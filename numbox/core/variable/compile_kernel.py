"""Compile a graph of variables into one numba function, the kernel, that calculates the requested variables.

:func:`compile_kernel` returns a :class:`CompiledKernel`, which holds the kernel and describes it. numba caches the
kernel on disk under a name derived from a digest of the formulas and of the values they read. An edit to a formula
therefore makes numba compile the kernel again. A formula that numba cannot compile runs in Python, between compiled
segments. With declared types, compile_kernel compiles the kernel before its first call and raises its type errors
itself. The module's private functions generate the kernel's source, compute the digest and decide when the kernel
cannot be cached; the module ``_kernel_partition`` builds the segments.
"""
import hashlib
import warnings

from collections import OrderedDict
from types import FunctionType, ModuleType
from typing import Any, Callable, NamedTuple

from numba import njit, typeof
from numba.core.caching import NullCache
from numba.core.ccallback import CFunc
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaError
from numba.np.ufunc.dufunc import DUFunc

from numbox.core.configurations import jit_options as _default_jit_options
from numbox.core.variable._kernel_partition import (
    PartitionReport, Segment, _ConePlan, _JitStep, _Plan, _PyStep,
    _evaluate, build_runs, compute_boundary, cone_liveness, discover, linearize, segment_liveness,
)
from numbox.core.variable.utils import (
    _assign_identifiers, _check_formula_arity, _validate_declared_return, _wrap_formula,
)
from numbox.core.variable.variable import (
    QUAL_SEP, CompiledGraph, CompiledNode, Graph, Variable, make_qual_name,
)
from numbox.core.cache_key import _settings_canon
from numbox.utils.fingerprint import (
    _Unfingerprintable, _canon_value, _effective_flags, _fingerprint_function,
    _flags_canon, _referenced_global_names, _safe_repr,
)
from numbox.utils.highlevel import _type_identity
from numbox.utils.preprocessing import (
    _anchor_root, _anchored_or_uncached, _orphan_anchor_sweep,
)

_ANCHOR_SUBDIR = "numbox-compile-kernel"
_orphan_anchor_sweep(_ANCHOR_SUBDIR)


class _KernelCtx(NamedTuple):
    """The results of compile_kernel that a `CompiledKernel` keeps for its searches, its store and its recompute plans.

    It stays in Python and never reaches compiled code.
    """
    compiled: Any
    idents: dict
    bindings_by_var: dict
    jit_options: dict | None
    cache: bool | None
    external: set


def _formula_fingerprint(formula) -> tuple[str, bool]:
    """Return the text that stands for `formula` in the kernel's digest, and whether the kernel can be cached with it.

    The text is the fingerprint that `_fingerprint_function` in numbox.utils.fingerprint computes for the formula's
    Python function. It holds the function's code, constants and names, its module and qualified name, its default
    arguments and its closure variables. It also holds the values of the globals that the function reads, recursively
    through the functions and dispatchers that it calls; a function already visited is not walked again. It holds a
    module attribute that the function reads only if the attribute is data, a dispatcher or a @vectorize function. A
    dispatcher or a @vectorize formula adds its target options. Builtins are not among a function's globals, so they
    are not hashed. A value that a module provides only through a module-level ``__getattr__`` (PEP 562) is not
    hashed either, because resolving it would run the ``__getattr__`` of unrelated modules at every compile. After a
    change of such a value, numba loads the cached kernel until the user deletes its cache files.

    If the formula has no Python function, or reads a value without a canonical form, the text names the object by
    its id, and the second value is False. The kernel then compiles in every process, and numba never loads it from a
    cache. numbox hashes a @cfunc formula, but the second value is False as well: the kernel calls it at an address
    that differs from process to process, and numba cannot cache that.
    """
    # A dispatcher keeps its Python function in py_func, and a @vectorize or @cfunc function in __wrapped__. A cres
    # function has neither, so it reaches the id branch below and leaves the kernel uncached, although _canon_value
    # in numbox.utils.fingerprint hashes a cres function that a formula calls, through its _numbox_py_func.
    target = getattr(formula, "py_func", None)
    extra = ""
    if target is None and isinstance(formula, (DUFunc, CFunc)):
        target = getattr(formula, "__wrapped__", None)
        extra = f";kind={type(formula).__name__}"
    if target is None:
        target = formula
    if not isinstance(target, FunctionType):
        return f"{_safe_repr(formula)} @{id(formula)}", False
    try:
        if isinstance(formula, (Dispatcher, DUFunc)):
            extra += ";targetoptions=" + _canon_value(dict(getattr(formula, "targetoptions", {}) or {}), set())
        return _fingerprint_function(target, set()) + extra, not isinstance(formula, CFunc)
    except (_Unfingerprintable, RecursionError):
        return f"{_safe_repr(formula)} @{id(formula)}", False


def _is_self_cached(formula) -> bool:
    """Return True if `formula` is a dispatcher with an on-disk cache of its own, one compiled with ``cache=True``.

    numba's key for such a cache leaves out the values of the globals that the dispatcher reads and the jit options
    that it compiles under, so the cache can hold stale code. The kernel's digest cannot make numba refresh that
    cache. numbox must therefore keep numba from caching a kernel that links the code, or the kernel's cache would
    hold the stale code as well. A dispatcher compiled with ``cache=False`` compiles in every process and is safe.
    The check covers dispatchers only: a @vectorize function compiled with ``cache=True`` keeps a cache of its own as
    well, but numba caches a kernel that calls it, and that kernel can hold stale code from the function's cache. The
    digest holds the target options of a @vectorize formula.
    """
    if isinstance(formula, Dispatcher):
        # numba gives a dispatcher compiled with cache=False a NullCache, and one compiled with cache=True a
        # FunctionCache.
        cache = getattr(formula, "_cache", None)
        return cache is not None and not isinstance(cache, NullCache)
    return False


def _pyfunc_of(formula):
    """Return the Python function behind a formula, or None.

    That is a dispatcher's `py_func`, the ``__wrapped__`` function of a @vectorize or @cfunc function, or the formula
    itself if it is a plain function. A cres function has neither attribute, so the result for it is None.
    """
    func = getattr(formula, "py_func", None)
    if func is None and isinstance(formula, (DUFunc, CFunc)):
        func = getattr(formula, "__wrapped__", None)
    if func is None and isinstance(formula, FunctionType):
        func = formula
    return func if isinstance(func, FunctionType) else None


def _references_self_cached(value, seen: set[int]) -> bool:
    """Return True if `value` is, holds or reaches a dispatcher that has a cache of its own.

    A formula reaches such a dispatcher through a call, a default argument, a closure variable, a global, a module
    attribute, or a container of these. The kernel then links the dispatcher's code, which can be stale, and the
    digest cannot make numba refresh the dispatcher's cache. The walk follows the channels that `_canon_value`
    follows when it computes the digest, because a channel missed here would let stale code into a cached kernel. It
    differs in two places: it reads only the values of a dict, not its keys, and it stops at a cres function. `seen`
    holds the ids of the objects already visited.
    """
    if id(value) in seen:
        return False
    seen.add(id(value))
    if _is_self_cached(value):
        return True
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_references_self_cached(v, seen) for v in value)
    if isinstance(value, dict):
        return any(_references_self_cached(v, seen) for v in value.values())
    func = _pyfunc_of(value)
    if func is None:
        return False
    for cell in func.__closure__ or ():
        try:
            contents = cell.cell_contents
        except ValueError:
            continue
        if _references_self_cached(contents, seen):
            return True
    for default in (func.__defaults__ or ()) + tuple((func.__kwdefaults__ or {}).values()):
        if _references_self_cached(default, seen):
            return True
    referenced = _referenced_global_names(func.__code__)
    for name in referenced:
        if name not in func.__globals__:
            continue
        gval = func.__globals__[name]
        if isinstance(gval, ModuleType):
            # numba links a dispatcher reached as a module attribute, such as helpers.helper(x), possibly through a
            # chain of modules, as it links a global one.
            if _module_reaches_self_cached(gval, referenced, seen, set()):
                return True
        elif _references_self_cached(gval, seen):
            return True
    return False


def _module_reaches_self_cached(mod, ref_names, seen: set[int], mods_seen: set[int]) -> bool:
    """Return True if a name in `ref_names` that `mod` holds reaches a dispatcher that has a cache of its own.

    The walk recurses into the modules that `mod` holds under those names, and `mods_seen` holds the ids of the
    modules already visited. It tests membership in the module's ``__dict__`` and never calls ``getattr``, because
    ``getattr`` would run the ``__getattr__`` of an unrelated module (PEP 562) for a name that the formula uses
    elsewhere. A dispatcher that a module provides only through ``__getattr__`` is therefore not found.
    """
    if id(mod) in mods_seen:
        return False
    mods_seen.add(id(mod))
    mod_dict = getattr(mod, "__dict__", {})
    for name in ref_names:
        if name not in mod_dict:
            continue
        val = mod_dict[name]
        if isinstance(val, ModuleType):
            if _module_reaches_self_cached(val, ref_names, seen, mods_seen):
                return True
        elif _references_self_cached(val, seen):
            return True
    return False


def _self_cached_name(formula) -> str:
    return getattr(getattr(formula, "py_func", formula), "__qualname__", None) or _safe_repr(formula)


# The return-type checks that passed, keyed by the formula's fingerprint, the input types, the declared type and the
# jit options, so that a later compile_kernel of the same declarations does not compile the check again.
# compile_kernel checks a formula without a canonical fingerprint every time.
_validated_returns: set = set()


def _is_typed(var: Variable) -> bool:
    return var.params is not None and var.params.type is not None


def _classify(compiled: CompiledGraph):
    """Classify the graph by its declarations, and return ``(case, dispositions, consumed)``.

    `dispositions` maps each calculated variable to "STATIC_JIT" if it is declared jittable with a type and all its
    inputs have types, to "STATIC_PY" if it is declared ``jitable=False``, and to "UNKNOWN" otherwise. `consumed` is
    the set of external variables that a formula reads. The case is "A" if every variable is "STATIC_JIT" and every
    consumed external variable has a type: compile_kernel then compiles one fused kernel. The case is "B" if no
    variable is "UNKNOWN", some are "STATIC_PY", and every consumed external variable has a type: compile_kernel
    then compiles the segments. Otherwise the case is "C", and the kernel decides at its first call.
    """
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    consumed = set()
    dispositions = {}
    for node in compiled.ordered_nodes:
        var = node.variable
        if var in external:
            continue
        for inp in node.inputs:
            if inp in external:
                consumed.add(inp)
        p = var.params
        if p is not None and not p.jitable:
            dispositions[var] = "STATIC_PY"
        elif p is not None and p.jitable and p.type is not None and all(_is_typed(i) for i in node.inputs):
            dispositions[var] = "STATIC_JIT"
        else:
            dispositions[var] = "UNKNOWN"
    vals = set(dispositions.values())
    # A request of external variables alone leaves `dispositions` empty. Such a graph takes case "C", as a graph
    # without declarations does: nothing compiles in compile_kernel, and `partition` stays None until the first call.
    if not dispositions or "UNKNOWN" in vals or any(not _is_typed(e) for e in consumed):
        case = "C"
    elif "STATIC_PY" in vals:
        case = "B"
    else:
        case = "A"
    return case, dispositions, consumed


def _assemble_source(params: list[tuple[str, str, str]], lines: list[str], out_ids: list[str]) -> str:
    """Return the source of a kernel or a segment: ``def _kernel(<params>):``, the body lines, and a return of a tuple.

    The fused kernel and every segment take their source from this function, so the same nodes always give the same
    text, and with it the same digest. `_compile` renames the function by replacing ``def _kernel(``.
    """
    sig = ", ".join(ident for _, _, ident in params)
    ret = f"    return ({', '.join(out_ids)},)"
    body = ("\n".join(lines) + "\n") if lines else ""
    return f"def _kernel({sig}):\n{body}{ret}\n"


def _emit_lines(
    nodes: list[CompiledNode], skip: set[Variable],
    idents: dict[Variable, str], bindings: dict[str, Any], flags: dict | None = None,
) -> list[str]:
    """Return a line of source for each node not in `skip`, and add the formula of each node to `bindings`.

    The function adds the formula of a node whose identifier is `t` under the name ``f_<t>``. It decorates a plain
    Python formula with ``@njit`` under `flags` first. Raises ``ValueError`` for a node without a formula or with a
    formula that cannot take its inputs, and ``TypeError`` for a formula that is not callable.
    """
    lines = []
    for node in nodes:
        var = node.variable
        if var in skip:
            continue
        if var.formula is None:
            raise ValueError(
                f"{var.qual_name()!r} has graph placement but no formula; a fused "
                f"kernel cannot compile it. Provide a formula, or use CompiledGraph."
            )
        temp = idents[var]
        fg = "f_" + temp
        try:
            bindings[fg] = _wrap_formula(var.formula, flags)
        except TypeError as e:
            raise TypeError(f"{var.qual_name()!r}: {e}") from e
        _check_formula_arity(var.formula, len(node.inputs), var.qual_name())
        arg_ids = ", ".join(idents[inp] for inp in node.inputs)
        in_names = ", ".join(repr(inp.qual_name()) for inp in node.inputs)
        # Each variable becomes a local variable of the kernel that holds the result of its formula, so the source
        # holds no types: numba infers the type of each local variable from the kernel's arguments.
        lines.append(f"    {temp} = {fg}({arg_ids})  # {var.qual_name()!r} = f({in_names})")
    return lines


def _generate_segment_body(
    run_nodes: list[CompiledNode], live_in: tuple[Variable, ...],
    live_out: tuple[Variable, ...], idents: dict[Variable, str], flags: dict | None = None,
) -> tuple[str, dict, list, list]:
    """Return the source of one compiled segment, with its bindings, parameters and outputs.

    The segment takes the variables of `live_in` as parameters and returns those of `live_out`, in the order given,
    which the callers sort by qualified name. The source has the shape of the fused kernel's, so `_compile` handles
    both alike. Returns ``(source, bindings, params, outputs)``, as `_generate_body` does.
    """
    params = [(v.source, v.name, idents[v]) for v in live_in]
    bindings = {}
    lines = _emit_lines(run_nodes, set(), idents, bindings, flags)
    outputs = [v.qual_name() for v in live_out]
    out_ids = [idents[v] for v in live_out]
    source = _assemble_source(params, lines, out_ids)
    return source, bindings, params, outputs


def _generate_body(
    compiled: CompiledGraph, required: list[str], idents: dict[Variable, str], flags: dict | None = None,
) -> tuple[str, dict, list, list]:
    """Return the source of the fused kernel, without its decorator, with its bindings, parameters and outputs.

    Returns ``(source, bindings, params, outputs)``. `source` defines a function named ``_kernel``. `bindings` maps
    the global name of each formula to the callable that compiled code calls. `params` holds ``(source name, variable
    name, identifier)`` for each external variable, sorted by qualified name, which is the order of the kernel's
    arguments. `outputs` holds the requested qualified names in the order of the returned tuple. Raises ``ValueError``
    if `required` is empty or holds a name that the compiled graph does not hold.
    """
    if not required:
        raise ValueError("compile_kernel requires at least one requested variable")

    external = set()
    for vars_ in compiled.required_external_variables.values():
        external.update(vars_.values())

    ext_sorted = sorted(external, key=lambda v: v.qual_name())
    params = [(v.source, v.name, idents[v]) for v in ext_sorted]

    bindings = {}
    lines = _emit_lines(compiled.ordered_nodes, external, idents, bindings, flags)

    by_qual = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    outputs, out_ids = [], []
    for q in required:
        var = by_qual.get(q)
        if var is None:
            raise ValueError(f"Requested variable {q!r} is not in the compiled graph")
        outputs.append(q)
        out_ids.append(idents[var])

    source = _assemble_source(params, lines, out_ids)
    return source, bindings, params, outputs


def _compile(
    source: str, bindings: dict[str, Any], jit_options: dict | None, cache: bool | None,
    declared_sigs: tuple = (),
) -> Dispatcher:
    """Compile the source of a kernel or a segment into an @njit dispatcher named after a digest of what it computes.

    The digest covers the source, the fingerprint of each formula in `bindings`, the jit options other than
    ``cache``, the ``NUMBA_*`` settings that change the compiled code, and `declared_sigs`. `declared_sigs` holds the
    declared signature of an eager fused kernel, or the declared types of an eager segment's inputs and outputs, and
    is empty when the types are undeclared (case "C"). One source compiled for two sets of declared types therefore
    gets two names.

    numba caches the dispatcher unless `cache`, `jit_options` or numbox's default options disable caching, or the
    digest misses something that the code depends on. That happens for:

    - a formula without a canonical fingerprint;
    - a formula that is or reaches a dispatcher with a cache of its own;
    - a jit option without a canonical form;
    - a declared type that is or holds a dispatcher type.

    numbox warns in the second and third cases. For a cached dispatcher, numbox writes the source to an anchor file
    named after the digest. If numba cannot cache there, numbox warns and compiles without a cache.
    """
    fingerprints = []
    cacheable = True
    self_cached = []
    for fg, formula in bindings.items():
        fp, ok = _formula_fingerprint(formula)
        fingerprints.append(f"{fg}: {fp}")
        cacheable = cacheable and ok
        if _references_self_cached(formula, set()):
            cacheable = False
            self_cached.append(_self_cached_name(formula))
    if self_cached:
        # numbox keeps the possibly stale code of such a dispatcher out of the kernel's cache, but not out of the
        # kernel: numba loads the dispatcher's own cached code when it compiles the kernel, in every process.
        warnings.warn(
            "compile_kernel: formula(s) " + ", ".join(sorted(set(self_cached)))
            + " are, or reference, a @njit(cache=True) dispatcher whose cache key is "
            "blind to referenced global values; compiling this kernel "
            "without an on-disk cache so a stale inner binary cannot be linked into a "
            "cached artifact. Pass cache=False to those dispatchers (or supply "
            "plain-Python formulas) to restore caching."
        )
    # The cache argument takes precedence over the cache option of jit_options, which takes precedence over numbox's
    # default options. If none of them sets it, caching is enabled.
    opts = {**_default_jit_options, **(jit_options or {})}
    if cache is not None:
        opts["cache"] = cache
    opts.setdefault("cache", True)
    flags = _effective_flags(jit_options)
    flags_canon, ok_flags = _flags_canon(flags)
    if not ok_flags:
        cacheable = False
        # The warning lists every jit option that the kernel compiles under, not only the options without a canonical
        # form.
        warnings.warn(
            f"compile_kernel: jit flags {sorted(flags)} have no canonical "
            "fingerprint; compiling this kernel without an on-disk cache."
        )
    # Each declared signature is a tuple of numba types. The digest takes each type through _type_identity, which is
    # the same in every process, and not through its repr: the repr of a dispatcher type, such as
    # type(CPUDispatcher(<function f at 0x...>)), holds an address that changes from process to process. A digest of
    # it would rename the kernel in every process and leave a new pair of cache files behind each time. A type that is
    # or holds a dispatcher type makes the unit uncached. _type_identity marks such a type uncacheable, because
    # numba's own cache index misses a signature that holds one in every process and adds an entry each time.
    declared_ids = [[_type_identity(t) for t in s] for s in declared_sigs]
    declared_canon = [[h for h, _ok in sig] for sig in declared_ids]
    if not all(ok for sig in declared_ids for _h, ok in sig):
        cacheable = False
    hash_text = (
        "ck-digest-v3\n" + source
        + "\n# formulas:\n" + "\n".join(fingerprints)
        + "\n# flags: " + flags_canon
        # The NUMBA_* settings that change the compiled code are not among the jit options, and numba's own cache key
        # leaves them out; NUMBA_BOUNDSCHECK even overrides an explicit boundscheck option. The kernel's numba key
        # holds them too, through the part that numbox adds for every function it compiles (numbox.core.cache_key).
        + "\n# settings: " + _settings_canon()
        + "\n# declared_sigs: " + repr(declared_canon)
    )
    if not cacheable:
        opts["cache"] = False
    digest = hashlib.sha256(hash_text.encode("utf-8")).hexdigest()[:16]
    name = f"_kernel_{digest}"
    final_src = "@njit(**_kernel_jit_options)\n" + source.replace(
        "def _kernel(", f"def {name}(", 1
    )
    anchor = _anchor_root(_ANCHOR_SUBDIR) / f"_kernel_{digest}.py"
    if opts["cache"]:
        # Only a cached kernel gets an anchor file. The structref and the SQLite registrations write theirs whether
        # cached or not, so that numba's messages can quote their source.
        opts = _anchored_or_uncached(anchor, final_src, opts)
    code = compile(final_src, str(anchor), "exec")
    # The namespace's __name__ must name an importable module: numba rebuilds the environment of a cached function in
    # another process with importlib.import_module, which fails for None. make_graph and make_structref set it the
    # same way.
    ns = {**bindings, "njit": njit, "_kernel_jit_options": opts, "__name__": __name__}
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns.pop(name)


def _build_plan(
    runs: list, idents: dict, flags: dict, jit_options: dict | None, cache: bool | None,
    liveness: Callable, live_in_type: Callable, declared_sigs: bool = False, demoted: dict | None = None,
) -> tuple[tuple, tuple]:
    """Turn the runs of `build_runs` into the steps of a plan, and return ``(steps, segments)``.

    Each Python run becomes a `_PyStep` for each of its nodes. Each compiled run becomes one `_JitStep`, a segment that
    numba compiles at once for the types that `live_in_type` gives for its inputs, so a compile failure raises here.
    `liveness` returns the inputs and outputs of a compiled run. With `declared_sigs` True, the digest of each segment
    holds its declared input and output types. With `demoted`, which maps each Python variable to its reason, the
    function records a `Segment` of each run for the partition report; without it, `segments` is empty.
    """
    steps, segments = [], []
    for kind, run_nodes in runs:
        quals = tuple(n.variable.qual_name() for n in run_nodes)
        if kind == "python":
            for n in run_nodes:
                steps.append(_PyStep(
                    var=n.variable,
                    py_callable=getattr(n.variable.formula, "py_func", n.variable.formula),
                    in_vars=tuple(n.inputs),
                ))
            if demoted is not None:
                produced = {n.variable for n in run_nodes}
                ins = sorted({i for n in run_nodes for i in n.inputs if i not in produced},
                             key=lambda v: v.qual_name())
                segments.append(Segment(
                    kind="python", nodes=quals,
                    inputs=tuple(v.qual_name() for v in ins),
                    outputs=quals, source=None,
                    reasons={n.variable.qual_name(): demoted[n.variable] for n in run_nodes},
                ))
            continue
        live_in, live_out = liveness(run_nodes)
        src, seg_bindings, _, _ = _generate_segment_body(run_nodes, live_in, live_out, idents, flags)
        seg_sigs = ((tuple(v.params.type for v in live_in), tuple(v.params.type for v in live_out))
                    if declared_sigs else ())
        disp = _compile(src, seg_bindings, jit_options, cache, seg_sigs)
        disp.compile(tuple(live_in_type(v) for v in live_in))
        steps.append(_JitStep(dispatcher=disp, in_vars=live_in, out_vars=live_out))
        if demoted is not None:
            segments.append(Segment(
                kind="jit", nodes=quals,
                inputs=tuple(v.qual_name() for v in live_in),
                outputs=tuple(v.qual_name() for v in live_out),
                source=src, reasons={},
            ))
    return tuple(steps), tuple(segments)


class CompiledKernel:
    """A kernel that :func:`compile_kernel` compiled from a graph, with a description of the kernel.

    The kernel runs in one of two modes. A fused kernel is one function that numba compiled. A segmented kernel runs
    the formulas that numba cannot compile in Python, between compiled segments. Without declared types, the first
    call decides the mode: numba compiles the kernel for that call's argument types, and if numba fails, numbox
    searches for the formulas that must run in Python. With declared types, compile_kernel decides the mode.

    A fused kernel stays fused: a later call with types that numba cannot compile raises numba's ``TypingError``. A
    segmented kernel without declared types searches again when a segment fails to compile for a call's types. The new
    search replaces the partition, and it fills the store of :meth:`recompute` with that call's values. A segmented
    kernel built from declared types raises instead.

    Attributes:

    - `params`: the qualified names of the external variables, sorted, which is the order of the kernel's arguments.
    - `outputs`: the requested qualified names, in the order of the returned tuple.
    - `source`: the kernel's source as numbox generates it, with the function named ``_kernel``.
    - `identifiers`: a dict that maps each qualified name to its identifier in `source`.
    - `partition`: a ``PartitionReport`` that describes the split into segments and Python formulas. It is None until
      the first call, unless compile_kernel decided the mode from declared types.
    - `is_declared`: True if compile_kernel decided the mode from declared types. Such a kernel checks each value that
      :meth:`recompute` receives against its declared type, and never searches.
    """

    def __init__(self, kernel: Dispatcher, params: list[tuple[str, str, str]],
                 outputs: list[str], source: str, identifiers: dict[str, str],
                 ctx: _KernelCtx, required_vars: list[Variable],
                 external_vars: list[Variable], is_declared: bool = False) -> None:
        self._fused = kernel
        self.is_declared = is_declared
        self._mode = "virgin"
        self._plan = None
        self.partition = None
        self._ctx = ctx
        self._param_keys = [(src, name) for src, name, _ in params]
        self.params = [make_qual_name(src, name) for src, name, _ in params]
        self.outputs = list(outputs)
        self.source = source
        self.identifiers = identifiers
        self._required_vars = required_vars
        self._external_vars = external_vars
        # The store, {Variable: value}, from which recompute reads the values that do not change. A search fills it
        # at the first call; otherwise the first recompute fills it from _last_args.
        self._store = None
        # {Variable: reason} for each variable that runs in Python. A search or the declarations set it, and the
        # recompute plans run those variables in Python too.
        self._demoted = {}
        # The external values of the first completed call, from which _ensure_store fills and refills the store. A
        # fused kernel without declared types records them only at its first call; see _resolve_and_call.
        self._last_args = None
        # The variables that a change can start from: the external variables, and each calculated variable that a
        # recompute has overridden. None until recompute needs it.
        self._sources = None
        # The variables whose values the recompute plans keep current in the store, computed when first needed.
        self._boundary = None
        # The recompute plans, keyed by the affected variables and the variables that they read from the store, in
        # the order of their last use.
        self._cone_cache = OrderedDict()
        # The number of recompute plans kept. Beyond it, _cone_plan_cached discards the plan used least recently.
        self._cone_cap = 64

    @property
    def kernel(self) -> Callable:
        """The callable that runs the kernel on the external values, given in the order of `params`.

        It returns the requested values as a tuple, in the order of `outputs`. Before the first call, it is a method
        of this object that compiles or searches as needed and records the call's arguments for :meth:`recompute`.
        Once the kernel is fused, it is numba's dispatcher itself; for a segmented kernel, it is a method that runs
        the segments. A name bound to it before the first call keeps working, but each call through that name passes
        through the method, so a loop should bind the name after the first call.
        """
        if self._mode == "fused":
            # A fused kernel stays fused. numba compiles another version for new argument types, or raises a typing
            # error; numbox does not fall back to segments.
            return self._fused
        if self._mode == "fused-pending":
            return self._fused_pending_call
        if self._mode == "segmented":
            return self._run_segmented
        return self._resolve_and_call

    def _fused_pending_call(self, *args) -> tuple:
        # A caller may hold this bound method from before the first call. Once the mode is "fused", the method calls
        # the dispatcher directly and leaves _last_args as the first call set it, because _ensure_store fills the
        # store from the first call's arguments.
        if self._mode == "fused":
            return self._fused(*args)
        result = self._fused(*args)
        self._last_args = args
        self._mode = "fused"
        return result

    def _fused_report(self) -> PartitionReport:
        compiled, external = self._ctx.compiled, self._ctx.external
        nodes = tuple(
            n.variable.qual_name() for n in compiled.ordered_nodes
            if n.variable not in external
        )
        return PartitionReport(mode="fused", segments=(Segment(
            kind="jit", nodes=nodes, inputs=tuple(self.params),
            outputs=tuple(self.outputs), source=self.source, reasons={},
        ),))

    def _resolve_and_call(self, *args) -> tuple:
        if self._mode != "virgin":
            return self.kernel(*args)
        # An argument that numba cannot type sends the call straight to the search, which runs the formulas that read
        # it in Python.
        try:
            arg_types = tuple(typeof(a) for a in args)
        except (ValueError, TypeError):
            arg_types = None
        if arg_types is not None:
            try:
                self._fused.compile(arg_types)
            except NumbaError:
                result = self._discover_and_run(args)
                self._last_args = args
                return result
            # The mode changes before the call runs. If the call raises, _last_args stays None. Every later call goes
            # to the dispatcher directly, which never sets it, so recompute raises RuntimeError on this kernel from
            # then on.
            self._mode = "fused"
            self.partition = self._fused_report()
            result = self._fused(*args)
            self._last_args = args
            return result
        result = self._discover_and_run(args)
        self._last_args = args
        return result

    def _run_segmented(self, *args) -> tuple:
        try:
            result = self._plan.run(args)
        except NumbaError:
            # The catch is broad on purpose: a segment that fails to compile for new argument types starts a new
            # search. If a Python formula raised the NumbaError, the search runs the same formula on the same values
            # and raises it again, so no error is hidden. A kernel built from declared types does not search again,
            # because a search would replace the demoted set that the declarations give; it raises instead.
            if self.is_declared:
                raise
            result = self._discover_and_run(args)
        if self._last_args is None:
            self._last_args = args
        return result

    def _discover_and_run(self, args: tuple) -> tuple:
        compiled, idents, bindings_by_var, jit_options, cache, external = self._ctx
        flags = _effective_flags(jit_options)
        values = dict(zip(self._external_vars, args))
        demoted = discover(compiled.ordered_nodes, external, values, bindings_by_var, flags)
        nodes = [n for n in compiled.ordered_nodes if n.variable not in external]
        order = linearize(nodes, set(demoted))
        runs = build_runs(order, set(demoted))
        steps, segments = _build_plan(
            runs, idents, flags, jit_options, cache,
            liveness=lambda run_nodes: segment_liveness(run_nodes, external, self._required_vars, order),
            live_in_type=lambda v: typeof(values[v]),
            demoted=demoted,
        )
        self._plan = _Plan(
            steps=steps,
            external_vars=tuple(self._external_vars),
            output_vars=tuple(self._required_vars),
        )
        # The search computed every value, so the values fill the store. A new search, after a segment failed for
        # new argument types, replaces the store with this call's values, while _last_args keeps the first call's.
        self._store = values
        # The recompute plans run the variables of this search in Python.
        self._demoted = demoted
        self._mode = "segmented"
        self.partition = PartitionReport(mode="segmented", segments=segments)
        return tuple(values[v] for v in self._required_vars)

    def _ensure_store(self):
        if self._store is not None:
            return
        if self._mode == "virgin" or self._last_args is None:
            raise RuntimeError(
                "CompiledKernel.recompute requires a prior full call: call the kernel "
                "once with the current inputs to seed the value store before recompute()."
            )
        compiled, _, bindings_by_var, jit_options, cache, external = self._ctx
        flags = _effective_flags(jit_options)
        # The store is empty until the first recompute, unless a search filled it, and again after _flush_and_reseed:
        # a fused kernel and a plan of segments return only the requested values. This method fills the store by
        # computing every variable once more from the first call's arguments, formula by formula. Each formula
        # therefore runs once more, with any side effect that it has.
        values = dict(zip(self._external_vars, self._last_args))
        if self.is_declared:
            # The declarations give the demoted set. A search could disagree with them, so _evaluate runs without one.
            _evaluate(compiled.ordered_nodes, external, values, bindings_by_var, flags, self._demoted)
        else:
            self._demoted = discover(compiled.ordered_nodes, external, values, bindings_by_var, flags)
        self._store = values

    def _apply_changes(self, changed: dict) -> set:
        """Write the changed values into the store, and return the set of changed variables.

        The method looks a name up among the required external variables first, then among the calculated variables
        of the compiled graph; a name found in neither changes nothing, with a warning. The first override of a
        calculated variable adds it to the variables that a change can start from. On a kernel built from declared
        types, the method raises ``ValueError``, before it writes anything, if numba cannot convert a value to its
        declared type.
        """
        compiled = self._ctx.compiled
        # The first pass resolves each (source, name) to its variable and checks each value against its declaration.
        # A ValueError therefore leaves the store untouched, for a caller that catches it and calls recompute again.
        resolved = []
        for src, vals in changed.items():
            for name, val in vals.items():
                var = compiled.required_external_variables.get(src, {}).get(name)
                is_external = var is not None
                if var is None:
                    qual = make_qual_name(src, name)
                    var = next((n.variable for n in compiled.ordered_nodes
                                if n.variable.qual_name() == qual), None)
                    if var is None:
                        warnings.warn(f"{qual} is not in the calculation path, update has no effect.")
                        continue
                if self.is_declared and _is_typed(var):
                    # The check asks numba only whether it can convert the value's type to the declared type. A lossy
                    # conversion passes, such as a float for an int64 variable. The second pass stores the value as it
                    # is, without a conversion, and numba compiles the plan for the value's own type.
                    try:
                        got = typeof(val)
                        ok = self._fused.typingctx.can_convert(got, var.params.type) is not None
                    except (ValueError, TypeError):
                        ok, got = False, type(val).__name__
                    if not ok:
                        raise ValueError(
                            f"declared type {var.params.type}, got {got} for {var.qual_name()}")
                resolved.append((var, val, is_external))
        # The second pass writes the values, collects the changed variables, and records the calculated variables that
        # this call overrides for the first time.
        changed_vars = set()
        new_interior_sources = set()
        for var, val, is_external in resolved:
            self._store[var] = val
            changed_vars.add(var)
            if not is_external and (self._sources is None or var not in self._sources):
                new_interior_sources.add(var)
        if new_interior_sources:
            self._expand_sources(new_interior_sources)
        return changed_vars

    def _expand_sources(self, new_sources: set):
        """Add overridden calculated variables to the variables that a change can start from.

        The method resets the boundary, which `_ensure_boundary` computes again when it is next needed, and it discards
        every recompute plan, because the plans depend on the old boundary.
        """
        if self._sources is None:
            self._sources = set(self._external_vars)
        self._sources |= new_sources
        self._boundary = None
        self._cone_cache.clear()

    def _ensure_boundary(self):
        if self._boundary is not None:
            return
        compiled = self._ctx.compiled
        if self._sources is None:
            self._sources = set(self._external_vars)
        self._boundary = compute_boundary(
            compiled.ordered_nodes, self._sources, set(self._required_vars)
        )

    def _cone_live_in_type(self, v: Variable):
        """Return the type for which numba compiles a recompute segment, for the segment's input `v` from the store.

        On a kernel built from declared types, that is the declared type of `v` if it has one, so that the plan matches
        the segments that compile_kernel compiled. Otherwise it is the type of the value in the store.
        """
        if self.is_declared and _is_typed(v):
            return v.params.type
        return typeof(self._store[v])

    def _build_cone_plan(self, affected: list) -> _ConePlan:
        """Build and compile the recompute plan for the `affected` nodes.

        The method orders the nodes and splits them into runs, as compile_kernel does for the kernel. A node that runs
        in Python according to `_demoted` runs in Python here as well, and numba compiles each compiled run for the
        types of its inputs, which `_cone_live_in_type` gives. `_cone_plan_cached` calls this when it holds no plan for
        the nodes.
        """
        compiled, idents, _, jit_options, cache, _ = self._ctx
        flags = _effective_flags(jit_options)
        cone_vars = {n.variable for n in affected}
        demoted_in_cone = {v for v in self._demoted if v in cone_vars}
        order = linearize(affected, demoted_in_cone)
        runs = build_runs(order, demoted_in_cone)
        steps, _ = _build_plan(
            runs, idents, flags, jit_options, cache,
            liveness=lambda run_nodes: cone_liveness(run_nodes, order, self._required_vars, self._boundary),
            live_in_type=self._cone_live_in_type,
        )
        return _ConePlan(steps=steps)

    def _cone_key(self, affected) -> tuple[frozenset, frozenset]:
        """Return the key of the recompute plan for the `affected` nodes.

        The key holds the qualified names of the nodes and of the variables that they read from the store. The second
        part keeps apart a change of an external value and an override of a calculated variable that affect the same
        nodes, because their plans read different variables from the store.
        """
        cone_vars = {n.variable for n in affected}
        live_in_boundary = {inp for n in affected for inp in n.inputs if inp not in cone_vars}
        return (frozenset(v.qual_name() for v in cone_vars),
                frozenset(v.qual_name() for v in live_in_boundary))

    def _cone_plan_cached(self, affected) -> _ConePlan:
        key = self._cone_key(affected)
        plan = self._cone_cache.get(key)
        if plan is not None:
            self._cone_cache.move_to_end(key)
            return plan
        plan = self._build_cone_plan(affected)
        self._cone_cache[key] = plan
        if len(self._cone_cache) > self._cone_cap:
            self._cone_cache.popitem(last=False)
        return plan

    def _flush_and_reseed(self):
        """Discard every recompute plan, and refill the store and the boundary from the first call's arguments.

        The refilled store does not hold the values that earlier recompute calls wrote. :meth:`recompute` calls this
        once, when a plan fails while it runs.
        """
        self._cone_cache.clear()
        self._store = None
        self._boundary = None
        self._ensure_store()
        self._ensure_boundary()

    def recompute(self, changed: dict) -> tuple:
        """Recompute only the variables that `changed` affects, and return the requested values in `outputs` order.

        `changed` holds the new values as ``{source: {name: value}}``, as :meth:`CompiledGraph.recompute
        <numbox.core.variable.variable.CompiledGraph.recompute>` takes them. numbox writes them into the kernel's
        store, which holds the values of the first call and every change since; a later call of `kernel` or
        :meth:`execute` does not update it. The values may change type, and so may the value that a Python formula
        returns: numba compiles the recompute plan again for new types. A changed name may be a calculated variable.
        Its formula then does not run, and numbox recomputes the variables that depend on it from the given value. If
        the same call also changes one of its inputs, its formula runs and replaces the given value. A name outside the
        requested part of the graph changes nothing, with a warning.

        :raises RuntimeError: if no call has given the store its starting values: before the first call that
            completes, and always on a kernel without declared types whose first call compiled it as one function and
            then raised.
        :raises ValueError: on a kernel built from declared types, if numba cannot convert a changed value to the
            variable's declared type. numbox then writes nothing. It stores a value that numba can convert as it is,
            without a conversion.

        numba's ``TypingError`` propagates if numba cannot compile a new plan for the changed values, and the values
        of that call stay in the store. If a plan that an earlier ``recompute`` built fails on the new values, numbox
        retries once. The retry refills the store from the first call's arguments, applies this call's changes, and
        builds a new plan; the changes of earlier ``recompute`` calls are lost. If the new plan fails as well, numba's
        error propagates. A kernel built from declared types does not retry. The section "Recomputing after a change"
        of the numbox.core.variable documentation gives examples.
        """
        self._ensure_store()
        changed_vars = self._apply_changes(changed)
        if not changed_vars:
            return tuple(self._store[v] for v in self._required_vars)
        compiled = self._ctx.compiled
        self._ensure_boundary()
        affected = compiled._collect_affected(changed_vars)
        if not affected:
            return tuple(self._store[v] for v in self._required_vars)
        # The plan recomputes only the affected nodes and reads every other value from the store. A node that runs in
        # Python in the kernel runs in Python in the plan, and numba compiles the other nodes in segments. A plan is
        # compiled at its first use and kept for the same affected nodes and store inputs, up to _cone_cap plans, so a
        # recurring change reuses its compiled plan.
        plan = self._cone_plan_cached(affected)
        try:
            plan.run_into(self._store)
        except NumbaError:
            # A plan fails here when numba cannot compile it for the types of the values it reads. That happens to a
            # plan built for earlier types. A new plan that fails to compile raises in _cone_plan_cached instead, with
            # no retry. The retry discards every plan and refills the store from the first call's arguments. It then
            # applies this call's changes again and builds a new plan for the new types. A kernel built from declared
            # types raises at once. numba compiles its plans for the declared types, which a retry would not change,
            # and _apply_changes has already refused the values that numba cannot convert to them.
            if self.is_declared:
                raise
            self._flush_and_reseed()
            self._apply_changes(changed)
            affected = compiled._collect_affected(changed_vars)
            plan = self._cone_plan_cached(affected)
            plan.run_into(self._store)
        return tuple(self._store[v] for v in self._required_vars)

    def execute(self, external_values: dict) -> dict:
        """Call the kernel on external values given as ``{source: {name: value}}``, and return a dict by qualified name.

        `external_values` may hold more values than the kernel needs, as for :meth:`CompiledGraph.execute
        <numbox.core.variable.variable.CompiledGraph.execute>`. Raises ``KeyError`` if the value of a parameter is
        missing. A call after the first does not update the store of :meth:`recompute`.
        """
        args = []
        for src, name in self._param_keys:
            try:
                args.append(external_values[src][name])
            except KeyError as e:
                raise KeyError(
                    f"Missing external value for {make_qual_name(src, name)!r}"
                ) from e
        result = self.kernel(*args)
        return dict(zip(self.outputs, result))


def compile_kernel(
    graph: Graph, required: str | list[str], *,
    jit_options: dict | None = None, cache: bool | None = None,
) -> CompiledKernel:
    """Compile `graph` into a kernel that calculates the `required` variables, and return it as a `CompiledKernel`.

    :param graph: the graph that holds the variables and their formulas.
    :param required: the qualified name of a variable, such as ``"calc.total"``, or a list of them, each named once.
        The kernel returns the variables in this order.
    :param jit_options: numba's jit options for the kernel, the keyword arguments of ``@njit``. numbox merges them over
        its default options, which it reads from the ``NUMBOX_JIT_OPTIONS`` environment variable, and decorates every
        plain Python formula with the result, ``cache`` left out.
    :param cache: ``True`` or ``False`` to enable or disable the kernel's cache. The default, ``None``, defers to the
        ``cache`` option of `jit_options`, then to that of numbox's default options. numbox sets the latter to
        ``False`` when numba can write no cache for numbox's own modules, even if the kernel's own file could be
        cached. If none of them is set, caching is enabled.
    :raises TypeError: if an entry of `required` is not a string, or a formula is not callable.
    :raises ValueError: if `required` is empty, names a variable twice, or holds a name that is not qualified or that
        the graph cannot resolve. compile_kernel raises it as well for a variable in the requested part of the graph
        that has no formula, or a formula that cannot take its inputs.
    :raises RuntimeError: if the requested part of the graph has a cycle.
    :raises RecursionError: if a chain of dependencies is deeper than Python's recursion limit.

    Without declared types, numba compiles the kernel at its first call, for the types of that call's arguments, and
    numba's typing errors surface there. :class:`~numbox.core.variable.variable.Params` declares the type of a
    variable. If every calculated variable in the requested part of the graph has a declared type, and so does every
    external variable that a formula reads, compile_kernel compiles the kernel or its segments for the declared
    types. It then raises ``ValueError`` for a formula that returns a type other than its declaration, and numba's
    ``TypingError`` for a formula that does not compile for the declared types. One case still compiles at the first
    call: a fused kernel with a requested external variable that no formula reads.

    numba caches the kernel on disk under a name derived from a digest. The digest covers the formulas and the values
    that they read, the jit options other than ``cache``, the ``NUMBA_*`` settings that change the compiled code, and
    the declared types. When caching is enabled, numbox writes the kernel's source to a file of that name, in the
    directory ``numbox-compile-kernel`` under ``NUMBA_CACHE_DIR`` or numba's user cache directory. The kernel compiles
    in every process, without a cache, if a formula is itself a ``@cfunc`` or a ``cres`` function, reads a value that
    numbox cannot hash, or is or calls a dispatcher compiled with ``cache=True``. numbox warns in the last case, when
    a jit option has no form that numbox can hash, and when numba cannot cache in the kernel's directory. A dispatcher
    compiled with ``cache=True`` keeps a cache of its own, which can be stale, and even an uncached kernel runs the
    code from that cache.

    With numba's default options, the values that numbox computes formula by formula equal those that a segment or a
    fused kernel computes. Under ``fastmath``, which lets LLVM reorder floating-point arithmetic within one compiled
    function, they can differ. numbox computes formula by formula in the search at the first call, and when it fills
    a fused kernel's store at its first ``recompute``. The numbox.core.variable documentation gives the details in
    "When the kernel compiles again", "When the kernel is not cached" and "If a formula cannot be compiled".
    """
    required = [required] if isinstance(required, str) else list(required)
    for entry in required:
        if not isinstance(entry, str):
            raise TypeError(f"required entries must be qualified-name strings; got {entry!r}")
        if QUAL_SEP not in entry:
            raise ValueError(
                f"required entry {entry!r} is not qualified (expected 'source{QUAL_SEP}name')"
            )
    dupes = sorted({entry for entry in required if required.count(entry) > 1})
    if dupes:
        raise ValueError(
            f"required has duplicate entries {dupes}; each requested output must "
            f"appear once (the kernel's return tuple is positional)"
        )
    try:
        compiled = graph.compile(required)
    except KeyError as e:
        raise ValueError(
            f"required name or one of its dependencies cannot be resolved in the graph: {e}"
        ) from e
    except RecursionError:
        raise RecursionError("Consider raising recursion limit.") from None
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    case, dispositions, consumed = _classify(compiled)
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    flags = _effective_flags(jit_options)
    flags_key, _ = _flags_canon(flags)
    if case in ("A", "B"):
        for node in compiled.ordered_nodes:
            var = node.variable
            if var in external or dispositions.get(var) != "STATIC_JIT":
                continue
            in_types = tuple(i.params.type for i in node.inputs)
            fp, fingerprintable = _formula_fingerprint(var.formula)
            key = (fp, in_types, var.params.type, flags_key)
            if fingerprintable and key in _validated_returns:
                continue
            _validate_declared_return(var.formula, in_types, var.params.type, flags)
            if fingerprintable:
                _validated_returns.add(key)
    source, bindings, params, outputs = _generate_body(compiled, required, idents, flags)

    def _registry_var(qual):
        src, name = qual.rsplit(QUAL_SEP, 1)
        return graph.registry[src][name]

    required_vars = [_registry_var(q) for q in outputs]
    external_vars = [graph.registry[src][name] for src, name, _ in params]
    # In case "A", the digest holds the declared types of the external variables that the formulas read, in the order
    # of the kernel's arguments, as the eager compile below uses them. One source that holds no types therefore gets one
    # name for int64 values and another for float64 values. Cases "B" and "C" pass no signature here; the segments of
    # case "B" carry their own.
    declared_sigs = ()
    if case == "A":
        consumed_sig = tuple(v.params.type for v in external_vars if v in consumed)
        declared_sigs = (consumed_sig,)
    kernel = _compile(source, bindings, jit_options, cache, declared_sigs)
    bindings_by_var = {
        n.variable: bindings["f_" + idents[n.variable]]
        for n in compiled.ordered_nodes if n.variable not in external
    }
    ctx = _KernelCtx(compiled, idents, bindings_by_var, jit_options, cache, external)
    identifiers = {v.qual_name(): ident for v, ident in idents.items()}
    if case == "A":
        ck = CompiledKernel(kernel, params, outputs, source, identifiers, ctx,
                            required_vars, external_vars, is_declared=True)
        sig_vars = [v for v in external_vars if v in consumed]
        # The eager compile needs a type for every parameter of the kernel. A requested external variable that no
        # formula reads is a parameter but not in `consumed`, so its type is not in the signature, even if it is
        # declared. A signature without it would have too few types, so such a kernel compiles at its first call.
        if len(sig_vars) == len(external_vars):
            ck._fused.compile(tuple(v.params.type for v in sig_vars))
        ck._mode = "fused-pending"
        ck.partition = ck._fused_report()
        return ck
    if case == "B":
        demoted = {n.variable: "declared non-jittable" for n in compiled.ordered_nodes
                   if dispositions.get(n.variable) == "STATIC_PY"}
        nodes = [n for n in compiled.ordered_nodes if n.variable not in external]
        order = linearize(nodes, demoted)
        runs = build_runs(order, demoted)
        # compile_kernel builds the segments as _discover_and_run does, but for the declared types and with the demoted
        # set that the declarations give, without a search.
        steps, segments = _build_plan(
            runs, idents, flags, jit_options, cache,
            liveness=lambda run_nodes: segment_liveness(run_nodes, external, required_vars, order),
            live_in_type=lambda v: v.params.type,
            declared_sigs=True,
            demoted=demoted,
        )
        ck = CompiledKernel(kernel, params, outputs, source, identifiers, ctx,
                            required_vars, external_vars, is_declared=True)
        ck._plan = _Plan(steps=steps, external_vars=tuple(external_vars),
                         output_vars=tuple(required_vars))
        ck._demoted = demoted
        ck._mode = "segmented"
        ck.partition = PartitionReport(mode="segmented", segments=segments)
        return ck
    return CompiledKernel(
        kernel, params, outputs, source, identifiers, ctx, required_vars, external_vars
    )
