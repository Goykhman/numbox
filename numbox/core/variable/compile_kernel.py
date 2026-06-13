"""Compile a `core.variable` Variable graph into fused @njit kernel(s).

Alongside `core.work` (a structref graph), this turns a `Graph`/`CompiledGraph`
into JIT-compiled straight-line code. When every formula is njit-able the
whole graph becomes a single fused @njit function whose interior nodes are SSA
temporaries (no per-node type info needed: numba infers every interior type
from the kernel's runtime argument types). When some formulas are not
njit-able, the first call detects them automatically -- numba compile errors
demote a node to plain Python, runtime errors always propagate -- and a Python
master orchestrates fused @njit segments around the demoted nodes, with a
fusion-maximizing linearization choosing the segment boundaries. The resulting
partition is described by `CompiledKernel.partition` (a PartitionReport with
per-node demotion reasons); formulas with no Python fallback
(cres/CompileResultWAP, CFunc, DUFunc) are always treated as jittable.

The on-disk cache is content-addressed per compiled unit (the fused kernel, or
each jit segment): the digest fingerprints each formula's code, constants,
default arguments, closure-cell values, referenced globals, and the kernel's
effective jit flags, so a stale binary is never reused and two distinct
kernels never collide. A formula with no canonical fingerprint forces its unit
uncached (no anchor, no numba cache) -- never reused, never wrong.
"""
import hashlib
import inspect
import keyword
import re
import sys
import warnings

from types import CodeType, FunctionType, ModuleType

import numpy as np

from numba import njit, typeof
from numba.core.ccallback import CFunc
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaError
from numba.core.types.function_type import CompileResultWAP
from numba.np.ufunc.dufunc import DUFunc

from numbox.core.configurations import jit_options as _default_jit_options
from numbox.core.variable._kernel_partition import (
    PartitionReport, Segment, _JitStep, _Plan, _PyStep,
    build_runs, discover, linearize, segment_liveness,
)
from numbox.core.variable.variable import QUAL_SEP, make_qual_name
from numbox.utils.preprocessing import (
    _anchor_root, _materialize_anchor, _orphan_anchor_sweep,
)

# Names injected into the kernel exec namespace; identifiers must avoid them.
_RESERVED = frozenset({"njit", "_kernel_jit_options"})

_ANCHOR_SUBDIR = "numbox-compile-kernel"
_orphan_anchor_sweep(_ANCHOR_SUBDIR)


def _sanitize(qual_name):
    s = re.sub(r"[^0-9A-Za-z_]", "_", qual_name)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    if not s or s[0].isdigit():
        s = "v_" + s
    return s


def _assign_identifiers(variables):
    """Map each Variable to a unique, valid, readable Python identifier.

    Readable (from the qual_name) with a minimal deterministic sha256 suffix
    only where names would otherwise collide. Reserves both the node temp `t`
    and its formula global `f_<t>` so those namespaces never clash, and avoids
    the injected reserved names.
    """
    used = set(_RESERVED)
    idents = {}
    for var in variables:
        base = _sanitize(var.qual_name())
        digest = hashlib.sha256(var.qual_name().encode("utf-8")).hexdigest()
        cand = base
        i = 0
        while cand in used or ("f_" + cand) in used or keyword.iskeyword(cand):
            i += 1
            if i > len(digest):
                raise RuntimeError(
                    f"Cannot assign a unique identifier for {var.qual_name()!r}; "
                    f"all sha256 prefixes exhausted"
                )
            cand = f"{base}_{digest[:i]}"
        used.add(cand)
        used.add("f_" + cand)
        idents[var] = cand
    return idents


def _wrap_formula(formula):
    """Return an njit-callable for `formula`; plain-Python callables are njit-wrapped."""
    if isinstance(formula, (Dispatcher, CompileResultWAP, DUFunc, CFunc)):
        return formula
    if not callable(formula):
        raise TypeError(f"formula {formula!r} is not callable")
    return njit(formula)


class _Unfingerprintable(Exception):
    """A value the cache digest cannot canonicalize; the kernel goes uncached."""


def _safe_repr(obj):
    """``repr(obj)`` that never raises -- the fingerprint fallback must always
    yield a string so an un-fingerprintable formula degrades to uncached rather
    than crashing when its ``__repr__`` itself raises."""
    try:
        return repr(obj)
    except Exception:  # noqa: BLE001 - fallback must not crash on a raising __repr__
        return f"<{type(obj).__name__} repr-failed>"


def _canon_value(value, seen):
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return repr(value)
    if isinstance(value, np.generic):
        return f"npscalar({value.dtype.str};{value.tobytes().hex()})"
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise _Unfingerprintable(f"object-dtype ndarray {value.dtype.str}")
        data = np.ascontiguousarray(value)
        try:
            raw = hashlib.sha256(data.tobytes()).hexdigest()
        except (ValueError, TypeError) as e:
            raise _Unfingerprintable(f"unhashable ndarray {value.dtype.str}") from e
        return f"ndarray({data.dtype.str};{value.shape};{raw})"
    if isinstance(value, (tuple, list)):
        return f"{type(value).__name__}[" + ",".join(_canon_value(v, seen) for v in value) + "]"
    if isinstance(value, (set, frozenset)):
        return f"{type(value).__name__}[" + ",".join(sorted(_canon_value(v, seen) for v in value)) + "]"
    if isinstance(value, dict):
        items = sorted((_canon_value(k, seen), _canon_value(v, seen)) for k, v in value.items())
        return "dict[" + ",".join(f"{k}:{v}" for k, v in items) + "]"
    if isinstance(value, ModuleType):
        return f"module({value.__name__})"
    if isinstance(value, Dispatcher):
        topts = _canon_value(dict(getattr(value, "targetoptions", {}) or {}), seen)
        return f"dispatcher({_fingerprint_function(value.py_func, seen)};{topts})"
    if isinstance(value, FunctionType):
        return f"function({_fingerprint_function(value, seen)})"
    raise _Unfingerprintable(type(value).__name__)


def _fingerprint_codeobj(code, seen):
    consts = ",".join(
        _fingerprint_codeobj(c, seen) if isinstance(c, CodeType) else _canon_value(c, seen)
        for c in code.co_consts
    )
    return (
        f"code({code.co_code.hex()};flags={code.co_flags};argc={code.co_argcount};"
        f"kwonly={code.co_kwonlyargcount};names={','.join(code.co_names)};consts=[{consts}])"
    )


def _referenced_global_names(code):
    names = set(code.co_names)
    for c in code.co_consts:
        if isinstance(c, CodeType):
            names |= _referenced_global_names(c)
    return names


def _fingerprint_function(func, seen):
    if id(func) in seen:
        return f"recursive({func.__qualname__})"
    seen = seen | {id(func)}
    code = func.__code__
    cells = []
    for name, cell in zip(code.co_freevars, func.__closure__ or ()):
        try:
            contents = cell.cell_contents
        except ValueError as e:
            raise _Unfingerprintable("empty closure cell") from e
        cells.append(f"{name}={_canon_value(contents, seen)}")
    hashed_globals = []
    for name in sorted(_referenced_global_names(code)):
        if name in func.__globals__:
            hashed_globals.append(f"{name}={_canon_value(func.__globals__[name], seen)}")
    return (
        f"func({func.__module__}:{func.__qualname__};{_fingerprint_codeobj(code, seen)};"
        f"defaults={_canon_value(func.__defaults__ or (), seen)};"
        f"kwdefaults={_canon_value(func.__kwdefaults__ or {}, seen)};"
        f"closure=[{';'.join(cells)}];globals=[{';'.join(hashed_globals)}])"
    )


def _formula_fingerprint(formula):
    """Behavioral identity of a formula for the cache digest.

    Returns ``(text, cacheable)``. The text covers every value channel
    numba freezes into a compiled artifact: code-object bytecode/consts/
    names, default-argument values, closure-cell values, the values of
    referenced module-level globals (recursing into helper functions and
    dispatchers, with cycle protection), the defining module, and
    dispatcher targetoptions. Builtins resolve outside ``__globals__``
    and are deliberately not hashed. Any value with no canonical form
    makes the formula un-fingerprintable: the returned text is then a
    per-object placeholder and ``cacheable`` is False, so the kernel is
    compiled without an on-disk cache -- never reused, never wrong.
    CFunc formulas embed an ASLR-randomized address, so numba can never
    disk-cache kernels that call them -- they are fingerprinted
    deterministically but marked uncacheable.
    """
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
        if isinstance(formula, Dispatcher):
            extra += ";targetoptions=" + _canon_value(dict(formula.targetoptions or {}), set())
        return _fingerprint_function(target, set()) + extra, not isinstance(formula, CFunc)
    except (_Unfingerprintable, RecursionError):
        return f"{_safe_repr(formula)} @{id(formula)}", False


def _check_formula_arity(formula, n_inputs, qual_name):
    target = getattr(formula, "py_func", None) or getattr(formula, "__wrapped__", None) or formula
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return
    try:
        sig.bind(*range(n_inputs))
    except TypeError as e:
        raise ValueError(
            f"{qual_name!r}: formula signature {sig} cannot accept its "
            f"{n_inputs} declared input(s) passed positionally ({e})"
        ) from None


def _assemble_source(params, lines, out_ids):
    """Assemble the canonical kernel source; both the fused kernel and every
    segment must share this exact shape so cache digests cannot drift."""
    sig = ", ".join(ident for _, _, ident in params)
    ret = f"    return ({', '.join(out_ids)},)"
    body = ("\n".join(lines) + "\n") if lines else ""
    return f"def _kernel({sig}):\n{body}{ret}\n"


def _emit_lines(nodes, skip, idents, bindings):
    """Emit one body line per node (excluding `skip`), filling `bindings`;
    raises on a missing formula, a non-callable formula, or an arity mismatch."""
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
            bindings[fg] = _wrap_formula(var.formula)
        except TypeError as e:
            raise TypeError(f"{var.qual_name()!r}: {e}") from e
        _check_formula_arity(var.formula, len(node.inputs), var.qual_name())
        arg_ids = ", ".join(idents[inp] for inp in node.inputs)
        in_names = ", ".join(repr(inp.qual_name()) for inp in node.inputs)
        lines.append(f"    {temp} = {fg}({arg_ids})  # {var.qual_name()!r} = f({in_names})")
    return lines


def _generate_segment_body(run_nodes, live_in, live_out, idents):
    """Like _generate_body, for one jit segment: live-ins are parameters,
    live-outs the return tuple. Same source shape so _compile applies verbatim.

    Returns (source, bindings, params, outputs) with params/outputs in the
    caller-provided (qual-sorted) live_in/live_out order.
    """
    params = [(v.source, v.name, idents[v]) for v in live_in]
    bindings = {}
    lines = _emit_lines(run_nodes, set(), idents, bindings)
    outputs = [v.qual_name() for v in live_out]
    out_ids = [idents[v] for v in live_out]
    source = _assemble_source(params, lines, out_ids)
    return source, bindings, params, outputs


def _generate_body(compiled, required, idents):
    """Generate `def _kernel(...): ...` source (no decorator) + bindings.

    Returns (source, bindings, params, outputs):
      source   - the kernel def as text (function name is the literal _kernel)
      bindings - {formula_global_name: njit-callable}
      params   - [(source_name, var_name, identifier)] in kernel-arg order
      outputs  - [requested_qual_name] in return-tuple order
    """
    if not required:
        raise ValueError("compile_kernel requires at least one requested variable")

    external = set()
    for vars_ in compiled.required_external_variables.values():
        external.update(vars_.values())

    ext_sorted = sorted(external, key=lambda v: v.qual_name())
    for var in ext_sorted:
        if var.formula is not None:
            raise ValueError(
                f"{var.qual_name()!r} is external but carries a formula; CompiledGraph "
                f"computes such a variable while a fused kernel treats it as a plain "
                f"input. Move it into a Variables namespace or drop the formula."
            )
    params = [(v.source, v.name, idents[v]) for v in ext_sorted]

    bindings = {}
    lines = _emit_lines(compiled.ordered_nodes, external, idents, bindings)

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


def _compile(source, bindings, jit_options, cache):
    """Content-addressed compile of the kernel source into an @njit dispatcher."""
    fingerprints = []
    cacheable = True
    for fg, formula in bindings.items():
        fp, ok = _formula_fingerprint(formula)
        fingerprints.append(f"{fg}: {fp}")
        cacheable = cacheable and ok
    opts = {**_default_jit_options, **(jit_options or {})}
    if cache is not None:
        opts["cache"] = cache
    opts.setdefault("cache", True)
    flags = {k: v for k, v in opts.items() if k != "cache"}
    try:
        flags_canon = _canon_value(flags, set())
    except (_Unfingerprintable, RecursionError):
        flags_canon = repr(sorted(flags.items(), key=repr))
        cacheable = False
    hash_text = (
        "ck-digest-v2\n" + source
        + "\n# formulas:\n" + "\n".join(fingerprints)
        + "\n# flags: " + flags_canon
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
        try:
            anchor.parent.mkdir(parents=True, exist_ok=True)
            _materialize_anchor(anchor, final_src)
        except OSError as e:
            warnings.warn(
                f"compile_kernel: cache directory unusable ({e}); "
                f"compiling without an on-disk cache"
            )
            opts["cache"] = False
    code = compile(final_src, str(anchor), "exec")
    # __name__ must be an importable module so numba can rebuild the cached
    # overload's environment in another process (importlib.import_module needs
    # a real name, not None); mirrors make_graph / make_structref.
    ns = {**bindings, "njit": njit, "_kernel_jit_options": opts, "__name__": __name__}
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns.pop(name)


class CompiledKernel:
    """A fused @njit kernel compiled from a Variable graph.

    Attributes::

      kernel      - hot-path callable: resolver before the first call, the
                    bare numba dispatcher once fused, the segmented master
                    otherwise. Positional external args (in `params` order)
                    -> tuple (in `outputs` order).
      params      - external input qual_names, kernel-argument order.
      outputs     - requested variable qual_names, return-tuple order.
      source      - generated kernel source text.
      identifiers - {qual_name: temp identifier} for inspection.
      partition   - PartitionReport describing what runs where (None until
                    the first call resolves the mode).
    """

    def __init__(self, kernel, params, outputs, source, identifiers, ctx,
                 required_vars, external_vars):
        self._fused = kernel
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

    @property
    def kernel(self):
        if self._mode == "fused":
            # Fused is permanent: later signatures go through numba's own
            # dispatch, and a typing failure there raises as in v1 -- no
            # segmentation fallback from fused mode.
            return self._fused
        if self._mode == "segmented":
            return self._run_segmented
        return self._resolve_and_call

    def _fused_report(self):
        compiled, _, _, _, _, external = self._ctx
        nodes = tuple(
            n.variable.qual_name() for n in compiled.ordered_nodes
            if n.variable not in external
        )
        return PartitionReport(mode="fused", segments=(Segment(
            kind="jit", nodes=nodes, inputs=tuple(self.params),
            outputs=tuple(self.outputs), source=self.source, reasons={},
        ),))

    def _resolve_and_call(self, *args):
        if self._mode != "virgin":
            return self.kernel(*args)
        try:
            arg_types = tuple(typeof(a) for a in args)
        except (ValueError, TypeError):
            arg_types = None
        if arg_types is not None:
            try:
                self._fused.compile(arg_types)
            except NumbaError:
                return self._discover_and_run(args)
            self._mode = "fused"
            self.partition = self._fused_report()
            return self._fused(*args)
        return self._discover_and_run(args)

    def _run_segmented(self, *args):
        try:
            return self._plan.run(args)
        except NumbaError:
            # Deliberately broad: a segment failing to compile for new input
            # types triggers re-discovery. A NumbaError raised inside a
            # python-step formula re-raises from re-discovery's own Python
            # evaluation of the same args, so nothing is masked.
            return self._discover_and_run(args)

    def _discover_and_run(self, args):
        compiled, idents, bindings_by_var, jit_options, cache, external = self._ctx
        values = dict(zip(self._external_vars, args))
        demoted = discover(compiled.ordered_nodes, external, values, bindings_by_var)
        nodes = [n for n in compiled.ordered_nodes if n.variable not in external]
        order = linearize(nodes, set(demoted))
        runs = build_runs(order, set(demoted))
        steps, segments = [], []
        for kind, run_nodes in runs:
            quals = tuple(n.variable.qual_name() for n in run_nodes)
            if kind == "python":
                ins = set()
                produced = set()
                for n in run_nodes:
                    ins.update(i for i in n.inputs if i not in produced)
                    produced.add(n.variable)
                    steps.append(_PyStep(
                        var=n.variable,
                        py_callable=getattr(n.variable.formula, "py_func", n.variable.formula),
                        in_vars=tuple(n.inputs),
                    ))
                reasons = {n.variable.qual_name(): demoted[n.variable] for n in run_nodes}
                key = lambda v: v.qual_name()   # noqa: E731 - tiny local sort key
                segments.append(Segment(
                    kind="python", nodes=quals,
                    inputs=tuple(v.qual_name() for v in sorted(ins, key=key)),
                    outputs=quals, source=None, reasons=reasons,
                ))
                continue
            live_in, live_out = segment_liveness(
                run_nodes, external, self._required_vars, order
            )
            src, seg_bindings, _, _ = _generate_segment_body(
                run_nodes, live_in, live_out, idents
            )
            disp = _compile(src, seg_bindings, jit_options, cache)
            disp.compile(tuple(typeof(values[v]) for v in live_in))
            steps.append(_JitStep(dispatcher=disp, in_vars=live_in, out_vars=live_out))
            segments.append(Segment(
                kind="jit", nodes=quals,
                inputs=tuple(v.qual_name() for v in live_in),
                outputs=tuple(v.qual_name() for v in live_out),
                source=src, reasons={},
            ))
        self._plan = _Plan(
            steps=tuple(steps),
            external_vars=tuple(self._external_vars),
            output_vars=tuple(self._required_vars),
        )
        self._mode = "segmented"
        self.partition = PartitionReport(mode="segmented", segments=tuple(segments))
        return tuple(values[v] for v in self._required_vars)

    def execute(self, external_values):
        """Dict-in / dict-out convenience, symmetric with CompiledGraph.execute."""
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


def compile_kernel(graph, required, *, jit_options=None, cache=None):
    """Compile `graph` into a fused @njit kernel for the `required` variables.

    :param graph: a `Graph`; its dependency structure and formulas are fused
        into one straight-line @njit function (see `CompiledKernel`).
    :param required: qualified name or list of qualified names. Order is
        preserved and fixes the order of `CompiledKernel.outputs` / the
        kernel's return tuple; a duplicate entry raises `ValueError` (each
        output is requested once -- the return tuple is positional, so a
        repeat carries no information).
    :param jit_options: merged over numbox's defaults
        (`NUMBOX_JIT_OPTIONS` env) and passed to @njit. All options except
        `cache` participate in the content-addressed digest.
    :param cache: tri-state. `None` (default) defers to
        `jit_options["cache"]`, then the `NUMBOX_JIT_OPTIONS` env default,
        then `True`. An explicit `True`/`False` wins over both.

    Error timing: structural problems raise here (unknown or malformed
    `required` entries, non-callable formulas, arity mismatches against the
    declared inputs, formula-bearing external variables, graphs deeper than
    the recursion limit); numba typing problems surface at the kernel's
    first call (auto-njit of plain-Python formulas is lazy).

    Caching: the kernel digest fingerprints each formula's bytecode,
    constants, default values, closure-cell values, referenced module-level
    globals (including helper functions, recursively), defining module, and
    the effective jit flags. A formula whose state cannot be fingerprinted
    (e.g. cres/CompileResultWAP objects, values with no canonical form)
    downgrades that one kernel to cache=False: always recompiled, never
    stale. When caching is enabled, a content-addressed anchor `.py` file is
    written under numba's cache directory; with caching off (or the cache
    dir unwritable, which warns and degrades) nothing is written.

    Non-jittable formulas: the first call resolves the execution mode. If the
    fully fused kernel cannot be typed for the actual argument types, each
    node is probed against the real intermediate values; nodes whose formulas
    fail to *compile* (or whose input values numba cannot type) run in plain
    Python, and the jittable remainder is fused into segments orchestrated
    from Python. `CompiledKernel.partition` describes the result, including
    per-node demotion reasons; it is `None` before the first call. Runtime
    errors never demote -- they propagate. `CompiledKernel.kernel` is the
    hot-path callable: the bare @njit dispatcher once the graph resolves
    fully fused, the Python master when segmented. A later call whose types
    break a segment re-learns and replaces the partition (one active plan);
    once fully fused, fused is permanent (a later-signature typing failure
    raises, as in v1). The discovery call computes jit-node values through
    per-node dispatchers while later calls use fused segments -- identical
    under default IEEE semantics, but non-default jit_options such as
    fastmath could in principle differ across fusion boundaries.
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
        raise RecursionError(
            f"graph dependency depth exceeds Python's recursion limit "
            f"({sys.getrecursionlimit()}); the traversal needs roughly one stack frame "
            f"per chained node - raise sys.setrecursionlimit(...) before compile_kernel"
        ) from None
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    source, bindings, params, outputs = _generate_body(compiled, required, idents)
    kernel = _compile(source, bindings, jit_options, cache)
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    bindings_by_var = {
        n.variable: bindings["f_" + idents[n.variable]]
        for n in compiled.ordered_nodes if n.variable not in external
    }

    def _registry_var(qual):
        src, _, name = qual.rpartition(QUAL_SEP)
        return graph.registry[src][name]

    required_vars = [_registry_var(q) for q in outputs]
    external_vars = [graph.registry[src][name] for src, name, _ in params]
    ctx = (compiled, idents, bindings_by_var, jit_options, cache, external)
    identifiers = {v.qual_name(): ident for v, ident in idents.items()}
    return CompiledKernel(
        kernel, params, outputs, source, identifiers, ctx, required_vars, external_vars
    )
