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

Jitability may be either discovered at the first call (above) or declared up
front. Each `Variable` carries an optional `params` (`Params(jitable, type)`).
A node with no `params` is discovered exactly as before -- byte-for-byte the
same behavior. When every node in the required cone is declared (and every
consumed external is typed), `compile_kernel()` resolves the execution mode at
build time instead of at the first call: an all-jittable graph compiles eagerly
into one fused kernel ("fused"), a declared jit/Python mix compiles eagerly into
a static segment plan ("segmented") with no probing, and `CompiledKernel.partition`
is populated at build (inspectable before any call). A declared `params.type`
that the formula does not naturally yield is caught at build by an explicit
unconstrained return-type probe -- not by binding the formula to the declared
signature, which would silently coerce a convertible-but-wrong scalar type (a
node declared int64 over a float-returning body would otherwise return a
truncated value). Any node left undeclared (or only partially typed) keeps the
runtime-discovery path; a graph that declares nothing behaves exactly as today.

The on-disk cache is content-addressed per compiled unit (the fused kernel, or
each jit segment): the digest fingerprints each formula's code, constants,
default arguments, closure-cell values, referenced globals, and the kernel's
effective jit flags, so a stale binary is never reused and two distinct
kernels never collide. The kernel source never mentions types, so declared
signatures are appended to the digest as well: two declared-type variants of one
graph therefore get distinct cache anchors. A formula with no canonical
fingerprint forces its unit uncached (no anchor, no numba cache) -- never
reused, never wrong.
"""
import hashlib
import warnings

from collections import OrderedDict
from types import FunctionType
from typing import Any, Callable

from numba import njit, typeof
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
from numbox.utils.fingerprint import (
    _Unfingerprintable, _canon_value, _fingerprint_function, _safe_repr,
)
from numbox.utils.preprocessing import (
    _anchor_root, _materialize_anchor, _orphan_anchor_sweep,
)

_ANCHOR_SUBDIR = "numbox-compile-kernel"
_orphan_anchor_sweep(_ANCHOR_SUBDIR)


def _effective_flags(jit_options: dict | None) -> dict:
    """The non-`cache` jit flags for the kernel. Threaded into the inner formula
    njit-wraps too, so a plain-Python formula computes identically whether reached
    via discovery, a fused segment, or the fully fused kernel (non-default flags such
    as `fastmath`/`error_model` otherwise diverge across the discovery boundary)."""
    opts = {**_default_jit_options, **(jit_options or {})}
    return {k: v for k, v in opts.items() if k != "cache"}


def _formula_fingerprint(formula) -> tuple[str, bool]:
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


# Build-time return-type validations are memoized by (formula fingerprint,
# input types, declared type) so repeated compile_kernel() calls do not re-probe.
_validated_returns: set = set()


def _is_typed(var: Variable) -> bool:
    return var.params is not None and var.params.type is not None


def _classify(compiled: CompiledGraph):
    """Label interior nodes and pick the case. Returns
    (case, dispositions: {Variable: str}, consumed_externals: set[Variable])."""
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
    # An empty `dispositions` is a fully-undeclared external-only graph (zero
    # interior nodes); route it to "C" so the "declares nothing = byte-for-byte
    # today" invariant holds (no eager build, partition None until first call).
    if not dispositions or "UNKNOWN" in vals or any(not _is_typed(e) for e in consumed):
        case = "C"
    elif "STATIC_PY" in vals:
        case = "B"
    else:
        case = "A"
    return case, dispositions, consumed


def _assemble_source(params: list[tuple[str, str, str]], lines: list[str], out_ids: list[str]) -> str:
    """Assemble the canonical kernel source; both the fused kernel and every
    segment must share this exact shape so cache digests cannot drift."""
    sig = ", ".join(ident for _, _, ident in params)
    ret = f"    return ({', '.join(out_ids)},)"
    body = ("\n".join(lines) + "\n") if lines else ""
    return f"def _kernel({sig}):\n{body}{ret}\n"


def _emit_lines(
    nodes: list[CompiledNode], skip: set[Variable],
    idents: dict[Variable, str], bindings: dict[str, Any], flags: dict | None = None,
) -> list[str]:
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
            bindings[fg] = _wrap_formula(var.formula, flags)
        except TypeError as e:
            raise TypeError(f"{var.qual_name()!r}: {e}") from e
        _check_formula_arity(var.formula, len(node.inputs), var.qual_name())
        arg_ids = ", ".join(idents[inp] for inp in node.inputs)
        in_names = ", ".join(repr(inp.qual_name()) for inp in node.inputs)
        lines.append(f"    {temp} = {fg}({arg_ids})  # {var.qual_name()!r} = f({in_names})")
    return lines


def _generate_segment_body(
    run_nodes: list[CompiledNode], live_in: tuple[Variable, ...],
    live_out: tuple[Variable, ...], idents: dict[Variable, str], flags: dict | None = None,
) -> tuple[str, dict, list, list]:
    """Like _generate_body, for one jit segment: live-ins are parameters,
    live-outs the return tuple. Same source shape so _compile applies verbatim.

    Returns (source, bindings, params, outputs) with params/outputs in the
    caller-provided (qual-sorted) live_in/live_out order.
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
    """Content-addressed compile of the kernel source into an @njit dispatcher.

    `declared_sigs` carries the unit's declared signature(s) -- the consumed
    external signature for an eager fused kernel, each segment's live-in/out
    signature for an eager segment -- folded into the digest via `repr` so two
    declared-type variants of one type-free source get distinct anchors. Empty
    for undeclared (Case C) units."""
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
    flags = _effective_flags(jit_options)
    try:
        flags_canon = _canon_value(flags, set())
    except (_Unfingerprintable, RecursionError):
        flags_canon = repr(sorted(flags.items(), key=repr))
        cacheable = False
    hash_text = (
        "ck-digest-v3\n" + source
        + "\n# formulas:\n" + "\n".join(fingerprints)
        + "\n# flags: " + flags_canon
        + "\n# declared_sigs: " + repr([repr(s) for s in declared_sigs])
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
      recompute   - value-only incremental refresh of only the cone affected
                    by a change, over a store seeded by a prior `kernel` call;
                    returns a tuple in `outputs` order (see `recompute`).
      params      - external input qual_names, kernel-argument order.
      outputs     - requested variable qual_names, return-tuple order.
      source      - generated kernel source text.
      identifiers - {qual_name: temp identifier} for inspection.
      partition   - PartitionReport describing what runs where. None until the
                    first call for undeclared graphs; set at build time for
                    fully-declared (eager) graphs.
      is_declared - True when the graph was fully declared and the mode resolved
                    eagerly at build; False for a discovery (undeclared) kernel.
                    Declared kernels enforce the `recompute()` type contract
                    instead of re-discovering.
    """

    def __init__(self, kernel: Dispatcher, params: list[tuple[str, str, str]],
                 outputs: list[str], source: str, identifiers: dict[str, str],
                 ctx: tuple, required_vars: list[Variable],
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
        self._store = None          # {Variable: value}; seeded on first recompute
        self._demoted = {}          # {Variable: reason}; frozen demotion verdicts
        self._last_args = None      # external args of the most recent full resolution
        self._sources = None        # change-source Variables (externals; may grow on interior override)
        self._boundary = None       # set[Variable]; persisted-node set, computed lazily
        self._cone_cache = OrderedDict()  # LRU cache of cone sub-plans, keyed on cone+live-in
        self._cone_cap = 64         # max distinct cone plans retained before LRU eviction

    @property
    def kernel(self) -> Callable:
        if self._mode == "fused":
            # Fused is permanent: later signatures go through numba's own
            # dispatch, and a typing failure there raises -- no segmentation
            # fallback from fused mode.
            return self._fused
        if self._mode == "fused-pending":
            return self._fused_pending_call
        if self._mode == "segmented":
            return self._run_segmented
        return self._resolve_and_call

    def _fused_pending_call(self, *args) -> tuple:
        # A caller may hold a reference to this bound method across calls; once
        # the mode has flipped to "fused", go straight to the dispatcher without
        # re-stamping _last_args (the first call already captured the seeding args).
        if self._mode == "fused":
            return self._fused(*args)
        result = self._fused(*args)
        self._last_args = args
        self._mode = "fused"
        return result

    def _fused_report(self) -> PartitionReport:
        compiled, _, _, _, _, external = self._ctx
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
            # Deliberately broad: a segment failing to compile for new input
            # types triggers re-discovery. A NumbaError raised inside a
            # python-step formula re-raises from re-discovery's own Python
            # evaluation of the same args, so nothing is masked. Declared
            # kernels do not re-discover (that would overwrite the authoritative
            # _demoted); an off-contract input re-raises instead.
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
                run_nodes, live_in, live_out, idents, flags
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
        self._store = values              # seed store from the already-computed values
        self._demoted = demoted           # freeze demotion verdicts for cone builds
        self._mode = "segmented"
        self.partition = PartitionReport(mode="segmented", segments=tuple(segments))
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
        # A fused first call discarded interior values, so seed by evaluating the
        # whole graph once from the captured args. Fused-graph formulas are njit-pure,
        # so the one extra evaluation is observationally safe. Seeding only the
        # persisted-node closure instead is a possible future optimization.
        values = dict(zip(self._external_vars, self._last_args))
        if self.is_declared:
            # Seed interior values with the frozen declared demotion set; re-running
            # discover would re-probe and could disagree with the declarations.
            _evaluate(compiled.ordered_nodes, external, values, bindings_by_var, flags, self._demoted)
        else:
            self._demoted = discover(compiled.ordered_nodes, external, values, bindings_by_var, flags)
        self._store = values

    def _apply_changes(self, changed: dict) -> set:
        """Write changed values into the store; return the set of changed Variables.
        External names resolve via required_external_variables; interior names via
        ordered_nodes (overriding an interior node expands the change-source set)."""
        compiled = self._ctx[0]
        # First pass: resolve every (src, name) -> var and run the declared
        # contract check, collecting (var, val, is_external). If ANY check
        # fails, raise here -- before mutating _store -- so a caught-and-retried
        # ValueError cannot leave the store partially written.
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
                    try:
                        got = typeof(val)
                        ok = self._fused.typingctx.can_convert(got, var.params.type) is not None
                    except (ValueError, TypeError):
                        ok, got = False, type(val).__name__
                    if not ok:
                        raise ValueError(
                            f"declared type {var.params.type}, got {got} for {var.qual_name()}")
                resolved.append((var, val, is_external))
        # Second pass: writes, changed_vars, and interior-override bookkeeping.
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
        """Add overridden interior nodes to the change-source set, force the
        persisted-node boundary to recompute, and invalidate cached cone plans
        (their boundaries have changed)."""
        if self._sources is None:
            self._sources = set(self._external_vars)
        self._sources |= new_sources
        self._boundary = None
        self._cone_cache.clear()

    def _ensure_boundary(self):
        if self._boundary is not None:
            return
        compiled = self._ctx[0]
        if self._sources is None:
            self._sources = set(self._external_vars)
        self._boundary = compute_boundary(
            compiled.ordered_nodes, self._sources, set(self._required_vars)
        )

    def _cone_live_in_type(self, v: Variable):
        """The numba type to compile a cone jit-segment live-in against. For a
        declared kernel whose live-in declares a type, use the declared type (so the
        cone matches the eager build's signature); otherwise fall back to the stored
        value's runtime type, as the undeclared path always does."""
        if self.is_declared and _is_typed(v):
            return v.params.type
        return typeof(self._store[v])

    def _build_cone_plan(self, affected: list) -> _ConePlan:
        """Linearize the affected cone, split into runs, and compile each jit run
        against its live-in types (declared types for a declared kernel, else the
        store's current runtime types; Python runs become _PyStep chains).
        `_demoted` is restricted to cone nodes so only nodes demoted at seed time stay
        Python. Called by `_cone_plan_cached` on a cache miss."""
        compiled, idents, _, jit_options, cache, _ = self._ctx
        flags = _effective_flags(jit_options)
        cone_vars = {n.variable for n in affected}
        demoted_in_cone = {v for v in self._demoted if v in cone_vars}
        order = linearize(affected, demoted_in_cone)
        runs = build_runs(order, demoted_in_cone)
        steps = []
        for kind, run_nodes in runs:
            if kind == "python":
                for n in run_nodes:
                    steps.append(_PyStep(
                        var=n.variable,
                        py_callable=getattr(n.variable.formula, "py_func", n.variable.formula),
                        in_vars=tuple(n.inputs),
                    ))
                continue
            live_in, live_out = cone_liveness(run_nodes, order, self._required_vars, self._boundary)
            src, seg_bindings, _, _ = _generate_segment_body(run_nodes, live_in, live_out, idents, flags)
            disp = _compile(src, seg_bindings, jit_options, cache)
            disp.compile(tuple(self._cone_live_in_type(v) for v in live_in))
            steps.append(_JitStep(dispatcher=disp, in_vars=live_in, out_vars=live_out))
        return _ConePlan(steps=tuple(steps))

    def _cone_key(self, affected) -> tuple[frozenset, frozenset]:
        """Cache key for a cone sub-plan: (cone node qual_names, live-in boundary
        qual_names). Including the live-in boundary keeps an external change and an
        interior override that share a cone from colliding (their boundaries differ)."""
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
        """Drop the cone-plan cache and re-seed the store + boundary from the last full
        call. Used to recover when a boundary live-in's type change makes a cached cone
        dispatcher fail to compile."""
        self._cone_cache.clear()
        self._store = None
        self._boundary = None
        self._ensure_store()
        self._ensure_boundary()

    def recompute(self, changed: dict) -> tuple:
        """Incrementally re-evaluate only the cone affected by `changed`.

        Mirrors :meth:`numbox.core.variable.variable.CompiledGraph.recompute`: this
        is a value-only refresh, not a recompile. `changed` is
        ``{source: {name: value}}``; the returned tuple is in `outputs` order. The
        same variables may carry different values across calls, but their numba types
        must stay the same as the seeding call (a type change is recovered from once,
        see below, but the contract is same-types).

        Precondition: a prior full call (``kernel(...)`` / ``execute(...)``) must have
        seeded the value store. Calling `recompute` first raises ``RuntimeError``.

        What it does: it writes the changed values into a persistent value store,
        collects the downstream cone of the changed nodes, and re-fuses just that cone
        -- reading every unchanged input from the store. The cone sub-plan is compiled
        on first use and kept in a bounded LRU cache keyed on the cone and its live-in
        boundary, so a recurring change pattern reuses its compiled plan without
        re-fusing. Nodes that were demoted to plain Python at seed time stay Python in
        the cone; the jittable remainder fuses into ``@njit`` segments.

        Interior overrides: a `changed` name may resolve to an interior (computed) node
        rather than an external input -- mirroring the interpreted path. Its value is
        overridden in the store and only its downstream cone recomputes; the overridden
        node's own formula is *not* re-run. The first override of a not-yet-seen interior
        node expands the change-source set and rebuilds the persisted-node boundary (and
        invalidates cached cone plans, whose boundaries have shifted).

        Limitations:

        - Do not interleave input-changing ``kernel(...)`` throughput calls between
          `recompute` calls. `recompute` is the stateful entry point: the store is
          seeded once, and a throughput call does not update it, so a subsequent
          `recompute` would read stale unchanged values. Use `recompute` for the
          incremental workflow and the bare kernel for independent one-shot calls.
        - An interior plain-Python (demoted) node must return a stable numba type across
          recomputes. The same-types contract extends to demoted outputs: a demoted
          node whose output type drifts between recomputes is not supported.

        On a live-in type change a cached cone dispatcher fails to compile; the whole
        cone-plan cache is flushed once, the store re-seeded from the last full call, the
        change re-applied, and the cone rebuilt against the new types.

        Declared kernels enforce a contract instead of recovering. For a kernel
        built from a fully-declared graph, a changed value is checked for numba
        assignability to the node's declared type: a value numba cannot assign to
        the declared type raises a crisp ``declared type X, got Y`` error, while a
        benign difference numba accepts -- a C-contiguous array against an
        ``'A'``-layout array declaration, or a safe scalar promotion -- is accepted.
        The check is convertibility, not type identity, and is scoped to declared
        (eager) kernels; an individually-declared node inside an otherwise
        discovered kernel keeps the flush-and-reseed recovery above.
        """
        self._ensure_store()
        changed_vars = self._apply_changes(changed)
        if not changed_vars:
            return tuple(self._store[v] for v in self._required_vars)
        compiled = self._ctx[0]
        self._ensure_boundary()
        affected = compiled._collect_affected(changed_vars)
        if not affected:
            return tuple(self._store[v] for v in self._required_vars)
        plan = self._cone_plan_cached(affected)
        try:
            plan.run_into(self._store)
        except NumbaError:
            # A live-in type change invalidates every compiled cone plan; flush
            # the whole cache once, reseed the store, re-apply the change against
            # the fresh store, and rebuild against the new types. Declared kernels
            # fix every type by contract, so the (d) contract check already
            # rejected off-contract types crisply; flushing here would be pointless.
            if self.is_declared:
                raise
            self._flush_and_reseed()
            self._apply_changes(changed)
            affected = compiled._collect_affected(changed_vars)
            plan = self._cone_plan_cached(affected)
            plan.run_into(self._store)
        return tuple(self._store[v] for v in self._required_vars)

    def execute(self, external_values: dict) -> dict:
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


def compile_kernel(
    graph: Graph, required: str | list[str], *,
    jit_options: dict | None = None, cache: bool | None = None,
) -> CompiledKernel:
    """Compile `graph` into a fused @njit kernel for the `required` variables.

    :param graph: its dependency structure and formulas are fused
        into one straight-line @njit function (see `CompiledKernel`).
    :param required: Order is preserved and fixes the order of
        `CompiledKernel.outputs` / the kernel's return tuple; a duplicate
        entry raises `ValueError` (each
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
    declared inputs, graphs deeper than the recursion limit). For an undeclared
    (or partially-declared) graph, numba typing problems surface at the kernel's
    first call (auto-njit of plain-Python formulas is lazy). For a fully-declared
    graph (every node carries `params`, every consumed external is typed) the
    mode resolves eagerly here, so type errors move to build time: a formula
    whose natural return at the declared input types is non-convertible to the
    declared type, a cross-node type mismatch, and -- crucially -- a
    coercible-but-wrong `params.type`. The last is caught by an explicit
    unconstrained return-type probe that compares the formula's naturally
    inferred return type against the declaration; binding the formula to the
    declared signature does not catch it, because numba silently coerces a
    convertible scalar (declaring int64 over a `x * 1.5` body would otherwise
    compute 7, not 7.5). Fully-declared graphs thus fail fast at build;
    any-undeclared graph fails at the first call, exactly as today. Runtime
    errors never demote -- they propagate.

    Caching: the kernel digest fingerprints each formula's bytecode,
    constants, default values, closure-cell values, referenced module-level
    globals (including helper functions, recursively), defining module, and
    the effective jit flags. Because the generated source never mentions types,
    a declared graph's signatures are appended to the digest too (the consumed
    external signature for an eager fused kernel, each segment's live-in/out
    signature for an eager segment), so two declared-type variants of one
    type-free graph get distinct cache anchors and never reuse each other's
    binary. A formula whose state cannot be fingerprinted (e.g.
    cres/CompileResultWAP objects, values with no canonical form) downgrades
    that one kernel to cache=False: always recompiled, never stale. When caching
    is enabled, a content-addressed anchor `.py` file is written under numba's
    cache directory; with caching off (or the cache dir unwritable, which warns
    and degrades) nothing is written.

    Non-jittable formulas: for an undeclared graph the first call resolves the
    execution mode. If the fully fused kernel cannot be typed for the actual
    argument types, each node is probed against the real intermediate values;
    nodes whose formulas fail to *compile* (or whose input values numba cannot
    type) run in plain Python, and the jittable remainder is fused into segments
    orchestrated from Python. A declared `jitable=False` node is instead demoted
    by declaration (no probing): a graph mixing declared jittable and
    declared-Python nodes resolves eagerly to a "segmented" plan at build,
    `CompiledKernel.partition` populated immediately with per-node reasons.
    `CompiledKernel.partition` describes the result, including per-node demotion
    reasons; it is `None` before the first call only for an undeclared graph
    (set at build for a fully-declared one). `CompiledKernel.kernel` is the
    hot-path callable: the bare @njit dispatcher once the graph resolves fully
    fused, the Python master when segmented. For an undeclared graph a later
    call whose types break a segment re-learns and replaces the partition (one
    active plan); a declared kernel does not re-discover. The crisp
    `declared type X, got Y` contract is enforced at build time and on
    `recompute()` (`can_convert` against each declared type), not on the
    throughput `kernel(...)` path: throughput retains numba's polymorphic
    widening across calls, and a later `kernel(...)` whose off-contract type
    breaks a jit segment raises numba's own typing error (the declared `_demoted`
    is left untouched -- no silent re-discovery). Once fully fused, fused is
    permanent (a later-signature typing failure raises). The discovery call computes
    jit-node values through per-node dispatchers while later calls use fused
    segments -- identical under default IEEE semantics, but non-default
    jit_options such as fastmath could in principle differ across fusion
    boundaries.
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
    if case in ("A", "B"):
        for node in compiled.ordered_nodes:
            var = node.variable
            if var in external or dispositions.get(var) != "STATIC_JIT":
                continue
            in_types = tuple(i.params.type for i in node.inputs)
            fp, fingerprintable = _formula_fingerprint(var.formula)
            key = (fp, in_types, var.params.type)
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
    # For Case A fold the consumed-external declared signature (kernel-arg order,
    # matching the eager `.compile` below) into the digest so int64/float64
    # variants of one type-free source get distinct anchors. Cases B/C pass ().
    declared_sigs = ()
    if case == "A":
        consumed_sig = tuple(v.params.type for v in external_vars if v in consumed)
        declared_sigs = (consumed_sig,)
    kernel = _compile(source, bindings, jit_options, cache, declared_sigs)
    bindings_by_var = {
        n.variable: bindings["f_" + idents[n.variable]]
        for n in compiled.ordered_nodes if n.variable not in external
    }
    ctx = (compiled, idents, bindings_by_var, jit_options, cache, external)
    identifiers = {v.qual_name(): ident for v, ident in idents.items()}
    if case == "A":
        ck = CompiledKernel(kernel, params, outputs, source, identifiers, ctx,
                            required_vars, external_vars, is_declared=True)
        sig_vars = [v for v in external_vars if v in consumed]
        # Eager-compile only when every kernel param is consumed; a pass-through
        # external output is a kernel arg but has no declared type, so a partial
        # signature would be wrong-arity. Such graphs compile lazily on first call.
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
        steps, segments = [], []
        # Same segment structure as _discover_and_run, but with declared types and a static demotion set (no probing).
        for kind, run_nodes in runs:
            quals = tuple(n.variable.qual_name() for n in run_nodes)
            if kind == "python":
                for n in run_nodes:
                    steps.append(_PyStep(
                        var=n.variable,
                        py_callable=getattr(n.variable.formula, "py_func", n.variable.formula),
                        in_vars=tuple(n.inputs)))
                reasons = {n.variable.qual_name(): demoted[n.variable] for n in run_nodes}
                ins = sorted({i for n in run_nodes for i in n.inputs
                              if i not in {x.variable for x in run_nodes}},
                             key=lambda v: v.qual_name())
                segments.append(Segment(kind="python", nodes=quals,
                                        inputs=tuple(v.qual_name() for v in ins),
                                        outputs=quals, source=None, reasons=reasons))
                continue
            live_in, live_out = segment_liveness(run_nodes, external, required_vars, order)
            src, seg_bindings, _, _ = _generate_segment_body(run_nodes, live_in, live_out, idents, flags)
            seg_sigs = (tuple(v.params.type for v in live_in),
                        tuple(v.params.type for v in live_out))
            disp = _compile(src, seg_bindings, jit_options, cache, seg_sigs)
            disp.compile(tuple(v.params.type for v in live_in))
            steps.append(_JitStep(dispatcher=disp, in_vars=live_in, out_vars=live_out))
            segments.append(Segment(kind="jit", nodes=quals,
                                    inputs=tuple(v.qual_name() for v in live_in),
                                    outputs=tuple(v.qual_name() for v in live_out),
                                    source=src, reasons={}))
        ck = CompiledKernel(kernel, params, outputs, source, identifiers, ctx,
                            required_vars, external_vars, is_declared=True)
        ck._plan = _Plan(steps=tuple(steps), external_vars=tuple(external_vars),
                         output_vars=tuple(required_vars))
        ck._demoted = demoted
        ck._mode = "segmented"
        ck.partition = PartitionReport(mode="segmented", segments=tuple(segments))
        return ck
    return CompiledKernel(
        kernel, params, outputs, source, identifiers, ctx, required_vars, external_vars
    )
