"""Partitioning machinery for segmented compile_kernel.

Private module: discovery (warm-up + probe), fusion-maximizing
linearization, run/liveness partitioning, the master execution plan, and
the PartitionReport surfaced as CompiledKernel.partition. Everything here
operates on `core.variable` Variables/CompiledNodes and plain values; the
only numba interaction is Dispatcher.compile probes and typeof.
"""
from bisect import insort
from dataclasses import dataclass, field
from typing import Any, Callable

from numba import njit, typeof
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaError

from numbox.core.variable.variable import Variable, CompiledNode


@dataclass(frozen=True)
class Segment:
    """One contiguous run of the partition, as reported to users."""
    kind: str                      # "jit" | "python"
    nodes: tuple[str, ...]         # qual_names, linear order
    inputs: tuple[str, ...]        # live-in qual_names
    outputs: tuple[str, ...]       # live-out qual_names
    source: str | None = None      # generated source, jit segments only
    reasons: dict[str, str] = field(default_factory=dict)   # python segments: demotion reasons


@dataclass(frozen=True)
class PartitionReport:
    """What actually runs where, and why -- see CompiledKernel.partition."""
    mode: str                      # "fused" | "segmented"
    segments: tuple[Segment, ...]  # of Segment

    @property
    def python_nodes(self) -> set[str]:
        names = set()
        for seg in self.segments:
            if seg.kind == "python":
                names.update(seg.nodes)
        return names

    def __str__(self) -> str:
        lines = [f"compile_kernel partition: mode={self.mode}, {len(self.segments)} segment(s)"]
        for i, seg in enumerate(self.segments):
            lines.append(
                f"  [{i}] {seg.kind}: nodes={list(seg.nodes)} "
                f"inputs={list(seg.inputs)} outputs={list(seg.outputs)}"
            )
            for qual, why in sorted(seg.reasons.items()):
                lines.append(f"      {qual}: {why}")
        return "\n".join(lines)


@dataclass(frozen=True)
class _JitStep:
    dispatcher: Dispatcher
    in_vars: tuple[Variable, ...]
    out_vars: tuple[Variable, ...]


@dataclass(frozen=True)
class _PyStep:
    var: Variable
    py_callable: Callable
    in_vars: tuple[Variable, ...]


@dataclass(frozen=True)
class _Plan:
    steps: tuple[_JitStep | _PyStep, ...]
    external_vars: tuple[Variable, ...]     # kernel-argument order
    output_vars: tuple[Variable, ...]       # required order

    def run(self, args: tuple) -> tuple:
        slots = dict(zip(self.external_vars, args))
        for step in self.steps:
            vals = [slots[v] for v in step.in_vars]
            if isinstance(step, _JitStep):
                slots.update(zip(step.out_vars, step.dispatcher(*vals)))
            else:
                slots[step.var] = step.py_callable(*vals)
        return tuple(slots[v] for v in self.output_vars)


@dataclass(frozen=True)
class _ConePlan:
    """A recompute sub-plan executed in place against a persistent store.
    Unlike _Plan.run (fresh dict from args, returns output_vars), run_into
    reads live-ins from and writes every step output back to the shared store."""
    steps: tuple[_JitStep | _PyStep, ...]

    def run_into(self, store: dict) -> None:
        for step in self.steps:
            vals = [store[v] for v in step.in_vars]
            if isinstance(step, _JitStep):
                store.update(zip(step.out_vars, step.dispatcher(*vals)))
            else:
                store[step.var] = step.py_callable(*vals)


def _qual(node: CompiledNode) -> str:
    return node.variable.qual_name()


def _linearize_from(nodes: list[CompiledNode], demoted: set[Variable], start_jit: bool) -> list[CompiledNode]:
    by_var = {n.variable: n for n in nodes}
    indeg = {n: 0 for n in nodes}
    dependents = {}
    for n in nodes:
        for inp in n.inputs:
            producer = by_var.get(inp)
            if producer is not None:    # external inputs are pre-satisfied
                indeg[n] += 1
                dependents.setdefault(producer, []).append(n)
    jit_q, py_q = [], []
    for n in sorted((n for n in nodes if indeg[n] == 0), key=_qual):
        (py_q if n.variable in demoted else jit_q).append(n)
    order = []
    on_jit = start_jit
    while jit_q or py_q:
        queue = jit_q if on_jit else py_q
        if not queue:
            on_jit = not on_jit
            continue
        n = queue.pop(0)
        order.append(n)
        for d in sorted(dependents.get(n, []), key=_qual):
            indeg[d] -= 1
            if indeg[d] == 0:
                insort(py_q if d.variable in demoted else jit_q, d, key=_qual)
    return order


def linearize(nodes: list[CompiledNode], demoted: set[Variable]) -> list[CompiledNode]:
    """Topological order clustering same-color nodes into minimal runs.

    Greedy color-sticky Kahn (drain the current color while possible,
    deterministic qual_name tie-break), evaluated from both possible
    starting colors; the candidate with fewer runs wins (jit-start on a
    tie). For two colors this is exactly optimal: starting from color c,
    run i holds precisely the nodes whose longest color-alternating
    ancestor chain spans i runs, so the run count meets the lower bound
    of 1 + the maximum number of color alternations along any directed
    path.
    """
    jit_first = _linearize_from(nodes, demoted, start_jit=True)
    py_first = _linearize_from(nodes, demoted, start_jit=False)
    if len(build_runs(py_first, demoted)) < len(build_runs(jit_first, demoted)):
        return py_first
    return jit_first


def build_runs(order: list[CompiledNode], demoted: set[Variable]) -> list[tuple[str, list[CompiledNode]]]:
    """Split a linear order into maximal same-color runs: [(kind, [nodes])]."""
    runs = []
    for n in order:
        kind = "python" if n.variable in demoted else "jit"
        if runs and runs[-1][0] == kind:
            runs[-1][1].append(n)
        else:
            runs.append((kind, [n]))
    return runs


def segment_liveness(
    run_nodes: list[CompiledNode],
    external: set[Variable],
    required_vars: list[Variable],
    order: list[CompiledNode],
) -> tuple[tuple[Variable, ...], tuple[Variable, ...]]:
    """(live_in, live_out) for one jit run, both sorted by qual_name.

    live_in: values the run consumes but does not produce (externals or
    earlier-step products). live_out: values the run produces that a later
    step consumes or that are required outputs.
    """
    produced = {n.variable for n in run_nodes}
    live_in = set()
    for n in run_nodes:
        for inp in n.inputs:
            if inp not in produced:
                live_in.add(inp)
    after = set()
    seen_any = False
    run_set = set(run_nodes)
    for n in order:
        if n in run_set:
            seen_any = True
            continue
        if seen_any:
            after.update(n.inputs)
    live_out = {v for v in produced if v in after or v in set(required_vars)}
    key = lambda v: v.qual_name()   # noqa: E731 - tiny local sort key
    return tuple(sorted(live_in, key=key)), tuple(sorted(live_out, key=key))


def cone_liveness(
    run_nodes: list[CompiledNode], cone_order: list[CompiledNode],
    required_vars, boundary: set[Variable],
) -> tuple[tuple[Variable, ...], tuple[Variable, ...]]:
    """(live_in, live_out) for one jit run inside a recompute cone.

    live_in: inputs consumed by the run but not produced within it (read from
    the store). live_out: produced values that are consumed by a later cone
    step, are required outputs, or are boundary nodes (persisted for other
    cones). Both sorted by qual_name."""
    produced = {n.variable for n in run_nodes}
    live_in = set()
    for n in run_nodes:
        for inp in n.inputs:
            if inp not in produced:
                live_in.add(inp)
    later = set()
    seen = False
    run_set = set(run_nodes)
    for n in cone_order:
        if n in run_set:
            seen = True
            continue
        if seen:
            later.update(n.inputs)
    req = set(required_vars)
    live_out = {v for v in produced if v in later or v in req or v in boundary}
    key = lambda v: v.qual_name()   # noqa: E731 - tiny local sort key
    return tuple(sorted(live_in, key=key)), tuple(sorted(live_out, key=key))


_REASON_LIMIT = 200


def _error_reason(exc: Exception) -> str:
    """First informative line of a numba error -- numba prefixes typing
    failures with a generic pipeline line, which would make every reason
    in a PartitionReport read identically."""
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    informative = next(
        (ln for ln in lines if not ln.startswith("Failed in nopython mode pipeline")),
        lines[0] if lines else "",
    )
    return f"{type(exc).__name__}: {informative[:_REASON_LIMIT]}"


def _untypeable_reason(node: CompiledNode, values: dict[Variable, Any]) -> str | None:
    for inp in node.inputs:
        try:
            typeof(values[inp])
        except (ValueError, TypeError):
            return (
                f"input '{inp.qual_name()}' value of type "
                f"{type(values[inp]).__name__} is not numba-typeable"
            )
    return None


def _call_exotic(binding, args: list, arg_types: tuple, flags: dict | None = None) -> Any:
    """Evaluate a CompileResultWAP/CFunc/DUFunc formula through a one-line
    @njit shim (the same global-binding shape segments use). No Python
    fallback exists for these, so a NumbaError here propagates. `flags` are the
    kernel's effective jit flags, so the shim matches the fused/segment hot path."""
    names = ", ".join(f"a{i}" for i in range(len(args)))
    ns = {"_formula": binding}
    exec(f"def _shim({names}):\n    return _formula({names})\n", ns)  # nosec B102
    shim = njit(**(flags or {}))(ns["_shim"])
    shim.compile(arg_types)
    return shim(*args)


def _dominators(nodes: list[CompiledNode], sources: set[Variable]) -> dict[Variable, frozenset]:
    """Dominator set per node for the DAG rooted at a virtual super-source
    (represented by None) with an edge to every node that is in `sources` or
    has an external input. `nodes` must be topologically ordered. Each node's
    set includes the node itself."""
    by_var = {n.variable: n for n in nodes}
    dom: dict[Variable, frozenset] = {}
    for n in nodes:
        var = n.variable
        seeds = [dom[inp] for inp in n.inputs if inp in by_var]
        if var in sources or any(inp not in by_var for inp in n.inputs):
            seeds.append(frozenset({None}))
        inter = frozenset.intersection(*seeds) if seeds else frozenset({None})
        dom[var] = inter | {var}
    return dom


def compute_boundary(
    nodes: list[CompiledNode], sources: set[Variable], required: set[Variable],
) -> set[Variable]:
    """Variables that must live in the recompute store: required outputs, plus
    any node that fails to dominate a consumer (some change in `sources` can
    reach the consumer without reaching the node, making it a cone live-in).
    Nodes that dominate all their consumers are fuse-through (not persisted)."""
    by_var = {n.variable: n for n in nodes}
    dom = _dominators(nodes, sources)
    consumers: dict[Variable, list[Variable]] = {}
    for n in nodes:
        for inp in n.inputs:
            if inp in by_var:
                consumers.setdefault(inp, []).append(n.variable)
    boundary = set(required)
    for n in nodes:
        var = n.variable
        if any(var not in dom[c] for c in consumers.get(var, ())):
            boundary.add(var)
    return boundary


def discover(
    ordered_nodes: list[CompiledNode],
    external: set[Variable],
    values: dict[Variable, Any],
    bindings_by_var: dict[Variable, Any],
    flags: dict | None = None,
) -> dict[Variable, str]:
    """One-pass warm-up + probe.

    Mutates `values` ({Variable: value}, pre-seeded with externals) to hold
    every node's value; returns {Variable: reason} for demoted nodes.
    Numba *compile* errors demote; runtime errors propagate.
    """
    demoted = {}
    for node in ordered_nodes:
        var = node.variable
        if var in external:
            continue
        args = [values[inp] for inp in node.inputs]
        binding = bindings_by_var[var]
        reason = _untypeable_reason(node, values)
        if reason is None:
            arg_types = tuple(typeof(a) for a in args)
            if isinstance(binding, Dispatcher):
                try:
                    binding.compile(arg_types)
                except NumbaError as e:
                    reason = _error_reason(e)
            else:
                values[var] = _call_exotic(binding, args, arg_types, flags)
                continue
        elif not isinstance(binding, Dispatcher):
            raise TypeError(
                f"{var.qual_name()!r}: formula {binding!r} has no Python fallback "
                f"and {reason}"
            )
        if reason is None:
            values[var] = binding(*args)
        else:
            demoted[var] = reason
            py = getattr(var.formula, "py_func", var.formula)
            values[var] = py(*args)
    return demoted
