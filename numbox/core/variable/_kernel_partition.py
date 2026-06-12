"""Partitioning machinery for segmented compile_kernel.

Private module: discovery (warm-up + probe), fusion-maximizing
linearization, run/liveness partitioning, the master execution plan, and
the PartitionReport surfaced as CompiledKernel.partition. Everything here
operates on `core.variable` Variables/CompiledNodes and plain values; the
only numba interaction is Dispatcher.compile probes and typeof.
"""
from bisect import insort
from dataclasses import dataclass, field

from numba import njit, typeof
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaError


@dataclass(frozen=True)
class Segment:
    """One contiguous run of the partition, as reported to users."""
    kind: str                      # "jit" | "python"
    nodes: tuple                   # qual_names, linear order
    inputs: tuple                  # live-in qual_names
    outputs: tuple                 # live-out qual_names
    source: str = None             # generated source, jit segments only
    reasons: dict = field(default_factory=dict)   # python segments: demotion reasons


@dataclass(frozen=True)
class PartitionReport:
    """What actually runs where, and why -- see CompiledKernel.partition."""
    mode: str                      # "fused" | "segmented"
    segments: tuple                # of Segment

    @property
    def python_nodes(self):
        names = set()
        for seg in self.segments:
            if seg.kind == "python":
                names.update(seg.nodes)
        return names

    def __str__(self):
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
    dispatcher: object
    in_vars: tuple
    out_vars: tuple


@dataclass(frozen=True)
class _PyStep:
    var: object
    py_callable: object
    in_vars: tuple


@dataclass(frozen=True)
class _Plan:
    steps: tuple
    external_vars: tuple           # kernel-argument order
    output_vars: tuple             # required order

    def run(self, args):
        slots = dict(zip(self.external_vars, args))
        for step in self.steps:
            vals = [slots[v] for v in step.in_vars]
            if isinstance(step, _JitStep):
                slots.update(zip(step.out_vars, step.dispatcher(*vals)))
            else:
                slots[step.var] = step.py_callable(*vals)
        return tuple(slots[v] for v in self.output_vars)


def _qual(node):
    return node.variable.qual_name()


def _linearize_from(nodes, demoted, start_jit):
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


def linearize(nodes, demoted):
    """Topological order clustering same-color nodes into few long runs.

    Greedy color-sticky Kahn (drain the current color while possible,
    deterministic qual_name tie-break), evaluated from both possible
    starting colors; the candidate with fewer runs wins (jit-start on a
    tie). Exact run-count minimization is NP-hard; this is the documented
    heuristic.
    """
    jit_first = _linearize_from(nodes, demoted, start_jit=True)
    py_first = _linearize_from(nodes, demoted, start_jit=False)
    if len(build_runs(py_first, demoted)) < len(build_runs(jit_first, demoted)):
        return py_first
    return jit_first


def build_runs(order, demoted):
    """Split a linear order into maximal same-color runs: [(kind, [nodes])]."""
    runs = []
    for n in order:
        kind = "python" if n.variable in demoted else "jit"
        if runs and runs[-1][0] == kind:
            runs[-1][1].append(n)
        else:
            runs.append((kind, [n]))
    return runs


def segment_liveness(run_nodes, external, required_vars, order):
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


_REASON_LIMIT = 200


def _error_reason(exc):
    first = str(exc).splitlines()[0] if str(exc) else ""
    return f"{type(exc).__name__}: {first[:_REASON_LIMIT]}"


def _untypeable_reason(node, values):
    for inp in node.inputs:
        try:
            typeof(values[inp])
        except (ValueError, TypeError):
            return (
                f"input '{inp.qual_name()}' value of type "
                f"{type(values[inp]).__name__} is not numba-typeable"
            )
    return None


def _call_exotic(binding, args, arg_types):
    """Evaluate a CompileResultWAP/CFunc/DUFunc formula through a one-line
    @njit shim (the same global-binding shape segments use). No Python
    fallback exists for these, so a NumbaError here propagates."""
    names = ", ".join(f"a{i}" for i in range(len(args)))
    ns = {"_formula": binding}
    exec(f"def _shim({names}):\n    return _formula({names})\n", ns)  # nosec B102
    shim = njit(ns["_shim"])
    shim.compile(arg_types)
    return shim(*args)


def discover(ordered_nodes, external, values, bindings_by_var):
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
        arg_types = None
        if reason is None:
            arg_types = tuple(typeof(a) for a in args)
            if isinstance(binding, Dispatcher):
                try:
                    binding.compile(arg_types)
                except NumbaError as e:
                    reason = _error_reason(e)
            else:
                values[var] = _call_exotic(binding, args, arg_types)
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
