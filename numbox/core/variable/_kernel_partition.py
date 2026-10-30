"""Split a kernel into compiled segments and Python formulas, and run the parts; a private module of compile_kernel.

The search, `discover`, runs every formula once to find the formulas that numba cannot compile. `linearize` orders the
nodes so that compiled and Python formulas form the fewest runs, and the liveness functions give each segment its
inputs and outputs. The plans run the segments and the Python formulas in turn, and `PartitionReport`, which
`CompiledKernel.partition` holds, reports the split. The code works on the variables and nodes of
`numbox.core.variable.variable` and on plain values. It uses numba only to type values, to compile formulas and to
call compiled code.
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
    """One part of a partition report: consecutive variables that numba compiles together, or that run in Python.

    `kind` is "jit" or "python". `nodes` holds the qualified names of the segment's variables in the order they run,
    `inputs` those of the values it reads, and `outputs` those of the values it provides to later steps. `source` is
    the generated source of a jit segment, and `reasons` maps each variable of a python segment to the reason it runs
    in Python.
    """
    kind: str
    nodes: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    source: str | None = None
    reasons: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PartitionReport:
    """The split of a kernel into compiled segments and Python formulas, which ``CompiledKernel.partition`` holds.

    `mode` is "fused" for a kernel that numba compiled as one function, and "segmented" otherwise. `segments` holds
    the segments in the order they run. The text of a report lists them, with the reason under each variable that runs
    in Python.
    """
    mode: str
    segments: tuple[Segment, ...]

    @property
    def python_nodes(self) -> set[str]:
        """The qualified names of the variables that run in Python."""
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
    # The external variables are in the order of the kernel's arguments, and the outputs in the order of the request.
    external_vars: tuple[Variable, ...]
    output_vars: tuple[Variable, ...]

    def run(self, args: tuple) -> tuple:
        """Run the steps on the external values `args`, in the order of `external_vars`, and return `output_vars`."""
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
    """The steps that recompute the variables a change affects, run against the store of the kernel.

    Unlike `_Plan.run`, which starts from the arguments and returns the outputs, `run_into` reads the inputs of each
    step from the store and writes the outputs of each step back into it.
    """
    steps: tuple[_JitStep | _PyStep, ...]

    def run_into(self, store: dict) -> None:
        """Run the steps, reading the inputs of each step from `store` and writing its outputs into `store`."""
        for step in self.steps:
            vals = [store[v] for v in step.in_vars]
            if isinstance(step, _JitStep):
                store.update(zip(step.out_vars, step.dispatcher(*vals)))
            else:
                store[step.var] = step.py_callable(*vals)


def _qual(node: CompiledNode) -> str:
    return node.variable.qual_name()


def _linearize_from(nodes: list[CompiledNode], demoted: set[Variable], start_jit: bool) -> list[CompiledNode]:
    # Kahn's topological sort that stays with one kind of node while it can: it takes the ready nodes of the current
    # kind, in qualified-name order, until none is ready, and only then switches to the other kind. `start_jit`
    # chooses the first kind. With the first kind fixed, each node lands in the earliest run that its inputs allow,
    # so no order that starts with that kind has fewer runs.
    by_var = {n.variable: n for n in nodes}
    indeg = {n: 0 for n in nodes}
    dependents = {}
    for n in nodes:
        for inp in n.inputs:
            producer = by_var.get(inp)
            if producer is not None:    # An external input is ready from the start.
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
    """Return `nodes` in an order that puts each node after its inputs, in the fewest runs.

    A run is a sequence of consecutive nodes that are all compiled, or all in `demoted` and so run in Python. Ties
    between ready nodes are broken by qualified name, so the order is deterministic.
    """
    # Each start gives the fewest runs for its first kind, so the better of the two gives the fewest runs overall. A
    # tie goes to the order that starts with compiled nodes.
    jit_first = _linearize_from(nodes, demoted, start_jit=True)
    py_first = _linearize_from(nodes, demoted, start_jit=False)
    if len(build_runs(py_first, demoted)) < len(build_runs(jit_first, demoted)):
        return py_first
    return jit_first


def build_runs(order: list[CompiledNode], demoted: set[Variable]) -> list[tuple[str, list[CompiledNode]]]:
    """Split `order` into runs of consecutive nodes of one kind, and return them as ``(kind, nodes)`` pairs.

    `kind` is "python" for a run of nodes in `demoted` and "jit" for any other run. Each run is as long as possible.
    """
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
    """Return the variables that a compiled segment reads and those that it returns, each sorted by qualified name.

    The segment reads every input of its nodes that it does not produce itself: an external value or the output of an
    earlier step. It returns every variable that it produces and that a later step in `order` reads or that is
    requested.
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
    """Return what a compiled segment of a recompute plan reads and returns, each sorted by qualified name.

    The segment reads, from the store, every input of its nodes that it does not produce itself. It returns every
    variable that it produces and that a later step in `cone_order` reads, that is requested, or that is in
    `boundary`, the variables whose values the store keeps current for other recompute plans.
    """
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
    """Return the type of `exc` and the first informative line of its message, cut to `_REASON_LIMIT` characters.

    numba opens the message of a typing error with a line about its nopython pipeline that every such error shares.
    The function skips that line, because otherwise every reason in a partition report would read the same.
    """
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
    """Call a cres, @cfunc or @vectorize formula on `args` through a one-line @njit function, and return the result.

    The function calls the formula through a global, as a segment does. numba compiles it for `arg_types` under
    `flags`, the kernel's jit options, so that the call computes as the fused kernel and the segments do. Such a formula
    has no Python version to fall back to, so a numba error propagates.
    """
    names = ", ".join(f"a{i}" for i in range(len(args)))
    ns = {"_formula": binding}
    exec(f"def _shim({names}):\n    return _formula({names})\n", ns)  # nosec B102
    shim = njit(**(flags or {}))(ns["_shim"])
    shim.compile(arg_types)
    return shim(*args)


def _dominators(nodes: list[CompiledNode], sources: set[Variable]) -> dict[Variable, frozenset]:
    """Return the set of dominators of each node, in the graph of `nodes` with a virtual source, None, added.

    The virtual source has an edge to every node that is in `sources` or reads an external value. A node d dominates
    a node v if every path from the virtual source to v passes through d. The set of each node holds the node itself.
    `nodes` must be in topological order.
    """
    by_var = {n.variable: n for n in nodes}
    dom: dict[Variable, frozenset] = {}
    for n in nodes:
        # The dominators of a node are the node itself and the dominators that all its predecessors share. The
        # virtual source is a predecessor of a node in `sources` and of a node with an external input.
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
    """Return the variables whose values recompute plans must keep current in the store.

    These are the variables in `required`, and every variable that fails to dominate one of the variables that read
    it. A change in `sources` can then reach that reader without passing through the variable, so a recompute plan
    reads the variable from the store. A plan recomputes a variable that dominates all its readers whenever it
    recomputes them, and passes the value to them within the plan.
    """
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


def _evaluate(
    ordered_nodes: list[CompiledNode],
    external: set[Variable],
    values: dict[Variable, Any],
    bindings_by_var: dict[Variable, Any],
    flags: dict | None,
    demoted: set[Variable] | dict[Variable, str],
) -> None:
    """Calculate the value of every calculated variable into `values`, keeping `demoted` as given, without a search.

    A cres, @cfunc or @vectorize formula always runs through the function of `_call_exotic`. A formula in `demoted`
    runs its Python function, and any other formula runs its dispatcher. `CompiledKernel._ensure_store` calls this to
    fill the store at the first recompute of a kernel built from declared types. The declarations give the demoted
    set, so no search is needed. This function is not a fallback after a numba error. On a kernel without declared
    types, `discover` searches after such an error, and a kernel built from declared types never searches.
    """
    for node in ordered_nodes:
        var = node.variable
        if var in external:
            continue
        args = [values[inp] for inp in node.inputs]
        binding = bindings_by_var[var]
        if not isinstance(binding, Dispatcher):
            arg_types = tuple(typeof(a) for a in args)
            values[var] = _call_exotic(binding, args, arg_types, flags)
        elif var in demoted:
            py = getattr(var.formula, "py_func", var.formula)
            values[var] = py(*args)
        else:
            values[var] = binding(*args)


def discover(
    ordered_nodes: list[CompiledNode],
    external: set[Variable],
    values: dict[Variable, Any],
    bindings_by_var: dict[Variable, Any],
    flags: dict | None = None,
) -> dict[Variable, str]:
    """Run every formula once, in order, and return the variables whose formulas must run in Python, with the reasons.

    `values` holds the external values on entry, and the value of every variable on return. numba compiles each
    formula separately, for the types of its input values. A formula runs in Python if numba fails to compile it or
    cannot type one of its input values, and the returned dict maps each such variable to the reason. A cres,
    @cfunc or @vectorize formula has no Python version: numba's error propagates, and an input value that numba cannot
    type raises ``TypeError``. An exception that a formula raises as it runs propagates, whether it runs compiled or in
    Python.
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
