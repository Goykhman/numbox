"""Build a graph of calculations from variable specifications, and evaluate it in Python.

A :class:`Graph` holds the variables in namespaces: a :class:`Variables` namespace for the variables that formulas
calculate, and an :class:`External` namespace for each external source of data. :meth:`Graph.compile` orders the
variables that a request needs into a :class:`CompiledGraph`, whose :meth:`~CompiledGraph.execute` and
:meth:`~CompiledGraph.recompute` call the formulas from Python and write the values into a :class:`Values` instance.
The module imports nothing from numba; :mod:`numbox.core.variable.compile_kernel` compiles the same graph into one
numba function.
"""
import sys
import warnings

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Iterator, Mapping,
    Protocol, TypeAlias, TypedDict
)


class Namespace(ABC):
    """A collection of variables keyed by name, the base class of :class:`Variables` and :class:`External`.

    A variable's qualified name joins the namespace's `name` and the variable's name with a dot. Iterating over a
    namespace yields the names of its variables.
    """
    name: str
    _variables: dict[str, 'Variable']

    def keys(self):
        """Return the names of the variables that the namespace holds."""
        return self._variables.keys()

    def update(self, key: str, var: 'Variable') -> None:
        """Store `var` under the name `key`, replacing any variable stored under that name.

        :meth:`External.declare` uses it to add a variable created after the namespace. A request that the graph has
        already compiled keeps the variable it found then.
        """
        self._variables[key] = var

    def __contains__(self, key: str) -> bool:
        """Return True if the namespace holds a variable named `key`, an unqualified name.

        A variable found there has this namespace's name as its source, so its qualified name starts with that name.
        Unlike a lookup with ``[]`` in an :class:`External` namespace, ``in`` never creates the variable.
        """
        return key in self._variables

    @abstractmethod
    def __getitem__(self, key: str) -> 'Variable':
        pass

    def __iter__(self) -> Iterator[str]:
        return iter(self._variables.keys())


class Storage(Protocol):
    """The storage that a :class:`CompiledGraph` calculates into: a :class:`Value` for each variable.

    :class:`Values` implements it, and so does any class with the same methods.
    """
    _values: dict['Variable', 'Value']

    def get(self, variable: 'Variable') -> 'Value':
        """Return the :class:`Value` of `variable`, and create it, holding ``_null``, on the first request for it."""

    def __iter__(self) -> Iterator['Variable']:
        pass


class VarSpecBase(TypedDict):
    """The required key of a variable specification: ``name``, the name of the variable."""
    name: str


@dataclass(frozen=True)
class Params:
    """The declaration of a variable for compile_kernel: whether numba compiles its formula, and its type.

    ``jitable=False`` declares a formula that runs in Python. `type` is the numba type of the variable's value, and
    None leaves it undeclared. :func:`~numbox.core.variable.compile_kernel.compile_kernel` uses the declarations only
    when every calculated variable in the requested part of the graph is declared with its type, and every external
    variable that a formula reads has a declared type; otherwise it ignores them all.

    A graph caches what it compiles, so a declaration must be in place before the first :meth:`Graph.compile` or
    compile_kernel of a request that involves the variable, as a formula must. A declaration added later has no
    effect on that request.
    """
    jitable: bool = True
    type: Any = None


class VarSpec(VarSpecBase, total=False):
    """A variable specification, the dictionary from which a :class:`Variables` namespace creates a :class:`Variable`.

    Besides ``name``, it may hold ``inputs``, ``formula``, ``metadata`` and ``params``, which have the meanings of the
    parameters of :class:`Variable`. The namespace sets the variable's ``source``.
    """
    inputs: dict[str, str]
    formula: Callable
    metadata: str
    params: Params


VarValue: TypeAlias = Any


class _Null:
    """The class of ``_null``, the marker that stands for the value of a variable not yet calculated."""
    pass


_null = _Null()


QUAL_SEP = "."


def make_qual_name(namespace_name: str, var_name: str) -> str:
    """Return the qualified name of a variable: `namespace_name` and `var_name` joined by a dot."""
    return f"{namespace_name}{QUAL_SEP}{var_name}"


class External(Namespace):
    """A namespace for the variables of one external source of data, whose values the caller supplies.

    A lookup of a name that the namespace does not hold creates a variable of that name, without inputs or formula,
    and stores it. The graph finds the external variables that a calculation needs through these lookups, so a
    misspelled name becomes one more external variable instead of an error.
    """
    def __init__(self, name: str):
        self.name = name
        self._variables: dict[str, 'Variable'] = {}

    def __getitem__(self, name) -> 'Variable':
        """Return the variable named `name`, an unqualified name, and create it if the namespace does not hold it.

        The variable's source is this namespace's name. External variables are best created through this lookup, or
        through :meth:`declare` when they need a declared type.
        """
        variable = self._variables.get(name)
        if variable is None:
            variable = Variable(
                name=name,
                source=self.name
            )
            self._variables[name] = variable
        return variable

    def declare(self, name: str, params: Params) -> 'Variable':
        """Create the variable `name` with the declaration `params`, store it, and return it.

        A lookup creates an external variable without a declaration, so this method is the supported way to declare
        the type of one. It replaces a variable already stored under `name`. Declare before the first compile of a
        request that involves the variable, because the graph keeps what it compiled.
        """
        variable = Variable(name=name, source=self.name, params=params)
        self.update(name, variable)
        return variable


@dataclass(frozen=True)
class Variable:
    """A node of the graph: a value that the caller supplies, or that a formula calculates from the values of inputs.

    A variable is best created by its namespace: a :class:`Variables` namespace creates one for each specification
    passed to it, and an :class:`External` namespace creates one when a name is first looked up in it. Two variables
    are equal, and have the same hash, when their names and sources are equal. A calculated value can be None, so the
    marker ``_null`` stands for a value not yet calculated.

    :param name: the variable's name, unique within its namespace. It must not contain `QUAL_SEP`, a dot.
    :param source: the name of the namespace that holds the variable.
    :param inputs: maps the name of each input variable to the name of the namespace that holds it. The formula
        receives the values of the inputs as positional arguments, in this order.
    :param formula: the function that calculates the variable's value from the values of its inputs, or None, as for
        an external variable.
    :param metadata: text that describes the variable. :meth:`Graph.explain` includes it, and a compiled graph in debug
        mode prints it.
    :param params: the declaration of the variable for compile_kernel, a :class:`Params`.
    :raises ValueError: if `name` contains `QUAL_SEP`.
    """
    name: str
    source: str = field(default="")
    inputs: Mapping[str, str] = field(default_factory=lambda: {})
    formula: Callable = field(default=None)
    metadata: str | None = field(default=None)
    params: Params | None = field(default=None)

    def __post_init__(self):
        # The check restricts only the variable name. numbox splits a qualified name with rsplit(QUAL_SEP, 1), at its
        # last separator, so a namespace name with a dot splits correctly, while a variable name with a dot would not.
        if QUAL_SEP in self.name:
            raise ValueError(
                f"Variable name {self.name!r} contains the reserved qualified-name "
                f"separator {QUAL_SEP!r}; variable names must not contain it "
                f"(qualified names are decomposed via rsplit({QUAL_SEP!r}, 1))."
            )

    def __hash__(self):
        return hash((self.source, self.name))

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name and self.source == other.source

    def qual_name(self) -> str:
        """Return the qualified name: the name of the variable's namespace and its own name, joined by a dot."""
        return make_qual_name(self.source, self.name)


class Variables(Namespace):
    """A namespace of calculated variables, created from variable specifications.

    :param name: the namespace's name, which becomes the source of each of its variables.
    :param variables: the specifications, a :class:`VarSpec` dictionary for each variable. A later specification
        with the same name replaces an earlier one.
    :raises TypeError: if a specification has a key that :class:`Variable` does not take, ``source`` included.
    """
    def __init__(
        self,
        name: str,
        variables: list[VarSpec],
    ):
        self.name = name
        self._variables = {variable["name"]: Variable(source=self.name, **variable) for variable in variables}

    def __getitem__(self, variable_name: str) -> Variable:
        """Return the variable named `variable_name`, an unqualified name.

        Raises ``KeyError`` if the namespace does not hold it.
        """
        try:
            return self._variables[variable_name]
        except KeyError as e:
            raise KeyError(f"{variable_name} is not stored in {self.name}.") from e


@dataclass
class Value:
    """The value of one variable in one :class:`Values` instance, ``_null`` until a calculation sets it.

    It is best created through :meth:`Values.get`.
    """
    variable: Variable
    value: VarValue | _Null = field(default=_null)


class Values:
    """The values of one calculation: a :class:`Value` for each variable, external or calculated.

    The instance starts empty and implements :class:`Storage`.
    """
    def __init__(self):
        self._values: dict[Variable, Value] = {}

    def get(self, variable: Variable) -> Value:
        """Return the :class:`Value` of `variable`, and create it, holding ``_null``, on the first request for it."""
        if variable not in self._values:
            self._values[variable] = Value(variable=variable)
        return self._values[variable]

    def __iter__(self) -> Iterator[Variable]:
        return iter(self._values.keys())


@dataclass(frozen=True)
class CompiledNode:
    """A variable of a :class:`CompiledGraph`, with its inputs: the variables whose values its formula takes, in order.

    Two nodes are equal, and have the same hash, when their variables have the same name and source.
    """
    variable: Variable
    inputs: list[Variable]

    def __hash__(self):
        return hash((self.variable.source, self.variable.name))

    def __eq__(self, other):
        return (
            isinstance(other, CompiledNode) and
            self.variable.name == other.variable.name and
            self.variable.source == other.variable.source
        )


@dataclass(frozen=True)
class CompiledGraph:
    """The variables that a request needs, each after its inputs, ready to be calculated from Python.

    :meth:`Graph.compile` creates it. `ordered_nodes` holds a :class:`CompiledNode` for each variable, the external
    ones included, and `required_external_variables` maps each external source to the variables that it must supply,
    as ``{source: {name: variable}}``. If `debug` is True, the instance prints each node and its metadata to stderr
    before the node's formula runs. The instance fills `dependents`, which maps each variable to the nodes that read
    it, when it is created, and it fills `affected_cache` as :meth:`recompute` runs.
    """
    ordered_nodes: list[CompiledNode]
    required_external_variables: dict[str, dict[str, Variable]]
    debug: bool = False
    dependents: dict[Variable, list[CompiledNode]] = field(default_factory=lambda: {})
    affected_cache: dict[frozenset[Variable], list[CompiledNode]] = field(default_factory=lambda: {})

    def __post_init__(self):
        for node in self.ordered_nodes:
            for inp in node.inputs:
                self.dependents.setdefault(inp, []).append(node)

    def execute(
        self,
        external_values: dict[str, dict[str, VarValue]],
        values: Storage,
    ):
        """Calculate every variable of the compiled graph, and write the values into `values`.

        :param external_values: the values of the required external variables, as ``{source: {name: value}}``. It may
            hold more values than the calculation needs.
        :param values: the storage to write into, such as a :class:`Values` instance.
        :raises KeyError: if `external_values` lacks a required source or variable.
        :raises RuntimeError: if an input of a formula holds ``_null``, as a variable without a formula does unless
            `values` already holds a value for it.

        The method writes the external values first. Then each formula runs after the formulas of its inputs. A
        variable without a formula keeps the value that `values` holds for it.
        """
        self._assign_external_values(external_values, values)
        self._calculate(self.ordered_nodes, values)

    def _assign_external_values(
        self,
        external_values: dict[str, dict[str, VarValue]],
        values: Storage
    ):
        """Write the value of each required external variable into `values`.

        :param external_values: the external values, as ``{source: {name: value}}``.
        :param values: the storage of the calculation, such as a :class:`Values` instance.
        :raises KeyError: if a required source or variable is missing from `external_values`.
        """
        for source_name, variables in self.required_external_variables.items():
            provided = external_values.get(source_name)
            if provided is None:
                raise KeyError(f"Missing external source '{source_name}'")
            for var_name, variable in variables.items():
                if var_name not in provided:
                    raise KeyError(
                        f"Missing value for external variable '{make_qual_name(source_name, var_name)}'"
                    )
                var_value = provided[var_name]
                values.get(variable).value = var_value

    def _collect_affected(self, changed_vars: set[Variable]) -> list[CompiledNode]:
        """Return the nodes that depend on `changed_vars`, directly or indirectly, in the order of `ordered_nodes`.

        A changed variable is among them only if it depends on another changed variable. The compiled graph caches the
        result for each set of changed variables.
        """
        key = frozenset(changed_vars)
        cached = self.affected_cache.get(key)
        if cached is not None:
            return cached
        affected = set()
        stack = list(changed_vars)
        while stack:
            var = stack.pop()
            for node in self.dependents.get(var, []):
                if node not in affected:
                    affected.add(node)
                    stack.append(node.variable)
        affected_list = [node for node in self.ordered_nodes if node in affected]
        self.affected_cache[key] = affected_list
        return affected_list

    def _calculate(self, nodes: list[CompiledNode], values: Storage):
        """Calculate the variable of each node in `nodes` from the values of its inputs, and write it into `values`.

        Every input must hold a value other than ``_null`` before a formula reads it, so `nodes` must be in
        topological order: `ordered_nodes`, or an ordered subset of it such as the one that :meth:`recompute` passes.
        The method skips a node without a formula. Raises ``RuntimeError`` if an input holds ``_null``.
        """
        for node in nodes:
            if node.variable.formula is None:
                continue
            args = [_null] * len(node.inputs)
            for i, input_ in enumerate(node.inputs):
                arg = values.get(input_).value
                if arg is _null:
                    raise RuntimeError(f"Uninitialized input {input_.qual_name()} for {node.variable}")
                args[i] = arg
            if self.debug:
                print(f"Calculating {node}\nwith metadata\n{node.variable.metadata}", file=sys.stderr, flush=True)
            result = node.variable.formula(*args)
            values.get(node.variable).value = result

    def recompute(self, changed: dict[str, dict[str, VarValue]], values: Storage):
        """Write the changed values into `values`, and calculate again the variables that depend on them.

        :param changed: the new values, as ``{source: {name: value}}``. A name may belong to an :class:`External` or
            to a :class:`Variables` namespace.
        :param values: the storage that holds the values of an earlier calculation.

        Each variable that depends on a changed variable is reset and calculated again from its formula, so the graph
        decides the final values. A given value of a calculated variable is therefore kept only if the variable does
        not depend on another variable changed in the same call; otherwise its formula replaces the value. A name that
        the compiled graph does not hold changes nothing, and a warning says that it is not in the calculation path.
        """
        changed_vars = set()
        for src, vals in changed.items():
            for name, val in vals.items():
                variable = self.required_external_variables.get(src, {}).get(name)
                qual_name = make_qual_name(src, name)
                if variable is None:
                    try:
                        variable = next(n.variable for n in self.ordered_nodes if n.variable.qual_name() == qual_name)
                    except StopIteration:
                        warnings.warn(f"{qual_name} is not in the calculation path, update has no effect.")
                        continue
                values.get(variable).value = val
                changed_vars.add(variable)
        affected_nodes = self._collect_affected(changed_vars)
        for node in affected_nodes:
            values.get(node.variable).value = _null
        self._calculate(affected_nodes, values)


class Graph:
    """A graph of variables, built from variable specifications, that orders the variables of each request.

    :param variables_lists: maps the name of each :class:`Variables` namespace to the specifications of its variables,
        :class:`VarSpec` dictionaries.
    :param external_source_names: the names of the external sources whose variables the formulas may read. Each becomes
        an :class:`External` namespace.
    :raises ValueError: if an external source has the name of a :class:`Variables` namespace.

    `registry` maps the name of each namespace to the namespace, and `external` maps the name of each external source
    to its namespace. `compiled_graphs` caches the result of :meth:`compile` for each set of requested names.
    """
    def __init__(
        self,
        variables_lists: dict[str, list[VarSpec]],
        external_source_names: list[str]
    ):
        self.external_source_names: list[str] = external_source_names
        self.registry: dict[str, Namespace] = {}
        self.external: dict[str, External] = {
            external_source_name: External(external_source_name) for external_source_name in external_source_names
        }
        for variables_name, variables_list in variables_lists.items():
            variables = Variables(
                name=variables_name,
                variables=variables_list
            )
            self.registry[variables_name] = variables
        for external_name, external_ in self.external.items():
            if external_name in self.registry:
                raise ValueError(f"Cannot use {external_name} for external, it's already used by variables_lists")
            self.registry[external_name] = external_
        self.compiled_graphs = {}
        self.reverse_dependencies = None

    def compile(self, required: list[str] | str, debug: bool = False) -> CompiledGraph:
        """Return a :class:`CompiledGraph` of the variables that `required` needs, each after its inputs.

        :param required: the qualified name of a variable to calculate, or a list of them.
        :param debug: if True, the compiled graph prints each node and its metadata to stderr before its formula runs.
        :raises ValueError: if a name is not qualified.
        :raises KeyError: if a name, or an input of a needed variable, lies in an unknown namespace or is missing from
            its :class:`Variables` namespace. An :class:`External` namespace creates any name looked up in it.
        :raises RuntimeError: if the needed variables form a cycle.
        :raises RecursionError: if a chain of dependencies is deeper than Python's recursion limit.

        The graph caches the result for each set of names, in any order, and returns the cached compiled graph for a
        later request of the same names, whatever `debug` that request passes. A variable or a declaration added
        afterwards therefore has no effect on that request.
        """
        if isinstance(required, str):
            required = [required]
        required_tup = tuple(sorted(required))
        compiled_graph = self.compiled_graphs.get(required_tup)
        if compiled_graph is not None:
            return compiled_graph
        ordered_variables, used_external_vars = self._topological_order(required)
        ordered_nodes = [
            CompiledNode(
                variable=var,
                inputs=[self.registry[var.inputs[input_name]][input_name] for input_name in var.inputs.keys()]
            ) for var in ordered_variables
        ]
        required_external_variables = self._required_external_variables(used_external_vars)
        compiled = CompiledGraph(
            ordered_nodes=ordered_nodes,
            required_external_variables=required_external_variables,
            debug=debug
        )
        self.compiled_graphs[required_tup] = compiled
        return compiled

    def _get_source(self, source_name: str) -> Namespace:
        """Return the namespace named `source_name`, a :class:`Variables` or an :class:`External` namespace.

        Raises ``KeyError`` if the graph has no namespace of that name.
        """
        variables_source = self.registry.get(source_name)
        if variables_source is not None:
            return variables_source
        raise KeyError(f"Unknown source {source_name}")

    def _topological_order(self, required: list[str]) -> tuple[list[Variable], set[Variable]]:
        """Return the variables that `required` needs, each after its inputs, and the set of the external ones.

        :param required: a qualified name, or a list of them.

        The walk is depth first and recursive, so a chain of dependencies deeper than Python's recursion limit raises
        ``RecursionError``. A cycle raises ``RuntimeError``.
        """
        if isinstance(required, str):
            required = [required]

        visited = set()
        visiting = set()
        ordered_variables = []

        used_external_vars: set[Variable] = set()

        def visit(qual_name: str):
            """Append the variable `qual_name` after its inputs, depth first, unless it is already in the order."""
            if qual_name in visited:
                return
            if qual_name in visiting:
                raise RuntimeError(f"Cycle detected at {qual_name}")
            visiting.add(qual_name)
            source_name, variable_name = qual_name.rsplit(QUAL_SEP, 1)
            source = self._get_source(source_name)
            variable = source[variable_name]
            if isinstance(source, External):
                used_external_vars.add(variable)
            for input_name, input_source in variable.inputs.items():
                visit(make_qual_name(input_source, input_name))
            visiting.remove(qual_name)
            visited.add(qual_name)
            ordered_variables.append(variable)

        for r in required:
            visit(r)
        return ordered_variables, used_external_vars

    @staticmethod
    def _required_external_variables(used_external_vars: set[Variable]) -> dict[str, dict[str, Variable]]:
        """Return the external variables grouped by source, as ``{source: {name: variable}}``."""
        required_external_variables = {}
        for variable in used_external_vars:
            variable_name = variable.name
            variable_source = variable.source
            required_external_variables.setdefault(variable_source, {})[variable_name] = variable
        return required_external_variables

    def _build_reverse_dependencies(self) -> dict[str, set[str]]:
        """Return a map from the qualified name of each input to the qualified names of the variables that read it.

        The graph builds the map from the registry on the first call and caches it, so a variable added or replaced
        afterwards is not in it.
        """
        if self.reverse_dependencies is not None:
            return self.reverse_dependencies
        reverse = {}
        for source_name, source in self.registry.items():
            for variable_name in source:
                variable = source[variable_name]
                qual_name = make_qual_name(source_name, variable.name)
                for input_name, input_source in variable.inputs.items():
                    input_qual = make_qual_name(input_source, input_name)
                    reverse.setdefault(input_qual, set()).add(qual_name)
        self.reverse_dependencies = reverse
        return reverse

    def dependents_of(self, qual_names: list[str] | set[str] | str) -> set[str]:
        """Return the qualified names of the variables that depend on any of `qual_names`, and `qual_names` themselves.

        :param qual_names: a qualified name, or a list or set of them.

        A dependency may be direct or pass through other variables. The search covers every namespace of the graph,
        not only a compiled request. The graph builds the map of dependents on the first call and keeps it, so a
        variable added or replaced afterwards is not seen.
        """
        if isinstance(qual_names, str):
            qual_names = {qual_names}
        else:
            qual_names = set(qual_names)
        reverse = self._build_reverse_dependencies()
        result = set(qual_names)
        stack = list(qual_names)
        while stack:
            current = stack.pop()
            for dep in reverse.get(current, ()):
                if dep not in result:
                    result.add(dep)
                    stack.append(dep)
        return result

    def explain(self, qual_name: str, right_to_left: bool = True) -> str:
        """Return a text that explains how the variable `qual_name` is derived from its dependencies.

        :param qual_name: the qualified name of the variable.
        :param right_to_left: if True, the default, the text starts with `qual_name` and moves toward the end nodes of
            the graph; if False, it starts with the end nodes.

        The text has one entry for `qual_name` and one for each variable that it depends on. The entry of an external
        variable names its source; the entry of any other variable names its inputs and gives its metadata.
        """
        derived = set()
        derivation = []

        def collect(qual_name_: str):
            """Add the entries of the dependencies of `qual_name_`, then its own, each variable once."""
            if qual_name_ in derived:
                return
            derived.add(qual_name_)
            source_name, variable_name = qual_name_.rsplit(QUAL_SEP, 1)
            variable_source = self.registry[source_name]
            variable = variable_source[variable_name]
            inputs_qual_names = []
            for input_name, input_source in variable.inputs.items():
                inputs_qual_names.append(make_qual_name(input_source, input_name))
                collect(make_qual_name(input_source, input_name))
            if isinstance(variable_source, External):
                derivation.append(f"'{variable_name}' comes from external source '{source_name}'\n")
            else:
                derivation.append(
                    f"""'{qual_name_}' depends on {tuple(sorted(inputs_qual_names))} via\n\n{variable.metadata}"""
                )

        collect(qual_name)
        derivation = reversed(derivation) if right_to_left else derivation
        derivation_txt = "\n" + "\n".join(derivation)
        return derivation_txt
