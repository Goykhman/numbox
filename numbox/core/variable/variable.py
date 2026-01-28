import warnings

from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Mapping, Set, Tuple, TypeAlias, Union
)


Namespace: TypeAlias = Union['External', 'Variables']
VarSpec: TypeAlias = Dict[str, Callable | Dict[str, str] | str]
VarValue: TypeAlias = Any


class _Null:
    """ Value of `Variable` that has not been calculated. """
    pass


_null = _Null()


QUAL_SEP = "."


def _make_qual_name(namespace_name: str, var_name: str) -> str:
    """ Each `Variable` instance is best contained in a
     mapping namespace, with the given `namespace_name`.
     This function thereby returns qualified name of the
     `Variable` instance. """
    return f"{namespace_name}{QUAL_SEP}{var_name}"


class External:
    """
    A dictionary that facilitates discovery of required names.
    When requested a `Variable` with the given name via a
    typical `__getitem__` call, if the `Variable` is not
    found, it will be created and added to this dictionary.
    This way the user will be able to infer which variables
    are required from the external source abstracted by this
    dictionary.
    """
    def __init__(self, name: str):
        self.name = name
        self._vars = {}

    def __getitem__(self, name):
        variable = self._vars.get(name)
        if variable is None:
            variable = Variable(
                name=name,
                source=self.name
            )
            self._vars[name] = variable
        return variable


@dataclass(frozen=True)
class Variable:
    """
    An instance of `Variable` is anything that can be calculated
    from the values of the given input dependencies using the
    provided formula (i.e., a function).

    Calculated value can be `None`, that is why a non-calculated
    value is designated with `_null`.

    An instance of `Variable` is best created directly within
    the given `Namespace`, that is, when it is instantiated upon
    initialization of that `Namespace` and made then available
    through a `__getitem__` call to either a collection of
    variables created as an instance of `Variables` (see below)
    or a non-`Variables` mapping, referred to (or abstracted by),
    in general, as an `External` collection. The namespace is
    also referred to as the 'source' of the `Variable`.

    Qualified name of a `Variable` incorporates both the name
    of the `Variable` and the name of its source / namespace.

    It is therefore recommended to create `Variable` only in
    the `External` or `Variables` containers rather than in
    isolation.

    :param name: name of the `Variable` instance.
    :param source: name of the `Variables` or `External` instance
    which is namespaces / source of this `Variable`.
    :param inputs: (optional) map from names of the `Variable`
    inputs (which are names of other `Variable` instances) to
    the names of their sources, i.e., names of either
    `Variables` or 'External' mapping collections referencing
    these variables.
    :param formula: (optional) function that calculates value
    of this `Variable` from its sources.
    :param cacheable: (default `False`) when `True`, the
    corresponding `Value` (see below) will be cached during
    calculation by the `id` of the corresponding Python object
    containing that value. When attempted to recompute with
    the same inputs, cached value will be returned instead.
    """
    name: str
    source: str = field(default="")
    inputs: Mapping[str, str] = field(default_factory=lambda: {})
    formula: Callable = field(default=None)
    cacheable: bool = field(default=False)

    def __hash__(self):
        return hash((self.source, self.name))

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name and self.source == other.source

    def qual_name(self) -> str:
        return _make_qual_name(self.source, self.name)


class Variables:
    def __init__(
        self,
        name: str,
        variables: List[VarSpec],
    ):
        """
        Namespace of `Variable` instances.

        :param name: name of the `Variables` instance namespace.
        :param variables: initializer list of `VarSpec` specs that can be used
        to create instances of `Variable` to be stored in this namespace.
        """
        self.name = name
        self.variables = {variable["name"]: Variable(source=self.name, **variable) for variable in variables}

    def __getitem__(self, variable_name: str) -> Variable:
        """
        :param variable_name: name of the `Variable` to retrieve,
        it is expected that `Variables` and `External` all expose
        this method that returns an instance of the `Variable` with
        the given name in either a `Variables` namespace or an
        `External` namespace, respectively.
        """
        return self.variables[variable_name]


@dataclass
class Value:
    """
    Value of the corresponding `Variable`.
    """
    variable: Variable
    value: VarValue | _Null = field(default_factory=lambda: _null)


class Values:
    """ Values of all `Variable`s, computed and external,
    will be held here. """
    def __init__(self):
        self.values: Dict[Variable, Value] = {}
        self.cache: Dict[tuple, VarValue] = {}

    def get(self, variable: Variable) -> Value:
        if variable not in self.values:
            self.values[variable] = Value(variable=variable)
        return self.values[variable]


@dataclass(frozen=True)
class CompiledNode:
    variable: Variable
    inputs: List[Variable]

    def __post_init__(self):
        if self.variable.formula is None and self.inputs:
            raise RuntimeError(f"{self.variable} contains inputs but no formula, how come?")

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
    ordered_nodes: List[CompiledNode]
    required_external_variables: Dict[str, Dict[str, Variable]]
    dependents: Dict[Variable, List[CompiledNode]] = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self):
        for node in self.ordered_nodes:
            for inp in node.inputs:
                self.dependents[inp].append(node)

    def execute(
        self,
        external_values: Dict[str, Dict[str, VarValue]],
        values: Values,
    ):
        """
        Main entry point to calculation of compiled graph.
        Calculation requires the following inputs:

        :param external_values: actual values of all required external
        variables, this can be a superset of what is really needed for
        the calculation. The map is first from the name of the external
        source and then from the name of the variable within that
        source to the variable's actual value.
        :param values: runtime storage of all values, instance of `Values`.
        :return:
        """
        self._assign_external_values(external_values, values)
        self._calculate(self.ordered_nodes, values)

    def _assign_external_values(
        self,
        external_values: Dict[str, Dict[str, VarValue]],
        values: Values
    ):
        """
        For the external variables required for this calculation,
        populate their values into the `Values` container.

        :param external_values: mapping from names of external sources
        to dictionary from names of external `Variable`s to their instances,
        that are needed for the given calculation.
        :param external_values: dictionary of `External` name to a mapping
        of `Variable` names to their values.
        :param values: an instance of `Values` storage of all calculated
        values.
        """
        for source_name, variables in self.required_external_variables.items():
            provided = external_values.get(source_name)
            if provided is None:
                raise KeyError(f"Missing external source '{source_name}'")
            for var_name, variable in variables.items():
                if var_name not in provided:
                    raise KeyError(
                        f"Missing value for external variable '{_make_qual_name(source_name, var_name)}'"
                    )
                values.get(variable).value = provided[var_name]

    def _collect_affected(self, changed_vars: Set[Variable]) -> List[CompiledNode]:
        """
        Return subset of `self.ordered_nodes` consisting of nodes
        affected by `changed_vars`, in the same order as in the original.
        """
        affected = set()
        stack = list(changed_vars)
        while stack:
            v = stack.pop()
            for node in self.dependents.get(v, []):
                if node not in affected:
                    affected.add(node)
                    stack.append(node.variable)
        return [node for node in self.ordered_nodes if node in affected]

    @staticmethod
    def _calculate(nodes: List[CompiledNode], values: Values):
        """
        Calculate the values of the `Variable`s using their own callables
        by evaluating them as functions of the values of the specified
        inputs.

        All inputs need to be calculated first (i.e., to be non-`_null`)
        before the value of the given `Variable` can be `calculate`d.

        This is possible because the `Variable`s are supplied as a
        topologically ordered list `ordered_variables`.
        """
        for node in nodes:
            if node.variable.formula is None:
                continue
            args = tuple(values.get(input_).value for input_ in node.inputs)
            assert not any(
                [arg is _null for arg in args]
            ), f"Uninitialized input for {node.variable}, args = {args}"
            cache_key = (node.variable, args)
            if node.variable.cacheable:
                if cache_key in values.cache:
                    values.get(node.variable).value = values.cache[cache_key]
                    continue
            result = node.variable.formula(*args)
            if node.variable.cacheable:
                values.cache[cache_key] = result
            values.get(node.variable).value = result

    def recompute(self, changed: Dict[str, Dict[str, VarValue]], values: Values):
        """
        :param changed: dict of sources to names to new values of changed `Variable`s.
        :param values: storage of all the `Variable` values.
        """
        changed_vars = set()
        for src, vals in changed.items():
            for name, val in vals.items():
                variable = self.required_external_variables.get(src, {}).get(name)
                qual = _make_qual_name(src, name)
                if variable is None:
                    try:
                        variable = next(n.variable for n in self.ordered_nodes if n.variable.qual_name() == qual)
                    except StopIteration:
                        warnings.warn(f"{qual} is not in the calculation path, update has no effect.")
                        continue
                values.get(variable).value = val
                changed_vars.add(variable)
        affected_nodes = self._collect_affected(changed_vars)
        for node in affected_nodes:
            values.get(node.variable).value = _null
        self._calculate(affected_nodes, values)


class Graph:
    def __init__(
        self,
        variables_lists: Dict[str, List[VarSpec]],
        external_source_names: List[str]
    ):
        """
        :param variables_lists: mapping of names of `Variables`
        namespace to the list of `Variable` instances to be added
        to that namespace.
        :param external_source_names: list of names of possible
        `External` sources from which `Variable` inputs might
        be coming from.
        """
        self.external_source_names = external_source_names
        self.registry = {}
        self.external = {
            external_source_name: External(external_source_name) for external_source_name in external_source_names
        }
        for variables_name, variables_list in variables_lists.items():
            assert variables_name not in self.registry, (
                f"Variables instance {variables_name} has already been created in this registry"
            )
            variables = Variables(
                name=variables_name,
                variables=variables_list
            )
            self.registry[variables_name] = variables
        for external_name, external_ in self.external.items():
            registered_external = self.registry.get(external_name)
            if registered_external is not None:
                assert registered_external == external_, (
                    f"{external_name} external already registered as {registered_external}"
                )
            else:
                self.registry[external_name] = external_
        self.compiled_graphs = {}

    def compile(self, required: List[str] | str) -> CompiledGraph:
        """
        :required: list of qualified variables names that need to be calculated.
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
        )
        self.compiled_graphs[required_tup] = compiled
        return compiled

    def _get_source(self, source_name: str) -> Namespace:
        """
        :param source_name: name of the source (either an instance
        of `Variables` or an `External` source) that is requested.
        """
        variables_source = self.registry.get(source_name)
        if variables_source is not None:
            return variables_source
        raise KeyError(f"Unknown source {source_name}")

    def _topological_order(self, required: List | Tuple | str):
        """
        :param required: qualified name(s) of `Variable` instance(s)
        for which a topological ordering of a DAG is to be determined.
        """
        if isinstance(required, str):
            required = [required]

        visited = set()
        visiting = set()
        ordered_variables = []

        used_external_vars: Set[Variable] = set()

        def visit(qual_name: str):
            if qual_name in visited:
                return
            if qual_name in visiting:
                raise RuntimeError(f"Cycle detected at {qual_name}")
            visiting.add(qual_name)
            source_name, variable_name = qual_name.split(QUAL_SEP)
            source = self._get_source(source_name)
            variable = source[variable_name]
            if isinstance(source, External):
                used_external_vars.add(variable)
            for input_name, input_source in variable.inputs.items():
                visit(_make_qual_name(input_source, input_name))
            visiting.remove(qual_name)
            visited.add(qual_name)
            ordered_variables.append(variable)

        for r in required:
            visit(r)
        return ordered_variables, used_external_vars

    @staticmethod
    def _required_external_variables(used_external_vars: Set[Variable]) -> Dict[str, Dict[str, Variable]]:
        """
        For requested `External` sources, return the list of required
        external `Variable` instances.
        """
        required_external_variables = defaultdict(dict)
        for variable in used_external_vars:
            variable_name = variable.name
            variable_source = variable.source
            required_external_variables[variable_source][variable_name] = variable
        return required_external_variables
