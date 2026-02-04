import warnings

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Iterator, List, Mapping,
    Protocol, Set, Tuple, TypeAlias, TypedDict
)


class Namespace(ABC):
    name: str
    _variables: Dict[str, 'Variable']

    def keys(self):
        return self._variables.keys()

    def update(self, key: str, var: 'Variable') -> None:
        """
        Post-initialization update for dynamically generated
        `Variable`s.
        """
        self._variables[key] = var

    def __contains__(self, key: str) -> bool:
        """
        :param key: un-qualified name of the `Variable`
        It is qualified with `self.name` - the name of this
        `Namespace` namespace that the `Variable` belongs to.
        """
        return key in self._variables

    @abstractmethod
    def __getitem__(self, key: str) -> 'Variable':
        pass

    def __iter__(self) -> Iterator[str]:
        return iter(self._variables.keys())


class Storage(Protocol):
    _values: Dict['Variable', 'Value']
    cache: Dict[Tuple['Variable', Tuple[Any, ...]], 'VarValue']

    def get(self, variable: 'Variable') -> 'Value':
        """
        Principal access point to the requested variable.
        Instantiates the corresponding value when first
        invoked for the given variable.
        """

    def __iter__(self) -> Iterator['Variable']:
        pass


class VarSpec(TypedDict, total=False):
    name: str
    inputs: Dict[str, str]
    formula: Callable
    metadata: str
    cacheable: bool


VarValue: TypeAlias = Any


class _Null:
    """ Value of `Variable` that has not been calculated. """
    pass


_null = _Null()


QUAL_SEP = "."


def make_qual_name(namespace_name: str, var_name: str) -> str:
    """ Each `Variable` instance is best initialized in
    and owned by a `Namespace` object (such as, instances
    of `External` and `Variables`), with the given
    `namespace_name`.

    This function thereby returns qualified name of the
    `Variable` instance. """
    return f"{namespace_name}{QUAL_SEP}{var_name}"


class External(Namespace):
    """
    An 'external' namespace that facilitates discovery
    of requested names.

    When requesting a `Variable` with the given name via a
    typical `__getitem__` call, if the `Variable` is not
    found, it will be created and added to this dictionary.
    This way the graph will be able to infer which variables
    are required from the external source abstracted by this
    namespace.
    """
    def __init__(self, name: str):
        self.name = name
        self._variables: Dict[str, 'Variable'] = {}

    def __getitem__(self, name) -> 'Variable':
        """
        :param name: un-qualified name of the `Variable`.
        Instances of `Variable` that are 'external' should
        be put in `External` namespace indirectly, through
        a call to this method. These `Variable`s will be
        qualified with the name `self.name` of this namespace.
        """
        variable = self._variables.get(name)
        if variable is None:
            variable = Variable(
                name=name,
                source=self.name
            )
            self._variables[name] = variable
        return variable


@dataclass(frozen=True)
class Variable:
    """
    An instance of `Variable` is anything that can be calculated
    from the values of the given `inputs` dependencies using the
    provided `formula` (i.e., a Python function).

    Calculated value can be `None`, that is why a non-calculated
    value is designated with `_null`.

    An instance of `Variable` is best created within the given
    `Namespace`. For example, when the `Variables` subtype of
    the `Namespace` is instantiated, it gets populated with
    the freshly created `Variable` instances per the `VarSpec`
    specifications passed to it. Or, when the `External` subtype
    of the `Namespace` is queried for the given variable name,
    if a `Variable` with such a name is not already present in
    that external namespace, it will be created and stored there.

    :param name: name of the `Variable` instance.
    :param source: name of the `Namespace` instance  which is
    the namespace / source of this `Variable`.
    :param inputs: (optional) map from names of the `Variable`
    inputs (which are names of other `Variable` instances) to
    names of their `Namespace`s.
    :param formula: (optional) function that calculates the
    value of this `Variable` from its sources.
    :param metadata: any possible metadata associated with
    this variable.
    :param cacheable: (default `False`) when `True`, the
    corresponding `Value` (see below) will be cached during
    calculation. When attempting to recompute with the same
    inputs, cached value will be returned instead. Use sparingly!
    """
    name: str
    source: str = field(default="")
    inputs: Mapping[str, str] = field(default_factory=lambda: {})
    formula: Callable = field(default=None)
    metadata: str | None = field(default=None)
    cacheable: bool = field(default=False)

    def __hash__(self):
        return hash((self.source, self.name))

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name and self.source == other.source

    def qual_name(self) -> str:
        """
        Qualified name of `Variable` incorporates both the name
        of the `Variable` and the name of its source / namespace.
        """
        return make_qual_name(self.source, self.name)


class Variables(Namespace):
    def __init__(
        self,
        name: str,
        variables: List[VarSpec],
    ):
        """
        Namespace of `Variable` instances.

        :param name: name of the `Variables` instance namespace.
        :param variables: initializer list of `VarSpec` specs
        used to create instances of `Variable` to be stored in
        this namespace.
        """
        self.name = name
        self._variables = {variable["name"]: Variable(source=self.name, **variable) for variable in variables}

    def __getitem__(self, variable_name: str) -> Variable:
        """
        :param variable_name: name of the `Variable` to retrieve.
        This is un-qualified name, as it is already looked up in
        this namespace.
        """
        return self._variables[variable_name]


@dataclass
class Value:
    """
    Value of the corresponding `Variable`.
    Best used when created indirectly by the
    `Values` storage.
    """
    variable: Variable
    value: VarValue | _Null = field(default_factory=lambda: _null)


class Values:
    """ Values of all `Variable`s, computed and external,
    will be held here. """
    def __init__(self):
        self._values: Dict[Variable, Value] = {}
        self.cache: Dict[Tuple['Variable', Tuple[Any, ...]], VarValue] = {}

    def get(self, variable: Variable) -> Value:
        if variable not in self._values:
            self._values[variable] = Value(variable=variable)
        return self._values[variable]

    def __iter__(self) -> Iterator[Variable]:
        return iter(self._values.keys())


@dataclass(frozen=True)
class CompiledNode:
    variable: Variable
    inputs: List[Variable]

    def __post_init__(self):
        if self.variable.formula and not self.variable.inputs:
            raise RuntimeError(f"{self.variable} contains formula but no inputs, how come?")

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
    dependents: Dict[Variable, List[CompiledNode]] = field(default_factory=lambda: {})

    def __post_init__(self):
        for node in self.ordered_nodes:
            for inp in node.inputs:
                self.dependents.setdefault(inp, []).append(node)

    def execute(
        self,
        external_values: Dict[str, Dict[str, VarValue]],
        values: Storage,
    ):
        """
        Main entry point to calculate values of nodes of the compiled
        graph. Calculation requires the following inputs:

        :param external_values: actual values of all required external
        variables, this can be a superset of what is really needed for
        the calculation. The map is first from the name of the external
        namespace and then from the name of the variable within that
        source to the variable's actual value.
        :param values: runtime storage of all values, e.g., an instance
        of `Values`.
        """
        self._assign_external_values(external_values, values)
        self._calculate(self.ordered_nodes, values)

    def _assign_external_values(
        self,
        external_values: Dict[str, Dict[str, VarValue]],
        values: Storage
    ):
        """
        For the external variables required for this calculation,
        populate their values into the `Values` storage.

        :param external_values: mapping from names of external sources
        to dictionary from names of external `Variable`s to their values
        that are needed for the given calculation.
        :param values: an instance of `Values` storage of all calculated
        values.
        """
        for source_name, variables in self.required_external_variables.items():
            provided = external_values.get(source_name)
            if provided is None:
                raise KeyError(f"Missing external source '{source_name}'")
            for var_name, variable in variables.items():
                var_value = provided.get(var_name)
                if var_value is None:
                    raise KeyError(
                        f"Missing value for external variable '{make_qual_name(source_name, var_name)}'"
                    )
                values.get(variable).value = var_value

    def _collect_affected(self, changed_vars: Set[Variable]) -> List[CompiledNode]:
        """
        Return subset of `self.ordered_nodes` consisting of nodes
        affected by `changed_vars`, in the same order as in the
        `self.ordered_nodes`.
        """
        affected = set()
        stack = list(changed_vars)
        while stack:
            var = stack.pop()
            for node in self.dependents.get(var, []):
                if node not in affected:
                    affected.add(node)
                    stack.append(node.variable)
        return [node for node in self.ordered_nodes if node in affected]

    @staticmethod
    def _calculate(nodes: List[CompiledNode], values: Storage):
        """
        Calculate the values of the `Variable`s using their own `formula`
        by evaluating them as functions of the values of the specified
        inputs.

        All inputs need to be calculated first (i.e., to be non-`_null`)
        before the value of the given `Variable` can be `_calculate`d.
        This is possible because the `Variable`s in the list `nodes` are
        supplied as a topologically ordered list `self.ordered_variables`,
        or as an ordered sub-set thereof (see, e.g., `recompute`).
        """
        for node in nodes:
            if node.variable.formula is None:
                continue
            args = tuple(values.get(input_).value for input_ in node.inputs)
            assert not any(
                [arg is _null for arg in args]
            ), f"Uninitialized input for {node.variable}, args = {args}"
            cache_key = (node.variable, args)
            if node.variable.cacheable and cache_key in values.cache:
                values.get(node.variable).value = values.cache[cache_key]
                continue
            result = node.variable.formula(*args)
            if node.variable.cacheable:
                values.cache[cache_key] = result
            values.get(node.variable).value = result

    def recompute(self, changed: Dict[str, Dict[str, VarValue]], values: Storage):
        """
        :param changed: dict of sources to names to new values of changed
        `Variable`s coming from either `External` or `Variables` source.
        :param values: storage of all the `Variable` values.
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
    def __init__(
        self,
        variables_lists: Dict[str, List[VarSpec]],
        external_source_names: List[str]
    ):
        """
        :param variables_lists: mapping of names of `Variables`
        namespaces to the lists of `Variable` instances to be
        added to those namespaces.
        :param external_source_names: list of names of possible
        `External` sources from which `Variable` inputs might
        be coming from.
        """
        self.external_source_names: List[str] = external_source_names
        self.registry: Dict[str, Namespace] = {}
        self.external: Dict[str, External] = {
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
        self.reverse_dependencies = None

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

    def _topological_order(self, required: List[str] | Tuple[str] | str):
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
            """ DFS traversal of graph nodes. """
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
    def _required_external_variables(used_external_vars: Set[Variable]) -> Dict[str, Dict[str, Variable]]:
        """
        For requested `External` sources, return the list of required
        external `Variable` instances.
        """
        required_external_variables = {}
        for variable in used_external_vars:
            variable_name = variable.name
            variable_source = variable.source
            required_external_variables.setdefault(variable_source, {})[variable_name] = variable
        return required_external_variables

    def _build_reverse_dependencies(self) -> Dict[str, set[str]]:
        """
        Utility to calculate set of qualified names of variables
        impacted by each of the encountered inputs.
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

    def dependents_of(self, qual_names: List[str] | Set[str] | str) -> Set[str]:
        """
        Return qualified names of `Variable`s that directly or indirectly
        depend on any of `qual_names`.
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
        """
        Follow the dependencies chain to explain how the given
        variable is derived.

        Uses `metadata` of the `Variable` instances.

        :param qual_name: qualified name of the `Variable`.
        :param right_to_left: when `True` (default), begin explanation
        with `qual_name`. That is, move towards the ends of the
        graph.
        """
        derived = set()
        derivation = []

        def collect(qual_name_: str):
            if qual_name_ in derived:
                return
            derived.add(qual_name_)
            source_name, variable_name = qual_name_.split(QUAL_SEP)
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
                    f"""'{qual_name_}' depends on {tuple(sorted(inputs_qual_names))} via \n\n{variable.metadata}"""
                )

        collect(qual_name)
        derivation = reversed(derivation) if right_to_left else derivation
        derivation_txt = "\n" + "\n".join(derivation)
        return derivation_txt
