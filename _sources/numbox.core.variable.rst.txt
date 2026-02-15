numbox.core.variable
====================

Overview
++++++++

Framework for Directed Acyclic Graph (DAG) in pure Python.
While this module does not contain any JIT-compiled
bits in particular, or anything imported from numba in general,
computationally heavy parts can be put on this graph as JIT-compiled functions
via the `formula` key of the graph variables specifications (see below).

Modules
++++++++

numbox.core.variable.variable
-----------------------------

Overview
********

A graph can be defined as follows::

    from numbox.core.variable.variable import Graph

    def derive_x(y_):
        return 2 * y_

    def derive_a(x_):
        return x_ - 74

    def derive_u(a_):
        return 2 * a_

    x = {"name": "x", "inputs": {"y": "basket"}, "formula": derive_x}
    a = {"name": "a", "inputs": {"x": "variables1"}, "formula": derive_a}
    u = {"name": "u", "inputs": {"a": "variables1"}, "formula": derive_u}

    graph = Graph(
        variables_lists={
            "variables1": [x, a],
            "variables2": [u],
        },
        external_source_names=["basket"]
    )

Here we have the variable `y` sourced externally from the `basket`, and calculated variables
`x` and `a` in the `variables1` namespace, and `u` in the `variables2` namespace.

The dictionaries
`x`, `a`, and `u` are called variable specifications. These specs on their own are agnostic about what
namespace they can be put in. The namespaces however need to be specified via the `variables_lists`
argument given to the `Graph` at the initialization time.

The full and unambiguous way to denote the variables is via their qualified
names, applicable both to externally sourced variables, `basket.y`, as well as
the calculated ones, `variables1.x`,
`variables1.a`, `variables2.u`.

One of the variables specifications, designated with the key `formula`, specifies the
function with the parameters that match the input variables (this graph node's dependencies)
that are in turn
designated with the key `inputs`. While the names of the parameters of the function assigned
to the `formula` key do not have to match the names of the `inputs`, their order is
expected to follow one-to-one correspondence. This way the graph is instructed
which inputs to use to get the values to be assigned to the parameters of the `formula`.

The Python function specified by the `formula`
can be a wrapper around numba JIT-compiled function, i.e.,
a proxy to the numba's `FunctionType` or `CPUDispatcher` objects [#f1]_.

The variable specification for `inputs` (if any) includes both the names of the dependencies variables
required to calculate the given variable via the function given by the `formula`,
as well as the namespaces where these variables are going to be looked for in.

Graph end nodes, located at the edge of the graph (a.k.a., leaf nodes) have neither `inputs`
nor `formula` in their specifications. Specifying `formula` without `inputs`
will result in an exception. It is possible, however, to specify `inputs`
but no formula, which technically defines the placement of the node
on the graph but leaves it up to the developer to defer specifying the node's calculation
logic until later in the runtime.

The variable can be specified as `cacheable` if its value calculated for the given tuple of
arguments can be cached and later retrieved without re-calculation provided
the arguments have not changed. The arguments types of the corresponding `formula` then need to be hashable -
custom type sub-classing with its own `__hash__` might be needed in certain cases, thereby providing the definition
of the identity of the arguments' values.
When `cacheable=True` (by default it is `False`), the graph will avoid recalculation of the
value provided the inputs haven't changed. It is not recommended to abuse the cache, especially
for the continuous or large-cardinality spaces of identities of the parameters of the node's `formula`.

It is worth noting here that the `cacheable` key is a rather brute force way
to avoid identical re-computations.
It is completely unrelated to the graph's dependency structure.
On the other hand, the graph's `recompute`
method, discussed below, only recomputes the values of variables that are dependent on the nodes
that have been updated. That is, the strategy of the `recompute` method
is determined by the graph's topology only
and is independent of the `cacheable` specifications of the nodes'
variables.

Names of the 'external' sources (of data values) need to be given to the `Graph` as well,
via the `external_source_names` argument.
When the :class:`numbox.core.variable.variable.Graph` is compiled
to the :class:`numbox.core.variable.variable.CompiledGraph`, it will automatically figure out which variables need to be sourced
from each of the specified external sources (such as, '`basket`') in order to perform the
required calculation::

    from numbox.core.variable.variable import CompiledGraph

    # What is required from this calculation, the names of qualified variables
    required = ["variables2.u"]

    # Compile the graph for the required variables
    compiled = graph.compile(required)
    assert isinstance(compiled, CompiledGraph)

    # The graph will figure out what external variables it needs to do the calculation
    required_external_variables = compiled.required_external_variables
    assert list(required_external_variables.keys()) == ["basket"]
    basket = required_external_variables["basket"]
    assert list(basket.keys()) == ["y"]
    assert basket["y"].name == "y"

`Graph` uses the variable specifications given to it to create instances of :class:`numbox.core.variable.variable.Variable`.
Namespaces of calculated `Variable` s are :class:`numbox.core.variable.variable.Variables`.
Namespaces of externally sourced `Variable` s are
:class:`numbox.core.variable.variable.External` .

Semantically, each `Variable` is defined by its scoped name, that is, a tuple of its namespace / source
name and its own name.

In DAG terminology, `External` scopes contain variables with no inputs, that is, edge (or end / leaf) nodes.

Instances of `Variable` s and `External` are stored in the `Graph`'s instance's `registry`::

    from numbox.core.variable.variable import Variables, Variable

    registry = graph.registry

    # Get the namespaces...
    variables1 = registry["variables1"]
    variables2 = registry["variables2"]

    # ... and the variables defined in these namespaces
    assert list(variables1.variables.keys()) == ["x", "a"]
    assert list(variables2.variables.keys()) == ["u"]

    assert isinstance(variables1, Variables)
    assert isinstance(variables1.variables["x"], Variable)

    basket_ = registry["basket"]
    ... # same `basket` as above
    assert basket_["y"] is basket["y"]

That is, users are not expected to instantiate neither `Variable` s nor `Variables` s,
although they are certainly allowed to do so if needed (it is recommended to design
one's code so that `Variable` instances when needed are simply retrieved from the `registry` of the
`Graph` instance).
Instead, users provide variable specifications, as the dictionaries `x`, `u`, `a`
in the example above (and the variable name "`y`" that is referred to and implied to be 'external')
that are given to the `Graph`. The `Graph` then creates instances of `Variables` (one per namespace)
and instances of `External` (one per an 'external' source). Finally, `Variables` and `External` in turn
create instances of `Variable` s and store them.

To calculate the required variables, one first needs to instantiate the execution-scope instance
of the storage :class:`numbox.core.variable.variable.Values` of the values of all variables
scoped in `Variables` and `External` namespaces. This storage will get automatically populated
with all calculated nodes
as a mapping from the corresponding `Variable` to instances of :class:`numbox.core.variable.variable.Value`.
The latter wraps the data. All the data of non-external variables is initialized to
the instance `_null` of the :class:`numbox.core.variable.variable._Null`.

Then, one needs to supply `external_values` of the leaf nodes that are needed for the calculation.
As discussed above, these required external variables are identified programmatically. Provided values for these
have been provided, one can calculate the graph as::

    from numbox.core.variable.variable import Values

    # Instantiate the storage
    values = Values()

    # Request the calculation by executing the graph
    compiled.execute(
        external_values={"basket": {"y": 137}},
        values=values,
    )

This populates the `values` with the correct data::

    x_var = variables1["x"]
    a_var = variables1["a"]
    u_var = variables2["u"]

    assert values.get(x_var).value == 274
    assert values.get(a_var).value == 200
    assert values.get(u_var).value == 400

The graph can be recomputed if some of its nodes have been changed.
Only the affected nodes will be re-evaluated::

    compiled.recompute({"basket": {"y": 1}}, values)
    assert values.get(basket["y"]).value == 1
    assert values.get(x_var).value == 2
    assert values.get(a_var).value == -72
    assert values.get(u_var).value == -144


.. rubric:: References

.. [#f1] It is straightforward to adapt the variables specifications given here in pure Python to build a fully-JIT'ed graph of :class:`numbox.core.work.work.Work` nodes, by using the :class:`numbox.core.work.builder.Derived`. See :ref:`builder`.

.. automodule:: numbox.core.variable.variable
   :members:
   :show-inheritance:
   :undoc-members:
