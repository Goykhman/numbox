numbox.core.variable
====================

Overview
++++++++

Framework for Directed Acyclic Graph (DAG) in pure Python.
Computationally heavy parts can be put on this graph as JIT-compiled functions.

Modules
++++++++

numbox.core.variable.variable
-----------------------------

Overview
********

A graph can be defined as follows::

    from numbox.core.variable.variable import Graph

    x = {"name": "x", "inputs": {"y": "basket"}, "formula": lambda y_: 2 * y_, "cacheable": True}
    a = {"name": "a", "inputs": {"x": "variables1"}, "formula": lambda x_: x_ - 74, "cacheable": True}
    u = {"name": "u", "inputs": {"a": "variables1"}, "formula": lambda a_: a_ * 2, "cacheable": True}

    graph = Graph(
        variables_lists={
            "variables1": [x, a],
            "variables2": [u],
        },
        external_source_names=["basket"]
    )

Here we have the variable `y` sourced externally from the `basket`, and calculated variables
`x` and `a` in the `variables1` namespace, and `u` in the `variables2` namespace. The dictionaries
`x`, `a`, and `u` are called variable specifications. These specs are on their own agnostic about what
namespace they can be put in. These namespaces however need to be defined in the `variables_lists`
argument given to the `Graph` at initialization time.

Names of the 'external' sources (of data values) need to be supplied to the `Graph` as well.
When the :class:`numbox.core.variable.variable.Graph` is compiled
to the :class:`numbox.core.variable.variable.CompiledGraph`, it will automatically figure out which variables need to be sourced
from each of the specified external sources (such as, '`basket`') in order to perform the
required calculation::

    from numbox.core.variable.variable import CompiledGraph

    required = ["variables2.u"]
    compiled = graph.compile(required)
    assert isinstance(compiled, CompiledGraph)

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

Instances of `Variable` s are stored in the `Graph`'s instance's `registry`::

    from numbox.core.variable.variable import Variables, Variable

    registry = graph.registry
    variables1 = registry["variables1"]
    variables2 = registry["variables2"]

    assert list(variables1.variables.keys()) == ["x", "a"]
    assert list(variables2.variables.keys()) == ["u"]

    assert isinstance(variables1, Variables)
    assert isinstance(variables1.variables["x"], Variable)

That is, users are not expected to create neither instances of `Variable` nor instances of `Variables`,
although they are certainly allowed to do so if needed.
Instead, users provide variable specifications, as the dictionaries `x`, `u`, `a`
in the example above (and the variable name "`y`" that is referred to and implied to be 'external')
that are given to the `Graph`. The `Graph` then creates instances of `Variables` (one per namespace)
and instances of `External` (one per 'external' source). Finally, `Variables` and `External` create instances
of `Variable` s and store them.

As shown above, `External` auto-discovers which external variables from the corresponding source need to be present,
and reports that information to the compiled graph.

To calculate the required variables, one first needs to instantiate the execution-scope instance
of :class:`numbox.core.variable.variable.Values`. This will contain all calculated nodes
as a mapping from the corresponding `Variable` to instances of :class:`numbox.core.variable.variable.Value`.
The latter wraps the data. All the data of non-external variables is initialized to
the instance `_null` of the :class:`numbox.core.variable.variable._Null`.

Then, one needs to supply `external_values` of the leaf nodes that are needed for the calculation.
As discussed above, these external variables are determined programmatically. Provided these
have been specified, one can calculate the graph as::

    from numbox.core.variable.variable import Values

    values = Values()
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

.. automodule:: numbox.core.variable.variable
   :members:
   :show-inheritance:
   :undoc-members:
