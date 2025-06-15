numbox.core.work
================

Overview
++++++++

Functionality for fully-jitted and light-weight calculation on a graph.

Modules
++++++++

numbox.core.work.node
---------------------

Overview
********

:class:`numbox.core.work.node.Node` represents a node on a directed acyclic graph
(`DAG <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_)
that exists in a fully jitted scope and is accessible both at the low-level and via a Python proxy.

`Node` can be used on its own (in which case the recommended way to
create it is via the factory function :func:`numbox.core.work.node.make_node`)
or as a base to more functionally-rich graph nodes,
such as :class:`numbox.core.work.work.Work`.

The logic of `Node` and its sub-classes follows a graph-optional design - no
graph orchestration structure is required to register and manage the graph
of `Node` instance objects - which in turn reduces unnecessary computation overhead
and simplifies the program design.

To that end, each node is identified by its name and contains a uniformly-typed
container member with all the input nodes references that it bears a directed dependency relationship to.
This enables a traversal not only of graphs of `Node` instances themselves
but also graphs of objects of its subclasses, such as, the graphs of `Work` nodes.

`Node` implementation makes heavy use of numba
`meminfo <https://numba.readthedocs.io/en/stable/developer/numba-runtime.html?highlight=meminfo#memory-management>`_
paradigm that manages memory-allocated
payload via smart pointer (pointer to numba's meminfo object) reference counting.
This allows users to reference the desired
memory location via a 'void' structref type, such as,
:class:`numbox.core.any.erased_type.ErasedType`, or :class:`numbox.core.utils.void_type.VoidType`,
and dereference its payload accordingly when needed via the appropriate :func:`numbox.utils.lowlevel.cast`.

.. automodule:: numbox.core.work.node
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.work.print\_tree
----------------------------

Overview
********

Provides utilities to print a tree from the given node's dependencies.
The node can be either instance of :class:`numbox.core.work.node.Node`
or :class:`numbox.core.work.work.Work`::

    from numbox.core.work.node import make_node
    from numbox.core.work.print_tree import make_image

    n1 = make_node("first")
    n2 = make_node("second")
    n3 = make_node("third", inputs=(n1, n2))
    n4 = make_node("fourth")
    n5 = make_node("fifth", inputs=(n3, n4))
    tree_image = make_image(n5)
    print(tree_image)

which outputs::

    fifth--third---first
           |       |
           |       second
           |
           fourth

Notice that the tree depth extends in horizontal direction,
the width extends in vertical direction and is aligned to
recursively fit images of the sub-trees.

For the sake of readability, if multiple nodes depend on the given node, the
latter will be accordingly displayed multiple times on the tree image, for instance::

    n1 = make_node("n1")
    n2 = make_node("n2", (n1,))
    n3 = make_node("n3", inputs=(n1,))
    n4 = make_node("n4", inputs=(n2, n3))
    tree_image = make_image(n4)

produces::

    n4--n2--n1
        |
        n3--n1

Here it is understood that both references to 'n1' point to the same node,
that happens to be a source of two other nodes, 'n2' and 'n3'.

.. automodule:: numbox.core.work.print_tree
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.work.work
---------------------

Overview
********

Defines :class:`numbox.core.work.work.Work` StructRef.
`Work` is a unit of calculation work that is designed to
be included as a node on a jitted graph of other `Work` nodes.

`Work` type subclasses :class:`numbox.core.work.node.Node` and follows the logic of its graph design.
However, since numba StructRef does not support low-level subclasses,
there is no inheritance relation between `NodeType` and `WorkType`,
leaving the data design to follow the composition pattern.
Namely, the members of the `Node` payload are a header in the payload of `Work`, allowing
to perform a meaningful :func:`numbox.utils.lowlevel.cast`.

The best way to create `Work` object instance is via the :func:`numbox.core.work.work.make_work` constructor
that can be invoked either from Python or jitted scope (plain-Python or jitted `run` function below)::

    import numpy
    from numba import float64, njit
    from numbox.core.work.work import make_work
    from numbox.utils.highlevel import cres

    @cres(float64(), cache=True)
    def derive_work():
        return 3.14

    @njit(cache=True)
    def run(derive_):
        work = make_work("work", 0.0, derive=derive_)
        work.calculate()
        return work.data

    assert numpy.isclose(run(derive_work), 3.14)

When called from jitted scope, if cacheability of the caller function
is a requirement, the `derive` function should be passed to `run` as
a `FunctionType` (not `njit`-produced `CPUDispatcher`) argument, i.e.,
decorated with :func:`numbox.utils.highlevel.cres`). Otherwise,
simply pulling `derive_work` from the global scope within
argument-less `run` will prevent its caching.

Graph manager
*************

While not a requirement, it is recommended that the `Work` instance's `name` attribute matches the name
of the variable to which that instance is assigned.
Moreoever, no out-of-the-box assertions for uniqueness of the `Work` names is provided.
The users are free to implement their own graph managers that register the `Work`
nodes and assert additional requirements on the names as needed. The core numbox library
maintains agnostic position to whether such an overhead is universally beneficial (and
is worth the performance tradeoff).

One option to build a graph manager would be via the constructor such as::

    from numba.core.errors import NumbaError
    from numbox.core.configurations import default_jit_options
    from numbox.core.work.node import NodeType
    from numbox.core.work.work import _make_work
    from numbox.utils.lowlevel import _cast
    from work_registry import _get_global, registry_type

    @njit(**default_jit_options)
    def make_registered_work(name, data, sources=(), derive=None):
        """ Optional graph manager. Consider using `make_work`
        where performance is more critical and name clashes are
        unlikely and/or inconsequential. """
        registry_ = _get_global(registry_type, "_work_registry")
        if name in registry_:
            raise NumbaError(f"{name} is already registered")
        work_ = _make_work(name, data, sources, derive)
        registry_[name] = _cast(work_, NodeType)
        return work_

Here :func:`numbox.core.work.work.ol_make_work` is the original `Work` constructor overload,
while the utility registry module can be defined as

.. literalinclude:: ./_static/work_registry.py
   :language: python
   :linenos:

Implementation details
**********************

Behind the scenes, `Work` accommodates individual access to its `sources`
(other `Work` nodes that are pointing to the given `Work` node on the DAG)
via a 'Python-native compiler' backdoor, which is essentially a relative pre-runtime
technique to leverage Python's `compile` and `exec` functions before preparing for
overload in the numba jitted scope. This technique is fully compatible with caching of
jitted functions and facilitates a natural Python counterpart to virtual functions (unsupported in numba).
Here it is extensively utilized in
:func:`numbox.core.work.work.ol_calculate` that overloads `calculate` method of
the `Work` class.

Invoking `calculate` method on the `Work` node triggers DFS calculation of its
sources - all of the sources are automatically calculated before the node itself is calculated.
Calculation of the `Work` node sets the value of its `data` attribute to the
outcome of the calculation, which in turn can depend on the `data` values of its
sources.

To avoid repeated calculation of the same node, `Work` has `derived` boolean flag
that is set to `True` once the node has been calculated, preventing subsequent
re-derivation. In particular, this ensures that DFS calculation of the node's
sources happens just once.

.. automodule:: numbox.core.work.work
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.work.work_utils
---------------------------

Overview
********

Convenience utilities for creating `Work`-graphs from Python scope.

The :func:`numbox.core.work.work.make_work` constructor accepts
`cres`-compiled derive function as an argument that requires
an explicitly provided signature of the `derive` function.
Return type of the `derive` function should match the type of the `data` attribute
of the corresponding `Work` instance while its argument types
should match the `data` types of the `Work` instance sources.

Utilities defined in this module make it easier to ensure these
requirements are met with a minimal amount of coding::

    import numpy
    from numbox.core.work.work_utils import make_init_data, make_work_helper


    pi = make_work_helper("pi", 3.1415)


    def derive_circumference(diameter_, pi_):
        return diameter_ * pi_


    def run(diameter_):
        diameter = make_work_helper("diameter", diameter_)
        circumference = make_work_helper(
            "circumference",
            make_init_data(),
            sources=(diameter, pi),
            derive_py=derive_circumference,
            jit_options={"cache": True}
        )
        circumference.calculate()
        return circumference.data


    if __name__ == "__main__":
        assert numpy.isclose(run(1.41), 3.1415 * 1.41)

.. automodule:: numbox.core.work.work_utils
   :members:
   :show-inheritance:
   :undoc-members:
