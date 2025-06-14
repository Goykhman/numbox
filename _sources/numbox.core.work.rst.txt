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
that can exist
in a fully jitted scope and optionally cross the border to the Python interpreter side.

`Node` can be used on its own (in which case the preferred way to
create it is via the factory function :func:`numbox.core.work.node.make_node`)
or as a base to more functionally-rich graph nodes,
such as :class:`numbox.core.work.work.Work`.

The logic of `Node` and its sub-classes follows a graph-optional design - no
graph orchestration structure is required to register and manage the graph
of `Node` instance objects - which in turn reduced unnecessary computation overhead
and simplifies the program design.

To that end, each node is identified by its name and contains a uniformly-typed
container member with all the input nodes references that it bears a directed dependency relationship too.
This enables a traversal not only of the graph of `Node` instances itself but also graphs of its subclasses,
such as, a graph of `Work` nodes.

`Node` implementation makes heavy use of numba
`meminfo <https://numba.readthedocs.io/en/stable/developer/numba-runtime.html?highlight=meminfo#memory-management>`_
paradigm that manages memory-allocated
payload via smart pointer (meminfo pointer) reference counting. This allows the user to reference the desired
memory location via a specially-designed structref type, such as,
:class:`numbox.core.any.erased_type.ErasedType`, or :class:`numbox.core.utils.void_type.VoidType`,
and dereference its payload accordingly and as needed via the appropriate :func:`numbox.utils.lowlevel.cast`.

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

which produces::

    fifth--third---first
           |       |
           |       second
           |
           fourth

Notice that the tree depth extends in horizontal direction.

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
However, since numba StructRef does not support low-level subclasses (between `NodeType` and `WorkType`),
the relation between `Node` and `Work` follows the composition pattern.
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
is a requirement, the `derive` function should be passed to it as
a `FunctionType` (not `njit`-produced `CPUDispatcher`) argument, i.e.,
decorated with :func:`numbox.utils.highlevel.cres`).

Behind the scenes, `Work` accommodates individual access to its `sources`
(other `Work` nodes that are pointing to the given `Work` node on the DAG)
via a 'Python-native compiler' backdoor, which is essentially a relative pre-runtime
technique to Python-compile and execute custom functions before preparing for
overload in the numba jitted scope. This technique is fully compatible with caching of
jitted functions and provides a natural Python counterpart to virtual functions that
are not supported in numba. Here it is extensively utilized in
:func:`numbox.core.work.work.ol_calculate` that overloads `calculate` method of
the `Work` class.

Invoking `calculate` method on the `Work` node triggers DFS calculation of its
sources - all of the sources are calculated before the node itself is calculated.
Calculation of the `Work` node sets the value of its `data` attribute to the
outcome of the calculation, which in turn can depend on the `data` values of its
sources.

To avoid repeated calculation of the same node, `Work` has `derived` boolean flag
that is set to `True` once the node has been calculated, preventing subsequent
re-derivation.

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
