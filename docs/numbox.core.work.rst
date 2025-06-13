numbox.core.work
================

numbox.core.work.node
---------------------

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
