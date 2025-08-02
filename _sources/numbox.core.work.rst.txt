numbox.core.work
================

Overview
++++++++

Functionality for fully-jitted and light-weight calculation on a graph.

Modules
++++++++

numbox.core.work.builder
------------------------

Overview
********

Pure Python abstraction for creation of JIT'ed graph of :class:`numbox.core.work.work.Work` nodes.
Users can define end nodes [#f1]_::

    from numba.core.types import int16
    from numbox.core.work.builder import End

    w1_ = End(name="w1", init_value=137, ty=int16)
    w2_ = End(name="w2", init_value=3.14)
    w5_ = End(name="w5", init_value=10)
    w6_ = End(name="w6", init_value=7.5)

where optional capability to specify numba types of the nodes has been illustrated.
Suppose more values, `w3`, `w4`, `w7`, `w8`, `w9`, `w10`, are derived as follows::

    def derive_w3(w1_, w2_):
        if w1_ < 0:
            return 0.0
        elif w1_ < 1:
            return 2 * w2_
        return 3 * w2_

    def derive_w4(w1_):
        return 2 * w1_

    def derive_w7(w3_, w5_):
        return w3_ + (w5_ ** 2)

    def derive_w8(w6_, w2_):
        if w6_ > 5:
            return w6_ * w2_
        else:
            return w6_ + w2_

    def derive_w9(w3_, w4_, w7_):
        return (w4_ - w3_) / (abs(w7_) + 1e-5)

    def derive_w10(w3_, w4_, w7_, w8_, w9_):
        return (w3_ + w4_ + w7_) * 0.1 + (w8_ - w9_)

Users can then declare the corresponding nodes (in fact `End` and `Derived`
are better termed as 'node specs', while instances of :class:`numbox.core.work.work.Work` are
the actual node objects) as::

    from numbox.core.work.builder import Derived

    w3_ = Derived(name="w3", init_value=0.0, derive=derive_w3, sources=(w1_, w2_))
    w4_ = Derived(name="w4", init_value=0.0, derive=derive_w4, sources=(w1_,))
    w7_ = Derived(name="w7", init_value=0.0, derive=derive_w7, sources=(w3_, w5_))
    w8_ = Derived(name="w8", init_value=0.0, derive=derive_w8, sources=(w6_, w2_))
    w9_ = Derived(name="w9", init_value=0.0, derive=derive_w9, sources=(w3_, w4_, w7_))
    w10_ = Derived(name="w10", init_value=0.0, derive=derive_w10, sources=(w3_, w4_, w7_, w8_, w9_))

DAG with the access nodes `w7`, `w9`, `w10`, can then be constructed and used as follows [#f2]_::

    from numbox.core.work.builder import make_graph

    access = make_graph(w7_, w9_, w10_)
    w7 = access.w7
    w9 = access.w9
    w10 = access.w10

Here `access` is a named tuple containing instances of :class:`numbox.core.work.work.Work` node structure::

    from numbox.core.work.work import Work

    assert isinstance(w7, Work)
    assert isinstance(w9, Work)
    assert isinstance(w10, Work)

One can then compute and access values of the derived nodes::

    from numpy import isclose

    assert w10.data == 0
    w10.calculate()
    assert isclose(w7.data, 109.42)
    assert isclose(w9.data, 2.418022)
    assert isclose(w10.data, 60.416)

Nodes are required to have unique `name` attribute. Attempting to declare multiple
nodes, either `End` or `Derived`, with the same `name` will raise `ValueError`.

Graph structure can be inspected as::

    from numbox.core.work.print_tree import make_image

    print(make_image(w10))

which outputs::

    w10--w3--w1
         |   |
         |   w2
         |
         w4--w1
         |
         w7--w3--w1
         |   |   |
         |   |   w2
         |   |
         |   w5
         |
         w8--w6
         |   |
         |   w2
         |
         w9--w3--w1
             |   |
             |   w2
             |
             w4--w1
             |
             w7--w3--w1
                 |   |
                 |   w2
                 |
                 w5

The visualization utility for a sub-graph tree :func:`numbox.core.work.print_tree.make_graph`
spans the breadth in the vertical direction and the depth in the horizontal direction.
The nodes are uniquely identified by their names.
To simplify the image structure, the same node can appear on one graph image multiple times.

Each `Work` can be represented as light-weight :class:`numbox.core.work.node.Node` type,
which stores its `node` attribute upon first invocation::

    w3_n = w3.as_node()

This enables an assortment of utilities, such as::

    assert w3.all_inputs_names() == ["w1", "w2"]
    assert w7.all_inputs_names() == ["w3", "w1", "w2", "w5"]
    assert w3.depends_on("w1")
    assert w3.get_input(0).name == "w1"
    assert w10.get_input(3).name == "w8"

From the given access node, values of nodes can be combined as follows::

    from numba.core.types import float64
    from numbox.core.work.combine_utils import make_sheaf_dict

    requested = ("w1", "w4", "w7", "w8")
    sheaf = make_sheaf_dict(requested)
    w10.combine(sheaf)
    assert isclose(sheaf["w4"].get_as(float64), 274)
    assert isclose(sheaf["w7"].get_as(float64), 109.42)
    assert isclose(sheaf["w8"].get_as(float64), 23.55)

Graph nodes can be loaded from the access node as follows [#f3]_::

    from numba.core.types import int16, unicode_type
    from numba.typed.typeddict import Dict
    from numbox.core.any.any_type import Any, make_any

    load_data = Dict.empty(key_type=unicode_type, value_type=AnyType)
    assert sheaf["w1"].get_as(int16) == 137
    load_data["w1"] = make_any(12)
    w10.load(load_data)
    w10.combine(sheaf)
    assert sheaf["w1"].get_as(int16) == 12

Recalculating the graph then renders new values of the affected nodes [#f4]_::

    w10.calculate()
    w10.combine(sheaf)
    assert isclose(sheaf["w4"].get_as(float64), 24)

Builder supports smart caching of compiled code that builds the graph from the specified
`End` and `Derived` nodes. If caching of the JIT'ed functions is configured (which is a default behavior),
the graph maker is re-compiled only when `init_value` or `derive` functions of the nodes are changed.

The `init_value` attribute of the `End` and `Derived` nodes can be assigned a variety of values,
including scalars, arrays, and instances of `StructRef`. In the latter case, it is recommended
to define `__repr__` method of the StructRef proxy class, so that the builder knows when the
values contained in the StructRef have changed. Otherwise, the default `__repr__` of the
StructRef containing dynamic address of the struct object will be used, making the builder
recompile every time it's invoked.

From each given accessor `Work` node, one can trace down its derivation to all the `End` nodes::

    from numbox.core.work.explain import explain

    derivation_of_w9 = explain(w9)
    print(derivation_of_w9)

will return::

    All required end nodes: ['w1', 'w2', 'w5']

    w1: end node

    w2: end node

    w3: derive_w3(w1, w2)

        def derive_w3(w1_, w2_):
            if w1_ < 0:
                return 0.0
            elif w1_ < 1:
                return 2 * w2_
            return 3 * w2_

    w4: derive_w4(w1)

        def derive_w4(w1_):
            return 2 * w1_

    w5: end node

    w7: derive_w7(w3, w5)

        def derive_w7(w3_, w5_):
            return w3_ + (w5_ ** 2)

    w9: derive_w9(w3, w4, w7)

        def derive_w9(w3_, w4_, w7_):
            return (w4_ - w3_) / (abs(w7_) + 1e-5)

This provides information of what pure inputs / end nodes are required to derive
the given node as well as the logical sequence of steps (as indicated by the graph structure)
to carry out the derivation.

By default, defining `End` or `Derived` node spec will store the created node spec instance
in the global registry `_specs_registry` (in :mod:`numbox.core.work.builder`) as a value paired to the key
given by the node's name.
Attempting to create more than one node with the same name will then raise an error.

Optionally, any number of (local) registries can be created to register given node specs in.
To do so, provide a dictionary-valued argument as the `registry` parameter of either `Derived` or `End` node.
Then the `make_graph` utility needs to be provided with *the same* registry as its *keyword* argument::

    reg_1 = {}
    end_1 = End(name="end_1", init_value=0.0, registry=reg_1)

    reg_2 = {}
    end_1_another = End(name="end_1", init_value=0.0, registry=reg_2)

    der_1 = Derived(name="der_1", init_value=0.0, sources=(end_1,), registry=reg_1, derive=lambda x: x + 2.17)
    accessors_1 = make_graph(der_1, registry=reg_1)
    der_1_ = accessors_1.der_1
    der_1_.calculate()
    assert isclose(der_1_.data, 2.17)

    der_1_another = Derived(
        name="der_1", init_value=0.0, sources=(end_1_another,), registry=reg_2, derive=lambda x: x + 3.14
    )
    accessors_2 = make_graph(der_1_another, registry=reg_2)
    der_1_another_ = accessors_2.der_1
    der_1_another_.calculate()
    assert isclose(der_1_another_.data, 3.14)

In this example, there are two nodes named "end_1" and two nodes named "der_1", but they
are defined in different registries, and the two corresponding graphs are built from the specs
obtained from their respective registries.

.. [#f1] For simplicity, this example illustrates nodes that contain scalar data (integers and floats). More complex types of data, like numpy arrays, strings, and aggregates (numba StructRef's, typed lists and dictionaries) are supported as well.
.. [#f2] Behind the scenes, :func:`numbox.core.work.builder.make_graph` compiles (and optionally caches) a graph maker with low-level intrinsic constructors of the individual work nodes inlined into it. All the Python 'derive' functions defined for the `Derived` nodes are compiled for the signatures inferred from the types of the derived nodes and their sources.
.. [#f3] For numpy-compatible data types, additional utilities are available in :mod:`numbox.core.work.loader_utils`.
.. [#f4] For numpy-compatible data types, additional utilities are available in :mod:`numbox.core.work.combine_utils`.

.. automodule:: numbox.core.work.builder
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.work.combine_utils
------------------------------

.. automodule:: numbox.core.work.combine_utils
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.work.loader_utils
-----------------------------

.. automodule:: numbox.core.work.loader_utils
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.work.node
---------------------

Overview
********

:class:`numbox.core.work.node.Node` represents a node on a directed acyclic graph
(`DAG <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_)
that exists in a fully jitted scope and is accessible both at the low-level and via a Python proxy.

`Node` can be used on its own (in which case the recommended way to
create it is via the factory function :func:`numbox.core.work.node.make_node`)
or as a prototype to more functionally-rich graph nodes,
such as :class:`numbox.core.work.work.Work`.

The logic of `Node` and its sub-classes follows a graph-optional design - no
graph orchestration structure is required to register and manage the graph
of `Node` instance objects - which in turn reduces unnecessary computation overhead
and simplifies the program design.

To that end, each node is identified by its name and contains a uniformly-typed vector-like
container member (rendered by the numba-native `numba.core.typed.List`) with all the
input nodes references that it bears a directed dependency relationship to.
This enables a traversal not only of graphs of `Node` instances themselves
but also graphs of objects representable by it, such as, the graphs of `Work` nodes.

`Node` implementation makes heavy use of the numba
`meminfo <https://numba.readthedocs.io/en/stable/developer/numba-runtime.html?highlight=meminfo#memory-management>`_
paradigm that manages memory-allocated
payload via smart pointer (pointer to numba's meminfo object) reference counting.
This allows users to reference the desired
memory location via a 'void' structref type, such as,
:class:`numbox.core.any.erased_type.ErasedType`, or :class:`numbox.core.utils.void_type.VoidType`,
or base structref type, such as, :class:`numbox.core.work.node_base.NodeBaseType`,
and dereference its payload accordingly when needed via the appropriate :func:`numbox.utils.lowlevel.cast`.

.. automodule:: numbox.core.work.node
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.work.node_base
--------------------------

Overview
********

Base class for :class:`numbox.core.work.node.Node` and :class:`numbox.core.work.work.Work`.
Contains functionality dependent only on the node name.

.. automodule:: numbox.core.work.node_base
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

`Work` type subclasses :class:`numbox.core.work.node_base.NodeBase`
and follows the logic of graph design of :class:`numbox.core.work.node.Node`.
However, since numba StructRef does not support low-level subclasses,
there is no inheritance relation between `NodeBaseType` and `WorkType`,
leaving the data design to follow the composition pattern.
Namely, the member (`name`) of the `NodeBase` payload is a header in the payload of `Work`, allowing
to perform a meaningful :func:`numbox.utils.lowlevel.cast`.

The main way to create `Work` object instance is via the :func:`numbox.core.work.work.make_work`
constructor (`Work(...)` instantiation is in fact disabled both in Python and jitted scope)
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

For performance-critical large graphs containing hundreds or more nodes created
in a jitted scope, using :func:`numbox.core.work.work.make_work` is not
feasible as it either results in large memory use (and takes up a lot of
disk space when the jitted caller is cached), or takes a substantial time to
compile when `make_work` is declared with `inline=True` directive (albeit
resulting in a much slimmer and optimized final compilation result).
For that purpose it is recommended to use a low-level intrinsic
:func:`numbox.core.work.lowlevel_work_utils.ll_make_work` as follows::

    from numba import njit
    from numba.core.types import float64
    from numpy import isclose

    from numbox.core.work.node_base import NodeBaseType
    from numbox.core.work.lowlevel_work_utils import ll_make_work, create_uniform_inputs
    from numbox.core.work.print_tree import make_image
    from numbox.utils.highlevel import cres

    @cres(float64())
    def derive_v0():
        return 3.14

    @njit(cache=True)
    def v0_maker(derive_):
        return ll_make_work("v0", 0.0, (), derive_)

    v0 = v0_maker(derive_v0)
    assert v0.data == 0
    assert v0.name == "v0"
    assert v0.inputs == ()
    assert not v0.derived
    v0.calculate()
    assert isclose(v0.data, 3.14)

Importantly, `Work` objects support :func:`numbox.core.work.work.ol_as_node`
rendition `as_node` that creates a :class:`numbox.core.work.node.Node` instance
with the same name as the `Work` instance and the vector of inputs of the
:class:`numbox.core.work.node_base.NodeBase` type referencing the original `Work`
instance's sources. Upon the first invocation of `as_node` on the given `Work`
instance, `Node` representations for itself and recursively all its sources
are created and stored in their `node` attributes only once. Subsequent invocations
of `as_node` on either the given `Work` node or any of nodes on its sub-graph
will return the previously created `Node` objects stored as the `node` attribute.

Graph manager
*************

While not a requirement, it is recommended that the `Work` instance's `name` attribute matches the name
of the variable to which that instance is assigned.
Moreover, no out-of-the-box assertions for uniqueness of the `Work` names is provided.
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
