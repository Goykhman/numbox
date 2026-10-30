numbox.core.work
================

Overview
++++++++

Functionality for fully-jitted and light-weight calculation on a graph. Fully jitted means that the whole
calculation, including the traversal of the graph, runs in numba-compiled code rather than in Python.

Modules
++++++++

.. _builder:

numbox.core.work.builder
------------------------

Overview
********

Pure Python abstraction for creation of JIT'ed graph of :class:`numbox.core.work.work.Work` nodes.
Later examples of this overview use names that earlier examples define, so run the examples in order, in one
session.
Users can define the specs of end nodes [#f1]_::

    from numba.core.types import int16
    from numbox.core.work.builder import End

    w1_ = End(name="w1", init_value=137, ty=int16)
    w2_ = End(name="w2", init_value=3.14)
    w5_ = End(name="w5", init_value=10)
    w6_ = End(name="w6", init_value=7.5)

where `ty=int16` illustrates the optional `ty` argument, which sets the numba type of a node.
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

`End` and `Derived` are node specs rather than nodes: the actual node objects are the instances of
:class:`numbox.core.work.work.Work` that the builder creates from them. Users can then declare the specs of the
derived nodes as::

    from numbox.core.work.builder import Derived

    w3_ = Derived(name="w3", init_value=0.0, derive=derive_w3, sources=(w1_, w2_))
    w4_ = Derived(name="w4", init_value=0.0, derive=derive_w4, sources=(w1_,))
    w7_ = Derived(name="w7", init_value=0.0, derive=derive_w7, sources=(w3_, w5_))
    w8_ = Derived(name="w8", init_value=0.0, derive=derive_w8, sources=(w6_, w2_))
    w9_ = Derived(name="w9", init_value=0.0, derive=derive_w9, sources=(w3_, w4_, w7_))
    w10_ = Derived(name="w10", init_value=0.0, derive=derive_w10, sources=(w3_, w4_, w7_, w8_, w9_))

A DAG can then be constructed and used as follows [#f2]_. Its access nodes are `w7`, `w9` and `w10`: their specs
`w7_`, `w9_` and `w10_` are passed to :func:`numbox.core.work.builder.make_graph`, which returns a `Work` node for
each of them::

    from numbox.core.work.builder import make_graph

    access = make_graph(w7_, w9_, w10_)
    w7 = access.w7
    w9 = access.w9
    w10 = access.w10

Here `access` is a named tuple with one :class:`numbox.core.work.work.Work` node for each access node::

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

Specs are required to have a unique `name` attribute within their registry, the dictionary in which the builder
keeps them by name. Attempting to declare a second spec, either `End` or `Derived`, with a `name` that is already in
the registry will raise `ValueError`. The paragraphs on the global registry `_specs_registry`, near the end of this
overview, describe registries.

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

The visualization utility for a sub-graph tree :func:`numbox.core.work.print_tree.make_image`
spans the breadth in the vertical direction and the depth in the horizontal direction.
The nodes are uniquely identified by their names.
To simplify the image structure, the same node can appear on one graph image multiple times.

Each `Work` can be represented as a light-weight :class:`numbox.core.work.node.Node`. The first call of `as_node`
creates the `Node` and stores it in the `node` attribute of the `Work`, and later calls return the stored `Node`.
Here `w3`, the first source of `w10`, is taken from `w10.sources`::

    w3 = w10.sources[0]
    w3_n = w3.as_node()

The `Node` representation enables an assortment of utilities, which `Work` provides as methods that delegate to the
`Node`. `all_inputs_names` lists the inputs transitively: the inputs of `w7` include `w1` and `w2`, the inputs of its
source `w3`::

    assert w3.all_inputs_names() == ["w1", "w2"]
    assert w7.all_inputs_names() == ["w3", "w1", "w2", "w5"]
    assert w3.depends_on("w1")
    assert w3.get_input(0).name == "w1"
    assert w10.get_input(3).name == "w8"

From an access node, here `w10`, values of nodes can be combined into a sheaf. A sheaf is a typed dictionary that
maps each requested node name to an `Any` value (:doc:`numbox.core.any`), and `w10.combine` sets each value to the
`data` of the node of that name in the graph of `w10`::

    from numba.core.types import float64
    from numbox.core.work.combine_utils import make_sheaf_dict

    requested = ("w1", "w4", "w7", "w8")
    sheaf = make_sheaf_dict(requested)
    w10.combine(sheaf)
    assert isclose(sheaf["w4"].get_as(float64), 274)
    assert isclose(sheaf["w7"].get_as(float64), 109.42)
    assert isclose(sheaf["w8"].get_as(float64), 23.55)

Values can be loaded into graph nodes through the access node as follows [#f3]_::

    from numba.core.types import int16, unicode_type
    from numba.typed.typeddict import Dict
    from numbox.core.any.any_type import AnyType, make_any

    load_data = Dict.empty(key_type=unicode_type, value_type=AnyType)
    assert sheaf["w1"].get_as(int16) == 137
    load_data["w1"] = make_any(12)
    w10.load(load_data)
    w10.combine(sheaf)
    assert sheaf["w1"].get_as(int16) == 12

`load` clears the `derived` flag of each loaded node and of each node that depends on it, up to the node on which
`load` is called. Recalculating the graph then computes new values of the affected nodes [#f4]_::

    w10.calculate()
    w10.combine(sheaf)
    assert isclose(sheaf["w4"].get_as(float64), 24)

Builder supports smart caching of compiled code that builds the graph from the specified
`End` and `Derived` specs. If caching of the JIT'ed functions is configured (which is a default behavior),
the graph maker is re-compiled when the name, sources, numba type or `derive` function of a node changes. It is
not re-compiled when only the value of an `init_value` changes, because the cached graph maker receives the
initial values as arguments.

The `init_value` attribute of the `End` and `Derived` specs can be assigned a variety of values,
including scalars, arrays, and instances of `StructRef`. Only the numba type of each value enters the builder's
cache key, so a changed value, a `StructRef` included, does not make the builder recompile.

From each given access node, one can trace its derivation to all the `End` nodes::

    from numbox.core.work.explain import explain

    derivation_of_w9 = explain(w9)
    print(derivation_of_w9)

prints::

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

This lists the end nodes that the derivation of the given node needs, that is, the inputs that no `derive`
computes, followed by the logical sequence of steps (as indicated by the graph structure) to carry out the
derivation.

By default, defining `End` or `Derived` node spec will store the created node spec instance
in the global registry `_specs_registry` (in :mod:`numbox.core.work.builder`) as a value paired to the key
given by the spec's name.
A second spec with the same name in the same registry raises the `ValueError` described above.

Optionally, any number of (local) registries can be created, and node specs can be registered in them.
To do so, provide a dictionary-valued argument as the `registry` parameter of either `Derived` or `End`.
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

In this example, there are two specs named "end_1" and two specs named "der_1", but they
are defined in different registries, and the two corresponding graphs are built from the specs
obtained from their respective registries.

.. [#f1] For simplicity, this example illustrates nodes that contain scalar data (integers and floats). More complex
   types of data, like numpy arrays, strings, and aggregates (numba StructRef's, typed lists and dictionaries) are
   supported as well.
.. [#f2] Internally, :func:`numbox.core.work.builder.make_graph` compiles (and optionally caches) a graph maker with
   low-level intrinsic constructors of the individual work nodes inlined into it. All the Python 'derive' functions
   defined for the `Derived` specs are compiled for the signatures inferred from the types of the derived nodes and
   their sources.
.. [#f3] For numpy-compatible data types, additional utilities are available in :mod:`numbox.core.work.loader_utils`.
.. [#f4] For numpy-compatible data types, additional utilities are available in :mod:`numbox.core.work.combine_utils`.

.. automodule:: numbox.core.work.builder
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.work.builder_utils
------------------------------

.. automodule:: numbox.core.work.builder_utils
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

The logic of `Node` and its sub-classes follows a graph-optional design: no graph orchestration structure is
required to register and manage the graph of `Node` instance objects. This in turn reduces unnecessary
computation overhead and simplifies the program design.

To that end, each node is identified by its name and contains a uniformly-typed vector-like
container member, a numba-native `numba.typed.List`, with references to all the input nodes that it depends on.
This enables a traversal not only of graphs of `Node` instances themselves
but also graphs of objects representable by it, such as, the graphs of `Work` nodes.

`Node` implementation makes heavy use of the numba
`meminfo <https://numba.readthedocs.io/en/stable/developer/numba-runtime.html?highlight=meminfo#memory-management>`_
paradigm that manages memory-allocated
payload via smart pointer (pointer to numba's meminfo object) reference counting.
This allows users to reference the desired
memory location via a 'void' structref type, such as,
``numbox.core.any.erased_type.ErasedType``, or ``numbox.utils.void_type.VoidType``,
or base structref type, such as, ``numbox.core.work.node_base.NodeBaseType``,
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

Both references to 'n1' point to the same node,
which is a source of two other nodes, 'n2' and 'n3'.

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

The Python class `Work` subclasses :class:`numbox.core.work.node_base.NodeBase`
and follows the logic of graph design of :class:`numbox.core.work.node.Node`.
However, since numba StructRef does not support low-level subclasses,
there is no inheritance relation between `NodeBaseType` and `WorkType`,
leaving the data design to follow the composition pattern.
Namely, the member (`name`) of the `NodeBase` payload is a header in the payload of `Work`, which makes
a :func:`numbox.utils.lowlevel.cast` between the two meaningful.

The main way to create a `Work` instance is via the :func:`numbox.core.work.work.make_work` constructor
(`Work(...)` instantiation is in fact disabled both in Python and jitted scope). `make_work` can be invoked either
from Python or from jitted scope: the `run` function below works with or without its `@njit` decorator. Its
`derive` is compiled with :func:`numbox.utils.highlevel.cres`, which compiles a function for one signature and
returns it as a first-class function value::

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
decorated with :func:`numbox.utils.highlevel.cres`. If `run` instead took no
argument and read `derive_work` from the global scope, numba could not cache `run`.

For performance-critical large graphs, with hundreds of nodes or more created in a jitted scope,
:func:`numbox.core.work.work.make_work` is not feasible. Called from jitted code, it results in large memory use,
and in large disk use when the jitted caller is cached. Declaring `make_work` with numba's `inline=True` directive
would produce a much smaller and optimized final compilation result, but its compilation would take a
substantial time. For such graphs it is recommended to use the low-level intrinsic
``numbox.core.work.lowlevel_work_utils.ll_make_work`` as follows::

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

Importantly, `Work` objects support the `as_node` method, implemented by :func:`numbox.core.work.work.ol_as_node`.
It creates a :class:`numbox.core.work.node.Node` instance with the same name as the `Work` instance and a vector
of inputs of the :class:`numbox.core.work.node_base.NodeBase` type that references the `Node` representations of
the original `Work` instance's sources. Upon the first invocation of `as_node` on the given `Work`
instance, `Node` representations for itself and recursively all its sources
are created and stored in their `node` attributes only once. Subsequent invocations
of `as_node` on either the given `Work` node or any of nodes on its sub-graph
will return the previously created `Node` objects stored as the `node` attribute.

Exception handling
******************

A node's ``derive`` is the function that computes the node's ``data`` from the ``data`` of its sources. If the
``derive`` raises an exception, :meth:`~numbox.core.work.work.Work.calculate` raises the same exception, with its type
and message. The node keeps its ``data``. Its ``derived`` flag, which ``calculate`` sets once the node is calculated,
stays unset, so a later ``calculate`` runs the ``derive`` again. Three ways of passing a ``derive`` lose the exception
instead, as `When the exception is lost`_ describes.

In the example below, :func:`~numbox.core.work.work_utils.make_work_helper` builds the nodes. Its ``derive_py``
argument takes the ``derive`` as a plain Python function, which the helper compiles with
:func:`~numbox.utils.highlevel.cres`. The source holds a one-element array, so that its value can change in place
between the two calls:

.. code-block:: python

    import numpy
    from numbox.core.work.work_utils import make_work_helper


    def derive_reciprocal(x_):
        if x_[0] == 0.0:
            raise ValueError("zero input")
        return 1.0 / x_[0]


    source = make_work_helper("source", numpy.zeros(1))
    node = make_work_helper("node", 1.0, sources=(source,), derive_py=derive_reciprocal)
    try:
        node.calculate()
    except ValueError as e:
        print(e)                     # zero input
    assert node.data == 1.0
    assert not node.derived

    source.data[0] = 4.0
    node.calculate()
    assert node.data == 0.25

``cres`` compiles a function for one signature and returns it as a `first-class function value
<https://numba.readthedocs.io/en/stable/reference/types.html#functions>`_, a value that jitted code can receive and
call. This page calls such a value a ``cres`` function. Exceptions propagate in the same way in four cases:

- a ``cres`` function passed to :func:`~numbox.core.work.work.make_work` from Python;
- a ``cres`` function passed to a jitted function that builds the node with ``make_work`` or ``ll_make_work``;
- a plain Python function given to :class:`~numbox.core.work.builder.Derived`, which compiles it with ``cres``;
- the ``.as_func`` value of a function decorated with :func:`~numbox.core.proxy.proxy.proxy`.

The sections below describe the limits of this rule: three ways of passing a ``derive`` that lose its exception,
tuples that mix kinds of function values, jitted code that catches the exception, and parallel loops. A ``derive``
that must report a failure in those situations can do so through its return value, for example by returning a NaN.

When the exception is lost
~~~~~~~~~~~~~~~~~~~~~~~~~~

In three cases ``calculate`` returns normally although the ``derive`` raised. Python prints the exception to stderr
below a line that begins ``Exception ignored in:``, which means that the exception was printed and then
discarded.
``calculate`` stores a zero-filled value of the ``derive``'s return type in ``data``, such as ``0.0`` or an empty
array, and sets ``derived``. A later ``calculate`` therefore does not run the ``derive`` again, and the node keeps the
zero-filled value.

.. warning::

   A zero-filled string holds a null pointer. Reading such a ``data`` from Python, as ``node.data``, ends the process
   with a segmentation fault.

The three cases are:

- A ``derive`` passed from Python to a jitted function whose parameter is declared as numba's ``FunctionType``, from
  ``numba.types``. This page calls numba's own type the plain ``FunctionType``; numbox assigns a ``cres`` function the
  type ``DeriveFunctionType``, a subclass of it. For example, ``@njit(float64(FunctionType(float64(float64))))``
  declares one parameter of the plain type, for a function that takes and returns a ``float64``. The exception is lost
  because numba converts the value to the plain type when Python passes it to the jitted function. The loss occurs only
  at that boundary between Python and jitted code: within jitted code, a ``derive`` passed on to a function that
  declares the plain ``FunctionType`` still propagates its exception. For the exception to propagate, let numba infer
  the parameter's type, or declare the parameter as ``DeriveFunctionType(float64(float64))``, with
  ``DeriveFunctionType`` from :mod:`numbox.utils.derive_wap`. A parameter declared that way accepts only ``cres``
  functions and ``.as_func`` values.
- A ``CompileResultWAP`` that you build with numba directly. ``CompileResultWAP``, from
  ``numba.core.types.function_type``, is numba's function value for one compiled function, and
  ``CompileResultWAP(f.get_compile_result(sig))`` builds one for an ``@njit`` function ``f``. The exception is lost
  when such a value reaches jitted code as an argument or as a global variable, because the value has the plain
  ``FunctionType``. :func:`~numbox.core.work.work.make_work` upgrades the value when Python calls ``make_work``, but a
  jitted function that receives the value cannot upgrade it. For the exception to propagate, build the ``derive``
  with ``cres``, or upgrade the value in Python with :func:`~numbox.utils.derive_wap.rewrap_derive` before jitted
  code receives it.
- A ``@cfunc`` used as a ``derive``, whether it reaches ``make_work`` from Python or from jitted code. The exception is
  lost because a ``@cfunc`` has the plain ``FunctionType`` and numbox does not upgrade a ``@cfunc``. For the exception
  to propagate, compile the function with ``cres`` instead.

numba can call a function value through two entry points. The numba entry point follows numba's own calling convention
and propagates an exception to the caller. The C entry point, the kind that ``@cfunc`` builds, does not propagate an
exception: numba discards the exception there, as `numba issue 8246 <https://github.com/numba/numba/issues/8246>`_
describes. numba records the numba entry point of a value only when the value comes from a dispatcher. For that reason
an ``@njit`` function compiled for a signature still propagates its exception when it is passed to a parameter of the
plain ``FunctionType``. A compile result that numba's own ``CompileResultWAP`` wraps is called through its C entry
point. ``cres`` therefore wraps its compile result in a :class:`~numbox.utils.derive_wap.DeriveWAP` instead, and numbox
assigns that wrapper the numba type :class:`~numbox.utils.derive_wap.DeriveFunctionType`. That type records the numba
entry point, and ``calculate`` calls a ``derive`` of that type through it. In the three cases above, the ``derive``
reaches ``calculate`` as a value of the plain ``FunctionType`` with only its C entry point.
:mod:`numbox.utils.derive_wap` describes the mechanism.

Other limits
~~~~~~~~~~~~

.. _derive_mixed_function_values:

Jitted code can receive several function values at once as the elements of a tuple, for example the ``derive`` functions
of the nodes it builds. Below numba 0.68.0, function values that take the same number of arguments can be elements of
one tuple only if they have the same type, such as ``cres`` functions of one signature. Values that take different
numbers of arguments are not affected, on any numba version. Four kinds of function value therefore cannot be elements
of one tuple with a ``cres`` function: a ``@cfunc``, an ``@njit`` function compiled for a signature, numba's own
``CompileResultWAP``, and a ``cres`` function of another signature. Such a tuple fails with an ``AssertionError`` that
has no message. The assertion is numba's own check, and two plain function values of different types fail it as well. A
``cres`` function fails the check more often, because numbox assigns it ``DeriveFunctionType``, while numba assigns the
other kinds the plain ``FunctionType``. numba 0.68.0 accepts every such tuple, and each ``cres`` function in it still
propagates its exception.

Under ``python -O``, which removes assertions, numba below 0.68.0 skips that check and assigns the whole tuple the type
of its first element. A ``cres`` function that comes after another kind in the tuple then loses its exception. Another
kind that comes after a ``cres`` function raises a ``TypeError`` that names the value's class.

A lazily compiled ``@njit`` function, one decorated without a signature, cannot be an element of one tuple with a
``cres`` function on any numba version. Passing such a tuple to a jitted function raises this error:

.. code-block:: text

    TypeError: DeriveFunctionType value must be a DeriveWAP, got <class 'numba.core.registry.CPUDispatcher'>

The message means that numba typed the whole tuple as ``DeriveFunctionType`` and numbox refused the ``CPUDispatcher``
value, the lazily compiled function. Compile that function with ``cres`` too.

The ``.as_func`` value of a proxied function counts as a ``cres`` function in these rules. The proxy page's section on
:ref:`passing the function as a value <proxy_passing_as_value>` states the same rules for ``.as_func``.

A jitted caller can catch the exception of a ``derive`` only with a bare ``except:`` or with ``except Exception:``.
numba accepts no other ``except`` clause, and it cannot bind the exception to a name with ``as``; see numba's
`try .. except <https://numba.readthedocs.io/en/stable/reference/pysupported.html#try-except>`_. Jitted code therefore
cannot read the exception's type or message.

Exceptions in parallel loops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``raise`` written directly in the body of a ``prange`` loop makes numba run that loop serially. numba runs a loop in
parallel only if the loop has a single exit (see numba's `explicit parallel loops
<https://numba.readthedocs.io/en/stable/user/parallel.html#explicit-parallel-loops>`_), and it warns with a
``NumbaPerformanceWarning`` when it cannot. A ``derive`` whose own ``prange`` loop raises in this way therefore
propagates its exception like any other ``derive``. A ``derive`` compiled with ``nogil=True`` also propagates its
exception like any other ``derive``, because ``nogil`` only releases Python's global interpreter lock while the
``derive`` runs. The ``derive`` still runs on the thread that called it, and no parallel loop is involved.

An exception raised in a function that the body of a parallel loop calls is different, because numba does run that loop
in parallel. The thread that ran the failing iteration skips the rest of the iterations assigned to it, so the result
does not include the computation of that iteration or of the later iterations assigned to that thread. With numba's omp
threading layer, the call from Python raises ``SystemError: ... returned a result with an exception set`` when the
failing iteration ran on the calling thread, and returns normally when it ran on another thread. With the workqueue
layer, the call always returns normally. The ``SystemError`` means that the jitted function returned while the exception
was still pending. Its ``__cause__`` is the original exception, and an ``except`` clause for the original exception's
type does not catch the ``SystemError``. Either way, the jitted code after the loop runs as if the loop had succeeded. A
plain jitted function that does not use numbox behaves the same way.

If ``calculate`` runs in the body of a parallel loop and the ``derive`` raises, the node is left intact: it keeps its
``data``, and ``derived`` stays unset. Check ``node.derived`` after the loop to learn whether the ``derive``
succeeded. A later ``calculate`` outside the loop raises the exception normally.

If the ``derive`` itself runs a parallel loop and a function called from its body raises, the ``derive`` returns
normally to ``calculate``, which cannot detect the failure. ``calculate`` stores the incomplete result in ``data`` and
sets ``derived``, so the failure stays cached, even when the call from Python then raises the ``SystemError``. In a
``derive``, do not raise exceptions in the functions that a parallel loop calls.

After you upgrade or downgrade numbox
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

numbox 0.7.3 is the first release in which the exception of a ``derive`` propagates, on numba 0.61 or later. A
function of your own that numba caches may keep the old behavior after the upgrade. Its cache entry holds the code
that numbox generated for its calls of ``calculate``, and numba's key for the entry leaves out the code of the
functions it calls (:ref:`what the key leaves out <cache_key_leaves_out>`). As long as the function's file and
signature stay unchanged, numba keeps loading that code.

A cached function that takes a ``cres`` function, or a node built from one, recompiles automatically, because numbox
0.7.3 assigns a ``cres`` function a new numba type on numba 0.61 or later. Consider instead a cached function whose
parameter is declared as the plain ``FunctionType``. If it receives an ``@njit`` function compiled for a signature,
numba continues to load its 0.7.2 code, which discards the exception that 0.7.3 propagates. Delete that function's cache
files, or save its source file again so that numba sees a newer modification time; the configurations page says
:ref:`where numba writes the cache <where_the_cache_lands>`. For later upgrades, decorate such a function with
:func:`~numbox.core.cache_key.numbox_njit`, which numbox provides from 0.7.8. Its key includes numbox's version, so the
function recompiles after every upgrade, as the cache_key page explains under
:ref:`After you upgrade numbox <cache_key_after_upgrade>`.

numbox 0.7.4 assigns the ``.as_func`` value of a proxied function the numba type of a ``cres`` function, on numba
0.61 or later; before, it was a plain function value. From 0.7.4 on, a tuple that mixes ``.as_func`` with a plain
``CompileResultWAP`` therefore fails below numba 0.68.0, as :ref:`the tuple rule <derive_mixed_function_values>`
describes.

A downgrade to numbox 0.7.2 or earlier, with numba's version unchanged, can leave a cache that numba cannot read. A call
of a cached function that received a ``cres`` function, or a node built from one, then raises
``ModuleNotFoundError: No module named 'numbox.utils.derive_wap'`` while numba loads its cache. The message means that
the cache index names a numbox type that the older numbox lacks. numba reads the types in the index before it checks the
source file, so saving the file again does not clear the error. Delete the function's cache files. An edit that moves
the function's decorator to another line also works, because numba includes that line number in the cache file names.
If numba's version changes as well, numba skips the old index without reading it, and the function recompiles. numbox
0.7.2 requires numba below 0.67, so downgrading numbox from an installation with numba 0.67 or later also changes
numba's version.

Graph manager
*************

While not a requirement, it is recommended that the `Work` instance's `name` attribute matches the name
of the variable to which that instance is assigned.
Moreover, no built-in assertion of the uniqueness of `Work` names is provided; only the builder's
registries refuse a repeated name, as :ref:`the builder overview <builder>` describes.
The users are free to implement their own graph managers that register the `Work`
nodes and assert additional requirements on the names as needed. The core numbox library
takes no position on whether such an overhead is universally beneficial (and
worth the performance tradeoff).

One option for building a graph manager is a constructor such as the following:

.. code-block:: python

    from numba import njit
    from numba.core.errors import NumbaError
    from numbox.core.configurations import jit_options
    from numbox.core.work.lowlevel_work_utils import ll_make_work
    from numbox.core.work.node import NodeType
    from numbox.utils.lowlevel import _cast
    from work_registry import _get_global, registry_type

    @njit(**jit_options)
    def make_registered_work(name, data, sources=(), derive=None):
        """ Optional graph manager. Consider using `make_work`
        where performance is more critical and name clashes are
        unlikely and/or inconsequential. """
        registry_ = _get_global(registry_type, "_work_registry")
        if name in registry_:
            raise NumbaError(f"{name} is already registered")
        work_ = ll_make_work(name, data, sources, derive)
        registry_[name] = _cast(work_, NodeType)
        return work_

The manager calls ``ll_make_work`` rather than ``make_work``, as recommended above for large graphs: numba emits the
code of the intrinsic directly into the jitted function that calls it. Saved as ``work_registry.py``, the registry
module that the example imports can be defined as follows:

.. literalinclude:: ./_static/work_registry.py
   :language: python
   :linenos:

Implementation details
**********************

Internally, `Work` accommodates individual access to its `sources`
(the other `Work` nodes on the DAG that the given `Work` node depends on)
via a 'Python-native compiler' technique: before numba compiles an overload, numbox generates its Python source
for the given number of sources and runs it with Python's `compile` and `exec` functions.
This technique is fully compatible with caching of
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
sources happens just once. numbox sets the flag only after the `derive` returns, so a
`derive` whose exception propagates leaves it unset, and the node can be calculated
again, as described under `Exception handling`_.

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
a `cres`-compiled derive function as an argument, and `cres` requires
an explicitly provided signature of the `derive` function.
The return type of the `derive` function should match the type of the `data` attribute
of the corresponding `Work` instance, and its argument types
should match the `data` types of the `Work` instance's sources.

Utilities defined in this module make it easier to ensure these
requirements are met with a minimal amount of coding. Below, `make_init_data()` returns `0.0`, the initial
`data` of the node; given a `shape`, it returns an array of that shape filled with `val`::

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
