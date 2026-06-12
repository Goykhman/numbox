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
will not result in an exception, accommodating for the case of a function
that computes and returns a value independent of any input parameters.
It is also possible to specify `inputs`
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

.. [#f1] It is straightforward to adapt the variables specifications given here in pure Python to build a fully-JIT'ed graph of :class:`numbox.core.work.work.Work` nodes, by using the :class:`numbox.core.work.builder.Derived`. See :ref:`builder`. As another route to fully-JIT'ed evaluation, :func:`numbox.core.variable.compile_kernel.compile_kernel` fuses the graph into a single ``@njit`` kernel (see below).

.. automodule:: numbox.core.variable.variable
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.variable.compile_kernel
-----------------------------------

Overview
********

Alongside :meth:`numbox.core.variable.variable.Graph.compile` (which produces a
:class:`numbox.core.variable.variable.CompiledGraph` evaluated node-by-node in pure Python),
:func:`numbox.core.variable.compile_kernel.compile_kernel` compiles a `Graph` into
fused ``@njit`` kernel code for a requested set of variables. It does not replace
`core.work` or `CompiledGraph`; it is an additional, JIT'ed evaluation path.

When every `formula` is njit-able the graph fuses into *one* ``@njit`` kernel that takes
the required external inputs as positional arguments and returns the requested variables
as a tuple, with every interior graph node lowered to an SSA temporary inside the single
compiled function. No per-node type information needs to be supplied: numba infers every
interior type from the runtime argument types. Plain-Python formulas are auto-wrapped with
``njit()``. When some formulas are not njit-able for the actual argument types, the first
call detects them and the graph is split into ``@njit`` segments orchestrated from Python
(see :ref:`compile_kernel_non_jittable` below).

The call returns a :class:`numbox.core.variable.compile_kernel.CompiledKernel`. It exposes
``.kernel`` (the hot-path callable — positional in, tuple out: the bare numba dispatcher
once the graph resolves fully fused, the Python master when the graph is segmented around
non-jittable nodes) and a dict-in / dict-out ``.execute`` convenience that mirrors
:meth:`numbox.core.variable.variable.CompiledGraph.execute`. The qualified names of the
kernel's positional inputs and tuple outputs are available as ``.params`` and ``.outputs``,
the generated kernel text as ``.source``, and the per-variable temporary identifiers as
``.identifiers``.

This fused path has two deliberate limitations: it does not honor the
`cacheable` memoization of individual nodes, and it offers no incremental recompute of
only the affected nodes. Use :class:`numbox.core.variable.variable.CompiledGraph` (or the
`core.work` graph) when either of those is needed.

**Caching.** The fused kernel is cached on disk, content-addressed by a
fingerprint of the generated kernel source, every formula's behavioral
state (bytecode, constants, default values, closure-cell values, referenced
module-level globals including helper functions, defining module), and the
effective jit flags. Changing any of these recompiles instead of reusing a
stale binary; cosmetic edits that do not change behavior (comments, local
renames) do not. Formulas whose state cannot be fingerprinted -- a
``cres``-compiled callable, or a value with no canonical form -- make that
one kernel uncacheable: always recompiled per process, never wrong. The
``cache`` keyword is tri-state: ``None`` (the default) defers to
``jit_options["cache"]``, then the ``NUMBOX_JIT_OPTIONS`` environment
default, then ``True``; an explicit ``True``/``False`` wins. Two costs are
worth knowing: a formula that references or closes over a **large array**
pays a per-compile ``sha256`` over that array's bytes (proportional to its
size) on every ``compile_kernel`` call; and numba itself declines to
disk-cache a kernel that calls a ``@cfunc`` formula or references a large
global array -- the kernel still computes correctly, it is simply
recompiled in each process regardless of the content-addressed anchor. A
``@vectorize`` (DUFunc) formula, by contrast, caches cleanly.

**Practical limits.** Graph traversal is recursive: dependency chains
deeper than roughly ``sys.getrecursionlimit()`` raise a ``RecursionError``
naming the remedy (raise the limit before compiling). Cold compilation of
the fused kernel costs on the order of 20 ms and ~1 MiB of memory per
formula node (numba 0.65, CPython 3.12); graphs beyond a few thousand
nodes compile increasingly slowly and are better split or evaluated via
:class:`numbox.core.variable.variable.CompiledGraph`.

A graph can be compiled to a fused kernel as follows:

.. code-block:: python

    from numba import njit
    from numbox.core.variable.variable import Graph
    from numbox.core.variable.compile_kernel import compile_kernel

    graph = Graph(
        variables_lists={"variables": [
            {"name": "x", "inputs": {"y": "basket"}, "formula": njit(lambda y: 2 * y)},
            {"name": "u", "inputs": {"x": "variables"}, "formula": njit(lambda x: x - 74)},
        ]},
        external_source_names=["basket"],
    )

    ck = compile_kernel(graph, ["variables.u"])
    assert ck.execute({"basket": {"y": 100}}) == {"variables.u": 126}
    assert ck.kernel(100) == (126,)

Here the dict-in / dict-out ``ck.execute`` looks up the required external value
``basket.y`` and returns the requested ``variables.u``, while ``ck.kernel``
is called positionally with the external input and returns the output tuple directly.

.. _compile_kernel_non_jittable:

Graphs with non-jittable nodes
******************************

``compile_kernel`` detects non-jittable formulas automatically at the first
call: it first tries to compile the fully fused kernel for the actual
argument types; if that fails, it probes each node against the real
intermediate values, runs the offenders in plain Python, and fuses the
jittable remainder into as few ``@njit`` segments as a greedy linearization
allows. A Python master then threads values between segments and Python
nodes. Compile-time failures demote a node; runtime errors always propagate.

.. code-block:: python

    import json

    from numbox.core.variable.compile_kernel import compile_kernel
    from numbox.core.variable.variable import Graph

    def n3(v):
        json.dumps({"k": 1})    # no nopython lowering for the json module
        return v * 3.0

    graph = Graph(
        variables_lists={"calc": [
            {"name": "n1", "inputs": {"x": "ext"}, "formula": lambda x: x + 1.0},
            {"name": "n2", "inputs": {"n1": "calc"}, "formula": lambda n1: n1 * 2.0},
            {"name": "n3", "inputs": {"n2": "calc"}, "formula": n3},
            {"name": "n4", "inputs": {"n3": "calc"}, "formula": lambda n3: n3 - 4.0},
            {"name": "n5", "inputs": {"n4": "calc"}, "formula": lambda n4: n4 / 2.0},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(graph, "calc.n5")
    ck.kernel(7.0)              # first call: probes, partitions, still correct
    print(str(ck.partition))    # 2 jit segments around the python n3, with reasons

``ck.partition`` is ``None`` until the first call resolves the mode; a fully
fused graph reports a single jit segment. Each jit segment is cached
content-addressed on disk exactly like a v1 kernel; the learned partition
itself is per-process. If a later call's types break a segment, the partition
is re-learned for those values and replaces the previous plan — workloads
alternating between type families whose partitions differ re-pay discovery on
each alternation.

.. automodule:: numbox.core.variable.compile_kernel
   :members:
   :show-inheritance:
   :undoc-members:
