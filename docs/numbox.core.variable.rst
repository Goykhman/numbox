numbox.core.variable
====================

Overview
++++++++

The module :mod:`numbox.core.variable.variable` is a framework for a Directed Acyclic Graph (DAG) of calculations,
in pure Python. It contains no JIT-compiled code and imports nothing from numba, but the functions that calculate a
graph's variables may be JIT-compiled, as the next section shows.

The module :mod:`numbox.core.variable.compile_kernel` evaluates such a graph in compiled code instead:
:func:`~numbox.core.variable.compile_kernel.compile_kernel` generates one ``@njit`` function that calls these
functions. numba compiles that function and caches it on disk, as described in `Compiling a graph into one kernel`_.

Building and evaluating a graph
+++++++++++++++++++++++++++++++

The examples of this section continue one another, as one Python session. The following defines a graph::

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

Here the variable `y` belongs to the external source `basket`, while `x` and `a` are calculated in the namespace
`variables1`, and `u` in the namespace `variables2`. No specification defines `y`: the input ``"y": "basket"`` of
`x` names it, and the graph creates `y` in `basket` when it first looks up that name.

The dictionaries `x`, `a` and `u` are variable specifications. A specification does not name the namespace it
belongs to; the `variables_lists` argument of `Graph` puts each specification in a namespace when the graph is
created.

A qualified name, the namespace name and the variable name joined by a dot, denotes a variable fully and
unambiguously: `basket.y` for the external variable, and `variables1.x`, `variables1.a` and `variables2.u` for the
calculated ones.

The key `formula` of a specification gives the function that calculates the variable. The key `inputs` maps the name
of each input variable, a dependency of this graph node, to the namespace in which the graph looks it up: in
``"inputs": {"x": "variables1"}``, the input `x` is looked up in `variables1`. The i-th parameter of the formula
receives the value of the i-th entry of `inputs`, and the names need not match: `derive_x` receives the value of `y`
as `y_`.

A formula may be an ordinary Python function, or one already compiled by numba: an ``@njit`` dispatcher, a
``@vectorize`` function or a ``@cfunc``. A first-class function value, such as one made by numbox's
:func:`~numbox.utils.highlevel.cres`, serves as well.

A variable without inputs is an end node, a leaf of the graph. An external variable such as `y` is an end node with
neither inputs nor formula, since no specification defines it. A calculated variable is an end node when its formula
takes no inputs: a formula without inputs raises no exception, so that a function can return a value that depends on
no input. A specification may also give inputs but no formula. It then places the node in the graph, and the developer
supplies the formula later, at run time.

The `Graph` takes the names of the external sources of data in its `external_source_names` argument. When
:meth:`Graph.compile <numbox.core.variable.variable.Graph.compile>` compiles the graph into a
:class:`~numbox.core.variable.variable.CompiledGraph` for the variables to calculate, it determines which variables
each external source, such as `basket`, must provide::

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

From the specifications, the graph and its namespaces build the objects listed in the first three rows below. The
last two hold the values of a calculation:

.. list-table::
   :header-rows: 1
   :widths: 18 46 36

   * - Class
     - What an instance holds
     - Who creates it
   * - :class:`~numbox.core.variable.variable.Variable`
     - One variable: its name, its namespace, its inputs and its formula.
     - Its namespace: a ``Variables`` namespace from the specifications, an ``External`` namespace the first time a
       variable is looked up in it.
   * - :class:`~numbox.core.variable.variable.Variables`
     - The calculated variables of one namespace, such as ``variables1``.
     - The ``Graph``, one for each key of ``variables_lists``.
   * - :class:`~numbox.core.variable.variable.External`
     - The variables of one external source, such as ``basket``. They have no inputs, so they are end nodes.
     - The ``Graph``, one for each name in ``external_source_names``.
   * - :class:`~numbox.core.variable.variable.Values`
     - The values of one calculation, as a mapping from each ``Variable`` instance to its ``Value``.
     - The user, before the calculation.
   * - :class:`~numbox.core.variable.variable.Value`
     - The value of one variable in one ``Values`` instance.
     - That ``Values`` instance, when the variable is first looked up in it.

Semantically, each `Variable` instance is defined by its scoped name, the pair of its namespace name and its own
name, which the qualified name writes with a dot; "scope" means namespace here.

The `registry` of the graph maps each namespace name to its `Variables` or `External` instance, and through those to
every `Variable` instance::

    from numbox.core.variable.variable import Variables, Variable

    registry = graph.registry

    # Get the namespaces...
    variables1 = registry["variables1"]
    variables2 = registry["variables2"]

    # ... and the variables defined in these namespaces
    assert list(variables1.keys()) == ["x", "a"]
    assert list(variables2.keys()) == ["u"]

    assert isinstance(variables1, Variables)
    assert isinstance(variables1["x"], Variable)

    basket_ = registry["basket"]
    ... # same `basket` as above
    assert basket_["y"] is basket["y"]

Users are not expected to create `Variable` or `Variables` instances themselves, although they may. They provide
variable specifications instead, and code that needs a `Variable` instance is best written to take it from the
`registry`, as above.

A calculation needs a `Values` instance to hold its values. The instance starts empty and creates the `Value` of a
variable when the variable is first looked up in it. A new `Value` holds `_null`, an instance of a private class that
marks a value not yet calculated, since a calculated value can itself be ``None``.
:meth:`CompiledGraph.execute <numbox.core.variable.variable.CompiledGraph.execute>` then populates the instance: it
writes the external values first, then the value of each calculated variable. `execute` takes the values of the
required external variables, which ``compiled.required_external_variables`` lists, as its `external_values`
argument, keyed by external source and variable name::

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

When some of the graph's nodes change, `recompute` evaluates again only the nodes that the change affects::

    compiled.recompute({"basket": {"y": 1}}, values)
    assert values.get(basket["y"]).value == 1
    assert values.get(x_var).value == 2
    assert values.get(a_var).value == -72
    assert values.get(u_var).value == -144

Two other parts of numbox evaluate such a graph in compiled code. It is straightforward to adapt the same variable
specifications to build a fully JIT-compiled graph of :class:`~numbox.core.work.work.Work` nodes with
:class:`~numbox.core.work.builder.Derived`, as described in :ref:`numbox.core.work.builder <builder>`.
:func:`~numbox.core.variable.compile_kernel.compile_kernel` instead compiles the formulas into one ``@njit`` function,
as described in `Compiling a graph into one kernel`_.

Compiling a graph into one kernel
+++++++++++++++++++++++++++++++++

:meth:`CompiledGraph.execute <numbox.core.variable.variable.CompiledGraph.execute>` calls the formulas from Python, one
node at a time. :func:`~numbox.core.variable.compile_kernel.compile_kernel` writes a Python function instead, the
kernel, that calls the formulas of the requested variables and of their dependencies, each after its inputs. numba
compiles the kernel with ``@njit``, and a call of the kernel then runs the formulas in compiled code, with no Python
between them.

The speedup is largest when the formulas are small. One call of ``execute`` on a chain of 50 small ``@njit`` formulas
took a few hundred times as long as one call of its kernel.

numba caches the compiled kernel on disk, so a later run of the program loads it instead of compiling it again. numba
finds a cached function by its name and by the file that holds its source. numbox derives the kernel's name, and with
it the name of that file, from a digest of the formulas and of what they read. An edit to a formula, or to a
value that a formula reads, therefore gives the kernel a new name. numba finds no cache entry under the new name and
compiles the kernel again. What the digest covers, and the few edits it misses, are listed in `When the kernel compiles
again`_.

A formula may be an ordinary Python function, which numbox decorates with ``@njit`` itself, or one already compiled
by numba: an ``@njit`` dispatcher, a `@vectorize <https://numba.readthedocs.io/en/stable/user/vectorize.html>`_
function or a `@cfunc <https://numba.readthedocs.io/en/stable/user/cfunc.html>`_. A function made by numbox's
:func:`~numbox.utils.highlevel.cres`, which compiles a function for one signature in advance, works as well.

The graph needs no types: numba infers the type of every variable from the arguments of the kernel's first call.

A formula that numba cannot compile runs in Python, unless it is a ``@cfunc``, ``@vectorize`` or ``cres`` function, as
described in `If a formula cannot be compiled`_.

Using compile_kernel
--------------------

``compile_kernel`` takes the graph and the variables to compute: one qualified name, such as ``"calc.total"``, or a
list of them. It returns a :class:`~numbox.core.variable.compile_kernel.CompiledKernel`, an object that holds the kernel
and describes it. The following example is saved as ``pricing.py``:

.. code-block:: python

    from numba import njit
    from numbox.core.variable.compile_kernel import compile_kernel
    from numbox.core.variable.variable import Graph


    @njit
    def cost(price, units):
        return price * units


    def with_tax(amount, rate):
        return amount * (1.0 + rate)


    graph = Graph(
        variables_lists={"calc": [
            {"name": "cost", "inputs": {"price": "basket", "units": "basket"}, "formula": cost},
            {"name": "total", "inputs": {"cost": "calc", "rate": "basket"}, "formula": with_tax},
        ]},
        external_source_names=["basket"],
    )
    ck = compile_kernel(graph, ["calc.total", "calc.cost"])

    print(ck.params)
    print(ck.kernel(20.0, 0.25, 3.0))
    print(ck.execute({"basket": {"price": 20.0, "rate": 0.25, "units": 3.0}}))
    print(ck.recompute({"basket": {"rate": 0.5}}))

``python pricing.py`` prints:

.. code-block:: text

    ['basket.price', 'basket.rate', 'basket.units']
    (75.0, 60.0)
    {'calc.total': 75.0, 'calc.cost': 60.0}
    (90.0, 60.0)

The kernel, ``ck.kernel``, takes the external values as positional arguments, in the order of ``ck.params``: the
qualified names sorted, so that ``rate`` comes before ``units`` here. It returns the requested variables as a tuple in
the order of ``ck.outputs``, which keeps the order of the list passed to ``compile_kernel``. When
``compile_kernel`` receives one name instead of a list, the tuple holds one value. ``ck.execute`` is the
dictionary-based counterpart: it takes the external values as ``{source: {name: value}}``, as ``CompiledGraph.execute``
does, and returns a dictionary keyed by qualified name. The last line of the example calls ``ck.recompute``, which
recomputes only what a change affects, as described in `Recomputing after a change`_.

In the example, ``cost`` is decorated with ``@njit``, not with ``@njit(cache=True)``. The kernel's cache holds the
compiled code of every formula, and a formula compiled with ``cache=True`` disables the kernel's cache, for the reason
given in `If numbox warns of a formula with its own cache`_.

The ``jit_options`` argument passes numba's jit options, the keyword arguments of ``@njit`` such as `fastmath
<https://numba.readthedocs.io/en/stable/user/performance-tips.html#fastmath>`_, to the kernel. numbox decorates each
plain Python formula, such as ``with_tax``, with the same options.

Whether numba caches the kernel depends on the first of these settings that is present:

1. the ``cache`` argument of ``compile_kernel``, ``True`` or ``False``; the default, ``None``, defers the decision to
   the settings below;
2. the ``cache`` option of ``jit_options``;
3. the ``cache`` option of numbox's default options. numbox reads those options from the ``NUMBOX_JIT_OPTIONS``
   environment variable for every function it compiles, and it sets their ``cache`` to ``False``, for all of those
   functions, when numba can write no cache for numbox's own modules (:doc:`numbox.core.configurations`);
4. if none of these settings is present, caching is enabled.

Even with ``cache=True``, the kernel compiles without a cache in the cases listed in `When the kernel is not cached`_.

Unless the types are declared, as described in `Declaring the types`_, numba compiles the kernel at its first call, for
the types of that call's arguments. A later call with other types, such as integers, makes numba compile and cache
another version, as for any numba function. Before the first call, ``ck.kernel`` is a numbox method that compiles the
kernel and then calls it. Once numba has compiled the kernel as one function, ``ck.kernel`` is numba's dispatcher
itself. A name bound to ``ck.kernel`` before the first call still calls the kernel, but every call through it passes
through numbox's method first, so a loop should bind the name after the first call.

``compile_kernel`` checks the graph's structure before anything compiles, and its error names the variable at fault.
It refuses:

- a requested name that is not qualified, that lies in an unknown namespace, that is missing from its ``Variables``
  namespace, or that is requested twice;
- a formula that is not callable, or that cannot take its inputs;
- a cycle in the graph;
- a variable without a formula, whether or not it has inputs, although ``CompiledGraph`` accepts one, as described in
  `Building and evaluating a graph`_.

An external name is never refused, because the namespace of an external source creates any variable that is looked
up in it (see `A misspelled external name`_).

When the kernel compiles again
------------------------------

numba checks a cached function against the file that defines it. An edit to that file makes numba compile the function
again. An edit to a function that it calls from another file does not, and neither does a new value of a global that
it reads, as numba's `caching limitations
<https://numba.readthedocs.io/en/stable/developer/caching.html#caching-limitations>`_ note. When a formula changes, the
kernel's own file stays as it was, so numba's file check alone would load a stale kernel.

numba loads a cached kernel only when two things match. The first is numba's key, which must match as for any cached
function: numba's own parts, and the part that numbox adds for every function it compiles, made of numbox's version, the
jit options and the ``NUMBA_*`` settings (:ref:`which functions numbox keys <cache_key_which_functions>`). Like numba's
own key, it leaves out the code of the functions called and the values of the globals read (:ref:`what the key leaves
out <cache_key_leaves_out>`). The second is the kernel's name, which is numbox's digest. A change to anything the digest
holds changes the name, and with it the file that numba looks for, in which numba finds nothing cached. The jit options
and the ``NUMBA_*`` settings are in both numba's key and the digest, so a change to any of them causes a cache miss.
The digest holds:

- the graph's structure, the names of its variables, and the requested variables in their order;
- each formula's code, its constants and docstring included, its module and name, and its default arguments and
  closure variables;
- the values of the globals and module attributes that a formula reads, such as ``SCALE`` or ``settings.SCALE``, and
  the code and values of each function and dispatcher that it calls, with the exceptions below;
- the options of a formula that is a numba dispatcher or a ``@vectorize`` function;
- the jit options that the kernel compiles under, from ``jit_options`` and numbox's defaults, except ``cache``;
- the ``NUMBA_*`` settings that change the code numba emits, listed in :data:`~numbox.core.cache_key.KEYED_SETTINGS`;
- the declared types, described in `Declaring the types`_.

If neither numba's key nor the kernel's name changes, numba loads the kernel from the cache. A new comment or a renamed
local variable keeps the name. A new or edited docstring changes it, because Python keeps a function's docstring among
its constants. numbox's version is in numba's key but not in the digest, so the first run after an upgrade from one
numbox release to another compiles each kernel once more under its old name
(:ref:`after you upgrade numbox <cache_key_after_upgrade>`).

Some edits do not change the digest. After one of them, the cached kernel returns the result from before the edit:

- an edit to a function decorated with numba's ``@register_jitable`` or ``@intrinsic`` that a formula calls as a
  module attribute, as in ``helpers.step(y)``;
- an edit to the implementation of an ``@overload``, however the formula reaches the function that it overloads;
- a new value of a module attribute that the module provides only through a module-level ``__getattr__``
  (:pep:`562`).

A ``@register_jitable`` function imported by name, as in ``from helpers import step``, is in the digest, and so is an
``@njit`` dispatcher reached either way. Delete the directories that hold the kernels' files, named in `Where the
kernel's files are`_, after an edit of one of the kinds listed.

Where the kernel's files are
----------------------------

When caching is enabled, numbox writes the source of each kernel to a file whose name is the kernel's name, ``_kernel_``
followed by the digest, in a directory ``numbox-compile-kernel``. numba then writes the compiled kernel to a directory
of its own:

- When `NUMBA_CACHE_DIR <https://numba.readthedocs.io/en/stable/reference/envvars.html#envvar-NUMBA_CACHE_DIR>`_ is
  set, ``numbox-compile-kernel`` is a directory under it, and numba writes the compiled kernel to another directory
  under ``NUMBA_CACHE_DIR``, whose name starts with ``numbox-compile-kernel_``.
- Otherwise, ``numbox-compile-kernel`` is under numba's user cache directory, and numba writes the compiled kernel to
  ``numbox-compile-kernel/__pycache__``. :ref:`Where the cache lands <where_the_cache_lands>` names the user cache
  directory for each system.

numbox never deletes the files of an old kernel, so they accumulate as the formulas change. To reclaim disk space,
delete ``numbox-compile-kernel`` and, when ``NUMBA_CACHE_DIR`` is set, the directories beside it whose names start
with ``numbox-compile-kernel_``, while no Python process is using them. The next run then compiles each kernel again.

``NUMBA_DEBUG_CACHE=1`` shows whether a kernel came from the cache. With it set, numba prints a line for each cache
file that it saves or loads, and each file name starts with the kernel's name. ``data saved`` marks a kernel that numba
compiled and cached, and ``data loaded`` one that it loaded from the cache. numba prints no line for a kernel that is
not cached. Among its results, a second run of ``pricing.py`` prints lines like these:

.. code-block:: text

    [cache] index loaded from '.../_kernel_6670ac166cbfc335._kernel_6670ac166cbfc335-1.py312.nbi'
    [cache] data loaded from '.../_kernel_6670ac166cbfc335._kernel_6670ac166cbfc335-1.py312.1.nbc'
    [cache] index loaded from '.../_kernel_4af2b237ca0f7be3._kernel_4af2b237ca0f7be3-1.py312.nbi'
    [cache] data loaded from '.../_kernel_4af2b237ca0f7be3._kernel_4af2b237ca0f7be3-1.py312.1.nbc'

The first two lines show numba loading the kernel, and the last two show it loading the code that ``ck.recompute``
compiled for its change.

When the kernel is not cached
-----------------------------

In the cases below, numba keeps no compiled kernel on disk, so every process compiles the kernel again. The results
are still correct.

If numbox warns of a formula with its own cache
***********************************************

numbox warns, here for a formula named ``double``:

.. code-block:: text

    UserWarning: compile_kernel: formula(s) double are, or reference, a @njit(cache=True) dispatcher whose cache key
    is blind to referenced global values; compiling this kernel without an on-disk cache so a stale inner binary
    cannot be linked into a cached artifact. Pass cache=False to those dispatchers (or supply plain-Python formulas)
    to restore caching.

The warning means that a formula is, or calls, a dispatcher made with ``@njit(cache=True)``, and that numbox has
disabled the kernel's cache. Such a dispatcher has a cache entry of its own, and numba's key for that entry leaves out
the globals that the dispatcher reads and the jit options that it compiles under. The entry can therefore be stale, and
a kernel cached with it would contain the stale code. The remedy is to drop ``cache=True`` from the formula's decorator,
or to pass as the formula the plain function that the dispatcher keeps as its `py_func
<https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#Dispatcher.py_func>`_, here ``double.py_func``.
The kernel's cache then holds the formula's code.

If numba warns of dynamic globals
*********************************

numba warns, naming the kernel:

.. code-block:: text

    NumbaWarning: Cannot cache compiled function "_kernel_a217a1320e74c590" as it uses dynamic globals (such as
    ctypes pointers and large global arrays)

The warning means that a formula reads a global array that numba refers to by its address, and code that contains an
address cannot be cached. numba copies a small array into the compiled code as a constant; it refers by address to an
array larger than 1 MB, or to one that is not contiguous (`global and closure variables
<https://numba.readthedocs.io/en/stable/reference/pysemantics.html#global-and-closure-variables>`_). The remedy is to
pass the array to the kernel as an external value, because an array argument does not prevent caching.

If numbox warns of jit flags with no canonical fingerprint
**********************************************************

numbox warns, listing every jit option that the kernel compiles under:

.. code-block:: text

    UserWarning: compile_kernel: jit flags ['fastmath', 'pipeline_class'] have no canonical fingerprint; compiling this
    kernel without an on-disk cache.

The warning means that one of the jit options, which the warning calls jit flags, has no form that numbox can put into
the digest, here the ``pipeline_class``. numbox has therefore disabled the kernel's cache; the list names every
option, not only the one at fault. The remedy is to leave such an option out, or to accept a kernel that compiles in
every process.

If numbox warns that numba cannot cache the kernel's file
*********************************************************

numbox warns, here for a read-only ``NUMBA_CACHE_DIR``:

.. code-block:: text

    RuntimeWarning: numba cannot cache _kernel_a7184ddece8d3cca.py here ([Errno 13] Permission denied:
    '.../numbox-compile-kernel'); it compiles without a cache. Set NUMBA_CACHE_DIR to a writable directory, or
    compile without a cache to silence this warning: "cache" off in the jit options this code was given, or in its
    cache argument where it takes one, or NUMBOX_JIT_OPTIONS='{"cache": false}' where the options are the package's

The warning means that numbox could not write the kernel's file, or that numba has no cache location for it, so the
kernel compiles without a cache. The text in parentheses is the error that numbox caught, here the
``PermissionError`` raised when numbox tried to create the directory. The warning names the remedy itself: a writable
``NUMBA_CACHE_DIR``, or caching disabled, which also silences the warning.

If the kernel compiles in every process without a warning
*********************************************************

numbox disables the kernel's cache without a warning in three cases, and ``NUMBA_DEBUG_CACHE=1`` then prints no line
for the kernel:

- A formula is a ``@cfunc``. The kernel calls such a function at its address in the process that compiled it, and
  numba cannot cache that.
- A formula is a function made by ``cres``, passed as the formula itself. A formula that calls one, such as
  ``lambda x: double(x)``, allows caching.
- A formula reads a value that numbox cannot hash. numbox hashes numbers, strings, bytes, ``None``, NumPy scalars and
  arrays other than object arrays, tuples, lists, sets and dicts of these, modules, Python functions, numba
  dispatchers and ``@vectorize`` functions.

One rule covers a function or a type that a module provides: calling it through its module allows caching, and
calling it by a bare name imported from the module does not.

==============================  ================================================================
Allows caching                  Prevents caching
==============================  ================================================================
``math.sqrt(x)``                ``sqrt(x)``, after ``from math import sqrt``
``np.exp(x)``                   ``exp(x)``, after ``from numpy import exp``
``numba.float64(x)``            ``float64(x)``, after ``from numba import float64``
``np.float64(x)``
==============================  ================================================================

An ``@intrinsic`` imported by name leaves the kernel uncached as well. Called through its module, it allows caching,
but an edit to it then does not change the digest, as listed in `When the kernel compiles again`_.

.. _compile_kernel_recompute:

Recomputing after a change
--------------------------

``ck.recompute`` does what :meth:`CompiledGraph.recompute <numbox.core.variable.variable.CompiledGraph.recompute>`
does. It takes the changed values as ``{source: {name: value}}``, recomputes only the variables that depend on them,
and returns the requested variables as a tuple in the order of ``ck.outputs``. In ``pricing.py``, the new ``rate``
makes numbox recompute ``calc.total`` and reuse ``calc.cost``.

``ck`` holds the values that ``recompute`` reuses in a dictionary, the store. numbox fills the store with the
values of the kernel's first call, and each ``recompute`` writes its changes into it. A fused kernel, one that numba
compiled as one function, returns only the requested variables, so numbox fills its store at the first
``recompute``. numbox then computes every variable once more from the first call's arguments, and numba compiles each
formula separately for this computation. numba does not cache these compilations, so every new process compiles the
formulas once. The store of a kernel that runs in segments, described in `If a formula cannot be compiled`_, fills
during the first call. Before any call, ``recompute`` raises
``RuntimeError: CompiledKernel.recompute requires a prior full call``.

A call of ``ck.kernel`` or ``ck.execute`` does not update the store. A ``recompute`` after ``ck.kernel`` with new
arguments therefore still starts from the first call's values and the changes passed to ``recompute`` since. To start
over from new values, pass every external value to ``recompute``, or build a new kernel with ``compile_kernel``, which
loads the compiled code from the cache.

A changed name may also be a computed variable: ``ck.recompute({"calc": {"cost": 50.0}})`` sets ``calc.cost`` to
``50.0`` and recomputes ``calc.total`` from it. If the same call also changes an input of that variable, numbox
recomputes the variable from its formula and discards the given value, as ``CompiledGraph.recompute`` does. A name
outside the requested part of the graph changes nothing, and numbox warns that it ``is not in the calculation path``.

For each new set of changed names, numbox builds a recompute plan, the steps that recompute the affected variables,
and numba compiles and caches the plan's code as it does the kernel's. A later ``recompute`` with the same names
reuses its plan. numbox keeps up to 64 recompute plans for each kernel. The first time a ``recompute`` overrides a
computed variable that no earlier ``recompute`` overrode, numbox discards all of the kernel's recompute plans, because
the override changes which intermediate values the plans must keep in the store.

A ``recompute`` raises numba's ``TypingError`` when numba cannot compile its plan for a changed value, such as a string
where a formula adds numbers. The value that failed stays in the store, so every later ``recompute`` that reads it
fails as well, until a ``recompute`` passes a new value for it. A changed value of another type that numba can
compile, such as an ``int`` where the first call passed a ``float``, causes no failure: numba compiles the plan again
for that type.

When a ``recompute`` reuses a plan that an earlier ``recompute`` built, and that plan fails on the new values, numbox
retries once, and the retry reverts the changes of earlier ``recompute`` calls. numbox discards every recompute plan,
refills the store from the first call's values, applies this call's changes again and builds a new plan; it keeps the
refilled store even when the retry succeeds. Consider a graph whose variable ``s`` adds the external values ``a`` and
``b``:

1. The first call passes ``a = 1.0`` and ``b = 2.0``, and ``s`` is ``3.0``.
2. A ``recompute`` sets ``a`` to ``"x"`` and ``b`` to ``"y"``. numba compiles a plan for two strings, and ``s`` is
   ``"xy"``.
3. A ``recompute`` sets ``b`` to ``3.0``. numbox reuses the plan of step 2 and runs it with the string ``"x"`` from the
   store and the float ``3.0``, a pair that numba cannot add. The retry refills ``a`` with ``1.0`` from the first call,
   and ``s`` is ``4.0``: the change of ``a`` in step 2 is lost.

A kernel built from declared types never retries; it checks each changed value against its declared type first, as
described in `Declaring the types`_.

.. _compile_kernel_non_jittable:

If a formula cannot be compiled
-------------------------------

A graph may hold a formula that numba cannot compile, such as one that uses Python's ``fractions`` module. The kernel
still runs: numbox runs that formula in Python and has numba compile the others into ``@njit`` segments, each a
compiled function for a run of consecutive formulas. ``ck.partition`` reports the split:

.. code-block:: python

    from fractions import Fraction

    from numbox.core.variable.compile_kernel import compile_kernel
    from numbox.core.variable.variable import Graph


    def to_eighths(v):
        return float(Fraction(v).limit_denominator(8))


    graph = Graph(
        variables_lists={"calc": [
            {"name": "n1", "inputs": {"x": "ext"}, "formula": lambda x: x * 2.0},
            {"name": "n2", "inputs": {"n1": "calc"}, "formula": lambda n1: n1 + 0.1},
            {"name": "n3", "inputs": {"n2": "calc"}, "formula": to_eighths},
            {"name": "n4", "inputs": {"n3": "calc"}, "formula": lambda n3: n3 * 10.0},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(graph, "calc.n4")
    print(ck.kernel(1.0))
    print(ck.partition)

It prints:

.. code-block:: text

    (21.25,)
    compile_kernel partition: mode=segmented, 3 segment(s)
      [0] jit: nodes=['calc.n1', 'calc.n2'] inputs=['ext.x'] outputs=['calc.n2']
      [1] python: nodes=['calc.n3'] inputs=['calc.n2'] outputs=['calc.n3']
          calc.n3: TypingError: Untyped global name 'Fraction': Cannot determine Numba type of <class 'abc.ABCMeta'>
      [2] jit: nodes=['calc.n4'] inputs=['calc.n3'] outputs=['calc.n4']

numbox finds such formulas in a search at the first call. When numba cannot compile the whole kernel for the call's
argument types, numbox runs the graph once with the call's values and has numba compile each formula separately. A
formula runs in Python when numba fails to compile it, or when numba cannot type one of its input values. If a formula
compiles but raises an exception while the search runs it, the formula does not run in Python, and the exception
propagates. numbox then groups the other formulas into as few segments as the graph's dependencies allow, numba compiles
each segment, and a Python function calls the segments and the Python formulas in order.

``ck.partition`` is ``None`` before the first call, unless the types are declared as described in `Declaring the
types`_. After the first call, the report gives the mode: ``fused`` when numba compiled the whole kernel as one
function, and ``segmented`` otherwise. It then lists the segments in order, with numba's reason under each formula that
runs in Python. A fused kernel reports one ``jit`` segment.

numbox never runs a ``@cfunc``, ``@vectorize`` or ``cres`` formula in Python, because such a function has no Python
version to run. If numba cannot compile one for the argument types, numba's error propagates: ``TypingError``, or
``ValueError: mismatch of argument types`` for a ``@cfunc`` or ``cres`` function. If numba cannot type an input value,
numbox raises ``TypeError``, with a message that the formula ``has no Python fallback`` and that the input value ``is
not numba-typeable``.

numba caches each segment on disk like a fused kernel, but numbox keeps the partition, the split of the graph into
segments and Python formulas, only in memory. Every new process searches again at its first call, compiling each
formula separately, and then loads the segments from the cache. The first call above took about 240 ms with an empty
cache and 150 ms with the segments in the cache. If a later call's types make a segment of a segmented kernel fail to
compile, numbox searches again with that call's values and replaces the partition; a fused kernel never searches
again, as the last paragraph of this section says. The new search also replaces the store of ``recompute`` with the
values of that call. With declared types, ``compile_kernel`` determines the partition instead, without a search, as
described in `Declaring the types`_.

Under ``fastmath``, which lets LLVM reorder floating-point arithmetic within one compiled function, a value computed
formula by formula can differ from the same value computed by a function that holds several formulas, and by more than
the last digit. The search, and the filling of a fused kernel's store at its first ``recompute``, compute formula by
formula; a segment and a fused kernel compute in one function. Consider ``n1 = x + 1e16``, ``n2 = n1 - 1e16`` and
``n3 = n2 * k``, with ``n3`` requested, ``x = 1.0`` and ``k = 2.0``. Formula by formula, ``n1`` rounds to ``1e16``, so
``n2`` is ``0.0`` and ``n3`` is ``0.0``. In one function, ``n2`` was ``1.0`` and ``n3`` was ``2.0``. A fused
kernel therefore returned ``2.0``, while its ``recompute`` with the same ``k`` returned ``0.0``. With a Python formula
after ``n3``, which makes the kernel segmented, the first call returned ``0.0`` and later calls returned ``2.0``.

Once the first call has compiled the whole kernel, numbox does not fall back to segments: a later call with types that
numba cannot compile raises numba's ``TypingError``, which numba's `troubleshooting guide
<https://numba.readthedocs.io/en/stable/user/troubleshoot.html#my-code-doesn-t-compile>`_ explains.

Declaring the types
-------------------

Declaring the variables' types moves both the type errors and the search of the first call into ``compile_kernel``.
With declarations, ``compile_kernel`` knows before any call which formulas run in Python, has numba compile the kernel
or its segments for the declared types, and raises when a formula does not return its declared type.

A variable declares its type in the ``params`` entry of its specification, with
:class:`~numbox.core.variable.variable.Params`: ``Params(type=float64)`` declares a formula that numba compiles, and
``Params(jitable=False, type=float64)`` one that runs in Python. An external variable takes its declaration from
:meth:`~numbox.core.variable.variable.External.declare`:

.. code-block:: python

    from numba import float64
    from numbox.core.variable.compile_kernel import compile_kernel
    from numbox.core.variable.variable import Graph, Params

    graph = Graph(
        variables_lists={"calc": [
            {"name": "half", "inputs": {"x": "basket"}, "formula": lambda x: x / 2, "params": Params(type=float64)},
        ]},
        external_source_names=["basket"],
    )
    graph.registry["basket"].declare("x", Params(type=float64))
    ck = compile_kernel(graph, "calc.half")
    print(ck.partition.mode)    # fused, before any call

Declare every computed variable in the requested part of the graph, with its type, and every external variable that
a formula reads. ``compile_kernel`` then determines the partition and sets ``ck.partition`` before any call. If no
formula is declared ``jitable=False``, it has numba compile the whole kernel for the declared types, as in the example;
otherwise, numba compiles the segments around the Python formulas. If a declaration is missing, the kernel searches at
its first call.

In one case, numba still compiles the kernel at the first call. numbox builds the kernel's declared signature from the
external variables that the formulas read. A requested external variable that no formula reads is also a parameter of
the kernel, and numbox leaves it out of that signature even when the variable is declared. Without a complete
signature, numba compiles the kernel for the types of the first call's arguments.

Declare before the first ``compile_kernel`` or :meth:`Graph.compile <numbox.core.variable.variable.Graph.compile>`
that involves these variables: the graph keeps its compiled result, and a later declaration does not affect it.

Given a declared signature, numba silently converts a result to the declared type: ``x * 1.5`` declared as ``int64``
returns 7 for 5. numbox therefore has numba compile each formula for the declared input types and compares the type
that the formula returns with the declaration. A formula that returns another type raises ``ValueError`` at
``compile_kernel``, such as ``declared type int64 but formula yields float64 at input types (int64,)``, and a formula
that does not compile for the declared types raises numba's ``TypingError`` there.

On a kernel built from declared types, ``recompute`` checks each changed value before it writes the value into the
store. A value that numba cannot convert to the declared type raises ``ValueError: declared type float64, got
complex128 for basket.x``. The check tests only whether numba can convert the value, so a lossy conversion passes, such
as ``7.5`` for an ``int64`` variable. A value that passes is not converted either: numba compiles the recompute plan
again for the value's own type, and ``x * 2`` declared as ``float64`` returned ``6``, an ``int``, for ``3``. numbox
does not check a call of ``ck.kernel``: numba compiles another version for new types, or raises ``TypingError`` if it
cannot, and a kernel built from declared types never searches again.

Limits
------

If compile_kernel raises RecursionError
***************************************

numbox walks the graph's dependencies recursively, so a chain of dependencies deeper than Python's recursion limit
makes ``compile_kernel`` raise ``RecursionError: Consider raising recursion limit.`` The remedy is to raise the limit
with `sys.setrecursionlimit <https://docs.python.org/3/library/sys.html#sys.setrecursionlimit>`_ before the call.

Large graphs
************

A cold compile, with nothing in the cache, takes about 20 to 25 ms and 1 MiB of memory per variable, measured with
numba 0.68 on CPython 3.12. A chain of 400 variables took about 8 s to compile, and a chain of 1600 took 39 s and
1.3 GiB. A graph of thousands of variables is better split into several kernels, each taking the outputs of the previous
kernel as external values, or evaluated with :class:`~numbox.core.variable.variable.CompiledGraph`.

Large arrays
************

numbox hashes the bytes of every array that a formula reads, at every call of ``compile_kernel``: about 90 ms for an
array of 80 MB. Passing a large array to the kernel as an external value avoids that cost, and it also keeps the
kernel cacheable when the array is larger than 1 MB, as described in `If numba warns of dynamic globals`_.

A misspelled external name
**************************

An :class:`~numbox.core.variable.variable.External` namespace creates any variable that is looked up in it, so a
misspelled external name, in a formula's inputs or among the requested variables, is not refused. It becomes one more
parameter of the kernel: ``ck.params`` shows it, ``ck.kernel`` takes one more positional argument, and ``ck.execute``
raises ``KeyError: "Missing external value for ..."`` with the misspelled name when the dictionary holds only the
correct one.

How the kernel is built
-----------------------

``ck.source`` holds the kernel's source as numbox writes it. For ``pricing.py`` it is:

.. code-block:: text

    def _kernel(basket_price, basket_rate, basket_units):
        calc_cost = f_calc_cost(basket_price, basket_units)  # 'calc.cost' = f('basket.price', 'basket.units')
        calc_total = f_calc_total(calc_cost, basket_rate)  # 'calc.total' = f('calc.cost', 'basket.rate')
        return (calc_total, calc_cost,)

Each line calls one formula through a global whose name is ``f_`` followed by the variable's identifier, and its
comment names the variable and its inputs. ``ck.identifiers`` maps each qualified name to its identifier in the
source. Before numba compiles the function, numbox renames it ``_kernel_`` followed by the digest, the name that appears
in numba's messages and in the names of the cache files. Each segment and each recompute plan is built the same way.

Modules
++++++++

numbox.core.variable.variable
-----------------------------

.. automodule:: numbox.core.variable.variable
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.variable.compile_kernel
-----------------------------------

.. automodule:: numbox.core.variable.compile_kernel
   :members:
   :show-inheritance:
   :undoc-members:

The identifier-assignment and formula helpers used by the compiler live in
:mod:`numbox.core.variable.utils`.

.. automodule:: numbox.core.variable.utils
   :members:
   :show-inheritance:
   :undoc-members:
