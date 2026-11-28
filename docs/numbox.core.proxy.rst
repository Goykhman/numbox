numbox.core.proxy
=================

Overview
++++++++

When one numba function calls another, numba links a copy of the callee's compiled code into the caller. For a few
small functions, the cost of these copies is negligible. For a function that many jitted callers share, each caller's
compiled code embeds its own copy of the callee, and each cached caller stores that copy on disk.

:func:`~numbox.core.proxy.proxy.proxy` separates the caller from the callee. The decorated function, called the proxied
function below, is compiled when its module is imported, for the signature passed to the decorator. A caller's compiled
code holds a declaration of the compiled function and a call to it, not a copy. The compiled function therefore exists
in one place.

The drawback is that LLVM cannot inline the call. The decorator is therefore suited to functions whose work takes longer
than the call itself. numbox uses it mainly for its bindings: a binding is a proxied function that wraps one C function,
such as the bindings to libc and to SQLite in :doc:`numbox.core.bindings`. Any function that many cached callers share
is a candidate.

Using the decorator
+++++++++++++++++++

Decorate the function with its signature. The signature is required, because the body is compiled at import time,
before any call could supply argument types. `Where the function may start`_ explains the length of the module docstring
below. The following example is saved as ``mymodule.py``:

.. code-block:: python

    """A module with a proxied function and a cached caller.

    The @proxy line must be on line 8 of the file or later; this docstring makes the room.
    """
    from numba import float64, njit
    from numbox.core.proxy.proxy import proxy


    @proxy(float64(float64), jit_options={"cache": True})
    def scale(x):
        return 2.0 * x


    @njit(cache=True)
    def total(xs):
        acc = 0.0
        for x in xs:
            acc += scale(x)
        return acc

After the decoration, the name ``scale`` refers to a numba dispatcher for a small wrapper that numbox generates around
the compiled body. The wrapper is named ``__scale``, so expect that name, not ``scale``, in numba's messages. Python
code calls ``scale`` as before. Jitted code such as ``total`` calls the body by a name that numbox registers in every
process, the alias that `Caching`_ describes. When ``total`` is cached, its cache entry holds that name instead of a
copy of the body's code. A call succeeds when numba can convert the argument to the signature's type, such as an ``int``
for a ``float64``. A call from Python with any other type raises numba's ``TypeError: No matching definition for
argument type(s)``; from jitted code, such a call fails to compile, and numba reports a failed cast to the signature's
type.

The body is cached only on request. The second argument, ``jit_options``, is passed to ``njit`` for the body and for
the generated wrapper. Pass ``{"cache": True}`` as above, or pass ``numbox.core.configurations.jit_options``, the
setting numbox's own bindings use. That setting enables caching unless the ``NUMBOX_JIT_OPTIONS`` environment variable
disables it. It also disables caching when numba can write no cache for numbox's own files, and your function is then
not cached either (:doc:`numbox.core.configurations`). Under ``NUMBA_OPT=0``, a debugging setting, numba does not cache
a caller of a proxied function and warns that the caller uses dynamic globals; the caller then compiles on every run.

A function that you define inside another function, such as a factory that makes one proxied function for each value it
receives, has its body cached but its wrapper compiled in every process. numba's cache cannot distinguish the wrappers
of two such functions, because they share a name, a line and their code, and differ only in the body that they call. The
wrapper is small and compiles quickly, but the compile time grows with the number of such functions that a process
makes. A function at module level is treated the same way if the process already decorated a function of the same name
in the same module that is not nested in another function, as when a module defines a function twice. The wrappers of
the two functions share a name too. Otherwise, in a process that loaded one of them from the cache and compiled the
other, a jitted function that received the first function as a first-class function value could run the body of the
second function. :func:`~numbox.core.cache_key.explain_cache_key` raises a ``ValueError`` for the dispatcher of such a
function. If the body is cached, pass the function's ``.as_func`` to see the key of the body.

.. _proxy_where_the_function_may_start:

Where the function may start
----------------------------

The function's first decorator line must be on line 8 of its file or later. That line is the ``@proxy`` line, or the
topmost decorator when others are stacked above it, because Python records the line of a decorated function as the line
of its first decorator. numbox compiles the generated wrapper as though it were defined at that line of your file, so
that numba caches the wrapper with your module and detects your edits. The wrapper of a function defined inside another
function is not cached, as described above, and is also compiled as though it were defined at that line. The wrapper's
own source has seven lines of helper code above its decorator. That decorator therefore cannot be above line 8. A
function nearer the top is refused at import with a ``ValueError`` that says where to move it. In the example, the
module docstring moves the ``@proxy`` line to line 8 or later.

Other decorators may be stacked above ``@proxy``. A decorator between ``@proxy`` and ``def`` runs first, so numba
compiles whatever that decorator returns in place of your function. Put a decorator there only if it returns the
function unchanged.

A proxied function cannot take ``*args`` or ``**kwargs``. Such a function fails at import.

Default arguments
-----------------

For a function with default arguments, pass ``@proxy`` a list of signatures. The first is the main signature. Each
further signature describes a call that omits trailing defaulted parameters. In such a signature, each omitted parameter
appears as numba's ``Omitted`` type, constructed with the default value:

.. code-block:: python

    """A proxied function with a default argument.

    The @proxy line must be on line 8 of the file or later; this docstring makes the room.
    """
    from numba import float64, njit
    from numba.types import Omitted
    from numbox.core.proxy.proxy import proxy


    @proxy([float64(float64, float64), float64(float64, Omitted(1.3))])
    def shift(x, y=1.3):
        return x + y


    @njit
    def use(x):
        return shift(x) + shift(x, 2.0)   # 2.3 + 3.0 for x = 1.0

Exceptions
----------

If the body raises an exception, a call to ``scale``, from Python or from jitted code, does not propagate it. The call
reaches the body through its C entry point, of the kind that ``@cfunc`` builds. That entry point cannot propagate a
Python exception to its caller. Python prints the exception to stderr below a line that begins
``Exception ignored in:``. The call returns a zero of the signature's return type. If the body can raise, pass the
function as a value, as the next section shows, or report failure through the return value.

.. _proxy_passing_as_value:

Passing the function as a value
+++++++++++++++++++++++++++++++

Jitted code that takes a function as an argument needs a function value, not a dispatcher. The dispatcher's attribute
``.as_func`` is such a value for the main signature: a `first-class function value
<https://numba.readthedocs.io/en/stable/reference/types.html#functions>`_. When you pass ``.as_func``, an exception
raised in the body propagates. The receiving function also remains cacheable. The following example is saved in a second
file, ``callers.py``:

.. code-block:: python

    from numba import njit

    from mymodule import scale


    @njit(cache=True)
    def apply(f, x):
        return f(x)


    apply(scale.as_func, 3.0)

Read ``.as_func`` in Python and pass the value to the jitted function as an argument. Jitted code cannot read an
attribute of a dispatcher, and ``scale.as_func`` inside a jitted function fails to compile. The value has only the main
signature, so a call through it must pass every parameter: ``shift.as_func`` takes two arguments, and the default for
``y`` applies only to calls of ``shift`` itself.

An exception raised inside the body propagates out of a call made through ``.as_func``. To keep that propagation, let
numba infer the type of the parameter that receives the value, as ``apply`` does. The value then keeps its own type,
which numba prints as ``DeriveFunctionType[float64(float64)]``, and the exception propagates.

If the parameter is declared as a plain numba function type instead, numba converts the value to a plain function
value when Python code passes it to the jitted function. The jitted function calls the body through its C entry point.
An exception is then discarded, as it is for a call to the dispatcher. Such a declaration is
``@njit(float64(FunctionType(float64(float64)), float64))``, with ``FunctionType`` from ``numba.types``.

Do not pass the dispatcher itself, as in ``apply(scale, 3.0)``. numba derives the type of that parameter from the
dispatcher object. The type's name includes the memory address of the function that the dispatcher wraps, an address
that differs in every process. A cached ``apply`` then recompiles on every run and adds a cache entry each time.
``.as_func`` has the same type in every process, so ``apply`` compiles once and loads from the cache after that.

Below numba 0.68.0, a tuple that mixes ``.as_func`` with a function value of another type, such as a ``@cfunc``, an
``@njit`` function compiled for a signature, numba's own ``CompileResultWAP`` or an ``.as_func`` of another signature,
can fail with an ``AssertionError`` that has no message. numba 0.68.0 accepts such a tuple. A lazily compiled ``@njit``
function that takes as many arguments as an ``.as_func`` cannot share a tuple with it on any numba version: the call
raises a ``TypeError``. The :ref:`work page <derive_mixed_function_values>` lists this limit alongside related ones.
For reference, the value's class is :class:`~numbox.utils.derive_wap.DeriveWAP` and its numba type is
:class:`~numbox.utils.derive_wap.DeriveFunctionType`.

.. _proxy_optional_c_symbols:

Optional C symbols
++++++++++++++++++

A binding to a C library can name a symbol that older versions of the library lack.
:func:`~numbox.core.proxy.proxy.proxy_if_available` takes the loaded library and first checks it for an attribute with
the decorated function's name. The function must therefore be named after the C symbol. The following example is saved
as ``mybindings.py``:

.. code-block:: python

    """A binding to a C symbol that may be absent.

    The decorator line must be on line 8 of the file or later; this docstring makes the room.
    """
    from numba import float64
    from numba.types import ExternalFunction
    from numbox.core.bindings.utils import load_lib
    from numbox.core.proxy.proxy import proxy_if_available

    lib = load_lib("m")
    _cos = ExternalFunction("cos", float64(float64))


    @proxy_if_available(lib, float64(float64))
    def cos(x):
        return _cos(x)

:func:`~numbox.core.bindings.utils.load_lib` opens the library in global symbol mode, so that compiled code can resolve
its symbols; a ``ctypes.CDLL`` opened without that mode is not visible to the compiled code. The body calls the C
function through numba's ``ExternalFunction``, which names the symbol and its C signature. numbox's own bindings call a
private helper instead, ``_call_lib_func``, which serves only the symbols in numbox's table
``numbox.core.bindings.signatures``. Every libm has ``cos``, so ``proxy_if_available`` never returns a stub in this
example. The example uses ``cos`` in place of a symbol such as ``sqlite3_changes64``, which SQLite added in
version 3.37. numbox guards that symbol in the same way.

If the symbol is present, the result is an ordinary proxied function. If it is absent, the result is a stub: a Python
call raises ``NotImplementedError``, and a jitted caller fails to compile with a ``TypingError`` that names the binding
and says that its C symbol is missing. The stub has no ``.as_func``, so code that passes the function as a value checks
for it first:

.. code-block:: python

    from callers import apply
    from mybindings import cos

    if hasattr(cos, "as_func"):
        apply(cos.as_func, 0.0)

.. _proxy_caching:

Caching
+++++++

After a caller is cached, three things can change: the proxied function, when you edit it; numbox, when you upgrade it;
and the C library that a binding calls. numbox repairs the first two cases automatically and reports the third.

After you edit a proxied function
---------------------------------

Cached callers in other files still refer to the old version. On the next run, numbox recompiles each of them and
warns once per stale cache entry::

    StaleProxyCacheWarning: numbox: discarding a stale numba cache entry and recompiling: pipeline.vec_scale-7.py312
    references numbox alias(es) this process cannot call (the compiled body, signature or jit_options changed
    since the caller was cached, the binding is no longer available, or two functions in this process share the
    alias): numbox_pxy_scale_e653de386ea6678b

The first name in the warning is the caller's cache entry: its module, its function, the line of its decorator and the
Python version, here a caller ``vec_scale`` in a module ``pipeline``. In later runs, numbox issues no such warning. You
do not need to clear the cache. numba itself recompiles the callers in the same file as the proxied function, because it
detects that the file has changed. The repair covers callers decorated with ``@njit``, ``@vectorize``, ``@guvectorize``
and ``@cfunc``. It covers a caller that calls the function and a caller that holds ``.as_func`` as a module-level
constant; see `Holding the function as a module-level constant`_.

The name at the end of the warning is the alias of the compiled body: a symbol that numbox registers in each process,
through which callers reach the body. numbox computes the alias from the body's code and the values it captures
(closure variables, defaults and the globals it reads), from the function's module and name, and from its signature and
jit options. The same body therefore has the same alias in every process. Any change to those inputs renames the alias.
A caller that holds ``.as_func`` as a module-level constant refers to the alias with ``_jit`` appended, and the warning
then shows that name.

numba's cache key for a caller :ref:`does not include the callees <cache_key_leaves_out>`, so numbox checks each cache
entry as numba loads it, and discards and recompiles an entry that refers to an alias this process has not registered.
Without the check, the entry would load with the old alias, and the process would end in a segmentation fault or an
``LLVM ERROR: Symbol not found`` abort. The check does not depend on import order: a caller imports the module of the
function it calls, that module imports ``numbox.core.proxy.proxy``, and the check is installed when that module is
imported.

.. _proxy_after_upgrade:

After you upgrade numbox
------------------------

numbox's version is part of the cache key of every proxied function, as :doc:`numbox.core.cache_key` explains. The first
run after an upgrade therefore compiles each proxied body and each cached wrapper once more and caches them again. The
alias does not include the version, so cached callers load as before. A release that changes the code of one of numbox's
bindings renames that binding's alias. A release that changes how aliases are computed renames every alias, including
the aliases of your own proxied functions. numbox repairs each affected cached caller in the first run after such a
release and issues one warning of the kind shown above per caller. The recompilation and the repairs are one-time costs.
A caller that holds a numbox binding as a module-level constant is repaired in the same way, unless numbox 0.7.9 or
earlier wrote its cache entry. Delete such an entry once, as `After an upgrade from, or a downgrade to, numbox 0.7.9 or
earlier`_ explains.

.. _proxy_alias_collision:

If numbox warns of an alias collision
-------------------------------------

An :class:`~numbox.core.proxy.proxy.AliasCollisionWarning` means that two different proxied functions computed the
same alias. The alias can include a captured value only if the value is the same in every process. A numba
``ExternalFunction`` is the same in every process, and so is a C function reached as an attribute of the loaded
library, such as ``lib.strlen``. A ctypes function built from a raw integer address is not the same in every process,
because address space layout randomization changes the address from one process to the next. numbox therefore excludes
such a function from the alias. Two bodies that differ only in such a value then have the same alias, because numbox
cannot distinguish them.

The remedy is to reach the C function as an attribute of the loaded library rather than through a pointer built from
an address. Until you apply it, numbox assigns the second body a name that is used only in this process and stops using
the shared alias. Cached callers of either body then recompile in every process instead of running the wrong body.
Registering the same body twice is not a collision. After a module reload, or when a function that builds proxied
functions is called twice with equal arguments, the alias is pointed at the new compilation and numbox keeps using it.
A caller that holds ``.as_func`` as a module-level constant is treated differently, as `If two functions share an
alias`_ explains.

The same warning can also report a ``proxy_if_available`` symbol whose availability differs between two decorations in
the same process; the message says which body callers reach.

If a C symbol disappears
------------------------

Suppose a caller of a ``proxy_if_available`` binding was cached where the library has the symbol, and later runs where
it does not. numbox discards the caller's cache entry with a ``StaleProxyCacheWarning``. The recompilation then fails
with the same ``TypingError`` as a compilation with an empty cache. A caller that holds the binding's ``.as_func`` as a
module-level constant is discarded in the same way.

Strict mode
-----------

Set ``NUMBOX_PROXY_CACHE_STRICT`` to make a stale entry fail instead of being repaired, for example to make a test run
fail or to keep the entry on disk for inspection. Any value except ``0``, ``false``, ``no``, ``off`` or the empty
string, in upper or lower case, enables strict mode. For a stale alias, numbox then raises
:class:`~numbox.core.proxy.proxy.StaleProxyCacheError` instead of discarding the entry. A test run can also turn the
warning itself into an error with pytest's ``-W error::numbox.core.proxy.proxy.StaleProxyCacheWarning``.

If numbox cannot check a cache entry
------------------------------------

The check reads the compiled code in each cache entry to find the aliases it refers to. If the check cannot read an
entry, because a numba release changed the entry's layout or the file is damaged, numbox loads the entry unchecked and
warns once per process with :class:`~numbox.core.proxy.proxy.UnvalidatedProxyCacheWarning`. In strict mode the load
raises :class:`~numbox.core.proxy.proxy.UnvalidatedProxyCacheError` instead.

.. _proxy_module_level_constant:

Holding the function as a module-level constant
+++++++++++++++++++++++++++++++++++++++++++++++

numba reads a module-level variable as a compile-time constant, so jitted code can read ``.as_func`` from one, although
it cannot read ``scale.as_func`` itself. The following example is saved as ``constant.py``:

.. code-block:: python

    from numba import njit

    from mymodule import scale

    SCALE = scale.as_func


    @njit(cache=True)
    def twice(x):
        return SCALE(x) + SCALE(x)

An exception raised in the body propagates, as it does when ``.as_func`` is passed as an argument, and ``twice`` is
cacheable. The compiled code of ``twice`` holds a declaration of the body's numba entry point and a call to it, under
the body's alias with ``_jit`` appended. It holds no copy of the body. LLVM therefore cannot inline the call, and a call
costs the same as a call through an argument. Choose between the two forms according to whether the caller should take
the function as a parameter.

A function that ``numba.pycc`` compiles ahead of time is the exception. It holds a copy of the body, because the loader
of an ahead-of-time module cannot resolve a name that numbox registers at run time. ``numba.pycc`` compiles only the
exported function that way. A jitted function that the exported function calls is compiled in the same way as for the
running process, and if it holds ``.as_func`` as a module-level constant, the module fails to import with an undefined
symbol. Read ``.as_func`` in the exported function and pass it to the jitted function as an argument.

numbox's check covers this form. After you edit the body, numbox discards the cache entry of ``twice``, and numba
compiles ``twice`` again, with the warning shown under `After you edit a proxied function`_. The same happens after a
numbox upgrade that changes a binding which the caller holds.

A binding that ``proxy_if_available`` may find absent is covered as well. Of the two guarded forms below, the first
defines the caller only where the binding exists, and the second defines it everywhere:

.. code-block:: python

    from numba import njit

    from mybindings import cos

    if hasattr(cos, "as_func"):
        COS = cos.as_func

        @njit(cache=True)
        def use(x):
            return COS(x)              # the caller exists only where the binding does


    MAYBE_COS = cos.as_func if hasattr(cos, "as_func") else None


    @njit(cache=True)
    def use_const(x):
        return MAYBE_COS(x)            # fails to compile where the binding is absent

Where the symbol is missing, a call of ``use_const`` fails to compile with a ``TypingError``. If ``use_const`` was
cached while the symbol was present, numbox first discards that cache entry with a ``StaleProxyCacheWarning``. A run
with a warm cache and a run with an empty cache therefore fail with the same error. In strict mode, the stale entry
raises :class:`~numbox.core.proxy.proxy.StaleProxyCacheError` instead.

.. _proxy_shared_alias:

If two functions share an alias
-------------------------------

``twice`` must call the function that ``SCALE`` holds, and the compiled code of ``twice`` refers to that function only
by name. numbox therefore never registers a ``_jit`` name a second time. The first function in a process that computes
an alias receives the name. Each later function that computes the same alias receives a name of its own, which is used
only in that process. Two functions compute one alias in four cases:

- a module is reloaded, which compiles its proxied functions again;
- a function that builds proxied functions is called twice with equal arguments;
- one function is decorated twice with the same signature and options;
- two bodies differ only in a value that the alias leaves out. Examples are a captured ufunc or ``math`` function, a
  module attribute that holds a function, a ufunc or a class, as in ``helpers.op(x)``, and a value that is not the same
  in every process, which `If numbox warns of an alias collision`_ describes.

In a process where two functions share an alias, numbox does not load a cached caller that holds either function as a
module-level constant. Another process may create the two functions in the other order, so the alias does not identify
the function that a cached caller was compiled against. numbox discards the entry with the ``StaleProxyCacheWarning``
shown above, and numba compiles the caller again. Both occur in every such process, and strict mode raises
:class:`~numbox.core.proxy.proxy.StaleProxyCacheError` there. A caller that takes ``.as_func`` as an argument is not
affected, and its cache entry loads in every process.

numbox detects a shared alias only in a process that creates both functions. It also cannot detect that ``SCALE``
holds a different function from the one it held in the run that cached ``twice``. That happens when ``SCALE`` is
imported from a module that selects the function from a setting, and a later run selects another function. numba
treats a global as a constant, and a function loaded from the cache `keeps the value from the run that compiled it
<https://numba.readthedocs.io/en/stable/developer/caching.html#caching-limitations>`_. numba therefore loads the entry
of ``twice`` for as long as the file that defines ``twice`` is unchanged. If the later run registers no function under
the name that the entry refers to, numbox discards the entry, and numba compiles ``twice`` again. If the run still
creates the old function, or creates another function that computes the same alias, numba loads the entry, and
``twice`` calls that function and not the one that ``SCALE`` holds. After such a change, delete the cache files of
``twice`` or pass the function as an argument.

After an upgrade from, or a downgrade to, numbox 0.7.9 or earlier
-----------------------------------------------------------------

numbox 0.7.9 and earlier linked a copy of the body into a caller of this form. A cache entry that one of those releases
wrote holds that copy and names no alias, so numbox's check finds nothing in it to test. numba keeps loading the entry
for as long as the caller's source file is unchanged. The caller then keeps running the body as it was when the entry
was written, without a warning, even after you edit the body or upgrade numbox. If the body is a ``proxy_if_available``
binding whose symbol is now missing, the process can end in a segmentation fault without any message on stderr.

After an upgrade from such a release, delete the cache files of each caller that holds a function value as a
module-level constant. One deletion is enough: numba compiles the caller again, and numbox checks the new entry from
then on. The names of the files are the module, the function, the line of its decorator and the Python version, here
``constant.twice-8.py312.nbi`` and ``constant.twice-8.py312.1.nbc``. They are:

- in a subdirectory of ``NUMBA_CACHE_DIR`` named after the source's directory, if that variable is set;
- otherwise in the ``__pycache__`` directory beside the caller's source;
- or in numba's user cache directory, if numba could not write that ``__pycache__``; the configurations page explains
  :ref:`where numba writes the cache <where_the_cache_lands>`.

Saving the caller's source file again also works, because numba compiles a function again when its file has a newer
modification time. A caller decorated with :func:`~numbox.core.cache_key.numbox_njit` needs neither step. Its cache key
includes numbox's version, so it compiles again after every upgrade.

A downgrade to numbox 0.7.9 or earlier needs no step for a caller that holds ``.as_func``. The older release does not
register the name that a newer cache entry refers to, so its check discards the entry, warns once and compiles the
caller again. A caller that holds a ``cres`` function as a constant is not safe after such a downgrade, as the utils
page explains under :ref:`holding the value as a module-level constant <cres_module_level_constant>`.

Modules
+++++++

numbox.core.proxy.proxy
-----------------------

.. automodule:: numbox.core.proxy.proxy
   :members:
   :show-inheritance:
   :undoc-members:
