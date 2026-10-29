numbox.core.proxy
=================

Overview
++++++++

When one numba function calls another, numba links a copy of the callee's compiled code into the caller. For a few
small functions that cost hardly matters. For a function that many jitted callers share, each caller's compiled code
embeds its own copy of the callee, and each cached caller stores that copy on disk.

:func:`~numbox.core.proxy.proxy.proxy` separates the caller from the callee. The decorated function, called the proxied
function below, is compiled when its module is imported, for the signature passed to the decorator. A caller's compiled
code holds a declaration of the compiled function and a call to it, not a copy, so the compiled function lives in one
place. The module-level constant form described in `Holding the function as a module-level constant`_ embeds a copy
after all.

The price is that LLVM cannot inline the call, so the decorator suits functions whose work outweighs a call. numbox
uses it mainly for its bindings: a binding is a proxied function that wraps one C function, such as the bindings to
libc and to SQLite in :doc:`numbox.core.bindings`. Any function that many cached callers share is a candidate.

Using the decorator
+++++++++++++++++++

Decorate the function with its signature. The signature is required, because the body is compiled at import time,
before any call could supply argument types. Saved as ``mymodule.py``, with a docstring whose length `Where the function
may start`_ explains::

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
copy of the body's code. A call with an argument numba can convert to the signature's type, such as an ``int`` for a
``float64``, goes through. A call from Python with any other type raises numba's ``TypeError: No matching definition
for argument type(s)``; from jitted code, such a call fails to compile, and numba reports a failed cast to the
signature's type.

The body is cached only on request. The second argument, ``jit_options``, is passed to ``njit`` for the body and for
the generated wrapper. Pass ``{"cache": True}`` as above, or pass ``numbox.core.configurations.jit_options``, the
setting numbox's own bindings use. That setting caches unless the ``NUMBOX_JIT_OPTIONS`` environment variable turns
caching off. It also turns caching off when numba can write no cache for numbox's own files, and your function is then
not cached either (:doc:`numbox.core.configurations`). Under ``NUMBA_OPT=0``, a debugging setting, numba does not
cache a caller of a proxied function and warns that the caller uses dynamic globals; the caller then compiles on every
run.

Where the function may start
----------------------------

The ``@proxy`` line must be on line 8 of its file or later. When other decorators are stacked above it, the topmost of
them must be on line 8 or later, because Python records the line of a decorated function as the line of its first
decorator. numbox compiles the generated wrapper as though it stood at that line of your file, so that numba caches
the wrapper with your module and notices your edits. The wrapper's own source has seven lines of helper code above its
decorator, so the decorator cannot stand above line 8. A function nearer the top is refused at import with a
``ValueError`` that says where to move it. The module docstring in the example makes the room.

Other decorators may be stacked above ``@proxy``, but no other decorator may sit between ``@proxy`` and ``def``. A
decorator there runs first and hands ``@proxy`` the function it returns: numba cannot compile that function, and its
source is in the decorator's file, not yours.

A proxied function cannot take ``*args`` or ``**kwargs``. Such a function fails at import; with an ordinary signature
the error is numba's signature-mismatch ``TypeError``.

Default arguments
-----------------

For a function with default arguments, pass ``@proxy`` a list of signatures. The first is the main signature. Each
further signature describes a call that omits trailing defaulted parameters; each omitted parameter appears there as
numba's ``Omitted`` type carrying the default value::

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
reaches the body through its C entry point, of the kind ``@cfunc`` builds, and that entry point cannot carry a Python
exception back. Python prints the exception to stderr as ignored, and the call returns a zero of the signature's return
type. If the body can raise, pass the function as a value, as the next section shows, or report failure through the
return value.

Passing the function as a value
+++++++++++++++++++++++++++++++

Jitted code that takes a function as an argument needs a function value, not a dispatcher. The dispatcher carries one
for the main signature, ``.as_func``, a `first-class function value
<https://numba.readthedocs.io/en/stable/reference/types.html#functions>`_. Passing it keeps the body's exceptions and
keeps the receiving function cacheable. In a second file, ``callers.py``::

    from numba import njit

    from mymodule import scale


    @njit(cache=True)
    def apply(f, x):
        return f(x)


    apply(scale.as_func, 3.0)

Read ``.as_func`` in Python and pass the value in. Jitted code cannot read an attribute of a dispatcher, and
``scale.as_func`` inside a jitted function fails to compile. The value carries the main signature only, so a call
through it must pass every parameter: ``shift.as_func`` takes two arguments, and the default for ``y`` applies only to
calls of ``shift`` itself.

An exception raised inside the body propagates out of a call made through ``.as_func``. To keep that propagation, let
numba infer the type of the parameter that receives the value, as ``apply`` does. The value then keeps its own type,
which numba prints as ``DeriveFunctionType[float64(float64)]``, and the exception propagates.

If the parameter is declared as a plain numba function type instead, numba converts the value to a plain function
value on the way in and calls the body through its C entry point. An exception is then dropped, as it is for a call to
the dispatcher. Such a declaration is ``@njit(float64(FunctionType(float64(float64)), float64))``, with
``FunctionType`` from ``numba.types``.

Do not pass the dispatcher itself, as in ``apply(scale, 3.0)``. numba gives that parameter a type derived from the
dispatcher object, and the type's name includes the memory address of the function the dispatcher wraps, which differs
in every process. A cached ``apply`` then recompiles on every run and adds a cache entry each time. ``.as_func`` has the
same type in every process, so ``apply`` compiles once and loads from the cache after that.

Below numba 0.68.0, every function value in a tuple must have the same type, such as ``.as_func`` values of one
signature. A tuple that mixes ``.as_func`` with a ``@cfunc``, an ``@njit`` function compiled for a signature, numba's
own ``CompileResultWAP`` or an ``.as_func`` of another signature fails with an ``AssertionError`` that has no message.
numba 0.68.0 accepts such a tuple. A lazily compiled ``@njit`` function cannot share a tuple with ``.as_func`` on any
numba version: the call raises a ``TypeError``. The :ref:`work page <derive_mixed_function_values>` lists this limit
alongside related ones. For reference, the value's class is :class:`~numbox.utils.derive_wap.DeriveWAP` and its numba
type is :class:`~numbox.utils.derive_wap.DeriveFunctionType`.

Optional C symbols
++++++++++++++++++

A binding to a C library can name a symbol that older versions of the library lack.
:func:`~numbox.core.proxy.proxy.proxy_if_available` takes the loaded library and, before anything else, checks it for
an attribute with the decorated function's name. The function must therefore be named after the C symbol. Saved as
``mybindings.py``::

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
``numbox.core.bindings.signatures``. Every libm has ``cos``, so this example never takes the stub path; it stands in
for a symbol such as ``sqlite3_changes64``, which SQLite added in version 3.37 and numbox guards this way.

If the symbol is present, the result is an ordinary proxied function. If it is absent, the result is a stub: a Python
call raises ``NotImplementedError``, and a jitted caller fails to compile with a ``TypingError`` that names the binding
and says that its C symbol is missing. The stub has no ``.as_func``, so code that passes the function as a value checks
for it first::

    from callers import apply
    from mybindings import cos

    if hasattr(cos, "as_func"):
        apply(cos.as_func, 0.0)

Caching
+++++++

Three things can change under a cached caller: the proxied function, after you edit it; numbox, after an upgrade; and
the C library behind a binding. numbox repairs the first two cases on its own and reports the third. The one form it
cannot repair is the module-level constant of `Holding the function as a module-level constant`_.

After you edit a proxied function
---------------------------------

Cached callers in other files still refer to the old version. On the next run, numbox recompiles each of them and
warns once per stale cache entry::

    StaleProxyCacheWarning: numbox: discarding a stale numba cache entry and recompiling: pipeline.vec_scale-7.py312
    references @proxy cfunc alias(es) this process cannot call (the proxied body, signature or jit_options changed
    since the caller was cached, or the binding is no longer available): numbox_pxy_scale_e653de386ea6678b

The first name in the warning is the caller's cache entry: its module, its function, the line of its decorator and the
Python version, here a caller ``vec_scale`` in a module ``pipeline``. Later runs are quiet, and you need not clear
anything. Callers in the same file as the proxied function recompile anyway, because numba sees that file change. The
repair covers callers decorated with ``@njit``, ``@vectorize``, ``@guvectorize`` and ``@cfunc``. The exception is a
caller that holds ``.as_func`` as a module-level constant; see `Holding the function as a module-level constant`_.

The name at the end of the warning is the alias of the compiled body: a symbol that numbox registers in each process,
through which callers reach the body. numbox computes the alias from the body's code and the values it captures
(closure variables, defaults and the globals it reads), from the function's module and name, and from its signature and
jit options. The same body therefore gets the same alias in every process, and any change to those renames it.

numba's cache key for a caller :ref:`does not include the callees <cache_key_leaves_out>`, so numbox checks each cache
entry as numba loads it, and discards and recompiles an entry that refers to an alias this process has not registered.
Without the check, the entry would load with the old alias, and the process would end in a segmentation fault or an
``LLVM ERROR: Symbol not found`` abort. No import order needs arranging: a caller imports the module of the function it
calls, that module imports ``numbox.core.proxy.proxy``, and the check is installed when that module is imported.

After you upgrade numbox
------------------------

numbox's version is part of the cache key of every proxied function, so the first run after an upgrade compiles each
proxied body and its wrapper once more and caches them again. The alias does not include the version, so cached callers
load as before. A release that changes the code of one of numbox's bindings renames that binding's alias, and one that
changes how aliases are computed renames every alias, yours included. That run repairs each affected cached caller with
one warning of the kind above. The recompilation and the repairs are one-time costs. A caller that holds a numbox
binding as a module-level constant is the exception again, and its cache entry must be cleared by hand; `Holding the
function as a module-level constant`_ says how.

If numbox warns of an alias collision
-------------------------------------

An :class:`~numbox.core.proxy.proxy.AliasCollisionWarning` means that two different proxied functions computed the
same alias. The alias can include a captured value only if the value is the same in every process. A numba
``ExternalFunction`` is the same in every process, and so is a C function reached as an attribute of the loaded
library, such as ``lib.strlen``. A ctypes function built from a raw integer address is not, because address space
layout randomization moves it between processes, so numbox leaves it out of the alias. Two bodies that differ only in
such a value are then indistinguishable and get one alias.

The remedy is to reach the C function as an attribute of the loaded library rather than through a pointer built from
an address. Until you apply it, numbox gives the second body a name used only in this process and stops using the
shared alias; cached callers of either body then recompile in every process instead of running the wrong body.
Registering the same body twice is not a collision. After a module reload, or when a function that builds proxied
functions is called twice with equal arguments, the alias is pointed at the new compilation and numbox keeps using it.

The same warning can also report a ``proxy_if_available`` symbol whose availability differs between two decorations in
the same process; the message says which body callers reach.

If a C symbol disappears
------------------------

Suppose a caller of a ``proxy_if_available`` binding was cached where the library has the symbol, and later runs where
it does not. numbox discards the caller's cache entry with a ``StaleProxyCacheWarning``, and the recompile fails with
the ``TypingError`` a fresh compile gives.

Strict mode
-----------

Set ``NUMBOX_PROXY_CACHE_STRICT`` to make a stale entry fail instead of being repaired, for example to make a test run
fail or to keep the entry on disk for inspection. Any value except ``0``, ``false``, ``no``, ``off`` or the empty
string, in upper or lower case, turns it on. A stale alias then raises
:class:`~numbox.core.proxy.proxy.StaleProxyCacheError` before the discard. A test run can also turn the warning itself
into an error with pytest's ``-W error::numbox.core.proxy.proxy.StaleProxyCacheWarning``.

If numbox cannot check a cache entry
------------------------------------

The check reads the compiled code in each cache entry to find the aliases it refers to. If the check cannot read an
entry, because a numba release changed the entry's layout or the file is damaged, numbox loads the entry unchecked and
warns once per process with :class:`~numbox.core.proxy.proxy.UnvalidatedProxyCacheWarning`. In strict mode the load
raises :class:`~numbox.core.proxy.proxy.UnvalidatedProxyCacheError` instead.

Holding the function as a module-level constant
+++++++++++++++++++++++++++++++++++++++++++++++

numba reads a module-level variable as a compile-time constant, so jitted code can read ``.as_func`` from one, although
it cannot read ``scale.as_func`` itself. Saved as ``constant.py``::

    from numba import njit

    from mymodule import scale

    SCALE = scale.as_func


    @njit(cache=True)
    def twice(x):
        return SCALE(x) + SCALE(x)

This form gives a direct call, which LLVM can inline, and an exception raised in the body propagates as it does
through an argument. ``twice`` is cacheable: when numbox lowers the constant, it links the body's compiled code into
the caller, as numba does for an ordinary callee. Because of the two costs below, prefer passing ``.as_func`` as an
argument unless the inlining matters.

The first cost is the one the overview describes: ``twice`` embeds its own copy of the compiled body. The second is
that the copy goes stale silently. The caller's compiled code names no alias, so numbox's check cannot see an edit to
the body, and numba checks only the caller's own source file for changes. After you edit the body, ``twice`` may keep
returning the old result with no warning. The same happens after a numbox upgrade when the constant is a numbox
binding. A ``StaleProxyCacheWarning`` in the same run names other callers, never this one.

The remedy is to delete the caller's cache files. Their names are the module, the function, the line of its decorator
and the Python version, here ``constant.twice-8.py312.nbi`` and ``constant.twice-8.py312.1.nbc``. They are:

- in a subdirectory of ``NUMBA_CACHE_DIR`` named after the source's directory, if that variable is set;
- otherwise in the ``__pycache__`` directory beside the caller's source;
- or in numba's user cache directory, if numba could not write that ``__pycache__``; the configurations page says
  :ref:`where the cache lands <where_the_cache_lands>`.

Even when ``twice`` and ``scale`` return the same result, the cache entry may still hold the old body, so delete the
files after every edit of the body.

A binding that ``proxy_if_available`` may find absent needs care in this form. Where the symbol is missing, a caller
cached while it was present still loads, and calling it can end the process in a segmentation fault with nothing on
stderr. With an empty cache, a caller such as ``use_const`` below fails to compile with a ``TypingError``, so a warm and
a cold run disagree. Strict mode cannot catch this, because the caller's compiled code names no alias for the check to
find.

Two forms cannot go stale. Reading ``.as_func`` with no check cannot, but it makes your module require the symbol:
where the symbol is absent, the import fails on the missing attribute before any cache is reached. Passing ``.as_func``
as an argument, as ``apply`` does, cannot either: the value arrives at call time, and a stub has no value to pass.

Of the two guarded forms below, one is safe and one is not::

    from numba import njit

    from mybindings import cos

    if hasattr(cos, "as_func"):
        COS = cos.as_func

        @njit(cache=True)
        def use(x):
            return COS(x)              # safe: the caller exists only where the binding does


    MAYBE_COS = cos.as_func if hasattr(cos, "as_func") else None


    @njit(cache=True)
    def use_const(x):
        return MAYBE_COS(x)            # not safe: the cached entry loads the binding from the run that had it

The last form is unsafe because numba's cache key does not include the value of a module-level variable such as
``MAYBE_COS``. The cached ``use_const`` loads with the body it embedded when it was compiled, whatever ``MAYBE_COS`` now
holds.

Modules
+++++++

numbox.core.proxy.proxy
-----------------------

.. automodule:: numbox.core.proxy.proxy
   :members:
   :show-inheritance:
   :undoc-members:
