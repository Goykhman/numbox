numbox.core.proxy
=================

Overview
++++++++

Implementation of :func:`numbox.core.proxy.proxy` decorator that swaps definition of a jit-compiled
function in-place for a declaration (while delegating the actual implementation
to a different function that is only accessible indirectly). As a result, statically linking in libraries
corresponding to proxy-jitted functions called from other jitted functions will
only paste a declaration rather than the entire LLVM IR code. One route is exempt: a jitted caller
that reaches a binding's ``.as_func`` as a compile-time constant links the body's code in rather than
a declaration, with the caching consequence described under `Cache invalidation`_ below.

The ``.as_func`` first-class value
++++++++++++++++++++++++++++++++++

Besides being callable, the dispatcher a ``@proxy`` decoration returns exposes
``.as_func``: a first-class function value for the main signature, to hand to a
jitted function that takes the binding as a ``FunctionType`` argument, or to
reference from jitted scope as a constant.

Passing it as an argument is what keeps the *receiving* function cacheable. Hand a
``@njit(cache=True)`` function the dispatcher itself and that parameter types as
``type(CPUDispatcher(...))``, whose name carries the dispatcher's address, so the
cache index key differs in every process. The receiving function then never matches
what an earlier run wrote: it recompiles on every run and leaves another overload
behind in the index. Hand it ``.as_func`` and the parameter types as a function type
instead, ``FunctionType[float64(float64)]`` on numba 0.60 and
``DeriveFunctionType[float64(float64)]`` from 0.61, which is structural and identical
in every process, so the receiving function is compiled once and loaded from the
cache in every later one.

From numba 0.61 onward ``.as_func`` is a
:class:`~numbox.utils.derive_wap.DeriveWAP` typed as
:class:`~numbox.utils.derive_wap.DeriveFunctionType`, so the ``jit_addr`` slot of
its data model carries the numba calling convention entry point and an exception
raised inside the proxied body propagates out of a first-class call instead of
being discarded. numba 0.60 has no such slot, so ``.as_func`` is a plain
``CompileResultWAP`` there and the exception is still discarded. ``DeriveWAP``
subclasses ``CompileResultWAP``, so ``isinstance`` passes on either version and
only ``type(...) is CompileResultWAP`` tells the two apart.

Only one of the binding's two handles propagates. **Calling the dispatcher itself
still discards**, on every numba version: that call reaches the proxied body
through its C-convention cfunc wrapper, which does not unwind, so the exception
surfaces only as an unraisable on stderr and the call returns a zero-filled
value. Handing ``.as_func`` from Python into a jitted parameter *declared* as a
plain ``FunctionType`` discards as well, because the value degrades to the C
convention on the way in; ``DeriveFunctionType.can_convert_to`` permits that
conversion deliberately, so such a call site goes on compiling unchanged. An
inferred-signature ``@njit`` argument keeps the numbox type and propagates, from
numba 0.61 onward; on 0.60 there is no numbox type to keep and it discards along
with the rest.

From numba 0.61 onward ``.as_func`` inherits the mixed-container limit of any
derive value: a tuple holding it alongside a plain ``CompileResultWAP`` no longer
unifies, failing with a message-less ``AssertionError`` from
``numba.core.utils.unified_function_type``. See :doc:`numbox.core.work` for that
limit and for why making the two types compare equal is not available as a fix.

Referencing ``.as_func`` as a constant carries a caching caveat that passing it
as a function-type argument does not; read `Cache invalidation`_ below before
capturing one into a module-level global.

Cache-anchor mechanism
++++++++++++++++++++++

The ``@proxy`` decorator generates a thin wrapper function via ``exec()``.
For numba to cache that wrapper across processes, the wrapper's bytecode
needs a ``co_filename`` and ``co_firstlineno`` that point at real Python
source — both because numba's cache stamp uses ``(st_mtime, st_size)`` of
``co_filename`` for invalidation, and because
``inspect.getsourcelines(wrapper)`` gets called during numba's annotation
pipeline.

Anchoring at the user's file
----------------------------

The wrapper anchors at ``inspect.getfile(func)`` — the user's ``.py``
file where the ``@proxy`` decoration sits. Blank lines are prepended to
the generated wrapper source so the wrapper's ``@njit`` decorator
(which is what Python records as ``co_firstlineno`` for a decorated
function) lands at ``func.__code__.co_firstlineno`` — i.e. exactly the
line of the user's ``@proxy`` decorator. ``inspect.findsource(wrapper)``
then matches that ``@proxy`` line on its first check via
``r'^(\s*@)'`` — no backward scan needed, tokenization proceeds from
real, syntactically valid Python.

The hazard this avoids
----------------------

``inspect.findsource`` searches backward from ``co_firstlineno`` for any
line matching its pattern, *including* lines inside docstrings that
happen to start with ``@``. A docstring mentioning
``@njit(parallel=True) workers`` indented four spaces would be matched
as if it were a real decorator, and the C tokenizer would then read
``worker's`` (the apostrophe) as an unterminated string literal and
raise ``TokenError``. Related to CPython issue
`#122981 <https://github.com/python/cpython/issues/122981>`_.

Placing the wrapper's ``co_firstlineno`` directly at the user's
``@proxy`` line means ``findsource`` matches without scanning, and the
docstring contents are never re-tokenized.

Cache invalidation
------------------

For a file-backed cached ``@njit``, numba's source stamp is
``(os.stat(co_filename).st_mtime, st_size)`` — see
``numba.core.caching._SourceFileBackedLocatorMixin.get_source_stamp``.
Any edit to the user's file (the file containing the ``@proxy``
decoration) invalidates the wrapper's cache. Edits to ``proxy.py``'s
wrapper template itself — without a corresponding user-file edit — do
*not* invalidate the cache (the user file's mtime is unchanged); treat
wrapper-template changes as developer-managed. Clear the affected
entries — the ``.nbc`` / ``.nbi`` files in the ``__pycache__`` beside
each binding's source, or under ``NUMBA_CACHE_DIR`` — when shipping a
template change to numbox.

A second staleness shape sits outside the wrapper's anchor entirely. A
``cache=True`` caller that reaches a binding's ``.as_func`` as a compile-time
constant is newly cacheable, and none of the stamps above notice when the proxied
body changes underneath it. Constant-lowering a plain ``CompileResultWAP`` bakes
the entry point into a dynamic global, and numba refuses to cache a function
carrying one, so such a caller used to recompile in every process and emit
``NumbaWarning: Cannot cache compiled function ... as it uses dynamic globals``.
``lower_constant_derive_function_type`` (``numbox/utils/derive_wap.py``) instead
declares the ``jit_addr`` slot symbolically with ``context.declare_function`` and
links the proxied body's compile result in with ``add_linking_library``. The call
site then uses only that slot, so the ``c_addr`` and ``py_addr`` dynamic globals
are dead and get eliminated before numba scans the final module, and the caller
caches. This holds from numba 0.61 onward; on 0.60 ``.as_func`` is a plain
``CompileResultWAP`` and the caller stays uncacheable as before.

What such a caller caches is the proxied body's machine code, linked in. The
inline at work on this path is not the ``inline='always'`` the decorator puts on
the generated wrapper: that one is a numba-IR inline of the wrapper's single
``@intrinsic`` call, it fires only where a jitted caller calls the dispatcher,
and what it leaves behind is the alias declaration the static-linking avoidance
is built on. A constant reference never reaches the wrapper. ``.as_func`` carries
the *body's* compile result, ``lower_constant_derive_function_type`` links that
library into the caller, and LLVM then inlines the body itself for an ordinary
body size, so on this one path the static-linking avoidance does not apply. The
cached binary binds the body as it stood at compile time, while numba's freshness
stamp watches only the caller's own source file. **This is the constant-reference
route alone: a caller reaching the same binding through its dispatcher is healed
on load by the stale-alias guard described below, and needs no manual clearing.
For a caller holding the body as a compile-time constant, clearing the numba
cache after an edit to the proxied body is mandatory rather than merely
advisable.**

Which body it runs is not single-valued. A body small enough to inline leaves no
separate definition behind, and the caller keeps serving the old numbers in
every later process. A larger body is embedded in the caller's object as a weak
``linkonce_odr`` definition under a mangled name carrying numba's per-process
``v<uid>`` abi-tag, which numba's cache key does not cover. Which body that
caller reaches then turns on the name the proxied body carries in the loading
process, and that name is minted by whichever process last recompiled the body
and frozen in the body's own cache entry, so it need not track the loading
process's own counter. Where it differs from the embedded name, the caller runs
its embedded copy while the dispatcher path runs the new body; where it matches,
the caller binds to the definition the already-loaded body registered and both
handles return the new value while the caller's cached object still holds only
the old code. The first process to recompile the body after an edit fixes which
of the two every later process gets. Both outcomes are silent, so the two
handles disagreeing within one process is one symptom and not a dependable one:
in the non-inlined case they can agree while the cached caller runs a body it
never embedded.

The stale-alias guard described below cannot cover this, by construction. It
scans a cached object's undefined symbols for the ``numbox_pxy_`` prefix, and a
constant reference emits no such symbol, because the body is linked into the
object rather than referenced through the alias. The single
:class:`~numbox.core.proxy.proxy.StaleProxyCacheWarning` such a run emits names
the dispatcher-path caller that healed correctly, never the const-referencing
caller serving the stale value, and the run after that is silent. Reverting to a
numbox that mints a plain ``CompileResultWAP`` is not a remedy either: numba's
index key is callee-blind, so it loads the entry already on disk. Clearing the
cache is the remedy.

The blind spot covers more than a body edit. A ``proxy_if_available`` binding
that was present when the caller was cached but is absent in a later process is
not caught either: the const-referencing caller cache-hits and calls into a
binding that is not there, returning a value computed by the vanished binding
where the body needs no external symbol, and dying on a bare segfault with an
empty stderr where it does. A cold cache in the same process raises a clean
typing error instead, so warm and cold disagree. ``NUMBOX_PROXY_CACHE_STRICT``
does not catch it either, for the same reason the guard does not: there is no
alias to check. The deterministic arm of this, a body that needs no external
symbol so the warm run hands back the vanished binding's own number instead of
segfaulting, is pinned in ``test/core/test_proxy_cache_stale.py`` by
``test_a_const_reference_caller_calls_a_binding_that_disappeared``.

Guarding the use with ``hasattr(binding, "as_func")`` protects only in two
shapes: when the jitted caller's *definition* sits inside that ``if``, so the
absent process never defines it; and when ``.as_func`` is passed as a
function-type argument rather than referenced as a constant, because the address
is then unboxed per call and the absent process takes the fallback. The shape
that fails is ``binding.as_func if hasattr(...) else fallback`` assigned to a
module global while the ``cache=True`` caller is defined unconditionally, since
numba loads the cache entry before it types anything and the ``else`` branch is
never consulted. Capturing ``.as_func`` with no guard at all is safe: the
attribute is missing in the absent process, so the import fails loudly and the
cache is never reached.

The body-edit case is pinned in ``test/core/test_proxy_cache_stale.py``, by
``test_a_const_reference_caller_becomes_cacheable`` and
``test_a_const_reference_caller_serves_a_stale_body_after_an_edit``.

Alias content-addressing and cross-file callers
------------------------------------------------

``@proxy`` references the proxied body through a deterministic symbol alias
(``numbox_pxy_<name>_<hash>``) registered per process via
``llvmlite.binding.add_symbol``. The hash folds the body's content fingerprint
alongside ``module``, ``qualname`` and the signature, so two different bodies
that share one identity — factory-made same-qualname closures, an in-process
redefinition, or ``fork()`` twins on a shared cache — get distinct aliases
instead of colliding on one (a collision let the last ``add_symbol`` win and
silently rebound callers to the wrong body).

Separating them needs the fingerprint to see the captured value that differs, and
what it can see is bounded by what has an identity that means the same thing in
the next process. A ``numba.core.types.ExternalFunction`` and a ctypes pointer
reached by attribute access on a loaded library both do, and both are folded. A
ctypes pointer built from a raw address does not: two of them differ only by an
address that ASLR moves, so folding it would buy discrimination at the cost of the
process-stability the alias exists for.

Bodies in that last class are therefore allowed to collide, and the collision is
caught rather than hidden. The newcomer is published under a process-local name of
its own, so both bodies stay callable and correct in this process, and the shared
name is retired into the absent-alias set, so a warm caller of either one is
discarded and recompiled rather than served against whichever body happens to hold
the name today. Both lose cross-process caching, which is the price of a name that
does not identify what it names, and a
:class:`~numbox.core.proxy.proxy.AliasCollisionWarning` says so.

Reaching an alias twice is usually not a collision at all. The same body is
registered again whenever a module is reloaded or a factory is called twice with
equal arguments, and both registrations name the same code, so the alias is simply
re-pointed at the newly compiled address and nothing is retired. The two cases are
told apart by the values the fingerprint could not canonicalize: two bodies that
fingerprint alike are the same body except in those, so comparing them answers
whether one body was compiled twice or two bodies met on one name. That comparison
only has to hold inside the current process, which is why it may look at an address
where the fingerprint may not. The address alone is not the body, though: one symbol
reached through two ctypes prototypes is two calls that numba types and lowers
differently, so the prototype's shape is compared beside the address, and only the
two together say that one body was compiled twice. A name already retired by a
collision stays retired through such a re-registration: re-pointing it does not
make it identify a body again, because another process may still build the two
bodies the other way round, so nothing takes an alias back out of the absent-alias
set.

Retiring the two names is not the whole of the containment, because a third body can
capture one of them. A ``@proxy`` body that merely calls a colliding binding closes
over its dispatcher, and folding that dispatcher by the name it happens to hold would
make the derived body's own alias a function of the order this process built the pair
in, so two such derived bodies exchange names between one process and the next.
Neither derived name is retired, each being freshly minted and resolving normally, so
unlike the pair it wraps a derived body would keep the cross-process cache that has
just stopped being right, and the collision would escape one composition step out.
The walker therefore declines to fold a retired alias at all. A body that captured one
falls through to the ordinary walk, where a dispatcher has no canonical form, so it
meets the same comparison every un-canonicalizable value meets and its own alias is
retired in its turn.

One window stays open. A derived body fingerprinted *before* the collision that
retires its dependency folded a name that still identified a body, and it is already
published under a name of its own that nothing revisits. Closing it would mean
recording which aliases each fingerprint consumed and retiring dependents
transitively, so what is documented here is the bound: containment reaches a derived
body built after the collision, and the warning's promise that callers of both
recompile in every process holds for everything built from that point on.

Because the alias encodes the body, changing a proxied binding's **signature or
body renames its alias**. numba's cache key for a *caller* is callee-blind, so a
``cache=True`` caller in another file cache-hits unchanged after such a change
and references the old alias, which the new process never registers. Left
unhandled this is a hard crash rather than a miss: RuntimeDyld resolves an
object's externals in one batch, so the single missing name zeroes *every*
relocation in the object and the process dies inside the argument-unpacking
wrapper — a bare segfault with an empty stderr, or, once any later cached object
is loaded, an ``LLVM ERROR: Symbol not found`` abort.

**numbox heals this on load; no manual cache clearing is needed.** Importing any
``@proxy`` binding imports numbox, which installs a guard around numba's cache
``rebuild``. Before a cached object reaches the execution engine its undefined
``numbox_pxy_*`` symbols are checked against the registered aliases; an object
referencing one this process never registered is discarded and recompiled in
place, emitting a :class:`~numbox.core.proxy.proxy.StaleProxyCacheWarning` that
names the retired alias. ``@njit``, ``@vectorize``, ``@guvectorize`` and
``@cfunc`` callers all heal.

Set ``NUMBOX_PROXY_CACHE_STRICT`` (truthy: anything other than unset / ``0`` /
``false`` / ``no`` / ``off``, case-insensitive) to make the guard fail loud
instead of healing: a stale alias then raises
:class:`~numbox.core.proxy.proxy.StaleProxyCacheError` *before* the discard,
leaving the stale entry on disk to inspect, and a payload the guard cannot read
raises :class:`~numbox.core.proxy.proxy.UnvalidatedProxyCacheError` rather than
loading unchecked.

**First run after an upgrade.** A numbox release that changes the alias
fingerprint renames every shipped alias at once, so the first process after the
upgrade heals every warm caller that reaches a numbox binding through its alias —
a one-time burst of recompiles and warnings, after which the cache is warm again.
A caller that reached ``.as_func`` as a compile-time constant is not among them:
it carries no alias, so it goes on serving the pre-upgrade body on every run
until its own cache entry is cleared. To clear it by hand instead, the entries
are the ``.nbc`` / ``.nbi`` files in the ``__pycache__`` directory beside each
caller's own source (or under ``NUMBA_CACHE_DIR`` if set);
``~/.cache/numba`` holds only callers numba cannot anchor to a source file.

The one variant that leaves the alias unchanged — a ``proxy_if_available``
binding present when the caller was cached but absent on reload — would otherwise
resolve to a diagnostic trap (a cfunc registered under the alias whose
``RuntimeError`` numba swallows at the C boundary, returning zero); the guard
treats such an alias as stale too, so a caller that reaches the binding through
its alias gets the same clean typing error a cold cache gives. A caller that
reached ``.as_func`` as a compile-time constant emits no alias and is not covered;
see `Cache invalidation`_ above for what it gets instead.

Multi-decorator support
-----------------------

When a user stacks decorators above ``@proxy(sig)``,
``func.__code__.co_firstlineno`` is the topmost decorator line (Python
records a decorated function's first line as its outermost decorator).
The anchor lands the wrapper at that topmost decorator.
``findsource`` matches it directly because every decorator line begins
with ``@``. Verified for single-, double-, and triple-stack outer
decorators.

``@proxy`` itself must be the innermost decorator (closest to ``def``).
A wrapping decorator between ``@proxy`` and ``def`` would hand
``@proxy`` a wrapped function whose ``__code__`` lives in the wrapping
decorator's source file, and would also break numba's ability to
JIT-compile through the intermediate Python wrapper.

Modules
+++++++

numbox.core.proxy.proxy
-----------------------

.. automodule:: numbox.core.proxy.proxy
   :members:
   :show-inheritance:
   :undoc-members:
