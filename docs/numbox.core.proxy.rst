numbox.core.proxy
=================

Overview
++++++++

Implementation of :func:`numbox.core.proxy.proxy` decorator that swaps definition of a jit-compiled
function in-place for a declaration (while delegating the actual implementation
to a different function that is only accessible indirectly). As a result, statically linking in libraries
corresponding to proxy-jitted functions called from other jitted functions will
only paste a declaration rather than the entire LLVM IR code.

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
wrapper-template changes as developer-managed (clear
``~/.cache/numba/`` when shipping a template change to numbox).

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
