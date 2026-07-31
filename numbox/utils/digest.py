"""Content-addressed digest for cache keys and anchor identifiers.

``digest`` produces a short, stable hash that invalidates when the subject's
``repr``, the user functions, the resolved ``jit_options``, or the numba/numbox
versions change. Plain Python functions are fingerprinted with the shared
closure/global-aware walker (``numbox.utils.fingerprint._fingerprint_function``)
-- so two callbacks with identical source but different captured closure-cell or
referenced-global values key distinctly, which a bare code-object hash would
miss. Callables with no canonical fingerprint (a partial, a builtin, a callable
object, or a function closing over an un-canonicalizable value) fall back to
cloudpickle of the object/code, which also captures bound state. The SQLite UDAF
registration anchors are one consumer; any content-addressed cache that mixes a
type/identifier with user callbacks can reuse it.
"""
import hashlib

from types import FunctionType

import numba
from numba.core.serialize import cloudpickle

import numbox
from numbox.core.configurations import jit_options
from numba.core.dispatcher import Dispatcher
from numba.np.ufunc.dufunc import DUFunc

from numbox.utils.fingerprint import (
    _Unfingerprintable, _canon_value, _fingerprint_function, _fingerprint_function_best_effort,
)


def digest(subject, fns):
    """Content hash that invalidates when ``subject`` (by ``repr``), the user
    functions (closure/global/const-sensitive via the shared fingerprint walker,
    with a cloudpickle fallback), the resolved ``jit_options``, or the
    numba/numbox versions change."""
    h = hashlib.sha256()
    h.update(repr(subject).encode("utf-8"))
    h.update(numba.__version__.encode("utf-8"))
    # numbox.__version__ is "" upstream (the package version derives from it via
    # pyproject's dynamic attr), so this fold is currently inert; it is kept so
    # digests auto-invalidate should numbox ever set a real __version__.
    h.update((numbox.__version__ or "").encode("utf-8"))
    # fold the resolved numbox-wide jit_options so flipping NUMBOX_JIT_OPTIONS
    # (e.g. cache off) re-keys the digest; numba's own cache also keys on flags.
    h.update(repr(sorted(jit_options.items())).encode("utf-8"))
    for fn in fns:
        if isinstance(fn, (Dispatcher, DUFunc)):
            # Route jitted callables through the shared value canonicalizer,
            # which folds targetoptions (and honours a @proxy's content-addressed
            # alias). The py_func shortcut below sees only the Python body, so
            # dispatchers over one body compiled with different jit flags --
            # nogil, fastmath, error_model, boundscheck -- all collided.
            try:
                h.update(_canon_value(fn, set()).encode("utf-8"))
                continue
            except (_Unfingerprintable, RecursionError):
                # Body not canonicalizable; fold the flags on their own so they
                # still re-key, then fall through to the best-effort walker.
                h.update(repr(sorted((getattr(fn, "targetoptions", None) or {}).items())).encode("utf-8"))
        py = getattr(fn, "py_func", fn)
        if isinstance(py, FunctionType):
            try:
                h.update(_fingerprint_function(py, set()).encode("utf-8"))
            except (_Unfingerprintable, RecursionError):
                # An un-canonicalizable closure/global aborts the strict walker;
                # the best-effort walker still captures the constants, closure
                # cells, defaults and globals it can (substituting opaque
                # placeholders for the rest, and sorting sets so it is
                # PYTHONHASHSEED-stable). A bare __code__ cloudpickle here dropped
                # exactly the closure/default state that forced the fallback and
                # leaked str-set iteration order.
                h.update(_fingerprint_function_best_effort(py).encode("utf-8"))
            continue
        # Codeless callable (partial / builtin / callable object): cloudpickle the
        # object to capture bound state. (Only these remain on the cloudpickle
        # path, so its uuid4 nondeterminism for __main__ objects is confined here.)
        h.update(cloudpickle.dumps(py))
    return h.hexdigest()[:16]
