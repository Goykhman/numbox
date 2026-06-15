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
from numbox.utils.fingerprint import _Unfingerprintable, _fingerprint_function


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
        py = getattr(fn, "py_func", fn)
        if isinstance(py, FunctionType):
            try:
                h.update(_fingerprint_function(py, set()).encode("utf-8"))
                continue
            except (_Unfingerprintable, RecursionError):
                pass  # un-canonicalizable closure/global -> fall back below
        # codeless callable (partial/builtin/callable object) or un-fingerprintable
        # function: cloudpickle the object (captures bound state) or its code object.
        h.update(cloudpickle.dumps(getattr(py, "__code__", py)))
    return h.hexdigest()[:16]
