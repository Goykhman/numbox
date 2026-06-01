"""Content-addressed digest for cache keys and anchor identifiers.

``digest`` produces a short, stable hash that invalidates when the subject's
``repr``, the user functions (``co_consts``-sensitive, via cloudpickle of the
code object -- NOT bare ``co_code``), the resolved ``jit_options``, or the
numba/numbox versions change. The SQLite UDAF registration anchors are one
consumer; any content-addressed cache that mixes a type/identifier with user
callbacks can reuse it.
"""
import hashlib

import numba
from numba.core.serialize import cloudpickle

import numbox
from numbox.core.configurations import jit_options


def digest(subject, fns):
    """Content hash that invalidates when ``subject`` (by ``repr``), the user
    functions (``co_consts``-sensitive, via cloudpickle of the code object --
    NOT bare ``co_code``), the resolved ``jit_options``, or the numba/numbox
    versions change."""
    h = hashlib.sha256()
    h.update(repr(subject).encode("utf-8"))
    h.update(numba.__version__.encode("utf-8"))
    # numbox.__version__ is "" upstream (the package version derives from it via
    # pyproject's dynamic attr), so this fold is currently inert; it is kept so
    # digests auto-invalidate should numbox ever set a real __version__. The
    # numba version above and the code-object hash below do the real
    # invalidating work (proven by test_invalidation_on_literal_edit).
    h.update((numbox.__version__ or "").encode("utf-8"))
    # fold the resolved numbox-wide jit_options so flipping NUMBOX_JIT_OPTIONS
    # (e.g. cache off) re-keys the digest; numba's own cache also keys on flags.
    h.update(repr(sorted(jit_options.items())).encode("utf-8"))
    for fn in fns:
        py = getattr(fn, "py_func", fn)
        code = getattr(py, "__code__", py)
        h.update(cloudpickle.dumps(code))
    return h.hexdigest()[:16]
