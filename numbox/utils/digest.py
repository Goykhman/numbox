"""A short hash that names generated code after the subject it serves and the user functions it calls.

:func:`digest` hashes the ``repr`` of a subject, such as a type, with the fingerprints of a list of functions. It also
folds numbox's default jit options and the versions of numba and numbox. Two functions with the same source but
different closure values or globals have different hashes, whereas a hash of the bare code object would be the same for
both. numbox names the anchors of the SQLite aggregate, window and table-valued functions with the hash. Any cache that
combines a type or an identifier with user callbacks can use it the same way.
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
    """Return a hash of 16 hexadecimal digits that changes when ``subject`` or one of the functions ``fns`` changes.

    The hash covers ``repr(subject)``, numba's version, numbox's version, numbox's default jit options and each entry
    of ``fns``. numbox's version is the empty string in a checkout or an editable install, so in those cases the hash
    does not change with the version. A released package holds its version, and the hash folds it. Each entry of ``fns``
    is folded as follows:

    - a dispatcher or a ``@vectorize`` function: the fingerprint of its Python function and the options it compiles
      under. If the Python function has no fingerprint, the options are folded with the weaker fingerprint below for a
      dispatcher, and with a cloudpickle of the object for a ``@vectorize`` function;
    - a Python function: its fingerprint, or the weaker fingerprint if the function reads a value that has none;
    - any other callable, such as a ``functools.partial``, a builtin or a callable object: a cloudpickle of it.

    For a Python function or a dispatcher, the hash is the same in every process. The weaker fingerprint puts the name
    of a value's type in place of each value without a fingerprint, and it leaves out module attributes, so an edit to
    such a value or attribute leaves the hash unchanged. A cloudpickle can differ in every process, as it does for an
    object whose class is defined in the script that runs.
    """
    h = hashlib.sha256()
    h.update(repr(subject).encode("utf-8"))
    h.update(numba.__version__.encode("utf-8"))
    # numbox.__version__ is "" in a checkout and in an editable install, because the packaging takes the version from
    # this attribute and the repository leaves it empty. The release workflow writes the version into
    # numbox/__init__.py before it builds the package, so a released package folds its version here.
    h.update((numbox.__version__ or "").encode("utf-8"))
    # A change of numbox's default jit options, such as NUMBOX_JIT_OPTIONS turning the cache off, changes the digest
    # and so the name of the anchor. numba's own cache key leaves out the jit options; numbox adds them to the key of
    # the code it keys (numbox.core.cache_key).
    h.update(repr(sorted(jit_options.items())).encode("utf-8"))
    for fn in fns:
        if isinstance(fn, (Dispatcher, DUFunc)):
            # _canon_value folds the fingerprint of the Python function with the options that the function compiles
            # under; for a @proxy function it folds the alias, which covers both. The Python function alone would give
            # one hash for two dispatchers of one body compiled with different options, such as nogil, fastmath,
            # error_model or boundscheck.
            try:
                h.update(_canon_value(fn, set()).encode("utf-8"))
                continue
            except (_Unfingerprintable, RecursionError):
                # The body has no fingerprint. The options are folded separately, so that a change of an option still
                # changes the digest. The weaker fingerprint below is then computed for a dispatcher's Python
                # function, and a @vectorize function, which has no py_func, is pickled.
                h.update(repr(sorted((getattr(fn, "targetoptions", None) or {}).items())).encode("utf-8"))
        py = getattr(fn, "py_func", fn)
        if isinstance(py, FunctionType):
            try:
                h.update(_fingerprint_function(py, set()).encode("utf-8"))
            except (_Unfingerprintable, RecursionError):
                # A value without a fingerprint makes _fingerprint_function raise. The best-effort fingerprint still
                # folds the constants, closure values, defaults and globals that have one, and it sorts the elements
                # of sets, so the result does not depend on PYTHONHASHSEED. It puts the name of the type in place of
                # each value without a fingerprint, and it omits the module attributes that the function reads.
                # An edit to such a value, or to such an attribute, therefore leaves the digest unchanged, and a
                # SQLite registration named after the digest loads the code compiled before the edit.
                h.update(_fingerprint_function_best_effort(py).encode("utf-8"))
            continue
        # A callable without code, such as a functools.partial, a builtin or a callable object, is hashed as a
        # cloudpickle of the object, which holds its bound state. cloudpickle assigns a random uuid4 identifier to a
        # class defined in __main__, so the pickle of such an object, and the digest, differ in every process. Only
        # these callables, and a @vectorize function without a fingerprint, take that path.
        h.update(cloudpickle.dumps(py))
    return h.hexdigest()[:16]
