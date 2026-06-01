"""Higher-level registration helpers for structref-backed SQLite UDAFs.

``register_aggregate`` / ``register_window`` generate the SQLite callback
functions (xStep/xInverse/xValue/xFinal) that perform the per-group state
lifecycle -- ``sqlite3_aggregate_context`` allocation + NULL guard, the single
intp slot, init-on-first-step via ``export_meminfo``, ``borrow_structref``, and
the release-in-xFinal-but-NOT-xValue rule -- so callers write only their
``init``/``step``/``finalize`` (and ``inverse``/``value`` for windows) state
logic, as plain Python or already-jitted (``@njit``/``@proxy``) functions.

**Exception handling.** Each generated callback wraps the user
step/inverse/value/finalize call in a ``try``/``except`` that reports a
descriptive error via ``sqlite3_result_error`` (e.g. "error in user step
callback", with code ``SQLITE_ERROR``) when the user callback raises.
A numba ``@cfunc`` otherwise SWALLOWS the exception (it prints "Exception
ignored" and returns the zero default without unwinding into SQLite), which would
be a silent wrong result; the in-body catch also lets numba run the borrowed
state's reference-count decrement, which the unwind would otherwise skip -- a
per-group meminfo leak. Only ``xFinal`` releases the slot.

Mechanism: per-UDAF callback source is generated with the state type and the user functions
baked in as module globals (so the calls inline), written to a content-addressed
anchor file under numba's cache dir (reusing numbox.utils.preprocessing), and the
impls (``@njit(**jit_options)`` -- the numbox-wide config, default ``cache=True``)
cache across processes. The anchor's content hash
folds a cloudpickle of the user functions' code objects so editing a body --
including a numeric literal -- invalidates correctly.

**Caller requirements.** Callbacks may be plain Python functions or already-jitted
callables (``@njit``/``@proxy``); plain ones are compiled with ``njit`` (see
``_prepare_callbacks``). The generated impls (``@njit(**jit_options)``, default
``cache=True``) cache and invalidate correctly even when the state-type class and
the callbacks are defined in ``__main__`` -- the anchor key is content-addressed
on a ``cloudpickle`` of each callback's ``__code__`` (plus ``repr(state_type)``,
the resolved ``jit_options``, and the numba/numbox versions), never on
``__module__`` (see ``_digest``); the
subprocess tests ``test_xprocess_cache_no_growth`` /
``test_invalidation_on_literal_edit`` exercise this ``__main__`` path. The one
case that does require a stable ``__module__`` is a *caller-side*
``@njit(cache=True)`` callback -- numba refuses to cache functions defined in
``__main__`` -- so for deployments where callbacks are themselves cached, define
the state-type class and callbacks in an importable module.
"""
import ctypes
import hashlib

import numba
from numba import cfunc, types
from numba.core.serialize import cloudpickle
from numba.core.types import StructRef
from numba.extending import is_jitted

import numbox
from numbox.core.configurations import jit_options
from numbox.core.bindings._sqlite_conn import sqlite3_errmsg
from numbox.core.bindings._sqlite_constants import (
    SQLITE_DETERMINISTIC,
    SQLITE_OK,
    SQLITE_UTF8,
)
from numbox.core.bindings._sqlite_udf import (
    sqlite3_create_function_v2,
    sqlite3_create_window_function,
)
from numbox.utils.cstrings import c_string
from numbox.utils.preprocessing import (
    _anchor_path,
    _materialize_anchor,
    _orphan_anchor_sweep,
)

# These names are referenced by the GENERATED anchor source; importing them here
# puts them in this module's __dict__, which seeds the exec namespace below.
import numpy as np  # noqa: F401
from numba import carray, njit  # noqa: F401
from numbox.core.bindings._sqlite_result import sqlite3_result_error  # noqa: F401
from numbox.core.bindings._sqlite_udf import sqlite3_aggregate_context  # noqa: F401
from numbox.utils.lowlevel import _cast_int_to_void_p, get_unicode_data_p  # noqa: F401
from numbox.utils.meminfo import (  # noqa: F401
    borrow_structref,
    export_meminfo,
    release_meminfo,
)

__all__ = ["register_aggregate", "register_window"]

_ANCHOR_SUBDIR = "numbox-sqlite-udaf"
_orphan_anchor_sweep(_ANCHOR_SUBDIR)


# Generated-source templates. Baked global names: _state_type, _init, _step,
# _finalize, _inverse, _value, jit_options. Each impl is @njit(**jit_options)
# (numbox-wide config, default cache=True) so it caches cross-process; the user
# fns inline because they are module globals here.
_XSTEP_SRC = '''
@njit(**jit_options)
def _xstep_impl(ctx, argc, argv_pp):
    agg = sqlite3_aggregate_context(ctx, 8)
    if agg == 0:
        return
    slot = carray(_cast_int_to_void_p(agg), (1,), dtype=np.intp)
    if slot[0] == 0:
        try:
            slot[0] = export_meminfo(_init())
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user init callback"), -1)
            return
    try:
        _step(borrow_structref(_state_type, slot[0]), ctx, argc, argv_pp)
    except Exception:
        sqlite3_result_error(ctx, get_unicode_data_p("error in user step callback"), -1)
'''

_XFINAL_SRC = '''
@njit(**jit_options)
def _xfinal_impl(ctx):
    agg = sqlite3_aggregate_context(ctx, 0)
    if agg == 0:
        try:
            _state = _init()
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user init callback"), -1)
            return
        try:
            _finalize(_state, ctx)
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user finalize callback"), -1)
        return
    slot = carray(_cast_int_to_void_p(agg), (1,), dtype=np.intp)
    if slot[0] == 0:
        try:
            _state = _init()
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user init callback"), -1)
            return
        try:
            _finalize(_state, ctx)
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user finalize callback"), -1)
        return
    try:
        _finalize(borrow_structref(_state_type, slot[0]), ctx)
    except Exception:
        sqlite3_result_error(ctx, get_unicode_data_p("error in user finalize callback"), -1)
    release_meminfo(slot[0])
'''

_XINVERSE_SRC = '''
@njit(**jit_options)
def _xinverse_impl(ctx, argc, argv_pp):
    agg = sqlite3_aggregate_context(ctx, 0)
    if agg == 0:
        return
    slot = carray(_cast_int_to_void_p(agg), (1,), dtype=np.intp)
    if slot[0] == 0:
        return
    try:
        _inverse(borrow_structref(_state_type, slot[0]), ctx, argc, argv_pp)
    except Exception:
        sqlite3_result_error(ctx, get_unicode_data_p("error in user inverse callback"), -1)
'''

_XVALUE_SRC = '''
@njit(**jit_options)
def _xvalue_impl(ctx):
    agg = sqlite3_aggregate_context(ctx, 0)
    if agg == 0:
        try:
            _state = _init()
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user init callback"), -1)
            return
        try:
            _value(_state, ctx)
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user value callback"), -1)
        return
    slot = carray(_cast_int_to_void_p(agg), (1,), dtype=np.intp)
    if slot[0] == 0:
        try:
            _state = _init()
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user init callback"), -1)
            return
        try:
            _value(_state, ctx)
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user value callback"), -1)
        return
    try:
        _value(borrow_structref(_state_type, slot[0]), ctx)
    except Exception:
        sqlite3_result_error(ctx, get_unicode_data_p("error in user value callback"), -1)
'''


def _validate_state_type(state_type):
    if not isinstance(state_type, StructRef):
        raise TypeError(
            "state_type must be a numba StructRef instance "
            "(e.g. MyStateType([('x', int64)])), got %r" % (state_type,))


def _prepare_callbacks(**fns):
    """Return ``fns`` with each callback ensured numba-compiled: an already
    jitted callable (``@njit``, ``@proxy``, ...) passes through; a plain Python
    function is wrapped with ``njit``. The callbacks are inlined into the
    generated ``@njit(cache=True)`` impls, so a lazy ``njit`` (no explicit
    signature) suffices -- the impl drives specialization -- and the
    content-addressed anchor (keyed on ``py_func.__code__``) provides the
    cross-process caching, so callbacks need no ``cache=True`` of their own."""
    prepared = {}
    for role, fn in fns.items():
        if is_jitted(fn):
            prepared[role] = fn
        elif callable(fn):
            prepared[role] = njit(fn)
        else:
            raise TypeError(
                "%s must be a callable (plain Python or @njit), got %r" % (role, fn))
    return prepared


def _digest(state_type, fns):
    """Content hash that invalidates when the state type, the user functions
    (co_consts-sensitive, via cloudpickle of the code object -- NOT bare
    co_code), the resolved ``jit_options``, or the numba/numbox versions change."""
    h = hashlib.sha256()
    h.update(repr(state_type).encode("utf-8"))
    h.update(numba.__version__.encode("utf-8"))
    # numbox.__version__ is "" upstream (the package version derives from it via
    # pyproject's dynamic attr), so this fold is currently inert; it is kept so
    # digests auto-invalidate should numbox ever set a real __version__. The
    # numba version above and the code-object hash below do the real
    # invalidating work (proven by test_invalidation_on_literal_edit).
    h.update((numbox.__version__ or "").encode("utf-8"))
    # fold the resolved numbox-wide jit_options so flipping NUMBOX_JIT_OPTIONS
    # (e.g. cache off) re-keys the anchor; numba's own cache also keys on flags.
    h.update(repr(sorted(jit_options.items())).encode("utf-8"))
    for fn in fns:
        py = getattr(fn, "py_func", fn)
        h.update(cloudpickle.dumps(py.__code__))
    return h.hexdigest()[:16]


def _compile_callbacks(stem, srcs, state_type, fns):
    """Generate + content-address-anchor + exec the @njit(cache=True) impls.

    ``fns`` maps generated global names (``_init``, ``_step``, ...) to user
    callables. Returns the exec namespace (contains ``_xstep_impl`` etc.)."""
    digest = _digest(state_type, list(fns.values()))
    code_txt = "# udaf-digest: %s\n%s" % (digest, "".join(srcs))
    # globals() is this module's __dict__; it carries __name__, which numba's
    # warm-cache Environment rebuild requires when reloading the cached impls.
    ns = {**globals(), "_state_type": state_type, **fns}
    anchor = _anchor_path(_ANCHOR_SUBDIR, stem, code_txt)
    _materialize_anchor(anchor, code_txt)
    code = compile(code_txt, str(anchor), mode="exec")
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns


def _raise_rc(db, name, rc):
    msg_p = sqlite3_errmsg(db)
    detail = ""
    if msg_p:
        detail = ": " + ctypes.cast(
            msg_p, ctypes.c_char_p).value.decode("utf-8", "replace")
    raise RuntimeError(
        "sqlite3 UDAF registration failed for %r (rc=%d)%s" % (name, rc, detail))


def _stem(prefix, name):
    return prefix + "".join(c if c.isalnum() else "_" for c in name)


def register_aggregate(db, name, n_arg, state_type, init, step, finalize,
                       *, deterministic=False):
    """Register a structref-backed aggregate UDAF.

    Callbacks may be plain Python or already-jitted (``@njit``/``@proxy``); plain
    functions are compiled with ``njit``.

    :param db: connection pointer (intp), as returned by ``sqlite3_open``.
    :param name: SQL function name (str); the C-string lifetime is handled here.
    :param n_arg: argument count, or -1 for variadic.
    :param state_type: the numba structref *instance* type for per-group state.
    :param init: ``() -> state`` returning a fresh state.
    :param step: ``(state, ctx, argc, argv_pp)`` updating state.
    :param finalize: ``(state, ctx)`` writing the result.
    :param deterministic: OR-in ``SQLITE_DETERMINISTIC``.

    Returns ``None``; the generated callbacks need not be retained -- numba keeps
    the compiled cfunc code and dispatchers resident for the process lifetime.
    """
    _validate_state_type(state_type)
    fns = _prepare_callbacks(init=init, step=step, finalize=finalize)
    ns = _compile_callbacks(
        _stem("udaf_", name), [_XSTEP_SRC, _XFINAL_SRC], state_type,
        {"_init": fns["init"], "_step": fns["step"], "_finalize": fns["finalize"]})
    xstep_impl = ns["_xstep_impl"]
    xfinal_impl = ns["_xfinal_impl"]

    @cfunc(types.void(types.intp, types.int32, types.intp))
    def step_cb(ctx, argc, argv):
        xstep_impl(ctx, argc, argv)

    @cfunc(types.void(types.intp))
    def final_cb(ctx):
        xfinal_impl(ctx)

    flags = SQLITE_UTF8 | (SQLITE_DETERMINISTIC if deterministic else 0)
    with c_string(name) as name_p:
        rc = sqlite3_create_function_v2(
            db, name_p, n_arg, flags, 0, 0,
            step_cb.address, final_cb.address, 0)
    if rc != SQLITE_OK:
        _raise_rc(db, name, rc)


def register_window(db, name, n_arg, state_type, init, step, inverse, value,
                    finalize, *, deterministic=False):
    """Register a structref-backed window UDAF.

    Same as :func:`register_aggregate` plus ``inverse(state, ctx, argc,
    argv_pp)`` (un-applies a row; state already exists) and ``value(state,
    ctx)`` (emits the running result WITHOUT releasing). Only ``xFinal``
    releases the meminfo.
    """
    _validate_state_type(state_type)
    fns = _prepare_callbacks(init=init, step=step, inverse=inverse, value=value,
                             finalize=finalize)
    ns = _compile_callbacks(
        _stem("wudaf_", name),
        [_XSTEP_SRC, _XINVERSE_SRC, _XVALUE_SRC, _XFINAL_SRC], state_type,
        {"_init": fns["init"], "_step": fns["step"], "_inverse": fns["inverse"],
         "_value": fns["value"], "_finalize": fns["finalize"]})
    xstep_impl = ns["_xstep_impl"]
    xinverse_impl = ns["_xinverse_impl"]
    xvalue_impl = ns["_xvalue_impl"]
    xfinal_impl = ns["_xfinal_impl"]

    @cfunc(types.void(types.intp, types.int32, types.intp))
    def step_cb(ctx, argc, argv):
        xstep_impl(ctx, argc, argv)

    @cfunc(types.void(types.intp, types.int32, types.intp))
    def inverse_cb(ctx, argc, argv):
        xinverse_impl(ctx, argc, argv)

    @cfunc(types.void(types.intp))
    def value_cb(ctx):
        xvalue_impl(ctx)

    @cfunc(types.void(types.intp))
    def final_cb(ctx):
        xfinal_impl(ctx)

    flags = SQLITE_UTF8 | (SQLITE_DETERMINISTIC if deterministic else 0)
    with c_string(name) as name_p:
        rc = sqlite3_create_window_function(
            db, name_p, n_arg, flags, 0,
            step_cb.address, final_cb.address,
            value_cb.address, inverse_cb.address, 0)
    if rc != SQLITE_OK:
        _raise_rc(db, name, rc)
