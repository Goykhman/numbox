"""Register SQLite aggregate and window functions whose state is a numba structref.

:func:`register_aggregate` and :func:`register_window` take a state type and the functions that create, update and
finalize the state. They generate and compile the C callbacks that SQLite calls. The callbacks keep each group's
state alive between rows, release it at the end of the group, and convert an exception in one of the functions into
a SQL error. The caller writes only the logic of the state, as plain Python functions or jitted ones.
"""
import ctypes

from numba import cfunc, types
from numba.core.types import StructRef
from numba.extending import is_jitted

from numbox.core.bindings.sqlite.conn import sqlite3_errmsg
from numbox.core.bindings.sqlite.constants import SQLITE_DETERMINISTIC, SQLITE_OK, SQLITE_UTF8
from numbox.core.bindings.sqlite.udf import sqlite3_create_function_v2, sqlite3_create_window_function
from numbox.utils.cstrings import c_string
from numbox.utils.digest import digest
from numbox.utils.preprocessing import (
    _anchor_path,
    _anchored_or_uncached,
    _orphan_anchor_sweep,
)

# The generated code names these. Importing them here puts them in this module's __dict__, from which
# _compile_callbacks seeds the namespace of the generated code.
import numpy as np  # noqa: F401
from numba import carray, njit  # noqa: F401
from numbox.core.configurations import jit_options  # noqa: F401
from numbox.core.bindings.sqlite.result import sqlite3_result_error  # noqa: F401
from numbox.core.bindings.sqlite.udf import sqlite3_aggregate_context  # noqa: F401
from numbox.utils.lowlevel import _cast_int_to_void_p, get_unicode_data_p  # noqa: F401
from numbox.utils.meminfo import (  # noqa: F401
    borrow_structref,
    export_meminfo,
    release_meminfo,
)

__all__ = ["register_aggregate", "register_window"]

_ANCHOR_SUBDIR = "numbox-sqlite-udaf"
_orphan_anchor_sweep(_ANCHOR_SUBDIR)


# Templates of the generated functions. The code runs in a namespace seeded from this module's globals, with
# _state_type, the caller's functions (_init, _step, _finalize, _inverse, _value) and jit_options added. Each
# function compiles under @njit(**jit_options), numbox's jit options, so numba caches it across processes when caching
# is enabled. The caller's functions are globals of that namespace, so numba links their code into the generated
# functions, and the cache entry of a generated function holds that code as well.
#
# A group's state is handled as follows. xStep requests 8 bytes from sqlite3_aggregate_context, which SQLite allocates
# and zeroes at the group's first call, and uses them as one intp slot. If the slot holds 0, xStep calls _init and
# stores the state's MemInfo there with export_meminfo, which takes a reference. Each call borrows the state with
# borrow_structref. xValue reports the running result and keeps the state; only xFinal releases the reference, with
# release_meminfo. xFinal, xValue and xInverse request 0 bytes, so they receive 0 when no xStep ran for the group.
# When no xStep ran, or when _init failed in xStep, xFinal and xValue report the result of a new state from _init, and
# xInverse does nothing.
#
# Each call of a caller's function is wrapped in a try/except that reports the failure with sqlite3_result_error,
# which sets SQLITE_ERROR. Without it, the cfunc that SQLite calls would catch the exception at its boundary, print
# "Exception ignored" and return without reporting an error. The SQL statement would then succeed with a wrong result.
# With the try/except, control stays in the generated function after an exception, so numba runs the decrement of the
# borrowed state's reference count. An exception that left the function would skip that decrement and leak the state.
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
    """Return ``fns`` with every callback as a numba dispatcher; a plain Python function is wrapped with ``njit``.

    A jitted callable, such as an ``@njit`` or ``@proxy`` function, is returned unchanged. A value that is not callable
    raises ``TypeError``.
    """
    prepared = {}
    for role, fn in fns.items():
        if is_jitted(fn):
            prepared[role] = fn
        elif callable(fn):
            # A lazy njit, without a signature, is enough, because the call in the generated function fixes the
            # argument types. The callback needs no cache of its own: the cache entry of the generated function
            # holds the callback's code.
            prepared[role] = njit(fn)
        else:
            raise TypeError(
                "%s must be a callable (plain Python or @njit), got %r" % (role, fn))
    return prepared


def _compile_callbacks(stem, srcs, state_type, fns):
    """Write the generated functions to an anchor file and compile them, and return the namespace that holds them.

    ``srcs`` are the templates to generate. ``fns`` maps the names that the templates use, ``_init``, ``_step`` and
    the rest, to the caller's callables. The namespace holds ``_xstep_impl`` and the other compiled functions.
    """
    # repr(state_type) holds only the class's qualified name and the fields, so two StructRef classes of one qualified
    # name in different modules would share a digest. The subject of the digest therefore adds the defining module and
    # the qualified name. The digest does not cover an @overload_method attached to the type, and that includes a
    # method that make_structref attaches. make_structref puts a method's hash only into the struct's own generated
    # code, so an edit to the body of such a method does not change this digest.
    state_id = "%s.%s;%r" % (type(state_type).__module__, type(state_type).__qualname__, state_type)
    # The digest covers that subject, numba's and numbox's versions, numbox's jit options and the fingerprint of each
    # callback. A fingerprint covers the callback's code, constants, defaults, closure, the globals it reads and the
    # module attributes it reads that hold data or a dispatcher. An edit to any of these, a numeric literal included,
    # therefore compiles new code. If a callback reads a value without a fingerprint, such as the @intrinsic
    # _cast_int_to_void_p of the usual argv_pp idiom, digest falls back to a weaker fingerprint. It names the type of
    # such a value instead of the value and omits module attributes, so an edit to such a value or to a module
    # attribute keeps the anchor and the cached code. The utils page describes the case under "If a SQLite function
    # keeps returning old results". A fingerprint names its function's module, and that name is __main__ in every run
    # of a script, so callbacks and a state type defined in a script's __main__ get the same digest in each run.
    # test_xprocess_cache_no_growth and test_invalidation_on_literal_edit run that case in subprocesses.
    udaf_digest = digest(state_id, list(fns.values()))
    # The anchor's name ends in a hash of the code text, which starts with the digest, so a new digest writes a new
    # anchor and compiles new code.
    code_txt = "# udaf-digest: %s\n%s" % (udaf_digest, "".join(srcs))
    # The namespace starts from this module's __dict__, because it carries __name__, which numba needs to rebuild
    # the Environment of a function that it loads from the cache. _anchored_or_uncached returns numbox's jit options,
    # or the options with caching off, after a warning, when numba cannot cache code of the anchor's file.
    anchor = _anchor_path(_ANCHOR_SUBDIR, stem, code_txt)
    ns = {**globals(), "_state_type": state_type, **fns,
          "jit_options": _anchored_or_uncached(anchor, code_txt, jit_options)}
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
    """Register the aggregate SQL function ``name`` on the connection ``db``, with a state of ``state_type`` per group.

    For each group of rows, SQLite calls ``init`` once to create the state, ``step`` once for each row, and
    ``finalize`` once to write the result. Each of the three is a plain Python function, which numbox compiles with
    ``njit``, or a jitted one, such as an ``@njit`` or ``@proxy`` function. numbox generates the C callbacks that
    SQLite calls. They keep the group's state alive between rows and release it after ``finalize``.

    :param db: the connection's ``sqlite3 *`` handle, an ``intp``.
    :param name: the SQL name of the function, a ``str``, which numbox passes to SQLite as a C string.
    :param n_arg: the number of SQL arguments, or -1 for any number.
    :param state_type: the numba ``StructRef`` instance type of the state.
    :param init: ``init()`` returns a new state.
    :param step: ``step(state, ctx, argc, argv_pp)`` updates the state with one row. ``ctx`` is the
        ``sqlite3_context *``, ``argc`` the number of SQL arguments, and ``argv_pp`` the address of SQLite's array of
        pointers to the argument values, which the bindings of :mod:`~numbox.core.bindings.sqlite.value` read.
    :param finalize: ``finalize(state, ctx)`` writes the result with a binding of
        :mod:`~numbox.core.bindings.sqlite.result`. If no row reaches the function, as in an aggregate over an empty
        table, ``finalize`` receives a new state from ``init``.
    :param deterministic: if true, the function is registered with ``SQLITE_DETERMINISTIC``.

    If one of the functions raises, the SQL statement fails with an error such as "error in user step callback".
    numba compiles the generated code and the functions during this call, so numba raises a typing error in a function
    from this call. The generated functions compile under numbox's jit options, and numbox names their cache after a
    digest of the state type and of the functions. The state type and the functions may be defined in a script's
    ``__main__``. If a function reads a value without a fingerprint, such as numbox's ``_cast_int_to_void_p``, the
    digest misses some edits, as the section "If a SQLite function keeps returning old results" of
    :doc:`numbox.utils` explains. The digest also misses an edit to a method of the state type. After an edit that
    the digest misses, the registration can run code compiled before the edit.

    Returns ``None``, and the caller need not keep anything. numba keeps the compiled code of the callbacks for the
    life of the process, although it does not document that behavior. Raises ``TypeError`` when ``state_type`` is not
    a ``StructRef`` instance type or a function is not callable. Raises ``RuntimeError``, with SQLite's message, when
    SQLite refuses the registration.
    """
    _validate_state_type(state_type)
    fns = _prepare_callbacks(init=init, step=step, finalize=finalize)
    ns = _compile_callbacks(
        _stem("udaf_", name), [_XSTEP_SRC, _XFINAL_SRC], state_type,
        {"_init": fns["init"], "_step": fns["step"], "_finalize": fns["finalize"]})
    xstep_impl = ns["_xstep_impl"]
    xfinal_impl = ns["_xfinal_impl"]

    # SQLite calls these wrappers. Each closes over a generated function of this registration, so numba compiles it at
    # every registration, without a cache; only the generated function that it calls can be cached.
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
    """Register the window SQL function ``name`` on the connection ``db``, with a state of ``state_type`` per group.

    The parameters are those of :func:`register_aggregate`, with two more functions.
    ``inverse(state, ctx, argc, argv_pp)`` removes from the state a row that leaves the window; SQLite calls it only
    for a row that ``step`` added, so the state exists. ``value(state, ctx)`` writes the current result of the window,
    as ``finalize`` does, but the state stays alive for the rows that follow. Only the call of ``finalize`` releases
    the state. The errors, the caching and its limits, the return value and the exceptions are those of
    :func:`register_aggregate`.
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

    # The wrappers that SQLite calls compile at every registration, as in register_aggregate.
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
