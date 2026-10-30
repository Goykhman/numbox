"""Tests of the bindings of :mod:`~numbox.core.bindings.sqlite.udf`, with callbacks written by hand.

The tests register a scalar function, an aggregate whose state is a structref and a window function whose state is a
NumPy array, and they read back the user data pointer and the connection handle in a callback. A test compares numba's
counts of MemInfo allocations and frees, to check that the aggregate and the window function release their states. The
helpers of udf_helpers, which generate such callbacks, have their tests in test_sqlite_udf_helpers.py.
"""
from ctypes import addressof, c_int64

import numpy as np
from numba import carray, cfunc, njit, types
from numba.core import types as nb_types
from numba.experimental import structref

from numbox.core.bindings.sqlite.constants import SQLITE_NULL, SQLITE_OK, SQLITE_UTF8
from numbox.core.bindings.sqlite.udf import (
    sqlite3_aggregate_context, sqlite3_context_db_handle, sqlite3_create_function_v2,
    sqlite3_create_window_function, sqlite3_user_data,
)
from numbox.core.bindings.sqlite.conn import sqlite3_close, sqlite3_open
from numbox.core.bindings.sqlite.exec import sqlite3_exec
from numbox.core.bindings.sqlite.result import sqlite3_result_int, sqlite3_result_int64, sqlite3_result_null
from numbox.core.bindings.sqlite.value import sqlite3_value_int, sqlite3_value_int64, sqlite3_value_type
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import _cast_int_to_void_p
from numbox.utils.meminfo import _incref_meminfo, borrow_structref, export_meminfo, release_meminfo, structref_meminfo


def _open_memory():
    db_p = c_int64(0)
    with c_string(":memory:") as name_p:
        rc = sqlite3_open(name_p, addressof(db_p))
    assert rc == SQLITE_OK
    return db_p.value


# ---------------------------------------------------------------------------
# The structref type of the aggregate's state, with one int64 field, total
# ---------------------------------------------------------------------------

@structref.register
class SumStateType(nb_types.StructRef):
    pass


class SumState(structref.StructRefProxy):
    def __new__(cls, total):
        return structref.StructRefProxy.__new__(cls, total)


structref.define_proxy(SumState, SumStateType, ["total"])
sum_state_type = SumStateType([("total", nb_types.int64)])


# ---------------------------------------------------------------------------
# Callbacks of the scalar functions
# ---------------------------------------------------------------------------

@njit
def _double_impl(ctx, argc, argv_pp):
    ud = sqlite3_user_data(ctx)
    out = carray(_cast_int_to_void_p(ud), (1,), dtype=np.int64)
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    val = sqlite3_value_int(args[0])
    result = val * 2
    out[0] = result
    sqlite3_result_int(ctx, result)


_double_cb = cfunc(types.void(types.intp, types.int32, types.intp))(_double_impl)


@njit
def _null_check_impl(ctx, argc, argv_pp):
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    if sqlite3_value_type(args[0]) == SQLITE_NULL:
        sqlite3_result_null(ctx)
    else:
        sqlite3_result_int(ctx, sqlite3_value_int(args[0]))


_null_check_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_null_check_impl)


# ---------------------------------------------------------------------------
# The probe, a scalar function that writes its first argument into the int64 at its user data
# ---------------------------------------------------------------------------

@njit
def _capture_impl(ctx, argc, argv_pp):
    ud = sqlite3_user_data(ctx)
    out = carray(_cast_int_to_void_p(ud), (1,), dtype=np.int64)
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    out[0] = sqlite3_value_int64(args[0])
    sqlite3_result_int(ctx, 0)


_capture_cb = cfunc(types.void(types.intp, types.int32, types.intp))(_capture_impl)


# ---------------------------------------------------------------------------
# Callbacks of the aggregate, whose state is a structref
# ---------------------------------------------------------------------------

@njit
def _sum_step_impl(ctx, argc, argv_pp):
    agg_ptr = sqlite3_aggregate_context(ctx, 8)
    if agg_ptr == 0:  # A result of 0 means a failed allocation; a carray at address 0 would crash the process.
        return
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    val = sqlite3_value_int64(args[0])
    slot = carray(_cast_int_to_void_p(agg_ptr), (1,), dtype=np.intp)
    if slot[0] == 0:
        s = SumState(np.int64(0))
        slot[0] = export_meminfo(s)
    state = borrow_structref(sum_state_type, slot[0])
    state.total += val


_sum_step_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_sum_step_impl)


@njit
def _sum_final_impl(ctx):
    agg_ptr = sqlite3_aggregate_context(ctx, 0)
    if agg_ptr == 0:
        sqlite3_result_int64(ctx, np.int64(0))
        return
    slot = carray(_cast_int_to_void_p(agg_ptr), (1,), dtype=np.intp)
    if slot[0] == 0:
        sqlite3_result_int64(ctx, np.int64(0))
        return
    state = borrow_structref(sum_state_type, slot[0])
    sqlite3_result_int64(ctx, state.total)
    release_meminfo(slot[0])


_sum_final_cb = cfunc(types.void(types.intp))(_sum_final_impl)


# ---------------------------------------------------------------------------
# Callbacks of the window function, whose state is a NumPy array
#
# The window function keeps its state in a NumPy array instead of the structref of the aggregate above. The state is a
# 1-element int64 array, the running total, which the step callback allocates with np.zeros. The aggregate context holds
# 16 bytes, two intp slots: the address of the array's MemInfo and the address of its data. The step callback takes a
# reference with _incref_meminfo, so that the array stays alive after the callback returns, and xFinal releases it with
# release_meminfo. The other callbacks read the total through a carray at the data address and build no structref. numba
# 0.65 and earlier run a pass that removes the reference count changes of a function that seems not to need them. The
# pass does not run on the step callback, because the callback calls np.zeros, which returns an array, and because
# _incref_meminfo marks the function, as its docstring explains. numba 0.66 removed the pass. numbduck's tests keep the
# state of a DuckDB aggregate in the same form.
# ---------------------------------------------------------------------------

_WSUM_SLOTS = 2  # slot[0] holds the MemInfo address, and slot[1] holds the data address.


@njit
def _wsum_step_impl(ctx, argc, argv_pp):
    agg_ptr = sqlite3_aggregate_context(ctx, 16)
    if agg_ptr == 0:  # A result of 0 means a failed allocation; a carray at address 0 would crash the process.
        return
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    val = sqlite3_value_int64(args[0])
    slot = carray(_cast_int_to_void_p(agg_ptr), (_WSUM_SLOTS,), dtype=np.intp)
    if slot[0] == 0:
        payload = np.zeros(1, dtype=np.int64)
        meminfo_p, data_p = structref_meminfo(payload)
        _incref_meminfo(meminfo_p)
        slot[0] = meminfo_p
        slot[1] = data_p
    state = carray(_cast_int_to_void_p(slot[1]), (1,), dtype=np.int64)
    state[0] += val


_wsum_step_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_wsum_step_impl)


@njit
def _wsum_inverse_impl(ctx, argc, argv_pp):
    agg_ptr = sqlite3_aggregate_context(ctx, 16)
    if agg_ptr == 0:  # A result of 0 means a failed allocation; a carray at address 0 would crash the process.
        return
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    val = sqlite3_value_int64(args[0])
    slot = carray(_cast_int_to_void_p(agg_ptr), (_WSUM_SLOTS,), dtype=np.intp)
    state = carray(_cast_int_to_void_p(slot[1]), (1,), dtype=np.int64)
    state[0] -= val


_wsum_inverse_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_wsum_inverse_impl)


@njit
def _wsum_value_impl(ctx):
    agg_ptr = sqlite3_aggregate_context(ctx, 0)
    if agg_ptr == 0:
        sqlite3_result_int64(ctx, np.int64(0))
        return
    slot = carray(_cast_int_to_void_p(agg_ptr), (_WSUM_SLOTS,), dtype=np.intp)
    if slot[0] == 0:
        sqlite3_result_int64(ctx, np.int64(0))
        return
    state = carray(_cast_int_to_void_p(slot[1]), (1,), dtype=np.int64)
    sqlite3_result_int64(ctx, state[0])


_wsum_value_cb = cfunc(types.void(types.intp))(_wsum_value_impl)


@njit
def _wsum_final_impl(ctx):
    agg_ptr = sqlite3_aggregate_context(ctx, 0)
    if agg_ptr == 0:
        sqlite3_result_int64(ctx, np.int64(0))
        return
    slot = carray(_cast_int_to_void_p(agg_ptr), (_WSUM_SLOTS,), dtype=np.intp)
    if slot[0] == 0:
        sqlite3_result_int64(ctx, np.int64(0))
        return
    state = carray(_cast_int_to_void_p(slot[1]), (1,), dtype=np.int64)
    sqlite3_result_int64(ctx, state[0])
    release_meminfo(slot[0])


_wsum_final_cb = cfunc(types.void(types.intp))(_wsum_final_impl)


# ---------------------------------------------------------------------------
# Callbacks that read the user data and the connection handle of their context
# ---------------------------------------------------------------------------

@njit
def _user_data_probe_impl(ctx, argc, argv_pp):
    ud = sqlite3_user_data(ctx)
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    target_p = sqlite3_value_int64(args[0])
    result = np.int64(1) if ud == target_p else np.int64(0)
    sqlite3_result_int64(ctx, result)


_user_data_probe_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_user_data_probe_impl)


@njit
def _db_handle_probe_impl(ctx, argc, argv_pp):
    ud = sqlite3_user_data(ctx)
    out = carray(_cast_int_to_void_p(ud), (1,), dtype=np.intp)
    out[0] = sqlite3_context_db_handle(ctx)
    sqlite3_result_int(ctx, 0)


_db_handle_probe_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_db_handle_probe_impl)


# ---------------------------------------------------------------------------
# Tests of the scalar functions
# ---------------------------------------------------------------------------

def test_scalar_udf_double_value():
    db_p = _open_memory()
    out = np.zeros(1, dtype=np.int64)
    with c_string("my_double") as name_p:
        rc = sqlite3_create_function_v2(
            db_p, name_p, 1, SQLITE_UTF8, out.ctypes.data,
            _double_cb.address, 0, 0, 0)
    assert rc == SQLITE_OK
    with c_string("SELECT my_double(21)") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert out[0] == 42
    sqlite3_close(db_p)


def test_scalar_udf_null_handling():
    """A scalar function that returns NULL for a NULL argument runs without an error.

    The test checks the result code of ``sqlite3_exec`` and does not read the NULL back.
    """
    db_p = _open_memory()
    with c_string("my_null") as name_p:
        rc = sqlite3_create_function_v2(
            db_p, name_p, 1, SQLITE_UTF8, 0,
            _null_check_cb.address, 0, 0, 0)
    assert rc == SQLITE_OK
    with c_string("SELECT my_null(NULL)") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    sqlite3_close(db_p)


# ---------------------------------------------------------------------------
# Tests of the aggregate
# ---------------------------------------------------------------------------

def test_udaf_sum_structref():
    db_p = _open_memory()

    with c_string("CREATE TABLE t(v INTEGER)") as sql_p:
        assert sqlite3_exec(db_p, sql_p, 0, 0, 0) == SQLITE_OK
    for i in range(1, 6):
        with c_string("INSERT INTO t VALUES (%d)" % i) as sql_p:
            assert sqlite3_exec(db_p, sql_p, 0, 0, 0) == SQLITE_OK

    with c_string("my_sum") as name_p:
        rc = sqlite3_create_function_v2(
            db_p, name_p, 1, SQLITE_UTF8, 0,
            0, _sum_step_cb.address, _sum_final_cb.address, 0)
    assert rc == SQLITE_OK

    out = np.zeros(1, dtype=np.int64)
    with c_string("capture") as name_p:
        rc = sqlite3_create_function_v2(
            db_p, name_p, 1, SQLITE_UTF8, out.ctypes.data,
            _capture_cb.address, 0, 0, 0)
    assert rc == SQLITE_OK

    with c_string("SELECT capture(my_sum(v)) FROM t") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert out[0] == 15
    sqlite3_close(db_p)


def test_udaf_empty_group():
    """An aggregate over an empty table returns 0.

    SQLite calls xFinal without an xStep, so ``sqlite3_aggregate_context`` returns 0, and xFinal sets the result 0.
    """
    db_p = _open_memory()

    with c_string("CREATE TABLE te(v INTEGER)") as sql_p:
        assert sqlite3_exec(db_p, sql_p, 0, 0, 0) == SQLITE_OK

    with c_string("my_sum2") as name_p:
        rc = sqlite3_create_function_v2(
            db_p, name_p, 1, SQLITE_UTF8, 0,
            0, _sum_step_cb.address, _sum_final_cb.address, 0)
    assert rc == SQLITE_OK

    out = np.zeros(1, dtype=np.int64)
    out[0] = -999
    with c_string("capture2") as name_p:
        rc = sqlite3_create_function_v2(
            db_p, name_p, 1, SQLITE_UTF8, out.ctypes.data,
            _capture_cb.address, 0, 0, 0)
    assert rc == SQLITE_OK

    with c_string("SELECT capture2(my_sum2(v)) FROM te") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert out[0] == 0
    sqlite3_close(db_p)


# ---------------------------------------------------------------------------
# Tests of the window function
# ---------------------------------------------------------------------------

def test_window_running_sum():
    db_p = _open_memory()

    with c_string("CREATE TABLE tw(v INTEGER)") as sql_p:
        assert sqlite3_exec(db_p, sql_p, 0, 0, 0) == SQLITE_OK
    for i in range(1, 6):
        with c_string("INSERT INTO tw VALUES (%d)" % i) as sql_p:
            assert sqlite3_exec(db_p, sql_p, 0, 0, 0) == SQLITE_OK

    with c_string("my_wsum") as name_p:
        rc = sqlite3_create_window_function(
            db_p, name_p, 1, SQLITE_UTF8, 0,
            _wsum_step_cb.address, _wsum_final_cb.address,
            _wsum_value_cb.address, _wsum_inverse_cb.address, 0)
    assert rc == SQLITE_OK

    # The frame ROWS BETWEEN 1 PRECEDING AND CURRENT ROW holds each row and the row before it. my_wsum therefore
    # returns 1, 1 + 2, 2 + 3, 3 + 4 and 4 + 5.
    expected = [1, 3, 5, 7, 9]

    @njit
    def _wcap_impl(ctx, argc, argv_pp):
        ud = sqlite3_user_data(ctx)
        meta = carray(_cast_int_to_void_p(ud), (6,), dtype=np.int64)
        i = meta[0]
        args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
        meta[1 + i] = sqlite3_value_int64(args[0])
        meta[0] = i + 1
        sqlite3_result_int(ctx, 0)

    wcap_cb = cfunc(
        types.void(types.intp, types.int32, types.intp))(_wcap_impl)

    meta = np.zeros(6, dtype=np.int64)
    with c_string("wcap") as name_p:
        rc = sqlite3_create_function_v2(
            db_p, name_p, 1, SQLITE_UTF8, meta.ctypes.data,
            wcap_cb.address, 0, 0, 0)
    assert rc == SQLITE_OK

    sql = ("SELECT wcap(my_wsum(v) OVER "
           "(ORDER BY v ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)) "
           "FROM tw ORDER BY v")
    with c_string(sql) as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK

    assert meta[0] == 5, f"expected 5 rows captured, got {meta[0]}"
    for i, exp in enumerate(expected):
        assert meta[1 + i] == exp, f"row {i}: expected {exp}, got {meta[1 + i]}"

    sqlite3_close(db_p)


def test_udaf_no_meminfo_leak():
    """The aggregate and the window function above release every state that they create.

    After a first run of their tests, ten more runs allocate as many MemInfos as they free. The aggregate keeps its
    state alive with the reference that ``export_meminfo`` takes. The window function keeps its state alive with the
    reference that ``_incref_meminfo`` takes. In both, xFinal releases the reference once with ``release_meminfo``. A
    missing release leaks the state and makes the counts differ, and a missing reference frees the state while SQLite
    still uses it.
    """
    from numba.core.runtime import nrt
    _nrt = nrt._nrt
    if not hasattr(_nrt, "memsys_enable_stats"):
        import pytest
        pytest.skip("NRT allocation stats unavailable")
    _nrt.memsys_enable_stats()
    # A first run of each test comes before the measurement, so one-time allocations do not count.
    test_udaf_sum_structref()
    test_window_running_sum()
    test_udaf_empty_group()
    before = nrt.rtsys.get_allocation_stats()
    for _ in range(10):
        test_udaf_sum_structref()
        test_window_running_sum()
        test_udaf_empty_group()
    after = nrt.rtsys.get_allocation_stats()
    allocated = after.mi_alloc - before.mi_alloc
    freed = after.mi_free - before.mi_free
    assert allocated == freed, (
        f"meminfo imbalance: {allocated} allocated, {freed} freed")


# ---------------------------------------------------------------------------
# Tests of sqlite3_user_data and sqlite3_context_db_handle
# ---------------------------------------------------------------------------

def test_user_data_round_trip():
    """``sqlite3_user_data`` returns the ``p_app`` of the registration, which the probe compares with its argument."""
    db_p = _open_memory()
    ctx_arr = np.array([12345], dtype=np.int64)
    papp = ctx_arr.ctypes.data

    with c_string("udprobe") as name_p:
        rc = sqlite3_create_function_v2(
            db_p, name_p, 1, SQLITE_UTF8, papp,
            _user_data_probe_cb.address, 0, 0, 0)
    assert rc == SQLITE_OK

    out = np.zeros(1, dtype=np.int64)
    with c_string("cap_ud") as name_p:
        rc = sqlite3_create_function_v2(
            db_p, name_p, 1, SQLITE_UTF8, out.ctypes.data,
            _capture_cb.address, 0, 0, 0)
    assert rc == SQLITE_OK

    sql2 = "SELECT cap_ud(udprobe(%d))" % papp
    with c_string(sql2) as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert out[0] == 1, f"user_data mismatch: got {out[0]}"
    sqlite3_close(db_p)


def test_context_db_handle():
    """``sqlite3_context_db_handle`` returns the handle of the connection that runs the function."""
    db_p = _open_memory()
    out = np.zeros(1, dtype=np.intp)

    with c_string("dbprobe") as name_p:
        rc = sqlite3_create_function_v2(
            db_p, name_p, 0, SQLITE_UTF8, out.ctypes.data,
            _db_handle_probe_cb.address, 0, 0, 0)
    assert rc == SQLITE_OK

    with c_string("SELECT dbprobe()") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert out[0] == db_p, f"context_db_handle={out[0]}, expected={db_p}"
    sqlite3_close(db_p)


if __name__ == "__main__":
    from test.auxiliary_utils import collect_and_run_tests
    collect_and_run_tests(__name__)
