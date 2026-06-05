"""UDF registration and integration tests for the SQLite buildout.

Covers scalar UDFs, aggregate UDFs (structref-backed), window UDFs,
and context helper round-trips (sqlite3_user_data, sqlite3_context_db_handle).
"""
from ctypes import addressof, c_int64

import numpy as np
from numba import carray, cfunc, njit, types
from numba.core import types as nb_types
from numba.experimental import structref

from numbox.core.bindings import (
    SQLITE_NULL,
    SQLITE_OK,
    SQLITE_UTF8,
    sqlite3_aggregate_context,
    sqlite3_close,
    sqlite3_context_db_handle,
    sqlite3_create_function_v2,
    sqlite3_create_window_function,
    sqlite3_exec,
    sqlite3_open,
    sqlite3_result_int,
    sqlite3_result_int64,
    sqlite3_result_null,
    sqlite3_user_data,
    sqlite3_value_int,
    sqlite3_value_int64,
    sqlite3_value_type,
)
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
# Structref for aggregate/window state
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
# Scalar UDF callbacks
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
# Capture probe: scalar UDF that writes its first arg into user_data
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
# Aggregate UDF callbacks (structref-backed)
# ---------------------------------------------------------------------------

@njit
def _sum_step_impl(ctx, argc, argv_pp):
    agg_ptr = sqlite3_aggregate_context(ctx, 8)
    if agg_ptr == 0:  # NULL on OOM; carray on a NULL pointer would segfault
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
# Window UDF callbacks (array-backed state)
#
# Alternative to the structref-backed aggregate above: the aggregate_context
# slot is 16 bytes holding two intp values -- [meminfo_p, data_p] -- instead
# of a single structref meminfo pointer. The payload is a 1-element int64
# array (the running total) allocated with np.zeros inside the step callback.
# Because np.zeros emits an NRT_MemInfo_alloc in that body, removerefctpass is
# disabled there, so the manual _incref_meminfo (the @intrinsic, which inlines)
# survives and pins the buffer past the callback; release_meminfo in xFinal
# frees it. Subsequent callbacks reach the payload through data_p via carray --
# no structref reconstruction. Same pattern as numbduck's array UDAF.
# ---------------------------------------------------------------------------

_WSUM_SLOTS = 2  # aggregate_context layout: slot[0]=meminfo_p, slot[1]=data_p


@njit
def _wsum_step_impl(ctx, argc, argv_pp):
    agg_ptr = sqlite3_aggregate_context(ctx, 16)
    if agg_ptr == 0:  # NULL on OOM; carray on a NULL pointer would segfault
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
    if agg_ptr == 0:  # NULL on OOM; carray on a NULL pointer would segfault
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
# Context helper callbacks
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
# Scalar UDF tests
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
# Aggregate UDF tests
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
# Window UDF test
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

    # Window: ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
    # Row 1: sum(1)     = 1
    # Row 2: sum(1,2)   = 3
    # Row 3: sum(2,3)   = 5
    # Row 4: sum(3,4)   = 7
    # Row 5: sum(4,5)   = 9
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
    """Regression guard for the aggregate/window state lifecycle.

    Exercises both bridge idioms: the structref-backed aggregate
    (``test_udaf_sum_structref``) and the array-backed window function
    (``test_window_running_sum``). Each keeps its state alive across callbacks
    with a +1 incref -- ``export_meminfo`` for the structref, the inlined
    ``_incref_meminfo`` for the array -- and frees it with a single
    ``release_meminfo`` in xFinal. A dropped incref frees the state early
    (use-after-free); a dropped release leaks. Either shows up as an imbalance
    between meminfo allocations and frees.
    """
    from numba.core.runtime import nrt
    _nrt = nrt._nrt
    if not hasattr(_nrt, "memsys_enable_stats"):
        import pytest
        pytest.skip("NRT allocation stats unavailable")
    _nrt.memsys_enable_stats()
    # warm up JIT / one-time allocations before measuring steady state
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
# Context helper tests
# ---------------------------------------------------------------------------

def test_user_data_round_trip():
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
