from ctypes import addressof, c_int64

import pytest
import numpy as np
from numba import njit

from numbox.utils.cstrings import c_string
from numbox.core.bindings import (
    sqlite3_open, sqlite3_close, register_tvf,
    sqlite3_prepare_v2, sqlite3_step, sqlite3_finalize,
    sqlite3_column_int64, sqlite3_column_double,
)
from numbox.core.bindings._sqlite_tvf import _make_xbestindex, _TVF_DESC_DTYPE
from numbox.core.bindings._sqlite_vtable import (
    _VTAB_DTYPE, _IDX_INFO_DTYPE, _CONSTRAINT_DTYPE, _USAGE_DTYPE,
)
from numbox.core.bindings._sqlite_constants import (
    SQLITE_OK, SQLITE_CONSTRAINT, SQLITE_INDEX_CONSTRAINT_EQ, SQLITE_ROW,
    SQLITE_ERROR, SQLITE_DONE,
)

_OUT = np.dtype([("n", "i8")])
_OUT2 = np.dtype([("n", "i8"), ("v", "f8")])


@njit
def _series(start, stop):
    out = np.empty(stop - start, _OUT)
    for i in range(stop - start):
        out[i].n = start + i
    return out


@njit
def _scaled(start, stop, scale):
    out = np.empty(stop - start, _OUT2)
    for i in range(stop - start):
        out[i].n = start + i
        out[i].v = (start + i) * scale
    return out


def _open():
    db = c_int64(0)
    with c_string(":memory:") as p:
        rc = sqlite3_open(p, addressof(db))
    assert rc == 0, rc
    return db


def _select_int(db, sql, ncol=1):
    stmt = c_int64(0)
    with c_string(sql) as p:
        rc = sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    rows = []
    if rc == 0:
        while sqlite3_step(stmt.value) == SQLITE_ROW:
            rows.append(tuple(sqlite3_column_int64(stmt.value, j) for j in range(ncol)))
    sqlite3_finalize(stmt.value)
    return rc, rows


def test_tvf_series():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series)
    stmt = c_int64(0)
    with c_string("SELECT n FROM series(2, 5)") as p:
        sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    got = []
    while sqlite3_step(stmt.value) == SQLITE_ROW:
        got.append(sqlite3_column_int64(stmt.value, 0))
    sqlite3_finalize(stmt.value)
    assert got == [2, 3, 4]
    sqlite3_close(db.value)
    del h


def test_tvf_two_calls_same_process():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series)
    _, r1 = _select_int(db, "SELECT n FROM series(2, 5)")
    _, r2 = _select_int(db, "SELECT n FROM series(10, 13)")
    assert [x[0] for x in r1] == [2, 3, 4]
    assert [x[0] for x in r2] == [10, 11, 12]
    sqlite3_close(db.value)
    del h


def test_tvf_two_calls_one_query_plan():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series)
    stmt = c_int64(0)
    sql = "SELECT a.n, b.n FROM series(2, 5) a, series(10, 13) b WHERE a.n = 2"
    with c_string(sql) as p:
        sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    got = []
    while sqlite3_step(stmt.value) == SQLITE_ROW:
        got.append((sqlite3_column_int64(stmt.value, 0), sqlite3_column_int64(stmt.value, 1)))
    sqlite3_finalize(stmt.value)
    assert got == [(2, 10), (2, 11), (2, 12)]
    sqlite3_close(db.value)
    del h


def test_tvf_multi_column_and_float_arg():
    db = _open()
    h = register_tvf(db.value, "scaled", (np.int64, np.int64, np.float64), _OUT2, _scaled)
    stmt = c_int64(0)
    with c_string("SELECT n, v FROM scaled(0, 3, 2.5)") as p:
        sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    got = []
    while sqlite3_step(stmt.value) == SQLITE_ROW:
        got.append((sqlite3_column_int64(stmt.value, 0), sqlite3_column_double(stmt.value, 1)))
    sqlite3_finalize(stmt.value)
    assert got == [(0, 0.0), (1, 2.5), (2, 5.0)]
    sqlite3_close(db.value)
    del h


def test_tvf_missing_hidden_arg():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series)
    stmt = c_int64(0)
    with c_string("SELECT n FROM series WHERE n < 100") as p:
        rc = sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    rows = []
    if rc == 0:
        while sqlite3_step(stmt.value) == SQLITE_ROW:
            rows.append(sqlite3_column_int64(stmt.value, 0))
    sqlite3_finalize(stmt.value)
    assert rows == []
    sqlite3_close(db.value)
    del h


def test_tvf_no_meminfo_leak():
    from numba.core.runtime import nrt
    _nrt = nrt._nrt
    if not hasattr(_nrt, "memsys_enable_stats"):
        pytest.skip("NRT allocation stats unavailable")
    _nrt.memsys_enable_stats()
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series)
    _select_int(db, "SELECT n FROM series(2, 5)")
    before = nrt.rtsys.get_allocation_stats()
    for _ in range(10):
        _select_int(db, "SELECT n FROM series(2, 5)")
    after = nrt.rtsys.get_allocation_stats()
    allocated = after.mi_alloc - before.mi_alloc
    freed = after.mi_free - before.mi_free
    sqlite3_close(db.value)
    del h
    assert allocated == freed, "meminfo imbalance: %d alloc, %d free" % (allocated, freed)


@njit
def _series_sliced(start, stop):
    # returns an offset slice: logical start != allocation base
    out = np.empty((stop - start) + 1, _OUT)
    for i in range((stop - start) + 1):
        out[i].n = (start - 1) + i
    return out[1:]


@njit
def _series_strided(start, stop):
    # returns a strided view: row stride = 2 * itemsize
    out = np.empty(2 * (stop - start), _OUT)
    for i in range(2 * (stop - start)):
        out[i].n = -1
    for i in range(stop - start):
        out[2 * i].n = start + i
    return out[::2]


@njit
def _series_empty(start, stop):
    return np.empty(0, _OUT)


def test_tvf_offset_slice_return():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series_sliced)
    _, rows = _select_int(db, "SELECT n FROM series(2, 5)")
    assert [x[0] for x in rows] == [2, 3, 4]
    sqlite3_close(db.value)
    del h


def test_tvf_strided_return():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series_strided)
    _, rows = _select_int(db, "SELECT n FROM series(2, 5)")
    assert [x[0] for x in rows] == [2, 3, 4]
    sqlite3_close(db.value)
    del h


def test_tvf_empty_return():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series_empty)
    _, rows = _select_int(db, "SELECT n FROM series(2, 5)")
    assert rows == []
    sqlite3_close(db.value)
    del h


def test_tvf_non_numeric_arg_type_raises():
    db = _open()
    with pytest.raises(TypeError):
        register_tvf(db.value, "f", (np.dtype("U4"),), _OUT, _series)
    sqlite3_close(db.value)


@njit
def _raises(start, stop):
    # raise explicitly: numba's default boundscheck is off, so an OOB index would
    # corrupt rather than raise.
    out = np.empty(stop - start, _OUT)
    if start < stop:
        raise ValueError("boom")
    return out


def _step_rc(db, sql):
    """Prepare `sql`, step once, return that step's result code (after finalize)."""
    stmt = c_int64(0)
    with c_string(sql) as p:
        assert sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0) == SQLITE_OK
    rc = sqlite3_step(stmt.value)
    sqlite3_finalize(stmt.value)
    return rc


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_tvf_user_fn_raises_yields_error():
    # a raising user fn must surface as a query error (xFilter returns
    # SQLITE_ERROR), not a silently-empty but successful result.
    db = _open()
    h = register_tvf(db.value, "boom", (np.int64, np.int64), _OUT, _raises)
    assert _step_rc(db, "SELECT n FROM boom(2, 5)") == SQLITE_ERROR
    sqlite3_close(db.value)
    del h


def test_tvf_empty_return_is_success_not_error():
    # a legitimately-empty TVF must still complete successfully (DONE), so the
    # error path above does not turn empty results into spurious failures.
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series_empty)
    assert _step_rc(db, "SELECT n FROM series(5, 5)") == SQLITE_DONE
    sqlite3_close(db.value)
    del h


def test_tvf_two_distinct_registrations_same_process():
    db = _open()
    h1 = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series)
    h2 = register_tvf(db.value, "scaled", (np.int64, np.int64, np.float64), _OUT2, _scaled)
    _, r1 = _select_int(db, "SELECT n FROM series(2, 5)")
    stmt = c_int64(0)
    with c_string("SELECT n, v FROM scaled(0, 3, 2.5)") as p:
        sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    r2 = []
    while sqlite3_step(stmt.value) == SQLITE_ROW:
        r2.append((sqlite3_column_int64(stmt.value, 0), sqlite3_column_double(stmt.value, 1)))
    sqlite3_finalize(stmt.value)
    assert [x[0] for x in r1] == [2, 3, 4]
    assert r2 == [(0, 0.0), (1, 2.5), (2, 5.0)]
    sqlite3_close(db.value)
    del h1, h2


def _call_xbestindex(ncols, n_hidden, constraints):
    """Drive _make_xbestindex()'s cfunc against a hand-built sqlite3_index_info.
    ``constraints`` is a list of (iColumn, op, usable). Returns (rc, usage_array).
    The numpy buffers stay referenced for the whole call so their data pointers
    remain valid. Hidden args occupy table columns ncols .. ncols+n_hidden-1."""
    desc = np.zeros(1, _TVF_DESC_DTYPE)
    desc[0]["ncols"] = ncols
    desc[0]["n_hidden"] = n_hidden
    vtab = np.zeros(1, _VTAB_DTYPE)
    vtab[0]["descriptor"] = desc.ctypes.data
    n = len(constraints)
    cons = np.zeros(n, _CONSTRAINT_DTYPE)
    for i, (col, op, usable) in enumerate(constraints):
        cons[i]["iColumn"] = col
        cons[i]["op"] = op
        cons[i]["usable"] = usable
    usage = np.zeros(n, _USAGE_DTYPE)
    ii = np.zeros(1, _IDX_INFO_DTYPE)
    ii[0]["nConstraint"] = n
    ii[0]["aConstraint"] = cons.ctypes.data
    ii[0]["aConstraintUsage"] = usage.ctypes.data
    rc = _make_xbestindex().ctypes(vtab.ctypes.data, ii.ctypes.data)
    return rc, usage


def test_tvf_xbestindex_rejects_unbound_arg_despite_duplicate_eq():
    # 2 visible cols, 3 hidden (cols 2,3,4). Duplicate usable EQ on arg0 (col 2)
    # plus a usable EQ on arg2 (col 4), with arg1 (col 3) left unbound. A naive
    # usable-EQ count reaches 3 == n_hidden and wrongly accepts the plan even
    # though one hidden arg is unbound. SQLite coalesces such constraints before
    # xBestIndex so this never arrives via SQL, but the contract is to reject it.
    EQ = SQLITE_INDEX_CONSTRAINT_EQ
    rc, _ = _call_xbestindex(2, 3, [(2, EQ, 1), (2, EQ, 1), (4, EQ, 1)])
    assert rc == SQLITE_CONSTRAINT, rc


def test_tvf_xbestindex_accepts_all_args_bound_with_duplicate_eq():
    # Every hidden arg (cols 2,3,4) has a usable EQ, with a redundant duplicate on
    # arg0: the plan must still be accepted (the duplicate must not change the
    # all-bound verdict).
    EQ = SQLITE_INDEX_CONSTRAINT_EQ
    rc, usage = _call_xbestindex(2, 3, [(2, EQ, 1), (2, EQ, 1), (3, EQ, 1), (4, EQ, 1)])
    assert rc == SQLITE_OK, rc
    assert [int(usage[i]["argvIndex"]) for i in range(4)] == [1, 1, 2, 3]
