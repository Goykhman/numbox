"""Tests of :func:`~numbox.core.bindings.sqlite.tvf.register_tvf`, which registers a table-valued function.

The tests register a function ``fn`` with ``register_tvf`` and query it with integer and float arguments, up to 64 of
them, and with one or two calls in a statement. ``fn`` returns full, sliced, strided or empty arrays. The tests also
check the hidden argument columns, the error of a query whose ``fn`` raises, and every column type of the output. Other
tests check that SQLite copies text cells and that the outer xFilter writes no cache entry. The xBestIndex tests check
which plans it accepts or rejects. test_sqlite_vtable.py tests the release of a registration when its connection closes.
"""
import os
import pickle
from ctypes import addressof, c_char_p, c_int32, c_int64, cast, string_at

import pytest
import numpy as np
from numba import njit
from numba.core import config

from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import array_data_p
from numbox.core.bindings.sqlite.conn import sqlite3_open, sqlite3_close
from numbox.core.bindings.sqlite import tvf
from numbox.core.bindings.sqlite.tvf import register_tvf
from numbox.core.bindings.sqlite.stmt import sqlite3_prepare_v2, sqlite3_step, sqlite3_finalize
from numbox.core.bindings.sqlite.column import (
    sqlite3_column_int64, sqlite3_column_double, sqlite3_column_text, sqlite3_column_type,
    sqlite3_column_count, sqlite3_column_blob, sqlite3_column_bytes,
)
from numbox.core.bindings.sqlite.tvf import _make_xbestindex, _TVF_DESC_DTYPE
from numbox.core.bindings.sqlite.vtable import (
    _VTAB_DTYPE, _IDX_INFO_DTYPE, _CONSTRAINT_DTYPE, _USAGE_DTYPE,
)
from numbox.core.bindings.sqlite.constants import (
    SQLITE_OK, SQLITE_CONSTRAINT, SQLITE_INDEX_CONSTRAINT_EQ, SQLITE_ROW,
    SQLITE_ERROR, SQLITE_DONE, SQLITE_INTEGER, SQLITE_TEXT, SQLITE_FLOAT,
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


@njit
def _sum_args(*args):
    out = np.empty(1, _OUT)
    total = 0
    for a in args:
        total += a
    out[0].n = total
    return out


def _open():
    db = c_int64(0)
    with c_string(":memory:") as p:
        rc = sqlite3_open(p, addressof(db))
    assert rc == 0, rc
    return db


def _select_int(db, sql, ncol=1):
    """Run ``sql``; return the result code of ``sqlite3_prepare_v2`` and the rows as tuples of ``ncol`` int64."""
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
    """Two calls of one function in one statement have separate cursors, so each call returns its own rows."""
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


def test_tvf_hidden_columns_read_back_their_arguments():
    """A query can select the hidden columns ``arg0``, ``arg1`` and ``arg2``, which return the arguments of the call.

    xColumn returns the value that fn received, an INTEGER for an integer argument and a REAL for a float argument, as
    SQLite's own table-valued functions do.
    """
    db = _open()
    h = register_tvf(db.value, "scaled", (np.int64, np.int64, np.float64), _OUT2, _scaled)
    stmt = c_int64(0)
    with c_string("SELECT n, arg0, arg1, arg2 FROM scaled(0, 3, 2.5)") as p:
        assert sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0) == SQLITE_OK
    got = []
    while sqlite3_step(stmt.value) == SQLITE_ROW:
        got.append((sqlite3_column_int64(stmt.value, 0),
                    sqlite3_column_type(stmt.value, 1), sqlite3_column_int64(stmt.value, 1),
                    sqlite3_column_type(stmt.value, 2), sqlite3_column_int64(stmt.value, 2),
                    sqlite3_column_type(stmt.value, 3), sqlite3_column_double(stmt.value, 3)))
    sqlite3_finalize(stmt.value)
    sqlite3_close(db.value)
    del h
    assert got == [(i, SQLITE_INTEGER, 0, SQLITE_INTEGER, 3, SQLITE_FLOAT, 2.5) for i in range(3)]


def test_tvf_where_on_an_argument_column_is_checked_against_the_argument():
    """SQLite checks a WHERE constraint on an argument column, such as ``arg0 = 2``, against the call's argument."""
    # The call binds arg0 and arg1, so an EQ constraint on either argument in the WHERE clause is a second EQ on that
    # argument. xBestIndex binds the first usable EQ constraint of each argument, and SQLite checks the other EQ
    # constraint against the hidden column.
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series)
    assert _select_int(db, "SELECT n FROM series(2, 5) WHERE arg0 = 2") == (SQLITE_OK, [(2,), (3,), (4,)])
    assert _select_int(db, "SELECT n FROM series(2, 5) WHERE arg1 = 5") == (SQLITE_OK, [(2,), (3,), (4,)])
    assert _select_int(db, "SELECT n FROM series(2, 5) WHERE arg0 = 3") == (SQLITE_OK, [])
    sqlite3_close(db.value)
    del h


def test_tvf_column_tag_without_a_branch_fails_the_query():
    """A column tag without a branch in ``_emit_cell`` fails a query of the function with ``SQLITE_ERROR``."""
    # _tvf_xcolumn passes each visible cell to _emit_cell of vtable.py, so a tag without a branch there fails this query
    # as it fails a query of register_table. The test writes the tag 99 over the first entry of the descriptor's
    # col_tags buffer.
    from numbox.core.bindings.sqlite import vtable as v
    db = _open()
    keys0 = set(v._DATA_ANCHOR)
    register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series)
    (key,) = set(v._DATA_ANCHOR) - keys0
    desc = next(o for o in v._DATA_ANCHOR[key]._keep if isinstance(o, np.ndarray) and o.dtype == _TVF_DESC_DTYPE)
    c_int32.from_address(int(desc["col_tags"][0])).value = 99
    rc = _step_rc(db, "SELECT n FROM series(2, 5)")
    sqlite3_close(db.value)
    assert rc == SQLITE_ERROR


def test_tvf_missing_hidden_arg():
    """A query that leaves the arguments unbound returns no rows.

    SQLite fails to prepare it with "no query solution", because xBestIndex rejects every plan with an unbound argument.
    """
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
    """Queries release every array that fn returns.

    After one query, ten more queries allocate as many MemInfos as they free.
    """
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
    # The function returns a slice whose data starts one row after the start of its allocation.
    out = np.empty((stop - start) + 1, _OUT)
    for i in range((stop - start) + 1):
        out[i].n = (start - 1) + i
    return out[1:]


@njit
def _series_strided(start, stop):
    # The function returns a view whose row stride is twice the itemsize.
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
    # The function raises with a raise statement, because numba checks no bounds by default, and an index out of bounds
    # would read or write past the array instead of raising.
    out = np.empty(stop - start, _OUT)
    if start < stop:
        raise ValueError("boom")
    return out


def _step_rc(db, sql):
    """Prepare ``sql``, step it once and finalize it, and return the result code of the step."""
    stmt = c_int64(0)
    with c_string(sql) as p:
        assert sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0) == SQLITE_OK
    rc = sqlite3_step(stmt.value)
    sqlite3_finalize(stmt.value)
    return rc


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_tvf_user_fn_raises_yields_error():
    """An exception in fn fails the query with ``SQLITE_ERROR`` instead of a successful query with no rows."""
    db = _open()
    h = register_tvf(db.value, "boom", (np.int64, np.int64), _OUT, _raises)
    assert _step_rc(db, "SELECT n FROM boom(2, 5)") == SQLITE_ERROR
    sqlite3_close(db.value)
    del h


def test_tvf_empty_return_is_success_not_error():
    """An empty array from fn completes the query with ``SQLITE_DONE``; the error path of xFilter does not fail it."""
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series_empty)
    assert _step_rc(db, "SELECT n FROM series(5, 5)") == SQLITE_DONE
    sqlite3_close(db.value)
    del h


def test_tvf_two_distinct_registrations_same_process():
    """Two functions with different dtypes, registered on one connection, return their own rows."""
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
    """Call the xBestIndex callback of ``_make_xbestindex`` on a ``sqlite3_index_info`` built from ``constraints``.

    ``constraints`` holds ``(iColumn, op, usable)`` triples, and the hidden arguments are the columns ``ncols`` to
    ``ncols + n_hidden - 1``. Returns the result code and the usage array.
    """
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
    # The NumPy arrays stay referenced until the call returns, so the addresses that the structs hold stay valid.
    rc = _make_xbestindex().ctypes(vtab.ctypes.data, ii.ctypes.data)
    return rc, usage


def test_tvf_xbestindex_rejects_unbound_arg_despite_duplicate_eq():
    """xBestIndex rejects a plan that leaves an argument unbound, although it has as many EQ constraints as arguments.

    The table has 2 visible columns and 3 hidden columns, the columns 2, 3 and 4. ``arg0`` has two usable EQ
    constraints, ``arg2`` has one, and ``arg1`` has none. An implementation that counted usable EQ constraints would
    reach 3, the number of arguments, and accept the plan. SQLite passes such a duplicate when a query filters an
    argument that the call binds, as in ``f(1) WHERE arg0 = 1``.
    """
    EQ = SQLITE_INDEX_CONSTRAINT_EQ
    rc, _ = _call_xbestindex(2, 3, [(2, EQ, 1), (2, EQ, 1), (4, EQ, 1)])
    assert rc == SQLITE_CONSTRAINT, rc


def test_tvf_xbestindex_accepts_all_args_bound_with_duplicate_eq():
    """xBestIndex accepts a plan that binds every argument, even when one argument has a duplicate EQ constraint.

    Every hidden argument, the columns 2, 3 and 4, has a usable EQ constraint, and ``arg0`` has a second EQ constraint.
    SQLite refuses a plan in which two constraints share an argvIndex. The duplicate therefore receives no argvIndex and
    is not omitted, so SQLite checks it against the hidden column.
    """
    EQ = SQLITE_INDEX_CONSTRAINT_EQ
    rc, usage = _call_xbestindex(2, 3, [(2, EQ, 1), (2, EQ, 1), (3, EQ, 1), (4, EQ, 1)])
    assert rc == SQLITE_OK, rc
    assert [(int(u["argvIndex"]), int(u["omit"])) for u in usage] == [(1, 1), (0, 0), (2, 1), (3, 1)]


def test_tvf_xbestindex_checks_every_arg_past_64():
    """xBestIndex binds each of 64 or 65 arguments, and it rejects the plan when the first or the last is unbound.

    The table has 1 visible column and n hidden columns. xBestIndex checks each argument separately, so the limit of a
    64-bit mask of bound arguments does not apply.
    """
    EQ = SQLITE_INDEX_CONSTRAINT_EQ
    for n in (64, 65):
        rc, usage = _call_xbestindex(1, n, [(1 + h, EQ, 1) for h in range(n)])
        assert rc == SQLITE_OK, (n, rc)
        assert [int(usage[i]["argvIndex"]) for i in range(n)] == list(range(1, n + 1))
        for unbound in (0, n - 1):
            rc, _ = _call_xbestindex(1, n, [(1 + h, EQ, 1) for h in range(n) if h != unbound])
            assert rc == SQLITE_CONSTRAINT, (n, unbound, rc)


def test_tvf_with_64_args_returns_its_row():
    """A function of 64 integer arguments returns one row, the sum of its arguments 0 to 63."""
    db = _open()
    h = register_tvf(db.value, "sum64", (np.int64,) * 64, _OUT, _sum_args)
    sql = "SELECT n FROM sum64(%s)" % ", ".join(str(i) for i in range(64))
    assert _select_int(db, sql) == (SQLITE_OK, [(64 * 63 // 2,)])
    sqlite3_close(db.value)
    del h


def _fetchall_text(db, sql):
    """Run ``sql`` and return its rows as tuples: an int for INTEGER, a str for TEXT and None for any other value.

    The text is read as the C string that ``sqlite3_column_text`` returns, so it ends at its first NUL.
    """
    stmt = c_int64(0)
    with c_string(sql) as p:
        assert sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0) == SQLITE_OK
    rows = []
    while sqlite3_step(stmt.value) == SQLITE_ROW:
        row = []
        for i in range(sqlite3_column_count(stmt.value)):
            t = sqlite3_column_type(stmt.value, i)
            if t == SQLITE_INTEGER:
                row.append(sqlite3_column_int64(stmt.value, i))
            elif t == SQLITE_TEXT:
                row.append(cast(sqlite3_column_text(stmt.value, i), c_char_p).value.decode("utf-8"))
            else:
                row.append(None)
        rows.append(tuple(row))
    sqlite3_finalize(stmt.value)
    return rows


_STR_OUT = np.dtype([("n", "i8"), ("u", "U6"), ("s", "S4")])
# numba cannot assign a str to a 'U' field of a record in jitted code, because it fails to cast the str literal to the
# fixed-width field. fn therefore returns a slice of an array that Python builds.
_STR_ROWS = np.array([(0, "héllo", b"ab"), (1, "\U0001F600", b"cd")], dtype=_STR_OUT)


@njit
def _str_series(start, stop):
    return _STR_ROWS[start:stop]


def test_tvf_string_and_unicode_output_columns():
    """A query reads a ``'U'`` and an ``'S'`` column of fn's array as TEXT, including ``'U'`` values outside ASCII."""
    # _emit_cell converts a 'U' cell to UTF-8 with utf32_to_utf8, in the cursor's scratch buffer. The UTF-8 encoding of
    # the test's 'U' values contains 2-byte and 4-byte forms. _emit_cell passes an 'S' cell as TEXT. register_tvf has no
    # text_as_blob parameter, so an 'S' column is always TEXT, and the BLOB branch of _emit_cell is unreachable through
    # register_tvf.
    db = _open()
    h = register_tvf(db.value, "strs", (np.int64, np.int64), _STR_OUT, _str_series)
    rows = _fetchall_text(db, "SELECT n, u, s FROM strs(0, 2)")
    assert rows == [(0, "héllo", "ab"), (1, "\U0001F600", "cd")]
    sqlite3_close(db.value)
    del h


def test_tvf_two_unicode_columns_share_cursor_scratch():
    """Two ``'U'`` columns of one row keep their own values, although _emit_cell converts both in one scratch buffer.

    _emit_cell passes a ``'U'`` result as ``SQLITE_TRANSIENT``, so SQLite copies it before the next xColumn call
    overwrites the buffer. A result passed as ``SQLITE_STATIC`` would point into the overwritten buffer.
    """
    db = _open()
    h = register_tvf(db.value, "twou", (np.int64, np.int64), _TWO_U_OUT, _two_u_series)
    rows = _fetchall_text(db, "SELECT a, b FROM twou(0, 2)")
    assert rows == [("αβ", "héllo"), ("😀", "wörld")]
    sqlite3_close(db.value)
    del h


_NUM_OUT = np.dtype([("i1", "i1"), ("i2", "i2"), ("i4", "i4"), ("i8", "i8"),
                     ("u1", "u1"), ("u2", "u2"), ("u4", "u4"), ("u8", "u8"),
                     ("f4", "f4"), ("f8", "f8"), ("b", "?")])
_NUM_ROWS = np.array([
    (-128, -32768, -2 ** 31, -2 ** 63, 255, 65535, 2 ** 32 - 1, 2 ** 64 - 1, 1.5, -2.25, True),
    (127, 32767, 2 ** 31 - 1, 2 ** 63 - 1, 0, 1, 2, 3, -0.5, 1e300, False),
], dtype=_NUM_OUT)


@njit
def _num_series(start, stop):
    return _NUM_ROWS[start:stop]


def test_tvf_every_numeric_and_bool_output_column():
    """A query reads each integer, float and bool column of fn's array with the expected value and storage class."""
    # SQLite stores an integer as an int64, so a query returns -1 for the largest uint64 and the INTEGER 0 or 1 for a
    # bool. The test compares the type codes as well, because -128.0 == -128 is true in Python.
    db = _open()
    h = register_tvf(db.value, "nums", (np.int64, np.int64), _NUM_OUT, _num_series)
    stmt = c_int64(0)
    with c_string("SELECT i1, i2, i4, i8, u1, u2, u4, u8, f4, f8, b FROM nums(0, 2)") as p:
        assert sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0) == SQLITE_OK
    rows, kinds = [], []
    while sqlite3_step(stmt.value) == SQLITE_ROW:
        n = sqlite3_column_count(stmt.value)
        kinds.append([sqlite3_column_type(stmt.value, i) for i in range(n)])
        rows.append(tuple(
            sqlite3_column_double(stmt.value, i) if kinds[-1][i] == SQLITE_FLOAT
            else sqlite3_column_int64(stmt.value, i) for i in range(n)
        ))
    sqlite3_finalize(stmt.value)
    sqlite3_close(db.value)
    del h
    assert kinds == [[SQLITE_INTEGER] * 8 + [SQLITE_FLOAT, SQLITE_FLOAT, SQLITE_INTEGER]] * 2
    assert rows == [
        (-128, -32768, -2 ** 31, -2 ** 63, 255, 65535, 2 ** 32 - 1, -1, 1.5, -2.25, 1),
        (127, 32767, 2 ** 31 - 1, 2 ** 63 - 1, 0, 1, 2, 3, -0.5, 1e300, 0),
    ]


_TWO_U_OUT = np.dtype([("a", "U6"), ("b", "U6")])
_TWO_U_ROWS = np.array([("αβ", "héllo"), ("\U0001F600", "wörld")], dtype=_TWO_U_OUT)


@njit
def _two_u_series(start, stop):
    return _TWO_U_ROWS[start:stop]


_ADDR_S_OUT = np.dtype([("addr", "i8"), ("s", "S4")])
_ADDR_S_ROWS = np.array([(0, b"ab"), (0, b"cd")], dtype=_ADDR_S_OUT)


@njit
def _addr_s_series(start, stop):
    # The function returns a new array for each call, as a typical fn does, and each row holds the array's address.
    out = _ADDR_S_ROWS[start:stop].copy()
    for i in range(out.shape[0]):
        out[i].addr = array_data_p(out)
    return out


def test_tvf_text_result_is_copied_by_sqlite():
    """SQLite copies an ``'S'`` cell of a table-valued function, so the cell's address lies outside fn's array.

    The next xFilter releases the array that fn returned, so _tvf_xcolumn passes an ``'S'`` cell as
    ``SQLITE_TRANSIENT``, and SQLite copies it. ``sqlite3_column_blob`` returns the address of the stored value, so a
    cell passed without a copy would have an address inside fn's array.
    """
    db = _open()
    h = register_tvf(db.value, "addrs", (np.int64, np.int64), _ADDR_S_OUT, _addr_s_series)
    stmt = c_int64(0)
    with c_string("SELECT addr, s FROM addrs(0, 2)") as p:
        assert sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0) == SQLITE_OK
    rows = []
    while sqlite3_step(stmt.value) == SQLITE_ROW:
        base = sqlite3_column_int64(stmt.value, 0)
        s_p = sqlite3_column_blob(stmt.value, 1)
        rows.append((base, s_p, string_at(s_p, sqlite3_column_bytes(stmt.value, 1))))
    sqlite3_finalize(stmt.value)
    sqlite3_close(db.value)
    del h
    assert [r[2] for r in rows] == [b"ab", b"cd"]
    for base, s_p, _ in rows:
        assert not base <= s_p < base + 2 * _ADDR_S_OUT.itemsize


def _outer_xfilter_overloads():
    """Return the number of overloads that numba's index files record for ``_tvf_xfilter``, a registration's cfunc.

    The count sums the index files under the directory of tvf.py and under numba's ``CACHE_DIR``, if it is set.
    """
    roots = [os.path.dirname(tvf.__file__)]
    if config.CACHE_DIR:
        roots.append(config.CACHE_DIR)
    total = 0
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if not fn.startswith("tvf.register_tvf.locals._tvf_xfilter-"):
                    continue
                if not fn.endswith(".nbi"):
                    continue
                # An index file holds a pickle of numba's version, then a pickle of the source stamp and the overloads.
                with open(os.path.join(dirpath, fn), "rb") as f:
                    pickle.load(f)
                    data = f.read()
                try:
                    _stamp, overloads = pickle.loads(data)
                except Exception:
                    continue
                total += len(overloads)
    return total


def test_tvf_outer_xfilter_does_not_grow_the_numba_cache():
    """A registration adds no entry for ``_tvf_xfilter`` to numba's cache index.

    The cfunc closes over xfilter_impl, a dispatcher made for each registration, so its cache key never repeats, and an
    entry written for it would never be read. If the cfunc were cached, each registration would add one overload to the
    index.
    """
    before = _outer_xfilter_overloads()
    db = _open()
    h = register_tvf(db.value, "nocache", (np.int64, np.int64), _OUT, _series)
    rc, rows = _select_int(db, "SELECT n FROM nocache(3, 6)")
    assert rc == SQLITE_OK
    assert rows == [(3,), (4,), (5,)]
    assert _outer_xfilter_overloads() == before
    sqlite3_close(db.value)
    del h
