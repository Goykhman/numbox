"""sqlite3_exec + sqlite3_free tests."""
from ctypes import addressof, c_int64

import numpy as np
from numba import carray, cfunc, types

from numbox.core.bindings._sqlite_conn import sqlite3_changes
from numbox.core.bindings._sqlite_constants import SQLITE_ABORT, SQLITE_OK
from numbox.core.bindings._sqlite_exec import sqlite3_exec, sqlite3_free
from numbox.utils.lowlevel import get_str_from_p_as_int
from test.auxiliary_utils import collect_and_run_tests, cstr


# Module-level cfuncs — must outlive exec for the hooks/exec tests.

@cfunc(types.int32(types.voidptr, types.int32, types.intp, types.intp))
def _count_rows_cb(ctx, n, values_pp, names_pp):
    """Increment ctx[0] (numpy int64) per row."""
    arr = carray(ctx, 1, dtype=np.int64)
    arr[0] += 1
    return 0


@cfunc(types.int32(types.voidptr, types.int32, types.intp, types.intp))
def _abort_cb(ctx, n, values_pp, names_pp):
    return 1


def test_exec_create_insert_null_callback(memory_db):
    _, sql_p = cstr("CREATE TABLE t(a INTEGER); INSERT INTO t VALUES (1);")
    assert sqlite3_exec(memory_db, sql_p, 0, 0, 0) == SQLITE_OK
    assert sqlite3_changes(memory_db) == 1


def test_exec_callback_collects_row_count(memory_db):
    _, setup_p = cstr("CREATE TABLE t(a INTEGER); INSERT INTO t VALUES (1), (2), (3);")
    sqlite3_exec(memory_db, setup_p, 0, 0, 0)

    ctx = np.zeros(1, dtype=np.int64)
    _, sel_p = cstr("SELECT a FROM t")
    rc = sqlite3_exec(memory_db, sel_p, _count_rows_cb.address,
                      ctx.ctypes.data, 0)
    assert rc == SQLITE_OK
    assert ctx[0] == 3


def test_exec_callback_can_abort(memory_db):
    _, setup_p = cstr("CREATE TABLE t(a INTEGER); INSERT INTO t VALUES (1);")
    sqlite3_exec(memory_db, setup_p, 0, 0, 0)

    _, sel_p = cstr("SELECT a FROM t")
    rc = sqlite3_exec(memory_db, sel_p, _abort_cb.address, 0, 0)
    assert rc == SQLITE_ABORT


def test_exec_invalid_sql_writes_errmsg(memory_db):
    _, sql_p = cstr("SELECT FROM WHERE")
    errmsg_p = c_int64(0)
    rc = sqlite3_exec(memory_db, sql_p, 0, 0, addressof(errmsg_p))
    assert rc != SQLITE_OK
    assert errmsg_p.value != 0
    msg = get_str_from_p_as_int(errmsg_p.value)
    assert "syntax" in msg.lower() or "error" in msg.lower(), msg
    sqlite3_free(errmsg_p.value)  # no return value; just exercise the path


if __name__ == "__main__":
    collect_and_run_tests(__name__)
