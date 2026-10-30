"""Tests of the bindings of :mod:`~numbox.core.bindings.sqlite.exec`.

The tests run SQL through ``sqlite3_exec`` without a row callback, with a callback that counts the rows and with a
callback that stops the run. For invalid SQL, ``sqlite3_exec`` writes an error message, which the test frees with
``sqlite3_free``. ``sqlite3_malloc`` has no test here; numbox's virtual tables call it in ``xConnect`` and ``xOpen``,
which test_sqlite_vtable.py and test_sqlite_tvf.py run.
"""
from ctypes import addressof, c_int64

import numpy as np
from numba import carray, cfunc, types

from numbox.core.bindings.sqlite.constants import SQLITE_ABORT, SQLITE_OK
from numbox.core.bindings.sqlite.conn import sqlite3_changes
from numbox.core.bindings.sqlite.exec import sqlite3_exec, sqlite3_free
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import get_str_from_p_as_int
from test.auxiliary_utils import collect_and_run_tests


# The row callbacks are cfuncs at module level. sqlite3_exec calls a callback only while it runs. numba also keeps a
# cfunc's compiled code after the cfunc object is collected, so the module-level reference is a precaution, as the
# docstring of numbox.core.bindings.sqlite.hooks describes.

@cfunc(types.int32(types.voidptr, types.int32, types.intp, types.intp))
def _count_rows_cb(ctx, n, values_pp, names_pp):
    """Add 1 to the int64 at ``ctx`` for each row, and return 0, so that ``sqlite3_exec`` continues."""
    arr = carray(ctx, 1, dtype=np.int64)
    arr[0] += 1
    return 0


@cfunc(types.int32(types.voidptr, types.int32, types.intp, types.intp))
def _abort_cb(ctx, n, values_pp, names_pp):
    """Return 1, which makes ``sqlite3_exec`` stop and return ``SQLITE_ABORT``."""
    return 1


def test_exec_create_insert_null_callback(memory_db):
    """With a ``cb_p`` of 0, ``sqlite3_exec`` runs both statements of the text, and the INSERT changes one row."""
    with c_string("CREATE TABLE t(a INTEGER); INSERT INTO t VALUES (1);") as sql_p:
        assert sqlite3_exec(memory_db, sql_p, 0, 0, 0) == SQLITE_OK
    assert sqlite3_changes(memory_db) == 1


def test_exec_callback_collects_row_count(memory_db):
    with c_string("CREATE TABLE t(a INTEGER); INSERT INTO t VALUES (1), (2), (3);") as setup_p:
        sqlite3_exec(memory_db, setup_p, 0, 0, 0)

    ctx = np.zeros(1, dtype=np.int64)
    with c_string("SELECT a FROM t") as sel_p:
        rc = sqlite3_exec(memory_db, sel_p, _count_rows_cb.address,
                          ctx.ctypes.data, 0)
    assert rc == SQLITE_OK
    assert ctx[0] == 3


def test_exec_callback_can_abort(memory_db):
    with c_string("CREATE TABLE t(a INTEGER); INSERT INTO t VALUES (1);") as setup_p:
        sqlite3_exec(memory_db, setup_p, 0, 0, 0)

    with c_string("SELECT a FROM t") as sel_p:
        rc = sqlite3_exec(memory_db, sel_p, _abort_cb.address, 0, 0)
    assert rc == SQLITE_ABORT


def test_exec_invalid_sql_writes_errmsg(memory_db):
    errmsg_p = c_int64(0)
    with c_string("SELECT FROM WHERE") as sql_p:
        rc = sqlite3_exec(memory_db, sql_p, 0, 0, addressof(errmsg_p))
    assert rc != SQLITE_OK
    assert errmsg_p.value != 0
    msg = get_str_from_p_as_int(errmsg_p.value)
    assert "syntax" in msg.lower() or "error" in msg.lower(), msg
    sqlite3_free(errmsg_p.value)  # The caller of sqlite3_exec frees the message. sqlite3_free returns nothing.


if __name__ == "__main__":
    collect_and_run_tests(__name__)
