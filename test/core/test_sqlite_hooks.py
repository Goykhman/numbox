"""Tests of the bindings of :mod:`~numbox.core.bindings.sqlite.hooks`.

Each test installs a hook on a connection to a database file and runs SQL that makes SQLite call it. The update hook
records the operations, the progress handler and the commit hook make their statements fail, and the rollback hook and
the trace callback count their calls. The busy handler is only installed, because a busy condition needs a second
connection that holds a lock. A callback's ``ctx`` is the address of a NumPy int64 array. The callback writes its record
into the array, the test reads it, and SQLite receives no Python object.
"""
from ctypes import addressof, c_int64

import numpy as np
import pytest
from numba import carray, cfunc, types

from numbox.core.bindings.sqlite.constants import SQLITE_INTERRUPT, SQLITE_OK, SQLITE_TRACE_STMT
from numbox.core.bindings.sqlite.hooks import (
    sqlite3_busy_handler, sqlite3_commit_hook, sqlite3_progress_handler, sqlite3_rollback_hook,
    sqlite3_trace_v2, sqlite3_update_hook,
)
from numbox.core.bindings.sqlite.conn import sqlite3_close, sqlite3_open
from numbox.core.bindings.sqlite.exec import sqlite3_exec
from numbox.utils.cstrings import c_string


# The callbacks are cfuncs at module level, so they stay referenced while their hooks are installed, as the docstring
# of numbox.core.bindings.sqlite.hooks recommends. numba keeps a cfunc's compiled code even after the object is
# collected, so the reference is a precaution.

@cfunc(types.void(types.voidptr, types.int32, types.intp, types.intp, types.int64))
def _update_cb(ctx, op, db_p, tbl_p, rowid):
    """Record ``op`` in the log ``ctx[0:64]`` at the index that ``ctx[64]`` holds, and increment ``ctx[64]``."""
    arr = carray(ctx, 128, dtype=np.int64)
    idx = arr[64]
    arr[idx] = op
    arr[64] = idx + 1


@cfunc(types.int32(types.voidptr))
def _progress_abort_cb(ctx):
    return 1  # A nonzero return value interrupts the statement.


@cfunc(types.int32(types.voidptr, types.int32))
def _busy_abort_cb(ctx, n):
    return 0  # A return value of 0 makes the waiting call fail with SQLITE_BUSY instead of trying again.


@cfunc(types.int32(types.voidptr))
def _commit_veto_cb(ctx):
    return 1  # A nonzero return value turns the commit into a rollback.


@cfunc(types.void(types.voidptr))
def _rollback_count_cb(ctx):
    arr = carray(ctx, 1, dtype=np.int64)
    arr[0] += 1


@cfunc(types.int32(types.uint32, types.voidptr, types.voidptr, types.voidptr))
def _trace_count_cb(mask, ctx, p, x):
    arr = carray(ctx, 1, dtype=np.int64)
    arr[0] += 1
    return 0


@pytest.fixture
def populated_db(tmp_path):
    """Yield a connection to a database file whose table ``t`` holds the integers 1, 2 and 3."""
    import sqlite3 as stdlib_sqlite3
    db_file = tmp_path / "hooks.sqlite"
    conn = stdlib_sqlite3.connect(str(db_file))
    conn.executescript(
        "CREATE TABLE t(a INTEGER);"
        "INSERT INTO t VALUES (1), (2), (3);"
    )
    conn.commit()
    conn.close()
    db_p = c_int64(0)
    with c_string(str(db_file)) as name_p:
        assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    yield db_p.value
    sqlite3_close(db_p.value)


def test_update_hook_records_ops(populated_db):
    # _update_cb logs each operation in ctx[0:64] and keeps the number of entries in ctx[64].
    ctx = np.zeros(128, dtype=np.int64)
    sqlite3_update_hook(populated_db, _update_cb.address, ctx.ctypes.data)
    with c_string("INSERT INTO t VALUES (99); DELETE FROM t WHERE a=99;") as sql_p:
        rc = sqlite3_exec(populated_db, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    # SQLite passes SQLITE_INSERT (18) for the INSERT and SQLITE_DELETE (9) for the DELETE; the constants module does
    # not define the two codes.
    assert ctx[64] == 2  # SQLite called the hook once for each changed row.
    assert ctx[0] == 18
    assert ctx[1] == 9


def test_progress_handler_aborts(populated_db):
    """A progress handler that returns a nonzero value makes ``sqlite3_exec`` fail with ``SQLITE_INTERRUPT``."""
    sqlite3_progress_handler(populated_db, 1, _progress_abort_cb.address, 0)
    with c_string("SELECT * FROM t") as sql_p:
        rc = sqlite3_exec(populated_db, sql_p, 0, 0, 0)
    assert rc == SQLITE_INTERRUPT


def test_busy_handler_registration_returns_ok(populated_db):
    # A busy condition needs a second connection that holds a lock on the file, so the test checks only that
    # sqlite3_busy_handler returns SQLITE_OK.
    rc = sqlite3_busy_handler(populated_db, _busy_abort_cb.address, 0)
    assert rc == SQLITE_OK


def test_commit_hook_vetoes(populated_db):
    sqlite3_commit_hook(populated_db, _commit_veto_cb.address, 0)
    with c_string("INSERT INTO t VALUES (42)") as sql_p:
        rc = sqlite3_exec(populated_db, sql_p, 0, 0, 0)
    # The commit hook turns the commit into a rollback, and sqlite3_exec returns SQLITE_CONSTRAINT, whose extended code
    # is SQLITE_CONSTRAINT_COMMITHOOK.
    assert rc != SQLITE_OK


def test_rollback_hook_fires(populated_db):
    ctx = np.zeros(1, dtype=np.int64)
    sqlite3_rollback_hook(populated_db, _rollback_count_cb.address, ctx.ctypes.data)
    with c_string("BEGIN; INSERT INTO t VALUES (77); ROLLBACK;") as sql_p:
        rc = sqlite3_exec(populated_db, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert ctx[0] == 1


def test_trace_v2_fires_for_stmt(populated_db):
    ctx = np.zeros(1, dtype=np.int64)
    rc = sqlite3_trace_v2(populated_db, SQLITE_TRACE_STMT,
                          _trace_count_cb.address, ctx.ctypes.data)
    assert rc == SQLITE_OK
    with c_string("SELECT 1") as sql_p:
        sqlite3_exec(populated_db, sql_p, 0, 0, 0)
    assert ctx[0] >= 1


if __name__ == "__main__":
    from test.auxiliary_utils import collect_and_run_tests
    collect_and_run_tests(__name__)
