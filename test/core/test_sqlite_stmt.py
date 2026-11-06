"""Tests of the bindings of :mod:`~numbox.core.bindings.sqlite.stmt`.

The tests prepare a statement, step it to its end, reset it and finalize it, and they read its SQL text, its
expanded text and whether it is busy. Invalid SQL fails to prepare and leaves no statement. The file also tests
``sqlite3_db_handle`` of :mod:`~numbox.core.bindings.sqlite.conn`, which returns the connection of a statement.
"""
from ctypes import addressof, c_int64

from numbox.core.bindings.sqlite.constants import SQLITE_DONE, SQLITE_OK, SQLITE_ROW
from numbox.core.bindings.sqlite.conn import sqlite3_db_handle
from numbox.core.bindings.sqlite.stmt import (
    sqlite3_expanded_sql, sqlite3_finalize, sqlite3_prepare_v2, sqlite3_reset, sqlite3_sql,
    sqlite3_step, sqlite3_stmt_busy,
)
from numbox.core.bindings.sqlite.exec import sqlite3_free
from numbox.utils.cstrings import c_string
from test.auxiliary_utils import collect_and_run_tests, str_from_p_as_int


def _prepare(db_p, sql):
    """Prepare ``sql`` on ``db_p``, check that the result code is ``SQLITE_OK``, and return the statement's handle."""
    stmt_p = c_int64(0)
    tail_p = c_int64(0)
    with c_string(sql) as sql_p:
        rc = sqlite3_prepare_v2(db_p, sql_p, -1, addressof(stmt_p), addressof(tail_p))
        assert rc == SQLITE_OK, f"prepare failed: rc={rc}"
    return stmt_p.value


def test_prepare_step_done_finalize_loop(memory_db):
    stmt_p = _prepare(memory_db, "SELECT 1")
    assert sqlite3_step(stmt_p) == SQLITE_ROW
    assert sqlite3_step(stmt_p) == SQLITE_DONE
    assert sqlite3_finalize(stmt_p) == SQLITE_OK


def test_reset_replays_query(memory_db):
    stmt_p = _prepare(memory_db, "SELECT 1")
    assert sqlite3_step(stmt_p) == SQLITE_ROW
    assert sqlite3_reset(stmt_p) == SQLITE_OK
    assert sqlite3_step(stmt_p) == SQLITE_ROW  # After the reset, the statement returns its row again.
    sqlite3_finalize(stmt_p)


def test_sql_returns_original_text(memory_db):
    original = "SELECT 1 AS one"
    stmt_p = _prepare(memory_db, original)
    sql_p = sqlite3_sql(stmt_p)
    assert str_from_p_as_int(sql_p) == original
    sqlite3_finalize(stmt_p)


def test_expanded_sql_substitutes_and_must_free(memory_db):
    """``sqlite3_expanded_sql`` returns text that the caller frees with ``sqlite3_free``.

    The statement has no parameters, so the expanded text equals the SQL, and the test does not check a substitution.
    """
    original = "SELECT 1"
    stmt_p = _prepare(memory_db, original)
    expanded_p = sqlite3_expanded_sql(stmt_p)
    assert expanded_p != 0
    assert str_from_p_as_int(expanded_p) == original
    sqlite3_free(expanded_p)
    sqlite3_finalize(stmt_p)


def test_stmt_busy_is_nonzero_mid_iteration(memory_db):
    """``sqlite3_stmt_busy`` returns 0 before the first step and a nonzero value after a step that returned a row."""
    stmt_p = _prepare(memory_db, "SELECT 1")
    assert sqlite3_stmt_busy(stmt_p) == 0
    sqlite3_step(stmt_p)
    assert sqlite3_stmt_busy(stmt_p) != 0
    sqlite3_finalize(stmt_p)


def test_prepare_invalid_sql_returns_error(memory_db):
    """For invalid SQL, ``sqlite3_prepare_v2`` returns an error code and writes 0 as the statement's handle."""
    stmt_p = c_int64(0)
    tail_p = c_int64(0)
    with c_string("SELECT FROM WHERE") as sql_p:
        rc = sqlite3_prepare_v2(memory_db, sql_p, -1,
                                addressof(stmt_p), addressof(tail_p))
    assert rc != SQLITE_OK
    assert stmt_p.value == 0


def test_db_handle_round_trip(memory_db):
    """``sqlite3_db_handle`` returns the connection on which the statement was prepared."""
    stmt_p = _prepare(memory_db, "SELECT 1")
    assert sqlite3_db_handle(stmt_p) == memory_db
    sqlite3_finalize(stmt_p)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
