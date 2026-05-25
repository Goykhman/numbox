"""Statement-lifecycle binding tests for the SQLite buildout."""
from ctypes import addressof, c_int64

from numbox.core.bindings import (
    SQLITE_DONE,
    SQLITE_OK,
    SQLITE_ROW,
    sqlite3_db_handle,
    sqlite3_expanded_sql,
    sqlite3_finalize,
    sqlite3_free,
    sqlite3_prepare_v2,
    sqlite3_reset,
    sqlite3_sql,
    sqlite3_step,
    sqlite3_stmt_busy,
)
from numbox.utils.cstrings import c_string
from test.auxiliary_utils import collect_and_run_tests, str_from_p_as_int


def _prepare(db_p, sql):
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
    assert sqlite3_step(stmt_p) == SQLITE_ROW  # replays
    sqlite3_finalize(stmt_p)


def test_sql_returns_original_text(memory_db):
    original = "SELECT 1 AS one"
    stmt_p = _prepare(memory_db, original)
    sql_p = sqlite3_sql(stmt_p)
    assert str_from_p_as_int(sql_p) == original
    sqlite3_finalize(stmt_p)


def test_expanded_sql_substitutes_and_must_free(memory_db):
    original = "SELECT 1"
    stmt_p = _prepare(memory_db, original)
    expanded_p = sqlite3_expanded_sql(stmt_p)
    assert expanded_p != 0
    assert str_from_p_as_int(expanded_p) == original
    sqlite3_free(expanded_p)
    sqlite3_finalize(stmt_p)


def test_stmt_busy_is_nonzero_mid_iteration(memory_db):
    stmt_p = _prepare(memory_db, "SELECT 1")
    assert sqlite3_stmt_busy(stmt_p) == 0  # before step
    sqlite3_step(stmt_p)                   # ROW
    assert sqlite3_stmt_busy(stmt_p) != 0  # mid-iteration
    sqlite3_finalize(stmt_p)


def test_prepare_invalid_sql_returns_error(memory_db):
    stmt_p = c_int64(0)
    tail_p = c_int64(0)
    with c_string("SELECT FROM WHERE") as sql_p:  # garbage
        rc = sqlite3_prepare_v2(memory_db, sql_p, -1,
                                addressof(stmt_p), addressof(tail_p))
    assert rc != SQLITE_OK
    assert stmt_p.value == 0  # no statement on failure


def test_db_handle_round_trip(memory_db):
    stmt_p = _prepare(memory_db, "SELECT 1")
    assert sqlite3_db_handle(stmt_p) == memory_db
    sqlite3_finalize(stmt_p)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
