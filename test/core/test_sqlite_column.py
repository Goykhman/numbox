"""Column accessor tests for the SQLite buildout."""
from ctypes import addressof, c_int64, c_ubyte

import pytest

from numbox.core.bindings import (
    SQLITE_BLOB,
    SQLITE_FLOAT,
    SQLITE_INTEGER,
    SQLITE_NULL,
    SQLITE_OK,
    SQLITE_ROW,
    SQLITE_TEXT,
    sqlite3_close,
    sqlite3_column_blob,
    sqlite3_column_bytes,
    sqlite3_column_count,
    sqlite3_column_database_name,
    sqlite3_column_decltype,
    sqlite3_column_double,
    sqlite3_column_int,
    sqlite3_column_int64,
    sqlite3_column_name,
    sqlite3_column_origin_name,
    sqlite3_column_table_name,
    sqlite3_column_text,
    sqlite3_column_type,
    sqlite3_finalize,
    sqlite3_open,
    sqlite3_prepare_v2,
    sqlite3_step,
)
from numbox.utils.cstrings import c_string
from test.auxiliary_utils import collect_and_run_tests, str_from_p_as_int


@pytest.fixture
def populated_table(tmp_path):
    """File db with t(i INT, big BIGINT, d REAL, s TEXT, b BLOB, n FLOAT)
    containing one row of known values."""
    import sqlite3 as stdlib_sqlite3
    db_file = tmp_path / "cols.sqlite"
    conn = stdlib_sqlite3.connect(str(db_file))
    conn.executescript(
        "CREATE TABLE t(i INT, big BIGINT, d REAL, s TEXT, b BLOB, n FLOAT);"
        "INSERT INTO t VALUES (42, 1099511627776, 3.14, 'hello', x'01020304', NULL);"
    )
    conn.commit()
    conn.close()
    db_p = c_int64(0)
    with c_string(str(db_file)) as name_p:
        assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    yield db_p.value
    sqlite3_close(db_p.value)


def _prepare_and_step(db_p, sql):
    stmt_p = c_int64(0)
    tail_p = c_int64(0)
    with c_string(sql) as sql_p:
        rc = sqlite3_prepare_v2(db_p, sql_p, -1, addressof(stmt_p), addressof(tail_p))
        assert rc == SQLITE_OK, f"prepare failed: rc={rc}"
    rc = sqlite3_step(stmt_p.value)
    assert rc == SQLITE_ROW
    return stmt_p.value


def test_column_int_int64_double_match_inserted(populated_table):
    stmt_p = _prepare_and_step(populated_table, "SELECT i, big, d FROM t")
    assert sqlite3_column_int(stmt_p, 0) == 42
    assert sqlite3_column_int64(stmt_p, 1) == 1099511627776
    assert abs(sqlite3_column_double(stmt_p, 2) - 3.14) < 1e-9
    sqlite3_finalize(stmt_p)


def test_column_text_decodes_utf8(populated_table):
    stmt_p = _prepare_and_step(populated_table, "SELECT s FROM t")
    text_p = sqlite3_column_text(stmt_p, 0)
    assert str_from_p_as_int(text_p) == "hello"
    sqlite3_finalize(stmt_p)


def test_column_blob_matches_inserted(populated_table):
    stmt_p = _prepare_and_step(populated_table, "SELECT b FROM t")
    blob_p = sqlite3_column_blob(stmt_p, 0)
    n = sqlite3_column_bytes(stmt_p, 0)
    assert n == 4
    # c_ubyte (not np.uint8) — np.uint8 is a numpy scalar dtype, not a ctypes type.
    buf = bytes((c_ubyte * n).from_address(blob_p))
    assert buf == bytes([1, 2, 3, 4])
    sqlite3_finalize(stmt_p)


def test_column_type_returns_type_codes(populated_table):
    stmt_p = _prepare_and_step(populated_table, "SELECT i, d, s, b, n FROM t")
    assert sqlite3_column_type(stmt_p, 0) == SQLITE_INTEGER
    assert sqlite3_column_type(stmt_p, 1) == SQLITE_FLOAT
    assert sqlite3_column_type(stmt_p, 2) == SQLITE_TEXT
    assert sqlite3_column_type(stmt_p, 3) == SQLITE_BLOB
    assert sqlite3_column_type(stmt_p, 4) == SQLITE_NULL
    sqlite3_finalize(stmt_p)


def test_column_count_matches_select_arity(populated_table):
    stmt_p = _prepare_and_step(populated_table, "SELECT i, big, d FROM t")
    assert sqlite3_column_count(stmt_p) == 3
    sqlite3_finalize(stmt_p)


def test_column_name_returns_label(populated_table):
    stmt_p = _prepare_and_step(populated_table, "SELECT i AS my_int FROM t")
    name_p = sqlite3_column_name(stmt_p, 0)
    assert str_from_p_as_int(name_p) == "my_int"
    sqlite3_finalize(stmt_p)


def test_column_decltype_returns_declared(populated_table):
    stmt_p = _prepare_and_step(populated_table, "SELECT i FROM t")
    dt_p = sqlite3_column_decltype(stmt_p, 0)
    # The declared type from CREATE TABLE was "INT"
    assert str_from_p_as_int(dt_p).upper() == "INT"
    sqlite3_finalize(stmt_p)


def test_column_database_name_when_available(populated_table):
    if not hasattr(sqlite3_column_database_name, "as_func"):
        pytest.skip("SQLITE_ENABLE_COLUMN_METADATA not in this SQLite build")
    stmt_p = _prepare_and_step(populated_table, "SELECT i FROM t")
    db_name_p = sqlite3_column_database_name(stmt_p, 0)
    assert str_from_p_as_int(db_name_p) == "main"
    sqlite3_finalize(stmt_p)


def test_column_table_name_when_available(populated_table):
    if not hasattr(sqlite3_column_table_name, "as_func"):
        pytest.skip("SQLITE_ENABLE_COLUMN_METADATA not in this SQLite build")
    stmt_p = _prepare_and_step(populated_table, "SELECT i FROM t")
    tbl_p = sqlite3_column_table_name(stmt_p, 0)
    assert str_from_p_as_int(tbl_p) == "t"
    sqlite3_finalize(stmt_p)


def test_column_origin_name_when_available(populated_table):
    if not hasattr(sqlite3_column_origin_name, "as_func"):
        pytest.skip("SQLITE_ENABLE_COLUMN_METADATA not in this SQLite build")
    stmt_p = _prepare_and_step(populated_table, "SELECT i FROM t")
    orig_p = sqlite3_column_origin_name(stmt_p, 0)
    assert str_from_p_as_int(orig_p) == "i"
    sqlite3_finalize(stmt_p)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
