"""BLOB incremental I/O tests for the SQLite buildout."""
from ctypes import addressof, c_int64

import numpy as np
import pytest

from numbox.core.bindings._sqlite_blob import (
    sqlite3_blob_bytes,
    sqlite3_blob_close,
    sqlite3_blob_open,
    sqlite3_blob_read,
    sqlite3_blob_reopen,
    sqlite3_blob_write,
)
from numbox.core.bindings._sqlite_conn import sqlite3_close, sqlite3_open
from numbox.core.bindings._sqlite_constants import (
    SQLITE_BLOB_READONLY,
    SQLITE_BLOB_READWRITE,
    SQLITE_OK,
)
from test.auxiliary_utils import cstr


@pytest.fixture
def populated_db(tmp_path):
    """Create a file-backed db (not :memory:, since stdlib + our bindings
    can't share an in-memory connection) with t(b BLOB) containing one row
    of known bytes at rowid 1."""
    import sqlite3 as stdlib_sqlite3
    db_file = tmp_path / "blob.sqlite"
    conn = stdlib_sqlite3.connect(str(db_file))
    conn.executescript(
        "CREATE TABLE t(b BLOB);"
        "INSERT INTO t(rowid, b) VALUES (1, x'01020304050607');"
        "INSERT INTO t(rowid, b) VALUES (2, x'AABBCCDD');"
    )
    conn.commit()
    conn.close()
    name_buf, name_p = cstr(str(db_file))
    db_p = c_int64(0)
    assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    yield db_p.value
    sqlite3_close(db_p.value)


def _blob_open(db_p, rowid, flags):
    # Retain buffer refs to keep them alive across the sqlite3_blob_open call —
    # binding to `_` only keeps the last call's buffer, and SQLite reads the
    # name strings inside the call, not just stores their addresses.
    main_buf, main_p = cstr("main")
    table_buf, table_p = cstr("t")
    col_buf, col_p = cstr("b")
    blob_p = c_int64(0)
    rc = sqlite3_blob_open(db_p, main_p, table_p, col_p, rowid, flags,
                           addressof(blob_p))
    return rc, blob_p.value


def test_blob_open_read_known_bytes(populated_db):
    rc, blob = _blob_open(populated_db, 1, SQLITE_BLOB_READONLY)
    assert rc == SQLITE_OK
    assert blob != 0
    n = sqlite3_blob_bytes(blob)
    assert n == 7
    buf = np.zeros(n, dtype=np.uint8)
    assert sqlite3_blob_read(blob, buf.ctypes.data, n, 0) == SQLITE_OK
    assert bytes(buf) == bytes(range(1, 8))
    assert sqlite3_blob_close(blob) == SQLITE_OK


def test_blob_write_then_reread(populated_db):
    rc, blob = _blob_open(populated_db, 1, SQLITE_BLOB_READWRITE)
    assert rc == SQLITE_OK
    new_bytes = np.array([0xFE] * 7, dtype=np.uint8)
    assert sqlite3_blob_write(blob, new_bytes.ctypes.data, 7, 0) == SQLITE_OK
    sqlite3_blob_close(blob)

    rc, blob = _blob_open(populated_db, 1, SQLITE_BLOB_READONLY)
    assert rc == SQLITE_OK
    got = np.zeros(7, dtype=np.uint8)
    sqlite3_blob_read(blob, got.ctypes.data, 7, 0)
    assert bytes(got) == bytes([0xFE] * 7)
    sqlite3_blob_close(blob)


def test_blob_bytes_matches_inserted(populated_db):
    rc, blob = _blob_open(populated_db, 2, SQLITE_BLOB_READONLY)
    assert rc == SQLITE_OK
    assert sqlite3_blob_bytes(blob) == 4
    sqlite3_blob_close(blob)


def test_blob_reopen_to_different_rowid(populated_db):
    rc, blob = _blob_open(populated_db, 1, SQLITE_BLOB_READONLY)
    assert rc == SQLITE_OK
    assert sqlite3_blob_bytes(blob) == 7
    assert sqlite3_blob_reopen(blob, 2) == SQLITE_OK
    assert sqlite3_blob_bytes(blob) == 4
    sqlite3_blob_close(blob)


def test_blob_open_bad_column_returns_error(populated_db):
    # Retain buffer refs (see _blob_open helper above).
    main_buf, main_p = cstr("main")
    table_buf, table_p = cstr("t")
    bad_col_buf, bad_col_p = cstr("nonexistent_column")
    blob_p = c_int64(0)
    rc = sqlite3_blob_open(populated_db, main_p, table_p, bad_col_p,
                           1, SQLITE_BLOB_READONLY, addressof(blob_p))
    assert rc != SQLITE_OK


def test_blob_close_returns_ok(populated_db):
    rc, blob = _blob_open(populated_db, 1, SQLITE_BLOB_READONLY)
    assert rc == SQLITE_OK
    assert sqlite3_blob_close(blob) == SQLITE_OK


if __name__ == "__main__":
    from test.auxiliary_utils import collect_and_run_tests
    collect_and_run_tests(__name__)
