"""Tests of the bindings of :mod:`~numbox.core.bindings.sqlite.conn`.

The tests read the library's version and threading mode, open and close connections, and read the error codes and
the message after a failed open. They also read a database's file name and read-only state, and the 64-bit change
counts of a new connection. The tests of ``sqlite3_changes64`` and ``sqlite3_total_changes64`` skip with a SQLite
older than 3.37, where the two bindings are stubs.
"""
from ctypes import addressof, c_int64

import pytest

from numbox.core.bindings.sqlite.constants import (
    SQLITE_CANTOPEN, SQLITE_OK, SQLITE_OPEN_CREATE, SQLITE_OPEN_READONLY, SQLITE_OPEN_READWRITE,
)
from numbox.core.bindings.sqlite.conn import (
    sqlite3_changes64, sqlite3_close, sqlite3_db_filename, sqlite3_db_readonly, sqlite3_errcode,
    sqlite3_errmsg, sqlite3_extended_errcode, sqlite3_libversion, sqlite3_libversion_number,
    sqlite3_open, sqlite3_open_v2, sqlite3_threadsafe, sqlite3_total_changes64,
)
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import get_str_from_p_as_int
from test.auxiliary_utils import collect_and_run_tests, str_from_p_as_int


def _open_memory():
    """Open a new in-memory database with ``sqlite3_open``, and return the connection's handle as an ``int``."""
    db_p = c_int64(0)
    with c_string(":memory:") as name_p:
        rc = sqlite3_open(name_p, addressof(db_p))
    assert rc == SQLITE_OK, f"sqlite3_open failed: rc={rc}"
    assert db_p.value != 0
    return db_p.value


def test_libversion_returns_dotted_string():
    version_p = sqlite3_libversion()
    version = str_from_p_as_int(version_p)
    assert "." in version, version


def test_libversion_number_returns_modern_int():
    n = sqlite3_libversion_number()
    # SQLite numbers version X.Y.Z as X * 1000000 + Y * 1000 + Z, so version 3.0.0 is 3000000.
    assert n >= 3_000_000, n


def test_open_close_memory_db():
    db_p = _open_memory()
    rc = sqlite3_close(db_p)
    assert rc == SQLITE_OK


def test_open_v2_with_create_flag(tmp_path):
    db_file = tmp_path / "create.sqlite"
    db_p = c_int64(0)
    with c_string(str(db_file)) as name_p:
        rc = sqlite3_open_v2(name_p, addressof(db_p),
                             SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, 0)
    assert rc == SQLITE_OK, rc
    assert db_p.value != 0
    assert sqlite3_close(db_p.value) == SQLITE_OK
    assert db_file.exists()


def test_open_v2_bad_path_returns_cantopen(tmp_path):
    """``sqlite3_open_v2`` returns ``SQLITE_CANTOPEN`` for a file in a missing directory.

    The handle that SQLite writes holds a message for the error, and the test closes the handle.
    """
    bad_path = tmp_path / "nonexistent_dir" / "x.sqlite"
    db_p = c_int64(0)
    with c_string(str(bad_path)) as name_p:
        rc = sqlite3_open_v2(name_p, addressof(db_p), SQLITE_OPEN_READONLY, 0)
    assert rc == SQLITE_CANTOPEN, rc
    # SQLite writes a handle even when the open fails, unless it cannot allocate one. The handle holds the error
    # message, and the caller closes it.
    if db_p.value != 0:
        errmsg_p = sqlite3_errmsg(db_p.value)
        assert get_str_from_p_as_int(errmsg_p)
        sqlite3_close(db_p.value)


def test_db_filename_returns_main_path(tmp_path):
    db_file = tmp_path / "named.sqlite"
    db_p = c_int64(0)
    with c_string(str(db_file)) as name_p:
        sqlite3_open_v2(name_p, addressof(db_p),
                        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, 0)
    with c_string("main") as main_p:
        got_p = sqlite3_db_filename(db_p.value, main_p)
    got = str_from_p_as_int(got_p)
    assert got == str(db_file), (got, str(db_file))
    sqlite3_close(db_p.value)


def test_db_readonly_zero_for_writable(tmp_path):
    db_file = tmp_path / "rw.sqlite"
    db_p = c_int64(0)
    with c_string(str(db_file)) as name_p:
        sqlite3_open_v2(name_p, addressof(db_p),
                        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, 0)
    with c_string("main") as main_p:
        assert sqlite3_db_readonly(db_p.value, main_p) == 0
    sqlite3_close(db_p.value)


def test_threadsafe_returns_nonzero():
    # sqlite3_threadsafe returns the compile-time SQLITE_THREADSAFE: 1 for the serialized mode, the default, and 2 for
    # the multi-thread mode. A library compiled with SQLITE_THREADSAFE=0, without mutexes, fails the test.
    assert sqlite3_threadsafe() in (1, 2)


def test_errcode_matches_errmsg_after_bad_open(tmp_path):
    """After a failed open, ``sqlite3_errcode`` returns the code of the open, and the extended code is nonzero."""
    bad_path = tmp_path / "nonexistent_dir" / "x.sqlite"
    db_p = c_int64(0)
    with c_string(str(bad_path)) as name_p:
        rc = sqlite3_open_v2(name_p, addressof(db_p), SQLITE_OPEN_READONLY, 0)
    assert rc != SQLITE_OK
    if db_p.value != 0:
        assert sqlite3_errcode(db_p.value) == rc
        # The low 8 bits of the extended code are the primary code, so the extended code is nonzero.
        assert sqlite3_extended_errcode(db_p.value) != 0
        sqlite3_close(db_p.value)


def test_changes64_when_available():
    if not hasattr(sqlite3_changes64, "as_func"):
        pytest.skip("sqlite3_changes64 not available (SQLite < 3.37)")
    db_p = _open_memory()
    n = sqlite3_changes64(db_p)
    assert n == 0  # No statement has run on the new connection.
    sqlite3_close(db_p)


def test_total_changes64_when_available():
    if not hasattr(sqlite3_total_changes64, "as_func"):
        pytest.skip("sqlite3_total_changes64 not available (SQLite < 3.37)")
    db_p = _open_memory()
    n = sqlite3_total_changes64(db_p)
    assert n == 0
    sqlite3_close(db_p)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
