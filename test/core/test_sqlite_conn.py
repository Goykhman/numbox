"""Connection + metadata binding tests for the SQLite buildout."""
from ctypes import addressof, c_int64

import pytest

from numbox.core.bindings._sqlite_conn import (
    sqlite3_changes64,
    sqlite3_close,
    sqlite3_db_filename,
    sqlite3_db_readonly,
    sqlite3_errcode,
    sqlite3_errmsg,
    sqlite3_extended_errcode,
    sqlite3_libversion,
    sqlite3_libversion_number,
    sqlite3_open,
    sqlite3_open_v2,
    sqlite3_threadsafe,
    sqlite3_total_changes64,
)
from numbox.core.bindings._sqlite_constants import (
    SQLITE_CANTOPEN,
    SQLITE_OK,
    SQLITE_OPEN_CREATE,
    SQLITE_OPEN_READONLY,
    SQLITE_OPEN_READWRITE,
)
from numbox.utils.lowlevel import get_str_from_p_as_int
from test.auxiliary_utils import collect_and_run_tests, cstr, str_from_p_as_int


def _open_memory():
    """Open ':memory:' via sqlite3_open. Returns the db_p as an int."""
    _, name_p = cstr(":memory:")
    db_p = c_int64(0)
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
    # SQLite 3.0.0 = 3_000_000; any modern build is far above this.
    assert n >= 3_000_000, n


def test_open_close_memory_db():
    db_p = _open_memory()
    rc = sqlite3_close(db_p)
    assert rc == SQLITE_OK


def test_open_v2_with_create_flag(tmp_path):
    db_file = tmp_path / "create.sqlite"
    _, name_p = cstr(str(db_file))
    db_p = c_int64(0)
    rc = sqlite3_open_v2(name_p, addressof(db_p),
                         SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, 0)
    assert rc == SQLITE_OK, rc
    assert db_p.value != 0
    assert sqlite3_close(db_p.value) == SQLITE_OK
    assert db_file.exists()


def test_open_v2_bad_path_returns_cantopen(tmp_path):
    bad_path = tmp_path / "nonexistent_dir" / "x.sqlite"
    _, name_p = cstr(str(bad_path))
    db_p = c_int64(0)
    rc = sqlite3_open_v2(name_p, addressof(db_p), SQLITE_OPEN_READONLY, 0)
    assert rc == SQLITE_CANTOPEN, rc
    # Even on failure, SQLite returns a (possibly bare) connection handle that
    # owns the errmsg. Caller must close it.
    if db_p.value != 0:
        errmsg_p = sqlite3_errmsg(db_p.value)
        assert get_str_from_p_as_int(errmsg_p)  # non-empty
        sqlite3_close(db_p.value)


def test_db_filename_returns_main_path(tmp_path):
    db_file = tmp_path / "named.sqlite"
    name_buf, name_p = cstr(str(db_file))
    db_p = c_int64(0)
    sqlite3_open_v2(name_p, addressof(db_p),
                    SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, 0)
    main_buf, main_p = cstr("main")
    got_p = sqlite3_db_filename(db_p.value, main_p)
    got = str_from_p_as_int(got_p)
    assert got == str(db_file), (got, str(db_file))
    sqlite3_close(db_p.value)


def test_db_readonly_zero_for_writable(tmp_path):
    db_file = tmp_path / "rw.sqlite"
    name_buf, name_p = cstr(str(db_file))
    db_p = c_int64(0)
    sqlite3_open_v2(name_p, addressof(db_p),
                    SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, 0)
    main_buf, main_p = cstr("main")
    assert sqlite3_db_readonly(db_p.value, main_p) == 0
    sqlite3_close(db_p.value)


def test_threadsafe_returns_nonzero():
    # Modern SQLite is always built with at least multi-thread (1) or
    # serialized (2) mode. Single-thread (0) is essentially extinct.
    assert sqlite3_threadsafe() in (1, 2)


def test_errcode_matches_errmsg_after_bad_open(tmp_path):
    bad_path = tmp_path / "nonexistent_dir" / "x.sqlite"
    _, name_p = cstr(str(bad_path))
    db_p = c_int64(0)
    rc = sqlite3_open_v2(name_p, addressof(db_p), SQLITE_OPEN_READONLY, 0)
    assert rc != SQLITE_OK
    if db_p.value != 0:
        assert sqlite3_errcode(db_p.value) == rc
        # extended_errcode may equal errcode or be a more specific code
        assert sqlite3_extended_errcode(db_p.value) != 0
        sqlite3_close(db_p.value)


def test_changes64_when_available():
    if not hasattr(sqlite3_changes64, "as_func"):
        pytest.skip("sqlite3_changes64 not available (SQLite < 3.37)")
    db_p = _open_memory()
    n = sqlite3_changes64(db_p)
    assert n == 0  # no statements executed yet
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
