import ctypes
import sqlite3

import pytest

from numbox.utils.pysqlite_bridge import extract_connection_ptr
from numbox.core.bindings._sqlite_conn import sqlite3_changes, sqlite3_libversion


def test_numbox_and_python_use_same_libsqlite3():
    """Library-coordination sanity check.

    numbox's ``sqlite3_libversion()`` and Python's ``sqlite3.sqlite_version``
    must report the same version string. If they disagree, the @njit
    bindings and Python's sqlite3 module are using different libsqlite3
    instances (typically the macOS Homebrew-static vs. system divergence
    documented in :mod:`numbox.utils.pysqlite_bridge`), and any
    ``sqlite3*`` extracted via :func:`extract_connection_ptr` would be
    unsafe to pass to numbox's bindings.
    """
    numbox_version = ctypes.c_char_p(sqlite3_libversion()).value.decode()
    assert numbox_version == sqlite3.sqlite_version, (
        f"numbox sees libsqlite3 {numbox_version!r}, Python sees "
        f"{sqlite3.sqlite_version!r}: libraries are not coordinated"
    )


def test_extract_connection_ptr_memory_db():
    conn = sqlite3.connect(":memory:")
    try:
        p = extract_connection_ptr(conn)
        assert isinstance(p, int)
        assert p != 0
    finally:
        conn.close()


def test_extract_connection_ptr_file_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    try:
        p = extract_connection_ptr(conn)
        assert isinstance(p, int)
        assert p != 0
    finally:
        conn.close()


def test_extract_connection_ptr_rejects_non_connection():
    with pytest.raises(TypeError, match="expected sqlite3.Connection"):
        extract_connection_ptr("not a connection")


def test_extracted_pointer_usable_with_numbox_bindings(tmp_path):
    """Cross-validation: extract pointer from Python's sqlite3.Connection,
    use it with numbox's @njit-callable bindings.

    Load-bearing test for the helper: if Python's sqlite and numbox's
    sqlite are different library instances (uncoordinated), this call
    crashes the interpreter (segfault). Library coordination is the
    point of :mod:`numbox.utils.pysqlite_bridge`.
    """
    db_path = tmp_path / "shared.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE t(x INTEGER)")
        conn.execute("INSERT INTO t VALUES (1), (2), (3)")
        conn.commit()
        p = extract_connection_ptr(conn)
        assert sqlite3_changes(p) == 3
    finally:
        conn.close()
