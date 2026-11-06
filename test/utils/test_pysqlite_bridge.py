"""Tests for ``numbox.utils.pysqlite_bridge``, which reads the ``sqlite3 *`` handle of a Python ``sqlite3.Connection``.

The tests check that numbox's bindings and Python's ``sqlite3`` module report the same SQLite version. They check
that ``extract_connection_ptr`` returns a handle that numbox's bindings can use, for an in-memory and for a file
database. The function must raise ``TypeError`` for an object that is not a connection and ``RuntimeError`` when the
two versions differ. One test checks the head fields that ``_pyobject_head_fields`` returns when
``sys.gettotalrefcount`` exists. When the versions differ, the tests that pass a handle to SQLite skip, and the version
test skips on macOS and fails elsewhere.
"""
import ctypes
import sqlite3
from platform import system

import pytest

from numbox.utils.pysqlite_bridge import extract_connection_ptr, libraries_coordinated
from numbox.core.bindings.sqlite.conn import sqlite3_changes, sqlite3_libversion


def test_numbox_and_python_use_same_libsqlite3():
    """numbox's bindings and Python's ``sqlite3`` module report the same SQLite version.

    If they differ, a handle from ``extract_connection_ptr`` is unsafe to pass to numbox's bindings. The test skips on
    macOS, where the versions can differ unless ``DYLD_INSERT_LIBRARIES`` loads Python's SQLite first, as the module
    docstring of ``numbox.utils.pysqlite_bridge`` describes. On other platforms, a difference fails the test.
    """
    numbox_version = ctypes.c_char_p(sqlite3_libversion()).value.decode()
    if numbox_version != sqlite3.sqlite_version:
        if system() == "Darwin":
            pytest.skip(
                f"macOS: numbox resolves sqlite {numbox_version!r}, Python uses "
                f"{sqlite3.sqlite_version!r}.  Set DYLD_INSERT_LIBRARIES to align "
                "(see numbox.utils.pysqlite_bridge docstring)."
            )
        pytest.fail(
            f"numbox sees libsqlite3 {numbox_version!r}, Python sees "
            f"{sqlite3.sqlite_version!r}: libraries are not coordinated"
        )


def test_libraries_coordinated_reflects_version_match(monkeypatch):
    """``libraries_coordinated`` returns True when the two versions are equal and False when they differ."""
    numbox_version = ctypes.c_char_p(sqlite3_libversion()).value.decode()
    monkeypatch.setattr(sqlite3, "sqlite_version", numbox_version)
    assert libraries_coordinated() is True
    monkeypatch.setattr(sqlite3, "sqlite_version", "0.0.0-mismatch")
    assert libraries_coordinated() is False


def test_extract_connection_ptr_memory_db():
    if not libraries_coordinated():
        pytest.skip("uncoordinated libraries — extract_connection_ptr guards against use")
    conn = sqlite3.connect(":memory:")
    try:
        p = extract_connection_ptr(conn)
        assert isinstance(p, int)
        assert p != 0
    finally:
        conn.close()


def test_extract_connection_ptr_file_db(tmp_path):
    if not libraries_coordinated():
        pytest.skip("uncoordinated libraries — extract_connection_ptr guards against use")
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


def test_extract_connection_ptr_raises_on_uncoordinated_libraries(monkeypatch):
    """``extract_connection_ptr`` raises ``RuntimeError`` naming ``DYLD_INSERT_LIBRARIES`` when the versions differ.

    The check runs before the function calls ``sqlite3_errmsg`` through numbox's bindings, which could crash the
    process with a handle from another copy of SQLite. The test sets Python's version to another value, so the check
    runs on every platform.
    """
    monkeypatch.setattr(sqlite3, "sqlite_version", "0.0.0-mismatch")
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(RuntimeError, match="DYLD_INSERT_LIBRARIES"):
            extract_connection_ptr(conn)
    finally:
        conn.close()


def test_extracted_pointer_usable_with_numbox_bindings(tmp_path):
    """Called with the extracted handle, numbox's ``sqlite3_changes`` counts the 3 rows of the connection's last INSERT.

    The test skips when the versions differ, because a handle passed to another copy of SQLite can crash the process.
    """
    if not libraries_coordinated():
        pytest.skip(
            "numbox and Python use different libsqlite3 instances — "
            "see numbox.utils.pysqlite_bridge docstring for workaround"
        )
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


def test_pyobject_head_fields_handles_debug_build(monkeypatch):
    # A release build has no sys.gettotalrefcount, so the test adds the attribute. It checks that _pyobject_head_fields
    # then puts _ob_next and _ob_prev first. That is the function's rule, and the comment in the function says on which
    # builds the rule matches CPython's layout.
    import sys
    from numbox.utils.pysqlite_bridge import _pyobject_head_fields

    monkeypatch.setattr(sys, "gettotalrefcount", lambda: 0, raising=False)
    names = [name for name, _ in _pyobject_head_fields()]
    assert names[:2] == ["_ob_next", "_ob_prev"]
