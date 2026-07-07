import ctypes
import sqlite3
from platform import system

import pytest

from numbox.utils.pysqlite_bridge import extract_connection_ptr, libraries_coordinated
from numbox.core.bindings.sqlite.conn import sqlite3_changes, sqlite3_libversion


def test_numbox_and_python_use_same_libsqlite3():
    """Library-coordination check.

    If numbox's bindings and Python's sqlite3 resolve to different libsqlite3
    instances, any pointer from :func:`extract_connection_ptr` is unsafe to
    pass to numbox's bindings.  On macOS this is expected unless the user
    sets ``DYLD_INSERT_LIBRARIES`` — see the
    :mod:`numbox.utils.pysqlite_bridge` module docstring.
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
    """:func:`libraries_coordinated` is a plain version-equality predicate."""
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
    """Guard: when numbox's bindings and Python's sqlite3 report different
    libsqlite3 versions, :func:`extract_connection_ptr` raises a helpful
    ``RuntimeError`` (pointing at the ``DYLD_INSERT_LIBRARIES`` workaround)
    instead of risking a segfault on its internal validation call.

    Forces a mismatch via monkeypatch so the guard is exercised on every
    platform, not just an uncoordinated macOS host.
    """
    monkeypatch.setattr(sqlite3, "sqlite_version", "0.0.0-mismatch")
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(RuntimeError, match="DYLD_INSERT_LIBRARIES"):
            extract_connection_ptr(conn)
    finally:
        conn.close()


def test_extracted_pointer_usable_with_numbox_bindings(tmp_path):
    """Cross-validation: extract pointer from Python's sqlite3.Connection,
    use it with numbox's @njit-callable bindings.

    Skipped when libraries are uncoordinated — passing the pointer to a
    different sqlite instance would segfault (see the
    :mod:`numbox.utils.pysqlite_bridge` docstring for the macOS workaround).
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
    # On a Py_DEBUG build sys.gettotalrefcount exists and CPython prepends the
    # _ob_next / _ob_prev trace pointers to PyObject_HEAD; _pyobject_head_fields
    # must include them so the db-field offset stays correct. A release build
    # lacks the attribute, so simulate a debug build by adding it.
    import sys
    from numbox.utils.pysqlite_bridge import _pyobject_head_fields

    monkeypatch.setattr(sys, "gettotalrefcount", lambda: 0, raising=False)
    names = [name for name, _ in _pyobject_head_fields()]
    assert names[:2] == ["_ob_next", "_ob_prev"]
