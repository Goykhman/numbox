"""Bindings that open and close a SQLite connection and report its state.

The module binds ``sqlite3_open``, ``sqlite3_open_v2`` and ``sqlite3_close``, and the functions that report the
library's version, the error of the most recent failed call, a database's file name and read-only state, and the counts
of changed rows. SQLite added ``sqlite3_changes64`` and ``sqlite3_total_changes64`` in version 3.37, released in
November 2021. Older libraries lack them, such as the SQLite 3.34 that CPython 3.10 ships on Windows, or a system
SQLite on Linux or macOS older than 3.37. Each of the two bindings is a stub if the loaded library lacks its function,
as its docstring describes.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy, proxy_if_available


__all__ = [
    "sqlite3_open", "sqlite3_open_v2", "sqlite3_close",
    "sqlite3_libversion", "sqlite3_libversion_number",
    "sqlite3_errmsg", "sqlite3_errcode", "sqlite3_extended_errcode",
    "sqlite3_threadsafe",
    "sqlite3_db_handle", "sqlite3_db_filename", "sqlite3_db_readonly",
    "sqlite3_changes", "sqlite3_last_insert_rowid", "sqlite3_total_changes",
    "sqlite3_changes64", "sqlite3_total_changes64",
]


# load_lib caches the handle of each library, so every binding module of this subpackage calls it for "sqlite3" and
# receives the same handle; no module depends on another having loaded the library first. conn, column and value keep
# the handle, because proxy_if_available looks up their optional symbols in it.
_sqlite3_lib = load_lib("sqlite3")


@proxy(signatures.get("sqlite3_open"), jit_options=jit_options)
def sqlite3_open(filename_p, db_pp):
    """Open the database file named at ``filename_p`` and write the new connection's handle to ``db_pp``.

    C reference: `sqlite3_open <https://sqlite.org/c3ref/open.html>`_.

    ``filename_p`` is the address of a NUL-terminated UTF-8 file name, and ``":memory:"`` opens a private in-memory
    database. SQLite opens the file for reading and writing and creates it if it does not exist. ``db_pp`` is the
    address of an ``intp`` that receives the handle, such as ``ctypes.addressof`` of a ``ctypes.c_int64``. Returns
    ``SQLITE_OK``, or an error code that :func:`sqlite3_errmsg` explains. SQLite writes a handle even when the open
    fails, unless it cannot allocate one, and the caller closes it with :func:`sqlite3_close` in either case.
    """
    return _call_lib_func("sqlite3_open", (filename_p, db_pp))


@proxy(signatures.get("sqlite3_open_v2"), jit_options=jit_options)
def sqlite3_open_v2(filename_p, db_pp, flags, vfs_p):
    """Open the database file named at ``filename_p`` in the mode that ``flags`` sets; write the handle to ``db_pp``.

    C reference: `sqlite3_open_v2 <https://sqlite.org/c3ref/open.html>`_.

    ``flags`` holds ``SQLITE_OPEN_READONLY``, ``SQLITE_OPEN_READWRITE``, or ``SQLITE_OPEN_READWRITE`` with
    ``SQLITE_OPEN_CREATE``, and may add other ``SQLITE_OPEN_*`` flags of
    :mod:`~numbox.core.bindings.sqlite.constants` with a bitwise OR. ``vfs_p`` is the address of the name of a VFS,
    or 0 for the default one. The file name, the handle and the result code have the meaning that
    :func:`sqlite3_open` describes.
    """
    return _call_lib_func("sqlite3_open_v2", (filename_p, db_pp, flags, vfs_p))


@proxy(signatures.get("sqlite3_close"), jit_options=jit_options)
def sqlite3_close(db_p):
    """Close the connection ``db_p`` and free its resources, and return ``SQLITE_OK``.

    C reference: `sqlite3_close <https://sqlite.org/c3ref/close.html>`_.

    SQLite returns ``SQLITE_BUSY`` and leaves the connection open while a prepared statement on it is not finalized
    or a BLOB handle is not closed. It rolls back a transaction that is still open. A call with a handle of 0 does
    nothing.
    """
    return _call_lib_func("sqlite3_close", (db_p,))


@proxy(signatures.get("sqlite3_libversion"), jit_options=jit_options)
def sqlite3_libversion():
    """Return the address of the loaded library's version string, such as ``"3.45.1"``.

    C reference: `sqlite3_libversion <https://sqlite.org/c3ref/libversion.html>`_.

    The string is a constant of the library, valid for the life of the process. Python code reads it with
    :func:`~numbox.utils.lowlevel.get_str_from_p_as_int`.
    """
    return _call_lib_func("sqlite3_libversion")


@proxy(signatures.get("sqlite3_libversion_number"), jit_options=jit_options)
def sqlite3_libversion_number():
    """Return the loaded library's version as an integer, such as 3045001 for version 3.45.1.

    C reference: `sqlite3_libversion_number <https://sqlite.org/c3ref/libversion.html>`_.
    """
    return _call_lib_func("sqlite3_libversion_number")


@proxy(signatures.get("sqlite3_errmsg"), jit_options=jit_options)
def sqlite3_errmsg(db_p):
    """Return the address of SQLite's English message, in UTF-8, for the most recent error on the connection ``db_p``.

    C reference: `sqlite3_errmsg <https://sqlite.org/c3ref/errcode.html>`_.

    SQLite owns the text, and the next call on the connection may overwrite or free it, so copy it at once, as
    :func:`~numbox.utils.lowlevel.get_str_from_p_as_int` does. Returns 0 when no message is available.
    """
    return _call_lib_func("sqlite3_errmsg", (db_p,))


@proxy(signatures.get("sqlite3_errcode"), jit_options=jit_options)
def sqlite3_errcode(db_p):
    """Return the result code of the most recent failed call on the connection ``db_p``.

    C reference: `sqlite3_errcode <https://sqlite.org/c3ref/errcode.html>`_.

    The code is a primary one, such as ``SQLITE_CONSTRAINT``, unless the connection uses extended result codes. Any
    later call on the connection may change it.
    """
    return _call_lib_func("sqlite3_errcode", (db_p,))


@proxy(signatures.get("sqlite3_extended_errcode"), jit_options=jit_options)
def sqlite3_extended_errcode(db_p):
    """Return the extended result code of the most recent failed call on the connection ``db_p``.

    C reference: `sqlite3_extended_errcode <https://sqlite.org/c3ref/errcode.html>`_.

    The low 8 bits of an extended code are its primary code, one of the result codes of
    :mod:`~numbox.core.bindings.sqlite.constants`.
    """
    return _call_lib_func("sqlite3_extended_errcode", (db_p,))


@proxy(signatures.get("sqlite3_threadsafe"), jit_options=jit_options)
def sqlite3_threadsafe():
    """Return 0 if the loaded library was compiled without its mutexes, and a nonzero value otherwise.

    C reference: `sqlite3_threadsafe <https://sqlite.org/c3ref/threadsafe.html>`_.

    The value reports the compile-time setting ``SQLITE_THREADSAFE``, not a threading mode set later at run time.
    """
    return _call_lib_func("sqlite3_threadsafe")


@proxy(signatures.get("sqlite3_db_handle"), jit_options=jit_options)
def sqlite3_db_handle(stmt_p):
    """Return the handle of the connection on which the statement ``stmt_p`` was prepared.

    C reference: `sqlite3_db_handle <https://sqlite.org/c3ref/db_handle.html>`_.
    """
    return _call_lib_func("sqlite3_db_handle", (stmt_p,))


@proxy(signatures.get("sqlite3_db_filename"), jit_options=jit_options)
def sqlite3_db_filename(db_p, name_p):
    """Return the address of the absolute file name of the database named at ``name_p`` on the connection ``db_p``.

    C reference: `sqlite3_db_filename <https://sqlite.org/c3ref/db_filename.html>`_.

    ``name_p`` is the address of the database's name: ``"main"``, ``"temp"`` or the name of an attached database.
    Returns 0 or an empty string for a temporary or in-memory database, and for a name that the connection lacks. The
    connection owns the text, which stays valid until the database is detached or the connection closes.
    """
    return _call_lib_func("sqlite3_db_filename", (db_p, name_p))


@proxy(signatures.get("sqlite3_db_readonly"), jit_options=jit_options)
def sqlite3_db_readonly(db_p, name_p):
    """Return 1 if the database named at ``name_p`` is read-only, 0 if it is writable, and -1 if ``db_p`` lacks it.

    C reference: `sqlite3_db_readonly <https://sqlite.org/c3ref/db_readonly.html>`_.

    ``name_p`` is the address of the database's name, as for :func:`sqlite3_db_filename`.
    """
    return _call_lib_func("sqlite3_db_readonly", (db_p, name_p))


@proxy(signatures.get("sqlite3_changes"), jit_options=jit_options)
def sqlite3_changes(db_p):
    """Return the number of rows that the most recent completed INSERT, UPDATE or DELETE on ``db_p`` changed.

    C reference: `sqlite3_changes <https://sqlite.org/c3ref/changes.html>`_.

    The count excludes the rows that triggers, foreign key actions and REPLACE conflict resolution changed. Above
    the int32 range the value is undefined; :func:`sqlite3_changes64` returns it as an int64.
    """
    return _call_lib_func("sqlite3_changes", (db_p,))


@proxy(signatures.get("sqlite3_last_insert_rowid"), jit_options=jit_options)
def sqlite3_last_insert_rowid(db_p):
    """Return the rowid of the most recent successful INSERT into a rowid table on ``db_p``, or 0 if there was none.

    C reference: `sqlite3_last_insert_rowid <https://sqlite.org/c3ref/last_insert_rowid.html>`_.
    """
    return _call_lib_func("sqlite3_last_insert_rowid", (db_p,))


@proxy(signatures.get("sqlite3_total_changes"), jit_options=jit_options)
def sqlite3_total_changes(db_p):
    """Return the number of rows changed by every INSERT, UPDATE and DELETE on ``db_p`` since the connection opened.

    C reference: `sqlite3_total_changes <https://sqlite.org/c3ref/total_changes.html>`_.

    The count includes the rows that triggers and foreign key actions changed, and it excludes the rows that REPLACE
    conflict resolution changed. Above the int32 range the value is undefined; :func:`sqlite3_total_changes64` returns
    it as an int64.
    """
    return _call_lib_func("sqlite3_total_changes", (db_p,))


# SQLite has these two from version 3.37. proxy_if_available makes each a stub where the loaded library lacks it.
@proxy_if_available(_sqlite3_lib, signatures.get("sqlite3_changes64"), jit_options=jit_options)
def sqlite3_changes64(db_p):
    """Return the count of :func:`sqlite3_changes` as an int64.

    C reference: `sqlite3_changes64 <https://sqlite.org/c3ref/changes.html>`_.

    SQLite has the function from version 3.37. If the loaded library is older, this binding is a stub: a call from
    Python raises ``NotImplementedError``, and a jitted call fails to compile with a ``TypingError``. A stub has no
    ``as_func``, so ``hasattr(sqlite3_changes64, "as_func")`` is true only when the binding is present.
    """
    return _call_lib_func("sqlite3_changes64", (db_p,))


@proxy_if_available(_sqlite3_lib, signatures.get("sqlite3_total_changes64"), jit_options=jit_options)
def sqlite3_total_changes64(db_p):
    """Return the count of :func:`sqlite3_total_changes` as an int64.

    C reference: `sqlite3_total_changes64 <https://sqlite.org/c3ref/total_changes.html>`_.

    SQLite has the function from version 3.37. If the loaded library is older, this binding is a stub, as
    :func:`sqlite3_changes64` describes.
    """
    return _call_lib_func("sqlite3_total_changes64", (db_p,))
