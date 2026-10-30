"""Bindings for SQLite's incremental BLOB I/O, which reads and writes part of a stored value without loading the rest.

The module binds ``sqlite3_blob_open``, ``sqlite3_blob_close``, ``sqlite3_blob_bytes``, ``sqlite3_blob_read``,
``sqlite3_blob_write`` and ``sqlite3_blob_reopen``. Each of them exists in SQLite from version 3.4.0, released in
2007, except ``sqlite3_blob_reopen``, which SQLite added in version 3.7.4, released in 2010. Both versions are far
older than 3.34, the oldest SQLite in numbox's test matrix, so every binding of the module is present.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy


__all__ = [
    "sqlite3_blob_open", "sqlite3_blob_close", "sqlite3_blob_bytes",
    "sqlite3_blob_read", "sqlite3_blob_write", "sqlite3_blob_reopen",
]


load_lib("sqlite3")


@proxy(signatures.get("sqlite3_blob_open"), jit_options=jit_options)
def sqlite3_blob_open(db_p, db_name_p, table_p, col_p, rowid, flags, blob_pp):
    """Open the value in column ``col_p`` of row ``rowid`` for incremental I/O, and write the handle to ``blob_pp``.

    C reference: `sqlite3_blob_open <https://sqlite.org/c3ref/blob_open.html>`_.

    ``db_name_p``, ``table_p`` and ``col_p`` are the addresses of UTF-8 names. The database's name is ``"main"``,
    ``"temp"`` or the name of an attached database, not a file name. ``flags`` is ``SQLITE_BLOB_READONLY`` (0) or
    ``SQLITE_BLOB_READWRITE`` (1). Returns ``SQLITE_OK``, or ``SQLITE_ERROR`` when the database, the table, the column
    or the row does not exist, or when the value is neither a BLOB nor text. After any error but ``SQLITE_MISUSE``,
    ``blob_pp`` holds 0. The caller closes the handle with :func:`sqlite3_blob_close`.
    """
    return _call_lib_func(
        "sqlite3_blob_open",
        (db_p, db_name_p, table_p, col_p, rowid, flags, blob_pp),
    )


@proxy(signatures.get("sqlite3_blob_close"), jit_options=jit_options)
def sqlite3_blob_close(blob_p):
    """Close the BLOB handle ``blob_p``, and return ``SQLITE_OK`` or an error code.

    C reference: `sqlite3_blob_close <https://sqlite.org/c3ref/blob_close.html>`_.

    The handle is closed even when the call returns an error. If the handle was writable and the connection is in
    autocommit mode, the close can commit the transaction. A call with a handle of 0 does nothing.
    """
    return _call_lib_func("sqlite3_blob_close", (blob_p,))


@proxy(signatures.get("sqlite3_blob_bytes"), jit_options=jit_options)
def sqlite3_blob_bytes(blob_p):
    """Return the size in bytes of the value open on the BLOB handle ``blob_p``.

    C reference: `sqlite3_blob_bytes <https://sqlite.org/c3ref/blob_bytes.html>`_.

    Incremental I/O cannot change the size; an ``UPDATE`` statement can.
    """
    return _call_lib_func("sqlite3_blob_bytes", (blob_p,))


@proxy(signatures.get("sqlite3_blob_read"), jit_options=jit_options)
def sqlite3_blob_read(blob_p, buf_p, n, offset):
    """Copy ``n`` bytes of the value open on ``blob_p``, from byte ``offset``, into the buffer at ``buf_p``.

    C reference: `sqlite3_blob_read <https://sqlite.org/c3ref/blob_read.html>`_.

    Returns ``SQLITE_OK``, or ``SQLITE_ERROR`` with nothing read when ``offset + n`` exceeds the size or either is
    negative. Returns ``SQLITE_ABORT`` when a change to the row has expired the handle. numba converts ``n`` and
    ``offset`` to int32 without a check.
    """
    return _call_lib_func("sqlite3_blob_read", (blob_p, buf_p, n, offset))


@proxy(signatures.get("sqlite3_blob_write"), jit_options=jit_options)
def sqlite3_blob_write(blob_p, buf_p, n, offset):
    """Copy ``n`` bytes from the buffer at ``buf_p`` into the value open on ``blob_p``, from byte ``offset``.

    C reference: `sqlite3_blob_write <https://sqlite.org/c3ref/blob_write.html>`_.

    Returns ``SQLITE_OK``, ``SQLITE_READONLY`` for a handle opened read-only, ``SQLITE_ERROR`` with nothing written
    when ``offset + n`` exceeds the size or either is negative, or ``SQLITE_ABORT`` when a change to the row has
    expired the handle. The write cannot increase the size of the value. numba converts ``n`` and ``offset`` to int32
    without a check.
    """
    return _call_lib_func("sqlite3_blob_write", (blob_p, buf_p, n, offset))


@proxy(signatures.get("sqlite3_blob_reopen"), jit_options=jit_options)
def sqlite3_blob_reopen(blob_p, new_rowid):
    """Move the BLOB handle ``blob_p`` to row ``new_rowid`` of the same table and column, and return a result code.

    C reference: `sqlite3_blob_reopen <https://sqlite.org/c3ref/blob_reopen.html>`_.

    The new row must exist and hold a BLOB or text in the column. An error aborts the handle: every later read, write
    or move returns ``SQLITE_ABORT``, and :func:`sqlite3_blob_bytes` returns 0.
    """
    return _call_lib_func("sqlite3_blob_reopen", (blob_p, new_rowid))
