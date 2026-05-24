"""SQLite BLOB incremental I/O bindings: blob_open / _close / _bytes / _read /
_write / _reopen.

All functions present in SQLite 3.4.0 (2007) except _reopen which arrived in
3.7.4 (2010). No version gating needed — far below the matrix floor of 3.34.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.proxy.proxy import proxy

__all__ = [
    "sqlite3_blob_open", "sqlite3_blob_close", "sqlite3_blob_bytes",
    "sqlite3_blob_read", "sqlite3_blob_write", "sqlite3_blob_reopen",
]


@proxy(signatures.get("sqlite3_blob_open"), jit_options={"cache": True})
def sqlite3_blob_open(db_p, db_name_p, table_p, col_p, rowid, flags, blob_pp):
    return _call_lib_func(
        "sqlite3_blob_open",
        (db_p, db_name_p, table_p, col_p, rowid, flags, blob_pp),
    )


@proxy(signatures.get("sqlite3_blob_close"), jit_options={"cache": True})
def sqlite3_blob_close(blob_p):
    return _call_lib_func("sqlite3_blob_close", (blob_p,))


@proxy(signatures.get("sqlite3_blob_bytes"), jit_options={"cache": True})
def sqlite3_blob_bytes(blob_p):
    return _call_lib_func("sqlite3_blob_bytes", (blob_p,))


@proxy(signatures.get("sqlite3_blob_read"), jit_options={"cache": True})
def sqlite3_blob_read(blob_p, buf_p, n, offset):
    return _call_lib_func("sqlite3_blob_read", (blob_p, buf_p, n, offset))


@proxy(signatures.get("sqlite3_blob_write"), jit_options={"cache": True})
def sqlite3_blob_write(blob_p, buf_p, n, offset):
    return _call_lib_func("sqlite3_blob_write", (blob_p, buf_p, n, offset))


@proxy(signatures.get("sqlite3_blob_reopen"), jit_options={"cache": True})
def sqlite3_blob_reopen(blob_p, new_rowid):
    return _call_lib_func("sqlite3_blob_reopen", (blob_p, new_rowid))
