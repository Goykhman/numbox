"""SQLite column accessors.

Three metadata accessors (column_database_name / column_table_name /
column_origin_name) require SQLite to be compiled with
SQLITE_ENABLE_COLUMN_METADATA. CPython's bundled sqlite3 has this enabled,
but external sqlite3.dlls on user PATH may not. proxy_if_available stubs
them out when absent so callers can hasattr-guard or fall back.

All other accessors are universally available across the matrix.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import get_loaded_lib
from numbox.core.proxy.proxy import proxy, proxy_if_available

__all__ = [
    "sqlite3_column_int", "sqlite3_column_int64", "sqlite3_column_double",
    "sqlite3_column_text", "sqlite3_column_blob", "sqlite3_column_bytes",
    "sqlite3_column_type", "sqlite3_column_count", "sqlite3_column_name",
    "sqlite3_column_decltype",
    "sqlite3_column_database_name", "sqlite3_column_table_name",
    "sqlite3_column_origin_name",
]


_sqlite3_lib = get_loaded_lib("sqlite3")


@proxy(signatures.get("sqlite3_column_int"), jit_options={"cache": True})
def sqlite3_column_int(stmt_p, idx):
    return _call_lib_func("sqlite3_column_int", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_int64"), jit_options={"cache": True})
def sqlite3_column_int64(stmt_p, idx):
    return _call_lib_func("sqlite3_column_int64", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_double"), jit_options={"cache": True})
def sqlite3_column_double(stmt_p, idx):
    return _call_lib_func("sqlite3_column_double", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_text"), jit_options={"cache": True})
def sqlite3_column_text(stmt_p, idx):
    return _call_lib_func("sqlite3_column_text", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_blob"), jit_options={"cache": True})
def sqlite3_column_blob(stmt_p, idx):
    return _call_lib_func("sqlite3_column_blob", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_bytes"), jit_options={"cache": True})
def sqlite3_column_bytes(stmt_p, idx):
    return _call_lib_func("sqlite3_column_bytes", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_type"), jit_options={"cache": True})
def sqlite3_column_type(stmt_p, idx):
    return _call_lib_func("sqlite3_column_type", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_count"), jit_options={"cache": True})
def sqlite3_column_count(stmt_p):
    return _call_lib_func("sqlite3_column_count", (stmt_p,))


@proxy(signatures.get("sqlite3_column_name"), jit_options={"cache": True})
def sqlite3_column_name(stmt_p, idx):
    return _call_lib_func("sqlite3_column_name", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_decltype"), jit_options={"cache": True})
def sqlite3_column_decltype(stmt_p, idx):
    return _call_lib_func("sqlite3_column_decltype", (stmt_p, idx))


# Compile-flag-gated (SQLITE_ENABLE_COLUMN_METADATA)
@proxy_if_available(_sqlite3_lib, signatures.get("sqlite3_column_database_name"), jit_options={"cache": True})
def sqlite3_column_database_name(stmt_p, idx):
    return _call_lib_func("sqlite3_column_database_name", (stmt_p, idx))


@proxy_if_available(_sqlite3_lib, signatures.get("sqlite3_column_table_name"), jit_options={"cache": True})
def sqlite3_column_table_name(stmt_p, idx):
    return _call_lib_func("sqlite3_column_table_name", (stmt_p, idx))


@proxy_if_available(_sqlite3_lib, signatures.get("sqlite3_column_origin_name"), jit_options={"cache": True})
def sqlite3_column_origin_name(stmt_p, idx):
    return _call_lib_func("sqlite3_column_origin_name", (stmt_p, idx))
