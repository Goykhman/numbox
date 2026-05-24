"""SQLite parameter-binding bindings.

The destructor arg in bind_text / bind_blob (last intp) is one of:
- SQLITE_STATIC = 0  -> SQLite assumes the buffer outlives the statement
- SQLITE_TRANSIENT = -1 -> SQLite makes a copy
- any other value -> a C function pointer SQLite calls to free the buffer

For numpy arrays passed via array_data_p, prefer SQLITE_TRANSIENT unless the
caller can guarantee the array outlives the prepared statement.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.proxy.proxy import proxy

__all__ = [
    "sqlite3_bind_int", "sqlite3_bind_int64", "sqlite3_bind_double",
    "sqlite3_bind_text", "sqlite3_bind_blob", "sqlite3_bind_null",
    "sqlite3_bind_parameter_count", "sqlite3_bind_parameter_index",
    "sqlite3_bind_parameter_name",
]


@proxy(signatures.get("sqlite3_bind_int"), jit_options={"cache": True})
def sqlite3_bind_int(stmt_p, idx, val):
    return _call_lib_func("sqlite3_bind_int", (stmt_p, idx, val))


@proxy(signatures.get("sqlite3_bind_int64"), jit_options={"cache": True})
def sqlite3_bind_int64(stmt_p, idx, val):
    return _call_lib_func("sqlite3_bind_int64", (stmt_p, idx, val))


@proxy(signatures.get("sqlite3_bind_double"), jit_options={"cache": True})
def sqlite3_bind_double(stmt_p, idx, val):
    return _call_lib_func("sqlite3_bind_double", (stmt_p, idx, val))


@proxy(signatures.get("sqlite3_bind_text"), jit_options={"cache": True})
def sqlite3_bind_text(stmt_p, idx, text_p, n, destructor):
    return _call_lib_func(
        "sqlite3_bind_text", (stmt_p, idx, text_p, n, destructor)
    )


@proxy(signatures.get("sqlite3_bind_blob"), jit_options={"cache": True})
def sqlite3_bind_blob(stmt_p, idx, data_p, n, destructor):
    return _call_lib_func(
        "sqlite3_bind_blob", (stmt_p, idx, data_p, n, destructor)
    )


@proxy(signatures.get("sqlite3_bind_null"), jit_options={"cache": True})
def sqlite3_bind_null(stmt_p, idx):
    return _call_lib_func("sqlite3_bind_null", (stmt_p, idx))


@proxy(signatures.get("sqlite3_bind_parameter_count"), jit_options={"cache": True})
def sqlite3_bind_parameter_count(stmt_p):
    return _call_lib_func("sqlite3_bind_parameter_count", (stmt_p,))


@proxy(signatures.get("sqlite3_bind_parameter_index"), jit_options={"cache": True})
def sqlite3_bind_parameter_index(stmt_p, name_p):
    return _call_lib_func("sqlite3_bind_parameter_index", (stmt_p, name_p))


@proxy(signatures.get("sqlite3_bind_parameter_name"), jit_options={"cache": True})
def sqlite3_bind_parameter_name(stmt_p, idx):
    return _call_lib_func("sqlite3_bind_parameter_name", (stmt_p, idx))
