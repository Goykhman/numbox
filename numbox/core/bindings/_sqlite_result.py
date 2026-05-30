"""SQLite result setter bindings.

Write the UDF return value inside xFunc / xFinal / xValue callbacks.
Each function takes a sqlite3_context* (as intp) as the first argument.

The destructor arg in result_text / result_blob (last intp before any
trailing args) is one of:
- SQLITE_STATIC = 0  -> SQLite assumes the buffer outlives the call
- SQLITE_TRANSIENT = -1 -> SQLite makes a copy
- any other value -> a C function pointer SQLite calls to free the buffer
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy


__all__ = [
    "sqlite3_result_int", "sqlite3_result_int64", "sqlite3_result_double",
    "sqlite3_result_text", "sqlite3_result_text64",
    "sqlite3_result_blob", "sqlite3_result_blob64",
    "sqlite3_result_null",
    "sqlite3_result_error", "sqlite3_result_error_nomem",
    "sqlite3_result_error_toobig", "sqlite3_result_error_code",
    "sqlite3_result_subtype", "sqlite3_result_value",
    "sqlite3_result_zeroblob", "sqlite3_result_zeroblob64",
]


load_lib("sqlite3")


@proxy(signatures.get("sqlite3_result_int"), jit_options=jit_options)
def sqlite3_result_int(ctx, val):
    return _call_lib_func("sqlite3_result_int", (ctx, val))


@proxy(signatures.get("sqlite3_result_int64"), jit_options=jit_options)
def sqlite3_result_int64(ctx, val):
    return _call_lib_func("sqlite3_result_int64", (ctx, val))


@proxy(signatures.get("sqlite3_result_double"), jit_options=jit_options)
def sqlite3_result_double(ctx, val):
    return _call_lib_func("sqlite3_result_double", (ctx, val))


@proxy(signatures.get("sqlite3_result_text"), jit_options=jit_options)
def sqlite3_result_text(ctx, text_p, n_bytes, destructor):
    return _call_lib_func("sqlite3_result_text", (ctx, text_p, n_bytes, destructor))


@proxy(signatures.get("sqlite3_result_text64"), jit_options=jit_options)
def sqlite3_result_text64(ctx, text_p, n_bytes, destructor, encoding):
    return _call_lib_func("sqlite3_result_text64", (ctx, text_p, n_bytes, destructor, encoding))


@proxy(signatures.get("sqlite3_result_blob"), jit_options=jit_options)
def sqlite3_result_blob(ctx, data_p, n_bytes, destructor):
    return _call_lib_func("sqlite3_result_blob", (ctx, data_p, n_bytes, destructor))


@proxy(signatures.get("sqlite3_result_blob64"), jit_options=jit_options)
def sqlite3_result_blob64(ctx, data_p, n_bytes, destructor):
    return _call_lib_func("sqlite3_result_blob64", (ctx, data_p, n_bytes, destructor))


@proxy(signatures.get("sqlite3_result_null"), jit_options=jit_options)
def sqlite3_result_null(ctx):
    return _call_lib_func("sqlite3_result_null", (ctx,))


@proxy(signatures.get("sqlite3_result_error"), jit_options=jit_options)
def sqlite3_result_error(ctx, msg_p, n_bytes):
    return _call_lib_func("sqlite3_result_error", (ctx, msg_p, n_bytes))


@proxy(signatures.get("sqlite3_result_error_nomem"), jit_options=jit_options)
def sqlite3_result_error_nomem(ctx):
    return _call_lib_func("sqlite3_result_error_nomem", (ctx,))


@proxy(signatures.get("sqlite3_result_error_toobig"), jit_options=jit_options)
def sqlite3_result_error_toobig(ctx):
    return _call_lib_func("sqlite3_result_error_toobig", (ctx,))


@proxy(signatures.get("sqlite3_result_error_code"), jit_options=jit_options)
def sqlite3_result_error_code(ctx, errcode):
    return _call_lib_func("sqlite3_result_error_code", (ctx, errcode))


@proxy(signatures.get("sqlite3_result_subtype"), jit_options=jit_options)
def sqlite3_result_subtype(ctx, subtype):
    return _call_lib_func("sqlite3_result_subtype", (ctx, subtype))


@proxy(signatures.get("sqlite3_result_value"), jit_options=jit_options)
def sqlite3_result_value(ctx, value_p):
    return _call_lib_func("sqlite3_result_value", (ctx, value_p))


@proxy(signatures.get("sqlite3_result_zeroblob"), jit_options=jit_options)
def sqlite3_result_zeroblob(ctx, n):
    return _call_lib_func("sqlite3_result_zeroblob", (ctx, n))


@proxy(signatures.get("sqlite3_result_zeroblob64"), jit_options=jit_options)
def sqlite3_result_zeroblob64(ctx, n):
    return _call_lib_func("sqlite3_result_zeroblob64", (ctx, n))
