"""SQLite value accessor bindings.

Read UDF arguments inside xFunc / xStep / xInverse callbacks. Each function
takes a sqlite3_value* (as intp) obtained by dereferencing the argv_pp array
at the appropriate index.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy


__all__ = [
    "sqlite3_value_int", "sqlite3_value_int64", "sqlite3_value_double",
    "sqlite3_value_text", "sqlite3_value_blob", "sqlite3_value_bytes",
    "sqlite3_value_type", "sqlite3_value_numeric_type",
    "sqlite3_value_nochange", "sqlite3_value_frombind",
    "sqlite3_value_subtype",
    "sqlite3_value_dup", "sqlite3_value_free",
]


load_lib("sqlite3")


@proxy(signatures.get("sqlite3_value_int"), jit_options=jit_options)
def sqlite3_value_int(value_p):
    return _call_lib_func("sqlite3_value_int", (value_p,))


@proxy(signatures.get("sqlite3_value_int64"), jit_options=jit_options)
def sqlite3_value_int64(value_p):
    return _call_lib_func("sqlite3_value_int64", (value_p,))


@proxy(signatures.get("sqlite3_value_double"), jit_options=jit_options)
def sqlite3_value_double(value_p):
    return _call_lib_func("sqlite3_value_double", (value_p,))


@proxy(signatures.get("sqlite3_value_text"), jit_options=jit_options)
def sqlite3_value_text(value_p):
    return _call_lib_func("sqlite3_value_text", (value_p,))


@proxy(signatures.get("sqlite3_value_blob"), jit_options=jit_options)
def sqlite3_value_blob(value_p):
    return _call_lib_func("sqlite3_value_blob", (value_p,))


@proxy(signatures.get("sqlite3_value_bytes"), jit_options=jit_options)
def sqlite3_value_bytes(value_p):
    return _call_lib_func("sqlite3_value_bytes", (value_p,))


@proxy(signatures.get("sqlite3_value_type"), jit_options=jit_options)
def sqlite3_value_type(value_p):
    return _call_lib_func("sqlite3_value_type", (value_p,))


@proxy(signatures.get("sqlite3_value_numeric_type"), jit_options=jit_options)
def sqlite3_value_numeric_type(value_p):
    return _call_lib_func("sqlite3_value_numeric_type", (value_p,))


@proxy(signatures.get("sqlite3_value_nochange"), jit_options=jit_options)
def sqlite3_value_nochange(value_p):
    return _call_lib_func("sqlite3_value_nochange", (value_p,))


@proxy(signatures.get("sqlite3_value_frombind"), jit_options=jit_options)
def sqlite3_value_frombind(value_p):
    return _call_lib_func("sqlite3_value_frombind", (value_p,))


@proxy(signatures.get("sqlite3_value_subtype"), jit_options=jit_options)
def sqlite3_value_subtype(value_p):
    return _call_lib_func("sqlite3_value_subtype", (value_p,))


@proxy(signatures.get("sqlite3_value_dup"), jit_options=jit_options)
def sqlite3_value_dup(value_p):
    return _call_lib_func("sqlite3_value_dup", (value_p,))


@proxy(signatures.get("sqlite3_value_free"), jit_options=jit_options)
def sqlite3_value_free(value_p):
    return _call_lib_func("sqlite3_value_free", (value_p,))
