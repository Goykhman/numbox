"""SQLite UDF registration and context bindings.

sqlite3_create_function_v2 -- register scalar (xFunc) and aggregate (xStep/xFinal) UDFs.
sqlite3_create_window_function -- register window UDFs (xStep/xFinal/xValue/xInverse).
sqlite3_aggregate_context -- allocate per-group state for aggregate/window UDFs.
sqlite3_user_data -- retrieve pApp from context.
sqlite3_context_db_handle -- retrieve db pointer from context.

Callback function pointers are passed as intp obtained from @cfunc(...).address.
Pass 0 for NULL (no callback / no pApp / no xDestroy).
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy


__all__ = [
    "sqlite3_create_function_v2", "sqlite3_create_window_function",
    "sqlite3_aggregate_context", "sqlite3_user_data",
    "sqlite3_context_db_handle",
]


load_lib("sqlite3")


@proxy(signatures.get("sqlite3_create_function_v2"), jit_options=jit_options)
def sqlite3_create_function_v2(
        db, name_p, n_arg, e_text_rep, p_app,
        x_func, x_step, x_final, x_destroy):
    return _call_lib_func(
        "sqlite3_create_function_v2",
        (db, name_p, n_arg, e_text_rep, p_app,
         x_func, x_step, x_final, x_destroy))


@proxy(signatures.get("sqlite3_create_window_function"),
       jit_options=jit_options)
def sqlite3_create_window_function(
        db, name_p, n_arg, e_text_rep, p_app,
        x_step, x_final, x_value, x_inverse, x_destroy):
    return _call_lib_func(
        "sqlite3_create_window_function",
        (db, name_p, n_arg, e_text_rep, p_app,
         x_step, x_final, x_value, x_inverse, x_destroy))


@proxy(signatures.get("sqlite3_aggregate_context"), jit_options=jit_options)
def sqlite3_aggregate_context(ctx, n_bytes):
    return _call_lib_func("sqlite3_aggregate_context", (ctx, n_bytes))


@proxy(signatures.get("sqlite3_user_data"), jit_options=jit_options)
def sqlite3_user_data(ctx):
    return _call_lib_func("sqlite3_user_data", (ctx,))


@proxy(signatures.get("sqlite3_context_db_handle"), jit_options=jit_options)
def sqlite3_context_db_handle(ctx):
    return _call_lib_func("sqlite3_context_db_handle", (ctx,))
