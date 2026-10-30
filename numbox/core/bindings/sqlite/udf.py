"""Bindings that register SQL functions on a SQLite connection and serve their callbacks.

``sqlite3_create_function_v2`` registers a scalar or an aggregate function, and ``sqlite3_create_window_function`` a
window function. The callbacks of such a function call ``sqlite3_aggregate_context`` for an aggregate's state,
``sqlite3_user_data`` for the pointer that the registration passed, and ``sqlite3_context_db_handle`` for the
connection. Each callback is the ``intp`` address of a C function, such as the ``.address`` of a function that numba's
``cfunc`` compiles, and 0 stands for an unused callback or pointer.
:func:`~numbox.core.bindings.sqlite.udf_helpers.register_aggregate` and
:func:`~numbox.core.bindings.sqlite.udf_helpers.register_window` generate the callbacks of an aggregate or a window
function whose state is a numba structref.
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
    """Register the scalar or aggregate SQL function named at ``name_p`` on the connection ``db``.

    C reference: `sqlite3_create_function_v2 <https://sqlite.org/c3ref/create_function.html>`_.

    A scalar function passes ``x_func`` and 0 for ``x_step`` and ``x_final``, and an aggregate passes ``x_step`` and
    ``x_final`` and 0 for ``x_func``; three zeros delete the function. ``x_func`` and ``x_step`` have the C type
    ``void (*)(sqlite3_context *, int argc, sqlite3_value **argv)``, and ``x_final`` the type
    ``void (*)(sqlite3_context *)``. ``n_arg`` is the number of SQL arguments, from 0 to 127, or -1 for any number.
    ``e_text_rep`` is ``SQLITE_UTF8``, combined with flags such as ``SQLITE_DETERMINISTIC`` by a bitwise OR. SQLite
    returns ``p_app`` from :func:`sqlite3_user_data`. Unless ``x_destroy`` is 0, SQLite calls it with ``p_app`` when
    the function is deleted or replaced, when the connection closes, and when the registration fails. Keep each
    ``cfunc`` referenced while the function is registered. Returns ``SQLITE_OK``, or an error code such as
    ``SQLITE_MISUSE`` for a name longer than 255 bytes.
    """
    return _call_lib_func(
        "sqlite3_create_function_v2",
        (db, name_p, n_arg, e_text_rep, p_app,
         x_func, x_step, x_final, x_destroy))


@proxy(signatures.get("sqlite3_create_window_function"),
       jit_options=jit_options)
def sqlite3_create_window_function(
        db, name_p, n_arg, e_text_rep, p_app,
        x_step, x_final, x_value, x_inverse, x_destroy):
    """Register the aggregate window SQL function named at ``name_p`` on the connection ``db``.

    C reference: `sqlite3_create_window_function <https://sqlite.org/c3ref/create_function.html>`_.

    ``x_step`` and ``x_final`` are required. ``x_value`` and ``x_inverse`` are both set for a window function, or both
    0 for a plain aggregate. ``x_step`` and ``x_inverse`` have the C type
    ``void (*)(sqlite3_context *, int argc, sqlite3_value **argv)``, and ``x_final`` and ``x_value`` the type
    ``void (*)(sqlite3_context *)``. The other arguments and the result codes are as for
    :func:`sqlite3_create_function_v2`.
    """
    return _call_lib_func(
        "sqlite3_create_window_function",
        (db, name_p, n_arg, e_text_rep, p_app,
         x_step, x_final, x_value, x_inverse, x_destroy))


@proxy(signatures.get("sqlite3_aggregate_context"), jit_options=jit_options)
def sqlite3_aggregate_context(ctx, n_bytes):
    """Return the address of the memory that holds the state of the current aggregate group on ``ctx``.

    C reference: `sqlite3_aggregate_context <https://sqlite.org/c3ref/aggregate_context.html>`_.

    The first call of a group with a positive ``n_bytes`` allocates that many bytes and zeroes them, and later calls
    return the same memory whatever their ``n_bytes``. A first call with an ``n_bytes`` of 0 or less returns 0, and so
    does a failed allocation. ``xFinal`` usually passes 0, because SQLite calls it once without any ``xStep`` when no
    row matches. SQLite frees the memory when the aggregate query ends.
    """
    return _call_lib_func("sqlite3_aggregate_context", (ctx, n_bytes))


@proxy(signatures.get("sqlite3_user_data"), jit_options=jit_options)
def sqlite3_user_data(ctx):
    """Return the ``p_app`` pointer that the registration of the SQL function on ``ctx`` passed.

    C reference: `sqlite3_user_data <https://sqlite.org/c3ref/user_data.html>`_.
    """
    return _call_lib_func("sqlite3_user_data", (ctx,))


@proxy(signatures.get("sqlite3_context_db_handle"), jit_options=jit_options)
def sqlite3_context_db_handle(ctx):
    """Return the handle of the connection on which the SQL function on ``ctx`` was registered.

    C reference: `sqlite3_context_db_handle <https://sqlite.org/c3ref/context_db_handle.html>`_.
    """
    return _call_lib_func("sqlite3_context_db_handle", (ctx,))
