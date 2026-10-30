"""Bindings that set the result of a SQL function, or the value of a virtual table's column.

A SQL function's ``xFunc``, ``xFinal`` and ``xValue`` callbacks report their result through these bindings, and a
virtual table's ``xColumn`` reports a cell's value the same way. Each binding takes, as its first argument, the
``sqlite3_context *`` that SQLite passed to the callback, and it must be called in that callback's thread. The text
and BLOB setters take a destructor after the byte count, an ``intp`` that tells SQLite what to do with the buffer:

- ``SQLITE_STATIC`` (0): SQLite uses the buffer in place, without a copy, and never frees it, so the buffer must stay
  valid and unchanged until SQLite has finished with the result.
- ``SQLITE_TRANSIENT`` (-1): SQLite copies the buffer before the call returns.
- Any other value: the address of a C function that SQLite calls to free the buffer when it has finished with it.
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
    """Set the result of the SQL function on ``ctx`` to the int32 ``val``.

    C reference: `sqlite3_result_int <https://sqlite.org/c3ref/result_blob.html>`_.

    numba converts ``val`` to int32 without a check, so a wider value keeps only its low 32 bits;
    :func:`sqlite3_result_int64` sets the whole value.
    """
    return _call_lib_func("sqlite3_result_int", (ctx, val))


@proxy(signatures.get("sqlite3_result_int64"), jit_options=jit_options)
def sqlite3_result_int64(ctx, val):
    """Set the result of the SQL function on ``ctx`` to the int64 ``val``.

    C reference: `sqlite3_result_int64 <https://sqlite.org/c3ref/result_blob.html>`_.
    """
    return _call_lib_func("sqlite3_result_int64", (ctx, val))


@proxy(signatures.get("sqlite3_result_double"), jit_options=jit_options)
def sqlite3_result_double(ctx, val):
    """Set the result of the SQL function on ``ctx`` to the float64 ``val``.

    C reference: `sqlite3_result_double <https://sqlite.org/c3ref/result_blob.html>`_.

    SQLite turns a NaN into SQL NULL, and an infinity stays a REAL.
    """
    return _call_lib_func("sqlite3_result_double", (ctx, val))


@proxy(signatures.get("sqlite3_result_text"), jit_options=jit_options)
def sqlite3_result_text(ctx, text_p, n_bytes, destructor):
    """Set the result of the SQL function on ``ctx`` to the UTF-8 text at ``text_p``, ``n_bytes`` long.

    C reference: `sqlite3_result_text <https://sqlite.org/c3ref/result_blob.html>`_.

    A negative ``n_bytes`` makes SQLite read up to the first NUL, and numba converts ``n_bytes`` to int32 without a
    check. A ``destructor`` of ``SQLITE_TRANSIENT`` makes SQLite copy the text before the call returns. With
    ``SQLITE_STATIC``, SQLite reads the text in place, and the caller keeps it alive and unchanged until SQLite has
    finished with the result. Any other value is the address of a C function that SQLite calls to free the text. A
    NUL before byte ``n_bytes`` stays in the text, and SQL expressions on such text give undefined results.
    """
    return _call_lib_func("sqlite3_result_text", (ctx, text_p, n_bytes, destructor))


@proxy(signatures.get("sqlite3_result_text64"), jit_options=jit_options)
def sqlite3_result_text64(ctx, text_p, n_bytes, destructor, encoding):
    """Set the result of the SQL function on ``ctx`` to the text at ``text_p``, ``n_bytes`` long, in ``encoding``.

    C reference: `sqlite3_result_text64 <https://sqlite.org/c3ref/result_blob.html>`_.

    ``encoding`` is ``SQLITE_UTF8`` for UTF-8 text. ``n_bytes`` is a uint64, so it must give the length: numba
    converts -1 to the largest uint64, and a length above 2**31 - 1 makes the function fail with ``SQLITE_TOOBIG``.
    ``destructor`` is as for :func:`sqlite3_result_text`.
    """
    return _call_lib_func("sqlite3_result_text64", (ctx, text_p, n_bytes, destructor, encoding))


@proxy(signatures.get("sqlite3_result_blob"), jit_options=jit_options)
def sqlite3_result_blob(ctx, data_p, n_bytes, destructor):
    """Set the result of the SQL function on ``ctx`` to a BLOB of the ``n_bytes`` bytes at ``data_p``.

    C reference: `sqlite3_result_blob <https://sqlite.org/c3ref/result_blob.html>`_.

    ``n_bytes`` must not be negative, and numba converts it to int32 without a check. ``destructor`` is as for
    :func:`sqlite3_result_text`: with ``SQLITE_STATIC`` the caller keeps the bytes alive and unchanged until SQLite
    has finished with the result.
    """
    return _call_lib_func("sqlite3_result_blob", (ctx, data_p, n_bytes, destructor))


@proxy(signatures.get("sqlite3_result_blob64"), jit_options=jit_options)
def sqlite3_result_blob64(ctx, data_p, n_bytes, destructor):
    """Set the result of the SQL function on ``ctx`` to a BLOB of the ``n_bytes`` bytes at ``data_p``, a uint64 length.

    C reference: `sqlite3_result_blob64 <https://sqlite.org/c3ref/result_blob.html>`_.

    A length above 2**31 - 1 makes the function fail with ``SQLITE_TOOBIG``. ``destructor`` is as for
    :func:`sqlite3_result_text`.
    """
    return _call_lib_func("sqlite3_result_blob64", (ctx, data_p, n_bytes, destructor))


@proxy(signatures.get("sqlite3_result_null"), jit_options=jit_options)
def sqlite3_result_null(ctx):
    """Set the result of the SQL function on ``ctx`` to SQL NULL.

    C reference: `sqlite3_result_null <https://sqlite.org/c3ref/result_blob.html>`_.
    """
    return _call_lib_func("sqlite3_result_null", (ctx,))


@proxy(signatures.get("sqlite3_result_error"), jit_options=jit_options)
def sqlite3_result_error(ctx, msg_p, n_bytes):
    """Make the SQL function on ``ctx`` fail with the UTF-8 error message at ``msg_p``, ``n_bytes`` long.

    C reference: `sqlite3_result_error <https://sqlite.org/c3ref/result_blob.html>`_.

    A negative ``n_bytes`` makes SQLite read the message up to the first NUL. SQLite copies the message before the
    call returns. The error code is ``SQLITE_ERROR`` unless a later :func:`sqlite3_result_error_code` sets another.
    """
    return _call_lib_func("sqlite3_result_error", (ctx, msg_p, n_bytes))


@proxy(signatures.get("sqlite3_result_error_nomem"), jit_options=jit_options)
def sqlite3_result_error_nomem(ctx):
    """Make the SQL function on ``ctx`` fail with ``SQLITE_NOMEM``, the error of a failed memory allocation.

    C reference: `sqlite3_result_error_nomem <https://sqlite.org/c3ref/result_blob.html>`_.
    """
    return _call_lib_func("sqlite3_result_error_nomem", (ctx,))


@proxy(signatures.get("sqlite3_result_error_toobig"), jit_options=jit_options)
def sqlite3_result_error_toobig(ctx):
    """Make the SQL function on ``ctx`` fail with ``SQLITE_TOOBIG``, the error of a string or BLOB that is too long.

    C reference: `sqlite3_result_error_toobig <https://sqlite.org/c3ref/result_blob.html>`_.
    """
    return _call_lib_func("sqlite3_result_error_toobig", (ctx,))


@proxy(signatures.get("sqlite3_result_error_code"), jit_options=jit_options)
def sqlite3_result_error_code(ctx, errcode):
    """Make the SQL function on ``ctx`` fail with the result code ``errcode``.

    C reference: `sqlite3_result_error_code <https://sqlite.org/c3ref/result_blob.html>`_.

    If no message was set, SQLite uses its standard message for the code. A later :func:`sqlite3_result_error` sets
    the code back to ``SQLITE_ERROR``, so call this binding after it.
    """
    return _call_lib_func("sqlite3_result_error_code", (ctx, errcode))


@proxy(signatures.get("sqlite3_result_subtype"), jit_options=jit_options)
def sqlite3_result_subtype(ctx, subtype):
    """Set the subtype of the result of the SQL function on ``ctx`` to the low 8 bits of ``subtype``.

    C reference: `sqlite3_result_subtype <https://sqlite.org/c3ref/result_subtype.html>`_.

    Register the function with ``SQLITE_RESULT_SUBTYPE``. If the flag is missing, the call can have no effect, and a
    SQLite built with ``-DSQLITE_STRICT_SUBTYPE=1`` makes the function fail with an error.
    """
    return _call_lib_func("sqlite3_result_subtype", (ctx, subtype))


@proxy(signatures.get("sqlite3_result_value"), jit_options=jit_options)
def sqlite3_result_value(ctx, value_p):
    """Set the result of the SQL function on ``ctx`` to a copy of the ``sqlite3_value`` at ``value_p``.

    C reference: `sqlite3_result_value <https://sqlite.org/c3ref/result_blob.html>`_.

    SQLite copies the value, so the value at ``value_p`` may change or be freed after the call.
    """
    return _call_lib_func("sqlite3_result_value", (ctx, value_p))


@proxy(signatures.get("sqlite3_result_zeroblob"), jit_options=jit_options)
def sqlite3_result_zeroblob(ctx, n):
    """Set the result of the SQL function on ``ctx`` to a BLOB of ``n`` zero bytes.

    C reference: `sqlite3_result_zeroblob <https://sqlite.org/c3ref/result_blob.html>`_.

    numba converts ``n`` to int32 without a check.
    """
    return _call_lib_func("sqlite3_result_zeroblob", (ctx, n))


@proxy(signatures.get("sqlite3_result_zeroblob64"), jit_options=jit_options)
def sqlite3_result_zeroblob64(ctx, n):
    """Set the result of the SQL function on ``ctx`` to a BLOB of ``n`` zero bytes, and return a result code.

    C reference: `sqlite3_result_zeroblob64 <https://sqlite.org/c3ref/result_blob.html>`_.

    Returns ``SQLITE_OK``, or ``SQLITE_TOOBIG`` for an ``n`` above SQLite's length limit, which also makes the
    function fail.
    """
    return _call_lib_func("sqlite3_result_zeroblob64", (ctx, n))
