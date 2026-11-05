"""Bindings that bind values to the parameters of a prepared SQLite statement.

Each binding takes the statement's handle and the parameter's index, which counts from 1, and returns ``SQLITE_OK``,
``SQLITE_RANGE`` for an index out of range, or ``SQLITE_MISUSE`` for a statement that has been stepped since its last
reset. ``sqlite3_bind_text`` and ``sqlite3_bind_blob`` take a destructor as their last argument, an ``intp`` that sets
how SQLite manages the lifetime of the buffer:

- ``SQLITE_STATIC`` (0): SQLite reads the buffer in place until the statement is finalized or the parameter is bound
  again, so the caller keeps the buffer alive and unchanged until then.
- ``SQLITE_TRANSIENT`` (-1): SQLite copies the buffer before the call returns.
- Any other value: the address of a C function that SQLite calls to free the buffer when SQLite no longer needs it.

For a NumPy array passed through ``array_data_p``, pass ``SQLITE_TRANSIENT`` unless the array outlives the prepared
statement.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy


__all__ = [
    "sqlite3_bind_int", "sqlite3_bind_int64", "sqlite3_bind_double",
    "sqlite3_bind_text", "sqlite3_bind_blob", "sqlite3_bind_null",
    "sqlite3_bind_parameter_count", "sqlite3_bind_parameter_index",
    "sqlite3_bind_parameter_name",
]


load_lib("sqlite3")


@proxy(signatures.get("sqlite3_bind_int"), jit_options=jit_options)
def sqlite3_bind_int(stmt_p, idx, val):
    """Bind the int32 ``val`` to the parameter ``idx`` of the statement ``stmt_p``, counted from 1.

    C reference: `sqlite3_bind_int <https://sqlite.org/c3ref/bind_blob.html>`_.

    numba converts ``val`` to int32 without a check, so a wider value is truncated to its low 32 bits; bind it with
    :func:`sqlite3_bind_int64` instead. Returns ``SQLITE_OK``, ``SQLITE_RANGE`` for an index out of range, or
    ``SQLITE_MISUSE`` if the statement has been stepped since its last reset.
    """
    return _call_lib_func("sqlite3_bind_int", (stmt_p, idx, val))


@proxy(signatures.get("sqlite3_bind_int64"), jit_options=jit_options)
def sqlite3_bind_int64(stmt_p, idx, val):
    """Bind the int64 ``val`` to the parameter ``idx`` of the statement ``stmt_p``, counted from 1.

    C reference: `sqlite3_bind_int64 <https://sqlite.org/c3ref/bind_blob.html>`_.

    Returns ``SQLITE_OK``, ``SQLITE_RANGE`` for an index out of range, or ``SQLITE_MISUSE`` if the statement has
    been stepped since its last reset.
    """
    return _call_lib_func("sqlite3_bind_int64", (stmt_p, idx, val))


@proxy(signatures.get("sqlite3_bind_double"), jit_options=jit_options)
def sqlite3_bind_double(stmt_p, idx, val):
    """Bind the float64 ``val`` to the parameter ``idx`` of the statement ``stmt_p``, counted from 1.

    C reference: `sqlite3_bind_double <https://sqlite.org/c3ref/bind_blob.html>`_.

    Returns ``SQLITE_OK``, ``SQLITE_RANGE`` for an index out of range, or ``SQLITE_MISUSE`` if the statement has
    been stepped since its last reset.
    """
    return _call_lib_func("sqlite3_bind_double", (stmt_p, idx, val))


@proxy(signatures.get("sqlite3_bind_text"), jit_options=jit_options)
def sqlite3_bind_text(stmt_p, idx, text_p, n, destructor):
    """Bind the UTF-8 text at ``text_p``, ``n`` bytes long, to the parameter ``idx`` of the statement ``stmt_p``.

    C reference: `sqlite3_bind_text <https://sqlite.org/c3ref/bind_blob.html>`_.

    A negative ``n`` makes SQLite read up to the first NUL, and numba converts ``n`` to int32 without a check. A
    ``destructor`` of ``SQLITE_TRANSIENT`` makes SQLite copy the text before the call returns. With ``SQLITE_STATIC``,
    SQLite reads the text in place, and the caller keeps it alive and unchanged until the statement is finalized or
    the parameter is bound again. Any other value is the address of a C function that SQLite calls to free the text.
    A ``text_p`` of 0 binds NULL. Returns ``SQLITE_OK``, ``SQLITE_RANGE`` for an index out of range, ``SQLITE_MISUSE``
    if the statement has been stepped since its last reset, or ``SQLITE_TOOBIG`` for text longer than SQLite's limit.
    """
    return _call_lib_func(
        "sqlite3_bind_text", (stmt_p, idx, text_p, n, destructor)
    )


@proxy(signatures.get("sqlite3_bind_blob"), jit_options=jit_options)
def sqlite3_bind_blob(stmt_p, idx, data_p, n, destructor):
    """Bind the ``n`` bytes at ``data_p`` as a BLOB to the parameter ``idx`` of the statement ``stmt_p``.

    C reference: `sqlite3_bind_blob <https://sqlite.org/c3ref/bind_blob.html>`_.

    ``n`` must not be negative, and numba converts it to int32 without a check. A ``destructor`` of
    ``SQLITE_TRANSIENT`` makes SQLite copy the bytes before the call returns. With ``SQLITE_STATIC``, SQLite reads the
    bytes in place, and the caller keeps them alive and unchanged until the statement is finalized or the parameter
    is bound again. Any other value is the address of a C function that SQLite calls to free the bytes. A ``data_p``
    of 0 binds NULL. Returns ``SQLITE_OK``, ``SQLITE_RANGE`` for an index out of range, ``SQLITE_MISUSE`` if the
    statement has been stepped since its last reset, or ``SQLITE_TOOBIG`` for a BLOB longer than SQLite's limit.
    """
    return _call_lib_func(
        "sqlite3_bind_blob", (stmt_p, idx, data_p, n, destructor)
    )


@proxy(signatures.get("sqlite3_bind_null"), jit_options=jit_options)
def sqlite3_bind_null(stmt_p, idx):
    """Bind SQL NULL to the parameter ``idx`` of the statement ``stmt_p``, counted from 1.

    C reference: `sqlite3_bind_null <https://sqlite.org/c3ref/bind_blob.html>`_.

    A parameter that is never bound is NULL as well. Returns ``SQLITE_OK``, ``SQLITE_RANGE`` for an index out of
    range, or ``SQLITE_MISUSE`` if the statement has been stepped since its last reset.
    """
    return _call_lib_func("sqlite3_bind_null", (stmt_p, idx))


@proxy(signatures.get("sqlite3_bind_parameter_count"), jit_options=jit_options)
def sqlite3_bind_parameter_count(stmt_p):
    """Return the largest parameter index of the statement ``stmt_p``, 0 if it has no parameters.

    C reference: `sqlite3_bind_parameter_count <https://sqlite.org/c3ref/bind_parameter_count.html>`_.

    The index equals the number of parameters unless parameters of the form ``?NNN`` leave gaps between indexes.
    """
    return _call_lib_func("sqlite3_bind_parameter_count", (stmt_p,))


@proxy(signatures.get("sqlite3_bind_parameter_index"), jit_options=jit_options)
def sqlite3_bind_parameter_index(stmt_p, name_p):
    """Return the index of the parameter of ``stmt_p`` named by the UTF-8 text at ``name_p``, or 0 if there is none.

    C reference: `sqlite3_bind_parameter_index <https://sqlite.org/c3ref/bind_parameter_index.html>`_.

    The name includes its first character, as in ``":rate"`` or ``"$rate"``.
    """
    return _call_lib_func("sqlite3_bind_parameter_index", (stmt_p, name_p))


@proxy(signatures.get("sqlite3_bind_parameter_name"), jit_options=jit_options)
def sqlite3_bind_parameter_name(stmt_p, idx):
    """Return the address of the UTF-8 name of the parameter ``idx`` of the statement ``stmt_p``, counted from 1.

    C reference: `sqlite3_bind_parameter_name <https://sqlite.org/c3ref/bind_parameter_name.html>`_.

    The name includes its first character, as in ``":rate"``. Returns 0 for a nameless ``?`` parameter and for an
    index out of range.
    """
    return _call_lib_func("sqlite3_bind_parameter_name", (stmt_p, idx))
