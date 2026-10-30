"""Bindings that read a ``sqlite3_value``, such as an argument of a SQL function or a value of a virtual table's filter.

A SQL function's ``xFunc``, ``xStep`` and ``xInverse`` callbacks receive their arguments as ``argv_pp``, the address
of an array of ``sqlite3_value *``, and a virtual table's ``xFilter`` receives its values the same way. Each binding
of this module takes one element of that array, the ``intp`` at the argument's index. A value is valid while the
callback that received it runs, and it must be read in that callback's thread.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy, proxy_if_available


__all__ = [
    "sqlite3_value_int", "sqlite3_value_int64", "sqlite3_value_double",
    "sqlite3_value_text", "sqlite3_value_blob", "sqlite3_value_bytes",
    "sqlite3_value_type", "sqlite3_value_numeric_type",
    "sqlite3_value_nochange", "sqlite3_value_frombind",
    "sqlite3_value_subtype",
    "sqlite3_value_dup", "sqlite3_value_free",
]


_sqlite3_lib = load_lib("sqlite3")


@proxy(signatures.get("sqlite3_value_int"), jit_options=jit_options)
def sqlite3_value_int(value_p):
    """Return the ``sqlite3_value`` at ``value_p`` as an int32.

    C reference: `sqlite3_value_int <https://sqlite.org/c3ref/value_blob.html>`_.

    SQLite converts a value of another type, and NULL reads as 0. A value outside the int32 range loses its high bits;
    :func:`sqlite3_value_int64` reads all 64.
    """
    return _call_lib_func("sqlite3_value_int", (value_p,))


@proxy(signatures.get("sqlite3_value_int64"), jit_options=jit_options)
def sqlite3_value_int64(value_p):
    """Return the ``sqlite3_value`` at ``value_p`` as an int64.

    C reference: `sqlite3_value_int64 <https://sqlite.org/c3ref/value_blob.html>`_.

    NULL reads as 0. SQLite converts a REAL or a text value as SQL's ``CAST`` to ``INTEGER`` does.
    """
    return _call_lib_func("sqlite3_value_int64", (value_p,))


@proxy(signatures.get("sqlite3_value_double"), jit_options=jit_options)
def sqlite3_value_double(value_p):
    """Return the ``sqlite3_value`` at ``value_p`` as a float64.

    C reference: `sqlite3_value_double <https://sqlite.org/c3ref/value_blob.html>`_.

    NULL reads as 0.0. SQLite converts an integer or a text value as SQL's ``CAST`` to ``REAL`` does.
    """
    return _call_lib_func("sqlite3_value_double", (value_p,))


@proxy(signatures.get("sqlite3_value_text"), jit_options=jit_options)
def sqlite3_value_text(value_p):
    """Return the address of the ``sqlite3_value`` at ``value_p`` as NUL-terminated UTF-8 text, or 0 for NULL.

    C reference: `sqlite3_value_text <https://sqlite.org/c3ref/value_blob.html>`_.

    SQLite owns the text. A later call that converts the same value can free it, so call
    :func:`sqlite3_value_bytes` after this call for the length in bytes, and use the text in the same callback.
    """
    return _call_lib_func("sqlite3_value_text", (value_p,))


@proxy(signatures.get("sqlite3_value_blob"), jit_options=jit_options)
def sqlite3_value_blob(value_p):
    """Return the address of the bytes of the ``sqlite3_value`` at ``value_p``, or 0 for NULL and for an empty BLOB.

    C reference: `sqlite3_value_blob <https://sqlite.org/c3ref/value_blob.html>`_.

    SQLite owns the bytes. A later call that converts the same value can free them, so call
    :func:`sqlite3_value_bytes` after this call for their number, and use the bytes in the same callback.
    """
    return _call_lib_func("sqlite3_value_blob", (value_p,))


@proxy(signatures.get("sqlite3_value_bytes"), jit_options=jit_options)
def sqlite3_value_bytes(value_p):
    """Return the length in bytes of the ``sqlite3_value`` at ``value_p`` as a BLOB or as UTF-8 text, 0 for NULL.

    C reference: `sqlite3_value_bytes <https://sqlite.org/c3ref/value_blob.html>`_.

    Call it after :func:`sqlite3_value_text` or :func:`sqlite3_value_blob`, because either of them can convert the
    value and change its length.
    """
    return _call_lib_func("sqlite3_value_bytes", (value_p,))


@proxy(signatures.get("sqlite3_value_type"), jit_options=jit_options)
def sqlite3_value_type(value_p):
    """Return the type code of the ``sqlite3_value`` at ``value_p``, such as ``SQLITE_INTEGER`` or ``SQLITE_NULL``.

    C reference: `sqlite3_value_type <https://sqlite.org/c3ref/value_blob.html>`_.

    The code is ``SQLITE_INTEGER``, ``SQLITE_FLOAT``, ``SQLITE_TEXT``, ``SQLITE_BLOB`` or ``SQLITE_NULL``, and it
    describes the value before any conversion. A later call that converts the value may change the code that this
    binding returns, so read the type first.
    """
    return _call_lib_func("sqlite3_value_type", (value_p,))


@proxy(signatures.get("sqlite3_value_numeric_type"), jit_options=jit_options)
def sqlite3_value_numeric_type(value_p):
    """Convert the ``sqlite3_value`` at ``value_p`` to a number if that loses nothing, and return its type code after.

    C reference: `sqlite3_value_numeric_type <https://sqlite.org/c3ref/value_blob.html>`_.

    Text that reads as a number becomes ``SQLITE_INTEGER`` or ``SQLITE_FLOAT``. Any other value keeps its type.
    """
    return _call_lib_func("sqlite3_value_numeric_type", (value_p,))


@proxy(signatures.get("sqlite3_value_nochange"), jit_options=jit_options)
def sqlite3_value_nochange(value_p):
    """Return a nonzero value if, in a virtual table's ``xUpdate``, the UPDATE leaves the value's column unchanged.

    C reference: `sqlite3_value_nochange <https://sqlite.org/c3ref/value_blob.html>`_.

    Such a value reads as NULL in every other respect. Outside an ``xUpdate`` call, the result means nothing.
    """
    return _call_lib_func("sqlite3_value_nochange", (value_p,))


# SQLite has sqlite3_value_frombind from version 3.28. proxy_if_available makes the binding a stub where the loaded
# library lacks it.
@proxy_if_available(_sqlite3_lib, signatures.get("sqlite3_value_frombind"), jit_options=jit_options)
def sqlite3_value_frombind(value_p):
    """Return a nonzero value if the ``sqlite3_value`` at ``value_p`` came from a bound parameter, and 0 otherwise.

    C reference: `sqlite3_value_frombind <https://sqlite.org/c3ref/value_blob.html>`_.

    A value from a SQL literal, a table column or an expression returns 0. SQLite has the function from version 3.28.
    If the loaded library is older, this binding is a stub: a call from Python raises ``NotImplementedError``, and a
    jitted call fails to compile with a ``TypingError``.
    """
    return _call_lib_func("sqlite3_value_frombind", (value_p,))


@proxy(signatures.get("sqlite3_value_subtype"), jit_options=jit_options)
def sqlite3_value_subtype(value_p):
    """Return the subtype of the SQL function argument at ``value_p``, or 0 if it has none.

    C reference: `sqlite3_value_subtype <https://sqlite.org/c3ref/value_subtype.html>`_.

    Register the function that reads it with ``SQLITE_SUBTYPE``. Without that flag, the call may return 0 in some
    cases where the argument has a subtype.
    """
    return _call_lib_func("sqlite3_value_subtype", (value_p,))


@proxy(signatures.get("sqlite3_value_dup"), jit_options=jit_options)
def sqlite3_value_dup(value_p):
    """Return the address of a copy of the ``sqlite3_value`` at ``value_p``, or 0 if ``value_p`` is 0 or the copy fails.

    C reference: `sqlite3_value_dup <https://sqlite.org/c3ref/value_dup.html>`_.

    The copy outlives the callback, and the caller frees it with :func:`sqlite3_value_free`.
    """
    return _call_lib_func("sqlite3_value_dup", (value_p,))


@proxy(signatures.get("sqlite3_value_free"), jit_options=jit_options)
def sqlite3_value_free(value_p):
    """Free a ``sqlite3_value`` that :func:`sqlite3_value_dup` returned; a ``value_p`` of 0 is a no-op.

    C reference: `sqlite3_value_free <https://sqlite.org/c3ref/value_dup.html>`_.
    """
    return _call_lib_func("sqlite3_value_free", (value_p,))
