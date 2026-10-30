"""Bindings that read the current result row of a prepared SQLite statement and describe its columns.

Each binding takes the statement's handle and a column index that counts from 0. A binding that reads a value may be
called only after ``sqlite3_step`` returned ``SQLITE_ROW``, and before the statement is stepped again, reset or
finalized. ``sqlite3_column_database_name``, ``sqlite3_column_table_name`` and ``sqlite3_column_origin_name`` exist
only in a SQLite compiled with ``SQLITE_ENABLE_COLUMN_METADATA``. CPython's ``sqlite3.dll`` on Windows has them, and
numbox loads only that DLL on Windows. On Linux and macOS, numbox loads the system's SQLite, and whether that library
has them depends on the distribution. If the loaded library lacks one of them, its binding is a stub, as its docstring
describes. Every other binding of the module exists in every SQLite that numbox supports.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy, proxy_if_available


__all__ = [
    "sqlite3_column_int", "sqlite3_column_int64", "sqlite3_column_double",
    "sqlite3_column_text", "sqlite3_column_blob", "sqlite3_column_bytes",
    "sqlite3_column_type", "sqlite3_column_count", "sqlite3_column_name",
    "sqlite3_column_decltype",
    "sqlite3_column_database_name", "sqlite3_column_table_name",
    "sqlite3_column_origin_name",
]


_sqlite3_lib = load_lib("sqlite3")


@proxy(signatures.get("sqlite3_column_int"), jit_options=jit_options)
def sqlite3_column_int(stmt_p, idx):
    """Return the value of column ``idx`` of the current row of ``stmt_p``, counted from 0, as an int32.

    C reference: `sqlite3_column_int <https://sqlite.org/c3ref/column_blob.html>`_.

    SQLite converts a value of another type, and the binding returns 0 for NULL. A value outside the int32 range loses
    its high bits; :func:`sqlite3_column_int64` reads all 64 bits.
    """
    return _call_lib_func("sqlite3_column_int", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_int64"), jit_options=jit_options)
def sqlite3_column_int64(stmt_p, idx):
    """Return the value of column ``idx`` of the current row of ``stmt_p``, counted from 0, as an int64.

    C reference: `sqlite3_column_int64 <https://sqlite.org/c3ref/column_blob.html>`_.

    The binding returns 0 for NULL. SQLite converts a REAL or a text value as SQL's ``CAST`` to ``INTEGER`` does.
    """
    return _call_lib_func("sqlite3_column_int64", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_double"), jit_options=jit_options)
def sqlite3_column_double(stmt_p, idx):
    """Return the value of column ``idx`` of the current row of ``stmt_p``, counted from 0, as a float64.

    C reference: `sqlite3_column_double <https://sqlite.org/c3ref/column_blob.html>`_.

    The binding returns 0.0 for NULL. SQLite converts an integer or a text value as SQL's ``CAST`` to ``REAL`` does.
    """
    return _call_lib_func("sqlite3_column_double", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_text"), jit_options=jit_options)
def sqlite3_column_text(stmt_p, idx):
    """Return the address of the value of column ``idx`` as NUL-terminated UTF-8 text, or 0 for NULL.

    C reference: `sqlite3_column_text <https://sqlite.org/c3ref/column_blob.html>`_.

    SQLite owns the text. It stays valid until the statement is stepped, reset or finalized, or until another
    accessor converts the same column. Call :func:`sqlite3_column_bytes` after this call for the length in bytes.
    """
    return _call_lib_func("sqlite3_column_text", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_blob"), jit_options=jit_options)
def sqlite3_column_blob(stmt_p, idx):
    """Return the address of the bytes of column ``idx`` of the current row, or 0 for NULL and for an empty BLOB.

    C reference: `sqlite3_column_blob <https://sqlite.org/c3ref/column_blob.html>`_.

    SQLite owns the bytes. They stay valid until the statement is stepped, reset or finalized, or until another
    accessor converts the same column. Call :func:`sqlite3_column_bytes` after this call for their number.
    """
    return _call_lib_func("sqlite3_column_blob", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_bytes"), jit_options=jit_options)
def sqlite3_column_bytes(stmt_p, idx):
    """Return the length in bytes of column ``idx`` as a BLOB or as UTF-8 text, without a NUL, and 0 for NULL.

    C reference: `sqlite3_column_bytes <https://sqlite.org/c3ref/column_blob.html>`_.

    Call it after :func:`sqlite3_column_text` or :func:`sqlite3_column_blob`, because either of them can convert the
    value and change its length.
    """
    return _call_lib_func("sqlite3_column_bytes", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_type"), jit_options=jit_options)
def sqlite3_column_type(stmt_p, idx):
    """Return the type code of the value of column ``idx`` of the current row of ``stmt_p``, counted from 0.

    C reference: `sqlite3_column_type <https://sqlite.org/c3ref/column_blob.html>`_.

    The code is ``SQLITE_INTEGER``, ``SQLITE_FLOAT``, ``SQLITE_TEXT``, ``SQLITE_BLOB`` or ``SQLITE_NULL``, and it
    describes the value before any conversion. After another accessor converts the value, the result is undefined, so
    read the type first.
    """
    return _call_lib_func("sqlite3_column_type", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_count"), jit_options=jit_options)
def sqlite3_column_count(stmt_p):
    """Return the number of columns in the result of the statement ``stmt_p``, 0 for a statement that returns no data.

    C reference: `sqlite3_column_count <https://sqlite.org/c3ref/column_count.html>`_.

    A positive count does not mean that the statement returns any rows.
    """
    return _call_lib_func("sqlite3_column_count", (stmt_p,))


@proxy(signatures.get("sqlite3_column_name"), jit_options=jit_options)
def sqlite3_column_name(stmt_p, idx):
    """Return the address of the UTF-8 name of column ``idx`` of the result of the statement ``stmt_p``.

    C reference: `sqlite3_column_name <https://sqlite.org/c3ref/column_name.html>`_.

    The name is the one that the column's ``AS`` clause sets. If the column has no ``AS`` clause, SQLite does not
    specify its name. The text is valid until the statement is finalized, until its first step prepares it again, or
    until the next call for the same column. Returns 0 when an allocation fails.
    """
    return _call_lib_func("sqlite3_column_name", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_decltype"), jit_options=jit_options)
def sqlite3_column_decltype(stmt_p, idx):
    """Return the address of the declared type of column ``idx`` of the result, or 0 when the column is an expression.

    C reference: `sqlite3_column_decltype <https://sqlite.org/c3ref/column_decltype.html>`_.

    The UTF-8 text is the type in the table's definition. SQLite assigns a type to each value, not to each column, so a
    value need not have the declared type of its column.
    """
    return _call_lib_func("sqlite3_column_decltype", (stmt_p, idx))


# These three exist only in a SQLite compiled with SQLITE_ENABLE_COLUMN_METADATA. numbox loads only the sqlite3.dll
# that CPython bundles on Windows, and that DLL has them (_resolve_lib_path in numbox/core/bindings/utils.py).
# proxy_if_available makes each a stub where the loaded library lacks it.
@proxy_if_available(_sqlite3_lib, signatures.get("sqlite3_column_database_name"), jit_options=jit_options)
def sqlite3_column_database_name(stmt_p, idx):
    """Return the address of the name of the database from which column ``idx`` of the statement ``stmt_p`` comes.

    C reference: `sqlite3_column_database_name <https://sqlite.org/c3ref/column_database_name.html>`_.

    Returns 0 for a column that is an expression or a subquery. The UTF-8 text is valid until the statement is
    finalized or prepared again. SQLite has the function only when it is compiled with
    ``SQLITE_ENABLE_COLUMN_METADATA``. If the loaded library lacks it, this binding is a stub: a call from Python
    raises ``NotImplementedError``, and a jitted call fails to compile with a ``TypingError``. A stub has no
    ``as_func``, so ``hasattr(sqlite3_column_database_name, "as_func")`` is true only when the binding is present.
    """
    return _call_lib_func("sqlite3_column_database_name", (stmt_p, idx))


@proxy_if_available(_sqlite3_lib, signatures.get("sqlite3_column_table_name"), jit_options=jit_options)
def sqlite3_column_table_name(stmt_p, idx):
    """Return the address of the name of the table from which column ``idx`` of the statement ``stmt_p`` comes.

    C reference: `sqlite3_column_table_name <https://sqlite.org/c3ref/column_database_name.html>`_.

    Returns 0 for a column that is an expression or a subquery. The name is the table's original name, not an alias.
    If the loaded library lacks the function, this binding is a stub, as :func:`sqlite3_column_database_name`
    describes.
    """
    return _call_lib_func("sqlite3_column_table_name", (stmt_p, idx))


@proxy_if_available(_sqlite3_lib, signatures.get("sqlite3_column_origin_name"), jit_options=jit_options)
def sqlite3_column_origin_name(stmt_p, idx):
    """Return the address of the name of the table column from which column ``idx`` of the statement ``stmt_p`` comes.

    C reference: `sqlite3_column_origin_name <https://sqlite.org/c3ref/column_database_name.html>`_.

    Returns 0 for a column that is an expression or a subquery. The name is the column's original name, not an alias.
    If the loaded library lacks the function, this binding is a stub, as :func:`sqlite3_column_database_name`
    describes.
    """
    return _call_lib_func("sqlite3_column_origin_name", (stmt_p, idx))
