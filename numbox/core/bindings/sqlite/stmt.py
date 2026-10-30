"""Bindings that prepare, step, reset and finalize a SQLite statement.

The module binds ``sqlite3_prepare_v2``, ``sqlite3_step``, ``sqlite3_reset`` and ``sqlite3_finalize``, and
``sqlite3_sql``, ``sqlite3_expanded_sql`` and ``sqlite3_stmt_busy``, which report a statement's text and state.
``sqlite3_expanded_sql`` returns a ``char *`` that the caller must free with ``sqlite3_free``, which
:mod:`~numbox.core.bindings.sqlite.exec` binds.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy


__all__ = [
    "sqlite3_prepare_v2", "sqlite3_finalize", "sqlite3_reset", "sqlite3_step",
    "sqlite3_sql", "sqlite3_expanded_sql", "sqlite3_stmt_busy",
]


load_lib("sqlite3")


@proxy(signatures.get("sqlite3_prepare_v2"), jit_options=jit_options)
def sqlite3_prepare_v2(db_p, sql_p, n_byte, stmt_pp, tail_pp):
    """Compile the first SQL statement of the text at ``sql_p`` on ``db_p`` and write its handle to ``stmt_pp``.

    C reference: `sqlite3_prepare_v2 <https://sqlite.org/c3ref/prepare.html>`_.

    ``sql_p`` is the address of UTF-8 SQL text, and ``n_byte`` its length in bytes, or -1 to read up to the first NUL.
    ``stmt_pp`` is the address of an ``intp`` that receives the statement's handle, or 0 after an error or for text
    that holds no statement. Unless ``tail_pp`` is 0, it receives the address of the text after the first statement.
    Returns ``SQLITE_OK`` or an error code. The caller finalizes the statement with :func:`sqlite3_finalize`.
    """
    return _call_lib_func("sqlite3_prepare_v2", (db_p, sql_p, n_byte, stmt_pp, tail_pp))


@proxy(signatures.get("sqlite3_finalize"), jit_options=jit_options)
def sqlite3_finalize(stmt_p):
    """Delete the prepared statement ``stmt_p``, and return ``SQLITE_OK`` or the error code of its last failed step.

    C reference: `sqlite3_finalize <https://sqlite.org/c3ref/finalize.html>`_.

    A handle of 0 is a no-op. Any use of the statement after this call is undefined, and can crash the process.
    """
    return _call_lib_func("sqlite3_finalize", (stmt_p,))


@proxy(signatures.get("sqlite3_reset"), jit_options=jit_options)
def sqlite3_reset(stmt_p):
    """Reset the statement ``stmt_p`` to run again from the start, and return ``SQLITE_OK`` or an error code.

    C reference: `sqlite3_reset <https://sqlite.org/c3ref/reset.html>`_.

    The parameters keep their bound values. The code is the error of the most recent step, if it failed. The reset
    itself can fail too, for example with ``SQLITE_BUSY`` when a statement stepped only once cannot commit, so check
    the code even when every step succeeded.
    """
    return _call_lib_func("sqlite3_reset", (stmt_p,))


@proxy(signatures.get("sqlite3_step"), jit_options=jit_options)
def sqlite3_step(stmt_p):
    """Run the statement ``stmt_p`` to its next result row, and return ``SQLITE_ROW``, ``SQLITE_DONE`` or an error code.

    C reference: `sqlite3_step <https://sqlite.org/c3ref/step.html>`_.

    After ``SQLITE_ROW``, the bindings of :mod:`~numbox.core.bindings.sqlite.column` read the row. After
    ``SQLITE_DONE`` or an error, the statement must be reset before it is stepped again. ``SQLITE_BUSY`` means that
    SQLite could not take a lock that it needs.
    """
    return _call_lib_func("sqlite3_step", (stmt_p,))


@proxy(signatures.get("sqlite3_sql"), jit_options=jit_options)
def sqlite3_sql(stmt_p):
    """Return the address of the UTF-8 SQL text from which the statement ``stmt_p`` was prepared.

    C reference: `sqlite3_sql <https://sqlite.org/c3ref/expanded_sql.html>`_.

    The statement owns the text, and SQLite frees it when the statement is finalized.
    """
    return _call_lib_func("sqlite3_sql", (stmt_p,))


# The binding returns SQLite's pointer as it is and leaves the free to the caller, as every other binding leaves
# ownership in view. A wrapper that freed the text itself would hide who owns the memory, so each call site frees it.
@proxy(signatures.get("sqlite3_expanded_sql"), jit_options=jit_options)
def sqlite3_expanded_sql(stmt_p):
    """Return the address of the SQL text of the statement ``stmt_p``, with the bound parameter values written in.

    C reference: `sqlite3_expanded_sql <https://sqlite.org/c3ref/expanded_sql.html>`_.

    SQLite allocates the text, and the caller frees it with ``sqlite3_free``. Returns 0 when memory runs out, when
    the text would exceed SQLite's length limit, and in a SQLite built with ``SQLITE_OMIT_TRACE``.
    """
    return _call_lib_func("sqlite3_expanded_sql", (stmt_p,))


@proxy(signatures.get("sqlite3_stmt_busy"), jit_options=jit_options)
def sqlite3_stmt_busy(stmt_p):
    """Return a nonzero value if ``stmt_p`` has been stepped but has neither finished nor been reset, and 0 otherwise.

    C reference: `sqlite3_stmt_busy <https://sqlite.org/c3ref/stmt_busy.html>`_.
    """
    return _call_lib_func("sqlite3_stmt_busy", (stmt_p,))
