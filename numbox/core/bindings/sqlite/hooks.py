"""Bindings that install SQLite's callback hooks on a connection.

The module binds ``sqlite3_update_hook``, ``sqlite3_progress_handler``, ``sqlite3_busy_handler``,
``sqlite3_commit_hook``, ``sqlite3_rollback_hook`` and ``sqlite3_trace_v2``. Each takes the callback as the ``intp``
address of a C function, such as the ``.address`` of a function that numba's ``cfunc`` compiles. Each binding's
docstring gives the C type that its callback must have, and neither numbox nor SQLite checks that type. A connection
has at most one callback of each kind, and installing a callback replaces the previous one. Keep the ``cfunc`` object
referenced for as long as its hook is installed, for example at module level in the caller's code. numba keeps a
cfunc's compiled code for the life of the process even after the object is collected, but it does not document that
behavior.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy


__all__ = [
    "sqlite3_update_hook", "sqlite3_progress_handler", "sqlite3_busy_handler",
    "sqlite3_commit_hook", "sqlite3_rollback_hook", "sqlite3_trace_v2",
]


load_lib("sqlite3")


@proxy(signatures.get("sqlite3_update_hook"), jit_options=jit_options)
def sqlite3_update_hook(db_p, cb_p, ctx_p):
    """Install ``cb_p`` to be called for every row inserted, updated or deleted in a rowid table on ``db_p``.

    C reference: `sqlite3_update_hook <https://sqlite.org/c3ref/update_hook.html>`_.

    The callback has the C type ``void (*)(void *ctx, int op, const char *db, const char *table, sqlite3_int64)``.
    It receives ``ctx_p``, the operation, the addresses of the database's and the table's names, and the rowid. The
    operation is ``SQLITE_INSERT`` (18), ``SQLITE_DELETE`` (9) or ``SQLITE_UPDATE`` (23), three codes that
    :mod:`~numbox.core.bindings.sqlite.constants` does not define. The callback must not change the connection, and
    preparing or stepping a statement counts as a change. A ``cb_p`` of 0 removes the hook. Keep the ``cfunc`` whose
    address is ``cb_p`` referenced while the hook is installed. Returns the ``ctx_p`` of the previous call on the
    connection, or 0 for the first call.
    """
    return _call_lib_func("sqlite3_update_hook", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_progress_handler"), jit_options=jit_options)
def sqlite3_progress_handler(db_p, n_ops, cb_p, ctx_p):
    """Install ``cb_p`` to be called about every ``n_ops`` virtual machine instructions during long calls on ``db_p``.

    C reference: `sqlite3_progress_handler <https://sqlite.org/c3ref/progress_handler.html>`_.

    The callback has the C type ``int (*)(void *ctx)`` and receives ``ctx_p``. A nonzero return interrupts the
    operation. An ``n_ops`` below 1 or a ``cb_p`` of 0 removes the handler. The callback must not change the
    connection. Keep the ``cfunc`` whose address is ``cb_p`` referenced while the handler is installed.
    """
    return _call_lib_func("sqlite3_progress_handler", (db_p, n_ops, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_busy_handler"), jit_options=jit_options)
def sqlite3_busy_handler(db_p, cb_p, ctx_p):
    """Install ``cb_p`` to be called when ``db_p`` meets a table that another thread or process has locked.

    C reference: `sqlite3_busy_handler <https://sqlite.org/c3ref/busy_handler.html>`_.

    The callback has the C type ``int (*)(void *ctx, int count)``. It receives ``ctx_p`` and the number of earlier
    calls for the same lock. A return of 0 makes the waiting call fail with ``SQLITE_BUSY``, and a nonzero return
    retries. A ``cb_p`` of 0 removes the handler, and ``sqlite3_busy_timeout`` or ``PRAGMA busy_timeout`` replaces
    it. The callback must neither change the connection nor close it. Keep the ``cfunc`` whose address is ``cb_p``
    referenced while the handler is installed. Returns a result code.
    """
    return _call_lib_func("sqlite3_busy_handler", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_commit_hook"), jit_options=jit_options)
def sqlite3_commit_hook(db_p, cb_p, ctx_p):
    """Install ``cb_p`` to be called whenever a transaction on ``db_p`` commits.

    C reference: `sqlite3_commit_hook <https://sqlite.org/c3ref/commit_hook.html>`_.

    The callback has the C type ``int (*)(void *ctx)`` and receives ``ctx_p``. A nonzero return turns the commit into
    a rollback. The callback must not change the connection, and running any statement counts as a change. A
    ``cb_p`` of 0 removes the hook. Keep the ``cfunc`` whose address is ``cb_p`` referenced while the hook is
    installed. Returns the ``ctx_p`` of the previous call on the connection, or 0 for the first call.
    """
    return _call_lib_func("sqlite3_commit_hook", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_rollback_hook"), jit_options=jit_options)
def sqlite3_rollback_hook(db_p, cb_p, ctx_p):
    """Install ``cb_p`` to be called whenever a transaction on ``db_p`` rolls back.

    C reference: `sqlite3_rollback_hook <https://sqlite.org/c3ref/commit_hook.html>`_.

    The callback has the C type ``void (*)(void *ctx)`` and receives ``ctx_p``. SQLite does not call it for the
    rollback that closing the connection causes. The callback must not change the connection. A ``cb_p`` of 0 removes
    the hook. Keep the ``cfunc`` whose address is ``cb_p`` referenced while the hook is installed. Returns the
    ``ctx_p`` of the previous call on the connection, or 0 for the first call.
    """
    return _call_lib_func("sqlite3_rollback_hook", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_trace_v2"), jit_options=jit_options)
def sqlite3_trace_v2(db_p, mask, cb_p, ctx_p):
    """Install ``cb_p`` to be called for the trace events in ``mask`` on ``db_p``, and return a result code.

    C reference: `sqlite3_trace_v2 <https://sqlite.org/c3ref/trace_v2.html>`_.

    ``mask`` combines ``SQLITE_TRACE_STMT``, ``SQLITE_TRACE_PROFILE``, ``SQLITE_TRACE_ROW`` and
    ``SQLITE_TRACE_CLOSE`` with a bitwise OR. The callback has the C type
    ``int (*)(unsigned event, void *ctx, void *p, void *x)``. It receives the event, ``ctx_p`` and two pointers whose
    meaning depends on the event, and it should return 0. A ``mask`` of 0 or a ``cb_p`` of 0 turns tracing off. Keep
    the ``cfunc`` whose address is ``cb_p`` referenced while tracing is on.
    """
    return _call_lib_func("sqlite3_trace_v2", (db_p, mask, cb_p, ctx_p))
