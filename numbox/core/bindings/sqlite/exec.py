"""Bindings for ``sqlite3_exec``, which runs SQL text in one call, and for SQLite's memory allocator.

``sqlite3_exec`` prepares, steps and finalizes each statement of a SQL text, which may hold several statements. It can
call a C function once for each result row, with this type::

    int (*callback)(void *ctx, int ncol, char **col_values, char **col_names)

``ctx`` is the pointer that the caller passed to ``sqlite3_exec``. ``col_values`` and ``col_names`` are arrays of
``ncol`` addresses of NUL-terminated UTF-8 strings, valid only during the call, and a NULL value has the address 0. The
callback returns 0 to continue. Any other value stops ``sqlite3_exec``, which then returns ``SQLITE_ABORT``. numba's
``cfunc`` compiles such a function, and its ``.address`` is the value to pass. Here the callback counts the rows into
a NumPy array::

    from ctypes import addressof, c_int64

    import numpy as np
    from numba import carray, cfunc, types

    from numbox.core.bindings.sqlite.conn import sqlite3_close, sqlite3_open
    from numbox.core.bindings.sqlite.exec import sqlite3_exec
    from numbox.utils.cstrings import c_string


    @cfunc(types.int32(types.voidptr, types.int32, types.intp, types.intp))
    def count_rows(ctx, ncol, col_values, col_names):
        count = carray(ctx, 1, dtype=np.int64)
        count[0] += 1
        return 0


    db_p = c_int64(0)
    with c_string(":memory:") as path_p:
        sqlite3_open(path_p, addressof(db_p))
    count = np.zeros(1, dtype=np.int64)
    with c_string("SELECT 1 UNION ALL SELECT 2") as sql_p:
        rc = sqlite3_exec(db_p.value, sql_p, count_rows.address, count.ctypes.data, 0)
    print(rc, count[0])   # 0 2
    sqlite3_close(db_p.value)

``sqlite3_free`` releases memory that SQLite allocated for the caller, such as the error message of ``sqlite3_exec``
and the text of ``sqlite3_expanded_sql``. ``sqlite3_malloc`` allocates such memory; numbox's virtual tables allocate
their ``sqlite3_vtab`` and cursor structs with it.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy


__all__ = [
    "sqlite3_exec", "sqlite3_free", "sqlite3_malloc",
]


load_lib("sqlite3")


@proxy(signatures.get("sqlite3_exec"), jit_options=jit_options)
def sqlite3_exec(db_p, sql_p, cb_p, ctx_p, errmsg_pp):
    """Run each SQL statement of the UTF-8 text at ``sql_p`` on the connection ``db_p``, and return a result code.

    C reference: `sqlite3_exec <https://sqlite.org/c3ref/exec.html>`_.

    ``cb_p`` is the address of a row callback of the type that the module docstring gives, or 0 for none, and SQLite
    passes ``ctx_p`` to it. A statement that fails stops the run, and SQLite skips the statements after it. Unless
    ``errmsg_pp`` is 0, SQLite then writes to it the address of an error message, which the caller frees with
    :func:`sqlite3_free`; after a success it writes 0 there. Returns ``SQLITE_OK``, ``SQLITE_ABORT`` when the callback
    returned a nonzero value, or the error code of the statement that failed.
    """
    return _call_lib_func(
        "sqlite3_exec", (db_p, sql_p, cb_p, ctx_p, errmsg_pp)
    )


@proxy(signatures.get("sqlite3_free"), jit_options=jit_options)
def sqlite3_free(mem_p):
    """Free memory that SQLite allocated, such as the error message of :func:`sqlite3_exec`.

    C reference: `sqlite3_free <https://sqlite.org/c3ref/free.html>`_.

    A pointer of 0 is a no-op. A pointer that SQLite's allocator did not return can corrupt memory or crash the
    process, and so can a read of the memory after it is freed.
    """
    return _call_lib_func("sqlite3_free", (mem_p,))


@proxy(signatures.get("sqlite3_malloc"), jit_options=jit_options)
def sqlite3_malloc(n):
    """Allocate at least ``n`` bytes with SQLite's allocator and return their address, or 0 if the allocation fails.

    C reference: `sqlite3_malloc <https://sqlite.org/c3ref/free.html>`_.

    Returns 0 as well for an ``n`` of 0 or less, and numba converts ``n`` to int32 without a check. The memory is not
    zeroed. It is aligned to at least 8 bytes unless SQLite was built with ``SQLITE_4_BYTE_ALIGNED_MALLOC``, and the
    caller frees it with :func:`sqlite3_free`.
    """
    return _call_lib_func("sqlite3_malloc", (n,))
