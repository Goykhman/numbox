"""SQLite exec + free bindings.

sqlite3_exec is the one-shot SQL escape hatch — it parses, prepares, steps,
and finalizes a (potentially multi-statement) SQL string. The third arg is a
function pointer to a per-row callback; pass 0 for no callback.

sqlite3_free releases memory SQLite allocated and returned to the caller —
notably the errmsg buffer from sqlite3_exec's out-param, and the result of
sqlite3_expanded_sql.

Callback shape (informational):
    int (*sqlite3_exec_callback)(void *ctx, int ncol,
                                 char **col_values, char **col_names)
Return 0 to continue, nonzero to abort with SQLITE_ABORT.

Produce the callback address from Python via:
    @cfunc(int32(voidptr, int32, intp, intp))
    def my_row_cb(ctx, n, values_pp, names_pp): ...
    sqlite3_exec(db, sql, my_row_cb.address, ctx_p, errmsg_pp)
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.proxy.proxy import proxy

__all__ = [
    "sqlite3_exec", "sqlite3_free",
]


load_lib("sqlite3")


@proxy(signatures.get("sqlite3_exec"), jit_options={"cache": True})
def sqlite3_exec(db_p, sql_p, cb_p, ctx_p, errmsg_pp):
    return _call_lib_func(
        "sqlite3_exec", (db_p, sql_p, cb_p, ctx_p, errmsg_pp)
    )


@proxy(signatures.get("sqlite3_free"), jit_options={"cache": True})
def sqlite3_free(mem_p):
    return _call_lib_func("sqlite3_free", (mem_p,))
