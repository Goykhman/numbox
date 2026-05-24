"""SQLite statement-lifecycle bindings: prepare_v2 / finalize / reset / step /
sql / expanded_sql / stmt_busy.

Note: ``sqlite3_expanded_sql`` returns a ``char *`` the caller MUST free via
``sqlite3_free`` (bound in _sqlite_exec.py). Document this with each call site
rather than building a wrapper that auto-frees — the wrapper would hide
ownership in a way the rest of the bindings don't.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.proxy.proxy import proxy

__all__ = [
    "sqlite3_prepare_v2", "sqlite3_finalize", "sqlite3_reset", "sqlite3_step",
    "sqlite3_sql", "sqlite3_expanded_sql", "sqlite3_stmt_busy",
]


load_lib("sqlite3")


@proxy(signatures.get("sqlite3_prepare_v2"), jit_options={"cache": True})
def sqlite3_prepare_v2(db_p, sql_p, n_byte, stmt_pp, tail_pp):
    return _call_lib_func("sqlite3_prepare_v2", (db_p, sql_p, n_byte, stmt_pp, tail_pp))


@proxy(signatures.get("sqlite3_finalize"), jit_options={"cache": True})
def sqlite3_finalize(stmt_p):
    return _call_lib_func("sqlite3_finalize", (stmt_p,))


@proxy(signatures.get("sqlite3_reset"), jit_options={"cache": True})
def sqlite3_reset(stmt_p):
    return _call_lib_func("sqlite3_reset", (stmt_p,))


@proxy(signatures.get("sqlite3_step"), jit_options={"cache": True})
def sqlite3_step(stmt_p):
    return _call_lib_func("sqlite3_step", (stmt_p,))


@proxy(signatures.get("sqlite3_sql"), jit_options={"cache": True})
def sqlite3_sql(stmt_p):
    return _call_lib_func("sqlite3_sql", (stmt_p,))


@proxy(signatures.get("sqlite3_expanded_sql"), jit_options={"cache": True})
def sqlite3_expanded_sql(stmt_p):
    return _call_lib_func("sqlite3_expanded_sql", (stmt_p,))


@proxy(signatures.get("sqlite3_stmt_busy"), jit_options={"cache": True})
def sqlite3_stmt_busy(stmt_p):
    return _call_lib_func("sqlite3_stmt_busy", (stmt_p,))
