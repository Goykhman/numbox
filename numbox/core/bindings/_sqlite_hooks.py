"""SQLite callback hooks: update_hook / progress_handler / busy_handler /
commit_hook / rollback_hook / trace_v2.

Each takes a function pointer (intp) the caller produces via @cfunc(...).address.
The cfunc instance MUST outlive the hook registration — keep it at module scope
in the caller.

Callback shapes (informational; signatures are caller's responsibility):
- update_hook:       void(void*, int op, const char* db, const char* tbl, sqlite3_int64 rowid)
- progress_handler:  int(void*) -- nonzero aborts
- busy_handler:      int(void*, int) -- 0 to abort, nonzero to retry
- commit_hook:       int(void*) -- nonzero vetoes commit
- rollback_hook:     void(void*)
- trace_v2:          int(unsigned, void*, void*, void*)
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.proxy.proxy import proxy

__all__ = [
    "sqlite3_update_hook", "sqlite3_progress_handler", "sqlite3_busy_handler",
    "sqlite3_commit_hook", "sqlite3_rollback_hook", "sqlite3_trace_v2",
]


load_lib("sqlite3")


@proxy(signatures.get("sqlite3_update_hook"), jit_options={"cache": True})
def sqlite3_update_hook(db_p, cb_p, ctx_p):
    return _call_lib_func("sqlite3_update_hook", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_progress_handler"), jit_options={"cache": True})
def sqlite3_progress_handler(db_p, n_ops, cb_p, ctx_p):
    return _call_lib_func("sqlite3_progress_handler", (db_p, n_ops, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_busy_handler"), jit_options={"cache": True})
def sqlite3_busy_handler(db_p, cb_p, ctx_p):
    return _call_lib_func("sqlite3_busy_handler", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_commit_hook"), jit_options={"cache": True})
def sqlite3_commit_hook(db_p, cb_p, ctx_p):
    return _call_lib_func("sqlite3_commit_hook", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_rollback_hook"), jit_options={"cache": True})
def sqlite3_rollback_hook(db_p, cb_p, ctx_p):
    return _call_lib_func("sqlite3_rollback_hook", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_trace_v2"), jit_options={"cache": True})
def sqlite3_trace_v2(db_p, mask, cb_p, ctx_p):
    return _call_lib_func("sqlite3_trace_v2", (db_p, mask, cb_p, ctx_p))
