from numbox.core.proxy.proxy import proxy
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib


load_lib("sqlite3")


@proxy(signatures.get("sqlite3_libversion"), jit_options={"cache": True})
def sqlite3_libversion_number():
    return _call_lib_func("sqlite3_libversion")


@proxy(signatures.get("sqlite3_open"), jit_options={"cache": True})
def sqlite3_open(db_name_p, db_pp):
    return _call_lib_func("sqlite3_open", (db_name_p, db_pp))


@proxy(signatures.get("sqlite3_close"), jit_options={"cache": True})
def sqlite3_close(db_p):
    return _call_lib_func("sqlite3_close", (db_p,))
