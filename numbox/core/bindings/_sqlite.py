from numbox.utils.highlevel import cres
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib


load_lib("sqlite3")


@cres(signatures.get("sqlite3_libversion"), cache=True)
def sqlite3_libversion_number():
    return _call_lib_func("sqlite3_libversion")


@cres(signatures.get("sqlite3_open"), cache=True)
def sqlite3_open(db_name_p, db_pp):
    return _call_lib_func("sqlite3_open", (db_name_p, db_pp))


@cres(signatures.get("sqlite3_close"), cache=True)
def sqlite3_close(db_p):
    return _call_lib_func("sqlite3_close", (db_p,))
