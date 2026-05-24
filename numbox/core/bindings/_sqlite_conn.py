"""SQLite connection + metadata bindings.

Resolves the shared library via ``get_loaded_lib("sqlite3")`` from
``utils._loaded_libs``. Other ``_sqlite_*.py`` modules (currently
``_sqlite_column``) call the same getter rather than importing
``_sqlite3_lib`` from here, so there's no cross-module dependency on
which file happens to load the library first.

Two functions are decorated with ``proxy_if_available``:
``sqlite3_changes64`` and ``sqlite3_total_changes64``, both added in
SQLite 3.37 (Nov 2021). Older library versions lack these symbols —
notably the SQLite 3.34 shipped with CPython 3.10 on Windows, and
distro-shipped system sqlite3 on Linux / macOS that predates 3.37.
The wrappers stub to ``NotImplementedError`` so callers can
``hasattr(...,"as_func")`` to decide whether to use them or fall back
to the int32 variants.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import get_loaded_lib
from numbox.core.proxy.proxy import proxy, proxy_if_available

__all__ = [
    "sqlite3_open", "sqlite3_open_v2", "sqlite3_close",
    "sqlite3_libversion", "sqlite3_libversion_number",
    "sqlite3_errmsg", "sqlite3_errcode", "sqlite3_extended_errcode",
    "sqlite3_threadsafe",
    "sqlite3_db_handle", "sqlite3_db_filename", "sqlite3_db_readonly",
    "sqlite3_changes", "sqlite3_last_insert_rowid", "sqlite3_total_changes",
    "sqlite3_changes64", "sqlite3_total_changes64",
]


_sqlite3_lib = get_loaded_lib("sqlite3")


@proxy(signatures.get("sqlite3_open"), jit_options={"cache": True})
def sqlite3_open(filename_p, db_pp):
    return _call_lib_func("sqlite3_open", (filename_p, db_pp))


@proxy(signatures.get("sqlite3_open_v2"), jit_options={"cache": True})
def sqlite3_open_v2(filename_p, db_pp, flags, vfs_p):
    return _call_lib_func("sqlite3_open_v2", (filename_p, db_pp, flags, vfs_p))


@proxy(signatures.get("sqlite3_close"), jit_options={"cache": True})
def sqlite3_close(db_p):
    return _call_lib_func("sqlite3_close", (db_p,))


@proxy(signatures.get("sqlite3_libversion"), jit_options={"cache": True})
def sqlite3_libversion():
    return _call_lib_func("sqlite3_libversion")


@proxy(signatures.get("sqlite3_libversion_number"), jit_options={"cache": True})
def sqlite3_libversion_number():
    return _call_lib_func("sqlite3_libversion_number")


@proxy(signatures.get("sqlite3_errmsg"), jit_options={"cache": True})
def sqlite3_errmsg(db_p):
    return _call_lib_func("sqlite3_errmsg", (db_p,))


@proxy(signatures.get("sqlite3_errcode"), jit_options={"cache": True})
def sqlite3_errcode(db_p):
    return _call_lib_func("sqlite3_errcode", (db_p,))


@proxy(signatures.get("sqlite3_extended_errcode"), jit_options={"cache": True})
def sqlite3_extended_errcode(db_p):
    return _call_lib_func("sqlite3_extended_errcode", (db_p,))


@proxy(signatures.get("sqlite3_threadsafe"), jit_options={"cache": True})
def sqlite3_threadsafe():
    return _call_lib_func("sqlite3_threadsafe")


@proxy(signatures.get("sqlite3_db_handle"), jit_options={"cache": True})
def sqlite3_db_handle(stmt_p):
    return _call_lib_func("sqlite3_db_handle", (stmt_p,))


@proxy(signatures.get("sqlite3_db_filename"), jit_options={"cache": True})
def sqlite3_db_filename(db_p, name_p):
    return _call_lib_func("sqlite3_db_filename", (db_p, name_p))


@proxy(signatures.get("sqlite3_db_readonly"), jit_options={"cache": True})
def sqlite3_db_readonly(db_p, name_p):
    return _call_lib_func("sqlite3_db_readonly", (db_p, name_p))


@proxy(signatures.get("sqlite3_changes"), jit_options={"cache": True})
def sqlite3_changes(db_p):
    return _call_lib_func("sqlite3_changes", (db_p,))


@proxy(signatures.get("sqlite3_last_insert_rowid"), jit_options={"cache": True})
def sqlite3_last_insert_rowid(db_p):
    return _call_lib_func("sqlite3_last_insert_rowid", (db_p,))


@proxy(signatures.get("sqlite3_total_changes"), jit_options={"cache": True})
def sqlite3_total_changes(db_p):
    return _call_lib_func("sqlite3_total_changes", (db_p,))


# SQLite 3.37+; stubbed via proxy_if_available on older library versions.
@proxy_if_available(_sqlite3_lib, signatures.get("sqlite3_changes64"), jit_options={"cache": True})
def sqlite3_changes64(db_p):
    return _call_lib_func("sqlite3_changes64", (db_p,))


@proxy_if_available(_sqlite3_lib, signatures.get("sqlite3_total_changes64"), jit_options={"cache": True})
def sqlite3_total_changes64(db_p):
    return _call_lib_func("sqlite3_total_changes64", (db_p,))
