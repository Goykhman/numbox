"""Extract raw SQLite C API pointers from Python ``sqlite3`` objects.

Bridges CPython's stdlib ``sqlite3`` module to the numba-callable
bindings layer by exposing the underlying ``sqlite3 *db`` handle that
sits inside a Python ``sqlite3.Connection``. Mirrors the pattern in
`numbduck.pybridge <https://github.com/Goykhman/numbduck/blob/main/numbduck/pybridge.py>`_,
which does the same for DuckDB.

**macOS library coordination.** On macOS, python.org framework Python
*statically* links libsqlite3 into ``_sqlite3.cpython-3X-darwin.so``
(see `actions/python-versions' macos-python-builder.psm1
<https://github.com/actions/python-versions/blob/main/builders/macos-python-builder.psm1>`_
which adds ``-L$(brew --prefix sqlite3)/lib`` at configure time).
Meanwhile ``ctypes.util.find_library("sqlite3")`` returns the system
``/usr/lib/libsqlite3.dylib``, often an older release with a different
internal ``sqlite3`` struct layout. Calling sqlite3 C API functions
from one library on a ``sqlite3*`` allocated by the other is undefined
behavior -- crashes at runtime.

Importing this module on macOS does two coordinated things *before*
any ``numbox.core.bindings._sqlite_*`` module loads:

1. **Pre-populate** ``numbox.core.bindings.utils._loaded_libs["sqlite3"]``
   with a handle to Python's ``_sqlite3.so``. This makes
   ``proxy_if_available``'s ``hasattr`` check (used to gate compile-flag-
   dependent bindings like ``sqlite3_column_database_name``) reflect
   *Python's* sqlite's symbol availability. Symbols not in Python's
   library get stubbed out cleanly instead of @njit-compiling against
   the system library.

2. **Register** each sqlite3_* symbol address from ``_sqlite3.so`` with
   :func:`llvmlite.binding.add_symbol`. LLVM's JIT linker consults its
   symbol registry *before* falling back to ``dlsym(RTLD_DEFAULT)`` --
   so numbox's eager-compiled ``@proxy`` bindings link to Python's
   addresses regardless of macOS dyld's shared-cache resolution.

To pre-populate ``_loaded_libs`` before ``_sqlite_conn`` runs, this
module loads ``numbox.core.bindings.utils`` directly via
:mod:`importlib.util` -- bypassing the package ``__init__.py`` which
would star-import the ``_sqlite_*`` modules and trigger their eager
``@proxy`` compilation. The pre-loaded utils module is registered in
``sys.modules`` so the subsequent ``from numbox.core.bindings._sqlite_conn
import ...`` reuses it.

**Order constraint.** ``@proxy`` compiles eagerly at decoration time
(see ``numbox.core.proxy.proxy``). Callers must import
:mod:`numbox.utils.pysqlite_bridge` *before* any direct or transitive
import of ``numbox.core.bindings._sqlite_*``. In the test environment,
the project-root ``conftest.py`` handles this.

Linux and Windows already coordinate naturally: Python's sqlite3 and
numbox's ``load_lib("sqlite3")`` resolve to the same shared library on
those platforms, so this module is a no-op there.
"""
import ctypes
import sqlite3  # noqa: F401  -- triggers _sqlite3 + libsqlite3 dependency load
from platform import system


_NUMBOX_SQLITE_SYMBOLS = (
    # Connection
    "sqlite3_open", "sqlite3_open_v2", "sqlite3_close",
    "sqlite3_libversion", "sqlite3_libversion_number",
    "sqlite3_errmsg", "sqlite3_errcode", "sqlite3_extended_errcode",
    "sqlite3_threadsafe", "sqlite3_db_handle", "sqlite3_db_filename",
    "sqlite3_db_readonly",
    "sqlite3_changes", "sqlite3_total_changes",
    "sqlite3_changes64", "sqlite3_total_changes64",
    "sqlite3_last_insert_rowid",
    # Statement
    "sqlite3_prepare_v2", "sqlite3_finalize", "sqlite3_reset",
    "sqlite3_step", "sqlite3_sql", "sqlite3_expanded_sql",
    "sqlite3_stmt_busy",
    # Bind
    "sqlite3_bind_int", "sqlite3_bind_int64", "sqlite3_bind_double",
    "sqlite3_bind_text", "sqlite3_bind_blob", "sqlite3_bind_null",
    "sqlite3_bind_parameter_count", "sqlite3_bind_parameter_index",
    "sqlite3_bind_parameter_name",
    # Column
    "sqlite3_column_int", "sqlite3_column_int64", "sqlite3_column_double",
    "sqlite3_column_text", "sqlite3_column_blob", "sqlite3_column_bytes",
    "sqlite3_column_type", "sqlite3_column_count",
    "sqlite3_column_name", "sqlite3_column_decltype",
    "sqlite3_column_database_name", "sqlite3_column_table_name",
    "sqlite3_column_origin_name",
    # Exec
    "sqlite3_exec", "sqlite3_free",
    # Blob
    "sqlite3_blob_open", "sqlite3_blob_close", "sqlite3_blob_bytes",
    "sqlite3_blob_read", "sqlite3_blob_write", "sqlite3_blob_reopen",
    # Hooks
    "sqlite3_update_hook", "sqlite3_progress_handler",
    "sqlite3_busy_handler", "sqlite3_commit_hook",
    "sqlite3_rollback_hook", "sqlite3_trace_v2",
)


def _patch_numbox_sqlite_for_python_libsqlite3():
    """Coordinate numbox's sqlite3 bindings with Python's libsqlite3 on macOS.

    See module docstring for the full rationale. Two-part fix:
    (1) pre-populate numbox's ``_loaded_libs`` so symbol-availability
    checks use Python's library; (2) register Python's symbol addresses
    with llvmlite so the JIT linker resolves to them.
    """
    if system() != "Darwin":
        return

    import _sqlite3
    import importlib.util
    import os
    import sys
    from os import RTLD_GLOBAL

    from llvmlite import binding as llvm_binding

    import numbox  # safe -- numbox/__init__.py only sets __version__

    # Load numbox.core.bindings.utils directly. Going through normal
    # ``from numbox.core.bindings.utils import ...`` would trigger
    # ``numbox.core.bindings.__init__.py`` which star-imports the
    # ``_sqlite_*`` modules and eager-compiles them with the wrong library.
    utils_path = os.path.join(
        numbox.__path__[0], "core", "bindings", "utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "numbox.core.bindings.utils", utils_path
    )
    utils_mod = importlib.util.module_from_spec(spec)
    sys.modules["numbox.core.bindings.utils"] = utils_mod
    spec.loader.exec_module(utils_mod)

    # RTLD_GLOBAL: promote Python's sqlite3_* symbols to the global
    # namespace as a defensive measure (the add_symbol path below is
    # the primary mechanism for symbol redirection).
    py_sqlite = ctypes.CDLL(_sqlite3.__file__, mode=RTLD_GLOBAL)

    # (1) Make numbox's load_lib("sqlite3") return Python's lib.
    # ``proxy_if_available`` decorators in _sqlite_conn / _sqlite_column
    # will then check hasattr against this handle; symbols absent from
    # Python's lib (e.g., column-metadata accessors when Python's sqlite
    # lacks SQLITE_ENABLE_COLUMN_METADATA) are stubbed out cleanly.
    utils_mod._loaded_libs["sqlite3"] = py_sqlite

    # (2) Register sqlite3_* addresses for numba's JIT linker. macOS
    # dyld's shared-cache resolution returns ``/usr/lib/libsqlite3.dylib``'s
    # symbols via ``dlsym(RTLD_DEFAULT)`` regardless of RTLD_GLOBAL load
    # order; add_symbol overrides that for LLVM-compiled JIT modules.
    for sym in _NUMBOX_SQLITE_SYMBOLS:
        func = getattr(py_sqlite, sym, None)
        if func is None:
            continue   # symbol absent in this sqlite version / build
        addr = ctypes.cast(func, ctypes.c_void_p).value
        if addr:
            llvm_binding.add_symbol(sym, addr)


_patch_numbox_sqlite_for_python_libsqlite3()


# Triggers numbox.core.bindings.__init__.py star-imports. Each _sqlite_*
# module does `_sqlite3_lib = load_lib("sqlite3")` which now returns
# our pre-populated Python-lib handle, and each @proxy decoration's JIT
# compile resolves sqlite3_* externs against our add_symbol-registered
# addresses.
from numbox.core.bindings._sqlite_conn import sqlite3_errmsg  # noqa: E402


class _PysqliteConnection(ctypes.Structure):
    """Model the first fields of CPython's ``pysqlite_Connection``.

    Mirrors the layout in `Modules/_sqlite/connection.h
    <https://github.com/python/cpython/blob/main/Modules/_sqlite/connection.h>`_.
    ``ctypes.Structure`` computes the ``db`` offset from the declared
    field types, so the layout is correct on both 32-bit (offset 8) and
    64-bit (offset 16) platforms without a hardcoded constant.

    Pattern from `numbsql
    <https://github.com/cpcloud/numbsql/blob/main/numbsql/sqlite.py#L282>`_.

    Assumes a release (non-debug, non-free-threaded) CPython build.
    ``Py_DEBUG`` builds prepend ``_PyObject_HEAD_EXTRA``; free-threaded
    (``Py_GIL_DISABLED``) builds use a larger ``PyObject``.
    """

    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("db", ctypes.c_void_p),
    ]


def extract_connection_ptr(conn):
    """Return the raw ``sqlite3*`` underlying a Python ``sqlite3.Connection``.

    Uses :class:`_PysqliteConnection` to read the ``db`` field at its
    platform-correct offset inside the PyObject struct, then validates
    the pointer by calling
    :func:`~numbox.core.bindings._sqlite_conn.sqlite3_errmsg` (a
    healthy connection returns ``"not an error"``).

    Parameters
    ----------
    conn : sqlite3.Connection

    Returns
    -------
    int
        ``sqlite3*`` as a Python int (``intp``-compatible).

    Raises
    ------
    TypeError
        If *conn* is not a ``sqlite3.Connection``.
    RuntimeError
        If the extracted pointer fails the validation call.
    """
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError(
            f"expected sqlite3.Connection, got {type(conn).__name__}"
        )
    db_ptr = _PysqliteConnection.from_address(id(conn)).db
    if db_ptr is None:
        raise RuntimeError("extracted null sqlite3* from sqlite3.Connection")
    errmsg = ctypes.c_char_p(sqlite3_errmsg(db_ptr)).value
    if errmsg != b"not an error":
        raise RuntimeError(
            f"extracted connection pointer failed validation: {errmsg!r}"
        )
    return db_ptr
