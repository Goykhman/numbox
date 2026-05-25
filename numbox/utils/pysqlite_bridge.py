"""Extract raw SQLite C API pointers from Python ``sqlite3`` objects.

Bridges CPython's stdlib ``sqlite3`` module to the numba-callable
bindings layer by exposing the underlying ``sqlite3 *db`` handle that
sits inside a Python ``sqlite3.Connection``. Mirrors the pattern in
`numbduck.pybridge <https://github.com/Goykhman/numbduck/blob/main/numbduck/pybridge.py>`_,
which does the same for DuckDB.

**macOS library coordination.** On macOS, Python's ``_sqlite3.so``
and ``ctypes.util.find_library("sqlite3")`` resolve to *different*
sqlite3 libraries. This happens on both common Python distributions:

- **python.org framework builds** statically link libsqlite3 into
  ``_sqlite3.so`` (no external dylib at all).
- **Homebrew Python** dynamically links ``_sqlite3.so`` against
  Homebrew's ``/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib``, but
  CPython loads extension modules with ``RTLD_LOCAL``, so those
  symbols stay invisible to ``dlsym(RTLD_DEFAULT)``.

In both cases, ``find_library("sqlite3")`` resolves to the *system*
sqlite from macOS's dyld shared cache -- a different library (often a
different version) from the one Python's ``_sqlite3`` actually uses.
Calling sqlite3 C API functions from one library on a ``sqlite3*``
allocated by the other is undefined behavior -- crashes at runtime.

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

    # Load numbox.core.bindings.utils and .signatures directly via
    # importlib.util. Going through ``from numbox.core.bindings import``
    # would trigger ``__init__.py`` which star-imports the ``_sqlite_*``
    # modules and eager-compiles them with the wrong library.
    bindings_dir = os.path.join(numbox.__path__[0], "core", "bindings")

    utils_path = os.path.join(bindings_dir, "utils.py")
    spec = importlib.util.spec_from_file_location(
        "numbox.core.bindings.utils", utils_path
    )
    utils_mod = importlib.util.module_from_spec(spec)
    sys.modules["numbox.core.bindings.utils"] = utils_mod
    spec.loader.exec_module(utils_mod)

    sigs_path = os.path.join(bindings_dir, "signatures.py")
    sigs_spec = importlib.util.spec_from_file_location(
        "numbox.core.bindings.signatures", sigs_path
    )
    sigs_mod = importlib.util.module_from_spec(sigs_spec)
    sys.modules["numbox.core.bindings.signatures"] = sigs_mod
    sigs_spec.loader.exec_module(sigs_mod)

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
    # dyld's shared-cache resolution returns the system sqlite's symbols
    # via ``dlsym(RTLD_DEFAULT)`` regardless of RTLD_GLOBAL load order;
    # add_symbol overrides that for LLVM-compiled JIT modules.
    # Symbol list derived from signatures_sqlite so new bindings are
    # picked up automatically without maintaining a separate list.
    for sym in sigs_mod.signatures_sqlite:
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


def _pyobject_head_fields():
    """Return ctypes field list for CPython's ``PyObject_HEAD``.

    Detects debug and free-threaded builds at runtime so the struct
    offset to ``db`` is correct in all configurations.
    """
    import sys
    import sysconfig

    fields = []

    if hasattr(sys, "gettotalrefcount"):
        fields += [
            ("_ob_next", ctypes.c_void_p),
            ("_ob_prev", ctypes.c_void_p),
        ]

    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        fields += [
            ("ob_tid", ctypes.c_size_t),
            ("_padding", ctypes.c_uint16),
            ("ob_mutex", ctypes.c_uint8),
            ("ob_gc_bits", ctypes.c_uint8),
            ("ob_ref_local", ctypes.c_uint32),
            ("ob_ref_shared", ctypes.c_ssize_t),
            ("ob_type", ctypes.c_void_p),
        ]
    else:
        fields += [
            ("ob_refcnt", ctypes.c_ssize_t),
            ("ob_type", ctypes.c_void_p),
        ]

    return fields


class _PysqliteConnection(ctypes.Structure):
    """Model the first fields of CPython's ``pysqlite_Connection``.

    Mirrors the layout in `Modules/_sqlite/connection.h
    <https://github.com/python/cpython/blob/main/Modules/_sqlite/connection.h>`_.
    ``ctypes.Structure`` computes the ``db`` offset from the declared
    field types, so the layout is correct across 32/64-bit, debug, and
    free-threaded builds without hardcoded constants.

    Pattern from `numbsql
    <https://github.com/cpcloud/numbsql/blob/main/numbsql/sqlite.py#L282>`_.
    """

    _fields_ = _pyobject_head_fields() + [("db", ctypes.c_void_p)]


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
