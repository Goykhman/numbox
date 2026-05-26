"""Extract raw SQLite C API pointers from Python ``sqlite3`` objects.

Bridges CPython's stdlib ``sqlite3`` module to the numba-callable bindings layer by exposing the underlying
``sqlite3 *db`` handle that sits inside a Python ``sqlite3.Connection``.  Mirrors the pattern in
`numbduck.pybridge <https://github.com/Goykhman/numbduck/blob/main/numbduck/pybridge.py>`_.

macOS symbol resolution
~~~~~~~~~~~~~~~~~~~~~~~

On macOS, the system sqlite (``/usr/lib/libsqlite3.dylib``) is part of the
`dyld shared cache <https://developer.apple.com/documentation/kernel/os_dyld_shared_cache_header>`_,
mapped into every process at launch.  Python's ``_sqlite3.so`` uses a *different* sqlite — either statically
linked (python.org framework builds) or dynamically linked to Homebrew's copy
(``/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib``).  Both are different libraries, often different versions,
from the shared-cache one.

LLVM's JIT linker resolves extern symbols via ``llvm::sys::DynamicLibrary::SearchForAddressOfSymbol``, which
ultimately calls ``dlsym(RTLD_DEFAULT, name)``.  Per Apple's
`dlsym(3) <https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/dlsym.3.html>`_,
``RTLD_DEFAULT`` searches all Mach-O images in load order and returns the first match.  The shared-cache sqlite
is always loaded first (at process launch), so its symbols win — regardless of ``RTLD_GLOBAL`` on any
later-loaded library.

``ctypes.CDLL(path).symbol`` works correctly because it calls ``dlsym(handle, name)`` with a specific handle,
which searches the library and its direct dependencies (two-level namespace).  But LLVM never uses a specific
handle — it uses the flat ``RTLD_DEFAULT`` path.

The fix is :func:`llvmlite.binding.add_symbol`, which inserts addresses into LLVM's ``ExplicitSymbols`` map —
checked *before* any ``dlsym`` fallback.  This module reads the correct addresses from ``_sqlite3.so`` via
ctypes (which follows the dependency chain) and registers them with ``add_symbol`` so the JIT linker uses
Python's sqlite, not the system's.

On Linux there is only one sqlite on the system, so ``RTLD_DEFAULT`` returns the correct address and no
patching is needed.

What this module does
~~~~~~~~~~~~~~~~~~~~~

On macOS, before any ``numbox.core.bindings._sqlite_*`` module loads:

1. **Pre-populate** ``numbox.core.bindings.utils._loaded_libs["sqlite3"]`` with a ctypes handle to
   ``_sqlite3.so``, so ``proxy_if_available``'s ``hasattr`` check reflects Python's sqlite's symbol availability.

2. **Register** each sqlite3_* symbol address from ``_sqlite3.so`` with ``add_symbol``, using
   ``signatures_sqlite`` keys so new bindings are picked up automatically.

``utils.py`` and ``signatures.py`` are loaded via :mod:`importlib.util` to avoid triggering ``__init__.py``'s
star-imports (which would eager-compile ``@proxy`` bindings against the wrong library).

**Order constraint.** ``@proxy`` compiles eagerly at decoration time.  Callers must import this module *before*
any ``_sqlite_*`` import.  The project-root ``conftest.py`` handles this for tests.
"""
import ctypes
import sqlite3  # noqa: F401  -- triggers _sqlite3 + libsqlite3 dependency load
from platform import system


def _patch_numbox_sqlite_for_python_libsqlite3():
    """Coordinate numbox's sqlite3 bindings with Python's libsqlite3 on macOS.  See module docstring."""
    if system() != "Darwin":
        return

    import _sqlite3
    import importlib.util
    import os
    import sys

    from llvmlite import binding as llvm_binding

    import numbox  # safe -- numbox/__init__.py only sets __version__

    # Load utils.py and signatures.py directly via importlib.util — going through
    # ``from numbox.core.bindings import`` would trigger ``__init__.py`` which star-imports the ``_sqlite_*``
    # modules and eager-compiles ``@proxy`` bindings against the wrong library.
    bindings_dir = os.path.join(numbox.__path__[0], "core", "bindings")

    utils_path = os.path.join(bindings_dir, "utils.py")
    spec = importlib.util.spec_from_file_location("numbox.core.bindings.utils", utils_path)
    utils_mod = importlib.util.module_from_spec(spec)
    sys.modules["numbox.core.bindings.utils"] = utils_mod
    spec.loader.exec_module(utils_mod)

    sigs_path = os.path.join(bindings_dir, "signatures.py")
    sigs_spec = importlib.util.spec_from_file_location("numbox.core.bindings.signatures", sigs_path)
    sigs_mod = importlib.util.module_from_spec(sigs_spec)
    sys.modules["numbox.core.bindings.signatures"] = sigs_mod
    sigs_spec.loader.exec_module(sigs_mod)

    # (1) Pre-populate _loaded_libs so proxy_if_available's hasattr checks use Python's library.
    py_sqlite = ctypes.CDLL(_sqlite3.__file__, mode=os.RTLD_GLOBAL)
    utils_mod._loaded_libs["sqlite3"] = py_sqlite

    # (2) Register correct symbol addresses with LLVM.  ctypes.CDLL(handle).symbol uses dlsym(handle, name)
    # which searches the library + its dependencies — returns Python's sqlite addresses on both python.org
    # (static) and Homebrew (dynamic dependency).  add_symbol puts them into LLVM's ExplicitSymbols map,
    # checked before dlsym(RTLD_DEFAULT) which would return the wrong (system shared-cache) addresses.
    for sym in sigs_mod.signatures_sqlite:
        func = getattr(py_sqlite, sym, None)
        if func is None:
            continue
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
