"""Read the ``sqlite3 *`` handle of a Python ``sqlite3.Connection``, for numbox's SQLite bindings.

numbox's SQLite bindings take a connection as the ``sqlite3 *`` handle of SQLite's C API, and a connection that
Python's ``sqlite3`` module opens keeps that handle in its C structure. :func:`extract_connection_ptr` reads it with
ctypes, as the `pybridge <https://github.com/Goykhman/numbduck/blob/main/numbduck/pybridge.py>`_ module of numbduck
reads the handle of a DuckDB connection. The handle is safe to use only if numbox's bindings and Python's ``sqlite3``
module use the same SQLite, which :func:`libraries_coordinated` checks by comparing the versions they report.

macOS caveat
~~~~~~~~~~~~

On macOS, the system's SQLite is in the `dyld shared cache <https://keith.github.io/xcode-man-pages/dyld.1.html>`_,
whose structure Apple's `DyldSharedCache.h
<https://github.com/apple-oss-distributions/dyld/blob/main/common/DyldSharedCache.h>`_ defines, and the cache is mapped
into every process at launch. LLVM's JIT linker resolves the ``sqlite3_*`` symbols with ``dlsym(RTLD_DEFAULT)``, which
returns the first match in load order: the system's copy. Python's ``_sqlite3`` module can use another SQLite. A
python.org framework build links SQLite statically, and a Homebrew or conda-forge build links its own shared library.
A handle from one copy passed to the other can give wrong results or crash the process.

The workaround is to make the process load Python's SQLite library first::

    DYLD_INSERT_LIBRARIES=/path/to/your/libsqlite3.dylib python my_script.py

The library is usually at one of these paths:

- Homebrew on Apple Silicon: ``/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib``
- Homebrew on Intel: ``/usr/local/opt/sqlite/lib/libsqlite3.dylib``
- conda-forge: ``$CONDA_PREFIX/lib/libsqlite3.dylib``

A python.org framework build has no shared library of its SQLite to insert. Inserting another library, such as
Homebrew's, satisfies the check only if that library reports the same version as the SQLite linked into Python.

On Linux, numbox's bindings load the library that ``ctypes.util.find_library("sqlite3")`` finds, usually the system's.
A Python whose ``sqlite3`` module links that library, as a Linux distribution's Python does, needs no workaround. A
Python that carries its own SQLite, as the builds that uv installs do, can report another version, and
:func:`extract_connection_ptr` then raises ``RuntimeError``.
"""
import ctypes
import sqlite3

from numbox.core.bindings.sqlite.conn import sqlite3_errmsg, sqlite3_libversion


def _pyobject_head_fields():
    """Return the ctypes fields of CPython's ``PyObject_HEAD`` for the build of the running interpreter.

    A release build gets ``ob_refcnt`` and ``ob_type``, and a free-threaded build gets its larger header. A debug build,
    which has ``sys.gettotalrefcount``, also gets the two pointers ``_ob_next`` and ``_ob_prev`` in front. Those
    pointers exist only in a build configured ``--with-trace-refs``, and only on CPython 3.10 to 3.12, so on any other
    debug build the offset of ``db`` is wrong.
    """
    import sys
    import sysconfig

    fields = []

    # sys.gettotalrefcount exists in every debug build, but _ob_next and _ob_prev exist only under Py_TRACE_REFS,
    # which a debug build does not imply, and CPython 3.13 removed them. On a plain debug build this layout puts db
    # at offset 32 instead of 16. test_pyobject_head_fields_handles_debug_build makes the same assumption.
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
    """The leading fields of CPython's ``pysqlite_Connection``: the object header and the ``db`` handle.

    The layout follows `Modules/_sqlite/connection.h
    <https://github.com/python/cpython/blob/main/Modules/_sqlite/connection.h>`_. ``ctypes.Structure`` computes the
    offset of ``db`` from the declared field types, so no offset is written in the code, and the offset is right on
    32-bit, 64-bit and free-threaded builds. :func:`_pyobject_head_fields` names the debug builds on which it is
    wrong. The approach comes from `numbsql <https://github.com/cpcloud/numbsql/blob/main/numbsql/sqlite.py#L282>`_.
    """

    _fields_ = _pyobject_head_fields() + [("db", ctypes.c_void_p)]


def libraries_coordinated():
    """Return True if numbox's bindings and Python's ``sqlite3`` module report the same SQLite version.

    The function compares the version that :func:`~numbox.core.bindings.sqlite.conn.sqlite3_libversion` returns
    through numbox's bindings with ``sqlite3.sqlite_version``, the version of the library that Python's module uses.
    Different versions mean different copies of SQLite, as on macOS without the workaround in the module docstring.
    :func:`extract_connection_ptr` then raises instead of passing a handle from one copy to the other. Two copies of
    the same version count as coordinated.
    """
    numbox_version = ctypes.c_char_p(sqlite3_libversion()).value.decode()
    return numbox_version == sqlite3.sqlite_version


def extract_connection_ptr(conn):
    """Return the ``sqlite3 *`` handle of the Python ``sqlite3.Connection`` ``conn``, as an ``int``.

    The handle is borrowed: ``conn`` owns it, and the ``int`` holds no reference to ``conn``. The caller must keep
    ``conn`` open and referenced for as long as any code, jitted or not, uses the handle. After ``conn.close()``, or
    once ``conn`` is garbage-collected, the handle points to freed memory, and a use of it is a use-after-free. The
    rule is the one that ``SQLITE_STATIC`` sets for ``bind_text`` and ``bind_blob``: borrowed memory must outlive
    every use.

    ``TypeError`` is raised if ``conn`` is not a ``sqlite3.Connection``. Before it reads the handle, the function
    checks with :func:`libraries_coordinated` that numbox's bindings and Python's ``sqlite3`` module use the same
    SQLite, and it raises ``RuntimeError`` if they do not. It reads the ``db`` field of the connection's C structure
    through :class:`_PysqliteConnection` and raises ``RuntimeError`` for a NULL handle, which a closed connection has.
    It then checks the handle by calling :func:`~numbox.core.bindings.sqlite.conn.sqlite3_errmsg`, which must return
    ``not an error``. That check also raises ``RuntimeError`` for a live connection whose most recent statement
    failed, until a statement on the connection succeeds.
    """
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError(
            f"expected sqlite3.Connection, got {type(conn).__name__}"
        )
    if not libraries_coordinated():
        numbox_version = ctypes.c_char_p(sqlite3_libversion()).value.decode()
        raise RuntimeError(
            f"numbox's @njit bindings resolve libsqlite3 {numbox_version!r}, but Python's "
            f"sqlite3 module uses {sqlite3.sqlite_version!r}.  Passing this connection "
            f"pointer to numbox's bindings would be unsafe (possible segfault).  On macOS, "
            f"set DYLD_INSERT_LIBRARIES to your Python's libsqlite3.dylib — see the "
            f"numbox.utils.pysqlite_bridge module docstring."
        )
    db_ptr = _PysqliteConnection.from_address(id(conn)).db
    if db_ptr is None:
        raise RuntimeError("extracted null sqlite3* from sqlite3.Connection")
    # sqlite3_errmsg returns the message of the connection's most recent error, so this check also rejects a healthy
    # connection whose last statement failed.
    errmsg = ctypes.c_char_p(sqlite3_errmsg(db_ptr)).value
    if errmsg != b"not an error":
        raise RuntimeError(
            f"extracted connection pointer failed validation: {errmsg!r}"
        )
    return db_ptr
