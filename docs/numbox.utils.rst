numbox.utils
============

numbox.utils.highlevel
----------------------

Dynamically defining StructRef
''''''''''''''''''''''''''''''

Defining numba `StructRef` requires writing a lot of boilerplate code.
A utility for concise definition of `StructRef` types that supports caching
is provided in :func:`numbox.utils.highlevel.make_structref`. To use it,
define a *separate module* such as type_classes.py such as::

    from numba.experimental.structref import register
    from numba.core.types import StructRef


    @register
    class DataStructTypeClass(StructRef):
        pass

Then in a *different module* main.py define::

    from numba.core.types import float32, unicode_type
    from numpy import isclose
    from numbox.utils.highlevel import make_structref
    from type_classes import DataStructTypeClass


    def derive_output(struct_):
        if struct_.control == "double":
            return struct_.value * 2
        else:
            return struct_.value


    data_struct = make_structref(
        "DataStruct",
        {"value": float32, "control": unicode_type},
        DataStructTypeClass,
        struct_methods={
            "derive_output": derive_output
        }
    )


    if __name__ == "__main__":
        data_1 = data_struct(3.14, "double")
        data_2 = data_struct(2.17, "something else")
        assert isclose(data_1.derive_output(), 6.28)
        assert isclose(data_2.derive_output(), 2.17)


.. automodule:: numbox.utils.highlevel
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.preprocessing
--------------------------

Cache-anchor mechanism
''''''''''''''''''''''

``make_structref`` writes the generated ``code_txt`` to a
content-addressed file under numba's cache directory and uses that
file -- not ``highlevel.py`` -- as the ``compile()`` anchor. The
content-addressing is what keeps numba's per-overload cache correct
when two versions of the generated code differ only in ``co_consts``.

Numba's per-overload cache key
(``numba.core.caching.Cache._index_key``) is::

    (sig, codegen.magic_tuple(), hash(co_code), hash(closure_cells))

It hashes ``co_code`` only, not ``co_consts``. Python's
`LOAD_CONST <https://docs.python.org/3/library/dis.html#opcode-LOAD_CONST>`_
opcode encodes an *index* into ``co_consts`` rather than the value
itself, so two methods differing only in a numeric literal
(``return self.x + 1`` vs ``return self.x + 1000``) produce identical
``co_code``. With a shared anchor file both would resolve to the same
``cache_subpath`` (numba's
``_CacheLocator.get_suitable_cache_subpath`` derives the cache subdir
from a hash of ``co_filename``) and the second to compile would
silently load the first's binary. Per-content anchors segregate the
two by ``cache_subpath`` so the collision never arises.

Structural body changes -- different operators, additional
statements, renamed variables -- produce different ``co_code`` and
therefore different ``_index_key`` values regardless of the anchoring
scheme; those invalidate cleanly even under a shared anchor. The
narrow failure mode protected by content-addressing is constant-only
edits (numeric or string literals, default arg values) where
``co_code`` is identical across versions.

Python 3.14's
`LOAD_SMALL_INT <https://docs.python.org/3.14/whatsnew/3.14.html>`_
opcode inlines small integers directly into ``co_code``, narrowing
the failure mode on that version to constants outside the inline
range. Earlier supported versions (3.10--3.13) collide on any
constant edit.

See also ``numba.core.caching.Cache._index_key`` and
``numba.core.caching._SourceFileBackedLocatorMixin.get_source_stamp``
in numba's source for the cache key construction and source-stamp
validity check.


.. automodule:: numbox.utils.preprocessing
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.digest
-------------------

.. automodule:: numbox.utils.digest
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.fingerprint
------------------------

.. automodule:: numbox.utils.fingerprint
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.lowlevel
---------------------

.. automodule:: numbox.utils.lowlevel
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.cstrings
---------------------

Allocating C strings for the bindings layer
'''''''''''''''''''''''''''''''''''''''''''

The bindings family (``numbox.core.bindings._c``, ``_sqlite_*``, etc.)
takes ``intp`` pointers for every text argument. Producing a valid
NUL-terminated UTF-8 C string from a Python ``str`` is non-trivial:
:func:`~numbox.utils.lowlevel.get_unicode_data_p` returns a pointer to
the Python string's internal data payload, which CPython stores as
UCS-1/2/4 depending on contents -- only safe for ASCII inputs.

:func:`~numbox.utils.cstrings.c_string` is a Python-side context
manager that allocates a real C buffer with the UTF-8 encoding of the
input and yields the pointer with safe lifetime tied to the ``with``
block::

    from numbox.utils.cstrings import c_string
    from numbox.core.bindings import sqlite3_exec

    with c_string("CREATE TABLE t(x INTEGER)") as sql_p:
        sqlite3_exec(db_p, sql_p, 0, 0, 0)
    # buffer freed here automatically

For concurrent multi-string calls, use a single ``with`` with multiple
context managers, or :class:`contextlib.ExitStack` when the count is
dynamic::

    with c_string("main") as schema_p, c_string("t") as table_p, c_string("b") as col_p:
        sqlite3_blob_open(db_p, schema_p, table_p, col_p, rowid, flags,
                          addressof(blob_p))

**Python-only.** ``c_string`` cannot be used inside ``@njit`` -- numba
does not support arbitrary context managers (raises
``UnsupportedBytecodeError``), and ``ctypes`` objects can't be
manipulated under JIT anyway. For ``@njit`` callers that need a C
string, pre-allocate a numpy ``uint8`` buffer containing the UTF-8
bytes plus a trailing NUL at Python level, then pass
:func:`~numbox.utils.lowlevel.array_data_p` into the JIT kernel.

.. automodule:: numbox.utils.cstrings
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.pysqlite_bridge
----------------------------

Bridging Python ``sqlite3`` connections to the bindings layer
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

CPython's stdlib ``sqlite3.Connection`` holds the underlying
``sqlite3 *db`` C handle as a private field inside its PyObject
struct. :func:`~numbox.utils.pysqlite_bridge.extract_connection_ptr`
reads that field via ``ctypes`` so callers can pass the handle into
numbox's JIT-callable SQLite bindings::

    import numbox.utils.pysqlite_bridge as pysqlite_bridge
    import sqlite3
    from numbox.core.bindings import sqlite3_changes

    conn = sqlite3.connect("app.db")
    conn.execute("CREATE TABLE t(x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1), (2), (3)")
    db_p = pysqlite_bridge.extract_connection_ptr(conn)
    assert sqlite3_changes(db_p) == 3

Mirrors the pattern in
`numbduck.pybridge <https://github.com/Goykhman/numbduck/blob/main/numbduck/pybridge.py>`_.

**Build configurations are handled at runtime.** The ``sqlite3 *db`` field
offset is computed by ``_pyobject_head_fields()``, which adapts to the running
interpreter: ``Py_DEBUG`` builds (detected via ``sys.gettotalrefcount``)
prepend the ``_ob_next`` / ``_ob_prev`` trace pointers, and free-threaded
builds (``Py_GIL_DISABLED``) use the no-GIL object header — so release, debug,
and free-threaded builds all read the field at the correct offset.

The macOS shared-cache caveat and its ``DYLD_INSERT_LIBRARIES`` workaround are
described in the module docstring below; :func:`extract_connection_ptr` also
validates (via :func:`libraries_coordinated`) that numbox's bindings and
Python's ``sqlite3`` resolve to the same libsqlite3 before returning a pointer,
raising rather than handing back a pointer to a mismatched library.

.. automodule:: numbox.utils.pysqlite_bridge
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.timer
------------------

.. automodule:: numbox.utils.timer
   :members:
   :show-inheritance:
   :undoc-members:
