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

numbox.utils.derive_wap
-----------------------

A first-class ``FunctionType`` value carries three addresses in its data model, and which
one numba calls decides whether an exception can leave the call. The ``jit_addr`` slot holds
the numba calling convention entry point, which unwinds; when it is empty numba calls the C
wrapper instead, which it documents as not supporting exceptions.

numba populates that slot for a ``Dispatcher`` and leaves it empty for everything else, so
the compile result behind :func:`numbox.utils.highlevel.cres` would arrive with the slot
unset. :class:`~numbox.utils.derive_wap.DeriveWAP` captures the entry point from the compile
result and :class:`~numbox.utils.derive_wap.DeriveFunctionType` fills the slot from it, on
both unboxing and constant lowering.

The three slots are directly observable through
:func:`numbox.utils.lowlevel.get_func_tuple`, which is the clearest way to see what the type
adds. ``jit_addr`` is populated and matches the entry point the wrapper captured, while
numba's own ``_get_jit_address`` still yields 0 for the same value, because it resolves an
address only for a ``Dispatcher``. The example needs numba 0.61 or later:

.. code-block:: python

    from numba import float64
    from numba.experimental.function_type import _get_jit_address

    from numbox.utils.derive_wap import DeriveWAP
    from numbox.utils.highlevel import cres
    from numbox.utils.lowlevel import get_func_tuple

    sig = float64(float64, float64)

    @cres(sig, cache=True)
    def add(x, y):
        return x + y

    assert isinstance(add, DeriveWAP)

    c_addr, py_addr, jit_addr = get_func_tuple(add)
    assert jit_addr != 0
    assert jit_addr == add.jit_address
    assert _get_jit_address(add, sig) == 0

Nothing above runs on an earlier numba. ``_get_jit_address`` does not exist there to import,
the tuple has two entries rather than three, and ``cres`` returns a plain
``CompileResultWAP`` rather than a :class:`~numbox.utils.derive_wap.DeriveWAP`, because
without the slot there is nothing for the mechanism to populate.

:func:`numbox.utils.highlevel.cres` is not the only source of one. From numba 0.61 onward a
``@proxy`` binding's ``.as_func`` is a :class:`~numbox.utils.derive_wap.DeriveWAP` too,
minted from the proxied body's compile result, with the same exception semantics and the
same effect on a jitted caller that reaches it as a constant. See :doc:`numbox.core.proxy`.

``test/utils/test_lowlevel.py::test_get_func_tuple`` pins every assertion above except the
first; the wrapper's type is pinned separately by
``test/utils/test_derive_wap.py::test_cres_mints_a_derive_wap``.

.. automodule:: numbox.utils.derive_wap
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

The anchor is written whenever it can be: numba caches from it, and
quotes the source from it in its messages, a typing error's among them,
so with caching off a write that fails is nothing and the path serves
as the code's name, as it did. With caching on numba is then asked
whether it can cache a function of that file, the question the package
puts for its own modules; where the write fails, for whatever reason,
or the answer is
no, an unwritable user cache directory or ``NUMBA_CACHE_DIR`` with no
other location left, the generated code compiles without a cache after
one warning naming the remedy, instead of dying at the write or at
numba's set-up. A path too long for the file system is one such
failure, and the warning says so where the file system does (Windows
reports a component too long as a syntax error, and the warning then
offers the directory, quoting the error); with the names bounded, as below,
the directory is the only part that can make it so, ``NUMBA_CACHE_DIR``
or the user's cache directory, and ``NUMBA_CACHE_DIR`` at a shorter
path moves the anchor out of either. The check writes names shorter
than the longest numba writes, so a directory within about 230 bytes
of the path limit, 4096 on Linux, passes it and overflows at numba's
first save instead, where numba's own error names the length; the
remedy is the same.

The anchor's name carries the struct's or the function's, and numba
names its cache files after the anchor and the qualified name of the
function it caches, which carries the struct's again through the class
whose body defines the jitted getters and method thunks, so a struct
named with about 93 characters overflowed the 255 bytes a file system
allows a name, in numba's own files past the anchor's check. Those
names are bounded now (``bounded_stem``): a name of 40 bytes or fewer
is used as it is, so nearly every struct keeps the file names it had,
and a longer one becomes the whole characters of it that fit in 31
bytes and a digest of the whole. The measure is the name's UTF-8,
which is the file system's: a character of another script takes up to
four bytes there. The generated class is defined under the bounded name and
takes the struct's full name back once its body is compiled, so
``__name__``, ``__qualname__`` and ``repr`` show the name the caller
gave, whatever its length, and the struct caches. A field's jitted
getter is defined under the bounded name the same way and the
property takes the field's, so a field's name of any length caches
too; a method's name is bounded in its thunk and its overload.
``compile_kernel``, the work builder's derives and the sqlite
aggregate, window and table-valued function registrations anchor their
generated code the same way and fall back the same way, the derive
without a warning, and the kernel and the derive write no anchor when
not caching, as before; ``make_graph``'s kernel, anchored to the builder's
own file, asks for that file under the options it was given. See :doc:`numbox.core.configurations` for the
package-wide rule the anchors follow.

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

numbox.utils.meminfo
--------------------

Handing a reference through a pointer
'''''''''''''''''''''''''''''''''''''

:func:`~numbox.utils.meminfo.export_meminfo` returns a structref's (or an array's) MemInfo
pointer as an ``intp`` and takes a reference for it,
:func:`~numbox.utils.meminfo.borrow_structref` rebuilds a live structref from that pointer,
and :func:`~numbox.utils.meminfo.release_meminfo` gives the reference back. The value stays
alive for as long as the reference is held.

On numba 0.65 and earlier, numba's ``removerefctpass`` deletes every ``NRT_incref`` and
``NRT_decref`` call in a function it judges not to need them: roughly, one whose arguments
are plain values or arrays, that returns a plain value, and that allocates nothing. The
reference ``export_meminfo`` takes is one of those calls, so numbox marks the function that
makes it with numba's ``numba_args_may_always_need_nrt`` metadata, which keeps the pass off
it. numba 0.66 removed the pass and every mention of that metadata, so from 0.66 on nothing
reads it.

**numba's cache does not see that mark arrive.** numbox releases without it lose the
reference when ``export_meminfo`` is given an array on numba 0.65 and earlier, so the
pointer it returns does not hold the array: the MemInfo behind it can be freed while the
pointer is still in use. A ``cache=True`` function that makes such a call and that numba
compiled against one of those releases keeps the lost reference after numbox is upgraded,
because numba's cache key covers the function's own source file, bytecode and closure,
never the code of the functions it calls
(`numba's caching limitations <https://numba.readthedocs.io/en/stable/user/jit.html#cache>`__).
Delete that function's cache files (``*.nbi`` and ``*.nbc``) so numba compiles it again.

.. automodule:: numbox.utils.meminfo
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.cstrings
---------------------

Allocating C strings for the bindings layer
'''''''''''''''''''''''''''''''''''''''''''

The bindings family (``numbox.core.bindings.libc``, ``numbox.core.bindings.sqlite.*``, etc.)
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
    from numbox.core.bindings.sqlite.exec import sqlite3_exec

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
    from numbox.core.bindings.sqlite.conn import sqlite3_changes

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
