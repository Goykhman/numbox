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

numbox.utils.lowlevel
---------------------

.. automodule:: numbox.utils.lowlevel
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.timer
------------------

.. automodule:: numbox.utils.timer
   :members:
   :show-inheritance:
   :undoc-members:
