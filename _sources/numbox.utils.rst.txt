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
