"""``VoidType``, a structref type with no fields, under which code can hold a structref of any type.

:func:`numbox.utils.lowlevel.cast` casts a structref to ``VoidType`` and back without changing the memory it refers
to, and :func:`numbox.utils.lowlevel.uniformize_tuple_of_structs` casts to ``VoidType`` by default. A ``VoidType``
value that reaches Python is a :class:`Void`.
"""
from numba.core.types import StructRef
from numba.experimental.structref import define_boxing, register, StructRefProxy


@register
class VoidTypeClass(StructRef):
    """The numba type class of ``VoidType``, a structref type with no fields."""
    pass


VoidType = VoidTypeClass([])


class Void(StructRefProxy):
    """The Python object for a ``VoidType`` value that reaches Python.

    Python code cannot create one: calling the class raises ``NotImplementedError``.
    """
    def __new__(cls, x):
        raise NotImplementedError("Not for use in Python")


# Box a VoidType value that reaches Python as a Void, and unbox a Void that Python passes to jitted code.
define_boxing(VoidTypeClass, Void)
