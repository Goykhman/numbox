"""``ErasedType``, a structref type with no fields that holds the value of an ``Any``.

:mod:`numbox.core.any.any_type` wraps each value in a structref of the value's own type and casts that structref to
``ErasedType``, so that every ``Any`` has the same numba type. numbox defines no boxing for ``ErasedType``, so jitted
code cannot return a value of this type to Python.
"""
from numba.core.types import StructRef
from numba.experimental.structref import register


@register
class ErasedTypeClass(StructRef):
    """The numba type class of ``ErasedType``, a structref type with no fields."""
    pass


ErasedType = ErasedTypeClass([])
