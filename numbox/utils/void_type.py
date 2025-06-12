from numba.core.types import StructRef
from numba.experimental.structref import define_boxing, register, StructRefProxy


@register
class VoidTypeClass(StructRef):
    pass


VoidType = VoidTypeClass([])


class Void(StructRefProxy):
    def __new__(cls, x):
        raise NotImplementedError("Not for use in Python")


# Allows using `VoidTypeClass` in Python context
define_boxing(VoidTypeClass, Void)
