from numba.core.types import StructRef
from numba.experimental.structref import register


@register
class ErasedTypeClass(StructRef):
    pass


ErasedType = ErasedTypeClass([])
