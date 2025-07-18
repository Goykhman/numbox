from numba.experimental.structref import register
from numba.core.types import StructRef


@register
class S1TypeClass(StructRef):
    pass


@register
class S2TypeClass(StructRef):
    pass
