from numba.core.types import StructRef
from numba.experimental.structref import new, register
from numba.extending import overload

from numbox.core.configurations import default_jit_options


@register
class ContentTypeClass(StructRef):
    pass


class _Content:
    pass


@overload(_Content, strict=False, jit_options=default_jit_options)
def ol_content(x_ty):
    content_type = ContentTypeClass([("x", x_ty)])

    def _(x):
        c = new(content_type)
        c.x = x
        return c
    return _
