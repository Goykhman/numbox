"""Wrap a value of any numba type in a structref of its own type, for :mod:`numbox.core.any.any_type`.

In jitted code, ``_Content(x)`` returns a new structref with one field, ``x``, that holds the value. Its type is
``ContentTypeClass([("x", <the type of x>)])``, so each value type gives a structref type of its own. ``any_type``
casts that structref to ``ErasedType`` to store it in an ``Any``. Only jitted code can call ``_Content`` with a value.
"""
from numba.core.types import StructRef
from numba.experimental.structref import new, register
from numba.extending import overload

from numbox.core.configurations import jit_options


@register
class ContentTypeClass(StructRef):
    """The numba type class of the structref that ``_Content`` returns, with one field ``x`` of the value's type."""
    pass


class _Content:
    pass


@overload(_Content, strict=False, jit_options=jit_options)
def ol_content(x_ty):
    """Implement ``_Content(x)`` for jitted code: return a new structref whose field ``x`` holds ``x``."""
    content_type = ContentTypeClass([("x", x_ty)])

    def _(x):
        c = new(content_type)
        c.x = x
        return c
    return _
