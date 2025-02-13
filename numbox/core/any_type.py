from numba import njit
from numba.core.errors import NumbaError
from numba.core.types import StructRef
from numba.experimental.structref import define_boxing, new, register, StructRefProxy
from numba.extending import overload, overload_method

from numbox.utils.highlevel import prune_type
from numbox.utils.lowlevel import cast, deref


@register
class ErasedTypeClass(StructRef):
    pass


ErasedType = ErasedTypeClass([])


@register
class ContentTypeClass(StructRef):
    pass


class _Content:
    pass


@overload(_Content, strict=False)
def ol_content(x_ty):
    x_ty = prune_type(x_ty)
    content_type = ContentTypeClass([("x", x_ty)])

    def _(x):
        c = new(content_type)
        c.x = x
        return c
    return _


@register
class AnyTypeClass(StructRef):
    pass


deleted_any_ctor_error = 'Use `make_any` instead'


class Any(StructRefProxy):
    def __new__(cls, x):
        raise NotImplementedError(deleted_any_ctor_error)

    @njit
    def get_as(self, ty):
        return self.get_as(ty)

    @njit
    def reset(self, val):
        return self.reset(val)


def _any_deleted_ctor(p):
    raise NumbaError(deleted_any_ctor_error)


overload(Any)(_any_deleted_ctor)
define_boxing(AnyTypeClass, Any)
AnyType = AnyTypeClass([("p", ErasedType)])


@overload_method(AnyTypeClass, "get_as", strict=False)
def ol_get_as(self_class, ty_class):
    def _(self, ty):
        return deref(self.p, ty)
    return _


@overload_method(AnyTypeClass, "reset", strict=False)
def ol_reset(self_class, x_class):
    def _(self, x):
        self.p = cast(_Content(x), ErasedType)
    return _


@njit
def make_any(x):
    any = new(AnyType)
    any.p = cast(_Content(x), ErasedType)
    return any
