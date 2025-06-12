from numba import njit
from numba.core.errors import NumbaError
from numba.core.types import StructRef, TypeRef, unicode_type
from numba.experimental.structref import define_boxing, new, register, StructRefProxy
from numba.extending import overload, overload_method

from numbox.core.configurations import default_jit_options
from numbox.core.any.content_wrap import _Content
from numbox.core.any.erased_type import ErasedType
from numbox.utils.lowlevel import _cast, _deref_payload


@register
class AnyTypeClass(StructRef):
    pass


deleted_any_ctor_error = 'Use `make_any` instead'


class Any(StructRefProxy):
    def __new__(cls, x):
        raise NotImplementedError(deleted_any_ctor_error)

    @njit(**default_jit_options)
    def get_as(self, ty):
        return self.get_as(ty)

    @njit(**default_jit_options)
    def reset(self, val):
        return self.reset(val)

    @property
    @njit(**default_jit_options)
    def type_info(self):
        return self.t


def _any_deleted_ctor(p):
    raise NumbaError(deleted_any_ctor_error)


overload(Any, jit_options=default_jit_options)(_any_deleted_ctor)
define_boxing(AnyTypeClass, Any)
AnyType = AnyTypeClass([("p", ErasedType), ("t", unicode_type)])


@overload_method(AnyTypeClass, "get_as", strict=False, jit_options=default_jit_options)
def ol_get_as(self_ty, ty_ref: TypeRef):
    ty_code = str(ty_ref.instance_type)

    def _(self, ty):
        if ty_code != self.t:
            raise NumbaError(f"Any stored type {self.t}, cannot decode as {ty_code}")
        return _deref_payload(self.p, ty)
    return _


@overload_method(AnyTypeClass, "reset", strict=False, jit_options=default_jit_options)
def ol_reset(self_ty, x_ty):
    ty_code = str(x_ty)

    def _(self, x):
        self.p = _cast(_Content(x), ErasedType)
        self.t = ty_code
    return _


def _make_any(x):
    raise NotImplementedError("Not callable from Python")


@overload(_make_any, strict=False, jit_options=default_jit_options)
def ol_make_any(x_ty):
    ty_code = str(x_ty)

    def _(x):
        any = new(AnyType)
        any.p = _cast(_Content(x), ErasedType)
        any.t = ty_code
        return any
    return _


@njit(**default_jit_options)
def make_any(x):
    return _make_any(x)
