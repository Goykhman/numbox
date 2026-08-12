from numba import njit
from numba.core.errors import NumbaError
from numba.core.types import StructRef, TypeRef, unicode_type
from numba.experimental.structref import define_boxing, new, register, StructRefProxy
from numba.extending import overload, overload_method

from numbox.core.configurations import jit_options
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

    @njit(**jit_options)
    def get_as(self, ty):
        return self.get_as(ty)

    @njit(**jit_options)
    def reset(self, val):
        return self.reset(val)

    @property
    @njit(**jit_options)
    def type_info(self):
        return self._get_type_info()


def _any_deleted_ctor(p):
    raise NumbaError(deleted_any_ctor_error)


overload(Any, jit_options=jit_options)(_any_deleted_ctor)
define_boxing(AnyTypeClass, Any)
AnyType = AnyTypeClass([("p", ErasedType), ("t", unicode_type)])
AnyTypeLite = AnyTypeClass([("p", ErasedType)])


@overload_method(AnyTypeClass, "_get_type_info", strict=False, jit_options=jit_options)
def ol_get_type_info(self_ty):
    if self_ty == AnyType:
        def _(self):
            return self.t
        return _

    if self_ty == AnyTypeLite:
        raise AttributeError("AnyTypeLite instance does not store type info.")


@overload_method(AnyTypeClass, "get_as", strict=False, jit_options=jit_options)
def ol_get_as(self_ty, ty_ref: TypeRef):
    if self_ty == AnyType:
        ty_code = str(ty_ref.instance_type)

        def _(self, ty):
            if ty_code != self.t:
                raise NumbaError(f"Any stored type {self.t}, cannot decode as {ty_code}")
            return _deref_payload(self.p, ty)

        return _

    if self_ty == AnyTypeLite:
        def _(self, ty):
            return _deref_payload(self.p, ty)

        return _


@overload_method(AnyTypeClass, "reset", strict=False, jit_options=jit_options)
def ol_reset(self_ty, x_ty):
    if self_ty == AnyType:
        ty_code = str(x_ty)

        def _(self, x):
            self.p = _cast(_Content(x), ErasedType)
            self.t = ty_code
        return _

    if self_ty == AnyTypeLite:
        def _(self, x):
            self.p = _cast(_Content(x), ErasedType)

        return _


def _make_any(x):
    raise NotImplementedError("Not callable from Python")


@overload(_make_any, strict=False, jit_options=jit_options)
def ol_make_any(x_ty):
    ty_code = str(x_ty)

    def _(x):
        any_ = new(AnyType)
        any_.p = _cast(_Content(x), ErasedType)
        any_.t = ty_code
        return any_
    return _


def _make_any_lite(x):
    raise NotImplementedError("Not callable from Python")


@overload(_make_any_lite, strict=False, jit_options=jit_options)
def ol_make_any_lite(x_ty):
    def _(x):
        any_ = new(AnyTypeLite)
        any_.p = _cast(_Content(x), ErasedType)
        return any_
    return _


@njit(**jit_options)
def make_any(x):
    return _make_any(x)


@njit(**jit_options)
def make_any_lite(x):
    return _make_any_lite(x)
