from inspect import getmodule, getsource
from numba import njit
from numba.core.errors import NumbaError
from numba.core.itanium_mangler import mangle_type_or_value
from numba.core.types import StructRef, Type
from numba.experimental.structref import define_boxing, new, register, StructRefProxy
from numba.extending import overload, overload_method

from numbox.core.configurations import default_jit_options
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


@overload(_Content, strict=False, jit_options=default_jit_options)
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

    @njit(**default_jit_options)
    def get_as(self, ty):
        return self.get_as(ty)

    @njit(**default_jit_options)
    def reset(self, val):
        return self.reset(val)


def _any_deleted_ctor(p):
    raise NumbaError(deleted_any_ctor_error)


overload(Any, jit_options=default_jit_options)(_any_deleted_ctor)
define_boxing(AnyTypeClass, Any)
AnyType = AnyTypeClass([("p", ErasedType)])


@overload_method(AnyTypeClass, "get_as", strict=False, jit_options=default_jit_options)
def ol_get_as(self_class, ty_class):
    def _(self, ty):
        return deref(self.p, ty)
    return _


@overload_method(AnyTypeClass, "reset", strict=False, jit_options=default_jit_options)
def ol_reset(self_class, x_class):
    def _(self, x):
        self.p = cast(_Content(x), ErasedType)
    return _


def make_any_prototype(x):
    any = new(AnyType)
    any.p = cast(_Content(x), ErasedType)
    return any


make_any = njit(**default_jit_options)(make_any_prototype)


def make_any_maker(arg_type=None, jit_options=None):
    if arg_type is None:
        return make_any
    if not isinstance(arg_type, Type):
        raise TypeError(f"Expected numba type, got {arg_type}")
    jit_options = {**default_jit_options, **(jit_options or {})}
    sig = AnyType(arg_type)
    custom_make_any_name = f"make_any_{mangle_type_or_value(arg_type)}"
    source_code = getsource(make_any_prototype)
    source_code = source_code.replace('make_any_prototype', custom_make_any_name)
    code_txt = f"""
@njit(sig, **jit_options)
{source_code}
"""
    code = compile(code_txt, __file__, mode='exec')
    ns = {**getmodule(make_any).__dict__, **{'sig': sig, 'jit_options': jit_options}}
    exec(code, ns)
    custom_make_any = ns[custom_make_any_name]
    return custom_make_any
