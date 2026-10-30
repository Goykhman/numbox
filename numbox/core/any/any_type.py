"""``Any``, a structref that holds one value of any numba type, the counterpart of C++'s ``std::any``.

:func:`make_any` wraps a value in a structref of the value's own type, casts that structref to ``ErasedType`` and
stores it in the field ``p`` of a new ``AnyType`` value, so every ``Any`` has the same numba type, whatever it holds.
``get_as(ty)`` reads the value back as the type ``ty``, and ``reset(x)`` replaces it with ``x``. ``get_as`` does not
check ``ty``: reading a value as another type reinterprets its bytes.
"""
from numba import njit
from numba.core.errors import NumbaError
from numba.core.types import StructRef, TypeRef
from numba.experimental.structref import define_boxing, new, register, StructRefProxy
from numba.extending import overload, overload_method

from numbox.core.configurations import jit_options
from numbox.core.any.content_wrap import _Content
from numbox.core.any.erased_type import ErasedType
from numbox.utils.lowlevel import _cast, _deref_payload


@register
class AnyTypeClass(StructRef):
    """The numba type class of ``AnyType``, the structref type of an ``Any``, with one field ``p`` of ``ErasedType``."""
    pass


deleted_any_ctor_error = 'Use `make_any` instead'


class Any(StructRefProxy):
    """The Python object for an ``AnyType`` value, a structref that holds one value of any numba type.

    Create one with :func:`make_any`. Calling the class raises ``NotImplementedError`` in Python, and jitted code that
    calls it fails to compile.
    """
    def __new__(cls, x):
        raise NotImplementedError(deleted_any_ctor_error)

    @njit(**jit_options)
    def get_as(self, ty):
        """Return the value that the ``Any`` holds, read as the numba type ``ty``.

        ``ty`` must be the type of the stored value. The type is not checked, and another type reinterprets the value's
        bytes.
        """
        return self.get_as(ty)

    @njit(**jit_options)
    def reset(self, val):
        """Replace the value that the ``Any`` holds with ``val``, which can be of another type."""
        return self.reset(val)


def _any_deleted_ctor(p):
    raise NumbaError(deleted_any_ctor_error)


overload(Any, jit_options=jit_options)(_any_deleted_ctor)
define_boxing(AnyTypeClass, Any)
AnyType = AnyTypeClass([("p", ErasedType)])


@overload_method(AnyTypeClass, "get_as", strict=False, jit_options=jit_options)
def ol_get_as(self_ty, ty_ref: TypeRef):
    """Implement ``Any.get_as(ty)`` for jitted code: read the stored value as the type ``ty``, which is not checked."""
    def _(self, ty):
        return _deref_payload(self.p, ty)
    return _


@overload_method(AnyTypeClass, "reset", strict=False, jit_options=jit_options)
def ol_reset(self_ty, x_ty):
    """Implement ``Any.reset(x)`` for jitted code: store a structref that holds ``x`` in ``p``, as ``ErasedType``."""
    def _(self, x):
        self.p = _cast(_Content(x), ErasedType)
    return _


def _make_any(x):
    raise NotImplementedError("Not callable from Python")


@overload(_make_any, strict=False, jit_options=jit_options)
def ol_make_any(x_ty):
    """Implement ``_make_any(x)`` for jitted code: return a new ``Any`` that holds ``x``."""
    def _(x):
        any_ = new(AnyType)
        any_.p = _cast(_Content(x), ErasedType)
        return any_
    return _


@njit(**jit_options)
def make_any(x):
    """Return a new ``Any`` that holds ``x``; Python code and jitted code can call it."""
    return _make_any(x)
