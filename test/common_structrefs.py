"""Structrefs written by hand that several test modules share.

``S1``, ``S12``, ``S2`` and ``S3`` follow numba's pattern for a StructRef: a type class registered with
``structref.register``, a proxy class with its boxing, an ``@overload`` constructor and a jitted constructor for one
signature. The one field of ``S12`` has the type of the first field of ``S1``, so ``cast`` can read an ``S1`` as an
``S12``. ``S2`` holds a two-dimensional array, and ``S3`` holds an ``S1``. Python code cannot create an ``S4``, and
jitted code creates one with ``ll_make_s4``.
"""
from numba import njit, float64, int16, int64
from numba.core import types
from numba.experimental import structref
from numba.extending import intrinsic, overload, overload_method

from numbox.utils.lowlevel import _new, populate_structref


@structref.register
class S1TypeClass(types.StructRef):
    pass


class S1(structref.StructRefProxy):
    def __new__(cls, x1, x2, x3):
        return s1_constructor(x1, x2, x3)

    def __repr__(self):
        return f"S1(x1={self.x1}, x2={self.x2}, x3={self.x3})"

    @property
    @njit
    def x1(self):
        return self.x1

    @x1.setter
    @njit
    def x1(self, val):
        self.x1 = val

    @property
    @njit
    def x2(self):
        return self.x2

    @property
    @njit
    def x3(self):
        return self.x3

    @njit
    def calculate(self, x):
        return self.calculate(x)


structref.define_boxing(S1TypeClass, S1)
fields_s1 = [("x1", int16), ("x2", int64), ("x3", float64)]
S1Type = S1TypeClass(fields_s1)


@overload(S1, strict=False)
def ol_s1(x1_ty, x2_ty, x3_ty):
    def _(x1, x2, x3):
        s1_ = structref.new(S1Type)
        s1_.x1 = x1
        s1_.x2 = x2
        s1_.x3 = x3
        return s1_
    return _


@njit(S1Type(int16, int64, float64))
def s1_constructor(x1, x2, x3):
    return S1(x1, x2, x3)


@overload_method(S1TypeClass, "calculate", strict=False)
def ol_calculate(self_ty, x_ty):
    def _(self, x):
        return self.x2 + x
    return _


@structref.register
class S12TypeClass(types.StructRef):
    pass


class S12(structref.StructRefProxy):
    def __new__(cls, x1):
        return s12_constructor(x1)

    @property
    @njit
    def x1(self):
        return self.x1


structref.define_boxing(S12TypeClass, S12)
fields_s12 = [("x1", int16)]
S12Type = S12TypeClass(fields_s12)


@overload(S12, strict=False)
def ol_s12(x1_ty):
    def _(x1):
        s12_ = structref.new(S12Type)
        s12_.x1 = x1
        return s12_
    return _


@njit(S12Type(int16))
def s12_constructor(x1):
    return S12(x1)


@structref.register
class S2TypeClass(types.StructRef):
    pass


class S2(structref.StructRefProxy):
    def __new__(cls, x1):
        return s2_constructor(x1)

    @property
    @njit
    def x1(self):
        return self.x1


structref.define_boxing(S2TypeClass, S2)
x1_array_type = types.npytypes.Array(int64, 2, "C")
fields_s2 = [("x1", x1_array_type)]
S2Type = S2TypeClass(fields_s2)


@overload(S2, strict=False)
def ol_s2(x1_ty):
    def _(x1):
        s2_ = structref.new(S2Type)
        s2_.x1 = x1
        return s2_
    return _


@njit(S2Type(x1_array_type))
def s2_constructor(x1):
    return S2(x1)


@structref.register
class S3TypeClass(types.StructRef):
    pass


class S3(structref.StructRefProxy):
    def __new__(cls, x1, x2):
        return s3_constructor(x1, x2)

    @property
    @njit
    def x1(self):
        return self.x1

    @property
    @njit
    def x2(self):
        return self.x2


structref.define_boxing(S3TypeClass, S3)
fields_s3 = [("x1", S1Type), ("x2", float64)]
S3Type = S3TypeClass(fields_s3)


@overload(S3, strict=False)
def ol_s3(x1_ty, x2_ty):
    def _(x1, x2):
        s3_ = structref.new(S3Type)
        s3_.x1 = x1
        s3_.x2 = x2
        return s3_
    return _


@njit(S3Type(S1Type, float64))
def s3_constructor(x1, x2):
    return S3(x1, x2)


class S4(structref.StructRefProxy):
    def __new__(cls, *args):
        raise NotImplementedError("Not intended to be instantiated in Python")

    @property
    @njit
    def x(self):
        return self.x


@structref.register
class S4TypeClass(types.StructRef):
    pass


structref.define_boxing(S4TypeClass, S4)


@intrinsic(prefer_literal=False)
def ll_make_s4(typingctx, x_ty):
    """Return a new ``S4`` whose field ``x`` holds ``x``, in jitted code.

    The field takes the type of the argument. A constructor of this kind uses less memory, cache disk space and
    compilation time than an ``@overload`` constructor when one jitted function creates many instances, as the
    docstring of ``ll_make_work`` says for ``Work`` nodes.
    """
    attributes_ = [
        ("x", x_ty)
    ]
    s4_type_ = S4TypeClass(attributes_)
    args_names = ["x"]

    def codegen(context, builder, signature, args):
        s4_, data_pointer = _new(context, builder, s4_type_)
        populate_structref(context, builder, signature, s4_type_, s4_, args, args_names)
        return s4_
    return s4_type_(x_ty), codegen
