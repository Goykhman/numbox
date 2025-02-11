from numba import njit, float64, int16, int64
from numba.core import types
from numba.experimental import structref


@structref.register
class S1TypeClass(types.StructRef):
    pass


class S1(structref.StructRefProxy):
    def __new__(cls, x1, x2, x3):
        return s1_constructor(x1, x2, x3)

    @property
    @njit
    def x1(self):
        return self.x1

    @property
    @njit
    def x2(self):
        return self.x2

    @property
    @njit
    def x3(self):
        return self.x3


fields_s1 = [("x1", int16), ("x2", int64), ("x3", float64)]
structref.define_proxy(S1, S1TypeClass, [field[0] for field in fields_s1])

S1Type = S1TypeClass(fields_s1)


@njit(S1Type(int16, int64, float64))
def s1_constructor(x1, x2, x3):
    return S1(x1, x2, x3)


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


fields_s12 = [("x1", int16)]
structref.define_proxy(S12, S12TypeClass, [field[0] for field in fields_s12])

S12Type = S12TypeClass(fields_s12)


@njit(S12Type(int16))
def s12_constructor(x1):
    return S12(x1)


@structref.register
class S2TypeClass(types.StructRef):
    pass


class S2(structref.StructRefProxy):
    def __new__(cls, x1):
        return s2_constructor(x1)


x1_array_type = types.npytypes.Array(int64, 2, 'C')
fields_s2 = [("x1", x1_array_type)]
structref.define_proxy(S2, S2TypeClass, [field[0] for field in fields_s2])

S2Type = S2TypeClass(fields_s2)


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


fields_s3 = [("x1", S1Type), ("x2", float64)]
structref.define_proxy(S3, S3TypeClass, [field[0] for field in fields_s3])

S3Type = S3TypeClass(fields_s3)


@njit(S3Type(S1Type, float64))
def s3_constructor(x1, x2):
    return S3(x1, x2)
