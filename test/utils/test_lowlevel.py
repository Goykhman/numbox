import numba
import numpy
from ctypes import c_char_p, c_void_p
from numba import float64
from numba.experimental.function_type import _get_jit_address, _get_wrapper_address
from numba.extending import intrinsic

from numbox.utils.meminfo import get_nrt_refcount, structref_meminfo
from numbox.utils.highlevel import cres
from numbox.utils.lowlevel import (
    cast, deref_payload, extract_struct_member, get_func_p_as_int_from_func_struct,
    get_func_tuple, get_str_from_p_as_int, get_unicode_data_p, tuple_of_struct_ptrs_as_int,
    uniformize_tuple_of_structs
)
from test.auxiliary_utils import collect_and_run_tests, deref_int64_intp, str_from_p_as_int
from test.common_structrefs import ll_make_s4, S1, S1Type, S12, S12Type, S2


def test_1():
    """ `cast` does not create a new `MemInfo`. Consistently with that, ref-count
     increments when a cast structure points at the same `MemInfo` """
    x1 = 23
    s1 = S1(x1, 567, 1.23)
    assert s1.x1 == x1
    assert get_nrt_refcount(s1) == 1
    s1_as_s12 = cast(s1, S12Type)
    assert get_nrt_refcount(s1) == 2, "`s1_as_s12` is an independent owner of `s1`'s payload"
    assert s1_as_s12.x1 == x1
    s1_meminfo_p = structref_meminfo(s1)[0]
    s1_as_s12_meminfo_p = structref_meminfo(s1_as_s12)[0]
    assert s1_meminfo_p == s1_as_s12_meminfo_p
    assert get_nrt_refcount(s1_as_s12) == 2
    del s1
    assert get_nrt_refcount(s1_as_s12) == 1


def test_2():
    """ Round-trip for `deref` returns the original object """
    a = numpy.array([[34], [56]], dtype=numpy.int64)
    s2 = S2(a)
    a1 = deref_payload(s2, numba.types.npytypes.Array(numba.int64, 2, 'C'))
    a_p = a.ctypes.data
    a1_p = a1.ctypes.data
    assert a_p == a1_p


def test_3():
    """ `deref` does not create a new `MemInfo`. It returns the original
     payload of the structref and therefore does not increment its ref-count.
     It's different from `cast` that creates an additional owner of the same payload.
     Instead, `deref` simply returns the original owner. """

    arr_ty = numba.types.npytypes.Array(numba.int64, 2, 'C')

    @numba.njit
    def aux():
        """ This is jitted because we want `a` to get wrapped in a `MemInfo`, which requires NRT """
        a = numpy.array([[34], [56]], dtype=numpy.int64)
        a_meminfo_p = structref_meminfo(a)[0]
        a_ref_ct_1_ = deref_int64_intp(a_meminfo_p)

        s2 = S2(a)
        a_ref_ct_2_ = deref_int64_intp(a_meminfo_p)

        s2_data_p = structref_meminfo(s2)[1]
        # `a` is wrapped in `MemInfo` pointed at by `s2`'s `MemInfo`'s data member
        a_wrap_meminfo_p_ = deref_int64_intp(s2_data_p)

        a1 = deref_payload(s2, arr_ty)
        a_ref_ct_3_ = deref_int64_intp(a_meminfo_p)

        a1_wrap_meminfo_p_ = structref_meminfo(a1)[0]

        return (
            a, a1, a_wrap_meminfo_p_, a1_wrap_meminfo_p_, a_ref_ct_1_, a_ref_ct_2_, a_ref_ct_3_
        )

    (
        a, a1, a_wrap_meminfo_p, a1_wrap_meminfo_p, a_ref_ct_1, a_ref_ct_2, a_ref_ct_3
    ) = aux()
    assert a_wrap_meminfo_p == a1_wrap_meminfo_p
    assert a_ref_ct_1 == 1
    assert a_ref_ct_2 == 2
    assert a_ref_ct_3 == 3


def test_extract_data_member():

    @intrinsic(prefer_literal=True)
    def _extract_data_member(typingctx, struct_ty, member_name_ty, member_type_ref):
        member_name_ = member_name_ty.literal_value
        member_ty = member_type_ref.instance_type
        sig = member_ty(struct_ty, member_name_ty, member_type_ref)

        def codegen(context, builder, signature, args):
            return extract_struct_member(context, builder, struct_ty, args[0], member_name_)
        return sig, codegen

    member_name = "x3"
    x1_ty = S1Type.field_dict[member_name]

    @numba.njit
    def get_x3(s_):
        return _extract_data_member(s_, member_name, x1_ty)

    x3 = 3.14
    s1 = S1(12, 137, x3)

    assert abs(get_x3(s1) - x3) < 1e-15


def test_get_func_p_from_func_struct():
    func_sig = float64(float64, float64)

    @cres(func_sig, cache=True)
    def func(x, y):
        return x + y

    assert get_func_p_as_int_from_func_struct(func) == _get_wrapper_address(func, func_sig)
    assert get_func_p_as_int_from_func_struct(func) == func.address


def test_get_func_tuple():
    func_sig = float64(float64, float64)

    @cres(func_sig, cache=True)
    def func(x, y):
        return x + y

    func_tup = get_func_tuple(func)
    assert func_tup[0] == _get_wrapper_address(func, func_sig)
    assert func_tup[1] == id(func)
    assert func_tup[2] == 0, "void `jit_addr` in `FunctionModel` for cres"
    assert func_tup[2] == _get_jit_address(func, func_sig), """
void `jit_addr` in `FunctionModel` for cres, added to FunctionType data model in numba==0.61
https://github.com/numba/numba/blob/release0.61/numba/experimental/function_type.py#L65
"""


def test_get_unicode_data_p():
    s1_ = "a random string"
    s1_a = get_unicode_data_p(s1_)
    assert str_from_p_as_int(s1_a) == s1_


def test_get_str_from_p_as_int():
    s1_ = "a random string"
    s1_b = s1_.encode("utf-8")
    s1 = c_char_p(s1_b)
    s1_p = c_void_p.from_buffer(s1).value
    s1 = get_str_from_p_as_int(s1_p)
    assert s1 == s1_


def test_populate_structref():
    @numba.njit
    def make_s4():
        """ In a jitted context `a` gets its own meminfo """
        a_ = numpy.array([2.17, 3.14], dtype=numpy.float64)
        a_meminfo_p = structref_meminfo(a_)[0]
        a_ref_ct_1_ = deref_int64_intp(a_meminfo_p)
        s4_ = ll_make_s4(a_)
        a_ref_ct_2_ = deref_int64_intp(a_meminfo_p)
        return a_, s4_, a_ref_ct_1_, a_ref_ct_2_

    a, s4, a_ref_ct_1, a_ref_ct_2 = make_s4()
    assert a_ref_ct_1 == 1
    assert a_ref_ct_2 == 2
    assert a.ctypes.data == s4.x.ctypes.data


def test_tuple_of_struct_ptrs_as_int():
    s1 = S1(1, 2, 3.14)
    s12 = S12(4)
    t = tuple_of_struct_ptrs_as_int((s1, s12))
    s1_meminfo_p_as_int = structref_meminfo(s1)[0]
    s12_meminfo_p_as_int = structref_meminfo(s12)[0]
    assert t[0] == s1_meminfo_p_as_int
    assert t[1] == s12_meminfo_p_as_int


def test_uniformize_tuple_of_structs():
    s1_x1 = 11
    s1_x2 = 137
    s1_x3 = 3.14
    s1 = S1(s1_x1, s1_x2, s1_x3)
    s12_x1 = 44
    s12 = S12(s12_x1)
    t = uniformize_tuple_of_structs((s1, s12))

    s1_meminfo_p_as_int = structref_meminfo(s1)[0]
    s12_meminfo_p_as_int = structref_meminfo(s12)[0]

    assert structref_meminfo(t[0])[0] == s1_meminfo_p_as_int
    assert structref_meminfo(t[1])[0] == s12_meminfo_p_as_int

    s1_ = cast(t[0], S1Type)
    s12_ = cast(t[1], S12Type)

    s1__meminfo_p_as_int = structref_meminfo(s1)[0]
    s12__meminfo_p_as_int = structref_meminfo(s12)[0]
    assert s1__meminfo_p_as_int == s1_meminfo_p_as_int
    assert s12__meminfo_p_as_int == s12_meminfo_p_as_int

    assert s1_.x1 == s1_x1

    x1_pr = -23
    s1.x1 = x1_pr
    assert s1_.x1 == x1_pr

    assert s1_.x2 == s1_x2
    assert abs(s1_.x3 - s1_x3) < 1e-15

    assert s12_.x1 == s12.x1


if __name__ == '__main__':
    collect_and_run_tests(__name__)
