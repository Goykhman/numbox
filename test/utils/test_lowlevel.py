import numba
import numpy

from test.common_structrefs import S1, S12Type, S2
from numbox.core.meminfo import get_nrt_refcount, structref_meminfo
from numbox.utils.lowlevel import cast, deref
from test.auxiliary_utils import deref_int64_intp


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
    a1 = deref(s2, numba.types.npytypes.Array(numba.int64, 2, 'C'))
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
        a_ref_ct = deref_int64_intp(a_meminfo_p)

        s2 = S2(a)
        a_ref_ct_2 = deref_int64_intp(a_meminfo_p)

        s2_data_p = structref_meminfo(s2)[1]
        # `a` is wrapped in `MemInfo` pointed at by `s2`'s `MemInfo`'s data member
        a_wrap_meminfo_p = deref_int64_intp(s2_data_p)

        a1 = deref(s2, arr_ty)
        a_ref_ct_3 = deref_int64_intp(a_meminfo_p)

        a1_wrap_meminfo_p = structref_meminfo(a1)[0]
        return a_wrap_meminfo_p, a1_wrap_meminfo_p, a_ref_ct, a_ref_ct_2, a_ref_ct_3

    a_wrap_meminfo_p, a1_wrap_meminfo_p, a_ref_ct, a_ref_ct_2, a_ref_ct_3 = aux()
    assert a_wrap_meminfo_p == a1_wrap_meminfo_p
    assert a_ref_ct == 1
    assert a_ref_ct_2 == 2
    assert a_ref_ct_3 == 2, "`deref` returns the original owner of the data payload"


if __name__ == '__main__':
    test_1()
    test_2()
    test_3()
