import ctypes
import numpy

from numbox.core.meminfo import structref_meminfo, get_nrt_refcount
from test.common_structrefs import S1, S2


# see https://github.com/numba/numba/blob/main/numba/core/runtime/nrt.cpp#L17
size_of_meminfo = 6
offset_to_data_p_in_meminfo_p = 3


def test_s1():
    x1 = 137
    x2 = 314
    x3 = 2.17
    s1 = S1(x1, x2, x3)
    s1_meminfo_p_as_int, s1_data_p_as_int = structref_meminfo(s1)

    data_pp = s1_meminfo_p_as_int + 8 * offset_to_data_p_in_meminfo_p
    data_p = ctypes.c_int64.from_address(data_pp).value
    assert data_p == s1_data_p_as_int

    assert s1_data_p_as_int - s1_meminfo_p_as_int == 8 * size_of_meminfo, """ 
        This just means that data pointed at by the `data_p` is lined up in memory
        right after the six size-8 fields of the `MemInfo` structure. This doesn't
        always have to be the case, for instance in `test_s2` a separate `MemInfo`
        is created to wrap `x1` array, and there is no reason to expect that the
        layout for that `MemInfo` in memory will be precisely such that the array
        data itself will follow right after it.
    """
    alignment_bytes = 8
    x1_retrieved = ctypes.c_int16.from_address(s1_data_p_as_int).value
    x2_retrieved = ctypes.c_int64.from_address(s1_data_p_as_int + alignment_bytes).value
    x3_retrieved = ctypes.c_double.from_address(s1_data_p_as_int + 2 * alignment_bytes).value
    for x_orig, x_retrieved in zip((x1, x2, x3), (x1_retrieved, x2_retrieved, x3_retrieved)):
        assert abs(x_retrieved - x_orig) < 1e-10, f"retrieved = {x_retrieved} orig = {x_orig}"


def test_s2():
    x1 = numpy.array([[137], [214], [317]], dtype=numpy.int64)
    s2 = S2(x1)
    assert get_nrt_refcount(s2) == 1
    s2_meminfo_p_as_int, s2_data_p_as_int = structref_meminfo(s2)

    # `x1` member in `s2` structure's data
    x1_member_meminfo_pp = s2_data_p_as_int
    x1_member_meminfo_p = ctypes.c_int64.from_address(x1_member_meminfo_pp).value
    x1_data_pp = x1_member_meminfo_p + offset_to_data_p_in_meminfo_p * 8
    x1_data_p = ctypes.c_int64.from_address(x1_data_pp).value
    assert x1_data_p == x1.ctypes.data
    assert x1.ctypes.data != s2_meminfo_p_as_int + size_of_meminfo * 6


if __name__ == "__main__":
    test_s1()
    test_s2()
