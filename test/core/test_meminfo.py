"""Tests for ``numbox.utils.meminfo``, which passes a structref through an integer address and reads its MemInfo.

The tests read numba's ``MemInfo`` structure and a structref's data with ``ctypes``, at the addresses that
``structref_meminfo`` returns. They check the reference counts that ``export_meminfo``, ``borrow_structref`` and
``release_meminfo`` produce, and the ``TypingError`` for an address of a narrower integer type. Two tests check the
marker that keeps the ``removerefctpass`` of numba 0.65 and earlier from deleting the increment of an exported
reference.
"""
import ctypes
import numba
import numpy

from numbox.utils.meminfo import structref_meminfo, get_nrt_refcount
from test.auxiliary_utils import collect_and_run_tests
from test.common_structrefs import S1, S2


# numba's MemInfo structure has six fields of eight bytes, and its data field is the fourth, at index 3:
# https://github.com/numba/numba/blob/main/numba/core/runtime/nrt.cpp#L17
size_of_meminfo = 6
offset_to_data_p_in_meminfo_p = 3


def test_s1():
    """The data field of an ``S1``'s MemInfo holds the address of the struct's data, which follows the MemInfo.

    The three fields are read with ``ctypes`` at offsets of 0, 8 and 16 bytes from the start of the data.
    """
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
    """The data of an ``S2`` starts with the address of its array's MemInfo.

    The data field of that MemInfo holds the address of the array's data.
    """
    x1 = numpy.array([[137], [214], [317]], dtype=numpy.int64)
    s2 = S2(x1)
    assert get_nrt_refcount(s2) == 1
    s2_meminfo_p_as_int, s2_data_p_as_int = structref_meminfo(s2)

    # The array is the first field of s2's data, and an array's data model starts with its MemInfo pointer.
    x1_member_meminfo_pp = s2_data_p_as_int
    x1_member_meminfo_p = ctypes.c_int64.from_address(x1_member_meminfo_pp).value
    x1_data_pp = x1_member_meminfo_p + offset_to_data_p_in_meminfo_p * 8
    x1_data_p = ctypes.c_int64.from_address(x1_data_pp).value
    assert x1_data_p == x1.ctypes.data
    assert x1.ctypes.data != s2_meminfo_p_as_int + size_of_meminfo * 6


def test_bridge_refcount_ladder():
    """``export_meminfo`` raises the count to 2, a borrow leaves it at 2, and ``release_meminfo`` returns it to 1."""
    from numbox.utils.meminfo import (
        borrow_structref, export_meminfo, get_nrt_refcount, release_meminfo,
    )
    from test.common_structrefs import S1, S1Type

    s = S1(1, 2, 3.0)
    assert get_nrt_refcount(s) == 1
    p = export_meminfo(s)
    assert get_nrt_refcount(s) == 2

    @numba.njit
    def bounce(p_):
        t = borrow_structref(S1Type, p_)
        return t.x1

    assert bounce(p) == 1
    release_meminfo(p)
    assert get_nrt_refcount(s) == 1


def test_bridge_round_trip_recovers_fields():
    from numbox.utils.meminfo import borrow_structref, export_meminfo
    from test.common_structrefs import S1, S1Type

    s = S1(42, 43, 3.14)
    p = export_meminfo(s)

    @numba.njit
    def read_all(p_):
        t = borrow_structref(S1Type, p_)
        return (t.x1, t.x2, t.x3)

    assert read_all(p) == (42, 43, 3.14)


def test_bridge_intrinsics_reject_non_intp():
    """``_incref_meminfo`` raises ``TypingError`` for an address typed ``int32`` when the caller compiles.

    An ``int32`` holds only the low 32 bits of a 64-bit address, so the increment would write to a wrong address.
    """
    import pytest
    from numba import int32, njit
    from numba.core.errors import TypingError
    from numbox.utils.meminfo import _incref_meminfo

    def caller(x):
        _incref_meminfo(x)
        return x

    with pytest.raises(TypingError, match=r"_incref_meminfo expects"):
        njit(int32(int32))(caller)


def test_incref_survives_refcount_pruning():
    """The ``removerefctpass`` of numba 0.65 and earlier does not delete the increment of ``_incref_meminfo``.

    The pass deletes every ``NRT_incref`` of a function whose types need no reference counting, and a ``void(intp)``
    function passes its check. The counts can fail only on a numba version that has the pass, so the test also checks
    the IR for the ``numba_args_may_always_need_nrt`` marker that stops the pass, which numbox adds on every numba
    version.
    """
    from numbox.utils.meminfo import _incref_meminfo, export_meminfo, get_nrt_refcount, release_meminfo
    from test.common_structrefs import S1

    @numba.njit
    def pin(p_):
        _incref_meminfo(p_)

    s = S1(1, 2, 3.0)
    p = export_meminfo(s)
    assert get_nrt_refcount(s) == 2
    pin(p)
    assert get_nrt_refcount(s) == 3
    release_meminfo(p)
    release_meminfo(p)
    assert get_nrt_refcount(s) == 1
    assert "!numba_args_may_always_need_nrt" in pin.inspect_llvm(pin.signatures[0])


def test_export_meminfo_keeps_its_incref_for_an_array():
    """``export_meminfo`` keeps its increment for an array: the array has one more Python reference after the call.

    A function with an array argument passes the check of the ``removerefctpass`` of numba 0.65 and earlier. Without
    the ``numba_args_may_always_need_nrt`` marker, the pass would delete the increment, and the returned address would
    refer to a MemInfo that is already freed. The unboxing wraps the array in a MemInfo that holds a Python reference to
    the array, and that MemInfo is the one exported.
    """
    import sys

    from numbox.utils.meminfo import export_meminfo, release_meminfo

    a = numpy.zeros(4)
    before = sys.getrefcount(a)
    p = export_meminfo(a)
    assert sys.getrefcount(a) == before + 1
    release_meminfo(p)
    assert sys.getrefcount(a) == before
    sig = next(s for s in export_meminfo.signatures if isinstance(s[0], numba.types.Array))
    assert "!numba_args_may_always_need_nrt" in export_meminfo.inspect_llvm(sig)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
