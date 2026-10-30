"""Tests for ``load_unaligned`` and ``store_unaligned`` in ``numbox.utils.lowlevel``.

The tests read an ``int64`` at a misaligned address, ``int32`` values at aligned addresses and a ``float64`` through
``load_unaligned``. Two tests check in the IR that ``load_unaligned`` and ``store_unaligned`` emit an alignment of 1,
while ``load_at`` and ``store_at`` emit their type's alignment. A misaligned access through ``load_at`` or ``store_at``
also gives the right value on x86-64, so only the IR distinguishes the two pairs of functions.
"""
import numpy as np
from numba import njit
from numba.core.types import int32, int64, float64
from numbox.utils.lowlevel import load_unaligned, store_unaligned, load_at, store_at, array_data_p


def test_load_unaligned_reads_misaligned_int64():
    # The int64 starts at byte offset 1 of the buffer, so its address is misaligned.
    buf = np.zeros(16, dtype=np.uint8)
    buf[1:9] = np.frombuffer(np.int64(0x0123456789ABCDEF).tobytes(), dtype=np.uint8)

    @njit
    def read(base):
        return load_unaligned(base + 1, int64)

    assert read(array_data_p(buf)) == 0x0123456789ABCDEF


def test_load_unaligned_matches_load_at_aligned():
    a = np.array([7, -3, 11], dtype=np.int32)

    @njit
    def read(base, i):
        return load_unaligned(base + 4 * i, int32)

    base = array_data_p(a)
    assert [read(base, i) for i in range(3)] == [7, -3, 11]


def test_load_unaligned_float64():
    x = np.array([3.5], dtype=np.float64)

    @njit
    def read(base):
        return load_unaligned(base, float64)

    assert read(array_data_p(x)) == 3.5


def test_store_unaligned_emits_align1_not_natural():
    # On x86-64, store_at at a misaligned address writes the same bytes as store_unaligned, so the stored value does
    # not distinguish the two. The test therefore reads the IR: store_unaligned must emit its store with align 1, and
    # store_at, whose store at a misaligned address is undefined behavior, emits the natural alignment.
    @njit
    def w_unaligned(p):
        store_unaligned(p + 1, int64(0x0102030405060708))

    @njit
    def w_at(p):
        store_at(p + 1, int64(0x0102030405060708))

    buf = np.zeros(16, dtype=np.uint8)
    w_unaligned(array_data_p(buf))
    assert int(np.frombuffer(buf[1:9].tobytes(), dtype=np.int64)[0]) == 0x0102030405060708
    w_at(array_data_p(buf))  # this call compiles w_at, whose IR the assertions below read

    needle = str(0x0102030405060708)  # LLVM prints the i64 constant in decimal

    def const_stores(f):
        ir = "\n".join(f.inspect_llvm().values())
        return [ln for ln in ir.splitlines() if needle in ln and " store " in ln]

    assert any("align 1" in ln for ln in const_stores(w_unaligned)), const_stores(w_unaligned)
    assert const_stores(w_at) and all("align 1" not in ln for ln in const_stores(w_at)), const_stores(w_at)


def test_load_unaligned_emits_align1_not_natural():
    # load_unaligned must emit its load with align 1, and load_at must not, as for the stores above.
    buf = np.zeros(16, dtype=np.uint8)
    buf[1:9] = np.frombuffer(np.int64(0x0102030405060708).tobytes(), dtype=np.uint8)

    @njit
    def r_unaligned(p):
        return load_unaligned(p + 1, int64)

    @njit
    def r_at(p):
        return load_at(p + 1, int64)

    assert r_unaligned(array_data_p(buf)) == 0x0102030405060708
    assert r_at(array_data_p(buf)) == 0x0102030405060708  # undefined behavior for load_at; x86-64 reads the value

    def i64_loads(f):
        ir = "\n".join(f.inspect_llvm().values())
        return [ln for ln in ir.splitlines() if "= load i64" in ln]

    assert any("align 1" in ln for ln in i64_loads(r_unaligned)), i64_loads(r_unaligned)
    assert i64_loads(r_at) and all("align 1" not in ln for ln in i64_loads(r_at)), i64_loads(r_at)
