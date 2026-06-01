import numpy as np
from numba import njit
from numba.core.types import int32, int64, float64
from numbox.utils.lowlevel import load_unaligned, array_data_p


def test_load_unaligned_reads_misaligned_int64():
    # int64 stored at byte offset 1 of a byte buffer (deliberately misaligned).
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
