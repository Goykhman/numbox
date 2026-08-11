from numba import int8, int64
from numbox.core.any.any_lite import make_any


def test_1():
    x = -65
    any1 = make_any(x)
    assert any1.get_as(int64) == x
    assert any1.get_as(int8) == x
