from numba import int16
from numpy import isclose

from numbox.core.work.builder import Derived, End, make_graph
from numbox.core.work.print_tree import make_image
from test.auxiliary_utils import collect_and_run_tests


w1_ = End(name="w1", init_value=137, ty=int16)
w2_ = End(name="w2", init_value=3.14)
inputs = [w1_, w2_]


def derive_w3(w1_, w2_):
    if w1_ < 0:
        return 0.0
    elif w1_ < 1:
        return 2 * w2_
    return 3 * w2_


def derive_w4(w1_):
    return 2 * w1_


w3_ = Derived(name="w3", init_value=0.0, derive=derive_w3, sources=(w1_, w2_))
w4_ = Derived(name="w4", init_value=0.0, derive=derive_w4, sources=(w1_,))
derived = [w3_, w4_]


def test_1():
    w3 = make_graph(inputs, derived, w3_)
    assert w3.data == 0
    w3.calculate()
    assert isclose(w3.data, 9.42)
    assert make_image(w3) == """
w3--w1
    |
    w2"""


def test_2():
    (w3, w4) = make_graph(inputs, derived, (w3_, w4_))
    assert w3.data == 0
    w3.calculate()
    assert isclose(w3.data, 9.42)
    assert w4.data == 0
    w4.calculate()
    assert isclose(w4.data, 274)
    assert make_image(w4) == """
w4--w1"""


if __name__ == "__main__":
    collect_and_run_tests(__name__)
