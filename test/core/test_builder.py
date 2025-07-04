from numba.core.types import float64, int16, unicode_type
from numba.typed.typeddict import Dict
from numpy import isclose

from numbox.core.any.any_type import AnyType, make_any
from numbox.core.work.builder import Derived, End, make_graph
from numbox.core.work.combine_utils import make_sheaf_dict
from numbox.core.work.print_tree import make_image
from test.auxiliary_utils import collect_and_run_tests


w1_ = End(name="w1", init_value=137, ty=int16)
w2_ = End(name="w2", init_value=3.14)
w5_ = End(name="w5", init_value=10)
w6_ = End(name="w6", init_value=7.5)
inputs = [w1_, w2_, w5_, w6_]


def derive_w3(w1_, w2_):
    if w1_ < 0:
        return 0.0
    elif w1_ < 1:
        return 2 * w2_
    return 3 * w2_


def derive_w4(w1_):
    return 2 * w1_


def derive_w7(w3_, w5_):
    return w3_ + (w5_ ** 2)


def derive_w8(w6_, w2_):
    if w6_ > 5:
        return w6_ * w2_
    else:
        return w6_ + w2_


def derive_w9(w3_, w4_, w7_):
    return (w4_ - w3_) / (abs(w7_) + 1e-5)


def derive_w10(w3_, w4_, w7_, w8_, w9_):
    return (w3_ + w4_ + w7_) * 0.1 + (w8_ - w9_)


w3_ = Derived(name="w3", init_value=0.0, derive=derive_w3, sources=(w1_, w2_))
w4_ = Derived(name="w4", init_value=0.0, derive=derive_w4, sources=(w1_,))
w7_ = Derived(name="w7", init_value=0.0, derive=derive_w7, sources=(w3_, w5_))
w8_ = Derived(name="w8", init_value=0.0, derive=derive_w8, sources=(w6_, w2_))
w9_ = Derived(name="w9", init_value=0.0, derive=derive_w9, sources=(w3_, w4_, w7_))
w10_ = Derived(name="w10", init_value=0.0, derive=derive_w10, sources=(w3_, w4_, w7_, w8_, w9_))
derived = [w3_, w4_, w7_, w8_, w9_, w10_]


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


def test_3():
    (w7, w9, w10) = make_graph(inputs, derived, (w7_, w9_, w10_))
    assert make_image(w9) == """
w9--w3--w1
    |   |
    |   w2
    |
    w4--w1
    |
    w7--w3--w1
        |   |
        |   w2
        |
        w5"""
    assert make_image(w10) == """
w10--w3--w1
     |   |
     |   w2
     |
     w4--w1
     |
     w7--w3--w1
     |   |   |
     |   |   w2
     |   |
     |   w5
     |
     w8--w6
     |   |
     |   w2
     |
     w9--w3--w1
         |   |
         |   w2
         |
         w4--w1
         |
         w7--w3--w1
             |   |
             |   w2
             |
             w5"""
    assert w10.data == 0
    w10.calculate()
    assert isclose(w7.data, 109.42)
    assert isclose(w9.data, 2.418022)
    assert isclose(w10.data, 60.416)
    assert w7.all_inputs_names() == ["w3", "w1", "w2", "w5"]
    w8_r = w10.get_input(3)
    assert w8_r.name == "w8"

    requested = ("w4", "w7", "w8")
    sheaf = make_sheaf_dict(requested)
    w10.combine(sheaf)
    assert isclose(sheaf["w4"].get_as(float64), 274)
    assert isclose(sheaf["w7"].get_as(float64), 109.42)
    assert isclose(sheaf["w8"].get_as(float64), 23.55)

    load_data = Dict.empty(key_type=unicode_type, value_type=AnyType)
    load_data["w1"] = make_any(12)
    w10.load(load_data)
    w10.calculate()
    w10.combine(sheaf)
    assert isclose(sheaf["w4"].get_as(float64), 24)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
