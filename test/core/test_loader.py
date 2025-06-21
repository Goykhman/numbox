from numba.core.types import unicode_type
from numba.typed.typeddict import Dict
from numpy import isclose

from numbox.core.any.any_type import AnyType, make_any
from numbox.core.work.work_utils import make_work_helper


def test_loader_1():
    w1 = make_work_helper("w1", 0.0)
    assert w1.data == 0
    d1 = Dict.empty(key_type=unicode_type, value_type=AnyType)
    d1["w1"] = make_any(3.14)
    reset = w1.load(d1)
    assert reset
    assert isclose(w1.data, 3.14)


def test_loader_2():
    w1 = make_work_helper("w1", 1.41)
    w2 = make_work_helper("w2", 1.72)
    assert isclose(w1.data, 1.41)
    assert isclose(w2.data, 1.72)
    w3 = make_work_helper("w3", 0.0, sources=(w1, w2), derive_py=lambda w1_, w2_: w1_ + 2 * w2_)
    assert w3.data == 0
    assert not w3.derived
    w3.calculate()
    assert w3.derived
    assert isclose(w3.data, 1.41 + 2 * 1.72)

    load_data = Dict.empty(key_type=unicode_type, value_type=AnyType)
    load_data["w1"] = make_any(3.14)
    reset = w3.load(load_data)
    assert reset
    assert not w3.derived
    assert isclose(w1.data, 3.14)
    assert isclose(w2.data, 1.72)
    assert isclose(w3.data, 1.41 + 2 * 1.72)
    assert not w3.derived
    w3.calculate()
    assert isclose(w3.data, 3.14 + 2 * 1.72)
