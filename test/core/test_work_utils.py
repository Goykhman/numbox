import numpy
import numpy as np
import pytest

from numba import float64
from numba.core.types import unicode_type
from numba.typed.typeddict import Dict
from numba.typed.typedlist import List

from numbox.core.configurations import default_jit_options
from numbox.core.work.node import Node
from numbox.core.work.print_tree import make_image
from numbox.core.work.work_utils import make_init_data, make_work_helper
from test.auxiliary_utils import collect_and_run_tests
from test.common_structrefs import S1


def test_make_work_helper():
    w = make_work_helper("", 0.0, sources=(), derive_py=lambda: 3.14)
    assert w.data == 0
    w.calculate()
    assert abs(w.data - 3.14) < 1e-15


def test_bad_make_work_helper():
    with pytest.raises(TypeError):
        _ = make_work_helper("", 0.0, sources=(), derive_py=lambda x: 2 * x)


h = 5
time_range = make_init_data(shape=(h,), ty=numpy.int16)
time = make_work_helper("time", time_range)


def calc_val_py(time_array):
    val_data_ = numpy.empty((h,), dtype=numpy.float64)
    for time_ind in range(time_array.shape[0]):
        if time_ind % 2 == 0:
            val_data_[time_ind] = 3.14
        else:
            val_data_[time_ind] = 2.17
    return val_data_


def test_simple_time_1():
    val_init_data = make_init_data(shape=(h,), ty=numpy.float64)
    val = make_work_helper("val", val_init_data, (time,), calc_val_py, default_jit_options)

    val.calculate()

    data_ref = numpy.array([3.14, 2.17, 3.14, 2.17, 3.14], dtype=numpy.float64)
    assert numpy.allclose(val.data, data_ref)


def test_simple_time_2():
    def calc_ptr(data_):
        return data_.ctypes.data

    ptr = make_work_helper("ptr", 0, (time,), calc_ptr)
    ptr.calculate()
    assert ptr.data == time_range.ctypes.data, "original time range array is expected to be used"


def test_simple_time_3():
    tau = make_work_helper("tau", h // 2)

    def initialize_x0():
        return 137.0
    x0 = make_work_helper("x0", 0.0, (), initialize_x0)

    def derive_y0(x0_):
        return x0_ * 1.41 + 1.72
    y0_init_data = make_init_data()
    y0 = make_work_helper("y0", y0_init_data, (x0,), derive_y0)

    def derive_y1(x0_, time_, tau_):
        y1_data_ = numpy.empty((h,), dtype=numpy.float64)
        for time_ind in range(time_.shape[0]):
            if time_ind < tau_:
                y1_data_[time_ind] = 2.3 + x0_
            else:
                y1_data_[time_ind] = 1.4 * x0_
        return y1_data_
    y1_init_data = make_init_data(shape=(h,), ty=numpy.float64)
    y1 = make_work_helper("y1", y1_init_data, (x0, time, tau), derive_y1)

    def derive_y2(y0_, y1_):
        return y0_ + 2 * y1_
    y2_init_data = make_init_data(shape=(h,), ty=numpy.float64)
    y2 = make_work_helper("y2", y2_init_data, (y0, y1), derive_y2)

    y2.calculate()

    assert abs(x0.data - 137) < 1e-15
    assert abs(y0.data - (x0.data * 1.41 + 1.72)) < 1e-15
    y1_1 = 2.3 + x0.data
    y1_2 = 1.4 * x0.data
    assert np.allclose(y1.data, [y1_1, y1_1, y1_2, y1_2, y1_2])
    assert np.allclose(y2.data, y0.data + 2 * y1.data)

    assert make_image(y2) == """
y2--y0--x0
    |
    y1--x0
        |
        time
        |
        tau"""

    assert y2.depends_on("time")
    assert not y0.depends_on(time)
    assert y2.all_inputs_names() == ["y0", "x0", "y1", "time", "tau"]

    y2_node = y2.as_node()
    assert y2_node.name == "y2"
    assert y2_node.inputs[0].name == "y0"


def test_dict_data():
    d1_ = {"pi": 3.14, "e": 2.17}
    d1 = Dict.empty(key_type=unicode_type, value_type=float64)
    for k, v in d1_.items():
        d1[k] = v
    w1 = make_work_helper("w1", init_data=d1)

    def derive_w2(w1_dict):
        return w1_dict["pi"] * w1_dict["e"]

    w2 = make_work_helper(
        "w2", make_init_data(), sources=(w1,), derive_py=derive_w2, jit_options=default_jit_options
    )
    w2.calculate()
    assert numpy.isclose(w2.data, 3.14 * 2.17)


def test_list_data():
    l1_ = [3.14, 2.17]
    l1 = List.empty_list(item_type=float64)
    for item in l1_:
        l1.append(item)
    w1 = make_work_helper("w1", init_data=l1)

    def derive_w2(w1_lst):
        return w1_lst[0] * w1_lst[1]

    w2 = make_work_helper(
        "w2", make_init_data(), sources=(w1,), derive_py=derive_w2, jit_options=default_jit_options
    )
    w2.calculate()
    assert numpy.isclose(w2.data, 3.14 * 2.17)


def test_structref_data_1():
    w1_data = S1(12, 137, 1.41)
    w1 = make_work_helper("w1", init_data=w1_data)

    def derive_w2(w1_struct):
        return w1_struct.x1 * w1_struct.x3 + w1_struct.x2 / (w1_struct.calculate(5.3) + 1)

    w2 = make_work_helper(
        "w2", make_init_data(), sources=(w1,), derive_py=derive_w2, jit_options=default_jit_options
    )
    w2.calculate()
    assert numpy.isclose(w2.data, 12 * 1.41 + 137 / ((137 + 5.3) + 1))


def test_structref_data_2():
    w1_data = S1(12, 137, 1.41)
    w1 = make_work_helper("w1", init_data=w1_data)
    w2 = make_work_helper("w2", init_data=2.31)

    def derive_w3(w1_struct, w2_data):
        return w1_struct.x1 * w1_struct.x3 + w1_struct.x2 / (w1_struct.calculate(5.3) + w2_data)

    w3 = make_work_helper(
        "w3", make_init_data(), sources=(w1, w2), derive_py=derive_w3, jit_options=default_jit_options
    )
    w3.calculate()
    assert numpy.isclose(w3.data, 12 * 1.41 + 137 / ((137 + 5.3) + 2.31))


def test_struct_array_data():
    w1_data_ty = numpy.dtype([("name", "|S16"), ("x", numpy.int32), ("y", numpy.float64)])
    w1_data = numpy.array(("fine_euler", 137, 2.17), dtype=w1_data_ty)  # structured scalar
    w1 = make_work_helper("w1", init_data=w1_data)

    def derive_w2(w1_struct):
        w2_ = w1_struct["x"] * w1_struct["y"]
        return w2_.item()

    w2 = make_work_helper(
        "w2", make_init_data(), sources=(w1,), derive_py=derive_w2, jit_options=default_jit_options
    )
    w2.calculate()
    assert numpy.isclose(w2.data, 137 * 2.17)


def test_as_node():
    w1 = make_work_helper("w1", 0.0)
    assert isinstance(w1.as_node(), Node)
    assert isinstance(w1.as_node(), Node)  # returns cached node

    w2 = make_work_helper("w2", 0.0)
    w3 = make_work_helper("w3", 0.0, (w2,), lambda w2_: 0.0)
    _ = w3.make_inputs_vector()  # stores w2 as node
    assert isinstance(w1.as_node(), Node)  # returns cached node


if __name__ == "__main__":
    collect_and_run_tests(__name__)
