import numpy
import numpy as np
import pytest

from numbox.core.configurations import default_jit_options
from numbox.core.print_tree import make_image
from numbox.core.work_utils import make_init_data, make_work_helper


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
    y0 = make_work_helper("y0", make_init_data(), (x0,), derive_y0)

    def derive_y1(x0_, time_, tau_):
        y1_data_ = numpy.empty((h,), dtype=numpy.float64)
        for time_ind in range(time_.shape[0]):
            if time_ind < tau_:
                y1_data_[time_ind] = 2.3 + x0_
            else:
                y1_data_[time_ind] = 1.4 * x0_
        return y1_data_
    y1 = make_work_helper("y1", make_init_data(shape=(h,), ty=numpy.float64), (x0, time, tau), derive_y1)

    def derive_y2(y0_, y1_):
        return y0_ + 2 * y1_
    y2 = make_work_helper("y2", make_init_data(shape=(h,), ty=numpy.float64), (y0, y1), derive_y2)

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
