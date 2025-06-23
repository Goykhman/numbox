from numba import float64, njit
from numpy import isclose

from numbox.core.work.lowlevel_work_utils import ll_make_work
from numbox.core.work.print_tree import make_image
from numbox.utils.highlevel import cres
from test.auxiliary_utils import collect_and_run_tests


def test_make_work_ll():

    @cres(float64())
    def derive_v0():
        return 3.14

    @njit(cache=True)
    def v0_maker(derive_):
        return ll_make_work("v0", 0.0, (), derive_)

    v0 = v0_maker(derive_v0)
    assert v0.data == 0
    assert v0.name == "v0"
    assert v0.inputs == ()
    assert not v0.derived
    v0.calculate()
    assert isclose(v0.data, 3.14)


@cres(float64())
def derive_w1():
    return 3.14


@cres(float64())
def derive_w2():
    return 1.41


@cres(float64(float64, float64))
def derive_w3(w1_, w2_):
    return w1_ * w2_


@cres(float64())
def derive_w4():
    return 1.72


@cres(float64(float64, float64))
def derive_w5(w3_, w4_):
    return w3_ + w4_


@njit(cache=True)
def run_test_work_calculate(derive_w1_, derive_w2_, derive_w3_, derive_w4_, derive_w5_):
    w1 = ll_make_work("w1", 0.0, (), derive_w1_)
    w2 = ll_make_work("w2", 0.0, (), derive_w2_)
    w3 = ll_make_work("w3", 0.0, (w1, w2), derive_w3_)
    w4 = ll_make_work("w4", 0.0, (), derive_w4_)
    w5 = ll_make_work("w5", 0.0, (w3, w4), derive_w5_)
    w1_init_data = w1.data
    w2_init_data = w2.data
    w3_init_data = w3.data
    w4_init_data = w4.data
    w5_init_data = w5.data
    w5.calculate()
    return (
        w1_init_data, w2_init_data, w3_init_data, w4_init_data, w5_init_data,
        w1, w2, w3, w4, w5
    )


def test_make_work_ll_sources():
    (
        w1_init_data, w2_init_data, w3_init_data, w4_init_data, w5_init_data,
        w1, w2, w3, w4, w5
    ) = run_test_work_calculate(derive_w1, derive_w2, derive_w3, derive_w4, derive_w5)
    assert w1_init_data == 0
    assert w2_init_data == 0
    assert w3_init_data == 0
    assert w4_init_data == 0
    assert w5_init_data == 0
    assert abs(w1.data - 3.14) < 1e-15
    assert abs(w2.data - 1.41) < 1e-15
    assert abs(w3.data - w1.data * w2.data) < 1e-15
    assert abs(w4.data - 1.72) < 1e-15
    assert abs(w5.data - (w3.data + w4.data)) < 1e-15


def test_make_work_ll_graph():
    @njit
    def aux():
        w1_ = ll_make_work("first", 0.0, (), None)
        w2_ = ll_make_work("second", 0.0, (w1_,), None)
        w3_ = ll_make_work("third", 0.0, (w2_,), None)
        w4_ = ll_make_work("fourth", 0.0, (), None)
        w5_ = ll_make_work("fifth", 0.0, (w3_, w4_), None)
        return w5_

    w = aux()
    assert w.get_inputs_names() == ["third", "fourth"]
    tree_image = make_image(w)
    tree_image_ref = """
fifth--third---second--first
       |
       fourth"""
    assert tree_image == tree_image_ref


if __name__ == "__main__":
    collect_and_run_tests(__name__)
