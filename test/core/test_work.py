import numpy
import pytest
from numba import njit
from numba.core.types import Array, float64, int64

from numbox.core.work.work import make_work
from numbox.utils.highlevel import cres
from test.auxiliary_utils import collect_and_run_tests


def test_make_work():
    @cres(Array(float64, 2, "C")(Array(float64, 2, "C")))
    def derive_p2(data_):
        ret = numpy.ones((100, 100), dtype=numpy.float64)
        return ret

    for _ in range(1_000):
        p1 = make_work("p", numpy.zeros((100, 100), dtype=numpy.float64))
        p1.calculate()
        p2 = make_work("p", numpy.zeros((100, 100), dtype=numpy.float64), sources=(p1,), derive=derive_p2)
        p2.calculate()
        assert abs(p2.data[1][2] - 1.0) < 1e-15


num_of_entities = 3
time_horizon = 5
num_of_inner_states = 3


p0_data_all = numpy.array([
    [0.25, 0.45, 0.3],
    [0.85, 0.05, 0.1],
    [0.5, 0.1, 0.4],
], dtype=numpy.float64)


transitions = [
    numpy.array([
        [0.05, 0.65, 0.25],
        [0.8, 0.1, 0.5],
        [0.15, 0.25, 0.25]
    ]),
    numpy.array([
        [0.75, 0.2, 0.8],
        [0.05, 0.75, 0.1],
        [0.2, 0.05, 0.1]
    ]),
    numpy.array([
        [0.45, 0.3, 0.55],
        [0.45, 0.45, 0.3],
        [0.1, 0.25, 0.15]
    ]),
]


p_ref_data_all = numpy.array([
    [
        [0.25, 0.45, 0.3],
        [0.38, 0.395, 0.225],
        [0.332, 0.456, 0.212],
        [0.366, 0.4172, 0.2168],
        [0.34368, 0.44292, 0.2134]
    ],
    [
        [0.85, 0.05, 0.1],
        [0.7275, 0.09, 0.1825],
        [0.709625, 0.122125, 0.16825],
        [0.69124375, 0.1439, 0.16485625],
        [0.67909781, 0.15897281, 0.16192938]
    ],
    [
        [0.5, 0.1, 0.4],
        [0.475, 0.39, 0.135],
        [0.405, 0.42975, 0.16525],
        [0.4020625, 0.4252125, 0.172725],
        [0.40349063, 0.42409125, 0.17241813],
    ]
], dtype=numpy.float64)


double_1_ty = Array(float64, 1, "C")
double_2_ty = Array(float64, 2, "C")
derive_p_sig = double_2_ty(double_1_ty, double_2_ty)


@cres(derive_p_sig, cache=True)
def derive_p(p0_, tr_):
    p_ = numpy.zeros(shape=(time_horizon, num_of_inner_states), dtype=numpy.float64)
    p_[0, :] = p0_
    for t in range(1, time_horizon):
        for i in range(num_of_inner_states):
            for j in range(num_of_inner_states):
                p_[t][i] += tr_[i][j] * p_[t - 1][j]
    return p_


def test_work():
    for transition in transitions:
        assert numpy.allclose(transition.sum(axis=0), 1.0), """complete set of states at `t`
conditional on state at `t-1`"""
    for entity_id in range(num_of_entities):
        p0_data = p0_data_all[entity_id]
        assert abs(p0_data.sum() - 1.0) == 0, "complete set of initial states"
        p0 = make_work("p0", p0_data)

        tr_data = transitions[entity_id]
        tr = make_work("tr", tr_data)

        p_data = numpy.zeros(shape=(time_horizon, num_of_inner_states), dtype=numpy.float64)
        p = make_work("p", p_data, sources=(p0, tr), derive=derive_p)
        p.calculate()

        print(f"\nentity {entity_id}:")
        for t in range(time_horizon):
            ref_data = p_ref_data_all[entity_id, t, :]
            assert numpy.allclose(p.data[t, :], ref_data)
            print(f"probabilities at time {t} <- {p.data[t, :]}")
            assert abs(p.data[t, :].sum() - 1.0) < 1e-15, "calculated probabilities are still normalized"


def test_work_sources():
    w1 = make_work("w1", 3.14)
    w2 = make_work("w2", 2)

    @cres(float64(float64, int64))
    def derive_w3(w1_, w2_):
        return w1_ + w2_
    w3 = make_work("w3", 0.0, sources=(w1, w2), derive=derive_w3)
    assert w3.get_inputs_names() == ["w1", "w2"]

    @cres(float64(float64))
    def derive_w4(w3_):
        return 137 * w3_
    w4 = make_work("w4", 0.0, sources=(w3,), derive=derive_w4)
    assert w4.get_inputs_names() == ["w3"]


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


def run_test_work_calculate(derive_w1_, derive_w2_, derive_w3_, derive_w4_, derive_w5_):
    w1 = make_work("w1", 0.0, derive=derive_w1_)
    w2 = make_work("w2", 0.0, derive=derive_w2_)
    w3 = make_work("w3", 0.0, sources=(w1, w2), derive=derive_w3_)
    w4 = make_work("w4", 0.0, derive=derive_w4_)
    w5 = make_work("w5", 0.0, sources=(w3, w4), derive=derive_w5_)
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


def test_work_calculate():
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


def test_work_calculate_jitted():
    (
        w1_init_data, w2_init_data, w3_init_data, w4_init_data, w5_init_data,
        w1, w2, w3, w4, w5
    ) = njit(cache=True)(run_test_work_calculate)(derive_w1, derive_w2, derive_w3, derive_w4, derive_w5)
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


def test_bad_signature_make_work():
    with pytest.raises(ValueError) as e:
        _ = make_work("", 0.0, sources=(), derive=derive_w3)
    assert str(e.value) == """Signatures do not match, derive defines (float64, float64) -> float64
but data and sources imply () -> float64"""


if __name__ == "__main__":
    collect_and_run_tests(__name__)
