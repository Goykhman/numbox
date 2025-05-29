import numba
import numpy
from numbox.core.work import make_work
from numbox.utils.highlevel import cres_njit


num_of_entities = 3
time_horizon = 5
num_of_inner_states = 3


p0_data_all = [
    numpy.array([0.25, 0.45, 0.3], dtype=numpy.float64),
    numpy.array([0.85, 0.05, 0.1], dtype=numpy.float64),
    numpy.array([0.5, 0.1, 0.4], dtype=numpy.float64),
]


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


double_1_ty = numba.types.Array(numba.types.float64, 1, "C")
double_2_ty = numba.types.Array(numba.types.float64, 2, "C")
derive_p_sig = double_2_ty(double_1_ty, double_2_ty)


@cres_njit(derive_p_sig, cache=True)
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
            print(f"probabilities at time {t} <- {p.data[t, :]}")
            assert abs(p.data[t, :].sum() - 1.0) < 1e-15, "calculated probabilities are still normalized"
