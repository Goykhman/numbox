""" Benchmarks vectorized (numpy) calculation over entities with internal state
evolving in time against compiled (numba) loop over individual entities """


import numba
import numpy
import sys

from numbox.core.work.work import make_work
from numbox.utils.highlevel import cres
from numbox.utils.timer import timer


# number of entities
N = 1_000

# number of time steps
H = 120

# number of internal states
M = 20


# **************************
# ***** numpy approach *****
# **************************


def calculate_numpy(states_, transitions_):
    for t in range(1, H):
        updated_states_ = [numpy.zeros(N, dtype=numpy.float64) for _ in range(M)]
        for i in range(M):
            for j in range(M):
                updated_states_[i] += transitions_[i + M * j] * states_[j]
        states_ = updated_states_
    return states_


@timer
def run_numpy():

    states = numpy.random.rand(M, N)
    states = states / states.sum(axis=0)  # normalize probabilities
    states = [states[i, :] for i in range(M)]
    # For instance, states[2][4] is the probability that entity 4 is in the state 2

    transitions = []
    # for each possible state at `t-1`...
    for _ in range(M):
        # ...create M probabilities of states at `t`
        tr_ = numpy.random.rand(M, N)
        tr_ = tr_ / tr_.sum(axis=0)  # normalize the total of probabilities to every possible state at `t`
        transitions.extend(tr_.tolist())

    assert len(transitions) == M ** 2, "each transition probability is N-vectorized across all entities"

    # uncomment these when not doing benchmarking
    # init_state = states[0][0]

    states = calculate_numpy(states, transitions)

    # fin_state = states[0][0]
    # assert abs(init_state - fin_state) > 1e-8, "some calculation indeed happened"
    # assert numpy.allclose(numpy.array(states).sum(axis=0), 1.0), "probabilities are still normalized"


# **************************
# ***** numba approach *****
# **************************


double_1_ty = numba.types.Array(numba.types.float64, 1, "C")
double_2_ty = numba.types.Array(numba.types.float64, 2, "C")
derive_p_sig = double_2_ty(double_1_ty, double_2_ty)


@cres(derive_p_sig, cache=True)
def derive_p(p0_, tr_):
    p_ = numpy.zeros(shape=(H, M), dtype=numpy.float64)
    p_[0, :] = p0_
    for t in range(1, H):
        for i in range(M):
            for j in range(M):
                p_[t][i] += tr_[i][j] * p_[t - 1][j]
    return p_


@numba.njit(cache=True)
def calculate_numba(p):
    p.calculate()


@timer
def run_numba():
    for entity_id in range(N):
        p0_data = numpy.random.rand(M)
        p0_data = p0_data / p0_data.sum()
        p0 = make_work("p0", p0_data)

        # uncomment these when not doing benchmarking
        # p0_data_0_init = p0_data[0]

        # row of `tr_data` is to-state, col is from-state
        tr_data = numpy.random.rand(M, M)
        # given from-state, normalize probabilities to all to-states
        tr_data = tr_data / tr_data.sum(axis=0)
        tr = make_work("tr", tr_data)

        p_data = numpy.zeros(shape=(H, M), dtype=numpy.float64)
        p = make_work("p", p_data, sources=(p0, tr), derive=derive_p)
        calculate_numba(p)

        # initial data is just stored in its column
        # assert numpy.allclose(p.data[0], p0_data)
        # p_data_0_fin = p.data[-1][0]
        # assert abs(p_data_0_fin - p0_data_0_init) > 1e-8, "some kind of calculation happened"
        # assert abs(p.data[-1].sum() - 1.0) < 1e-14, f"final probabilities are still normalized"


def benchmark():
    run_numba()  # first run, to cache compiled functions
    run_numba()
    run_numpy()
    rel_factor_ = timer.times["run_numpy"] / timer.times["run_numba"]
    print(f"time(numpy) / time(numba) = {rel_factor_}", file=sys.stderr)


if __name__ == "__main__":
    benchmark()
