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
# ***** get numpy data *****
# **************************


def init_vectorized_states():
    states_ = numpy.random.rand(M, N)
    states_ = states_ / states_.sum(axis=0)  # normalize probabilities
    states_ = [states_[i, :] for i in range(M)]
    # For instance, states[2][4] is the probability that entity 4 is in the state 2
    return states_


def vectorized_transitions():
    transitions_ = []
    # for each possible state at `t-1`...
    for _ in range(M):
        # ...create M probabilities of states at `t`
        tr_ = numpy.random.rand(M, N)
        tr_ = tr_ / tr_.sum(axis=0)  # normalize the total of probabilities to every possible state at `t`
        transitions_.extend(tr_.tolist())
    return transitions_


# **************************
# ***** numpy approach *****
# **************************


@timer
def _run_numpy(init_states_, transitions_):
    fin_states = numpy.empty((H, M, N), dtype=numpy.float64)
    for i in range(M):
        fin_states[0, i, :] = init_states_[i]
    states_ = init_states_
    for t in range(1, H):
        updated_states_ = [numpy.zeros(N, dtype=numpy.float64) for _ in range(M)]
        for i in range(M):
            for j in range(M):
                updated_states_[i] += transitions_[i + M * j] * states_[j]
        for i in range(M):
            fin_states[t, i, :] = updated_states_[i]
        states_ = updated_states_
    return fin_states


def run_numpy():
    init_states_ = init_vectorized_states()
    transitions_ = vectorized_transitions()
    fin_states = _run_numpy(init_states_, transitions_)
    assert numpy.allclose(fin_states.sum(axis=1), 1.0), "calculated probabilities are not normalized"


# **************************
# ***** get numba data *****
# **************************


def init_states():
    states_ = numpy.empty((N, M), dtype=numpy.float64)
    for i in range(N):
        init_state_ = numpy.random.rand(M)
        init_state_ = init_state_ / init_state_.sum()
        states_[i, :] = init_state_[:]
    return states_


def transitions():
    transitions_ = numpy.empty((N, M, M), dtype=numpy.float64)
    for i in range(N):
        tr_ = numpy.random.rand(M, M)
        tr_ = tr_ / tr_.sum(axis=0)
        transitions_[i, :, :] = tr_[:, :]
    return transitions_


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


@timer
@numba.njit(cache=True)
def _run_numba(init_states_, transitions_, derive_p_):
    fin_states = numpy.empty((N, H, M), dtype=numpy.float64)
    for entity_id in range(N):
        p0 = make_work(f"p0_{entity_id}", init_states_[entity_id])
        tr = make_work(f"tr_{entity_id}", transitions_[entity_id])
        p_data = numpy.zeros(shape=(H, M), dtype=numpy.float64)
        p = make_work(f"p_{entity_id}", p_data, sources=(p0, tr), derive=derive_p_)
        p.calculate()
        fin_states[entity_id, :, :] = p.data
    return fin_states


def run_numba():
    init_states_ = init_states()
    transitions_ = transitions()
    fin_states = _run_numba(init_states_, transitions_, derive_p)
    assert numpy.allclose(numpy.array(fin_states).sum(axis=2), 1.0), "calculated probabilities are not normalized"


def benchmark():
    # run_numba()
    run_numpy()
    run_numba()
    rel_factor_ = timer.times["_run_numpy"] / timer.times["_run_numba"]
    print(f"time(numpy) / time(numba) = {rel_factor_}", file=sys.stderr)


if __name__ == "__main__":
    benchmark()
