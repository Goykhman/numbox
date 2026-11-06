"""Compare a NumPy calculation over many entities with a jitted loop that builds a ``Work`` graph for each entity.

Each of ``N`` entities is in one of ``M`` internal states. The probabilities of the states evolve over ``H`` time
steps, and each entity has a transition matrix of its own. ``run_numpy`` computes all entities at once, with arrays
over the entities. ``run_numba`` loops over the entities in a cached jitted function, which creates two end nodes and a
derived node with ``make_work`` for each entity and calculates the derived node. Both check that the probabilities at
each step sum to 1. Run the script with ``python test/benchmark_work.py``. It logs the time of each approach and prints
the ratio of the NumPy time to the numba time to stderr.
"""


import logging
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


# *********************************************
# ***** Input data for the NumPy approach *****
# *********************************************


def init_vectorized_states():
    """Return the initial state probabilities as a list of ``M`` arrays over the ``N`` entities.

    Element ``[i][e]`` is the probability that entity ``e`` is in state ``i``, and the probabilities of each entity sum
    to 1.
    """
    states_ = numpy.random.rand(M, N)
    states_ = states_ / states_.sum(axis=0)
    states_ = [states_[i, :] for i in range(M)]
    return states_


def vectorized_transitions():
    """Return the transition probabilities as a list of ``M * M`` lists of ``N`` values, one value per entity.

    Element ``[i + M * j][e]`` is the probability that entity ``e`` moves from state ``j`` at ``t-1`` to state ``i`` at
    ``t``. For each entity and each state ``j``, the probabilities over the states ``i`` sum to 1.
    """
    transitions_ = []
    # Iteration j of the loop appends the M rows of the state j at `t-1`, one row for each state i at `t`.
    for _ in range(M):
        tr_ = numpy.random.rand(M, N)
        tr_ = tr_ / tr_.sum(axis=0)
        transitions_.extend(tr_.tolist())
    return transitions_


# ******************************
# ***** The NumPy approach *****
# ******************************


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
    """Create the input data, run the NumPy approach, and check that the probabilities at each step sum to 1."""
    init_states_ = init_vectorized_states()
    transitions_ = vectorized_transitions()
    fin_states = _run_numpy(init_states_, transitions_)
    assert numpy.allclose(fin_states.sum(axis=1), 1.0), "calculated probabilities are not normalized"


# *********************************************
# ***** Input data for the numba approach *****
# *********************************************


def init_states():
    """Return the initial state probabilities as an array of shape ``(N, M)``, whose rows sum to 1."""
    states_ = numpy.empty((N, M), dtype=numpy.float64)
    for i in range(N):
        init_state_ = numpy.random.rand(M)
        init_state_ = init_state_ / init_state_.sum()
        states_[i, :] = init_state_[:]
    return states_


def transitions():
    """Return the transition matrices as an array of shape ``(N, M, M)``, one matrix for each entity.

    Element ``[e, i, j]`` is the probability that entity ``e`` moves from state ``j`` to state ``i``, so each column
    of a matrix sums to 1.
    """
    transitions_ = numpy.empty((N, M, M), dtype=numpy.float64)
    for i in range(N):
        tr_ = numpy.random.rand(M, M)
        tr_ = tr_ / tr_.sum(axis=0)
        transitions_[i, :, :] = tr_[:, :]
    return transitions_


# ******************************
# ***** The numba approach *****
# ******************************


double_1_ty = numba.types.Array(numba.types.float64, 1, "C")
double_2_ty = numba.types.Array(numba.types.float64, 2, "C")
derive_p_sig = double_2_ty(double_1_ty, double_2_ty)


@cres(derive_p_sig, cache=True)
def derive_p(p0_, tr_):
    """Return the state probabilities of one entity at each of the ``H`` time steps, as an array of shape ``(H, M)``.

    ``p0_`` holds the initial probabilities, and ``tr_`` is the entity's transition matrix.
    """
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
    """Create the input data, run the numba approach, and check that the probabilities at each step sum to 1."""
    init_states_ = init_states()
    transitions_ = transitions()
    fin_states = _run_numba(init_states_, transitions_, derive_p)
    assert numpy.allclose(numpy.array(fin_states).sum(axis=2), 1.0), "calculated probabilities are not normalized"


def benchmark():
    """Run both approaches and print the ratio of the NumPy time to the numba time to stderr."""
    # Timer keeps only the duration of the last call of each function. If the call below is enabled, the first call of
    # _run_numba compiles it or loads it from the cache, and the ratio excludes that time.
    # run_numba()
    run_numpy()
    run_numba()
    rel_factor_ = timer.times["_run_numpy"] / timer.times["_run_numba"]
    print(f"time(numpy) / time(numba) = {rel_factor_}", file=sys.stderr)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    benchmark()
