"""Tests of cached jitted functions that calculate ``Work`` nodes.

A node created from Python can be passed to such a function, and a ``cres`` derive passed as an argument leaves the
function cacheable. A function that reads its ``cres`` derive from the module's globals cannot be cached, and numba
warns about it.
"""
from numba import float64, njit
from numbox.core.configurations import jit_options
from numbox.core.work.work import make_work
from numbox.core.work.work_utils import make_work_helper, make_init_data
from numbox.utils.highlevel import cres
from test.auxiliary_utils import collect_and_run_tests


def derive_w1():
    return 3.14


@njit(cache=True)
def run_1(w_):
    w_.calculate()
    return w_.data


def test_cache_1():
    """A cached jitted function calculates a node that ``make_work_helper`` created from Python."""
    w1 = make_work_helper("w1", make_init_data(), derive_py=derive_w1, jit_options=jit_options)
    assert w1.name == "w1"
    w1_data = run_1(w1)
    assert abs(w1_data - 3.14) < 1e-15


@cres(float64(), **jit_options)
def derive_w2():
    return 3.14


@njit(cache=True)
def run_2():
    """Create a node whose derive is the module global ``derive_w2``, calculate it, and return its value.

    numba cannot cache this function, because the function reads a ``cres`` function from the module's globals. numba
    warns::

        NumbaWarning: Cannot cache compiled function "run_2" as it uses dynamic globals
        (such as ctypes pointers and large global arrays)

    ``run_3`` receives the derive as an argument instead, and numba caches it.
    """
    w2_ = make_work("w2", 0.0, derive=derive_w2)
    w2_.calculate()
    return w2_.data


def test_cache_2():
    """``run_2`` computes the node's value, although numba cannot cache it."""
    w2_data = run_2()
    assert abs(w2_data - 3.14) < 1e-15


@njit(cache=True)
def run_3(derive_):
    w2_ = make_work("w2", 0.0, derive=derive_)
    w2_.calculate()
    return w2_.data


def test_cache_3():
    """``run_3`` receives the ``cres`` derive as an argument and computes the node's value."""
    w2_data = run_3(derive_w2)
    assert abs(w2_data - 3.14) < 1e-15


if __name__ == "__main__":
    collect_and_run_tests(__name__)
