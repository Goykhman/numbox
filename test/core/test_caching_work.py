from numba import float64, njit
from numbox.core.configurations import default_jit_options
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
    w1 = make_work_helper("w1", make_init_data(), derive_py=derive_w1, jit_options=default_jit_options)
    assert w1.name == "w1"
    w1_data = run_1(w1)
    assert abs(w1_data - 3.14) < 1e-15


@cres(float64(), **default_jit_options)
def derive_w2():
    return 3.14


@njit(cache=True)
def run_2():
    """
    Will log warning:

    NumbaWarning: Cannot cache compiled function "run_2" as it uses dynamic globals
    (such as ctypes pointers and large global arrays)

    To make it cacheable, pass `cres` derive function `derive_w2` as an argument, see
    `run_3` and `test_cache_3`
    """
    w2_ = make_work("w2", 0.0, derive=derive_w2)
    w2_.calculate()
    return w2_.data


def test_cache_2():
    w2_data = run_2()
    assert abs(w2_data - 3.14) < 1e-15


@njit(cache=True)
def run_3(derive_):
    w2_ = make_work("w2", 0.0, derive=derive_)
    w2_.calculate()
    return w2_.data


def test_cache_3():
    w2_data = run_3(derive_w2)
    assert abs(w2_data - 3.14) < 1e-15


if __name__ == "__main__":
    collect_and_run_tests(__name__)
