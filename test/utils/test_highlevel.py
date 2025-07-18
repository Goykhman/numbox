from numba import njit, typeof
from numba.core.types import float64
from numba.core.types.function_type import FunctionType
from numba.core.types.functions import Dispatcher
from numpy import isclose

from numbox.utils.highlevel import cres, determine_field_index, make_structref
from test.auxiliary_utils import collect_and_run_tests
from test.common_structrefs import S1Type
from test.utils.common_struct_type_classes import S1TypeClass, S2TypeClass


def test_cres_njit():
    aux_sig = float64(float64)

    @cres(aux_sig, cache=True)
    def aux_1(x):
        return 2 * x

    @njit(aux_sig, cache=True)
    def aux_2(x):
        return 2 * x

    assert abs(aux_1(3.14) - 2 * 3.14) < 1e-15
    assert abs(aux_2(3.14) - 2 * 3.14) < 1e-15

    assert isinstance(typeof(aux_1), FunctionType)
    assert isinstance(typeof(aux_2), Dispatcher)

    @njit
    def run(func):
        return func(3.14)

    assert abs(run(aux_1) - 2 * 3.14) < 1e-15
    assert abs(run(aux_2) - 2 * 3.14) < 1e-15

    assert isinstance(run.nopython_signatures[0].args[0], FunctionType)
    assert isinstance(run.nopython_signatures[1].args[0], Dispatcher)


def test_determine_field_index():
    assert determine_field_index(S1Type, "x1") == 0
    assert determine_field_index(S1Type, "x2") == 1
    assert determine_field_index(S1Type, "x3") == 2


@njit(cache=True)
def aux_test_make_structref(s):
    return s.y


def test_make_structref_1():
    make_s1 = make_structref("S1", ("x", "y"), S1TypeClass)
    s1_1 = make_s1(3.141, 2)
    s1_2 = make_s1(2.17, 3)
    assert isclose(s1_1.x, 3.141)
    assert s1_2.y == 3
    assert aux_test_make_structref(s1_1) == 2
    make_s2 = make_structref("S2", ("x", "y", "z"), S2TypeClass)
    s2_1_z = "hello"
    s2_1 = make_s2(1.41, 45, s2_1_z)
    assert isclose(s2_1.x, 1.41)
    assert s2_1.y == 45
    assert s2_1.z == s2_1_z
    assert aux_test_make_structref(s2_1) == 45


if __name__ == '__main__':
    collect_and_run_tests(__name__)
