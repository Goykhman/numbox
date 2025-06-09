from numba import float64, njit, typeof
from numba.core.types.function_type import FunctionType
from numba.core.types.functions import Dispatcher
from numbox.utils.highlevel import cres, determine_field_index
from test.common_structrefs import S1Type


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
