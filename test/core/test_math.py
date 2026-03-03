import math
import numpy as np
from numba import njit

from numbox.core.bindings._math import (
    cos, sin, tan,
    acos, asin, atan,
    cosh, sinh, tanh, acosh, asinh, atanh,
    exp, exp2, expm1,
    log, log2, log10, log1p, logb,
    sqrt, cbrt,
    ceil, floor, trunc, round, rint, nearbyint,
    erf, erfc, lgamma, tgamma,
    fabs,
    atan2,
    pow, fmod, remainder,
    hypot,
    fmax, fmin, fdim,
    copysign,
)
from test.auxiliary_utils import collect_and_run_tests


INF = float("inf")
NAN = float("nan")


def assert_close(actual, expected):
    if np.isnan(expected):
        assert np.isnan(actual), f"expected NaN, got {actual}"
    elif np.isinf(expected):
        assert actual == expected, f"expected {expected}, got {actual}"
    else:
        assert np.isclose(actual, expected), \
            f"expected {expected}, got {actual}"


# --- Trig ---

class TestTrig:
    @staticmethod
    def test_sin_cos_tan_identity():
        x = 3.1415
        assert np.isclose(sin(x) / cos(x), tan(x))

    @staticmethod
    def test_cos():
        for x in [0.0, 1.0, -1.0, math.pi, math.pi / 2]:
            assert_close(cos(x), math.cos(x))

    @staticmethod
    def test_sin():
        for x in [0.0, 1.0, -1.0, math.pi, math.pi / 2]:
            assert_close(sin(x), math.sin(x))

    @staticmethod
    def test_tan():
        for x in [0.0, 1.0, -1.0, math.pi / 4]:
            assert_close(tan(x), math.tan(x))

    @staticmethod
    def test_nan():
        assert np.isnan(cos(NAN))
        assert np.isnan(sin(NAN))
        assert np.isnan(tan(NAN))

    @staticmethod
    def test_inf():
        assert np.isnan(cos(INF))
        assert np.isnan(cos(-INF))
        assert np.isnan(sin(INF))
        assert np.isnan(sin(-INF))
        assert np.isnan(tan(INF))
        assert np.isnan(tan(-INF))

    @staticmethod
    def test_jit():
        @njit
        def _cos(x):
            return cos(x)

        @njit
        def _sin(x):
            return sin(x)

        @njit
        def _tan(x):
            return tan(x)

        x = 3.1415
        assert np.isclose(_sin(x) / _cos(x), _tan(x))
        for x in [0.0, 1.0, -1.0, math.pi, math.pi / 2]:
            assert_close(_cos(x), math.cos(x))
        for x in [0.0, 1.0, -1.0, math.pi, math.pi / 2]:
            assert_close(_sin(x), math.sin(x))
        for x in [0.0, 1.0, -1.0, math.pi / 4]:
            assert_close(_tan(x), math.tan(x))
        assert np.isnan(_cos(NAN))
        assert np.isnan(_sin(NAN))
        assert np.isnan(_tan(NAN))
        assert np.isnan(_cos(INF))
        assert np.isnan(_sin(INF))
        assert np.isnan(_tan(INF))


# --- Inverse trig ---

class TestAcos:
    @staticmethod
    def test_typical():
        for x in [0.0, 0.5, -0.5, 1.0, -1.0]:
            assert_close(acos(x), math.acos(x))

    @staticmethod
    def test_nan():
        assert np.isnan(acos(NAN))

    @staticmethod
    def test_out_of_domain():
        assert np.isnan(acos(2.0))
        assert np.isnan(acos(-2.0))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return acos(x)

        for x in [0.0, 0.5, -0.5, 1.0, -1.0]:
            assert_close(_jit(x), math.acos(x))
        assert np.isnan(_jit(NAN))
        assert np.isnan(_jit(2.0))
        assert np.isnan(_jit(-2.0))


class TestAsin:
    @staticmethod
    def test_typical():
        for x in [0.0, 0.5, -0.5, 1.0, -1.0]:
            assert_close(asin(x), math.asin(x))

    @staticmethod
    def test_nan():
        assert np.isnan(asin(NAN))

    @staticmethod
    def test_out_of_domain():
        assert np.isnan(asin(2.0))
        assert np.isnan(asin(-2.0))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return asin(x)

        for x in [0.0, 0.5, -0.5, 1.0, -1.0]:
            assert_close(_jit(x), math.asin(x))
        assert np.isnan(_jit(NAN))
        assert np.isnan(_jit(2.0))
        assert np.isnan(_jit(-2.0))


class TestAtan:
    @staticmethod
    def test_typical():
        for x in [0.0, 1.0, -1.0, 1e-300, 1e300]:
            assert_close(atan(x), math.atan(x))

    @staticmethod
    def test_inf():
        assert_close(atan(INF), math.pi / 2)
        assert_close(atan(-INF), -math.pi / 2)

    @staticmethod
    def test_nan():
        assert np.isnan(atan(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return atan(x)

        for x in [0.0, 1.0, -1.0, 1e-300, 1e300]:
            assert_close(_jit(x), math.atan(x))
        assert_close(_jit(INF), math.pi / 2)
        assert_close(_jit(-INF), -math.pi / 2)
        assert np.isnan(_jit(NAN))


# --- Hyperbolic ---

class TestCosh:
    @staticmethod
    def test_typical():
        for x in [0.0, 1.0, -1.0, 0.5]:
            assert_close(cosh(x), math.cosh(x))

    @staticmethod
    def test_inf():
        assert cosh(INF) == INF
        assert cosh(-INF) == INF

    @staticmethod
    def test_nan():
        assert np.isnan(cosh(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return cosh(x)

        for x in [0.0, 1.0, -1.0, 0.5]:
            assert_close(_jit(x), math.cosh(x))
        assert _jit(INF) == INF
        assert _jit(-INF) == INF
        assert np.isnan(_jit(NAN))


class TestSinh:
    @staticmethod
    def test_typical():
        for x in [0.0, 1.0, -1.0, 0.5]:
            assert_close(sinh(x), math.sinh(x))

    @staticmethod
    def test_inf():
        assert sinh(INF) == INF
        assert sinh(-INF) == -INF

    @staticmethod
    def test_nan():
        assert np.isnan(sinh(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return sinh(x)

        for x in [0.0, 1.0, -1.0, 0.5]:
            assert_close(_jit(x), math.sinh(x))
        assert _jit(INF) == INF
        assert _jit(-INF) == -INF
        assert np.isnan(_jit(NAN))


class TestTanh:
    @staticmethod
    def test_typical():
        for x in [0.0, 1.0, -1.0, 0.5]:
            assert_close(tanh(x), math.tanh(x))

    @staticmethod
    def test_inf():
        assert_close(tanh(INF), 1.0)
        assert_close(tanh(-INF), -1.0)

    @staticmethod
    def test_nan():
        assert np.isnan(tanh(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return tanh(x)

        for x in [0.0, 1.0, -1.0, 0.5]:
            assert_close(_jit(x), math.tanh(x))
        assert_close(_jit(INF), 1.0)
        assert_close(_jit(-INF), -1.0)
        assert np.isnan(_jit(NAN))


class TestAcosh:
    @staticmethod
    def test_typical():
        for x in [1.0, 2.0, 10.0, 1e300]:
            assert_close(acosh(x), math.acosh(x))

    @staticmethod
    def test_inf():
        assert acosh(INF) == INF

    @staticmethod
    def test_out_of_domain():
        assert np.isnan(acosh(0.5))

    @staticmethod
    def test_nan():
        assert np.isnan(acosh(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return acosh(x)

        for x in [1.0, 2.0, 10.0, 1e300]:
            assert_close(_jit(x), math.acosh(x))
        assert _jit(INF) == INF
        assert np.isnan(_jit(0.5))
        assert np.isnan(_jit(NAN))


class TestAsinh:
    @staticmethod
    def test_typical():
        for x in [0.0, 1.0, -1.0, 1e-300, 1e300]:
            assert_close(asinh(x), math.asinh(x))

    @staticmethod
    def test_inf():
        assert asinh(INF) == INF
        assert asinh(-INF) == -INF

    @staticmethod
    def test_nan():
        assert np.isnan(asinh(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return asinh(x)

        for x in [0.0, 1.0, -1.0, 1e-300, 1e300]:
            assert_close(_jit(x), math.asinh(x))
        assert _jit(INF) == INF
        assert _jit(-INF) == -INF
        assert np.isnan(_jit(NAN))


class TestAtanh:
    @staticmethod
    def test_typical():
        for x in [0.0, 0.5, -0.5, 1e-300]:
            assert_close(atanh(x), math.atanh(x))

    @staticmethod
    def test_boundary():
        assert atanh(1.0) == INF
        assert atanh(-1.0) == -INF

    @staticmethod
    def test_out_of_domain():
        assert np.isnan(atanh(2.0))

    @staticmethod
    def test_nan():
        assert np.isnan(atanh(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return atanh(x)

        for x in [0.0, 0.5, -0.5, 1e-300]:
            assert_close(_jit(x), math.atanh(x))
        assert _jit(1.0) == INF
        assert _jit(-1.0) == -INF
        assert np.isnan(_jit(2.0))
        assert np.isnan(_jit(NAN))


# --- Exponential/log ---

class TestExp:
    @staticmethod
    def test_typical():
        for x in [0.0, 1.0, -1.0, 2.0, 1e-300]:
            assert_close(exp(x), math.exp(x))

    @staticmethod
    def test_inf():
        assert exp(INF) == INF
        assert exp(-INF) == 0.0

    @staticmethod
    def test_nan():
        assert np.isnan(exp(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return exp(x)

        for x in [0.0, 1.0, -1.0, 2.0, 1e-300]:
            assert_close(_jit(x), math.exp(x))
        assert _jit(INF) == INF
        assert _jit(-INF) == 0.0
        assert np.isnan(_jit(NAN))


class TestExp2:
    @staticmethod
    def test_typical():
        assert_close(exp2(0.0), 1.0)
        assert_close(exp2(1.0), 2.0)
        assert_close(exp2(10.0), 1024.0)
        assert_close(exp2(-1.0), 0.5)

    @staticmethod
    def test_inf():
        assert exp2(INF) == INF
        assert exp2(-INF) == 0.0

    @staticmethod
    def test_nan():
        assert np.isnan(exp2(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return exp2(x)

        assert_close(_jit(0.0), 1.0)
        assert_close(_jit(1.0), 2.0)
        assert_close(_jit(10.0), 1024.0)
        assert_close(_jit(-1.0), 0.5)
        assert _jit(INF) == INF
        assert _jit(-INF) == 0.0
        assert np.isnan(_jit(NAN))


class TestExpm1:
    @staticmethod
    def test_typical():
        for x in [0.0, 1.0, -1.0, 1e-300]:
            assert_close(expm1(x), math.expm1(x))

    @staticmethod
    def test_inf():
        assert expm1(INF) == INF
        assert expm1(-INF) == -1.0

    @staticmethod
    def test_nan():
        assert np.isnan(expm1(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return expm1(x)

        for x in [0.0, 1.0, -1.0, 1e-300]:
            assert_close(_jit(x), math.expm1(x))
        assert _jit(INF) == INF
        assert _jit(-INF) == -1.0
        assert np.isnan(_jit(NAN))


class TestLog:
    @staticmethod
    def test_typical():
        for x in [1.0, 2.0, math.e, 10.0, 1e-300, 1e300]:
            assert_close(log(x), math.log(x))

    @staticmethod
    def test_zero():
        assert log(0.0) == -INF

    @staticmethod
    def test_negative():
        assert np.isnan(log(-1.0))

    @staticmethod
    def test_inf():
        assert log(INF) == INF

    @staticmethod
    def test_nan():
        assert np.isnan(log(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return log(x)

        for x in [1.0, 2.0, math.e, 10.0, 1e-300, 1e300]:
            assert_close(_jit(x), math.log(x))
        assert _jit(0.0) == -INF
        assert np.isnan(_jit(-1.0))
        assert _jit(INF) == INF
        assert np.isnan(_jit(NAN))


class TestLog2:
    @staticmethod
    def test_typical():
        assert_close(log2(1.0), 0.0)
        assert_close(log2(2.0), 1.0)
        assert_close(log2(1024.0), 10.0)

    @staticmethod
    def test_zero():
        assert log2(0.0) == -INF

    @staticmethod
    def test_negative():
        assert np.isnan(log2(-1.0))

    @staticmethod
    def test_inf():
        assert log2(INF) == INF

    @staticmethod
    def test_nan():
        assert np.isnan(log2(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return log2(x)

        assert_close(_jit(1.0), 0.0)
        assert_close(_jit(2.0), 1.0)
        assert_close(_jit(1024.0), 10.0)
        assert _jit(0.0) == -INF
        assert np.isnan(_jit(-1.0))
        assert _jit(INF) == INF
        assert np.isnan(_jit(NAN))


class TestLog10:
    @staticmethod
    def test_typical():
        assert_close(log10(1.0), 0.0)
        assert_close(log10(10.0), 1.0)
        assert_close(log10(1000.0), 3.0)

    @staticmethod
    def test_zero():
        assert log10(0.0) == -INF

    @staticmethod
    def test_negative():
        assert np.isnan(log10(-1.0))

    @staticmethod
    def test_inf():
        assert log10(INF) == INF

    @staticmethod
    def test_nan():
        assert np.isnan(log10(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return log10(x)

        assert_close(_jit(1.0), 0.0)
        assert_close(_jit(10.0), 1.0)
        assert_close(_jit(1000.0), 3.0)
        assert _jit(0.0) == -INF
        assert np.isnan(_jit(-1.0))
        assert _jit(INF) == INF
        assert np.isnan(_jit(NAN))


class TestLog1p:
    @staticmethod
    def test_typical():
        for x in [0.0, 1.0, 1e-300, -0.5]:
            assert_close(log1p(x), math.log1p(x))

    @staticmethod
    def test_minus_one():
        assert log1p(-1.0) == -INF

    @staticmethod
    def test_inf():
        assert log1p(INF) == INF

    @staticmethod
    def test_nan():
        assert np.isnan(log1p(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return log1p(x)

        for x in [0.0, 1.0, 1e-300, -0.5]:
            assert_close(_jit(x), math.log1p(x))
        assert _jit(-1.0) == -INF
        assert _jit(INF) == INF
        assert np.isnan(_jit(NAN))


class TestLogb:
    @staticmethod
    def test_typical():
        assert_close(logb(1.0), 0.0)
        assert_close(logb(2.0), 1.0)
        assert_close(logb(8.0), 3.0)
        assert_close(logb(0.5), -1.0)

    @staticmethod
    def test_negative():
        assert_close(logb(-2.0), 1.0)
        assert_close(logb(-8.0), 3.0)

    @staticmethod
    def test_zero():
        assert logb(0.0) == -INF

    @staticmethod
    def test_inf():
        assert logb(INF) == INF

    @staticmethod
    def test_nan():
        assert np.isnan(logb(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return logb(x)

        assert_close(_jit(1.0), 0.0)
        assert_close(_jit(2.0), 1.0)
        assert_close(_jit(8.0), 3.0)
        assert_close(_jit(0.5), -1.0)
        assert_close(_jit(-2.0), 1.0)
        assert_close(_jit(-8.0), 3.0)
        assert _jit(0.0) == -INF
        assert _jit(INF) == INF
        assert np.isnan(_jit(NAN))


# --- Power/root ---

class TestSqrt:
    @staticmethod
    def test_typical():
        for x in [0.0, 1.0, 4.0, 2.0, 1e300]:
            assert_close(sqrt(x), math.sqrt(x))

    @staticmethod
    def test_negative():
        assert np.isnan(sqrt(-1.0))

    @staticmethod
    def test_inf():
        assert sqrt(INF) == INF

    @staticmethod
    def test_nan():
        assert np.isnan(sqrt(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return sqrt(x)

        for x in [0.0, 1.0, 4.0, 2.0, 1e300]:
            assert_close(_jit(x), math.sqrt(x))
        assert np.isnan(_jit(-1.0))
        assert _jit(INF) == INF
        assert np.isnan(_jit(NAN))


class TestCbrt:
    @staticmethod
    def test_typical():
        assert_close(cbrt(0.0), 0.0)
        assert_close(cbrt(1.0), 1.0)
        assert_close(cbrt(8.0), 2.0)
        assert_close(cbrt(27.0), 3.0)
        assert_close(cbrt(-8.0), -2.0)

    @staticmethod
    def test_inf():
        assert cbrt(INF) == INF
        assert cbrt(-INF) == -INF

    @staticmethod
    def test_nan():
        assert np.isnan(cbrt(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return cbrt(x)

        assert_close(_jit(0.0), 0.0)
        assert_close(_jit(1.0), 1.0)
        assert_close(_jit(8.0), 2.0)
        assert_close(_jit(27.0), 3.0)
        assert_close(_jit(-8.0), -2.0)
        assert _jit(INF) == INF
        assert _jit(-INF) == -INF
        assert np.isnan(_jit(NAN))


# --- Rounding ---

class TestCeil:
    @staticmethod
    def test_typical():
        assert ceil(2.3) == 3.0
        assert ceil(-2.3) == -2.0
        assert ceil(0.0) == 0.0
        assert ceil(2.0) == 2.0

    @staticmethod
    def test_inf():
        assert ceil(INF) == INF
        assert ceil(-INF) == -INF

    @staticmethod
    def test_nan():
        assert np.isnan(ceil(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return ceil(x)

        assert _jit(2.3) == 3.0
        assert _jit(-2.3) == -2.0
        assert _jit(0.0) == 0.0
        assert _jit(2.0) == 2.0
        assert _jit(INF) == INF
        assert _jit(-INF) == -INF
        assert np.isnan(_jit(NAN))


class TestFloor:
    @staticmethod
    def test_typical():
        assert floor(2.7) == 2.0
        assert floor(-2.7) == -3.0
        assert floor(0.0) == 0.0
        assert floor(2.0) == 2.0

    @staticmethod
    def test_inf():
        assert floor(INF) == INF
        assert floor(-INF) == -INF

    @staticmethod
    def test_nan():
        assert np.isnan(floor(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return floor(x)

        assert _jit(2.7) == 2.0
        assert _jit(-2.7) == -3.0
        assert _jit(0.0) == 0.0
        assert _jit(2.0) == 2.0
        assert _jit(INF) == INF
        assert _jit(-INF) == -INF
        assert np.isnan(_jit(NAN))


class TestTrunc:
    @staticmethod
    def test_typical():
        assert trunc(2.7) == 2.0
        assert trunc(-2.7) == -2.0
        assert trunc(0.0) == 0.0

    @staticmethod
    def test_inf():
        assert trunc(INF) == INF
        assert trunc(-INF) == -INF

    @staticmethod
    def test_nan():
        assert np.isnan(trunc(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return trunc(x)

        assert _jit(2.7) == 2.0
        assert _jit(-2.7) == -2.0
        assert _jit(0.0) == 0.0
        assert _jit(INF) == INF
        assert _jit(-INF) == -INF
        assert np.isnan(_jit(NAN))


class TestRound:
    @staticmethod
    def test_typical():
        assert round(2.3) == 2.0
        assert round(2.5) == 3.0  # C round: ties away from zero
        assert round(-2.5) == -3.0
        assert round(0.0) == 0.0

    @staticmethod
    def test_inf():
        assert round(INF) == INF
        assert round(-INF) == -INF

    @staticmethod
    def test_nan():
        assert np.isnan(round(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return round(x)

        assert _jit(2.3) == 2.0
        assert _jit(2.5) == 3.0  # C round: ties away from zero
        assert _jit(-2.5) == -3.0
        assert _jit(0.0) == 0.0
        assert _jit(INF) == INF
        assert _jit(-INF) == -INF
        assert np.isnan(_jit(NAN))


class TestRint:
    @staticmethod
    def test_typical():
        assert rint(2.3) == 2.0
        assert rint(2.7) == 3.0
        assert rint(0.0) == 0.0

    @staticmethod
    def test_halfway():
        assert rint(2.5) == 2.0  # ties to even
        assert rint(3.5) == 4.0  # ties to even

    @staticmethod
    def test_inf():
        assert rint(INF) == INF
        assert rint(-INF) == -INF

    @staticmethod
    def test_nan():
        assert np.isnan(rint(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return rint(x)

        assert _jit(2.3) == 2.0
        assert _jit(2.7) == 3.0
        assert _jit(0.0) == 0.0
        assert _jit(2.5) == 2.0  # ties to even
        assert _jit(3.5) == 4.0  # ties to even
        assert _jit(INF) == INF
        assert _jit(-INF) == -INF
        assert np.isnan(_jit(NAN))


class TestNearbyint:
    @staticmethod
    def test_typical():
        assert nearbyint(2.3) == 2.0
        assert nearbyint(2.7) == 3.0
        assert nearbyint(0.0) == 0.0

    @staticmethod
    def test_halfway():
        assert nearbyint(2.5) == 2.0  # ties to even
        assert nearbyint(3.5) == 4.0  # ties to even

    @staticmethod
    def test_inf():
        assert nearbyint(INF) == INF
        assert nearbyint(-INF) == -INF

    @staticmethod
    def test_nan():
        assert np.isnan(nearbyint(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return nearbyint(x)

        assert _jit(2.3) == 2.0
        assert _jit(2.7) == 3.0
        assert _jit(0.0) == 0.0
        assert _jit(2.5) == 2.0  # ties to even
        assert _jit(3.5) == 4.0  # ties to even
        assert _jit(INF) == INF
        assert _jit(-INF) == -INF
        assert np.isnan(_jit(NAN))


# --- Error/gamma ---

class TestErf:
    @staticmethod
    def test_typical():
        for x in [0.0, 1.0, -1.0, 0.5, 2.0]:
            assert_close(erf(x), math.erf(x))

    @staticmethod
    def test_inf():
        assert_close(erf(INF), 1.0)
        assert_close(erf(-INF), -1.0)

    @staticmethod
    def test_nan():
        assert np.isnan(erf(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return erf(x)

        for x in [0.0, 1.0, -1.0, 0.5, 2.0]:
            assert_close(_jit(x), math.erf(x))
        assert_close(_jit(INF), 1.0)
        assert_close(_jit(-INF), -1.0)
        assert np.isnan(_jit(NAN))


class TestErfc:
    @staticmethod
    def test_typical():
        for x in [0.0, 1.0, -1.0, 0.5, 2.0]:
            assert_close(erfc(x), math.erfc(x))

    @staticmethod
    def test_inf():
        assert_close(erfc(INF), 0.0)
        assert_close(erfc(-INF), 2.0)

    @staticmethod
    def test_nan():
        assert np.isnan(erfc(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return erfc(x)

        for x in [0.0, 1.0, -1.0, 0.5, 2.0]:
            assert_close(_jit(x), math.erfc(x))
        assert_close(_jit(INF), 0.0)
        assert_close(_jit(-INF), 2.0)
        assert np.isnan(_jit(NAN))


class TestLgamma:
    @staticmethod
    def test_typical():
        for x in [1.0, 2.0, 0.5, 10.0]:
            assert_close(lgamma(x), math.lgamma(x))

    @staticmethod
    def test_inf():
        assert lgamma(INF) == INF

    @staticmethod
    def test_nan():
        assert np.isnan(lgamma(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return lgamma(x)

        for x in [1.0, 2.0, 0.5, 10.0]:
            assert_close(_jit(x), math.lgamma(x))
        assert _jit(INF) == INF
        assert np.isnan(_jit(NAN))


class TestTgamma:
    @staticmethod
    def test_typical():
        for x in [1.0, 2.0, 0.5, 5.0]:
            assert_close(tgamma(x), math.gamma(x))

    @staticmethod
    def test_inf():
        assert tgamma(INF) == INF

    @staticmethod
    def test_zero():
        assert np.isinf(tgamma(0.0))

    @staticmethod
    def test_nan():
        assert np.isnan(tgamma(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return tgamma(x)

        for x in [1.0, 2.0, 0.5, 5.0]:
            assert_close(_jit(x), math.gamma(x))
        assert _jit(INF) == INF
        assert np.isinf(_jit(0.0))
        assert np.isnan(_jit(NAN))


# --- Absolute value ---

class TestFabs:
    @staticmethod
    def test_typical():
        assert fabs(1.0) == 1.0
        assert fabs(-1.0) == 1.0
        assert fabs(0.0) == 0.0
        assert fabs(-0.0) == 0.0
        assert fabs(1e300) == 1e300
        assert fabs(-1e300) == 1e300

    @staticmethod
    def test_inf():
        assert fabs(INF) == INF
        assert fabs(-INF) == INF

    @staticmethod
    def test_nan():
        assert np.isnan(fabs(NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x):
            return fabs(x)

        assert _jit(1.0) == 1.0
        assert _jit(-1.0) == 1.0
        assert _jit(0.0) == 0.0
        assert _jit(-0.0) == 0.0
        assert _jit(1e300) == 1e300
        assert _jit(-1e300) == 1e300
        assert _jit(INF) == INF
        assert _jit(-INF) == INF
        assert np.isnan(_jit(NAN))


# --- Two-argument: trig ---

class TestAtan2:
    @staticmethod
    def test_typical():
        pairs = [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0),
                 (-1.0, -1.0), (0.0, 1.0), (1.0, 0.0)]
        for y, x in pairs:
            assert_close(atan2(y, x), math.atan2(y, x))

    @staticmethod
    def test_zero():
        assert_close(atan2(0.0, 0.0), math.atan2(0.0, 0.0))
        assert_close(atan2(-0.0, 0.0), math.atan2(-0.0, 0.0))

    @staticmethod
    def test_inf():
        assert_close(atan2(1.0, INF), 0.0)
        assert_close(atan2(INF, 1.0), math.pi / 2)

    @staticmethod
    def test_nan():
        assert np.isnan(atan2(NAN, 1.0))
        assert np.isnan(atan2(1.0, NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(y, x):
            return atan2(y, x)

        pairs = [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0),
                 (-1.0, -1.0), (0.0, 1.0), (1.0, 0.0)]
        for y, x in pairs:
            assert_close(_jit(y, x), math.atan2(y, x))
        assert_close(_jit(0.0, 0.0), math.atan2(0.0, 0.0))
        assert_close(_jit(-0.0, 0.0), math.atan2(-0.0, 0.0))
        assert_close(_jit(1.0, INF), 0.0)
        assert_close(_jit(INF, 1.0), math.pi / 2)
        assert np.isnan(_jit(NAN, 1.0))
        assert np.isnan(_jit(1.0, NAN))


# --- Two-argument: power ---

class TestPow:
    @staticmethod
    def test_typical():
        for x, y in [(2.0, 3.0), (2.0, 0.5), (10.0, 2.0), (2.0, -1.0)]:
            assert_close(pow(x, y), math.pow(x, y))

    @staticmethod
    def test_zero_exponent():
        assert_close(pow(5.0, 0.0), 1.0)
        assert_close(pow(0.0, 0.0), 1.0)

    @staticmethod
    def test_one_base():
        assert_close(pow(1.0, 1e300), 1.0)

    @staticmethod
    def test_inf():
        assert pow(2.0, INF) == INF
        assert pow(2.0, -INF) == 0.0

    @staticmethod
    def test_nan():
        assert np.isnan(pow(NAN, 2.0))

    @staticmethod
    def test_negative_base_noninteger():
        assert np.isnan(pow(-1.0, 0.5))

    @staticmethod
    def test_zero_base_negative_exponent():
        assert pow(0.0, -1.0) == INF

    @staticmethod
    def test_jit():
        @njit
        def _jit(x, y):
            return pow(x, y)

        for x, y in [(2.0, 3.0), (2.0, 0.5), (10.0, 2.0), (2.0, -1.0)]:
            assert_close(_jit(x, y), math.pow(x, y))
        assert_close(_jit(5.0, 0.0), 1.0)
        assert_close(_jit(0.0, 0.0), 1.0)
        assert_close(_jit(1.0, 1e300), 1.0)
        assert _jit(2.0, INF) == INF
        assert _jit(2.0, -INF) == 0.0
        assert np.isnan(_jit(NAN, 2.0))
        assert np.isnan(_jit(-1.0, 0.5))
        assert _jit(0.0, -1.0) == INF


# --- Two-argument: modular ---

class TestFmod:
    @staticmethod
    def test_typical():
        for x, y in [(5.0, 3.0), (-5.0, 3.0), (5.0, -3.0), (10.0, 2.5)]:
            assert_close(fmod(x, y), math.fmod(x, y))

    @staticmethod
    def test_zero_dividend():
        assert_close(fmod(0.0, 1.0), 0.0)

    @staticmethod
    def test_inf_dividend():
        assert np.isnan(fmod(INF, 1.0))

    @staticmethod
    def test_nan():
        assert np.isnan(fmod(NAN, 1.0))
        assert np.isnan(fmod(1.0, NAN))
        assert np.isnan(fmod(1.0, 0.0))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x, y):
            return fmod(x, y)

        for x, y in [(5.0, 3.0), (-5.0, 3.0), (5.0, -3.0), (10.0, 2.5)]:
            assert_close(_jit(x, y), math.fmod(x, y))
        assert_close(_jit(0.0, 1.0), 0.0)
        assert np.isnan(_jit(INF, 1.0))
        assert np.isnan(_jit(NAN, 1.0))
        assert np.isnan(_jit(1.0, NAN))
        assert np.isnan(_jit(1.0, 0.0))


class TestRemainder:
    @staticmethod
    def test_typical():
        for x, y in [(5.0, 3.0), (-5.0, 3.0), (5.0, -3.0), (10.0, 3.0)]:
            assert_close(remainder(x, y), math.remainder(x, y))

    @staticmethod
    def test_zero_dividend():
        assert_close(remainder(0.0, 1.0), 0.0)

    @staticmethod
    def test_inf_dividend():
        assert np.isnan(remainder(INF, 1.0))

    @staticmethod
    def test_nan():
        assert np.isnan(remainder(NAN, 1.0))
        assert np.isnan(remainder(1.0, NAN))
        assert np.isnan(remainder(1.0, 0.0))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x, y):
            return remainder(x, y)

        for x, y in [(5.0, 3.0), (-5.0, 3.0), (5.0, -3.0), (10.0, 3.0)]:
            assert_close(_jit(x, y), math.remainder(x, y))
        assert_close(_jit(0.0, 1.0), 0.0)
        assert np.isnan(_jit(INF, 1.0))
        assert np.isnan(_jit(NAN, 1.0))
        assert np.isnan(_jit(1.0, NAN))
        assert np.isnan(_jit(1.0, 0.0))


# --- Two-argument: geometry ---

class TestHypot:
    @staticmethod
    def test_typical():
        for x, y in [(3.0, 4.0), (1.0, 1.0), (0.0, 5.0), (5.0, 0.0)]:
            assert_close(hypot(x, y), math.hypot(x, y))

    @staticmethod
    def test_negative():
        assert_close(hypot(-3.0, -4.0), 5.0)

    @staticmethod
    def test_inf():
        assert hypot(INF, 1.0) == INF
        assert hypot(1.0, INF) == INF
        assert hypot(INF, NAN) == INF

    @staticmethod
    def test_nan():
        assert np.isnan(hypot(NAN, 1.0))
        assert np.isnan(hypot(1.0, NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x, y):
            return hypot(x, y)

        for x, y in [(3.0, 4.0), (1.0, 1.0), (0.0, 5.0), (5.0, 0.0)]:
            assert_close(_jit(x, y), math.hypot(x, y))
        assert_close(_jit(-3.0, -4.0), 5.0)
        assert _jit(INF, 1.0) == INF
        assert _jit(1.0, INF) == INF
        assert _jit(INF, NAN) == INF
        assert np.isnan(_jit(NAN, 1.0))
        assert np.isnan(_jit(1.0, NAN))


# --- Two-argument: comparison ---

class TestFmax:
    @staticmethod
    def test_typical():
        assert_close(fmax(2.0, 3.0), 3.0)
        assert_close(fmax(-1.0, -2.0), -1.0)
        assert_close(fmax(0.0, -0.0), 0.0)

    @staticmethod
    def test_nan_ignored():
        assert_close(fmax(NAN, 1.0), 1.0)
        assert_close(fmax(1.0, NAN), 1.0)

    @staticmethod
    def test_both_nan():
        assert np.isnan(fmax(NAN, NAN))

    @staticmethod
    def test_inf():
        assert fmax(INF, 1.0) == INF
        assert fmax(-INF, 1.0) == 1.0

    @staticmethod
    def test_jit():
        @njit
        def _jit(x, y):
            return fmax(x, y)

        assert_close(_jit(2.0, 3.0), 3.0)
        assert_close(_jit(-1.0, -2.0), -1.0)
        assert_close(_jit(0.0, -0.0), 0.0)
        assert_close(_jit(NAN, 1.0), 1.0)
        assert_close(_jit(1.0, NAN), 1.0)
        assert np.isnan(_jit(NAN, NAN))
        assert _jit(INF, 1.0) == INF
        assert _jit(-INF, 1.0) == 1.0


class TestFmin:
    @staticmethod
    def test_typical():
        assert_close(fmin(2.0, 3.0), 2.0)
        assert_close(fmin(-1.0, -2.0), -2.0)

    @staticmethod
    def test_nan_ignored():
        assert_close(fmin(NAN, 1.0), 1.0)
        assert_close(fmin(1.0, NAN), 1.0)

    @staticmethod
    def test_both_nan():
        assert np.isnan(fmin(NAN, NAN))

    @staticmethod
    def test_inf():
        assert fmin(-INF, 1.0) == -INF
        assert fmin(INF, 1.0) == 1.0

    @staticmethod
    def test_jit():
        @njit
        def _jit(x, y):
            return fmin(x, y)

        assert_close(_jit(2.0, 3.0), 2.0)
        assert_close(_jit(-1.0, -2.0), -2.0)
        assert_close(_jit(NAN, 1.0), 1.0)
        assert_close(_jit(1.0, NAN), 1.0)
        assert np.isnan(_jit(NAN, NAN))
        assert _jit(-INF, 1.0) == -INF
        assert _jit(INF, 1.0) == 1.0


class TestFdim:
    @staticmethod
    def test_typical():
        assert_close(fdim(5.0, 3.0), 2.0)
        assert_close(fdim(3.0, 5.0), 0.0)
        assert_close(fdim(0.0, 0.0), 0.0)
        assert_close(fdim(-1.0, -5.0), 4.0)

    @staticmethod
    def test_inf():
        assert fdim(INF, 1.0) == INF
        assert fdim(1.0, INF) == 0.0

    @staticmethod
    def test_nan():
        assert np.isnan(fdim(NAN, 1.0))
        assert np.isnan(fdim(1.0, NAN))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x, y):
            return fdim(x, y)

        assert_close(_jit(5.0, 3.0), 2.0)
        assert_close(_jit(3.0, 5.0), 0.0)
        assert_close(_jit(0.0, 0.0), 0.0)
        assert_close(_jit(-1.0, -5.0), 4.0)
        assert _jit(INF, 1.0) == INF
        assert _jit(1.0, INF) == 0.0
        assert np.isnan(_jit(NAN, 1.0))
        assert np.isnan(_jit(1.0, NAN))


# --- Two-argument: utility ---

class TestCopysign:
    @staticmethod
    def test_typical():
        assert_close(copysign(1.0, -1.0), -1.0)
        assert_close(copysign(-1.0, 1.0), 1.0)
        assert_close(copysign(5.0, -0.0), -5.0)
        assert_close(copysign(-5.0, 0.0), 5.0)

    @staticmethod
    def test_zero_sign():
        r = copysign(0.0, -1.0)
        assert math.copysign(1.0, r) == -1.0  # result is -0.0

    @staticmethod
    def test_inf():
        assert copysign(INF, -1.0) == -INF
        assert copysign(-INF, 1.0) == INF

    @staticmethod
    def test_nan():
        assert np.isnan(copysign(NAN, 1.0))
        assert np.isnan(copysign(NAN, -1.0))

    @staticmethod
    def test_jit():
        @njit
        def _jit(x, y):
            return copysign(x, y)

        assert_close(_jit(1.0, -1.0), -1.0)
        assert_close(_jit(-1.0, 1.0), 1.0)
        assert_close(_jit(5.0, -0.0), -5.0)
        assert_close(_jit(-5.0, 0.0), 5.0)
        assert math.copysign(1.0, _jit(0.0, -1.0)) == -1.0
        assert _jit(INF, -1.0) == -INF
        assert _jit(-INF, 1.0) == INF
        assert np.isnan(_jit(NAN, 1.0))
        assert np.isnan(_jit(NAN, -1.0))


if __name__ == "__main__":
    collect_and_run_tests(__name__)
