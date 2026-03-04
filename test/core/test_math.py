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
        plain = sin(x) / cos(x)
        jitted = _sin(x) / _cos(x)
        assert_close(plain, jitted)
        assert np.isclose(plain, tan(x))
        assert np.isclose(jitted, _tan(x))

    @staticmethod
    def test_cos():
        @njit
        def _jit(x):
            return cos(x)

        for x in [0.0, 1.0, -1.0, math.pi, math.pi / 2]:
            p, j = cos(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.cos(x))
            assert_close(j, math.cos(x))

    @staticmethod
    def test_sin():
        @njit
        def _jit(x):
            return sin(x)

        for x in [0.0, 1.0, -1.0, math.pi, math.pi / 2]:
            p, j = sin(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.sin(x))
            assert_close(j, math.sin(x))

    @staticmethod
    def test_tan():
        @njit
        def _jit(x):
            return tan(x)

        for x in [0.0, 1.0, -1.0, math.pi / 4]:
            p, j = tan(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.tan(x))
            assert_close(j, math.tan(x))

    @staticmethod
    def test_nan():
        @njit
        def _cos(x):
            return cos(x)

        @njit
        def _sin(x):
            return sin(x)

        @njit
        def _tan(x):
            return tan(x)

        assert_close(cos(NAN), _cos(NAN))
        assert np.isnan(cos(NAN))
        assert_close(sin(NAN), _sin(NAN))
        assert np.isnan(sin(NAN))
        assert_close(tan(NAN), _tan(NAN))
        assert np.isnan(tan(NAN))

    @staticmethod
    def test_inf():
        @njit
        def _cos(x):
            return cos(x)

        @njit
        def _sin(x):
            return sin(x)

        @njit
        def _tan(x):
            return tan(x)

        for x in [INF, -INF]:
            assert_close(cos(x), _cos(x))
            assert np.isnan(cos(x))
            assert_close(sin(x), _sin(x))
            assert np.isnan(sin(x))
            assert_close(tan(x), _tan(x))
            assert np.isnan(tan(x))


# --- Inverse trig ---

class TestAcos:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return acos(x)

        for x in [0.0, 0.5, -0.5, 1.0, -1.0]:
            p, j = acos(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.acos(x))
            assert_close(j, math.acos(x))

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return acos(x)

        assert_close(acos(NAN), _jit(NAN))
        assert np.isnan(acos(NAN))

    @staticmethod
    def test_out_of_domain():
        @njit
        def _jit(x):
            return acos(x)

        for x in [2.0, -2.0]:
            assert_close(acos(x), _jit(x))
            assert np.isnan(acos(x))


class TestAsin:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return asin(x)

        for x in [0.0, 0.5, -0.5, 1.0, -1.0]:
            p, j = asin(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.asin(x))
            assert_close(j, math.asin(x))

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return asin(x)

        assert_close(asin(NAN), _jit(NAN))
        assert np.isnan(asin(NAN))

    @staticmethod
    def test_out_of_domain():
        @njit
        def _jit(x):
            return asin(x)

        for x in [2.0, -2.0]:
            assert_close(asin(x), _jit(x))
            assert np.isnan(asin(x))


class TestAtan:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return atan(x)

        for x in [0.0, 1.0, -1.0, 1e-300, 1e300]:
            p, j = atan(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.atan(x))
            assert_close(j, math.atan(x))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return atan(x)

        p, j = atan(INF), _jit(INF)
        assert_close(p, j)
        assert_close(p, math.pi / 2)
        assert_close(j, math.pi / 2)
        p, j = atan(-INF), _jit(-INF)
        assert_close(p, j)
        assert_close(p, -math.pi / 2)
        assert_close(j, -math.pi / 2)

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return atan(x)

        assert_close(atan(NAN), _jit(NAN))
        assert np.isnan(atan(NAN))


# --- Hyperbolic ---

class TestCosh:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return cosh(x)

        for x in [0.0, 1.0, -1.0, 0.5]:
            p, j = cosh(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.cosh(x))
            assert_close(j, math.cosh(x))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return cosh(x)

        p, j = cosh(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF
        p, j = cosh(-INF), _jit(-INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return cosh(x)

        assert_close(cosh(NAN), _jit(NAN))
        assert np.isnan(cosh(NAN))


class TestSinh:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return sinh(x)

        for x in [0.0, 1.0, -1.0, 0.5]:
            p, j = sinh(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.sinh(x))
            assert_close(j, math.sinh(x))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return sinh(x)

        p, j = sinh(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF
        p, j = sinh(-INF), _jit(-INF)
        assert_close(p, j)
        assert p == -INF
        assert j == -INF

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return sinh(x)

        assert_close(sinh(NAN), _jit(NAN))
        assert np.isnan(sinh(NAN))


class TestTanh:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return tanh(x)

        for x in [0.0, 1.0, -1.0, 0.5]:
            p, j = tanh(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.tanh(x))
            assert_close(j, math.tanh(x))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return tanh(x)

        p, j = tanh(INF), _jit(INF)
        assert_close(p, j)
        assert_close(p, 1.0)
        assert_close(j, 1.0)
        p, j = tanh(-INF), _jit(-INF)
        assert_close(p, j)
        assert_close(p, -1.0)
        assert_close(j, -1.0)

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return tanh(x)

        assert_close(tanh(NAN), _jit(NAN))
        assert np.isnan(tanh(NAN))


class TestAcosh:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return acosh(x)

        for x in [1.0, 2.0, 10.0, 1e300]:
            p, j = acosh(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.acosh(x))
            assert_close(j, math.acosh(x))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return acosh(x)

        p, j = acosh(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF

    @staticmethod
    def test_out_of_domain():
        @njit
        def _jit(x):
            return acosh(x)

        assert_close(acosh(0.5), _jit(0.5))
        assert np.isnan(acosh(0.5))

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return acosh(x)

        assert_close(acosh(NAN), _jit(NAN))
        assert np.isnan(acosh(NAN))


class TestAsinh:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return asinh(x)

        for x in [0.0, 1.0, -1.0, 1e-300, 1e300]:
            p, j = asinh(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.asinh(x))
            assert_close(j, math.asinh(x))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return asinh(x)

        p, j = asinh(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF
        p, j = asinh(-INF), _jit(-INF)
        assert_close(p, j)
        assert p == -INF
        assert j == -INF

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return asinh(x)

        assert_close(asinh(NAN), _jit(NAN))
        assert np.isnan(asinh(NAN))


class TestAtanh:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return atanh(x)

        for x in [0.0, 0.5, -0.5, 1e-300]:
            p, j = atanh(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.atanh(x))
            assert_close(j, math.atanh(x))

    @staticmethod
    def test_boundary():
        @njit
        def _jit(x):
            return atanh(x)

        p, j = atanh(1.0), _jit(1.0)
        assert_close(p, j)
        assert p == INF
        assert j == INF
        p, j = atanh(-1.0), _jit(-1.0)
        assert_close(p, j)
        assert p == -INF
        assert j == -INF

    @staticmethod
    def test_out_of_domain():
        @njit
        def _jit(x):
            return atanh(x)

        assert_close(atanh(2.0), _jit(2.0))
        assert np.isnan(atanh(2.0))

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return atanh(x)

        assert_close(atanh(NAN), _jit(NAN))
        assert np.isnan(atanh(NAN))


# --- Exponential/log ---

class TestExp:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return exp(x)

        for x in [0.0, 1.0, -1.0, 2.0, 1e-300]:
            p, j = exp(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.exp(x))
            assert_close(j, math.exp(x))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return exp(x)

        p, j = exp(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF
        p, j = exp(-INF), _jit(-INF)
        assert_close(p, j)
        assert p == 0.0
        assert j == 0.0

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return exp(x)

        assert_close(exp(NAN), _jit(NAN))
        assert np.isnan(exp(NAN))


class TestExp2:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return exp2(x)

        cases = [(0.0, 1.0), (1.0, 2.0),
                 (10.0, 1024.0), (-1.0, 0.5)]
        for x, expected in cases:
            p, j = exp2(x), _jit(x)
            assert_close(p, j)
            assert_close(p, expected)
            assert_close(j, expected)

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return exp2(x)

        p, j = exp2(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF
        p, j = exp2(-INF), _jit(-INF)
        assert_close(p, j)
        assert p == 0.0
        assert j == 0.0

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return exp2(x)

        assert_close(exp2(NAN), _jit(NAN))
        assert np.isnan(exp2(NAN))


class TestExpm1:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return expm1(x)

        for x in [0.0, 1.0, -1.0, 1e-300]:
            p, j = expm1(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.expm1(x))
            assert_close(j, math.expm1(x))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return expm1(x)

        p, j = expm1(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF
        p, j = expm1(-INF), _jit(-INF)
        assert_close(p, j)
        assert p == -1.0
        assert j == -1.0

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return expm1(x)

        assert_close(expm1(NAN), _jit(NAN))
        assert np.isnan(expm1(NAN))


class TestLog:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return log(x)

        for x in [1.0, 2.0, math.e, 10.0, 1e-300, 1e300]:
            p, j = log(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.log(x))
            assert_close(j, math.log(x))

    @staticmethod
    def test_zero():
        @njit
        def _jit(x):
            return log(x)

        p, j = log(0.0), _jit(0.0)
        assert_close(p, j)
        assert p == -INF
        assert j == -INF

    @staticmethod
    def test_negative():
        @njit
        def _jit(x):
            return log(x)

        assert_close(log(-1.0), _jit(-1.0))
        assert np.isnan(log(-1.0))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return log(x)

        p, j = log(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return log(x)

        assert_close(log(NAN), _jit(NAN))
        assert np.isnan(log(NAN))


class TestLog2:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return log2(x)

        cases = [(1.0, 0.0), (2.0, 1.0), (1024.0, 10.0)]
        for x, expected in cases:
            p, j = log2(x), _jit(x)
            assert_close(p, j)
            assert_close(p, expected)
            assert_close(j, expected)

    @staticmethod
    def test_zero():
        @njit
        def _jit(x):
            return log2(x)

        p, j = log2(0.0), _jit(0.0)
        assert_close(p, j)
        assert p == -INF
        assert j == -INF

    @staticmethod
    def test_negative():
        @njit
        def _jit(x):
            return log2(x)

        assert_close(log2(-1.0), _jit(-1.0))
        assert np.isnan(log2(-1.0))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return log2(x)

        p, j = log2(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return log2(x)

        assert_close(log2(NAN), _jit(NAN))
        assert np.isnan(log2(NAN))


class TestLog10:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return log10(x)

        cases = [(1.0, 0.0), (10.0, 1.0), (1000.0, 3.0)]
        for x, expected in cases:
            p, j = log10(x), _jit(x)
            assert_close(p, j)
            assert_close(p, expected)
            assert_close(j, expected)

    @staticmethod
    def test_zero():
        @njit
        def _jit(x):
            return log10(x)

        p, j = log10(0.0), _jit(0.0)
        assert_close(p, j)
        assert p == -INF
        assert j == -INF

    @staticmethod
    def test_negative():
        @njit
        def _jit(x):
            return log10(x)

        assert_close(log10(-1.0), _jit(-1.0))
        assert np.isnan(log10(-1.0))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return log10(x)

        p, j = log10(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return log10(x)

        assert_close(log10(NAN), _jit(NAN))
        assert np.isnan(log10(NAN))


class TestLog1p:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return log1p(x)

        for x in [0.0, 1.0, 1e-300, -0.5]:
            p, j = log1p(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.log1p(x))
            assert_close(j, math.log1p(x))

    @staticmethod
    def test_minus_one():
        @njit
        def _jit(x):
            return log1p(x)

        p, j = log1p(-1.0), _jit(-1.0)
        assert_close(p, j)
        assert p == -INF
        assert j == -INF

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return log1p(x)

        p, j = log1p(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return log1p(x)

        assert_close(log1p(NAN), _jit(NAN))
        assert np.isnan(log1p(NAN))


class TestLogb:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return logb(x)

        cases = [(1.0, 0.0), (2.0, 1.0),
                 (8.0, 3.0), (0.5, -1.0)]
        for x, expected in cases:
            p, j = logb(x), _jit(x)
            assert_close(p, j)
            assert_close(p, expected)
            assert_close(j, expected)

    @staticmethod
    def test_negative():
        @njit
        def _jit(x):
            return logb(x)

        for x, expected in [(-2.0, 1.0), (-8.0, 3.0)]:
            p, j = logb(x), _jit(x)
            assert_close(p, j)
            assert_close(p, expected)
            assert_close(j, expected)

    @staticmethod
    def test_zero():
        @njit
        def _jit(x):
            return logb(x)

        p, j = logb(0.0), _jit(0.0)
        assert_close(p, j)
        assert p == -INF
        assert j == -INF

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return logb(x)

        p, j = logb(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return logb(x)

        assert_close(logb(NAN), _jit(NAN))
        assert np.isnan(logb(NAN))


# --- Power/root ---

class TestSqrt:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return sqrt(x)

        for x in [0.0, 1.0, 4.0, 2.0, 1e300]:
            p, j = sqrt(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.sqrt(x))
            assert_close(j, math.sqrt(x))

    @staticmethod
    def test_negative():
        @njit
        def _jit(x):
            return sqrt(x)

        assert_close(sqrt(-1.0), _jit(-1.0))
        assert np.isnan(sqrt(-1.0))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return sqrt(x)

        p, j = sqrt(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return sqrt(x)

        assert_close(sqrt(NAN), _jit(NAN))
        assert np.isnan(sqrt(NAN))


class TestCbrt:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return cbrt(x)

        cases = [(0.0, 0.0), (1.0, 1.0), (8.0, 2.0),
                 (27.0, 3.0), (-8.0, -2.0)]
        for x, expected in cases:
            p, j = cbrt(x), _jit(x)
            assert_close(p, j)
            assert_close(p, expected)
            assert_close(j, expected)

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return cbrt(x)

        p, j = cbrt(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF
        p, j = cbrt(-INF), _jit(-INF)
        assert_close(p, j)
        assert p == -INF
        assert j == -INF

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return cbrt(x)

        assert_close(cbrt(NAN), _jit(NAN))
        assert np.isnan(cbrt(NAN))


# --- Rounding ---

class TestCeil:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return ceil(x)

        cases = [(2.3, 3.0), (-2.3, -2.0),
                 (0.0, 0.0), (2.0, 2.0)]
        for x, expected in cases:
            p, j = ceil(x), _jit(x)
            assert_close(p, j)
            assert p == expected
            assert j == expected

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return ceil(x)

        for x in [INF, -INF]:
            p, j = ceil(x), _jit(x)
            assert_close(p, j)
            assert p == x
            assert j == x

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return ceil(x)

        assert_close(ceil(NAN), _jit(NAN))
        assert np.isnan(ceil(NAN))


class TestFloor:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return floor(x)

        cases = [(2.7, 2.0), (-2.7, -3.0),
                 (0.0, 0.0), (2.0, 2.0)]
        for x, expected in cases:
            p, j = floor(x), _jit(x)
            assert_close(p, j)
            assert p == expected
            assert j == expected

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return floor(x)

        for x in [INF, -INF]:
            p, j = floor(x), _jit(x)
            assert_close(p, j)
            assert p == x
            assert j == x

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return floor(x)

        assert_close(floor(NAN), _jit(NAN))
        assert np.isnan(floor(NAN))


class TestTrunc:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return trunc(x)

        cases = [(2.7, 2.0), (-2.7, -2.0), (0.0, 0.0)]
        for x, expected in cases:
            p, j = trunc(x), _jit(x)
            assert_close(p, j)
            assert p == expected
            assert j == expected

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return trunc(x)

        for x in [INF, -INF]:
            p, j = trunc(x), _jit(x)
            assert_close(p, j)
            assert p == x
            assert j == x

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return trunc(x)

        assert_close(trunc(NAN), _jit(NAN))
        assert np.isnan(trunc(NAN))


class TestRound:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return round(x)

        # C round: ties away from zero
        cases = [(2.3, 2.0), (2.5, 3.0),
                 (-2.5, -3.0), (0.0, 0.0)]
        for x, expected in cases:
            p, j = round(x), _jit(x)
            assert_close(p, j)
            assert p == expected
            assert j == expected

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return round(x)

        for x in [INF, -INF]:
            p, j = round(x), _jit(x)
            assert_close(p, j)
            assert p == x
            assert j == x

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return round(x)

        assert_close(round(NAN), _jit(NAN))
        assert np.isnan(round(NAN))


class TestRint:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return rint(x)

        cases = [(2.3, 2.0), (2.7, 3.0), (0.0, 0.0)]
        for x, expected in cases:
            p, j = rint(x), _jit(x)
            assert_close(p, j)
            assert p == expected
            assert j == expected

    @staticmethod
    def test_halfway():
        @njit
        def _jit(x):
            return rint(x)

        # ties to even
        for x, expected in [(2.5, 2.0), (3.5, 4.0)]:
            p, j = rint(x), _jit(x)
            assert_close(p, j)
            assert p == expected
            assert j == expected

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return rint(x)

        for x in [INF, -INF]:
            p, j = rint(x), _jit(x)
            assert_close(p, j)
            assert p == x
            assert j == x

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return rint(x)

        assert_close(rint(NAN), _jit(NAN))
        assert np.isnan(rint(NAN))


class TestNearbyint:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return nearbyint(x)

        cases = [(2.3, 2.0), (2.7, 3.0), (0.0, 0.0)]
        for x, expected in cases:
            p, j = nearbyint(x), _jit(x)
            assert_close(p, j)
            assert p == expected
            assert j == expected

    @staticmethod
    def test_halfway():
        @njit
        def _jit(x):
            return nearbyint(x)

        # ties to even
        for x, expected in [(2.5, 2.0), (3.5, 4.0)]:
            p, j = nearbyint(x), _jit(x)
            assert_close(p, j)
            assert p == expected
            assert j == expected

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return nearbyint(x)

        for x in [INF, -INF]:
            p, j = nearbyint(x), _jit(x)
            assert_close(p, j)
            assert p == x
            assert j == x

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return nearbyint(x)

        assert_close(nearbyint(NAN), _jit(NAN))
        assert np.isnan(nearbyint(NAN))


# --- Error/gamma ---

class TestErf:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return erf(x)

        for x in [0.0, 1.0, -1.0, 0.5, 2.0]:
            p, j = erf(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.erf(x))
            assert_close(j, math.erf(x))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return erf(x)

        p, j = erf(INF), _jit(INF)
        assert_close(p, j)
        assert_close(p, 1.0)
        assert_close(j, 1.0)
        p, j = erf(-INF), _jit(-INF)
        assert_close(p, j)
        assert_close(p, -1.0)
        assert_close(j, -1.0)

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return erf(x)

        assert_close(erf(NAN), _jit(NAN))
        assert np.isnan(erf(NAN))


class TestErfc:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return erfc(x)

        for x in [0.0, 1.0, -1.0, 0.5, 2.0]:
            p, j = erfc(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.erfc(x))
            assert_close(j, math.erfc(x))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return erfc(x)

        p, j = erfc(INF), _jit(INF)
        assert_close(p, j)
        assert_close(p, 0.0)
        assert_close(j, 0.0)
        p, j = erfc(-INF), _jit(-INF)
        assert_close(p, j)
        assert_close(p, 2.0)
        assert_close(j, 2.0)

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return erfc(x)

        assert_close(erfc(NAN), _jit(NAN))
        assert np.isnan(erfc(NAN))


class TestLgamma:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return lgamma(x)

        for x in [1.0, 2.0, 0.5, 10.0]:
            p, j = lgamma(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.lgamma(x))
            assert_close(j, math.lgamma(x))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return lgamma(x)

        p, j = lgamma(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return lgamma(x)

        assert_close(lgamma(NAN), _jit(NAN))
        assert np.isnan(lgamma(NAN))


class TestTgamma:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return tgamma(x)

        for x in [1.0, 2.0, 0.5, 5.0]:
            p, j = tgamma(x), _jit(x)
            assert_close(p, j)
            assert_close(p, math.gamma(x))
            assert_close(j, math.gamma(x))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return tgamma(x)

        p, j = tgamma(INF), _jit(INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF

    @staticmethod
    def test_zero():
        @njit
        def _jit(x):
            return tgamma(x)

        p, j = tgamma(0.0), _jit(0.0)
        assert_close(p, j)
        assert np.isinf(p)

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return tgamma(x)

        assert_close(tgamma(NAN), _jit(NAN))
        assert np.isnan(tgamma(NAN))


# --- Absolute value ---

class TestFabs:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x):
            return fabs(x)

        cases = [(1.0, 1.0), (-1.0, 1.0),
                 (0.0, 0.0), (-0.0, 0.0),
                 (1e300, 1e300), (-1e300, 1e300)]
        for x, expected in cases:
            p, j = fabs(x), _jit(x)
            assert_close(p, j)
            assert p == expected
            assert j == expected

    @staticmethod
    def test_inf():
        @njit
        def _jit(x):
            return fabs(x)

        for x in [INF, -INF]:
            p, j = fabs(x), _jit(x)
            assert_close(p, j)
            assert p == INF
            assert j == INF

    @staticmethod
    def test_nan():
        @njit
        def _jit(x):
            return fabs(x)

        assert_close(fabs(NAN), _jit(NAN))
        assert np.isnan(fabs(NAN))


# --- Two-argument: trig ---

class TestAtan2:
    @staticmethod
    def test_typical():
        @njit
        def _jit(y, x):
            return atan2(y, x)

        pairs = [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0),
                 (-1.0, -1.0), (0.0, 1.0), (1.0, 0.0)]
        for y, x in pairs:
            p, j = atan2(y, x), _jit(y, x)
            assert_close(p, j)
            assert_close(p, math.atan2(y, x))
            assert_close(j, math.atan2(y, x))

    @staticmethod
    def test_zero():
        @njit
        def _jit(y, x):
            return atan2(y, x)

        for y, x in [(0.0, 0.0), (-0.0, 0.0)]:
            p, j = atan2(y, x), _jit(y, x)
            assert_close(p, j)
            assert_close(p, math.atan2(y, x))
            assert_close(j, math.atan2(y, x))

    @staticmethod
    def test_inf():
        @njit
        def _jit(y, x):
            return atan2(y, x)

        p, j = atan2(1.0, INF), _jit(1.0, INF)
        assert_close(p, j)
        assert_close(p, 0.0)
        assert_close(j, 0.0)
        p, j = atan2(INF, 1.0), _jit(INF, 1.0)
        assert_close(p, j)
        assert_close(p, math.pi / 2)
        assert_close(j, math.pi / 2)

    @staticmethod
    def test_nan():
        @njit
        def _jit(y, x):
            return atan2(y, x)

        for y, x in [(NAN, 1.0), (1.0, NAN)]:
            assert_close(atan2(y, x), _jit(y, x))
            assert np.isnan(atan2(y, x))


# --- Two-argument: power ---

class TestPow:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x, y):
            return pow(x, y)

        pairs = [(2.0, 3.0), (2.0, 0.5),
                 (10.0, 2.0), (2.0, -1.0)]
        for x, y in pairs:
            p, j = pow(x, y), _jit(x, y)
            assert_close(p, j)
            assert_close(p, math.pow(x, y))
            assert_close(j, math.pow(x, y))

    @staticmethod
    def test_zero_exponent():
        @njit
        def _jit(x, y):
            return pow(x, y)

        for x in [5.0, 0.0]:
            p, j = pow(x, 0.0), _jit(x, 0.0)
            assert_close(p, j)
            assert_close(p, 1.0)
            assert_close(j, 1.0)

    @staticmethod
    def test_one_base():
        @njit
        def _jit(x, y):
            return pow(x, y)

        p, j = pow(1.0, 1e300), _jit(1.0, 1e300)
        assert_close(p, j)
        assert_close(p, 1.0)
        assert_close(j, 1.0)

    @staticmethod
    def test_inf():
        @njit
        def _jit(x, y):
            return pow(x, y)

        p, j = pow(2.0, INF), _jit(2.0, INF)
        assert_close(p, j)
        assert p == INF
        assert j == INF
        p, j = pow(2.0, -INF), _jit(2.0, -INF)
        assert_close(p, j)
        assert p == 0.0
        assert j == 0.0

    @staticmethod
    def test_nan():
        @njit
        def _jit(x, y):
            return pow(x, y)

        assert_close(pow(NAN, 2.0), _jit(NAN, 2.0))
        assert np.isnan(pow(NAN, 2.0))

    @staticmethod
    def test_negative_base_noninteger():
        @njit
        def _jit(x, y):
            return pow(x, y)

        assert_close(pow(-1.0, 0.5), _jit(-1.0, 0.5))
        assert np.isnan(pow(-1.0, 0.5))

    @staticmethod
    def test_zero_base_negative_exponent():
        @njit
        def _jit(x, y):
            return pow(x, y)

        p, j = pow(0.0, -1.0), _jit(0.0, -1.0)
        assert_close(p, j)
        assert p == INF
        assert j == INF


# --- Two-argument: modular ---

class TestFmod:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x, y):
            return fmod(x, y)

        pairs = [(5.0, 3.0), (-5.0, 3.0),
                 (5.0, -3.0), (10.0, 2.5)]
        for x, y in pairs:
            p, j = fmod(x, y), _jit(x, y)
            assert_close(p, j)
            assert_close(p, math.fmod(x, y))
            assert_close(j, math.fmod(x, y))

    @staticmethod
    def test_zero_dividend():
        @njit
        def _jit(x, y):
            return fmod(x, y)

        p, j = fmod(0.0, 1.0), _jit(0.0, 1.0)
        assert_close(p, j)
        assert_close(p, 0.0)
        assert_close(j, 0.0)

    @staticmethod
    def test_inf_dividend():
        @njit
        def _jit(x, y):
            return fmod(x, y)

        assert_close(fmod(INF, 1.0), _jit(INF, 1.0))
        assert np.isnan(fmod(INF, 1.0))

    @staticmethod
    def test_nan():
        @njit
        def _jit(x, y):
            return fmod(x, y)

        for x, y in [(NAN, 1.0), (1.0, NAN), (1.0, 0.0)]:
            assert_close(fmod(x, y), _jit(x, y))
            assert np.isnan(fmod(x, y))


class TestRemainder:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x, y):
            return remainder(x, y)

        pairs = [(5.0, 3.0), (-5.0, 3.0),
                 (5.0, -3.0), (10.0, 3.0)]
        for x, y in pairs:
            p, j = remainder(x, y), _jit(x, y)
            assert_close(p, j)
            assert_close(p, math.remainder(x, y))
            assert_close(j, math.remainder(x, y))

    @staticmethod
    def test_zero_dividend():
        @njit
        def _jit(x, y):
            return remainder(x, y)

        p, j = remainder(0.0, 1.0), _jit(0.0, 1.0)
        assert_close(p, j)
        assert_close(p, 0.0)
        assert_close(j, 0.0)

    @staticmethod
    def test_inf_dividend():
        @njit
        def _jit(x, y):
            return remainder(x, y)

        assert_close(remainder(INF, 1.0), _jit(INF, 1.0))
        assert np.isnan(remainder(INF, 1.0))

    @staticmethod
    def test_nan():
        @njit
        def _jit(x, y):
            return remainder(x, y)

        for x, y in [(NAN, 1.0), (1.0, NAN), (1.0, 0.0)]:
            assert_close(remainder(x, y), _jit(x, y))
            assert np.isnan(remainder(x, y))


# --- Two-argument: geometry ---

class TestHypot:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x, y):
            return hypot(x, y)

        pairs = [(3.0, 4.0), (1.0, 1.0),
                 (0.0, 5.0), (5.0, 0.0)]
        for x, y in pairs:
            p, j = hypot(x, y), _jit(x, y)
            assert_close(p, j)
            assert_close(p, math.hypot(x, y))
            assert_close(j, math.hypot(x, y))

    @staticmethod
    def test_negative():
        @njit
        def _jit(x, y):
            return hypot(x, y)

        p, j = hypot(-3.0, -4.0), _jit(-3.0, -4.0)
        assert_close(p, j)
        assert_close(p, 5.0)
        assert_close(j, 5.0)

    @staticmethod
    def test_inf():
        @njit
        def _jit(x, y):
            return hypot(x, y)

        for x, y in [(INF, 1.0), (1.0, INF), (INF, NAN)]:
            p, j = hypot(x, y), _jit(x, y)
            assert_close(p, j)
            assert p == INF
            assert j == INF

    @staticmethod
    def test_nan():
        @njit
        def _jit(x, y):
            return hypot(x, y)

        for x, y in [(NAN, 1.0), (1.0, NAN)]:
            assert_close(hypot(x, y), _jit(x, y))
            assert np.isnan(hypot(x, y))


# --- Two-argument: comparison ---

class TestFmax:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x, y):
            return fmax(x, y)

        cases = [((2.0, 3.0), 3.0),
                 ((-1.0, -2.0), -1.0),
                 ((0.0, -0.0), 0.0)]
        for (x, y), expected in cases:
            p, j = fmax(x, y), _jit(x, y)
            assert_close(p, j)
            assert_close(p, expected)
            assert_close(j, expected)

    @staticmethod
    def test_nan_ignored():
        @njit
        def _jit(x, y):
            return fmax(x, y)

        for x, y in [(NAN, 1.0), (1.0, NAN)]:
            p, j = fmax(x, y), _jit(x, y)
            assert_close(p, j)
            assert_close(p, 1.0)
            assert_close(j, 1.0)

    @staticmethod
    def test_both_nan():
        @njit
        def _jit(x, y):
            return fmax(x, y)

        assert_close(fmax(NAN, NAN), _jit(NAN, NAN))
        assert np.isnan(fmax(NAN, NAN))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x, y):
            return fmax(x, y)

        p, j = fmax(INF, 1.0), _jit(INF, 1.0)
        assert_close(p, j)
        assert p == INF
        assert j == INF
        p, j = fmax(-INF, 1.0), _jit(-INF, 1.0)
        assert_close(p, j)
        assert p == 1.0
        assert j == 1.0


class TestFmin:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x, y):
            return fmin(x, y)

        cases = [((2.0, 3.0), 2.0), ((-1.0, -2.0), -2.0)]
        for (x, y), expected in cases:
            p, j = fmin(x, y), _jit(x, y)
            assert_close(p, j)
            assert_close(p, expected)
            assert_close(j, expected)

    @staticmethod
    def test_nan_ignored():
        @njit
        def _jit(x, y):
            return fmin(x, y)

        for x, y in [(NAN, 1.0), (1.0, NAN)]:
            p, j = fmin(x, y), _jit(x, y)
            assert_close(p, j)
            assert_close(p, 1.0)
            assert_close(j, 1.0)

    @staticmethod
    def test_both_nan():
        @njit
        def _jit(x, y):
            return fmin(x, y)

        assert_close(fmin(NAN, NAN), _jit(NAN, NAN))
        assert np.isnan(fmin(NAN, NAN))

    @staticmethod
    def test_inf():
        @njit
        def _jit(x, y):
            return fmin(x, y)

        p, j = fmin(-INF, 1.0), _jit(-INF, 1.0)
        assert_close(p, j)
        assert p == -INF
        assert j == -INF
        p, j = fmin(INF, 1.0), _jit(INF, 1.0)
        assert_close(p, j)
        assert p == 1.0
        assert j == 1.0


class TestFdim:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x, y):
            return fdim(x, y)

        cases = [((5.0, 3.0), 2.0), ((3.0, 5.0), 0.0),
                 ((0.0, 0.0), 0.0), ((-1.0, -5.0), 4.0)]
        for (x, y), expected in cases:
            p, j = fdim(x, y), _jit(x, y)
            assert_close(p, j)
            assert_close(p, expected)
            assert_close(j, expected)

    @staticmethod
    def test_inf():
        @njit
        def _jit(x, y):
            return fdim(x, y)

        p, j = fdim(INF, 1.0), _jit(INF, 1.0)
        assert_close(p, j)
        assert p == INF
        assert j == INF
        p, j = fdim(1.0, INF), _jit(1.0, INF)
        assert_close(p, j)
        assert p == 0.0
        assert j == 0.0

    @staticmethod
    def test_nan():
        @njit
        def _jit(x, y):
            return fdim(x, y)

        for x, y in [(NAN, 1.0), (1.0, NAN)]:
            assert_close(fdim(x, y), _jit(x, y))
            assert np.isnan(fdim(x, y))


# --- Two-argument: utility ---

class TestCopysign:
    @staticmethod
    def test_typical():
        @njit
        def _jit(x, y):
            return copysign(x, y)

        cases = [((1.0, -1.0), -1.0), ((-1.0, 1.0), 1.0),
                 ((5.0, -0.0), -5.0), ((-5.0, 0.0), 5.0)]
        for (x, y), expected in cases:
            p, j = copysign(x, y), _jit(x, y)
            assert_close(p, j)
            assert_close(p, expected)
            assert_close(j, expected)

    @staticmethod
    def test_zero_sign():
        @njit
        def _jit(x, y):
            return copysign(x, y)

        p = copysign(0.0, -1.0)
        j = _jit(0.0, -1.0)
        assert math.copysign(1.0, p) == -1.0  # -0.0
        assert math.copysign(1.0, j) == -1.0

    @staticmethod
    def test_inf():
        @njit
        def _jit(x, y):
            return copysign(x, y)

        p, j = copysign(INF, -1.0), _jit(INF, -1.0)
        assert_close(p, j)
        assert p == -INF
        assert j == -INF
        p, j = copysign(-INF, 1.0), _jit(-INF, 1.0)
        assert_close(p, j)
        assert p == INF
        assert j == INF

    @staticmethod
    def test_nan():
        @njit
        def _jit(x, y):
            return copysign(x, y)

        for y in [1.0, -1.0]:
            assert_close(copysign(NAN, y), _jit(NAN, y))
            assert np.isnan(copysign(NAN, y))


if __name__ == "__main__":
    collect_and_run_tests(__name__)
