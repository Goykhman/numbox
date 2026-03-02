import math
import numpy as np

from numbox.core.bindings._math import (
    acos, asin, atan,
    cosh, sinh, tanh, acosh, asinh, atanh,
    exp, exp2, expm1,
    log, log2, log10, log1p, logb,
    sqrt, cbrt,
    ceil, floor, trunc, round, rint, nearbyint,
    erf, erfc, lgamma, tgamma,
    fabs,
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
        assert np.isclose(actual, expected), f"expected {expected}, got {actual}"


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
    def test_inf():
        assert log2(INF) == INF

    @staticmethod
    def test_nan():
        assert np.isnan(log2(NAN))


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
    def test_inf():
        assert log10(INF) == INF

    @staticmethod
    def test_nan():
        assert np.isnan(log10(NAN))


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


class TestLogb:
    @staticmethod
    def test_typical():
        assert_close(logb(1.0), 0.0)
        assert_close(logb(2.0), 1.0)
        assert_close(logb(8.0), 3.0)
        assert_close(logb(0.5), -1.0)

    @staticmethod
    def test_zero():
        assert logb(0.0) == -INF

    @staticmethod
    def test_inf():
        assert logb(INF) == INF

    @staticmethod
    def test_nan():
        assert np.isnan(logb(NAN))


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


if __name__ == "__main__":
    collect_and_run_tests(__name__)
