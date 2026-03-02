import math

import numpy as np

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


INF = float("inf")
NAN = float("nan")


def assert_close(actual, expected):
    if np.isnan(expected):
        assert np.isnan(actual), f"expected NaN, got {actual}"
    elif np.isinf(expected):
        assert actual == expected, f"expected {expected}, got {actual}"
    else:
        assert np.isclose(actual, expected), f"expected {expected}, got {actual}"


# --- Trig ---

class TestTrig:
    def test_sin_cos_tan_identity(self):
        x = 3.1415
        assert np.isclose(sin(x) / cos(x), tan(x))

    def test_cos(self):
        for x in [0.0, 1.0, -1.0, math.pi, math.pi / 2]:
            assert_close(cos(x), math.cos(x))

    def test_sin(self):
        for x in [0.0, 1.0, -1.0, math.pi, math.pi / 2]:
            assert_close(sin(x), math.sin(x))

    def test_tan(self):
        for x in [0.0, 1.0, -1.0, math.pi / 4]:
            assert_close(tan(x), math.tan(x))


# --- Inverse trig ---

class TestAcos:
    def test_typical(self):
        for x in [0.0, 0.5, -0.5, 1.0, -1.0]:
            assert_close(acos(x), math.acos(x))

    def test_nan(self):
        assert np.isnan(acos(NAN))

    def test_out_of_domain(self):
        assert np.isnan(acos(2.0))
        assert np.isnan(acos(-2.0))


class TestAsin:
    def test_typical(self):
        for x in [0.0, 0.5, -0.5, 1.0, -1.0]:
            assert_close(asin(x), math.asin(x))

    def test_nan(self):
        assert np.isnan(asin(NAN))

    def test_out_of_domain(self):
        assert np.isnan(asin(2.0))


class TestAtan:
    def test_typical(self):
        for x in [0.0, 1.0, -1.0, 1e-300, 1e300]:
            assert_close(atan(x), math.atan(x))

    def test_inf(self):
        assert_close(atan(INF), math.pi / 2)
        assert_close(atan(-INF), -math.pi / 2)

    def test_nan(self):
        assert np.isnan(atan(NAN))


# --- Hyperbolic ---

class TestCosh:
    def test_typical(self):
        for x in [0.0, 1.0, -1.0, 0.5]:
            assert_close(cosh(x), math.cosh(x))

    def test_inf(self):
        assert cosh(INF) == INF
        assert cosh(-INF) == INF

    def test_nan(self):
        assert np.isnan(cosh(NAN))


class TestSinh:
    def test_typical(self):
        for x in [0.0, 1.0, -1.0, 0.5]:
            assert_close(sinh(x), math.sinh(x))

    def test_inf(self):
        assert sinh(INF) == INF
        assert sinh(-INF) == -INF

    def test_nan(self):
        assert np.isnan(sinh(NAN))


class TestTanh:
    def test_typical(self):
        for x in [0.0, 1.0, -1.0, 0.5]:
            assert_close(tanh(x), math.tanh(x))

    def test_inf(self):
        assert_close(tanh(INF), 1.0)
        assert_close(tanh(-INF), -1.0)

    def test_nan(self):
        assert np.isnan(tanh(NAN))


class TestAcosh:
    def test_typical(self):
        for x in [1.0, 2.0, 10.0, 1e300]:
            assert_close(acosh(x), math.acosh(x))

    def test_inf(self):
        assert acosh(INF) == INF

    def test_out_of_domain(self):
        assert np.isnan(acosh(0.5))

    def test_nan(self):
        assert np.isnan(acosh(NAN))


class TestAsinh:
    def test_typical(self):
        for x in [0.0, 1.0, -1.0, 1e-300, 1e300]:
            assert_close(asinh(x), math.asinh(x))

    def test_inf(self):
        assert asinh(INF) == INF
        assert asinh(-INF) == -INF

    def test_nan(self):
        assert np.isnan(asinh(NAN))


class TestAtanh:
    def test_typical(self):
        for x in [0.0, 0.5, -0.5, 1e-300]:
            assert_close(atanh(x), math.atanh(x))

    def test_boundary(self):
        assert atanh(1.0) == INF
        assert atanh(-1.0) == -INF

    def test_out_of_domain(self):
        assert np.isnan(atanh(2.0))

    def test_nan(self):
        assert np.isnan(atanh(NAN))


# --- Exponential/log ---

class TestExp:
    def test_typical(self):
        for x in [0.0, 1.0, -1.0, 2.0, 1e-300]:
            assert_close(exp(x), math.exp(x))

    def test_inf(self):
        assert exp(INF) == INF
        assert exp(-INF) == 0.0

    def test_nan(self):
        assert np.isnan(exp(NAN))


class TestExp2:
    def test_typical(self):
        assert_close(exp2(0.0), 1.0)
        assert_close(exp2(1.0), 2.0)
        assert_close(exp2(10.0), 1024.0)
        assert_close(exp2(-1.0), 0.5)

    def test_inf(self):
        assert exp2(INF) == INF
        assert exp2(-INF) == 0.0

    def test_nan(self):
        assert np.isnan(exp2(NAN))


class TestExpm1:
    def test_typical(self):
        for x in [0.0, 1.0, -1.0, 1e-300]:
            assert_close(expm1(x), math.expm1(x))

    def test_inf(self):
        assert expm1(INF) == INF
        assert expm1(-INF) == -1.0

    def test_nan(self):
        assert np.isnan(expm1(NAN))


class TestLog:
    def test_typical(self):
        for x in [1.0, 2.0, math.e, 10.0, 1e-300, 1e300]:
            assert_close(log(x), math.log(x))

    def test_zero(self):
        assert log(0.0) == -INF

    def test_negative(self):
        assert np.isnan(log(-1.0))

    def test_inf(self):
        assert log(INF) == INF

    def test_nan(self):
        assert np.isnan(log(NAN))


class TestLog2:
    def test_typical(self):
        assert_close(log2(1.0), 0.0)
        assert_close(log2(2.0), 1.0)
        assert_close(log2(1024.0), 10.0)

    def test_zero(self):
        assert log2(0.0) == -INF

    def test_inf(self):
        assert log2(INF) == INF

    def test_nan(self):
        assert np.isnan(log2(NAN))


class TestLog10:
    def test_typical(self):
        assert_close(log10(1.0), 0.0)
        assert_close(log10(10.0), 1.0)
        assert_close(log10(1000.0), 3.0)

    def test_zero(self):
        assert log10(0.0) == -INF

    def test_inf(self):
        assert log10(INF) == INF

    def test_nan(self):
        assert np.isnan(log10(NAN))


class TestLog1p:
    def test_typical(self):
        for x in [0.0, 1.0, 1e-300, -0.5]:
            assert_close(log1p(x), math.log1p(x))

    def test_minus_one(self):
        assert log1p(-1.0) == -INF

    def test_inf(self):
        assert log1p(INF) == INF

    def test_nan(self):
        assert np.isnan(log1p(NAN))


class TestLogb:
    def test_typical(self):
        assert_close(logb(1.0), 0.0)
        assert_close(logb(2.0), 1.0)
        assert_close(logb(8.0), 3.0)
        assert_close(logb(0.5), -1.0)

    def test_zero(self):
        assert logb(0.0) == -INF

    def test_inf(self):
        assert logb(INF) == INF

    def test_nan(self):
        assert np.isnan(logb(NAN))


# --- Power/root ---

class TestSqrt:
    def test_typical(self):
        for x in [0.0, 1.0, 4.0, 2.0, 1e300]:
            assert_close(sqrt(x), math.sqrt(x))

    def test_negative(self):
        assert np.isnan(sqrt(-1.0))

    def test_inf(self):
        assert sqrt(INF) == INF

    def test_nan(self):
        assert np.isnan(sqrt(NAN))


class TestCbrt:
    def test_typical(self):
        assert_close(cbrt(0.0), 0.0)
        assert_close(cbrt(1.0), 1.0)
        assert_close(cbrt(8.0), 2.0)
        assert_close(cbrt(27.0), 3.0)
        assert_close(cbrt(-8.0), -2.0)

    def test_inf(self):
        assert cbrt(INF) == INF
        assert cbrt(-INF) == -INF

    def test_nan(self):
        assert np.isnan(cbrt(NAN))


# --- Rounding ---

class TestCeil:
    def test_typical(self):
        assert ceil(2.3) == 3.0
        assert ceil(-2.3) == -2.0
        assert ceil(0.0) == 0.0
        assert ceil(2.0) == 2.0

    def test_inf(self):
        assert ceil(INF) == INF
        assert ceil(-INF) == -INF

    def test_nan(self):
        assert np.isnan(ceil(NAN))


class TestFloor:
    def test_typical(self):
        assert floor(2.7) == 2.0
        assert floor(-2.7) == -3.0
        assert floor(0.0) == 0.0
        assert floor(2.0) == 2.0

    def test_inf(self):
        assert floor(INF) == INF
        assert floor(-INF) == -INF

    def test_nan(self):
        assert np.isnan(floor(NAN))


class TestTrunc:
    def test_typical(self):
        assert trunc(2.7) == 2.0
        assert trunc(-2.7) == -2.0
        assert trunc(0.0) == 0.0

    def test_inf(self):
        assert trunc(INF) == INF
        assert trunc(-INF) == -INF

    def test_nan(self):
        assert np.isnan(trunc(NAN))


class TestRound:
    def test_typical(self):
        assert round(2.3) == 2.0
        assert round(2.5) == 3.0  # C round: ties away from zero
        assert round(-2.5) == -3.0
        assert round(0.0) == 0.0

    def test_inf(self):
        assert round(INF) == INF
        assert round(-INF) == -INF

    def test_nan(self):
        assert np.isnan(round(NAN))


class TestRint:
    def test_typical(self):
        assert rint(2.3) == 2.0
        assert rint(2.7) == 3.0
        assert rint(0.0) == 0.0

    def test_halfway(self):
        assert rint(2.5) == 2.0  # ties to even
        assert rint(3.5) == 4.0  # ties to even

    def test_inf(self):
        assert rint(INF) == INF
        assert rint(-INF) == -INF

    def test_nan(self):
        assert np.isnan(rint(NAN))


class TestNearbyint:
    def test_typical(self):
        assert nearbyint(2.3) == 2.0
        assert nearbyint(2.7) == 3.0
        assert nearbyint(0.0) == 0.0

    def test_halfway(self):
        assert nearbyint(2.5) == 2.0  # ties to even
        assert nearbyint(3.5) == 4.0  # ties to even

    def test_inf(self):
        assert nearbyint(INF) == INF
        assert nearbyint(-INF) == -INF

    def test_nan(self):
        assert np.isnan(nearbyint(NAN))


# --- Error/gamma ---

class TestErf:
    def test_typical(self):
        for x in [0.0, 1.0, -1.0, 0.5, 2.0]:
            assert_close(erf(x), math.erf(x))

    def test_inf(self):
        assert_close(erf(INF), 1.0)
        assert_close(erf(-INF), -1.0)

    def test_nan(self):
        assert np.isnan(erf(NAN))


class TestErfc:
    def test_typical(self):
        for x in [0.0, 1.0, -1.0, 0.5, 2.0]:
            assert_close(erfc(x), math.erfc(x))

    def test_inf(self):
        assert_close(erfc(INF), 0.0)
        assert_close(erfc(-INF), 2.0)

    def test_nan(self):
        assert np.isnan(erfc(NAN))


class TestLgamma:
    def test_typical(self):
        for x in [1.0, 2.0, 0.5, 10.0]:
            assert_close(lgamma(x), math.lgamma(x))

    def test_inf(self):
        assert lgamma(INF) == INF

    def test_nan(self):
        assert np.isnan(lgamma(NAN))


class TestTgamma:
    def test_typical(self):
        for x in [1.0, 2.0, 0.5, 5.0]:
            assert_close(tgamma(x), math.gamma(x))

    def test_inf(self):
        assert tgamma(INF) == INF

    def test_zero(self):
        assert np.isinf(tgamma(0.0))

    def test_nan(self):
        assert np.isnan(tgamma(NAN))


# --- Absolute value ---

class TestFabs:
    def test_typical(self):
        assert fabs(1.0) == 1.0
        assert fabs(-1.0) == 1.0
        assert fabs(0.0) == 0.0
        assert fabs(-0.0) == 0.0
        assert fabs(1e300) == 1e300
        assert fabs(-1e300) == 1e300

    def test_inf(self):
        assert fabs(INF) == INF
        assert fabs(-INF) == INF

    def test_nan(self):
        assert np.isnan(fabs(NAN))


# --- Two-argument: trig ---

class TestAtan2:
    def test_typical(self):
        for y, x in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (0.0, 1.0), (1.0, 0.0)]:
            assert_close(atan2(y, x), math.atan2(y, x))

    def test_zero(self):
        assert_close(atan2(0.0, 0.0), math.atan2(0.0, 0.0))
        assert_close(atan2(-0.0, 0.0), math.atan2(-0.0, 0.0))

    def test_inf(self):
        assert_close(atan2(1.0, INF), 0.0)
        assert_close(atan2(INF, 1.0), math.pi / 2)

    def test_nan(self):
        assert np.isnan(atan2(NAN, 1.0))
        assert np.isnan(atan2(1.0, NAN))


# --- Two-argument: power ---

class TestPow:
    def test_typical(self):
        for x, y in [(2.0, 3.0), (2.0, 0.5), (10.0, 2.0), (2.0, -1.0)]:
            assert_close(pow(x, y), math.pow(x, y))

    def test_zero_exponent(self):
        assert_close(pow(5.0, 0.0), 1.0)
        assert_close(pow(0.0, 0.0), 1.0)

    def test_one_base(self):
        assert_close(pow(1.0, 1e300), 1.0)

    def test_inf(self):
        assert pow(2.0, INF) == INF
        assert pow(2.0, -INF) == 0.0

    def test_nan(self):
        assert np.isnan(pow(NAN, 2.0))

    def test_negative_base_noninteger(self):
        assert np.isnan(pow(-1.0, 0.5))

    def test_zero_base_negative_exponent(self):
        assert pow(0.0, -1.0) == INF


# --- Two-argument: modular ---

class TestFmod:
    def test_typical(self):
        for x, y in [(5.0, 3.0), (-5.0, 3.0), (5.0, -3.0), (10.0, 2.5)]:
            assert_close(fmod(x, y), math.fmod(x, y))

    def test_zero_dividend(self):
        assert_close(fmod(0.0, 1.0), 0.0)

    def test_inf_dividend(self):
        assert np.isnan(fmod(INF, 1.0))

    def test_nan(self):
        assert np.isnan(fmod(NAN, 1.0))
        assert np.isnan(fmod(1.0, NAN))
        assert np.isnan(fmod(1.0, 0.0))


class TestRemainder:
    def test_typical(self):
        for x, y in [(5.0, 3.0), (-5.0, 3.0), (5.0, -3.0), (10.0, 3.0)]:
            assert_close(remainder(x, y), math.remainder(x, y))

    def test_zero_dividend(self):
        assert_close(remainder(0.0, 1.0), 0.0)

    def test_nan(self):
        assert np.isnan(remainder(NAN, 1.0))
        assert np.isnan(remainder(1.0, NAN))
        assert np.isnan(remainder(1.0, 0.0))


# --- Two-argument: geometry ---

class TestHypot:
    def test_typical(self):
        for x, y in [(3.0, 4.0), (1.0, 1.0), (0.0, 5.0), (5.0, 0.0)]:
            assert_close(hypot(x, y), math.hypot(x, y))

    def test_negative(self):
        assert_close(hypot(-3.0, -4.0), 5.0)

    def test_inf(self):
        assert hypot(INF, 1.0) == INF
        assert hypot(1.0, INF) == INF
        assert hypot(INF, NAN) == INF

    def test_nan(self):
        assert np.isnan(hypot(NAN, 1.0))
        assert np.isnan(hypot(1.0, NAN))


# --- Two-argument: comparison ---

class TestFmax:
    def test_typical(self):
        assert_close(fmax(2.0, 3.0), 3.0)
        assert_close(fmax(-1.0, -2.0), -1.0)
        assert_close(fmax(0.0, -0.0), 0.0)

    def test_nan_ignored(self):
        assert_close(fmax(NAN, 1.0), 1.0)
        assert_close(fmax(1.0, NAN), 1.0)

    def test_both_nan(self):
        assert np.isnan(fmax(NAN, NAN))

    def test_inf(self):
        assert fmax(INF, 1.0) == INF
        assert fmax(-INF, 1.0) == 1.0


class TestFmin:
    def test_typical(self):
        assert_close(fmin(2.0, 3.0), 2.0)
        assert_close(fmin(-1.0, -2.0), -2.0)

    def test_nan_ignored(self):
        assert_close(fmin(NAN, 1.0), 1.0)
        assert_close(fmin(1.0, NAN), 1.0)

    def test_both_nan(self):
        assert np.isnan(fmin(NAN, NAN))

    def test_inf(self):
        assert fmin(-INF, 1.0) == -INF
        assert fmin(INF, 1.0) == 1.0


class TestFdim:
    def test_typical(self):
        assert_close(fdim(5.0, 3.0), 2.0)
        assert_close(fdim(3.0, 5.0), 0.0)
        assert_close(fdim(0.0, 0.0), 0.0)
        assert_close(fdim(-1.0, -5.0), 4.0)

    def test_inf(self):
        assert fdim(INF, 1.0) == INF
        assert fdim(1.0, INF) == 0.0

    def test_nan(self):
        assert np.isnan(fdim(NAN, 1.0))
        assert np.isnan(fdim(1.0, NAN))


# --- Two-argument: utility ---

class TestCopysign:
    def test_typical(self):
        assert_close(copysign(1.0, -1.0), -1.0)
        assert_close(copysign(-1.0, 1.0), 1.0)
        assert_close(copysign(5.0, -0.0), -5.0)
        assert_close(copysign(-5.0, 0.0), 5.0)

    def test_zero_sign(self):
        assert math.copysign(1.0, copysign(0.0, -1.0)) == -1.0  # result is -0.0

    def test_inf(self):
        assert copysign(INF, -1.0) == -INF
        assert copysign(-INF, 1.0) == INF

    def test_nan(self):
        assert np.isnan(copysign(NAN, 1.0))
        assert np.isnan(copysign(NAN, -1.0))
