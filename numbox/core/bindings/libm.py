"""Bindings for the functions of the C math library, on ``float64``.

Each binding calls the ``double`` form of the C function of its name, so it takes and returns ``float64``, and numba
converts an integer or ``float32`` argument without a warning. The bindings behave as C's functions, not Python's. C
returns NaN or an infinity where Python's ``math`` module raises ``ValueError`` or ``OverflowError``, as for
``sqrt(-1.0)``, ``log(0.0)`` or ``exp(1000.0)``, and ``round`` and ``pow`` differ from Python's built-ins. Import the
module, as in ``from numbox.core.bindings import libm``, and call ``libm.round``, so that the name ``round`` still
refers to Python's built-in function. numba registers 26 of these names, among them ``cos``, ``pow`` and ``sqrt``, for
its own math support, and LLVM finds a registered name before it searches any library. On Windows, those 26 bindings
therefore call the functions of CPython's C runtime instead of those of the ``msvcrt.dll`` that this module loads for
``"m"``.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy


__all__ = [
    "cos", "sin", "tan",
    "acos", "asin", "atan",
    "cosh", "sinh", "tanh", "acosh", "asinh", "atanh",
    "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "logb",
    "sqrt", "cbrt",
    "ceil", "floor", "trunc", "round", "rint", "nearbyint",
    "erf", "erfc", "lgamma", "tgamma",
    "fabs",
    "atan2",
    "pow", "fmod", "remainder",
    "hypot",
    "fmax", "fmin", "fdim",
    "copysign",
]

# numba registers the names in numba.core.intrinsics.INTR_MATH, 26 of the 42 bound below, with ll.add_symbol
# (numba/core/externals.py), and LLVM resolves a registered name before it searches the loaded libraries. On Linux the
# registered addresses point into the same libm that this call loads. On Windows, they point into CPython's C runtime,
# so LLVM resolves none of those 26 names to the msvcrt.dll that this call loads.
load_lib("m")


@proxy(signatures.get("cos"), jit_options=jit_options)
def cos(x):
    """Return the cosine of ``x``, an angle in radians.

    C reference: `cos(3) <https://man7.org/linux/man-pages/man3/cos.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning. On Windows, the binding calls the function in CPython's C runtime, not the
    one in ``msvcrt.dll``, because numba registers the name ``cos`` for its own math support.
    """
    return _call_lib_func("cos", (x,))


@proxy(signatures.get("sin"), jit_options=jit_options)
def sin(x):
    """Return the sine of ``x``, an angle in radians.

    C reference: `sin(3) <https://man7.org/linux/man-pages/man3/sin.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning. On Windows, the binding calls the function in CPython's C runtime, not the
    one in ``msvcrt.dll``, because numba registers the name ``sin`` for its own math support.
    """
    return _call_lib_func("sin", (x,))


@proxy(signatures.get("tan"), jit_options=jit_options)
def tan(x):
    """Return the tangent of ``x``, an angle in radians.

    C reference: `tan(3) <https://man7.org/linux/man-pages/man3/tan.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning. On Windows, the binding calls the function in CPython's C runtime, not the
    one in ``msvcrt.dll``, because numba registers the name ``tan`` for its own math support.
    """
    return _call_lib_func("tan", (x,))


@proxy(signatures.get("acos"), jit_options=jit_options)
def acos(x):
    """Return the arc cosine of ``x`` in radians, from 0 to pi.

    C reference: `acos(3) <https://man7.org/linux/man-pages/man3/acos.3.html>`_.

    An ``x`` outside [-1, 1] gives NaN. The binding calls C's ``double`` form, so it takes and returns ``float64``, and
    numba converts an integer or ``float32`` argument without a warning. On Windows, the binding calls the function in
    CPython's C runtime, not the one in ``msvcrt.dll``, because numba registers the name ``acos`` for its own math
    support.
    """
    return _call_lib_func("acos", (x,))


@proxy(signatures.get("asin"), jit_options=jit_options)
def asin(x):
    """Return the arc sine of ``x`` in radians, from -pi/2 to pi/2.

    C reference: `asin(3) <https://man7.org/linux/man-pages/man3/asin.3.html>`_.

    An ``x`` outside [-1, 1] gives NaN. The binding calls C's ``double`` form, so it takes and returns ``float64``, and
    numba converts an integer or ``float32`` argument without a warning. On Windows, the binding calls the function in
    CPython's C runtime, not the one in ``msvcrt.dll``, because numba registers the name ``asin`` for its own math
    support.
    """
    return _call_lib_func("asin", (x,))


@proxy(signatures.get("atan"), jit_options=jit_options)
def atan(x):
    """Return the arc tangent of ``x`` in radians, from -pi/2 to pi/2.

    C reference: `atan(3) <https://man7.org/linux/man-pages/man3/atan.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning. On Windows, the binding calls the function in CPython's C runtime, not the
    one in ``msvcrt.dll``, because numba registers the name ``atan`` for its own math support.
    """
    return _call_lib_func("atan", (x,))


@proxy(signatures.get("cosh"), jit_options=jit_options)
def cosh(x):
    """Return the hyperbolic cosine of ``x``.

    C reference: `cosh(3) <https://man7.org/linux/man-pages/man3/cosh.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning. On Windows, the binding calls the function in CPython's C runtime, not the
    one in ``msvcrt.dll``, because numba registers the name ``cosh`` for its own math support.
    """
    return _call_lib_func("cosh", (x,))


@proxy(signatures.get("sinh"), jit_options=jit_options)
def sinh(x):
    """Return the hyperbolic sine of ``x``.

    C reference: `sinh(3) <https://man7.org/linux/man-pages/man3/sinh.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning. On Windows, the binding calls the function in CPython's C runtime, not the
    one in ``msvcrt.dll``, because numba registers the name ``sinh`` for its own math support.
    """
    return _call_lib_func("sinh", (x,))


@proxy(signatures.get("tanh"), jit_options=jit_options)
def tanh(x):
    """Return the hyperbolic tangent of ``x``.

    C reference: `tanh(3) <https://man7.org/linux/man-pages/man3/tanh.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning. On Windows, the binding calls the function in CPython's C runtime, not the
    one in ``msvcrt.dll``, because numba registers the name ``tanh`` for its own math support.
    """
    return _call_lib_func("tanh", (x,))


@proxy(signatures.get("acosh"), jit_options=jit_options)
def acosh(x):
    """Return the inverse hyperbolic cosine of ``x``, a value of 0 or more.

    C reference: `acosh(3) <https://man7.org/linux/man-pages/man3/acosh.3.html>`_.

    An ``x`` below 1 gives NaN. The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba
    converts an integer or ``float32`` argument without a warning. On Windows, the binding calls the function in
    CPython's C runtime, not the one in ``msvcrt.dll``, because numba registers the name ``acosh`` for its own math
    support.
    """
    return _call_lib_func("acosh", (x,))


@proxy(signatures.get("asinh"), jit_options=jit_options)
def asinh(x):
    """Return the inverse hyperbolic sine of ``x``.

    C reference: `asinh(3) <https://man7.org/linux/man-pages/man3/asinh.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning. On Windows, the binding calls the function in CPython's C runtime, not the
    one in ``msvcrt.dll``, because numba registers the name ``asinh`` for its own math support.
    """
    return _call_lib_func("asinh", (x,))


@proxy(signatures.get("atanh"), jit_options=jit_options)
def atanh(x):
    """Return the inverse hyperbolic tangent of ``x``.

    C reference: `atanh(3) <https://man7.org/linux/man-pages/man3/atanh.3.html>`_.

    An ``x`` of 1 or -1 gives an infinity of the same sign, and an ``x`` outside [-1, 1] gives NaN. The binding calls
    C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or ``float32`` argument
    without a warning. On Windows, the binding calls the function in CPython's C runtime, not the one in ``msvcrt.dll``,
    because numba registers the name ``atanh`` for its own math support.
    """
    return _call_lib_func("atanh", (x,))


@proxy(signatures.get("exp"), jit_options=jit_options)
def exp(x):
    """Return e raised to the power ``x``.

    C reference: `exp(3) <https://man7.org/linux/man-pages/man3/exp.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning. On Windows, the binding calls the function in CPython's C runtime, not the
    one in ``msvcrt.dll``, because numba registers the name ``exp`` for its own math support.
    """
    return _call_lib_func("exp", (x,))


@proxy(signatures.get("exp2"), jit_options=jit_options)
def exp2(x):
    """Return 2 raised to the power ``x``.

    C reference: `exp2(3) <https://man7.org/linux/man-pages/man3/exp2.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning.
    """
    return _call_lib_func("exp2", (x,))


@proxy(signatures.get("expm1"), jit_options=jit_options)
def expm1(x):
    """Return e raised to the power ``x``, minus 1, computed accurately when ``x`` is near 0.

    C reference: `expm1(3) <https://man7.org/linux/man-pages/man3/expm1.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning. On Windows, the binding calls the function in CPython's C runtime, not the
    one in ``msvcrt.dll``, because numba registers the name ``expm1`` for its own math support.
    """
    return _call_lib_func("expm1", (x,))


@proxy(signatures.get("log"), jit_options=jit_options)
def log(x):
    """Return the natural logarithm of ``x``.

    C reference: `log(3) <https://man7.org/linux/man-pages/man3/log.3.html>`_.

    An ``x`` of 0 gives negative infinity, and a negative ``x`` gives NaN. The binding calls C's ``double`` form, so it
    takes and returns ``float64``, and numba converts an integer or ``float32`` argument without a warning. On Windows,
    the binding calls the function in CPython's C runtime, not the one in ``msvcrt.dll``, because numba registers the
    name ``log`` for its own math support.
    """
    return _call_lib_func("log", (x,))


@proxy(signatures.get("log2"), jit_options=jit_options)
def log2(x):
    """Return the base-2 logarithm of ``x``.

    C reference: `log2(3) <https://man7.org/linux/man-pages/man3/log2.3.html>`_.

    An ``x`` of 0 gives negative infinity, and a negative ``x`` gives NaN. The binding calls C's ``double`` form, so it
    takes and returns ``float64``, and numba converts an integer or ``float32`` argument without a warning.
    """
    return _call_lib_func("log2", (x,))


@proxy(signatures.get("log10"), jit_options=jit_options)
def log10(x):
    """Return the base-10 logarithm of ``x``.

    C reference: `log10(3) <https://man7.org/linux/man-pages/man3/log10.3.html>`_.

    An ``x`` of 0 gives negative infinity, and a negative ``x`` gives NaN. The binding calls C's ``double`` form, so it
    takes and returns ``float64``, and numba converts an integer or ``float32`` argument without a warning. On Windows,
    the binding calls the function in CPython's C runtime, not the one in ``msvcrt.dll``, because numba registers the
    name ``log10`` for its own math support.
    """
    return _call_lib_func("log10", (x,))


@proxy(signatures.get("log1p"), jit_options=jit_options)
def log1p(x):
    """Return the natural logarithm of ``1 + x``, computed accurately when ``x`` is near 0.

    C reference: `log1p(3) <https://man7.org/linux/man-pages/man3/log1p.3.html>`_.

    An ``x`` of -1 gives negative infinity, and an ``x`` below -1 gives NaN. The binding calls C's ``double`` form, so
    it takes and returns ``float64``, and numba converts an integer or ``float32`` argument without a warning. On
    Windows, the binding calls the function in CPython's C runtime, not the one in ``msvcrt.dll``, because numba
    registers the name ``log1p`` for its own math support.
    """
    return _call_lib_func("log1p", (x,))


@proxy(signatures.get("logb"), jit_options=jit_options)
def logb(x):
    """Return the base-2 exponent of ``x`` as a float: ``floor(log2(abs(x)))`` for a finite ``x`` other than 0.

    C reference: `logb(3) <https://man7.org/linux/man-pages/man3/logb.3.html>`_.

    An ``x`` of 0 gives negative infinity. The binding calls C's ``double`` form, so it takes and returns ``float64``,
    and numba converts an integer or ``float32`` argument without a warning.
    """
    return _call_lib_func("logb", (x,))


@proxy(signatures.get("sqrt"), jit_options=jit_options)
def sqrt(x):
    """Return the square root of ``x``.

    C reference: `sqrt(3) <https://man7.org/linux/man-pages/man3/sqrt.3.html>`_.

    A negative ``x`` gives NaN. The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba
    converts an integer or ``float32`` argument without a warning. On Windows, the binding calls the function in
    CPython's C runtime, not the one in ``msvcrt.dll``, because numba registers the name ``sqrt`` for its own math
    support.
    """
    return _call_lib_func("sqrt", (x,))


@proxy(signatures.get("cbrt"), jit_options=jit_options)
def cbrt(x):
    """Return the cube root of ``x``, which is negative for a negative ``x``.

    C reference: `cbrt(3) <https://man7.org/linux/man-pages/man3/cbrt.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning.
    """
    return _call_lib_func("cbrt", (x,))


@proxy(signatures.get("ceil"), jit_options=jit_options)
def ceil(x):
    """Return the smallest integral value that is not less than ``x``, as a float.

    C reference: `ceil(3) <https://man7.org/linux/man-pages/man3/ceil.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning. On Windows, the binding calls the function in CPython's C runtime, not the
    one in ``msvcrt.dll``, because numba registers the name ``ceil`` for its own math support.
    """
    return _call_lib_func("ceil", (x,))


@proxy(signatures.get("floor"), jit_options=jit_options)
def floor(x):
    """Return the largest integral value that is not greater than ``x``, as a float.

    C reference: `floor(3) <https://man7.org/linux/man-pages/man3/floor.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning. On Windows, the binding calls the function in CPython's C runtime, not the
    one in ``msvcrt.dll``, because numba registers the name ``floor`` for its own math support.
    """
    return _call_lib_func("floor", (x,))


@proxy(signatures.get("trunc"), jit_options=jit_options)
def trunc(x):
    """Return ``x`` rounded toward zero to an integral value, as a float.

    C reference: `trunc(3) <https://man7.org/linux/man-pages/man3/trunc.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning. On Windows, the binding calls the function in CPython's C runtime, not the
    one in ``msvcrt.dll``, because numba registers the name ``trunc`` for its own math support.
    """
    return _call_lib_func("trunc", (x,))


@proxy(signatures.get("round"), jit_options=jit_options)
def round(x):
    """Return ``x`` rounded to the nearest integral value, with halfway cases away from zero, as a float.

    C reference: `round(3) <https://man7.org/linux/man-pages/man3/round.3.html>`_.

    This is C's ``round``, not Python's: ``round(2.5)`` is 3.0 here and 2 in Python. The binding calls C's ``double``
    form, so it takes and returns ``float64``, and numba converts an integer or ``float32`` argument without a warning.
    On Windows, the binding calls the function in CPython's C runtime, not the one in ``msvcrt.dll``, because numba
    registers the name ``round`` for its own math support.
    """
    return _call_lib_func("round", (x,))


@proxy(signatures.get("rint"), jit_options=jit_options)
def rint(x):
    """Return ``x`` rounded to an integral value in the current rounding mode, as a float.

    C reference: `rint(3) <https://man7.org/linux/man-pages/man3/rint.3.html>`_.

    In the default mode, which rounds to nearest, a halfway case goes to the even value: ``rint(2.5)`` is 2.0. The
    binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or ``float32``
    argument without a warning.
    """
    return _call_lib_func("rint", (x,))


@proxy(signatures.get("nearbyint"), jit_options=jit_options)
def nearbyint(x):
    """Return ``x`` rounded to an integral value in the current rounding mode, as a float.

    C reference: `nearbyint(3) <https://man7.org/linux/man-pages/man3/nearbyint.3.html>`_.

    It differs from ``rint`` only in that it does not set C's inexact floating-point exception flag. In the default
    mode, a halfway case goes to the even value. The binding calls C's ``double`` form, so it takes and returns
    ``float64``, and numba converts an integer or ``float32`` argument without a warning.
    """
    return _call_lib_func("nearbyint", (x,))


@proxy(signatures.get("erf"), jit_options=jit_options)
def erf(x):
    """Return the error function of ``x``.

    C reference: `erf(3) <https://man7.org/linux/man-pages/man3/erf.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning.
    """
    return _call_lib_func("erf", (x,))


@proxy(signatures.get("erfc"), jit_options=jit_options)
def erfc(x):
    """Return the complementary error function of ``x``, ``1 - erf(x)``, computed accurately for large ``x``.

    C reference: `erfc(3) <https://man7.org/linux/man-pages/man3/erfc.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning.
    """
    return _call_lib_func("erfc", (x,))


@proxy(signatures.get("lgamma"), jit_options=jit_options)
def lgamma(x):
    """Return the natural logarithm of the absolute value of the gamma function at ``x``.

    C reference: `lgamma(3) <https://man7.org/linux/man-pages/man3/lgamma.3.html>`_.

    The sign of the gamma function is not returned: C stores it in the global variable ``signgam``, which numbox does
    not read. Because of that global variable, C does not require ``lgamma`` to be thread-safe. The binding calls C's
    ``double`` form, so it takes and returns ``float64``, and numba converts an integer or ``float32`` argument without
    a warning.
    """
    return _call_lib_func("lgamma", (x,))


@proxy(signatures.get("tgamma"), jit_options=jit_options)
def tgamma(x):
    """Return the gamma function at ``x``.

    C reference: `tgamma(3) <https://man7.org/linux/man-pages/man3/tgamma.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning.
    """
    return _call_lib_func("tgamma", (x,))


@proxy(signatures.get("fabs"), jit_options=jit_options)
def fabs(x):
    """Return the absolute value of ``x``.

    C reference: `fabs(3) <https://man7.org/linux/man-pages/man3/fabs.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning. On Windows, the binding calls the function in CPython's C runtime, not the
    one in ``msvcrt.dll``, because numba registers the name ``fabs`` for its own math support.
    """
    return _call_lib_func("fabs", (x,))


@proxy(signatures.get("atan2"), jit_options=jit_options)
def atan2(y, x):
    """Return the arc tangent of ``y / x`` in radians, in the quadrant that the signs of ``x`` and ``y`` determine.

    C reference: `atan2(3) <https://man7.org/linux/man-pages/man3/atan2.3.html>`_.

    The result is in the range from -pi to pi, and the first argument is ``y``, as in C. The binding calls C's
    ``double`` form, so it takes and returns ``float64``, and numba converts an integer or ``float32`` argument without
    a warning. On Windows, the binding calls the function in CPython's C runtime, not the one in ``msvcrt.dll``, because
    numba registers the name ``atan2`` for its own math support.
    """
    return _call_lib_func("atan2", (y, x))


@proxy(signatures.get("pow"), jit_options=jit_options)
def pow(x, y):
    """Return ``x`` raised to the power ``y``.

    C reference: `pow(3) <https://man7.org/linux/man-pages/man3/pow.3.html>`_.

    This is C's ``pow``, not Python's: a negative ``x`` with a non-integral ``y`` gives NaN, and 0 with a negative ``y``
    gives infinity, where Python's ``**`` gives a complex number or raises ``ZeroDivisionError``. The binding calls C's
    ``double`` form, so it takes and returns ``float64``, and numba converts an integer or ``float32`` argument without
    a warning. On Windows, the binding calls the function in CPython's C runtime, not the one in ``msvcrt.dll``, because
    numba registers the name ``pow`` for its own math support.
    """
    return _call_lib_func("pow", (x, y))


@proxy(signatures.get("fmod"), jit_options=jit_options)
def fmod(x, y):
    """Return ``x - n * y``, where ``n`` is ``x / y`` rounded toward zero, so the result has the sign of ``x``.

    C reference: `fmod(3) <https://man7.org/linux/man-pages/man3/fmod.3.html>`_.

    Python's ``%`` gives the result the sign of ``y`` instead. A ``y`` of 0 gives NaN. The binding calls C's ``double``
    form, so it takes and returns ``float64``, and numba converts an integer or ``float32`` argument without a warning.
    On Windows, the binding calls the function in CPython's C runtime, not the one in ``msvcrt.dll``, because numba
    registers the name ``fmod`` for its own math support.
    """
    return _call_lib_func("fmod", (x, y))


@proxy(signatures.get("remainder"), jit_options=jit_options)
def remainder(x, y):
    """Return ``x - n * y``, where ``n`` is ``x / y`` rounded to the nearest integer, halfway cases to even.

    C reference: `remainder(3) <https://man7.org/linux/man-pages/man3/remainder.3.html>`_.

    This is the IEEE 754 remainder: ``remainder(7.0, 2.0)`` is -1.0. A ``y`` of 0 gives NaN. The binding calls C's
    ``double`` form, so it takes and returns ``float64``, and numba converts an integer or ``float32`` argument without
    a warning.
    """
    return _call_lib_func("remainder", (x, y))


@proxy(signatures.get("hypot"), jit_options=jit_options)
def hypot(x, y):
    """Return the square root of ``x * x + y * y``, without overflow or underflow in the intermediate steps.

    C reference: `hypot(3) <https://man7.org/linux/man-pages/man3/hypot.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning.
    """
    return _call_lib_func("hypot", (x, y))


@proxy(signatures.get("fmax"), jit_options=jit_options)
def fmax(x, y):
    """Return the larger of ``x`` and ``y``; if one of them is NaN, return the other.

    C reference: `fmax(3) <https://man7.org/linux/man-pages/man3/fmax.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning.
    """
    return _call_lib_func("fmax", (x, y))


@proxy(signatures.get("fmin"), jit_options=jit_options)
def fmin(x, y):
    """Return the smaller of ``x`` and ``y``; if one of them is NaN, return the other.

    C reference: `fmin(3) <https://man7.org/linux/man-pages/man3/fmin.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning.
    """
    return _call_lib_func("fmin", (x, y))


@proxy(signatures.get("fdim"), jit_options=jit_options)
def fdim(x, y):
    """Return ``x - y`` if ``x`` is greater than ``y``, and 0 otherwise.

    C reference: `fdim(3) <https://man7.org/linux/man-pages/man3/fdim.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning.
    """
    return _call_lib_func("fdim", (x, y))


@proxy(signatures.get("copysign"), jit_options=jit_options)
def copysign(x, y):
    """Return a value with the magnitude of ``x`` and the sign of ``y``.

    C reference: `copysign(3) <https://man7.org/linux/man-pages/man3/copysign.3.html>`_.

    The binding calls C's ``double`` form, so it takes and returns ``float64``, and numba converts an integer or
    ``float32`` argument without a warning.
    """
    return _call_lib_func("copysign", (x, y))
