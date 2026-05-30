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

load_lib("m")


@proxy(signatures.get("cos"), jit_options=jit_options)
def cos(x):
    return _call_lib_func("cos", (x,))


@proxy(signatures.get("sin"), jit_options=jit_options)
def sin(x):
    return _call_lib_func("sin", (x,))


@proxy(signatures.get("tan"), jit_options=jit_options)
def tan(x):
    return _call_lib_func("tan", (x,))


@proxy(signatures.get("acos"), jit_options=jit_options)
def acos(x):
    return _call_lib_func("acos", (x,))


@proxy(signatures.get("asin"), jit_options=jit_options)
def asin(x):
    return _call_lib_func("asin", (x,))


@proxy(signatures.get("atan"), jit_options=jit_options)
def atan(x):
    return _call_lib_func("atan", (x,))


@proxy(signatures.get("cosh"), jit_options=jit_options)
def cosh(x):
    return _call_lib_func("cosh", (x,))


@proxy(signatures.get("sinh"), jit_options=jit_options)
def sinh(x):
    return _call_lib_func("sinh", (x,))


@proxy(signatures.get("tanh"), jit_options=jit_options)
def tanh(x):
    return _call_lib_func("tanh", (x,))


@proxy(signatures.get("acosh"), jit_options=jit_options)
def acosh(x):
    return _call_lib_func("acosh", (x,))


@proxy(signatures.get("asinh"), jit_options=jit_options)
def asinh(x):
    return _call_lib_func("asinh", (x,))


@proxy(signatures.get("atanh"), jit_options=jit_options)
def atanh(x):
    return _call_lib_func("atanh", (x,))


@proxy(signatures.get("exp"), jit_options=jit_options)
def exp(x):
    return _call_lib_func("exp", (x,))


@proxy(signatures.get("exp2"), jit_options=jit_options)
def exp2(x):
    return _call_lib_func("exp2", (x,))


@proxy(signatures.get("expm1"), jit_options=jit_options)
def expm1(x):
    return _call_lib_func("expm1", (x,))


@proxy(signatures.get("log"), jit_options=jit_options)
def log(x):
    return _call_lib_func("log", (x,))


@proxy(signatures.get("log2"), jit_options=jit_options)
def log2(x):
    return _call_lib_func("log2", (x,))


@proxy(signatures.get("log10"), jit_options=jit_options)
def log10(x):
    return _call_lib_func("log10", (x,))


@proxy(signatures.get("log1p"), jit_options=jit_options)
def log1p(x):
    return _call_lib_func("log1p", (x,))


@proxy(signatures.get("logb"), jit_options=jit_options)
def logb(x):
    return _call_lib_func("logb", (x,))


@proxy(signatures.get("sqrt"), jit_options=jit_options)
def sqrt(x):
    return _call_lib_func("sqrt", (x,))


@proxy(signatures.get("cbrt"), jit_options=jit_options)
def cbrt(x):
    return _call_lib_func("cbrt", (x,))


@proxy(signatures.get("ceil"), jit_options=jit_options)
def ceil(x):
    return _call_lib_func("ceil", (x,))


@proxy(signatures.get("floor"), jit_options=jit_options)
def floor(x):
    return _call_lib_func("floor", (x,))


@proxy(signatures.get("trunc"), jit_options=jit_options)
def trunc(x):
    return _call_lib_func("trunc", (x,))


@proxy(signatures.get("round"), jit_options=jit_options)
def round(x):
    return _call_lib_func("round", (x,))


@proxy(signatures.get("rint"), jit_options=jit_options)
def rint(x):
    return _call_lib_func("rint", (x,))


@proxy(signatures.get("nearbyint"), jit_options=jit_options)
def nearbyint(x):
    return _call_lib_func("nearbyint", (x,))


@proxy(signatures.get("erf"), jit_options=jit_options)
def erf(x):
    return _call_lib_func("erf", (x,))


@proxy(signatures.get("erfc"), jit_options=jit_options)
def erfc(x):
    return _call_lib_func("erfc", (x,))


@proxy(signatures.get("lgamma"), jit_options=jit_options)
def lgamma(x):
    return _call_lib_func("lgamma", (x,))


@proxy(signatures.get("tgamma"), jit_options=jit_options)
def tgamma(x):
    return _call_lib_func("tgamma", (x,))


@proxy(signatures.get("fabs"), jit_options=jit_options)
def fabs(x):
    return _call_lib_func("fabs", (x,))


@proxy(signatures.get("atan2"), jit_options=jit_options)
def atan2(y, x):
    return _call_lib_func("atan2", (y, x))


@proxy(signatures.get("pow"), jit_options=jit_options)
def pow(x, y):
    return _call_lib_func("pow", (x, y))


@proxy(signatures.get("fmod"), jit_options=jit_options)
def fmod(x, y):
    return _call_lib_func("fmod", (x, y))


@proxy(signatures.get("remainder"), jit_options=jit_options)
def remainder(x, y):
    return _call_lib_func("remainder", (x, y))


@proxy(signatures.get("hypot"), jit_options=jit_options)
def hypot(x, y):
    return _call_lib_func("hypot", (x, y))


@proxy(signatures.get("fmax"), jit_options=jit_options)
def fmax(x, y):
    return _call_lib_func("fmax", (x, y))


@proxy(signatures.get("fmin"), jit_options=jit_options)
def fmin(x, y):
    return _call_lib_func("fmin", (x, y))


@proxy(signatures.get("fdim"), jit_options=jit_options)
def fdim(x, y):
    return _call_lib_func("fdim", (x, y))


@proxy(signatures.get("copysign"), jit_options=jit_options)
def copysign(x, y):
    return _call_lib_func("copysign", (x, y))
