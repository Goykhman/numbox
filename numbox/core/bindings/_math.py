from numbox.utils.highlevel import cres
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib

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


@cres(signatures.get("cos"), cache=True)
def cos(x):
    return _call_lib_func("cos", (x,))


@cres(signatures.get("sin"), cache=True)
def sin(x):
    return _call_lib_func("sin", (x,))


@cres(signatures.get("tan"), cache=True)
def tan(x):
    return _call_lib_func("tan", (x,))


@cres(signatures.get("acos"), cache=True)
def acos(x):
    return _call_lib_func("acos", (x,))


@cres(signatures.get("asin"), cache=True)
def asin(x):
    return _call_lib_func("asin", (x,))


@cres(signatures.get("atan"), cache=True)
def atan(x):
    return _call_lib_func("atan", (x,))


@cres(signatures.get("cosh"), cache=True)
def cosh(x):
    return _call_lib_func("cosh", (x,))


@cres(signatures.get("sinh"), cache=True)
def sinh(x):
    return _call_lib_func("sinh", (x,))


@cres(signatures.get("tanh"), cache=True)
def tanh(x):
    return _call_lib_func("tanh", (x,))


@cres(signatures.get("acosh"), cache=True)
def acosh(x):
    return _call_lib_func("acosh", (x,))


@cres(signatures.get("asinh"), cache=True)
def asinh(x):
    return _call_lib_func("asinh", (x,))


@cres(signatures.get("atanh"), cache=True)
def atanh(x):
    return _call_lib_func("atanh", (x,))


@cres(signatures.get("exp"), cache=True)
def exp(x):
    return _call_lib_func("exp", (x,))


@cres(signatures.get("exp2"), cache=True)
def exp2(x):
    return _call_lib_func("exp2", (x,))


@cres(signatures.get("expm1"), cache=True)
def expm1(x):
    return _call_lib_func("expm1", (x,))


@cres(signatures.get("log"), cache=True)
def log(x):
    return _call_lib_func("log", (x,))


@cres(signatures.get("log2"), cache=True)
def log2(x):
    return _call_lib_func("log2", (x,))


@cres(signatures.get("log10"), cache=True)
def log10(x):
    return _call_lib_func("log10", (x,))


@cres(signatures.get("log1p"), cache=True)
def log1p(x):
    return _call_lib_func("log1p", (x,))


@cres(signatures.get("logb"), cache=True)
def logb(x):
    return _call_lib_func("logb", (x,))


@cres(signatures.get("sqrt"), cache=True)
def sqrt(x):
    return _call_lib_func("sqrt", (x,))


@cres(signatures.get("cbrt"), cache=True)
def cbrt(x):
    return _call_lib_func("cbrt", (x,))


@cres(signatures.get("ceil"), cache=True)
def ceil(x):
    return _call_lib_func("ceil", (x,))


@cres(signatures.get("floor"), cache=True)
def floor(x):
    return _call_lib_func("floor", (x,))


@cres(signatures.get("trunc"), cache=True)
def trunc(x):
    return _call_lib_func("trunc", (x,))


@cres(signatures.get("round"), cache=True)
def round(x):
    return _call_lib_func("round", (x,))


@cres(signatures.get("rint"), cache=True)
def rint(x):
    return _call_lib_func("rint", (x,))


@cres(signatures.get("nearbyint"), cache=True)
def nearbyint(x):
    return _call_lib_func("nearbyint", (x,))


@cres(signatures.get("erf"), cache=True)
def erf(x):
    return _call_lib_func("erf", (x,))


@cres(signatures.get("erfc"), cache=True)
def erfc(x):
    return _call_lib_func("erfc", (x,))


@cres(signatures.get("lgamma"), cache=True)
def lgamma(x):
    return _call_lib_func("lgamma", (x,))


@cres(signatures.get("tgamma"), cache=True)
def tgamma(x):
    return _call_lib_func("tgamma", (x,))


@cres(signatures.get("fabs"), cache=True)
def fabs(x):
    return _call_lib_func("fabs", (x,))


@cres(signatures.get("atan2"), cache=True)
def atan2(y, x):
    return _call_lib_func("atan2", (y, x))


@cres(signatures.get("pow"), cache=True)
def pow(x, y):
    return _call_lib_func("pow", (x, y))


@cres(signatures.get("fmod"), cache=True)
def fmod(x, y):
    return _call_lib_func("fmod", (x, y))


@cres(signatures.get("remainder"), cache=True)
def remainder(x, y):
    return _call_lib_func("remainder", (x, y))


@cres(signatures.get("hypot"), cache=True)
def hypot(x, y):
    return _call_lib_func("hypot", (x, y))


@cres(signatures.get("fmax"), cache=True)
def fmax(x, y):
    return _call_lib_func("fmax", (x, y))


@cres(signatures.get("fmin"), cache=True)
def fmin(x, y):
    return _call_lib_func("fmin", (x, y))


@cres(signatures.get("fdim"), cache=True)
def fdim(x, y):
    return _call_lib_func("fdim", (x, y))


@cres(signatures.get("copysign"), cache=True)
def copysign(x, y):
    return _call_lib_func("copysign", (x, y))
