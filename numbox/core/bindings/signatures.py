from numba.core.types import (
    float64, int32, intp, void
)


signatures_c = {
    "rand": int32(),
    "srand": void(int32),
    "strlen": intp(intp),
}

signatures_m = {
    # Trig
    "cos": float64(float64),
    "sin": float64(float64),
    "tan": float64(float64),
    # Inverse trig
    "acos": float64(float64),
    "asin": float64(float64),
    "atan": float64(float64),
    # Hyperbolic
    "cosh": float64(float64),
    "sinh": float64(float64),
    "tanh": float64(float64),
    "acosh": float64(float64),
    "asinh": float64(float64),
    "atanh": float64(float64),
    # Exponential/log
    "exp": float64(float64),
    "exp2": float64(float64),
    "expm1": float64(float64),
    "log": float64(float64),
    "log2": float64(float64),
    "log10": float64(float64),
    "log1p": float64(float64),
    "logb": float64(float64),
    # Power/root
    "sqrt": float64(float64),
    "cbrt": float64(float64),
    # Rounding
    "ceil": float64(float64),
    "floor": float64(float64),
    "trunc": float64(float64),
    "round": float64(float64),
    "rint": float64(float64),
    "nearbyint": float64(float64),
    # Error/gamma
    "erf": float64(float64),
    "erfc": float64(float64),
    "lgamma": float64(float64),
    "tgamma": float64(float64),
    # Absolute value
    "fabs": float64(float64),
}

signatures_sqlite = {
    "sqlite3_close": int32(intp),
    "sqlite3_libversion": intp(),
    "sqlite3_open": int32(intp, intp),
}

signatures = {
    **signatures_c,
    **signatures_m,
    **signatures_sqlite
}
