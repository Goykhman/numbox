from numba.core.types import (
    float64, int32, intp, void
)


signatures_c = {
    "rand": int32(),
    "srand": void(int32),
    "strlen": intp(intp),
}

signatures_m = {
    "cos": float64(float64),
    "sin": float64(float64),
    "tan": float64(float64),
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
