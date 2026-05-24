from numba.core.types import (
    Tuple, float64, int32, int64, intp, void
)


lldiv_t = Tuple((int64, int64))


signatures_c = {
    "rand": int32(),
    "srand": void(int32),
    "strlen": intp(intp),
    "lldiv": lldiv_t(int64, int64),
    # === stdio (non-variadic) ===
    "puts": int32(intp),
    "fputs": int32(intp, intp),
    "fputc": int32(int32, intp),
    "putchar": int32(int32),
    "fwrite": intp(intp, intp, intp, intp),
    "fread": intp(intp, intp, intp, intp),
    "fflush": int32(intp),
    "fopen": intp(intp, intp),
    "fclose": int32(intp),
    "feof": int32(intp),
    "ferror": int32(intp),
    "clearerr": void(intp),
    # === strings ===
    "strcmp": int32(intp, intp),
    "strncmp": int32(intp, intp, intp),
    "strchr": intp(intp, int32),
    "strrchr": intp(intp, int32),
    "strstr": intp(intp, intp),
    "strncpy": intp(intp, intp, intp),
    "strerror": intp(int32),
    # === memory ===
    "memcpy": intp(intp, intp, intp),
    "memmove": intp(intp, intp, intp),
    "memset": intp(intp, int32, intp),
    "memcmp": int32(intp, intp, intp),
    "memchr": intp(intp, int32, intp),
    # === env ===
    "getenv": intp(intp),
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
    # Two-argument: trig
    "atan2": float64(float64, float64),
    # Two-argument: power
    "pow": float64(float64, float64),
    # Two-argument: modular
    "fmod": float64(float64, float64),
    "remainder": float64(float64, float64),
    # Two-argument: geometry
    "hypot": float64(float64, float64),
    # Two-argument: comparison
    "fmax": float64(float64, float64),
    "fmin": float64(float64, float64),
    "fdim": float64(float64, float64),
    # Two-argument: utility
    "copysign": float64(float64, float64),
}

signatures_sqlite = {
    # === Connection ===
    "sqlite3_open": int32(intp, intp),
    "sqlite3_open_v2": int32(intp, intp, int32, intp),
    "sqlite3_close": int32(intp),
    "sqlite3_libversion": intp(),
    "sqlite3_libversion_number": int32(),
    "sqlite3_errmsg": intp(intp),
    "sqlite3_errcode": int32(intp),
    "sqlite3_extended_errcode": int32(intp),
    "sqlite3_threadsafe": int32(),
    "sqlite3_db_handle": intp(intp),
    "sqlite3_db_filename": intp(intp, intp),
    "sqlite3_db_readonly": int32(intp, intp),
    "sqlite3_changes": int32(intp),
    "sqlite3_last_insert_rowid": int64(intp),
    "sqlite3_total_changes": int32(intp),
    # 64-bit row counts — SQLite 3.37+ (Nov 2021); guarded via proxy_if_available
    "sqlite3_changes64": int64(intp),
    "sqlite3_total_changes64": int64(intp),
    # === Statement lifecycle ===
    "sqlite3_prepare_v2": int32(intp, intp, int32, intp, intp),
    "sqlite3_finalize": int32(intp),
    "sqlite3_reset": int32(intp),
    "sqlite3_step": int32(intp),
    "sqlite3_sql": intp(intp),
    "sqlite3_expanded_sql": intp(intp),
    "sqlite3_stmt_busy": int32(intp),
    # === Parameter binding ===
    "sqlite3_bind_int": int32(intp, int32, int32),
    "sqlite3_bind_int64": int32(intp, int32, int64),
    "sqlite3_bind_double": int32(intp, int32, float64),
    "sqlite3_bind_text": int32(intp, int32, intp, int32, intp),
    "sqlite3_bind_blob": int32(intp, int32, intp, int32, intp),
    "sqlite3_bind_null": int32(intp, int32),
    "sqlite3_bind_parameter_count": int32(intp),
    "sqlite3_bind_parameter_index": int32(intp, intp),
    "sqlite3_bind_parameter_name": intp(intp, int32),
    # === Column accessors ===
    "sqlite3_column_int": int32(intp, int32),
    "sqlite3_column_int64": int64(intp, int32),
    "sqlite3_column_double": float64(intp, int32),
    "sqlite3_column_text": intp(intp, int32),
    "sqlite3_column_blob": intp(intp, int32),
    "sqlite3_column_bytes": int32(intp, int32),
    "sqlite3_column_type": int32(intp, int32),
    "sqlite3_column_count": int32(intp),
    "sqlite3_column_name": intp(intp, int32),
    "sqlite3_column_decltype": intp(intp, int32),
    # Compile-flag-gated (SQLITE_ENABLE_COLUMN_METADATA); via proxy_if_available
    "sqlite3_column_database_name": intp(intp, int32),
    "sqlite3_column_table_name": intp(intp, int32),
    "sqlite3_column_origin_name": intp(intp, int32),
    # === Exec + free ===
    "sqlite3_exec": int32(intp, intp, intp, intp, intp),
    "sqlite3_free": void(intp),
    # === BLOB incremental I/O ===
    "sqlite3_blob_open": int32(intp, intp, intp, intp, int64, int32, intp),
    "sqlite3_blob_close": int32(intp),
    "sqlite3_blob_bytes": int32(intp),
    "sqlite3_blob_read": int32(intp, intp, int32, int32),
    "sqlite3_blob_write": int32(intp, intp, int32, int32),
    "sqlite3_blob_reopen": int32(intp, int64),
    # === Callback hooks ===
    "sqlite3_update_hook": intp(intp, intp, intp),
    "sqlite3_progress_handler": void(intp, int32, intp, intp),
    "sqlite3_busy_handler": int32(intp, intp, intp),
    "sqlite3_commit_hook": intp(intp, intp, intp),
    "sqlite3_rollback_hook": intp(intp, intp, intp),
    "sqlite3_trace_v2": int32(intp, int32, intp, intp),
}

signatures = {
    **signatures_c,
    **signatures_m,
    **signatures_sqlite
}
