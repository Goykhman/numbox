"""Value accessor tests for the SQLite UDF bindings.

Exercises each ``sqlite3_value_*`` accessor inside a ``@cfunc`` callback
registered via ``sqlite3_create_function_v2``.  The capture pattern writes
the accessor's result into a numpy array via ``sqlite3_user_data`` +
``carray``, then the Python-level test asserts the captured value.
"""
from ctypes import addressof, c_int64

import numpy as np
from numba import carray, cfunc, njit, types
from numba.core import types as nb_types

from numbox.core.bindings._sqlite_conn import sqlite3_close, sqlite3_open
from numbox.core.bindings._sqlite_constants import (
    SQLITE_BLOB,
    SQLITE_FLOAT,
    SQLITE_INTEGER,
    SQLITE_NULL,
    SQLITE_OK,
    SQLITE_TEXT,
    SQLITE_UTF8,
)
from numbox.core.bindings._sqlite_exec import sqlite3_exec
from numbox.core.bindings._sqlite_result import sqlite3_result_int
from numbox.core.bindings._sqlite_udf import (
    sqlite3_create_function_v2,
    sqlite3_user_data,
)
from numbox.core.bindings._sqlite_value import (
    sqlite3_value_blob,
    sqlite3_value_bytes,
    sqlite3_value_double,
    sqlite3_value_dup,
    sqlite3_value_free,
    sqlite3_value_int,
    sqlite3_value_int64,
    sqlite3_value_text,
    sqlite3_value_type,
)
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import _cast_int_to_void_p
from test.auxiliary_utils import collect_and_run_tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_memory():
    db_p = c_int64(0)
    with c_string(":memory:") as name_p:
        rc = sqlite3_open(name_p, addressof(db_p))
    assert rc == SQLITE_OK
    return db_p.value


def _register(db_p, name, narg, cb_address, ud_ptr):
    with c_string(name) as name_p:
        rc = sqlite3_create_function_v2(
            db_p, name_p, narg, SQLITE_UTF8, ud_ptr,
            cb_address, 0, 0, 0)
    assert rc == SQLITE_OK


# ---------------------------------------------------------------------------
# Callback: capture int
# ---------------------------------------------------------------------------

@njit
def _cap_int_impl(ctx, argc, argv_pp):
    ud = sqlite3_user_data(ctx)
    out = carray(_cast_int_to_void_p(ud), (1,), dtype=np.int64)
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    out[0] = nb_types.int64(sqlite3_value_int(args[0]))
    sqlite3_result_int(ctx, 0)


_cap_int_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_cap_int_impl)


# ---------------------------------------------------------------------------
# Callback: capture int64
# ---------------------------------------------------------------------------

@njit
def _cap_int64_impl(ctx, argc, argv_pp):
    ud = sqlite3_user_data(ctx)
    out = carray(_cast_int_to_void_p(ud), (1,), dtype=np.int64)
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    out[0] = sqlite3_value_int64(args[0])
    sqlite3_result_int(ctx, 0)


_cap_int64_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_cap_int64_impl)


# ---------------------------------------------------------------------------
# Callback: capture double
# ---------------------------------------------------------------------------

@njit
def _cap_double_impl(ctx, argc, argv_pp):
    ud = sqlite3_user_data(ctx)
    out = carray(_cast_int_to_void_p(ud), (1,), dtype=np.float64)
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    out[0] = sqlite3_value_double(args[0])
    sqlite3_result_int(ctx, 0)


_cap_double_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_cap_double_impl)


# ---------------------------------------------------------------------------
# Callback: capture text bytes into ud array
# ud layout: [0] = byte count, [1..] = text bytes as uint8
# ---------------------------------------------------------------------------

@njit
def _cap_text_impl(ctx, argc, argv_pp):
    ud = sqlite3_user_data(ctx)
    meta = carray(_cast_int_to_void_p(ud), (64,), dtype=np.uint8)
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    text_p = sqlite3_value_text(args[0])
    nbytes = sqlite3_value_bytes(args[0])
    meta[0] = nb_types.uint8(nbytes)
    src = carray(_cast_int_to_void_p(text_p), (nbytes,), dtype=np.uint8)
    for i in range(nbytes):
        meta[1 + i] = src[i]
    sqlite3_result_int(ctx, 0)


_cap_text_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_cap_text_impl)


# ---------------------------------------------------------------------------
# Callback: capture blob bytes into ud array
# ud layout: [0] = byte count, [1..] = blob bytes as uint8
# ---------------------------------------------------------------------------

@njit
def _cap_blob_impl(ctx, argc, argv_pp):
    ud = sqlite3_user_data(ctx)
    meta = carray(_cast_int_to_void_p(ud), (64,), dtype=np.uint8)
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    blob_p = sqlite3_value_blob(args[0])
    nbytes = sqlite3_value_bytes(args[0])
    meta[0] = nb_types.uint8(nbytes)
    src = carray(_cast_int_to_void_p(blob_p), (nbytes,), dtype=np.uint8)
    for i in range(nbytes):
        meta[1 + i] = src[i]
    sqlite3_result_int(ctx, 0)


_cap_blob_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_cap_blob_impl)


# ---------------------------------------------------------------------------
# Callback: capture type codes for up to 5 args
# ---------------------------------------------------------------------------

@njit
def _cap_types_impl(ctx, argc, argv_pp):
    ud = sqlite3_user_data(ctx)
    out = carray(_cast_int_to_void_p(ud), (6,), dtype=np.int64)
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    out[0] = nb_types.int64(argc)
    for i in range(argc):
        out[1 + i] = nb_types.int64(sqlite3_value_type(args[i]))
    sqlite3_result_int(ctx, 0)


_cap_types_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_cap_types_impl)


# ---------------------------------------------------------------------------
# Callback: dup a value, read int from the dup, then free
# ---------------------------------------------------------------------------

@njit
def _cap_dup_impl(ctx, argc, argv_pp):
    ud = sqlite3_user_data(ctx)
    out = carray(_cast_int_to_void_p(ud), (1,), dtype=np.int64)
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    dup_p = sqlite3_value_dup(args[0])
    out[0] = sqlite3_value_int64(dup_p)
    sqlite3_value_free(dup_p)
    sqlite3_result_int(ctx, 0)


_cap_dup_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_cap_dup_impl)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_value_int_roundtrip():
    db_p = _open_memory()
    out = np.zeros(1, dtype=np.int64)
    _register(db_p, "probe_int", 1, _cap_int_cb.address, out.ctypes.data)
    with c_string("SELECT probe_int(42)") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert out[0] == 42
    sqlite3_close(db_p)


def test_value_int64_roundtrip():
    db_p = _open_memory()
    out = np.zeros(1, dtype=np.int64)
    _register(db_p, "probe_i64", 1, _cap_int64_cb.address, out.ctypes.data)
    with c_string("SELECT probe_i64(9999999999)") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert out[0] == 9999999999
    sqlite3_close(db_p)


def test_value_double_roundtrip():
    db_p = _open_memory()
    out = np.zeros(1, dtype=np.float64)
    _register(db_p, "probe_dbl", 1, _cap_double_cb.address, out.ctypes.data)
    with c_string("SELECT probe_dbl(3.14)") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert abs(out[0] - 3.14) < 1e-12
    sqlite3_close(db_p)


def test_value_text_decodes_utf8():
    db_p = _open_memory()
    buf = np.zeros(64, dtype=np.uint8)
    _register(db_p, "probe_txt", 1, _cap_text_cb.address, buf.ctypes.data)
    with c_string("SELECT probe_txt('hello')") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    nbytes = int(buf[0])
    text = bytes(buf[1:1 + nbytes]).decode("utf-8")
    assert text == "hello"
    sqlite3_close(db_p)


def test_value_blob_matches_inserted():
    db_p = _open_memory()
    buf = np.zeros(64, dtype=np.uint8)
    _register(db_p, "probe_blob", 1, _cap_blob_cb.address, buf.ctypes.data)
    with c_string("SELECT probe_blob(X'DEADBEEF')") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    nbytes = int(buf[0])
    assert nbytes == 4
    raw = bytes(buf[1:1 + nbytes])
    assert raw == b'\xDE\xAD\xBE\xEF'
    sqlite3_close(db_p)


def test_value_type_returns_correct_codes():
    db_p = _open_memory()
    out = np.zeros(6, dtype=np.int64)
    _register(db_p, "probe_types", 5, _cap_types_cb.address, out.ctypes.data)
    sql = "SELECT probe_types(42, 3.14, 'hi', X'FF', NULL)"
    with c_string(sql) as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert out[0] == 5
    assert out[1] == SQLITE_INTEGER
    assert out[2] == SQLITE_FLOAT
    assert out[3] == SQLITE_TEXT
    assert out[4] == SQLITE_BLOB
    assert out[5] == SQLITE_NULL
    sqlite3_close(db_p)


def test_value_dup_and_free():
    db_p = _open_memory()
    out = np.zeros(1, dtype=np.int64)
    _register(db_p, "probe_dup", 1, _cap_dup_cb.address, out.ctypes.data)
    with c_string("SELECT probe_dup(12345)") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert out[0] == 12345
    sqlite3_close(db_p)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
