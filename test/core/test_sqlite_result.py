"""Tests of the bindings of :mod:`~numbox.core.bindings.sqlite.result`.

Each test registers a scalar SQL function, the producer, that sets its result with one binding, and a second function,
the probe, that receives the result as its argument. The probe reads the value with an accessor of
:mod:`~numbox.core.bindings.sqlite.value` and writes it into a NumPy array at its user data, so the test checks the
value that SQLite passed on. A test of an error setter checks the result code of ``sqlite3_exec`` instead. The file also
tests ``sqlite3_value_numeric_type`` and ``sqlite3_value_subtype`` on values that a producer sets.
"""
from ctypes import addressof, c_int64

import numpy as np
from numba import carray, cfunc, njit, types
from numba.core import types as nb_types

from numbox.core.bindings.sqlite.constants import (
    SQLITE_NOMEM, SQLITE_NULL, SQLITE_OK, SQLITE_RESULT_SUBTYPE, SQLITE_SUBTYPE, SQLITE_TOOBIG,
    SQLITE_TRANSIENT, SQLITE_UTF8,
)
from numbox.core.bindings.sqlite.conn import sqlite3_close, sqlite3_libversion_number, sqlite3_open
from numbox.core.bindings.sqlite.udf import sqlite3_create_function_v2, sqlite3_user_data
from numbox.core.bindings.sqlite.exec import sqlite3_exec
from numbox.core.bindings.sqlite.result import (
    sqlite3_result_blob, sqlite3_result_blob64, sqlite3_result_double, sqlite3_result_error,
    sqlite3_result_error_code, sqlite3_result_error_nomem, sqlite3_result_error_toobig,
    sqlite3_result_int, sqlite3_result_int64, sqlite3_result_null, sqlite3_result_subtype,
    sqlite3_result_text, sqlite3_result_text64, sqlite3_result_value, sqlite3_result_zeroblob,
    sqlite3_result_zeroblob64,
)
from numbox.core.bindings.sqlite.value import (
    sqlite3_value_blob, sqlite3_value_bytes, sqlite3_value_double, sqlite3_value_int64,
    sqlite3_value_numeric_type, sqlite3_value_subtype, sqlite3_value_text, sqlite3_value_type,
)
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import _cast_int_to_void_p, get_unicode_data_p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_memory():
    db_p = c_int64(0)
    with c_string(":memory:") as name_p:
        rc = sqlite3_open(name_p, addressof(db_p))
    assert rc == SQLITE_OK
    return db_p.value


def _register(db_p, name, narg, cb_address, ud_ptr, flags=SQLITE_UTF8):
    with c_string(name) as name_p:
        rc = sqlite3_create_function_v2(
            db_p, name_p, narg, flags, ud_ptr,
            cb_address, 0, 0, 0)
    assert rc == SQLITE_OK


def _register_and_exec(db_p, name, cb, sql, n_arg=0):
    """Register ``cb`` as the scalar function ``name``, run ``sql``, and return the result code of ``sqlite3_exec``."""
    with c_string(name) as name_p:
        rc = sqlite3_create_function_v2(
            db_p, name_p, n_arg, SQLITE_UTF8, 0,
            cb.address, 0, 0, 0)
    assert rc == SQLITE_OK, "create_function_v2 failed: rc=%d" % rc
    with c_string(sql) as sql_p:
        return sqlite3_exec(db_p, sql_p, 0, 0, 0)


# ---------------------------------------------------------------------------
# Probes, which read their first argument and write it into the array at their user data. The probes and the producers
# below are cfuncs at module level, so each compiles once. numba keeps a cfunc's compiled code after the object is
# collected, so the module-level reference is a precaution.
# ---------------------------------------------------------------------------

@njit
def _cap_int64_impl(ctx, argc, argv_pp):
    """Write the first argument, read as an int64, into the int64 at the user data."""
    ud = sqlite3_user_data(ctx)
    out = carray(_cast_int_to_void_p(ud), (1,), dtype=np.int64)
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    out[0] = sqlite3_value_int64(args[0])
    sqlite3_result_int(ctx, 0)


_cap_int64_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_cap_int64_impl)


@njit
def _cap_double_impl(ctx, argc, argv_pp):
    """Write the first argument, read as a float64, into the float64 at the user data."""
    ud = sqlite3_user_data(ctx)
    out = carray(_cast_int_to_void_p(ud), (1,), dtype=np.float64)
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    out[0] = sqlite3_value_double(args[0])
    sqlite3_result_int(ctx, 0)


_cap_double_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_cap_double_impl)


@njit
def _cap_text_impl(ctx, argc, argv_pp):
    """Write the byte count of the first argument's text into byte 0 of the user data, and the bytes after it."""
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


@njit
def _cap_type_impl(ctx, argc, argv_pp):
    """Write the type code of the first argument into the int64 at the user data."""
    ud = sqlite3_user_data(ctx)
    out = carray(_cast_int_to_void_p(ud), (1,), dtype=np.int64)
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    out[0] = nb_types.int64(sqlite3_value_type(args[0]))
    sqlite3_result_int(ctx, 0)


_cap_type_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_cap_type_impl)


@njit
def _cap_blob_impl(ctx, argc, argv_pp):
    """Write the byte count of the first argument's BLOB into byte 0 of the user data, and the bytes after it."""
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


@njit
def _cap_subtype_impl(ctx, argc, argv_pp):
    """Write the subtype of the first argument into the int64 at the user data."""
    ud = sqlite3_user_data(ctx)
    out = carray(_cast_int_to_void_p(ud), (1,), dtype=np.int64)
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    out[0] = nb_types.int64(sqlite3_value_subtype(args[0]))
    sqlite3_result_int(ctx, 0)


_cap_subtype_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_cap_subtype_impl)


@njit
def _cap_numeric_type_impl(ctx, argc, argv_pp):
    """Write the type code that ``sqlite3_value_numeric_type`` returns for the first argument into the user data."""
    ud = sqlite3_user_data(ctx)
    out = carray(_cast_int_to_void_p(ud), (1,), dtype=np.int64)
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    out[0] = nb_types.int64(sqlite3_value_numeric_type(args[0]))
    sqlite3_result_int(ctx, 0)


_cap_numeric_type_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_cap_numeric_type_impl)


# ---------------------------------------------------------------------------
# Producers, each of which sets its result with one binding of the result module
# ---------------------------------------------------------------------------

@cfunc(types.void(types.intp, types.int32, types.intp))
def _result_int_cb(ctx, argc, argv_pp):
    sqlite3_result_int(ctx, 42)


@cfunc(types.void(types.intp, types.int32, types.intp))
def _result_int64_cb(ctx, argc, argv_pp):
    sqlite3_result_int64(ctx, 9999999999)


@cfunc(types.void(types.intp, types.int32, types.intp))
def _result_double_cb(ctx, argc, argv_pp):
    sqlite3_result_double(ctx, 2.718281828)


@cfunc(types.void(types.intp, types.int32, types.intp))
def _result_null_cb(ctx, argc, argv_pp):
    sqlite3_result_null(ctx)


@njit
def _result_text_impl(ctx, argc, argv_pp):
    s = "world"
    p = get_unicode_data_p(s)
    sqlite3_result_text(ctx, p, 5, SQLITE_TRANSIENT)


_result_text_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_result_text_impl)


@njit
def _result_error_impl(ctx, argc, argv_pp):
    s = "fail"
    p = get_unicode_data_p(s)
    sqlite3_result_error(ctx, p, 4)


_result_error_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_result_error_impl)


@njit
def _result_value_impl(ctx, argc, argv_pp):
    argv_p = _cast_int_to_void_p(argv_pp)
    args = carray(argv_p, (argc,), dtype=np.intp)
    sqlite3_result_value(ctx, args[0])


_result_value_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_result_value_impl)


@cfunc(types.void(types.intp, types.int32, types.intp))
def _result_zeroblob_cb(ctx, argc, argv_pp):
    sqlite3_result_zeroblob(ctx, 16)


@njit
def _result_blob_impl(ctx, argc, argv_pp):
    ud = sqlite3_user_data(ctx)
    sqlite3_result_blob(ctx, ud, 4, SQLITE_TRANSIENT)


_result_blob_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_result_blob_impl)


@njit
def _result_blob64_impl(ctx, argc, argv_pp):
    ud = sqlite3_user_data(ctx)
    sqlite3_result_blob64(ctx, ud, nb_types.uint64(4), SQLITE_TRANSIENT)


_result_blob64_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_result_blob64_impl)


@njit
def _result_text64_impl(ctx, argc, argv_pp):
    s = "hello64"
    p = get_unicode_data_p(s)
    sqlite3_result_text64(ctx, p, nb_types.uint64(7), SQLITE_TRANSIENT,
                          nb_types.uint8(SQLITE_UTF8))


_result_text64_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_result_text64_impl)


@cfunc(types.void(types.intp, types.int32, types.intp))
def _result_zeroblob64_cb(ctx, argc, argv_pp):
    sqlite3_result_zeroblob64(ctx, nb_types.uint64(20))


@cfunc(types.void(types.intp, types.int32, types.intp))
def _result_error_code_cb(ctx, argc, argv_pp):
    sqlite3_result_error_code(ctx, SQLITE_NOMEM)


@cfunc(types.void(types.intp, types.int32, types.intp))
def _result_error_nomem_cb(ctx, argc, argv_pp):
    sqlite3_result_error_nomem(ctx)


@cfunc(types.void(types.intp, types.int32, types.intp))
def _result_error_toobig_cb(ctx, argc, argv_pp):
    sqlite3_result_error_toobig(ctx)


@njit
def _result_subtype_impl(ctx, argc, argv_pp):
    sqlite3_result_int(ctx, 99)
    sqlite3_result_subtype(ctx, nb_types.uint32(74))


_result_subtype_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_result_subtype_impl)


@njit
def _result_text_for_numeric_impl(ctx, argc, argv_pp):
    s = "123"
    p = get_unicode_data_p(s)
    sqlite3_result_text(ctx, p, 3, SQLITE_TRANSIENT)


_result_text_for_numeric_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_result_text_for_numeric_impl)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_result_int():
    db_p = _open_memory()
    out = np.zeros(1, dtype=np.int64)
    _register(db_p, "ri", 0, _result_int_cb.address, 0)
    _register(db_p, "cap_ri", 1, _cap_int64_cb.address, out.ctypes.data)
    with c_string("SELECT cap_ri(ri())") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert out[0] == 42
    sqlite3_close(db_p)


def test_result_int64():
    db_p = _open_memory()
    out = np.zeros(1, dtype=np.int64)
    _register(db_p, "ri64", 0, _result_int64_cb.address, 0)
    _register(db_p, "cap_ri64", 1, _cap_int64_cb.address, out.ctypes.data)
    with c_string("SELECT cap_ri64(ri64())") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert out[0] == 9999999999
    sqlite3_close(db_p)


def test_result_double():
    db_p = _open_memory()
    out = np.zeros(1, dtype=np.float64)
    _register(db_p, "rd", 0, _result_double_cb.address, 0)
    _register(db_p, "cap_rd", 1, _cap_double_cb.address, out.ctypes.data)
    with c_string("SELECT cap_rd(rd())") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert abs(out[0] - 2.718281828) < 1e-9
    sqlite3_close(db_p)


def test_result_null():
    db_p = _open_memory()
    out = np.zeros(1, dtype=np.int64)
    out[0] = -1
    _register(db_p, "rn", 0, _result_null_cb.address, 0)
    _register(db_p, "cap_rn", 1, _cap_type_cb.address, out.ctypes.data)
    with c_string("SELECT cap_rn(rn())") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert out[0] == SQLITE_NULL
    sqlite3_close(db_p)


def test_result_text_transient():
    db_p = _open_memory()
    buf = np.zeros(64, dtype=np.uint8)
    _register(db_p, "rt", 0, _result_text_cb.address, 0)
    _register(db_p, "cap_rt", 1, _cap_text_cb.address, buf.ctypes.data)
    with c_string("SELECT cap_rt(rt())") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    nbytes = int(buf[0])
    text = bytes(buf[1:1 + nbytes]).decode("utf-8")
    assert text == "world"
    sqlite3_close(db_p)


def test_result_error_aborts():
    db_p = _open_memory()
    rc = _register_and_exec(db_p, "re", _result_error_cb, "SELECT re()")
    assert rc != SQLITE_OK
    sqlite3_close(db_p)


def test_result_value_passthrough():
    db_p = _open_memory()
    out = np.zeros(1, dtype=np.int64)
    _register(db_p, "rv", 1, _result_value_cb.address, 0)
    _register(db_p, "cap_rv", 1, _cap_int64_cb.address, out.ctypes.data)
    with c_string("SELECT cap_rv(rv(123))") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert out[0] == 123
    sqlite3_close(db_p)


def test_result_zeroblob():
    db_p = _open_memory()
    buf = np.zeros(64, dtype=np.uint8)
    buf[0] = 0xFF
    _register(db_p, "rz", 0, _result_zeroblob_cb.address, 0)
    _register(db_p, "cap_rz", 1, _cap_blob_cb.address, buf.ctypes.data)
    with c_string("SELECT cap_rz(rz())") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    nbytes = int(buf[0])
    assert nbytes == 16
    assert all(buf[1:1 + nbytes] == 0)
    sqlite3_close(db_p)


def test_result_blob():
    db_p = _open_memory()
    src = np.array([0xDE, 0xAD, 0xBE, 0xEF], dtype=np.uint8)
    _register(db_p, "rb", 0, _result_blob_cb.address, src.ctypes.data)
    buf = np.zeros(64, dtype=np.uint8)
    _register(db_p, "cap_rb", 1, _cap_blob_cb.address, buf.ctypes.data)
    with c_string("SELECT cap_rb(rb())") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    nbytes = int(buf[0])
    assert nbytes == 4
    assert bytes(buf[1:5]) == b'\xDE\xAD\xBE\xEF'
    sqlite3_close(db_p)


def test_result_blob64():
    db_p = _open_memory()
    src = np.array([0xCA, 0xFE, 0xBA, 0xBE], dtype=np.uint8)
    _register(db_p, "rb64", 0, _result_blob64_cb.address, src.ctypes.data)
    buf = np.zeros(64, dtype=np.uint8)
    _register(db_p, "cap_rb64", 1, _cap_blob_cb.address, buf.ctypes.data)
    with c_string("SELECT cap_rb64(rb64())") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    nbytes = int(buf[0])
    assert nbytes == 4
    assert bytes(buf[1:5]) == b'\xCA\xFE\xBA\xBE'
    sqlite3_close(db_p)


def test_result_text64():
    db_p = _open_memory()
    buf = np.zeros(64, dtype=np.uint8)
    _register(db_p, "rt64", 0, _result_text64_cb.address, 0)
    _register(db_p, "cap_rt64", 1, _cap_text_cb.address, buf.ctypes.data)
    with c_string("SELECT cap_rt64(rt64())") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    nbytes = int(buf[0])
    text = bytes(buf[1:1 + nbytes]).decode("utf-8")
    assert text == "hello64"
    sqlite3_close(db_p)


def test_result_zeroblob64():
    db_p = _open_memory()
    buf = np.zeros(64, dtype=np.uint8)
    buf[0] = 0xFF
    _register(db_p, "rz64", 0, _result_zeroblob64_cb.address, 0)
    _register(db_p, "cap_rz64", 1, _cap_blob_cb.address, buf.ctypes.data)
    with c_string("SELECT cap_rz64(rz64())") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    nbytes = int(buf[0])
    assert nbytes == 20
    assert all(buf[1:1 + nbytes] == 0)
    sqlite3_close(db_p)


def test_result_error_code():
    db_p = _open_memory()
    rc = _register_and_exec(
        db_p, "rec", _result_error_code_cb, "SELECT rec()")
    assert rc == SQLITE_NOMEM
    sqlite3_close(db_p)


def test_result_error_nomem():
    db_p = _open_memory()
    rc = _register_and_exec(
        db_p, "renom", _result_error_nomem_cb, "SELECT renom()")
    assert rc == SQLITE_NOMEM
    sqlite3_close(db_p)


def test_result_error_toobig():
    db_p = _open_memory()
    rc = _register_and_exec(
        db_p, "retb", _result_error_toobig_cb, "SELECT retb()")
    assert rc == SQLITE_TOOBIG
    sqlite3_close(db_p)


def test_result_subtype():
    db_p = _open_memory()
    out = np.zeros(1, dtype=np.int64)
    # A SQLite of version 3.45 or later built with -DSQLITE_STRICT_SUBTYPE=1, such as the conda-forge build, makes the
    # producer fail at its call of sqlite3_result_subtype unless the producer is registered with SQLITE_RESULT_SUBTYPE.
    # sqlite3_value_subtype checks no flag, but without SQLITE_SUBTYPE the probe may read 0 for an argument that has a
    # subtype. The test sets each flag only on a version that defines it: 3.45.0 for SQLITE_RESULT_SUBTYPE and 3.30.0
    # for SQLITE_SUBTYPE.
    ver = sqlite3_libversion_number()
    rsub_flags = SQLITE_UTF8 | (SQLITE_RESULT_SUBTYPE if ver >= 3045000 else 0)
    cap_flags = SQLITE_UTF8 | (SQLITE_SUBTYPE if ver >= 3030000 else 0)
    _register(db_p, "rsub", 0, _result_subtype_cb.address, 0, flags=rsub_flags)
    _register(db_p, "cap_sub", 1, _cap_subtype_cb.address, out.ctypes.data,
              flags=cap_flags)
    with c_string("SELECT cap_sub(rsub())") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert out[0] == 74
    sqlite3_close(db_p)


def test_value_numeric_type():
    """``sqlite3_value_numeric_type`` converts the text '123' to an integer and returns ``SQLITE_INTEGER``."""
    from numbox.core.bindings.sqlite.constants import SQLITE_INTEGER
    db_p = _open_memory()
    out = np.zeros(1, dtype=np.int64)
    _register(db_p, "rtnum", 0, _result_text_for_numeric_cb.address, 0)
    _register(db_p, "cap_num", 1, _cap_numeric_type_cb.address,
              out.ctypes.data)
    with c_string("SELECT cap_num(rtnum())") as sql_p:
        rc = sqlite3_exec(db_p, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert out[0] == SQLITE_INTEGER
    sqlite3_close(db_p)


# ---------------------------------------------------------------------------
# Value accessors without a test
# ---------------------------------------------------------------------------
# sqlite3_value_nochange has a meaning only in the xUpdate of a writable virtual table, and numbox's virtual tables are
# read-only, without an xUpdate.
# sqlite3_value_frombind reports whether a value came from a bound parameter. The tests of this file run SQL through
# sqlite3_exec, which binds no parameter.


if __name__ == "__main__":
    from test.auxiliary_utils import collect_and_run_tests
    collect_and_run_tests(__name__)
