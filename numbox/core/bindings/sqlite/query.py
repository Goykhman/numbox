"""Collect the rows of a SQL query into a NumPy structured array.

:func:`query_to_array` runs one query on a connection and returns its rows, one field of the dtype for each result
column. The jitted helpers that it calls step the statement and write the bytes of each cell into the array.
"""
import ctypes

import numpy as np
from numba import njit, carray
from numba.core.types import uint8

from numbox.core.bindings.sqlite.constants import SQLITE_ROW, SQLITE_NULL, SQLITE_OK, SQLITE_DONE
from numbox.core.bindings.sqlite.stmt import sqlite3_prepare_v2, sqlite3_step, sqlite3_finalize
from numbox.core.bindings.sqlite.column import (
    sqlite3_column_count, sqlite3_column_type, sqlite3_column_int64, sqlite3_column_double,
    sqlite3_column_text, sqlite3_column_blob, sqlite3_column_bytes,
)
from numbox.core.bindings.sqlite.conn import sqlite3_errmsg
from numbox.core.bindings.sqlite._typemap import (
    _col_tag,
    _TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64, _TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64,
    _TAG_F32, _TAG_F64, _TAG_BOOL, _TAG_S, _TAG_U, _TAG_BLOB,
    _utf8_decode_one,
)
from numbox.core.configurations import jit_options
from numbox.utils.lowlevel import _cast_int_to_void_p

__all__ = ["query_to_array"]


@njit(**jit_options)
def _copy_bytes(dst, off, src, nbytes):
    """Copy ``nbytes`` bytes from the start of ``src`` into ``dst`` at ``off``."""
    for k in range(nbytes):
        dst[off + k] = src[k]


@njit(**jit_options)
def _put_unicode(buf, off, scratch8, src_p, nbytes, width_cp):
    """Decode the ``nbytes`` UTF-8 bytes at ``src_p`` into at most ``width_cp`` code points in ``buf`` at ``off``.

    Each code point is written as a 4-byte unsigned integer in the machine's byte order, as NumPy's ``'U'`` dtype
    stores it. Malformed input decodes to U+FFFD.
    """
    inp = carray(_cast_int_to_void_p(src_p), (nbytes,), dtype=np.uint8)
    # The decoder is _typemap._utf8_decode_one, as in utf8_to_utf32, but each code point goes into buf through the
    # uint32 view of scratch8 and _copy_bytes, writes to arrays that numba tracks. utf8_to_utf32 stores through a raw
    # pointer instead, and the optimizer for macOS on arm64 has removed such stores as dead code.
    cps = scratch8.view(np.uint32)
    i = 0
    k = 0
    while i < nbytes and k < width_cp:
        cp, i = _utf8_decode_one(inp, i, nbytes)
        cps[0] = cp
        _copy_bytes(buf, off + 4 * k, scratch8, 4)
        k += 1


@njit(**jit_options)
def _store_cell(buf, off, tag, width, stmt, j, scratch8):
    """Write column ``j`` of the current row of ``stmt`` into the field at byte ``off`` of the ``uint8`` array ``buf``.

    ``tag`` and ``width`` describe the field. ``scratch8`` is an 8-byte ``uint8`` array for the conversion of one
    value. The caller has zeroed the row, so a SQL NULL leaves 0 in an integer or ``bool`` field and an empty value in
    a string field; only a float field receives NaN for NULL.
    """
    # Each scalar is written through a typed view of scratch8, so that its bytes are in the machine's byte order, and
    # _copy_bytes then copies them into buf. Every write goes to an array that numba tracks, so the optimizer for
    # macOS on arm64 cannot remove it as it removes a store through a raw pointer. An integer keeps the low bits that
    # fit its field. The text or BLOB accessor runs before sqlite3_column_bytes, as SQLite advises, because the
    # accessor can convert the value and change its length.
    ctype = sqlite3_column_type(stmt, j)
    if ctype == SQLITE_NULL:
        if tag == _TAG_F32:
            scratch8.view(np.float32)[0] = np.float32(np.nan)
            _copy_bytes(buf, off, scratch8, 4)
        elif tag == _TAG_F64:
            scratch8.view(np.float64)[0] = np.float64(np.nan)
            _copy_bytes(buf, off, scratch8, 8)
        return
    if tag == _TAG_I8 or tag == _TAG_U8:
        buf[off] = uint8(sqlite3_column_int64(stmt, j))
    elif tag == _TAG_I16 or tag == _TAG_U16:
        scratch8.view(np.int16)[0] = np.int16(sqlite3_column_int64(stmt, j))
        _copy_bytes(buf, off, scratch8, 2)
    elif tag == _TAG_I32 or tag == _TAG_U32:
        scratch8.view(np.int32)[0] = np.int32(sqlite3_column_int64(stmt, j))
        _copy_bytes(buf, off, scratch8, 4)
    elif tag == _TAG_I64 or tag == _TAG_U64:
        scratch8.view(np.int64)[0] = np.int64(sqlite3_column_int64(stmt, j))
        _copy_bytes(buf, off, scratch8, 8)
    elif tag == _TAG_BOOL:
        buf[off] = uint8(1) if sqlite3_column_int64(stmt, j) != 0 else uint8(0)
    elif tag == _TAG_F32:
        scratch8.view(np.float32)[0] = np.float32(sqlite3_column_double(stmt, j))
        _copy_bytes(buf, off, scratch8, 4)
    elif tag == _TAG_F64:
        scratch8.view(np.float64)[0] = sqlite3_column_double(stmt, j)
        _copy_bytes(buf, off, scratch8, 8)
    elif tag == _TAG_U:
        _put_unicode(buf, off, scratch8, sqlite3_column_text(stmt, j), sqlite3_column_bytes(stmt, j), width // 4)
    elif tag == _TAG_S:
        src_p = sqlite3_column_text(stmt, j)
        nbytes = sqlite3_column_bytes(stmt, j)
        src = carray(_cast_int_to_void_p(src_p), (nbytes,), dtype=np.uint8)
        nn = nbytes if nbytes < width else width
        for b in range(nn):
            buf[off + b] = src[b]
    elif tag == _TAG_BLOB:
        src_p = sqlite3_column_blob(stmt, j)
        nbytes = sqlite3_column_bytes(stmt, j)
        src = carray(_cast_int_to_void_p(src_p), (nbytes,), dtype=np.uint8)
        nn = nbytes if nbytes < width else width
        for b in range(nn):
            buf[off + b] = src[b]


@njit(**jit_options)
def _query_core(stmt, ncols, offsets, tags, widths, itemsize):
    """Step ``stmt`` to its end, and return its rows as bytes and the last code of ``sqlite3_step``.

    The rows are ``n * itemsize`` bytes in the machine's byte order, which the caller views as the structured dtype.
    The code is ``SQLITE_DONE`` when every step succeeded.
    """
    # The buffer starts with room for 16 rows and doubles whenever it is full.
    cap = 16
    out_u8 = np.zeros(cap * itemsize, np.uint8)
    scratch8 = np.empty(8, np.uint8)
    n = 0
    rc = sqlite3_step(stmt)
    while rc == SQLITE_ROW:
        if n == cap:
            cap = cap * 2
            new_u8 = np.zeros(cap * itemsize, np.uint8)
            new_u8[:n * itemsize] = out_u8[:n * itemsize]
            out_u8 = new_u8
        row_off = n * itemsize
        for j in range(ncols):
            _store_cell(out_u8, row_off + offsets[j], tags[j], widths[j], stmt, j, scratch8)
        n += 1
        rc = sqlite3_step(stmt)
    # The slice alone would keep the memory alive, because it shares the parent's NRT meminfo (make_view and
    # impl_ret_borrowed). The copy only drops the unused rows, fewer than one doubling: fewer than n rows when n > 16,
    # and at most 16 rows when n <= 16, because the capacity starts at 16.
    return out_u8[:n * itemsize].copy(), rc


def _raise_rc(db, rc):
    msg_p = sqlite3_errmsg(db)
    detail = ""
    if msg_p:
        detail = ": " + ctypes.cast(msg_p, ctypes.c_char_p).value.decode("utf-8", "replace")
    raise RuntimeError("query_to_array failed (rc=%d)%s" % (rc, detail))


def query_to_array(db, sql_p, dtype):
    """Run the SQL query at ``sql_p`` on ``db`` and return its rows as a 1-D NumPy structured array of ``dtype``.

    ``db`` is the connection's ``sqlite3 *`` handle. ``sql_p`` is the address of NUL-terminated UTF-8 SQL text, not a
    Python ``str``: :func:`~numbox.utils.cstrings.c_string` makes one from any text, and
    :func:`~numbox.utils.lowlevel.get_unicode_data_p` gives one for ASCII text. Only the first statement of the text
    runs. The fields of ``dtype`` take the result columns in order, one field for each column.

    A SQL NULL becomes NaN in a float field, 0 in an integer or ``bool`` field, and an empty value in an ``'S'`` or
    ``'U'`` field. An integer field keeps the low bits of a value too wide for it, and a ``bool`` field is true for any
    nonzero value. A string longer than its field is cut to the field's width, and the text for a ``'U'`` field is
    decoded from UTF-8, with U+FFFD for malformed bytes.

    Raises ``TypeError`` for a ``dtype`` without fields or with a field type that has no SQLite counterpart, such as
    a complex number. Raises ``ValueError`` when the number of fields differs from the number of result columns, and
    ``RuntimeError``, with SQLite's message, when the query fails to prepare or to run. Once the statement is
    prepared, the function finalizes it on every path.
    """
    if dtype.fields is None or dtype.names is None:
        raise TypeError("dtype must be a structured numpy dtype, got %r" % (dtype,))
    names = list(dtype.names)
    offsets = np.array([dtype.fields[nm][1] for nm in names], dtype=np.int64)
    subs = [dtype.fields[nm][0] for nm in names]
    tags = np.array([_col_tag(s, False) for s in subs], dtype=np.int64)
    widths = np.array([int(s.itemsize) for s in subs], dtype=np.int64)
    stmt = ctypes.c_int64(0)
    rc = sqlite3_prepare_v2(db, sql_p, -1, ctypes.addressof(stmt), 0)
    if rc != SQLITE_OK:
        _raise_rc(db, rc)
    try:
        ncols = sqlite3_column_count(stmt.value)
        if ncols != len(names):
            raise ValueError("dtype has %d fields but query returns %d columns" % (len(names), ncols))
        buf, last_rc = _query_core(stmt.value, ncols, offsets, tags, widths, int(dtype.itemsize))
        if last_rc != SQLITE_DONE:
            _raise_rc(db, last_rc)
        return buf.view(dtype)
    finally:
        sqlite3_finalize(stmt.value)
