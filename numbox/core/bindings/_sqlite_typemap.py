"""Shared numpy-dtype ↔ SQLite type mapping + fixed-width string helpers.

Used by the read-only vtable (_sqlite_vtable), the table-valued-function
mechanism, and query_to_array. The dtype tags are the single source of truth
for how a numpy column maps to a SQLite column type and how its bytes are
read/written.
"""
import numpy as np
from numba import carray, njit
from numba.core.types import uint8, uint32

from numbox.core.configurations import jit_options
from numbox.utils.lowlevel import _cast_int_to_void_p, load_unaligned, store_unaligned

_TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64 = 0, 1, 2, 3
_TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64 = 4, 5, 6, 7
_TAG_F32, _TAG_F64, _TAG_BOOL = 8, 9, 10
_TAG_S, _TAG_U, _TAG_BLOB = 11, 12, 13


@njit(**jit_options)
def _nul_trimmed_len(p, width):
    buf = carray(_cast_int_to_void_p(p), (width,), dtype=np.uint8)
    n = width
    while n > 0 and buf[n - 1] == 0:
        n -= 1
    return n


@njit(**jit_options)
def utf32_to_utf8(src, n_codepoints, dst):
    """``dst`` must hold at least ``4 * n_codepoints + 1`` bytes."""
    m = n_codepoints
    while m > 0 and load_unaligned(src + 4 * (m - 1), uint32) == 0:
        m -= 1
    out = carray(_cast_int_to_void_p(dst), (4 * n_codepoints + 1,), dtype=np.uint8)
    k = 0
    for i in range(m):
        cp = load_unaligned(src + 4 * i, uint32)
        if cp > 0x10FFFF or (0xD800 <= cp <= 0xDFFF):
            cp = 0xFFFD
        if cp < 0x80:
            out[k] = uint8(cp)
            k += 1
        elif cp < 0x800:
            out[k] = uint8(0xC0 | (cp >> 6))
            out[k + 1] = uint8(0x80 | (cp & 0x3F))
            k += 2
        elif cp < 0x10000:
            out[k] = uint8(0xE0 | (cp >> 12))
            out[k + 1] = uint8(0x80 | ((cp >> 6) & 0x3F))
            out[k + 2] = uint8(0x80 | (cp & 0x3F))
            k += 3
        else:
            out[k] = uint8(0xF0 | (cp >> 18))
            out[k + 1] = uint8(0x80 | ((cp >> 12) & 0x3F))
            out[k + 2] = uint8(0x80 | ((cp >> 6) & 0x3F))
            out[k + 3] = uint8(0x80 | (cp & 0x3F))
            k += 4
    return k


@njit(**jit_options)
def utf8_to_utf32(src, nbytes, dst, width_cp):
    """Decode the UTF-8 bytes at ``src`` (length ``nbytes``) into up to
    ``width_cp`` little-endian uint32 code points at ``dst``; NUL-pad the
    remainder. Malformed input (bad continuation byte, surrogate, overlong
    encoding, out-of-range) decodes to U+FFFD. Returns the number of code points
    written. ``dst`` may be misaligned (writes are align=1)."""
    inp = carray(_cast_int_to_void_p(src), (nbytes,), dtype=np.uint8)
    for k in range(width_cp):
        store_unaligned(dst + 4 * k, uint32(0))
    i = 0
    k = 0
    while i < nbytes and k < width_cp:
        b0 = uint32(inp[i])
        if b0 < 0x80:
            cp = b0
            i += 1
        elif b0 >> 5 == 0x6 and i + 1 < nbytes and (inp[i + 1] >> 6) == 0x2:
            cp = ((b0 & 0x1F) << 6) | (uint32(inp[i + 1]) & 0x3F)
            if cp < 0x80:
                cp = 0xFFFD
            i += 2
        elif b0 >> 4 == 0xE and i + 2 < nbytes and (inp[i + 1] >> 6) == 0x2 and (inp[i + 2] >> 6) == 0x2:
            cp = ((b0 & 0x0F) << 12) | ((uint32(inp[i + 1]) & 0x3F) << 6) | (uint32(inp[i + 2]) & 0x3F)
            if cp < 0x800 or (0xD800 <= cp <= 0xDFFF):
                cp = 0xFFFD
            i += 3
        elif (b0 >> 3 == 0x1E and i + 3 < nbytes and (inp[i + 1] >> 6) == 0x2
              and (inp[i + 2] >> 6) == 0x2 and (inp[i + 3] >> 6) == 0x2):
            cp = (((b0 & 0x07) << 18) | ((uint32(inp[i + 1]) & 0x3F) << 12)
                  | ((uint32(inp[i + 2]) & 0x3F) << 6) | (uint32(inp[i + 3]) & 0x3F))
            if cp < 0x10000 or cp > 0x10FFFF:
                cp = 0xFFFD
            i += 4
        else:
            cp = 0xFFFD
            i += 1
        store_unaligned(dst + 4 * k, uint32(cp))
        k += 1
    return k


_NUMERIC_TAGS = {
    np.dtype("int8"): _TAG_I8, np.dtype("int16"): _TAG_I16,
    np.dtype("int32"): _TAG_I32, np.dtype("int64"): _TAG_I64,
    np.dtype("uint8"): _TAG_U8, np.dtype("uint16"): _TAG_U16,
    np.dtype("uint32"): _TAG_U32, np.dtype("uint64"): _TAG_U64,
    np.dtype("float32"): _TAG_F32, np.dtype("float64"): _TAG_F64,
    np.dtype("bool"): _TAG_BOOL,
}
_SQL_TYPE = {
    _TAG_I8: "INTEGER", _TAG_I16: "INTEGER", _TAG_I32: "INTEGER", _TAG_I64: "INTEGER",
    _TAG_U8: "INTEGER", _TAG_U16: "INTEGER", _TAG_U32: "INTEGER", _TAG_U64: "INTEGER",
    _TAG_BOOL: "INTEGER", _TAG_F32: "REAL", _TAG_F64: "REAL",
    _TAG_S: "TEXT", _TAG_U: "TEXT", _TAG_BLOB: "BLOB",
}


def _col_tag(dt, text_as_blob):
    if dt.kind == "S":
        return _TAG_BLOB if text_as_blob else _TAG_S
    if dt.kind == "U":
        return _TAG_U
    if dt in _NUMERIC_TAGS:
        return _NUMERIC_TAGS[dt]
    raise TypeError("unsupported column dtype %r" % (dt,))
