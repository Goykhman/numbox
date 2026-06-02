"""Expose a numpy array as a read-only SQLite virtual table (register_table).

A single generic sqlite3_module (built once at import) serves every table; the
per-table base pointer / strides / dtype tags / schema live in a ctypes
descriptor passed as pClientData.
"""
import ctypes

import numpy as np
from numba import carray, cfunc, njit, types
from numba.core.types import (
    int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    float32, float64,
)

from numbox.core.bindings._sqlite_constants import (
    SQLITE_OK, SQLITE_TRANSIENT, SQLITE_ERROR, SQLITE_NOMEM,
)
from numbox.core.bindings._sqlite_conn import sqlite3_errmsg
from numbox.core.bindings._sqlite_exec import sqlite3_free
from numbox.core.bindings._sqlite_result import (
    sqlite3_result_int64, sqlite3_result_double,
    sqlite3_result_text, sqlite3_result_blob, sqlite3_result_error,
)
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import (
    _cast_int_to_void_p, get_unicode_data_p, load_at, load_unaligned, store_at,
)

__all__ = ["register_table"]

load_lib("sqlite3")

# dtype tags (col_tags[j])
_TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64 = 0, 1, 2, 3
_TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64 = 4, 5, 6, 7
_TAG_F32, _TAG_F64, _TAG_BOOL = 8, 9, 10
_TAG_S, _TAG_U, _TAG_BLOB = 11, 12, 13

# descriptor field byte offsets (mirror of _NdarrayTableDescriptor)
_D_NROWS, _D_NCOLS, _D_ROW_STRIDE, _D_DATA_BASE = 0, 8, 16, 24
_D_COL_OFFSETS, _D_COL_TAGS, _D_COL_WIDTHS, _D_SCHEMA, _D_SCRATCH = 32, 40, 48, 56, 64

# vtab layout: base sqlite3_vtab is 24 bytes; the descriptor_ptr is appended at +24
_VTAB_DESC, _VTAB_SIZE = 24, 32
# cursor layout: pVtab(+0), descriptor(+8), rowid(+16), scratch(+24)
_CUR_PVTAB, _CUR_DESC, _CUR_ROWID, _CUR_SCRATCH = 0, 8, 16, 24
assert _VTAB_DESC + 8 == _VTAB_SIZE
assert _CUR_DESC == _CUR_PVTAB + 8 and _CUR_ROWID == _CUR_DESC + 8 and _CUR_SCRATCH == _CUR_ROWID + 8


@proxy(signatures.get("sqlite3_create_module"), jit_options={"cache": True})
def sqlite3_create_module(db, z_name, p_module, p_client_data):
    return _call_lib_func("sqlite3_create_module", (db, z_name, p_module, p_client_data))


@proxy(signatures.get("sqlite3_declare_vtab"), jit_options={"cache": True})
def sqlite3_declare_vtab(db, z_sql):
    return _call_lib_func("sqlite3_declare_vtab", (db, z_sql))


@proxy(signatures.get("sqlite3_malloc"), jit_options={"cache": True})
def sqlite3_malloc(n):
    return _call_lib_func("sqlite3_malloc", (n,))


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


class _NdarrayTableDescriptor(ctypes.Structure):
    _fields_ = [
        ("nrows", ctypes.c_int64),
        ("ncols", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
        ("row_stride", ctypes.c_int64),
        ("data_base", ctypes.c_int64),
        ("col_offsets", ctypes.c_int64),
        ("col_tags", ctypes.c_int64),
        ("col_widths", ctypes.c_int64),
        ("schema_ptr", ctypes.c_int64),
        ("scratch_bytes", ctypes.c_int64),
    ]


def _assert_descriptor_layout():
    f = _NdarrayTableDescriptor
    assert ctypes.sizeof(f) == 72
    assert (f.nrows.offset, f.ncols.offset, f.row_stride.offset, f.data_base.offset) == \
        (_D_NROWS, _D_NCOLS, _D_ROW_STRIDE, _D_DATA_BASE)
    assert (f.col_offsets.offset, f.col_tags.offset, f.col_widths.offset) == \
        (_D_COL_OFFSETS, _D_COL_TAGS, _D_COL_WIDTHS)
    assert (f.schema_ptr.offset, f.scratch_bytes.offset) == (_D_SCHEMA, _D_SCRATCH)


_assert_descriptor_layout()

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


class _BuiltDescriptor:
    """The ctypes descriptor plus every buffer whose pointer it holds."""
    __slots__ = ("c", "offsets", "tags", "widths", "schema",
                 "nrows", "ncols", "row_stride", "scratch_bytes", "arr")

    def __init__(self, c, offsets, tags, widths, schema, arr):
        self.c = c
        self.offsets = offsets
        self.tags = tags
        self.widths = widths
        self.schema = schema
        self.arr = arr
        self.nrows = c.nrows
        self.ncols = c.ncols
        self.row_stride = c.row_stride
        self.scratch_bytes = c.scratch_bytes


def _build_descriptor(arr, columns, text_as_blob):
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a numpy.ndarray, got %r" % (type(arr),))
    fields = arr.dtype.fields
    if fields is not None:
        if arr.ndim != 1:
            raise ValueError("structured array must be 1-D, got ndim=%d" % arr.ndim)
        names = list(arr.dtype.names)
        sub = [arr.dtype.fields[n][0] for n in names]
        offs = [arr.dtype.fields[n][1] for n in names]
        col_names = list(columns) if columns is not None else names
        if len(col_names) != len(names):
            raise ValueError("columns length %d != field count %d" % (len(col_names), len(names)))
    else:
        if arr.ndim != 2:
            raise ValueError("plain array must be 2-D, got ndim=%d" % arr.ndim)
        if columns is None or len(columns) != arr.shape[1]:
            raise ValueError("columns must list all %d column names for a 2-D array" % arr.shape[1])
        col_names = list(columns)
        sub = [arr.dtype] * arr.shape[1]
        offs = [j * arr.strides[1] for j in range(arr.shape[1])]

    if not col_names:
        raise ValueError("array must have at least one column")
    tags = [_col_tag(dt, text_as_blob) for dt in sub]
    widths = [int(dt.itemsize) for dt in sub]
    scratch = max([w + 1 for w, t in zip(widths, tags) if t == _TAG_U], default=0)
    if _CUR_SCRATCH + scratch > 2 ** 31 - 1:
        raise ValueError("unicode column too wide: per-cursor scratch buffer would overflow int32")

    offsets_buf = np.array(offs, dtype=np.int64)
    tags_buf = np.array(tags, dtype=np.int32)  # the xColumn cfunc reads int32 elements; do not widen
    widths_buf = np.array(widths, dtype=np.int64)
    cols_sql = ", ".join('"%s" %s' % (n.replace('"', '""'), _SQL_TYPE[t]) for n, t in zip(col_names, tags))
    schema = ("CREATE TABLE x(%s)" % cols_sql).encode("utf-8") + b"\x00"

    c = _NdarrayTableDescriptor()
    c.nrows = int(arr.shape[0])
    c.ncols = len(col_names)
    c.row_stride = int(arr.strides[0])
    c.data_base = arr.ctypes.data
    c.col_offsets = offsets_buf.ctypes.data
    c.col_tags = tags_buf.ctypes.data
    c.col_widths = widths_buf.ctypes.data
    c.schema_ptr = ctypes.cast(ctypes.c_char_p(schema), ctypes.c_void_p).value
    c.scratch_bytes = int(scratch)
    return _BuiltDescriptor(c, offsets_buf, tags_buf, widths_buf, schema, arr)


@cfunc(types.int32(types.intp, types.intp, types.int32, types.intp, types.intp, types.intp), cache=True)
def _xconnect(db, p_aux, argc, argv, pp_vtab, pz_err):
    vtab = 0
    try:
        schema_p = load_at(p_aux + _D_SCHEMA, int64)
        rc = sqlite3_declare_vtab(db, schema_p)
        if rc != SQLITE_OK:
            return rc
        vtab = sqlite3_malloc(int32(_VTAB_SIZE))
        if vtab == 0:
            return SQLITE_NOMEM
        store_at(vtab + 0, int64(0))
        store_at(vtab + 8, int64(0))
        store_at(vtab + 16, int64(0))
        store_at(vtab + _VTAB_DESC, int64(p_aux))
        slot = carray(_cast_int_to_void_p(pp_vtab), (1,), dtype=np.intp)
        slot[0] = vtab
        return SQLITE_OK
    except Exception:
        sqlite3_free(vtab)
        return SQLITE_ERROR


@cfunc(types.int32(types.intp, types.intp), cache=True)
def _xbestindex(vtab, idx_info):
    return SQLITE_OK


@cfunc(types.int32(types.intp), cache=True)
def _xdisconnect(vtab):
    try:
        sqlite3_free(vtab)
        return SQLITE_OK
    except Exception:
        return SQLITE_ERROR


@cfunc(types.int32(types.intp, types.intp), cache=True)
def _xopen(vtab, pp_cursor):
    cur = 0
    try:
        desc = load_at(vtab + _VTAB_DESC, int64)
        scratch = load_at(desc + _D_SCRATCH, int64)
        cur = sqlite3_malloc(int32(_CUR_SCRATCH + scratch))  # _build_descriptor caps scratch: int32 cast cannot overflow
        if cur == 0:
            return SQLITE_NOMEM
        store_at(cur + _CUR_PVTAB, int64(vtab))
        store_at(cur + _CUR_DESC, int64(desc))
        store_at(cur + _CUR_ROWID, int64(0))
        slot = carray(_cast_int_to_void_p(pp_cursor), (1,), dtype=np.intp)
        slot[0] = cur
        return SQLITE_OK
    except Exception:
        sqlite3_free(cur)
        return SQLITE_ERROR


@cfunc(types.int32(types.intp), cache=True)
def _xclose(cur):
    try:
        sqlite3_free(cur)
        return SQLITE_OK
    except Exception:
        return SQLITE_ERROR


@cfunc(types.int32(types.intp, types.int32, types.intp, types.int32, types.intp), cache=True)
def _xfilter(cur, idx_num, idx_str, argc, argv):
    store_at(cur + _CUR_ROWID, int64(0))
    return SQLITE_OK


@cfunc(types.int32(types.intp), cache=True)
def _xnext(cur):
    store_at(cur + _CUR_ROWID, load_at(cur + _CUR_ROWID, int64) + 1)
    return SQLITE_OK


@cfunc(types.int32(types.intp), cache=True)
def _xeof(cur):
    desc = load_at(cur + _CUR_DESC, int64)
    rowid = load_at(cur + _CUR_ROWID, int64)
    nrows = load_at(desc + _D_NROWS, int64)
    if rowid >= nrows:
        return 1
    return 0


@cfunc(types.int32(types.intp, types.intp), cache=True)
def _xrowid(cur, p_rowid):
    store_at(p_rowid, load_at(cur + _CUR_ROWID, int64))
    return SQLITE_OK


@cfunc(types.int32(types.intp, types.intp, types.int32), cache=True)
def _xcolumn(cur, ctx, j):
    try:
        desc = load_at(cur + _CUR_DESC, int64)
        rowid = load_at(cur + _CUR_ROWID, int64)
        ncols = load_at(desc + _D_NCOLS, int32)
        base = load_at(desc + _D_DATA_BASE, int64)
        row_stride = load_at(desc + _D_ROW_STRIDE, int64)
        offsets = carray(_cast_int_to_void_p(load_at(desc + _D_COL_OFFSETS, int64)), (ncols,), dtype=np.int64)
        tags = carray(_cast_int_to_void_p(load_at(desc + _D_COL_TAGS, int64)), (ncols,), dtype=np.int32)
        widths = carray(_cast_int_to_void_p(load_at(desc + _D_COL_WIDTHS, int64)), (ncols,), dtype=np.int64)
        addr = base + rowid * row_stride + offsets[j]
        tag = tags[j]
        if tag == _TAG_I8:
            sqlite3_result_int64(ctx, int64(load_unaligned(addr, int8)))
        elif tag == _TAG_I16:
            sqlite3_result_int64(ctx, int64(load_unaligned(addr, int16)))
        elif tag == _TAG_I32:
            sqlite3_result_int64(ctx, int64(load_unaligned(addr, int32)))
        elif tag == _TAG_I64:
            sqlite3_result_int64(ctx, load_unaligned(addr, int64))
        elif tag == _TAG_U8:
            sqlite3_result_int64(ctx, int64(load_unaligned(addr, uint8)))
        elif tag == _TAG_U16:
            sqlite3_result_int64(ctx, int64(load_unaligned(addr, uint16)))
        elif tag == _TAG_U32:
            sqlite3_result_int64(ctx, int64(load_unaligned(addr, uint32)))
        elif tag == _TAG_U64:
            sqlite3_result_int64(ctx, int64(load_unaligned(addr, uint64)))
        elif tag == _TAG_BOOL:
            sqlite3_result_int64(ctx, int64(1) if load_unaligned(addr, uint8) != 0 else int64(0))
        elif tag == _TAG_F32:
            sqlite3_result_double(ctx, float64(load_unaligned(addr, float32)))
        elif tag == _TAG_F64:
            sqlite3_result_double(ctx, load_unaligned(addr, float64))
        elif tag == _TAG_S:
            n = _nul_trimmed_len(addr, widths[j])
            sqlite3_result_text(ctx, addr, int32(n), SQLITE_TRANSIENT)
        elif tag == _TAG_BLOB:
            n = _nul_trimmed_len(addr, widths[j])
            sqlite3_result_blob(ctx, addr, int32(n), SQLITE_TRANSIENT)
        elif tag == _TAG_U:
            scratch = cur + _CUR_SCRATCH
            n = utf32_to_utf8(addr, widths[j] // 4, scratch)
            sqlite3_result_text(ctx, scratch, int32(n), SQLITE_TRANSIENT)
        return SQLITE_OK
    except Exception:
        sqlite3_result_error(ctx, get_unicode_data_p("error reading vtable column"), -1)
        return SQLITE_ERROR


class _Sqlite3Module(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in (
        "xCreate", "xConnect", "xBestIndex", "xDisconnect", "xDestroy",
        "xOpen", "xClose", "xFilter", "xNext", "xEof", "xColumn", "xRowid",
        "xUpdate", "xBegin", "xSync", "xCommit", "xRollback",
        "xFindFunction", "xRename")]
    # reassigned to prepend iVersion; ctypes reads only the final value
    _fields_ = [("iVersion", ctypes.c_int)] + _fields_


THE_MODULE = _Sqlite3Module()
THE_MODULE.iVersion = 1
THE_MODULE.xCreate = _xconnect.address
THE_MODULE.xConnect = _xconnect.address
THE_MODULE.xBestIndex = _xbestindex.address
THE_MODULE.xDisconnect = _xdisconnect.address
THE_MODULE.xDestroy = _xdisconnect.address
THE_MODULE.xOpen = _xopen.address
THE_MODULE.xClose = _xclose.address
THE_MODULE.xFilter = _xfilter.address
THE_MODULE.xNext = _xnext.address
THE_MODULE.xEof = _xeof.address
THE_MODULE.xColumn = _xcolumn.address
THE_MODULE.xRowid = _xrowid.address
_THE_MODULE_P = ctypes.addressof(THE_MODULE)
_CFUNCS = (_xconnect, _xbestindex, _xdisconnect, _xopen, _xclose,
           _xfilter, _xnext, _xeof, _xcolumn, _xrowid)


class _VTableHandle:
    """Keeps the array + descriptor + buffers alive. SQLite reads the array
    buffer directly; if this handle is GC'd the data frees and the next query
    reads freed memory. The caller MUST retain it."""
    __slots__ = ("_keep",)

    def __init__(self, *objs):
        self._keep = objs


def _raise_rc(db, name, rc):
    msg_p = sqlite3_errmsg(db)
    detail = ""
    if msg_p:
        detail = ": " + ctypes.cast(msg_p, ctypes.c_char_p).value.decode("utf-8", "replace")
    raise RuntimeError("register_table failed for %r (rc=%d)%s" % (name, rc, detail))


def register_table(db, name, arr, columns=None, *, text_as_blob=False):
    """Expose a numpy array as a read-only eponymous SQLite virtual table.

    The caller MUST retain the returned handle for as long as the table is used,
    and must not mutate or resize the array while the table is registered -- the
    view is zero-copy, so queries read the array's buffer directly.

    Registering a second table under an existing name follows SQLite's
    eponymous-module semantics: the later registration replaces the earlier one
    (it does not raise). Column names may be any string (they are quoted in the
    generated schema).

    Value semantics:

    - ``uint64`` values >= 2**63 are stored as SQLite's signed INTEGER and
      therefore wrap to negative.
    - Floats pass through as REAL, including +/-inf -- EXCEPT NaN: SQLite coerces
      a NaN ``REAL`` to SQL NULL (via ``sqlite3IsNaN``), so a NaN cell reads back
      as NULL, not as a REAL NaN. There is no other source of SQL NULL (numpy has
      no missing value), so every non-NaN cell is non-NULL.

    String columns:

    - ``text_as_blob`` affects only bytes (``'S'``) columns; unicode (``'U'``) is
      always TEXT. By default ``'S'`` becomes TEXT and its raw bytes pass through
      unvalidated -- pass ``text_as_blob=True`` for non-UTF-8 bytes so they are
      stored as BLOB rather than malformed TEXT.
    - Fixed-width ``'S'``/``'U'`` columns are NUL-padded by numpy; trailing NUL
      padding is trimmed on read while interior NULs are preserved.
    - A TEXT value with an interior NUL is stored faithfully (an explicit byte
      length is passed), but C-string readers and most SQL text functions
      truncate at the first NUL; read it via ``sqlite3_column_bytes`` + the
      text/blob pointer, or use ``text_as_blob=True`` for full fidelity.
    """
    built = _build_descriptor(arr, columns, text_as_blob)
    with c_string(name) as name_p:
        rc = sqlite3_create_module(db, name_p, _THE_MODULE_P, ctypes.addressof(built.c))
    if rc != SQLITE_OK:
        _raise_rc(db, name, rc)
    return _VTableHandle(built)
