"""Expose a numpy array as a read-only SQLite virtual table (register_table).

A single generic sqlite3_module (built once at import) serves every table; the
per-table base pointer / strides / dtype tags / schema live in a numpy
structured-array descriptor whose data pointer is passed as pClientData. The
sqlite3_vtab and sqlite3_vtab_cursor SQLite allocates through us are likewise
numpy structured dtypes, with their SQLite-owned base nested as field 'base'.
See https://www.sqlite.org/vtab.html.
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
from numbox.core.bindings._sqlite_exec import sqlite3_free, sqlite3_malloc
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
    _cast_int_to_void_p, get_unicode_data_p, load_unaligned, store_at,
)

__all__ = ["register_table"]

load_lib("sqlite3")

# dtype tags (col_tags[j])
_TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64 = 0, 1, 2, 3
_TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64 = 4, 5, 6, 7
_TAG_F32, _TAG_F64, _TAG_BOOL = 8, 9, 10
_TAG_S, _TAG_U, _TAG_BLOB = 11, 12, 13

# per-table descriptor passed as pClientData; the cfuncs read it field-by-name
# via carray(ptr, (1,), dtype=_DESC_DTYPE). The dtype is the single source of
# truth for the layout (numpy owns the offsets). ('_pad', 'i4') holds the
# alignment slot after the i4 ncols and is load-bearing -- do not drop it or
# pass align=True (either would silently shift every field offset).
_DESC_DTYPE = np.dtype([
    ("nrows", "i8"), ("ncols", "i4"), ("_pad", "i4"),
    ("row_stride", "i8"), ("data_base", "i8"),
    ("col_offsets", "i8"), ("col_tags", "i8"), ("col_widths", "i8"),
    ("schema_ptr", "i8"), ("scratch_bytes", "i8"),
])
assert _DESC_DTYPE.itemsize == 72

# The vtab and cursor SQLite allocates through us are C structs whose first
# member is the SQLite-owned base (sqlite3_vtab / sqlite3_vtab_cursor); we append
# our own members after it. Each is a numpy structured dtype with the base nested
# as field 'base', so the cfuncs address members by name instead of raw offsets
# (align=True reproduces the C padding, e.g. the int nRef in sqlite3_vtab).

# struct sqlite3_vtab { const sqlite3_module *pModule; int nRef; char *zErrMsg; }
# https://www.sqlite.org/c3ref/vtab.html -- SQLite owns/sets these fields.
_SQLITE3_VTAB_DTYPE = np.dtype([("pModule", "i8"), ("nRef", "i4"), ("zErrMsg", "i8")], align=True)
_VTAB_DTYPE = np.dtype([("base", _SQLITE3_VTAB_DTYPE), ("descriptor", "i8")], align=True)
assert _SQLITE3_VTAB_DTYPE.itemsize == 24 and _VTAB_DTYPE.itemsize == 32
_VTAB_SIZE = _VTAB_DTYPE.itemsize

# struct sqlite3_vtab_cursor { sqlite3_vtab *pVtab; }
# https://www.sqlite.org/c3ref/vtab_cursor.html -- one-member SQLite base. The
# per-cursor UTF-8 scratch buffer is allocated right after this fixed header,
# starting at offset _CUR_HEADER.
_SQLITE3_VTAB_CURSOR_DTYPE = np.dtype([("pVtab", "i8")])
_CUR_DTYPE = np.dtype([("base", _SQLITE3_VTAB_CURSOR_DTYPE), ("descriptor", "i8"), ("rowid", "i8")], align=True)
assert _CUR_DTYPE.itemsize == 24
_CUR_HEADER = _CUR_DTYPE.itemsize

# struct sqlite3_index_info -- https://www.sqlite.org/c3ref/index_info.html
# Modelled only through estimatedRows: xBestIndex writes the two cardinality
# outputs and leaves everything else at SQLite's zero/default init. Fields after
# estimatedRows (idxFlags 3.9.0, colUsed 3.10.0) are omitted -- never addressed;
# estimatedRows needs SQLite 3.8.2+, met by the >=3.34 support floor.
_IDX_INFO_DTYPE = np.dtype([
    ("nConstraint", "i4"), ("_pad0", "i4"), ("aConstraint", "i8"),
    ("nOrderBy", "i4"), ("_pad1", "i4"), ("aOrderBy", "i8"),
    ("aConstraintUsage", "i8"),
    ("idxNum", "i4"), ("_pad2", "i4"), ("idxStr", "i8"),
    ("needToFreeIdxStr", "i4"), ("orderByConsumed", "i4"),
    ("estimatedCost", "f8"), ("estimatedRows", "i8"),
])
assert _IDX_INFO_DTYPE.fields["estimatedCost"][1] == 64
assert _IDX_INFO_DTYPE.fields["estimatedRows"][1] == 72

# @cfunc takes a cache bool (not a jit_options dict like @njit/@proxy); thread
# the package-wide cache setting through it (default True).
_CACHE = jit_options.get("cache", True)


@proxy(signatures.get("sqlite3_create_module"), jit_options=jit_options)
def sqlite3_create_module(db, z_name, p_module, p_client_data):
    return _call_lib_func("sqlite3_create_module", (db, z_name, p_module, p_client_data))


@proxy(signatures.get("sqlite3_declare_vtab"), jit_options=jit_options)
def sqlite3_declare_vtab(db, z_sql):
    return _call_lib_func("sqlite3_declare_vtab", (db, z_sql))


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
    """The numpy structured-array descriptor plus every buffer whose pointer it holds."""
    __slots__ = ("c", "offsets", "tags", "widths", "schema",
                 "nrows", "ncols", "row_stride", "scratch_bytes", "arr")

    def __init__(self, c, offsets, tags, widths, schema, arr):
        self.c = c
        self.offsets = offsets
        self.tags = tags
        self.widths = widths
        self.schema = schema
        self.arr = arr
        self.nrows = int(c["nrows"][0])
        self.ncols = int(c["ncols"][0])
        self.row_stride = int(c["row_stride"][0])
        self.scratch_bytes = int(c["scratch_bytes"][0])


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
    if _CUR_HEADER + scratch > 2 ** 31 - 1:
        raise ValueError("unicode column too wide: per-cursor scratch buffer would overflow int32")

    offsets_buf = np.array(offs, dtype=np.int64)
    tags_buf = np.array(tags, dtype=np.int32)  # the xColumn cfunc reads int32 elements; do not widen
    widths_buf = np.array(widths, dtype=np.int64)
    cols_sql = ", ".join('"%s" %s' % (n.replace('"', '""'), _SQL_TYPE[t]) for n, t in zip(col_names, tags))
    schema = ("CREATE TABLE x(%s)" % cols_sql).encode("utf-8") + b"\x00"

    c = np.zeros(1, _DESC_DTYPE)
    c["nrows"] = int(arr.shape[0])
    c["ncols"] = len(col_names)
    c["row_stride"] = int(arr.strides[0])
    c["data_base"] = arr.ctypes.data
    c["col_offsets"] = offsets_buf.ctypes.data
    c["col_tags"] = tags_buf.ctypes.data
    c["col_widths"] = widths_buf.ctypes.data
    c["schema_ptr"] = ctypes.cast(ctypes.c_char_p(schema), ctypes.c_void_p).value
    c["scratch_bytes"] = int(scratch)
    return _BuiltDescriptor(c, offsets_buf, tags_buf, widths_buf, schema, arr)


@cfunc(types.int32(types.intp, types.intp, types.int32, types.intp, types.intp, types.intp), cache=_CACHE)
def _xconnect(db, p_aux, argc, argv, pp_vtab, pz_err):
    vtab = 0
    try:
        d = carray(_cast_int_to_void_p(p_aux), (1,), dtype=_DESC_DTYPE)
        rc = sqlite3_declare_vtab(db, d[0].schema_ptr)
        if rc != SQLITE_OK:
            return rc
        vtab = sqlite3_malloc(int32(_VTAB_SIZE))
        if vtab == 0:
            return SQLITE_NOMEM
        # canonical sqlite3_vtab init: zero the whole struct (sqlite3_malloc does
        # not), then set our appended member; SQLite owns/sets the base fields.
        raw = carray(_cast_int_to_void_p(vtab), (_VTAB_SIZE,), dtype=np.uint8)
        for i in range(_VTAB_SIZE):
            raw[i] = 0
        v = carray(_cast_int_to_void_p(vtab), (1,), dtype=_VTAB_DTYPE)
        v[0].descriptor = p_aux
        slot = carray(_cast_int_to_void_p(pp_vtab), (1,), dtype=np.intp)
        slot[0] = vtab
        return SQLITE_OK
    except Exception:
        sqlite3_free(vtab)
        return SQLITE_ERROR


@cfunc(types.int32(types.intp, types.intp), cache=_CACHE)
def _xbestindex(vtab, idx_info):
    # No constraints are usable (read-only full scan), so leave SQLite's zeroed
    # outputs as-is except the cardinality: feed the real row count so joins and
    # subqueries over a large table aren't mis-costed (SQLite defaults
    # estimatedRows to 25 and estimatedCost to a huge value).
    v = carray(_cast_int_to_void_p(vtab), (1,), dtype=_VTAB_DTYPE)
    d = carray(_cast_int_to_void_p(v[0].descriptor), (1,), dtype=_DESC_DTYPE)
    ii = carray(_cast_int_to_void_p(idx_info), (1,), dtype=_IDX_INFO_DTYPE)
    ii[0].estimatedRows = d[0].nrows
    ii[0].estimatedCost = float64(d[0].nrows)
    return SQLITE_OK


@cfunc(types.int32(types.intp), cache=_CACHE)
def _xdisconnect(vtab):
    try:
        sqlite3_free(vtab)
        return SQLITE_OK
    except Exception:
        return SQLITE_ERROR


@cfunc(types.int32(types.intp, types.intp), cache=_CACHE)
def _xopen(vtab, pp_cursor):
    cur = 0
    try:
        v = carray(_cast_int_to_void_p(vtab), (1,), dtype=_VTAB_DTYPE)
        desc = v[0].descriptor
        d = carray(_cast_int_to_void_p(desc), (1,), dtype=_DESC_DTYPE)
        scratch = d[0].scratch_bytes
        cur = sqlite3_malloc(int32(_CUR_HEADER + scratch))  # _build_descriptor caps scratch: int32 cast cannot overflow
        if cur == 0:
            return SQLITE_NOMEM
        c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
        c[0].base.pVtab = vtab
        c[0].descriptor = desc
        c[0].rowid = 0
        slot = carray(_cast_int_to_void_p(pp_cursor), (1,), dtype=np.intp)
        slot[0] = cur
        return SQLITE_OK
    except Exception:
        sqlite3_free(cur)
        return SQLITE_ERROR


@cfunc(types.int32(types.intp), cache=_CACHE)
def _xclose(cur):
    try:
        sqlite3_free(cur)
        return SQLITE_OK
    except Exception:
        return SQLITE_ERROR


@cfunc(types.int32(types.intp, types.int32, types.intp, types.int32, types.intp), cache=_CACHE)
def _xfilter(cur, idx_num, idx_str, argc, argv):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    c[0].rowid = 0
    return SQLITE_OK


@cfunc(types.int32(types.intp), cache=_CACHE)
def _xnext(cur):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    c[0].rowid = c[0].rowid + 1
    return SQLITE_OK


@cfunc(types.int32(types.intp), cache=_CACHE)
def _xeof(cur):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    d = carray(_cast_int_to_void_p(c[0].descriptor), (1,), dtype=_DESC_DTYPE)
    if c[0].rowid >= d[0].nrows:
        return 1
    return 0


@cfunc(types.int32(types.intp, types.intp), cache=_CACHE)
def _xrowid(cur, p_rowid):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    store_at(p_rowid, c[0].rowid)
    return SQLITE_OK


@cfunc(types.int32(types.intp, types.intp, types.int32), cache=_CACHE)
def _xcolumn(cur, ctx, j):
    try:
        c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
        rowid = c[0].rowid
        d = carray(_cast_int_to_void_p(c[0].descriptor), (1,), dtype=_DESC_DTYPE)
        ncols = d[0].ncols
        base = d[0].data_base
        row_stride = d[0].row_stride
        offsets = carray(_cast_int_to_void_p(d[0].col_offsets), (ncols,), dtype=np.int64)
        tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=np.int32)
        widths = carray(_cast_int_to_void_p(d[0].col_widths), (ncols,), dtype=np.int64)
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
            scratch = cur + _CUR_HEADER
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
        rc = sqlite3_create_module(db, name_p, _THE_MODULE_P, built.c.ctypes.data)
    if rc != SQLITE_OK:
        _raise_rc(db, name, rc)
    return _VTableHandle(built)
