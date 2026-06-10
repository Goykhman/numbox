"""Expose a numpy array as a read-only SQLite virtual table (register_table).

A single generic sqlite3_module (built once at import) serves every table; the
per-table base pointer / strides / dtype tags / schema live in a numpy
structured-array descriptor whose data pointer is passed as pClientData.

The sqlite3_vtab and sqlite3_vtab_cursor that our xCreate/xOpen callbacks
allocate are C structs whose first member is the SQLite-owned base; we append
our own members after it. Each is modelled as a numpy structured dtype with that
base nested as field 'base', so the cfuncs address members by name rather than
raw offsets. See https://www.sqlite.org/vtab.html.
"""
import ctypes

import numpy as np
from numba import carray, cfunc, njit, types
from numba.core.types import (
    int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    float32, float64,
)

from numbox.core.bindings import (
    SQLITE_OK, SQLITE_STATIC, SQLITE_TRANSIENT, SQLITE_ERROR, SQLITE_NOMEM,
    SQLITE_INDEX_CONSTRAINT_EQ, SQLITE_INDEX_CONSTRAINT_GT,
    SQLITE_INDEX_CONSTRAINT_GE, SQLITE_INDEX_CONSTRAINT_LT,
    SQLITE_INDEX_CONSTRAINT_LE,
)
from numbox.core.bindings import (
    sqlite3_errmsg, sqlite3_free, sqlite3_malloc,
    sqlite3_result_int64, sqlite3_result_double,
    sqlite3_result_text, sqlite3_result_blob, sqlite3_result_error,
    sqlite3_value_double, sqlite3_value_int64,
)
from numbox.core.bindings._sqlite_typemap import (
    _TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64, _TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64,
    _TAG_F32, _TAG_F64, _TAG_BOOL, _TAG_S, _TAG_U, _TAG_BLOB,
    _SQL_TYPE, _col_tag, utf32_to_utf8, _nul_trimmed_len, tags_buf_t,
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

# per-table descriptor passed as pClientData; the cfuncs read it field-by-name
# via carray(ptr, (1,), dtype=_DESC_DTYPE). The dtype is the single source of
# truth for the layout (numpy owns the offsets); align=True inserts the natural
# padding after the i4 ncols so every i8 field stays 8-aligned.
_DESC_DTYPE = np.dtype([
    ("nrows", "i8"), ("ncols", "i4"),
    ("row_stride", "i8"), ("data_base", "i8"),
    ("col_offsets", "i8"), ("col_tags", "i8"), ("col_widths", "i8"),
    ("schema_ptr", "i8"), ("scratch_bytes", "i8"),
], align=True)
assert _DESC_DTYPE.itemsize == 72

# struct sqlite3_vtab { const sqlite3_module *pModule; int nRef; char *zErrMsg; }
# https://www.sqlite.org/c3ref/vtab.html -- SQLite owns/sets these fields.
_SQLITE3_VTAB_DTYPE = np.dtype([("pModule", "i8"), ("nRef", "i4"), ("zErrMsg", "i8")], align=True)
_VTAB_DTYPE = np.dtype([("base", _SQLITE3_VTAB_DTYPE), ("descriptor", "i8")], align=True)
assert _SQLITE3_VTAB_DTYPE.itemsize == 24 and _VTAB_DTYPE.itemsize == 32
_VTAB_SIZE = _VTAB_DTYPE.itemsize

# struct sqlite3_vtab_cursor { sqlite3_vtab *pVtab; }
# https://www.sqlite.org/c3ref/vtab_cursor.html -- one-member SQLite base.
_SQLITE3_VTAB_CURSOR_DTYPE = np.dtype([("pVtab", "i8")])
_CUR_DTYPE = np.dtype([
    ("base", _SQLITE3_VTAB_CURSOR_DTYPE), ("descriptor", "i8"), ("rowid", "i8"), ("scratch_p", "i8"),
    ("pred_p", "i8"), ("n_pred", "i8"),
], align=True)
assert _CUR_DTYPE.itemsize == 48
_CUR_SIZE = _CUR_DTYPE.itemsize

# struct sqlite3_index_info -- https://www.sqlite.org/c3ref/index_info.html
# Modelled only through estimatedRows: later fields (idxFlags 3.9.0, colUsed
# 3.10.0) are never addressed. estimatedRows needs SQLite 3.8.2+, met by the
# >=3.34 support floor.
_IDX_INFO_DTYPE = np.dtype([
    ("nConstraint", "i4"), ("aConstraint", "i8"), ("nOrderBy", "i4"),
    ("aOrderBy", "i8"), ("aConstraintUsage", "i8"), ("idxNum", "i4"),
    ("idxStr", "i8"), ("needToFreeIdxStr", "i4"), ("orderByConsumed", "i4"),
    ("estimatedCost", "f8"), ("estimatedRows", "i8"),
], align=True)
assert _IDX_INFO_DTYPE.fields["estimatedCost"][1] == 64
assert _IDX_INFO_DTYPE.fields["estimatedRows"][1] == 72

# Element layouts of the three arrays sqlite3_index_info points at. align=True
# reproduces the C padding so each element matches sqlite3.h byte-for-byte:
#   struct sqlite3_index_constraint { int iColumn; unsigned char op, usable; int iTermOffset; }   -> 12
#   struct sqlite3_index_constraint_usage { int argvIndex; unsigned char omit; }                  -> 8
#   struct sqlite3_index_orderby { int iColumn; unsigned char desc; }                             -> 8
_CONSTRAINT_DTYPE = np.dtype([("iColumn", "i4"), ("op", "u1"), ("usable", "u1"), ("iTermOffset", "i4")], align=True)
_USAGE_DTYPE = np.dtype([("argvIndex", "i4"), ("omit", "u1")], align=True)
_ORDERBY_DTYPE = np.dtype([("iColumn", "i4"), ("desc", "u1")], align=True)
assert _CONSTRAINT_DTYPE.itemsize == 12 and _USAGE_DTYPE.itemsize == 8 and _ORDERBY_DTYPE.itemsize == 8

# A claimed (col, op, value) predicate, carried per-cursor in pred_p (n_pred of
# them). xBestIndex serialises the (col, op) pairs into the idxStr channel;
# xFilter fills ival or fval (per the column's int/float domain) from the
# matching argv[] sqlite3_value, setting is_int to pick the comparison domain.
_PRED_DTYPE = np.dtype([("col", "i4"), ("op", "i4"), ("is_int", "i4"),
                        ("ival", "i8"), ("fval", "f8")], align=True)
_PRED_SIZE = _PRED_DTYPE.itemsize

# The (col, op) pair xBestIndex serialises per claimed constraint into the
# idxStr side channel; xFilter reads it back through the same dtype.
_SPEC_DTYPE = np.dtype([("col", "i4"), ("op", "i4")], align=True)
_SPEC_SIZE = _SPEC_DTYPE.itemsize
assert _SPEC_SIZE == 8

_CACHE = jit_options.get("cache", True)


@proxy(signatures.get("sqlite3_create_module_v2"), jit_options=jit_options)
def sqlite3_create_module_v2(db, z_name, p_module, p_client_data, x_destroy):
    return _call_lib_func("sqlite3_create_module_v2", (db, z_name, p_module, p_client_data, x_destroy))


_DATA_ANCHOR = {}


def _xdestroy_py(p_aux):
    _DATA_ANCHOR.pop(p_aux, None)


_XDESTROY_CFUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(_xdestroy_py)
_XDESTROY_ADDR = ctypes.cast(_XDESTROY_CFUNC, ctypes.c_void_p).value


def _register_with_destroy(db, name, module_p, client_ptr, handle):
    _DATA_ANCHOR[client_ptr] = handle
    with c_string(name) as name_p:
        rc = sqlite3_create_module_v2(db, name_p, module_p, client_ptr, _XDESTROY_ADDR)
    if rc != SQLITE_OK:
        _DATA_ANCHOR.pop(client_ptr, None)
        _raise_rc(db, name, rc)


@proxy(signatures.get("sqlite3_declare_vtab"), jit_options=jit_options)
def sqlite3_declare_vtab(db, z_sql):
    return _call_lib_func("sqlite3_declare_vtab", (db, z_sql))


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

    offsets_buf = np.array(offs, dtype=np.int64)
    tags_buf = np.array(tags, dtype=tags_buf_t)
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
        # no zeroing needed: SQLite memsets the sqlite3_vtab base after this
        # callback returns; descriptor is the only member we own.
        v = carray(_cast_int_to_void_p(vtab), (1,), dtype=_VTAB_DTYPE)
        v[0].descriptor = p_aux
        slot = carray(_cast_int_to_void_p(pp_vtab), (1,), dtype=np.intp)
        slot[0] = vtab
        return SQLITE_OK
    except Exception:
        sqlite3_free(vtab)
        return SQLITE_ERROR


@njit(**jit_options)
def _is_numeric_tag(tag):
    return tag <= _TAG_BOOL


@njit(**jit_options)
def _is_int_tag(tag):
    return tag <= _TAG_U64


@njit(**jit_options)
def _is_supported_op(op):
    return (op == SQLITE_INDEX_CONSTRAINT_EQ or op == SQLITE_INDEX_CONSTRAINT_GT
            or op == SQLITE_INDEX_CONSTRAINT_GE or op == SQLITE_INDEX_CONSTRAINT_LT
            or op == SQLITE_INDEX_CONSTRAINT_LE)


@cfunc(types.int32(types.intp, types.intp), cache=_CACHE)
def _xbestindex(vtab, idx_info):
    # Claim every usable _is_supported_op constraint on a numeric column: assign it an
    # argvIndex (so xFilter receives its value) and serialise the (col, op) pair
    # into idxStr (the xBestIndex -> xFilter side channel). Cardinality is
    # reported regardless (joins/subqueries otherwise mis-cost; SQLite defaults
    # estimatedRows to 25).
    idx_p = 0
    try:
        v = carray(_cast_int_to_void_p(vtab), (1,), dtype=_VTAB_DTYPE)
        d = carray(_cast_int_to_void_p(v[0].descriptor), (1,), dtype=_DESC_DTYPE)
        ii = carray(_cast_int_to_void_p(idx_info), (1,), dtype=_IDX_INFO_DTYPE)
        ncols = d[0].ncols
        tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=tags_buf_t)
        n_constraint = ii[0].nConstraint
        cons = carray(_cast_int_to_void_p(ii[0].aConstraint), (n_constraint,), dtype=_CONSTRAINT_DTYPE)
        usage = carray(_cast_int_to_void_p(ii[0].aConstraintUsage), (n_constraint,), dtype=_USAGE_DTYPE)

        idx_p = sqlite3_malloc(int32(n_constraint * _SPEC_SIZE)) if n_constraint > 0 else 0
        if n_constraint > 0 and idx_p == 0:
            return SQLITE_NOMEM
        spec = carray(_cast_int_to_void_p(idx_p), (n_constraint,), dtype=_SPEC_DTYPE)

        nbound = 0
        for i in range(n_constraint):
            col = cons[i].iColumn
            op = cons[i].op
            if cons[i].usable != 0 and _is_supported_op(op) and 0 <= col < ncols and _is_numeric_tag(tags[col]):
                usage[i].argvIndex = int32(nbound + 1)
                # omit MUST stay 0: SQLite re-checks every surfaced row, the
                # correctness net behind the cursor's pruning. Keep it 0 even
                # though _row_matches now prunes exactly (int64 for integer
                # columns) so future predicate widening can't silently drop rows.
                usage[i].omit = 0
                spec[nbound].col = int32(col)
                spec[nbound].op = int32(op)
                nbound += 1

        ii[0].idxNum = int32(nbound)
        if nbound > 0:
            ii[0].idxStr = idx_p
            ii[0].needToFreeIdxStr = int32(1)
            idx_p = 0  # SQLite owns it now; the except handler must not free it
        else:
            sqlite3_free(idx_p)
            idx_p = 0

        nrows = d[0].nrows
        # heuristic: full scan if no predicates, else nrows/(nbound+1)+1 (floor, min 1)
        ii[0].estimatedRows = nrows if nbound == 0 else nrows // (nbound + 1) + 1
        ii[0].estimatedCost = float64(nrows)
        return SQLITE_OK
    except Exception:
        sqlite3_free(idx_p)
        return SQLITE_ERROR


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
    scratch_p = 0
    try:
        v = carray(_cast_int_to_void_p(vtab), (1,), dtype=_VTAB_DTYPE)
        desc = v[0].descriptor
        d = carray(_cast_int_to_void_p(desc), (1,), dtype=_DESC_DTYPE)
        scratch = d[0].scratch_bytes
        cur = sqlite3_malloc(int32(_CUR_SIZE))
        if cur == 0:
            return SQLITE_NOMEM
        if scratch > 0:
            scratch_p = sqlite3_malloc(int32(scratch))  # scratch <= numpy's int32 'U' itemsize cap: cast is safe
            if scratch_p == 0:
                sqlite3_free(cur)
                return SQLITE_NOMEM
        c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
        c[0].base.pVtab = vtab
        c[0].descriptor = desc
        c[0].rowid = 0
        c[0].scratch_p = scratch_p
        c[0].pred_p = 0
        c[0].n_pred = 0
        slot = carray(_cast_int_to_void_p(pp_cursor), (1,), dtype=np.intp)
        slot[0] = cur
        return SQLITE_OK
    except Exception:
        sqlite3_free(scratch_p)
        sqlite3_free(cur)
        return SQLITE_ERROR


@cfunc(types.int32(types.intp), cache=_CACHE)
def _xclose(cur):
    try:
        c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
        sqlite3_free(c[0].pred_p)
        sqlite3_free(c[0].scratch_p)
        sqlite3_free(cur)
        return SQLITE_OK
    except Exception:
        return SQLITE_ERROR


@njit(**jit_options)
def _cell_value_f64(d, rowid, col):
    """Read the cell at (rowid, col) as float64, mirroring _xcolumn's full tag
    ladder (same addr math, same load_unaligned widths). xBestIndex only
    claims tags up to _TAG_BOOL; the string/blob tags fall through to 0."""
    ncols = d[0].ncols
    base = d[0].data_base
    row_stride = d[0].row_stride
    offsets = carray(_cast_int_to_void_p(d[0].col_offsets), (ncols,), dtype=np.int64)
    tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=tags_buf_t)
    addr = base + rowid * row_stride + offsets[col]
    tag = tags[col]
    if tag == _TAG_I8:
        return float64(load_unaligned(addr, int8))
    elif tag == _TAG_I16:
        return float64(load_unaligned(addr, int16))
    elif tag == _TAG_I32:
        return float64(load_unaligned(addr, int32))
    elif tag == _TAG_I64:
        return float64(load_unaligned(addr, int64))
    elif tag == _TAG_U8:
        return float64(load_unaligned(addr, uint8))
    elif tag == _TAG_U16:
        return float64(load_unaligned(addr, uint16))
    elif tag == _TAG_U32:
        return float64(load_unaligned(addr, uint32))
    elif tag == _TAG_U64:
        return float64(load_unaligned(addr, uint64))
    elif tag == _TAG_BOOL:
        return float64(1) if load_unaligned(addr, uint8) != 0 else float64(0)
    elif tag == _TAG_F32:
        return float64(load_unaligned(addr, float32))
    elif tag == _TAG_F64:
        return load_unaligned(addr, float64)
    return float64(0)


@njit(**jit_options)
def _cell_value_i64(d, rowid, col):
    """Read an integer cell at (rowid, col) as int64, mirroring _xcolumn's
    sqlite3_result_int64 (uint64 wrapped to the same int64 SQLite sees)."""
    ncols = d[0].ncols
    base = d[0].data_base
    row_stride = d[0].row_stride
    offsets = carray(_cast_int_to_void_p(d[0].col_offsets), (ncols,), dtype=np.int64)
    tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=tags_buf_t)
    addr = base + rowid * row_stride + offsets[col]
    tag = tags[col]
    if tag == _TAG_I8:
        return int64(load_unaligned(addr, int8))
    elif tag == _TAG_I16:
        return int64(load_unaligned(addr, int16))
    elif tag == _TAG_I32:
        return int64(load_unaligned(addr, int32))
    elif tag == _TAG_I64:
        return load_unaligned(addr, int64)
    elif tag == _TAG_U8:
        return int64(load_unaligned(addr, uint8))
    elif tag == _TAG_U16:
        return int64(load_unaligned(addr, uint16))
    elif tag == _TAG_U32:
        return int64(load_unaligned(addr, uint32))
    elif tag == _TAG_U64:
        return int64(load_unaligned(addr, uint64))
    return int64(0)


@njit(**jit_options)
def _cmp(op, cv, rv):
    """Compare cv and rv in their native (call-site) type.

    The int and float call sites stay separate so numba never phi-unifies an
    int64 cell up to float64: above 2**53 that widening is lossy, and a
    predicate evaluated on the rounded value prunes rows the omit=0 re-check
    never gets to see (SQLite only re-checks rows the cursor surfaces).
    """
    if op == SQLITE_INDEX_CONSTRAINT_EQ:
        return cv == rv
    elif op == SQLITE_INDEX_CONSTRAINT_GT:
        return cv > rv
    elif op == SQLITE_INDEX_CONSTRAINT_GE:
        return cv >= rv
    elif op == SQLITE_INDEX_CONSTRAINT_LT:
        return cv < rv
    return cv <= rv


@njit(**jit_options)
def _row_matches(cur):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    if c[0].n_pred == 0:
        return True
    d = carray(_cast_int_to_void_p(c[0].descriptor), (1,), dtype=_DESC_DTYPE)
    preds = carray(_cast_int_to_void_p(c[0].pred_p), (c[0].n_pred,), dtype=_PRED_DTYPE)
    for k in range(c[0].n_pred):
        op = preds[k].op
        if preds[k].is_int != 0:
            ok = _cmp(op, _cell_value_i64(d, c[0].rowid, preds[k].col), preds[k].ival)
        else:
            ok = _cmp(op, _cell_value_f64(d, c[0].rowid, preds[k].col), preds[k].fval)
        if not ok:
            return False
    return True


@njit(**jit_options)
def _seek_match(cur):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    d = carray(_cast_int_to_void_p(c[0].descriptor), (1,), dtype=_DESC_DTYPE)
    while c[0].rowid < d[0].nrows and not _row_matches(cur):
        c[0].rowid = c[0].rowid + 1


@cfunc(types.int32(types.intp, types.int32, types.intp, types.int32, types.intp), cache=_CACHE)
def _xfilter(cur, idx_num, idx_str, argc, argv):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    sqlite3_free(c[0].pred_p)
    c[0].pred_p = 0
    c[0].n_pred = 0
    c[0].rowid = 0
    try:
        if idx_num > 0 and argc > 0:
            pred_p = sqlite3_malloc(int32(argc * _PRED_SIZE))
            if pred_p == 0:
                return SQLITE_NOMEM
            c[0].pred_p = pred_p
            c[0].n_pred = argc
            preds = carray(_cast_int_to_void_p(pred_p), (argc,), dtype=_PRED_DTYPE)
            spec = carray(_cast_int_to_void_p(idx_str), (argc,), dtype=_SPEC_DTYPE)
            vals = carray(_cast_int_to_void_p(argv), (argc,), dtype=np.intp)
            d = carray(_cast_int_to_void_p(c[0].descriptor), (1,), dtype=_DESC_DTYPE)
            ncols = d[0].ncols
            col_tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=tags_buf_t)
            for k in range(argc):
                col = spec[k].col
                preds[k].col = col
                preds[k].op = spec[k].op
                if _is_int_tag(col_tags[col]):
                    preds[k].is_int = 1
                    preds[k].ival = sqlite3_value_int64(vals[k])
                    preds[k].fval = 0.0
                else:
                    preds[k].is_int = 0
                    preds[k].ival = 0
                    preds[k].fval = sqlite3_value_double(vals[k])
        _seek_match(cur)
        return SQLITE_OK
    except Exception:
        sqlite3_free(c[0].pred_p)
        c[0].pred_p = 0
        c[0].n_pred = 0
        return SQLITE_ERROR


@cfunc(types.int32(types.intp), cache=_CACHE)
def _xnext(cur):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    c[0].rowid = c[0].rowid + 1
    _seek_match(cur)
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
        tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=tags_buf_t)
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
            # S/BLOB results point into the registered array, which outlives
            # the statement (_DATA_ANCHOR + the no-mutation contract), so
            # STATIC hands SQLite the pointer zero-copy. U must stay TRANSIENT:
            # it serves the per-cursor scratch, overwritten by the next xColumn.
            n = _nul_trimmed_len(addr, widths[j])
            sqlite3_result_text(ctx, addr, int32(n), SQLITE_STATIC)
        elif tag == _TAG_BLOB:
            n = _nul_trimmed_len(addr, widths[j])
            sqlite3_result_blob(ctx, addr, int32(n), SQLITE_STATIC)
        elif tag == _TAG_U:
            scratch = c[0].scratch_p
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
    reads freed memory. The keep-alive lives in the module-level
    ``_DATA_ANCHOR``, released by SQLite via ``xDestroy``."""
    __slots__ = ("_keep",)

    def __init__(self, *objs):
        self._keep = objs


def _raise_rc(db, name, rc):
    msg_p = sqlite3_errmsg(db)
    detail = ""
    if msg_p:
        detail = ": " + ctypes.cast(msg_p, ctypes.c_char_p).value.decode("utf-8", "replace")
    raise RuntimeError("registration failed for %r (rc=%d)%s" % (name, rc, detail))


def register_table(db, name, arr, columns=None, *, text_as_blob=False):
    """Expose a numpy array as a read-only eponymous SQLite virtual table.

    The registration's keep-alive lives in the module-level ``_DATA_ANCHOR``
    and is released by SQLite via ``xDestroy`` (on connection close
    or re-registration of the same name). The caller must not
    mutate or resize the array while the table is registered -- the view is
    zero-copy, so queries read the array's buffer directly (numeric reads
    alias it, and ``'S'``/BLOB values are handed to SQLite as
    ``SQLITE_STATIC`` pointers into it).

    Registering a second table under an existing name follows SQLite's
    module-registration semantics: the later registration replaces the earlier one
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
    handle = _VTableHandle(built)
    _register_with_destroy(db, name, _THE_MODULE_P, built.c.ctypes.data, handle)
