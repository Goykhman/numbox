"""Expose a NumPy array, or a mapping of 1-D arrays, as a read-only SQLite virtual table.

:func:`register_table` registers the data under a name on a connection. SQL then queries the name as a table, and no
data is copied. :mod:`~numbox.core.bindings.sqlite.tvf` reuses the module's NumPy layouts of SQLite's virtual table
structs, its function that sets a cell as a column's result, and its registration. The registration keeps a table's
data alive while SQLite can read it.
"""
import ctypes
from collections.abc import Mapping

import numpy as np
from numba import carray, cfunc, njit, types
from numba.core.types import (
    int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    float32, float64,
)

from numbox.core.bindings.sqlite.constants import (
    SQLITE_OK, SQLITE_STATIC, SQLITE_TRANSIENT, SQLITE_ERROR, SQLITE_NOMEM, SQLITE_FLOAT,
    SQLITE_INDEX_CONSTRAINT_EQ, SQLITE_INDEX_CONSTRAINT_GT, SQLITE_INDEX_CONSTRAINT_GE,
    SQLITE_INDEX_CONSTRAINT_LT, SQLITE_INDEX_CONSTRAINT_LE,
)
from numbox.core.bindings.sqlite.conn import sqlite3_errmsg
from numbox.core.bindings.sqlite.exec import sqlite3_free, sqlite3_malloc
from numbox.core.bindings.sqlite.result import (
    sqlite3_result_int64, sqlite3_result_double, sqlite3_result_text, sqlite3_result_blob,
    sqlite3_result_error,
)
from numbox.core.bindings.sqlite.value import sqlite3_value_double, sqlite3_value_int64, sqlite3_value_numeric_type
from numbox.core.bindings.sqlite._typemap import (
    _TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64, _TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64,
    _TAG_F32, _TAG_F64, _TAG_BOOL, _TAG_S, _TAG_U, _TAG_BLOB,
    _SQL_TYPE, _col_tag, utf32_to_utf8, _nul_trimmed_len, tags_buf_t,
    _INLINE_JIT_OPTIONS,
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

# Each table has a descriptor, a one-element NumPy structured array. It holds the row and column counts and the
# address of the schema text. It also holds the addresses of four buffers with one entry per column: base addresses,
# strides, type tags and widths. The last field is the size of the scratch buffer for 'U' cells. SQLite passes the
# descriptor's address to xConnect as the module's client data (pClientData). The callbacks read the descriptor field
# by name through carray(ptr, (1,), dtype=_DESC_DTYPE). This dtype alone therefore defines the layout, and NumPy
# computes the offsets. align=True inserts the natural padding after the i4 ncols, so that every i8 field stays
# 8-byte aligned.
_DESC_DTYPE = np.dtype([
    ("nrows", "i8"), ("ncols", "i4"),
    ("col_bases", "i8"), ("col_strides", "i8"),
    ("col_tags", "i8"), ("col_widths", "i8"),
    ("schema_ptr", "i8"), ("scratch_bytes", "i8"),
], align=True)
assert _DESC_DTYPE.itemsize == 64

# The sqlite3_vtab and sqlite3_vtab_cursor that xConnect and xOpen allocate are C structs whose first member is
# SQLite's base struct, followed by numbox's own members (https://www.sqlite.org/vtab.html). A NumPy structured dtype
# describes each of them, with the base as its field "base".
# struct sqlite3_vtab { const sqlite3_module *pModule; int nRef; char *zErrMsg; } is the base of a table
# (https://www.sqlite.org/c3ref/vtab.html). SQLite owns these fields and sets them.
_SQLITE3_VTAB_DTYPE = np.dtype([("pModule", "i8"), ("nRef", "i4"), ("zErrMsg", "i8")], align=True)
_VTAB_DTYPE = np.dtype([("base", _SQLITE3_VTAB_DTYPE), ("descriptor", "i8")], align=True)
assert _SQLITE3_VTAB_DTYPE.itemsize == 24 and _VTAB_DTYPE.itemsize == 32
_VTAB_SIZE = _VTAB_DTYPE.itemsize

# struct sqlite3_vtab_cursor { sqlite3_vtab *pVtab; } is the base of a cursor, with one member
# (https://www.sqlite.org/c3ref/vtab_cursor.html).
_SQLITE3_VTAB_CURSOR_DTYPE = np.dtype([("pVtab", "i8")])
_CUR_DTYPE = np.dtype([
    ("base", _SQLITE3_VTAB_CURSOR_DTYPE), ("descriptor", "i8"), ("rowid", "i8"), ("scratch_p", "i8"),
    ("pred_p", "i8"), ("n_pred", "i8"),
], align=True)
assert _CUR_DTYPE.itemsize == 48
_CUR_SIZE = _CUR_DTYPE.itemsize

# struct sqlite3_index_info (https://www.sqlite.org/c3ref/index_info.html), described up to estimatedRows. The code
# never reads or writes the later fields, idxFlags (SQLite 3.9.0) and colUsed (3.10.0). estimatedRows needs SQLite
# 3.8.2 or later, and 3.34, the oldest version in numbox's test matrix, has it.
_IDX_INFO_DTYPE = np.dtype([
    ("nConstraint", "i4"), ("aConstraint", "i8"), ("nOrderBy", "i4"),
    ("aOrderBy", "i8"), ("aConstraintUsage", "i8"), ("idxNum", "i4"),
    ("idxStr", "i8"), ("needToFreeIdxStr", "i4"), ("orderByConsumed", "i4"),
    ("estimatedCost", "f8"), ("estimatedRows", "i8"),
], align=True)
assert _IDX_INFO_DTYPE.fields["estimatedCost"][1] == 64
assert _IDX_INFO_DTYPE.fields["estimatedRows"][1] == 72

# The element layouts of the three arrays that sqlite3_index_info points to. align=True reproduces the C padding, so
# each element matches sqlite3.h byte for byte.
#   struct sqlite3_index_constraint { int iColumn; unsigned char op, usable; int iTermOffset; } has 12 bytes.
#   struct sqlite3_index_constraint_usage { int argvIndex; unsigned char omit; } has 8 bytes.
#   struct sqlite3_index_orderby { int iColumn; unsigned char desc; } has 8 bytes.
_CONSTRAINT_DTYPE = np.dtype([("iColumn", "i4"), ("op", "u1"), ("usable", "u1"), ("iTermOffset", "i4")], align=True)
_USAGE_DTYPE = np.dtype([("argvIndex", "i4"), ("omit", "u1")], align=True)
_ORDERBY_DTYPE = np.dtype([("iColumn", "i4"), ("desc", "u1")], align=True)
assert _CONSTRAINT_DTYPE.itemsize == 12 and _USAGE_DTYPE.itemsize == 8 and _ORDERBY_DTYPE.itemsize == 8

# A constraint that xBestIndex claimed, as (col, op, value); each cursor keeps n_pred of them at pred_p. xBestIndex
# writes the (col, op) pairs into idxStr. xFilter fills ival or fval from the matching sqlite3_value of argv, and it
# sets is_int to the comparison mode: 1 compares as int64, 0 as float64, and 2 compares an integer column with a REAL
# value exactly.
_PRED_DTYPE = np.dtype([("col", "i4"), ("op", "i4"), ("is_int", "i4"),
                        ("ival", "i8"), ("fval", "f8")], align=True)
_PRED_SIZE = _PRED_DTYPE.itemsize

# The (col, op) pair that xBestIndex writes into idxStr for each constraint that it claims. SQLite passes idxStr on to
# xFilter, which reads the pairs back through the same dtype.
_SPEC_DTYPE = np.dtype([("col", "i4"), ("op", "i4")], align=True)
_SPEC_SIZE = _SPEC_DTYPE.itemsize
assert _SPEC_SIZE == 8

# The callbacks below compile with the cache option alone, taken from numbox's jit options and treated as enabled when
# the key is absent; no other jit option reaches them. The jitted helpers that they call compile under all of
# numbox's jit options.
_CACHE = jit_options.get("cache", True)


@proxy(signatures.get("sqlite3_create_module_v2"), jit_options=jit_options)
def sqlite3_create_module_v2(db, z_name, p_module, p_client_data, x_destroy):
    """Register the virtual table module at ``p_module`` under the UTF-8 name at ``z_name`` on the connection ``db``.

    C reference: `sqlite3_create_module_v2 <https://sqlite.org/c3ref/create_module.html>`_.

    ``p_module`` is the address of a ``sqlite3_module`` struct, which SQLite keeps using without a copy, so the struct
    must stay alive while the module is registered. SQLite passes ``p_client_data`` to the module's ``xCreate`` and
    ``xConnect``. Unless ``x_destroy`` is 0, SQLite calls it with ``p_client_data`` when it no longer needs the client
    data, and also when the registration fails. A ``p_module`` of 0 drops the module of that name. Returns
    ``SQLITE_OK`` or an error code.
    """
    return _call_lib_func("sqlite3_create_module_v2", (db, z_name, p_module, p_client_data, x_destroy))


# _DATA_ANCHOR maps the client-data address of each registration to the handle that keeps the registration's data
# alive. _register_with_destroy passes _XDESTROY_ADDR to sqlite3_create_module_v2 as the destructor of the client
# data. SQLite calls it with that address when it no longer needs the module, that is when the connection closes or
# when the name is registered again, and the destructor drops the handle.
_DATA_ANCHOR = {}


def _xdestroy_py(p_aux):
    _DATA_ANCHOR.pop(p_aux, None)


# A ctypes callback must stay referenced for as long as C code can call it, because ctypes frees the callback's code
# when the object is collected, so the object lives at module level.
_XDESTROY_CFUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(_xdestroy_py)
_XDESTROY_ADDR = ctypes.cast(_XDESTROY_CFUNC, ctypes.c_void_p).value


def _register_with_destroy(db, name, module_p, client_ptr, handle):
    _DATA_ANCHOR[client_ptr] = handle
    with c_string(name) as name_p:
        rc = sqlite3_create_module_v2(db, name_p, module_p, client_ptr, _XDESTROY_ADDR)
    if rc != SQLITE_OK:
        # SQLite may already have called the destructor, which it does when the registration fails after the
        # arguments are checked, so the pop takes a default.
        _DATA_ANCHOR.pop(client_ptr, None)
        _raise_rc(db, name, rc)


@proxy(signatures.get("sqlite3_declare_vtab"), jit_options=jit_options)
def sqlite3_declare_vtab(db, z_sql):
    """Declare the columns of the virtual table being created, from the ``CREATE TABLE`` statement at ``z_sql``.

    C reference: `sqlite3_declare_vtab <https://sqlite.org/c3ref/declare_vtab.html>`_.

    Only a module's ``xCreate`` or ``xConnect`` may call it. Returns ``SQLITE_OK`` or an error code.
    """
    return _call_lib_func("sqlite3_declare_vtab", (db, z_sql))


class _BuiltDescriptor:
    """A table's descriptor ``c``, with the buffers that it points into and the column arrays, kept alive together.

    The callbacks read ``c`` through its address and the buffers through the addresses that ``c`` holds; they read
    none of these attributes. ``nrows``, ``ncols`` and ``scratch_bytes`` repeat values of ``c`` for Python code.
    """
    __slots__ = ("c", "bases", "strides", "tags", "widths", "schema",
                 "nrows", "ncols", "scratch_bytes", "arrays")

    def __init__(self, c, bases, strides, tags, widths, schema, arrays):
        self.c = c
        self.bases = bases
        self.strides = strides
        self.tags = tags
        self.widths = widths
        self.schema = schema
        self.arrays = arrays
        self.nrows = int(c["nrows"][0])
        self.ncols = int(c["ncols"][0])
        self.scratch_bytes = int(c["scratch_bytes"][0])


def _finalize_descriptor(nrows, col_names, tags, widths, bases, strides, arrays):
    """Build a table's descriptor from the layout of its columns, and return it in a ``_BuiltDescriptor``.

    Raises ``ValueError`` for a table without columns and ``TypeError`` for a column name that is not a string.
    """
    if not col_names:
        raise ValueError("table must have at least one column")
    for n in col_names:
        if not isinstance(n, str):
            raise TypeError("column name must be a string, got %r" % (type(n),))
    scratch = max((w + 1 for w, t in zip(widths, tags) if t == _TAG_U), default=0)
    bases_buf = np.array(bases, dtype=np.int64)
    strides_buf = np.array(strides, dtype=np.int64)
    tags_buf = np.array(tags, dtype=tags_buf_t)
    widths_buf = np.array(widths, dtype=np.int64)
    cols_sql = ", ".join('"%s" %s' % (n.replace('"', '""'), _SQL_TYPE[t]) for n, t in zip(col_names, tags))
    schema = ("CREATE TABLE x(%s)" % cols_sql).encode("utf-8") + b"\x00"

    c = np.zeros(1, _DESC_DTYPE)
    c["nrows"] = int(nrows)
    c["ncols"] = len(col_names)
    c["col_bases"] = bases_buf.ctypes.data
    c["col_strides"] = strides_buf.ctypes.data
    c["col_tags"] = tags_buf.ctypes.data
    c["col_widths"] = widths_buf.ctypes.data
    c["schema_ptr"] = ctypes.cast(ctypes.c_char_p(schema), ctypes.c_void_p).value
    c["scratch_bytes"] = int(scratch)
    return _BuiltDescriptor(c, bases_buf, strides_buf, tags_buf, widths_buf, schema, tuple(arrays))


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

    tags = [_col_tag(dt, text_as_blob) for dt in sub]
    widths = [int(dt.itemsize) for dt in sub]
    base = arr.ctypes.data
    row_stride = int(arr.strides[0])
    bases = [base + int(o) for o in offs]
    strides = [row_stride] * len(col_names)
    return _finalize_descriptor(arr.shape[0], col_names, tags, widths, bases, strides, (arr,))


def _build_descriptor_columnar(mapping, text_as_blob):
    names = list(mapping.keys())
    if not names:
        raise ValueError("mapping must have at least one column")
    bases, strides, tags, widths = [], [], [], []
    col_arrays = []
    nrows = None
    for n in names:
        col = mapping[n]
        if not isinstance(col, np.ndarray):
            raise TypeError("column %r must be a 1-D numpy array, got %r" % (n, type(col)))
        if col.ndim != 1:
            raise ValueError("column %r must be 1-D, got ndim=%d" % (n, col.ndim))
        if nrows is None:
            nrows = int(col.shape[0])
        elif int(col.shape[0]) != nrows:
            raise ValueError("column %r length %d != %d" % (n, col.shape[0], nrows))
        tags.append(_col_tag(col.dtype, text_as_blob))
        widths.append(int(col.dtype.itemsize))
        bases.append(col.ctypes.data)
        strides.append(int(col.strides[0]))
        col_arrays.append(col)
    cols = tuple(col_arrays)
    return _finalize_descriptor(nrows, names, tags, widths, bases, strides, cols)


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
        # SQLite zeroes the sqlite3_vtab base after this callback returns and then sets its fields, so the code sets
        # only descriptor, numbox's own member.
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
    return _TAG_I8 <= tag <= _TAG_U64


@njit(**jit_options)
def _is_supported_op(op):
    return (op == SQLITE_INDEX_CONSTRAINT_EQ or op == SQLITE_INDEX_CONSTRAINT_GT
            or op == SQLITE_INDEX_CONSTRAINT_GE or op == SQLITE_INDEX_CONSTRAINT_LT
            or op == SQLITE_INDEX_CONSTRAINT_LE)


@cfunc(types.int32(types.intp, types.intp), cache=_CACHE)
def _xbestindex(vtab, idx_info):
    # The callback claims every usable constraint on a numeric column whose operator _is_supported_op accepts. Each
    # claimed constraint gets an argvIndex, so that xFilter receives its value, and its (col, op) pair goes into
    # idxStr, which SQLite passes on to xFilter. The callback reports a row estimate in every case, because SQLite
    # assumes 25 rows otherwise and then misjudges the cost of joins and subqueries.
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
                # omit stays at the 0 that SQLite set, so SQLite checks the constraint again on every row that the
                # cursor returns. The filtering in xFilter only skips rows that cannot match.
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
        # A heuristic, because the table keeps no column statistics: each bound constraint is assumed to divide the
        # remaining rows, which favors plans that bind more constraints. The + 1 keeps the estimate above 0.
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
            scratch_p = sqlite3_malloc(int32(scratch))  # fits int32: NumPy caps a 'U' itemsize at 2147483644 bytes
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


@njit(**_INLINE_JIT_OPTIONS)
def _load_cell_i64(addr, tag):
    """Return the integer or ``bool`` cell at ``addr`` as the int64 that SQLite receives.

    A ``uint64`` of 2**63 or more wraps to a negative value, and a ``bool`` is 0 or 1. Any other tag reads as 0.
    """
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
    elif tag == _TAG_BOOL:
        return int64(1) if load_unaligned(addr, uint8) != 0 else int64(0)
    return int64(0)


@njit(**_INLINE_JIT_OPTIONS)
def _load_cell_f64(addr, tag):
    """Return the float or ``bool`` cell at ``addr`` as a float64.

    No other numeric tag reaches it: ``_emit_cell`` sends it the float tags only, and the predicate scan compares every
    integer column as int64, through ``_cell_value_i64``. Any other tag reads as 0.
    """
    if tag == _TAG_BOOL:
        return float64(1) if load_unaligned(addr, uint8) != 0 else float64(0)
    elif tag == _TAG_F32:
        return float64(load_unaligned(addr, float32))
    elif tag == _TAG_F64:
        return load_unaligned(addr, float64)
    return float64(0)


@njit(**_INLINE_JIT_OPTIONS)
def _cell_at(d, rowid, col):
    """Return the address and the tag of the cell at ``(rowid, col)`` in the table of the descriptor ``d``."""
    ncols = d[0].ncols
    bases = carray(_cast_int_to_void_p(d[0].col_bases), (ncols,), dtype=np.int64)
    strides = carray(_cast_int_to_void_p(d[0].col_strides), (ncols,), dtype=np.int64)
    tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=tags_buf_t)
    return bases[col] + rowid * strides[col], tags[col]


@njit(**jit_options)
def _cell_value_f64(d, rowid, col):
    """Return the cell at ``(rowid, col)`` as a float64, loaded as ``_xcolumn`` loads it.

    xBestIndex claims only the tags up to ``_TAG_BOOL``, and a string or BLOB tag reads as 0.
    """
    addr, tag = _cell_at(d, rowid, col)
    return _load_cell_f64(addr, tag)


@njit(**jit_options)
def _cell_value_i64(d, rowid, col):
    """Return the integer cell at ``(rowid, col)`` as the int64 that ``_xcolumn`` passes to ``sqlite3_result_int64``.

    A ``uint64`` wraps to the same int64 that SQLite receives.
    """
    addr, tag = _cell_at(d, rowid, col)
    return _load_cell_i64(addr, tag)


@njit(**jit_options)
def _cmp(op, cv, rv):
    """Return the truth of ``cv OP rv`` for the constraint operator ``op``, comparing in the arguments' own types."""
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
def _int_float_cmp(i, r):
    """Return the sign of ``i - r``, -1, 0 or 1, for an int64 cell ``i`` and a REAL constraint value ``r``.

    The result is exact at every magnitude, as in SQLite's ``sqlite3IntFloatCompare``, so an integer column
    constrained by a REAL value never loses a matching row.
    """
    # The function never converts i to float64, which would lose precision above 2**53.
    if r < -9.223372036854775808e18:   # r < INT64_MIN, so i > r
        return 1
    if r >= 9.223372036854775808e18:   # r >= 2**63 > INT64_MAX, so i < r
        return -1
    y = int64(r)                       # truncates toward zero; r is in [INT64_MIN, 2**63)
    if i < y:
        return -1
    if i > y:
        return 1
    s = float64(i)                     # i == y here, and y converts to float64 exactly, so s == i
    if s < r:
        return -1
    if s > r:
        return 1
    return 0


@njit(**jit_options)
def _cmp_if(op, i, r):
    """Return the truth of ``i OP r`` for an int64 cell ``i`` and a REAL constraint value ``r``, compared exactly."""
    c = _int_float_cmp(i, r)
    if op == SQLITE_INDEX_CONSTRAINT_EQ:
        return c == 0
    elif op == SQLITE_INDEX_CONSTRAINT_GT:
        return c > 0
    elif op == SQLITE_INDEX_CONSTRAINT_GE:
        return c >= 0
    elif op == SQLITE_INDEX_CONSTRAINT_LT:
        return c < 0
    return c <= 0


@njit(**jit_options)
def _row_matches(cur):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    if c[0].n_pred == 0:
        return True
    d = carray(_cast_int_to_void_p(c[0].descriptor), (1,), dtype=_DESC_DTYPE)
    preds = carray(_cast_int_to_void_p(c[0].pred_p), (c[0].n_pred,), dtype=_PRED_DTYPE)
    for k in range(c[0].n_pred):
        op = preds[k].op
        mode = preds[k].is_int
        # Each mode calls _cmp from a call site of its own, so numba never unifies an int64 cell with a float64 one
        # in a phi node. That widening loses precision above 2**53, and a predicate evaluated on the rounded value
        # would drop rows that SQLite's own check of the omit=0 constraints never sees, because SQLite checks only
        # the rows that the cursor returns.
        if mode == 1:
            ok = _cmp(op, _cell_value_i64(d, c[0].rowid, preds[k].col), preds[k].ival)
        elif mode == 0:
            ok = _cmp(op, _cell_value_f64(d, c[0].rowid, preds[k].col), preds[k].fval)
        else:
            ok = _cmp_if(op, _cell_value_i64(d, c[0].rowid, preds[k].col), preds[k].fval)
        if not ok:
            return False
    return True


@njit(**jit_options)
def _seek_match(cur):
    # No try/except: the loop does pointer arithmetic and loads within bounds (0 <= col < ncols, rowid < nrows), and
    # it allocates nothing that could raise.
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
        # argc equals the idxNum that xBestIndex set, the number of claimed constraints. The code relies on argc
        # alone, because idxStr holds argc (col, op) pairs and argv holds argc values.
        if argc > 0:
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
                # The mode depends on the type of the constraint's value as well as on the column's tag. An integer
                # column constrained by a REAL value must not truncate the value to int64, because a row dropped
                # here never reaches SQLite's own check, which sees only the rows that the cursor returns.
                if _is_int_tag(col_tags[col]):
                    if sqlite3_value_numeric_type(vals[k]) == SQLITE_FLOAT:
                        # An integer column against a REAL value compares exactly, in mode 2. Truncating the value
                        # to int64 would drop rows for a fractional or out-of-range value, and widening the cell to
                        # float64 would drop rows for an integer value above 2**53.
                        preds[k].is_int = 2
                        preds[k].ival = 0
                        preds[k].fval = sqlite3_value_double(vals[k])
                    else:
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
    # No try/except: the body advances the row and calls _seek_match, which does pointer arithmetic and loads within
    # bounds, and nothing here allocates memory that could raise.
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    c[0].rowid = c[0].rowid + 1
    _seek_match(cur)
    return SQLITE_OK


@cfunc(types.int32(types.intp), cache=_CACHE)
def _xeof(cur):
    # No try/except: the body reads the cursor and the descriptor and allocates nothing that could raise.
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    d = carray(_cast_int_to_void_p(c[0].descriptor), (1,), dtype=_DESC_DTYPE)
    if c[0].rowid >= d[0].nrows:
        return 1
    return 0


@cfunc(types.int32(types.intp, types.intp), cache=_CACHE)
def _xrowid(cur, p_rowid):
    # No try/except: the body reads the cursor, writes the rowid and allocates nothing that could raise.
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    store_at(p_rowid, c[0].rowid)
    return SQLITE_OK


_UNSUPPORTED_TAG_MSG = "unsupported column tag"


@njit(**_INLINE_JIT_OPTIONS)
def _emit_cell(ctx, addr, tag, width, scratch_p, s_blob_destructor):
    """Set the cell at ``addr`` as the ``xColumn`` result on ``ctx``, and return the code that ``xColumn`` returns.

    An ``'S'`` or BLOB result points at ``addr`` itself, so the caller passes the destructor value that matches how
    long ``addr`` stays valid. A ``'U'`` result is converted to UTF-8 in ``scratch_p``, which the next call
    overwrites, so it is always ``SQLITE_TRANSIENT``. Each tag that ``_col_tag`` returns has a branch. Any other tag
    fails the query with an error instead of reading back as NULL.
    """
    if _is_int_tag(tag) or tag == _TAG_BOOL:
        sqlite3_result_int64(ctx, _load_cell_i64(addr, tag))
    elif tag == _TAG_F32 or tag == _TAG_F64:
        sqlite3_result_double(ctx, _load_cell_f64(addr, tag))
    elif tag == _TAG_S:
        n = _nul_trimmed_len(addr, width)
        sqlite3_result_text(ctx, addr, int32(n), s_blob_destructor)
    elif tag == _TAG_BLOB:
        n = _nul_trimmed_len(addr, width)
        sqlite3_result_blob(ctx, addr, int32(n), s_blob_destructor)
    elif tag == _TAG_U:
        n = utf32_to_utf8(addr, width // 4, scratch_p)
        sqlite3_result_text(ctx, scratch_p, int32(n), SQLITE_TRANSIENT)
    else:
        sqlite3_result_error(ctx, get_unicode_data_p(_UNSUPPORTED_TAG_MSG), -1)
        return SQLITE_ERROR
    return SQLITE_OK


@cfunc(types.int32(types.intp, types.intp, types.int32), cache=_CACHE)
def _xcolumn(cur, ctx, j):
    try:
        c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
        rowid = c[0].rowid
        d = carray(_cast_int_to_void_p(c[0].descriptor), (1,), dtype=_DESC_DTYPE)
        ncols = d[0].ncols
        bases = carray(_cast_int_to_void_p(d[0].col_bases), (ncols,), dtype=np.int64)
        strides = carray(_cast_int_to_void_p(d[0].col_strides), (ncols,), dtype=np.int64)
        tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=tags_buf_t)
        widths = carray(_cast_int_to_void_p(d[0].col_widths), (ncols,), dtype=np.int64)
        addr = bases[j] + rowid * strides[j]
        # An 'S' or BLOB result points into the registered array, which stays alive and unchanged while the table is
        # registered: _DATA_ANCHOR holds it, and the caller must not change it. SQLITE_STATIC therefore lets SQLite
        # read the value without a copy.
        return _emit_cell(ctx, addr, tags[j], widths[j], c[0].scratch_p, SQLITE_STATIC)
    except Exception:
        sqlite3_result_error(ctx, get_unicode_data_p("error reading vtable column"), -1)
        return SQLITE_ERROR


class _Sqlite3Module(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in (
        "xCreate", "xConnect", "xBestIndex", "xDisconnect", "xDestroy",
        "xOpen", "xClose", "xFilter", "xNext", "xEof", "xColumn", "xRowid",
        "xUpdate", "xBegin", "xSync", "xCommit", "xRollback",
        "xFindFunction", "xRename")]
    # _fields_ is assigned twice to put iVersion first; ctypes reads only the final value. With iVersion 1, SQLite
    # reads no member after xRename, so the struct ends there.
    _fields_ = [("iVersion", ctypes.c_int)] + _fields_


# One sqlite3_module, built when this module is imported, serves every table that register_table registers, because
# each callback reads its table's layout from the descriptor that SQLite passes as client data. xCreate is the same
# function as xConnect, which makes every table eponymous, and xDestroy is xDisconnect, because a table has no storage
# of its own to delete.
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
    """The objects that a registered table needs while SQLite can read it: its descriptor, buffers and column arrays.

    SQLite and the callbacks read the arrays' memory directly, so if the handle were collected, the next query would
    read freed memory. ``_DATA_ANCHOR`` holds the handle until SQLite calls the destructor that the registration
    passed to ``sqlite3_create_module_v2``.
    """
    __slots__ = ("_keep",)

    def __init__(self, *objs):
        self._keep = objs


def _raise_rc(db, name, rc):
    msg_p = sqlite3_errmsg(db)
    detail = ""
    if msg_p:
        detail = ": " + ctypes.cast(msg_p, ctypes.c_char_p).value.decode("utf-8", "replace")
    raise RuntimeError("registration failed for %r (rc=%d)%s" % (name, rc, detail))


def register_table(db, name, data, columns=None, *, text_as_blob=False):
    """Register ``data`` on the connection ``db`` as the read-only SQLite virtual table ``name``, without a copy.

    The table is eponymous: SQL queries it as ``name``, with no ``CREATE VIRTUAL TABLE``, because the module's
    ``xCreate`` is its ``xConnect``. ``data`` is one of:

    - a 1-D NumPy structured array, one row for each element, with the field names as column names. ``columns`` may
      rename the columns, but they keep the order of the fields; to reorder them, reorder the fields of the array or
      pass a mapping.
    - a 2-D NumPy array, one table row for each row of the array. ``columns`` is required and names each column.
    - a mapping, such as a ``dict``, from column name to 1-D NumPy array, all of one length. The keys name the
      columns in their order, so passing ``columns`` raises ``ValueError``. Each value must be a ``numpy.ndarray``,
      because numbox converts nothing: another value raises ``TypeError``, and an array that is not 1-D raises
      ``ValueError``.

    numbox keeps the data and the table's descriptor alive until SQLite drops the module, when the connection closes
    or when the name is registered again. The caller must not change or resize the arrays while the table is
    registered, because each query reads the columns' memory directly. ``'S'`` and BLOB values reach SQLite as
    ``SQLITE_STATIC`` pointers into that memory.

    Values:

    - A ``uint64`` value of 2**63 or more becomes SQLite's signed INTEGER, so it wraps to a negative value.
    - A float becomes a REAL, infinities included, except NaN: SQLite turns a NaN REAL into SQL NULL, so a NaN cell
      reads back as NULL. NumPy has no missing value, so every other cell is non-NULL. Masked arrays are not
      supported, and the table ignores the mask.

    String columns:

    - ``text_as_blob`` affects only bytes (``'S'``) columns, and a unicode (``'U'``) column is always TEXT. By default
      an ``'S'`` column is TEXT, and its bytes reach SQLite without a check that they are UTF-8. Pass
      ``text_as_blob=True`` for bytes that are not UTF-8, so that they reach SQLite as a BLOB rather than as
      malformed TEXT.
    - NumPy pads a fixed-width ``'S'`` or ``'U'`` value with NULs. The table drops the trailing NULs of each value and
      keeps the interior ones.
    - SQLite receives the whole of a TEXT value with an interior NUL, because numbox passes its length in bytes. C
      string functions and most SQL text functions stop at the first NUL, though. Read such a value with
      ``sqlite3_column_bytes`` and the text or BLOB pointer, or pass ``text_as_blob=True`` to keep every byte.

    Raises ``TypeError`` for ``data`` of another type, for a column name that is not a string and for a dtype without
    a SQLite type. Raises ``ValueError`` for a shape, a length or a ``columns`` list that does not fit the data, and
    ``RuntimeError``, with SQLite's message, when SQLite refuses the registration.
    """
    if isinstance(data, np.ndarray):
        built = _build_descriptor(data, columns, text_as_blob)
    elif isinstance(data, Mapping):
        if columns is not None:
            raise ValueError("columns must be omitted when data is a mapping; "
                             "the dict keys name the columns")
        built = _build_descriptor_columnar(data, text_as_blob)
    else:
        raise TypeError("data must be a numpy.ndarray or a mapping of name -> 1-D array, "
                        "got %r" % (type(data),))
    handle = _VTableHandle(built)
    _register_with_destroy(db, name, _THE_MODULE_P, built.c.ctypes.data, handle)
