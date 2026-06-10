"""Expose a per-query-computed numpy structured array as a SQLite table-valued
function (register_tvf).

``register_tvf(db, name, arg_types, out_dtype, fn)`` registers an eponymous
virtual table whose rows are NOT static -- they are produced per query by the
user's ``fn(*args)``, which returns a 1-D numpy structured array of ``out_dtype``.
The function arguments are exposed as HIDDEN columns: ``SELECT * FROM f(2, 5)``
turns ``2``/``5`` into EQ constraints on the hidden columns, which ``xBestIndex``
binds into ``argv`` (in declaration order) and ``xFilter`` decodes and feeds to
``fn``. The returned array is held alive for the cursor's lifetime via an
NRT-backed ``[meminfo_p, data_p]`` slot (pinned with the inlined
``_incref_meminfo`` intrinsic so numba's refcount pass cannot strip it), and
released exactly once in ``xClose``.

Unlike the read-only ``register_table`` (one shared module), each
``register_tvf`` GENERATES its own ``xFilter`` / ``xColumn`` (the user ``fn`` and
``out_dtype`` are baked in as codegen globals so the allocator specialises on the
dtype and the impls cache cross-process) and builds its own ``sqlite3_module``
from this registration's cfunc addresses. The handle retains the module struct,
every cfunc object (SQLite stores their addresses), the descriptor buffers, and
``fn``; its keep-alive lives in the module-level ``_DATA_ANCHOR`` (released by
SQLite via ``xDestroy``).
"""
import ctypes

import numpy as np
from numba import cfunc, types

from numbox.core.bindings._sqlite_constants import (
    SQLITE_OK, SQLITE_ERROR, SQLITE_NOMEM, SQLITE_CONSTRAINT,
    SQLITE_INDEX_CONSTRAINT_EQ,
)
from numbox.core.bindings._sqlite_typemap import _col_tag, _SQL_TYPE, tags_buf_t
from numbox.core.bindings._sqlite_vtable import (
    _Sqlite3Module, _SQLITE3_VTAB_CURSOR_DTYPE, _VTAB_DTYPE, _VTAB_SIZE,
    _IDX_INFO_DTYPE, _CONSTRAINT_DTYPE, _USAGE_DTYPE,
    _register_with_destroy,
)
from numbox.utils.digest import digest
from numbox.utils.preprocessing import (
    _anchor_path, _materialize_anchor, _orphan_anchor_sweep,
)

# Names referenced by the GENERATED source; importing them here puts them in
# this module's __dict__, which seeds the exec namespace below.
from numba import carray, njit  # noqa: F401
from numba.core.types import (  # noqa: F401
    int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64,
)
from numbox.core.bindings._sqlite_vtable import sqlite3_declare_vtab  # noqa: F401
from numbox.core.bindings._sqlite_exec import sqlite3_malloc, sqlite3_free  # noqa: F401
from numbox.core.bindings._sqlite_value import (  # noqa: F401
    sqlite3_value_int64, sqlite3_value_double,
)
from numbox.core.bindings._sqlite_result import (  # noqa: F401
    sqlite3_result_int64, sqlite3_result_double,
    sqlite3_result_text, sqlite3_result_blob, sqlite3_result_error,
)
from numbox.core.bindings._sqlite_typemap import (  # noqa: F401
    _TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64, _TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64,
    _TAG_F32, _TAG_F64, _TAG_BOOL, _TAG_S, _TAG_U, _TAG_BLOB,
    _nul_trimmed_len, utf32_to_utf8,
)
from numbox.core.configurations import jit_options  # noqa: F401
from numbox.utils.lowlevel import (  # noqa: F401
    _cast_int_to_void_p, get_unicode_data_p, load_unaligned, store_at, array_data_p,
)
from numbox.utils.meminfo import (  # noqa: F401
    structref_meminfo, _incref_meminfo, release_meminfo,
)

__all__ = ["register_tvf"]

_CACHE = jit_options.get("cache", True)
_SQLITE_TRANSIENT = -1

_ANCHOR_SUBDIR = "numbox-sqlite-tvf"
_orphan_anchor_sweep(_ANCHOR_SUBDIR)


# per-registration descriptor passed as pClientData; xConnect/xBestIndex/xColumn
# read it by-name via carray(ptr, (1,), dtype=_TVF_DESC_DTYPE). The visible
# (out_dtype) columns come first in the schema, the n_hidden arg columns after.
_TVF_DESC_DTYPE = np.dtype([
    ("ncols", "i4"), ("n_hidden", "i4"), ("itemsize", "i8"),
    ("col_offsets", "i8"), ("col_tags", "i8"), ("col_widths", "i8"),
    ("schema_ptr", "i8"), ("scratch_bytes", "i8"),
], align=True)

_TVF_CUR_DTYPE = np.dtype([
    ("base", _SQLITE3_VTAB_CURSOR_DTYPE), ("descriptor", "i8"), ("rowid", "i8"),
    ("mi_p", "i8"), ("data_p", "i8"), ("n_rows", "i8"), ("row_stride", "i8"), ("scratch_p", "i8"),
], align=True)
_TVF_CUR_SIZE = _TVF_CUR_DTYPE.itemsize

_INT_TAGS = frozenset((_TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64,
                       _TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64, _TAG_BOOL))
_FLOAT_TAGS = frozenset((_TAG_F32, _TAG_F64))


# --- generated-source templates -------------------------------------------
# Baked globals (seeded into the exec namespace): _fn (the user callable),
# _N_HIDDEN, jit_options, and every helper imported above. The {arg_decode} /
# {fn_call} substitutions are the only arity-varying parts; they are generated
# from arg_types before exec, exactly as the UDAF helpers bake per-UDAF source.
# Cache-correctness across out_dtypes is not a shared-function hazard here: each
# registration gets a DISTINCT anchor/cache key from digest((out_dtype,
# tuple(arg_tags)), [fn]), and the baked _fn's fixed return type (set by
# out_dtype) makes numba specialise the impl, so two out_dtypes never collide.
_XFILTER_SRC = '''
@njit(**jit_options)
def _tvf_xfilter_impl(cur, argc, argv):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_TVF_CUR_DTYPE)
    if c[0].mi_p != 0:
        release_meminfo(c[0].mi_p)
        c[0].mi_p = 0
        c[0].data_p = 0
        c[0].n_rows = 0
        c[0].row_stride = 0
    c[0].rowid = 0
    if argc < _N_HIDDEN:
        return
    vals = carray(_cast_int_to_void_p(argv), (argc,), dtype=np.intp)
{arg_decode}
    result = {fn_call}
    mi_p, _base = structref_meminfo(result)
    _incref_meminfo(mi_p)
    c[0].mi_p = mi_p
    c[0].data_p = array_data_p(result)
    c[0].n_rows = result.shape[0]
    c[0].row_stride = result.strides[0]
'''


def _gen_arg_decode(arg_tags):
    lines = []
    for i, tag in enumerate(arg_tags):
        if tag in _INT_TAGS:
            lines.append("    a%d = sqlite3_value_int64(vals[%d])" % (i, i))
        else:
            lines.append("    a%d = sqlite3_value_double(vals[%d])" % (i, i))
    return "\n".join(lines)


# --- per-registration cfunc factories -------------------------------------
def _make_xconnect():
    @cfunc(types.int32(types.intp, types.intp, types.int32, types.intp, types.intp, types.intp),
           cache=_CACHE)
    def _tvf_xconnect(db, p_aux, argc, argv, pp_vtab, pz_err):
        vtab = 0
        try:
            d = carray(_cast_int_to_void_p(p_aux), (1,), dtype=_TVF_DESC_DTYPE)
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
    return _tvf_xconnect


def _make_xbestindex():
    @cfunc(types.int32(types.intp, types.intp), cache=_CACHE)
    def _tvf_xbestindex(vtab, idx_info):
        # Require a usable EQ constraint on every hidden arg column; bind each to
        # argv in declaration order (argvIndex = hidden-arg-index + 1, omit=1 so
        # SQLite passes the value and drops the synthetic constraint). If any
        # hidden arg is unbound, return SQLITE_CONSTRAINT so the plan is rejected
        # cleanly -- xFilter is then never called with too few args.
        v = carray(_cast_int_to_void_p(vtab), (1,), dtype=_VTAB_DTYPE)
        d = carray(_cast_int_to_void_p(v[0].descriptor), (1,), dtype=_TVF_DESC_DTYPE)
        ii = carray(_cast_int_to_void_p(idx_info), (1,), dtype=_IDX_INFO_DTYPE)
        ncols = d[0].ncols
        n_hidden = d[0].n_hidden
        n_constraint = ii[0].nConstraint
        cons = carray(_cast_int_to_void_p(ii[0].aConstraint), (n_constraint,), dtype=_CONSTRAINT_DTYPE)
        usage = carray(_cast_int_to_void_p(ii[0].aConstraintUsage), (n_constraint,), dtype=_USAGE_DTYPE)

        # Track each hidden arg's binding as its own bit, not a running count: a
        # duplicate usable EQ on one arg must not mask another arg being unbound.
        # argvIndex is position-based (h + 1), so duplicates overwrite one slot.
        bound_mask = uint64(0)
        for i in range(n_constraint):
            col = cons[i].iColumn
            op = cons[i].op
            h = col - ncols
            if cons[i].usable != 0 and op == SQLITE_INDEX_CONSTRAINT_EQ and 0 <= h < n_hidden:
                usage[i].argvIndex = int32(h + 1)
                usage[i].omit = 1
                bound_mask |= uint64(1) << uint64(h)

        if bound_mask != (uint64(1) << uint64(n_hidden)) - uint64(1):
            return SQLITE_CONSTRAINT
        ii[0].idxNum = int32(1)
        ii[0].estimatedCost = float64(1)
        ii[0].estimatedRows = 16
        return SQLITE_OK
    return _tvf_xbestindex


def _make_static_cfuncs():
    @cfunc(types.int32(types.intp), cache=_CACHE)
    def _tvf_xdisconnect(vtab):
        try:
            sqlite3_free(vtab)
            return SQLITE_OK
        except Exception:
            return SQLITE_ERROR

    @cfunc(types.int32(types.intp, types.intp), cache=_CACHE)
    def _tvf_xopen(vtab, pp_cursor):
        cur = 0
        scratch_p = 0
        try:
            v = carray(_cast_int_to_void_p(vtab), (1,), dtype=_VTAB_DTYPE)
            desc = v[0].descriptor
            d = carray(_cast_int_to_void_p(desc), (1,), dtype=_TVF_DESC_DTYPE)
            scratch = d[0].scratch_bytes
            cur = sqlite3_malloc(int32(_TVF_CUR_SIZE))
            if cur == 0:
                return SQLITE_NOMEM
            if scratch > 0:
                scratch_p = sqlite3_malloc(int32(scratch))
                if scratch_p == 0:
                    sqlite3_free(cur)
                    return SQLITE_NOMEM
            c = carray(_cast_int_to_void_p(cur), (1,), dtype=_TVF_CUR_DTYPE)
            c[0].base.pVtab = vtab
            c[0].descriptor = desc
            c[0].rowid = 0
            c[0].mi_p = 0
            c[0].data_p = 0
            c[0].n_rows = 0
            c[0].row_stride = 0
            c[0].scratch_p = scratch_p
            slot = carray(_cast_int_to_void_p(pp_cursor), (1,), dtype=np.intp)
            slot[0] = cur
            return SQLITE_OK
        except Exception:
            sqlite3_free(scratch_p)
            sqlite3_free(cur)
            return SQLITE_ERROR

    @cfunc(types.int32(types.intp), cache=_CACHE)
    def _tvf_xclose(cur):
        try:
            c = carray(_cast_int_to_void_p(cur), (1,), dtype=_TVF_CUR_DTYPE)
            if c[0].mi_p != 0:
                release_meminfo(c[0].mi_p)
                c[0].mi_p = 0
                c[0].data_p = 0
                c[0].n_rows = 0
                c[0].row_stride = 0
            sqlite3_free(c[0].scratch_p)
            sqlite3_free(cur)
            return SQLITE_OK
        except Exception:
            return SQLITE_ERROR

    @cfunc(types.int32(types.intp), cache=_CACHE)
    def _tvf_xnext(cur):
        c = carray(_cast_int_to_void_p(cur), (1,), dtype=_TVF_CUR_DTYPE)
        c[0].rowid = c[0].rowid + 1
        return SQLITE_OK

    @cfunc(types.int32(types.intp), cache=_CACHE)
    def _tvf_xeof(cur):
        c = carray(_cast_int_to_void_p(cur), (1,), dtype=_TVF_CUR_DTYPE)
        if c[0].data_p == 0 or c[0].rowid >= c[0].n_rows:
            return 1
        return 0

    @cfunc(types.int32(types.intp, types.intp), cache=_CACHE)
    def _tvf_xrowid(cur, p_rowid):
        c = carray(_cast_int_to_void_p(cur), (1,), dtype=_TVF_CUR_DTYPE)
        store_at(p_rowid, c[0].rowid)
        return SQLITE_OK

    return _tvf_xdisconnect, _tvf_xopen, _tvf_xclose, _tvf_xnext, _tvf_xeof, _tvf_xrowid


def _make_xcolumn():
    @cfunc(types.int32(types.intp, types.intp, types.int32), cache=_CACHE)
    def _tvf_xcolumn(cur, ctx, j):
        try:
            c = carray(_cast_int_to_void_p(cur), (1,), dtype=_TVF_CUR_DTYPE)
            data_p = c[0].data_p
            if data_p == 0:
                return SQLITE_OK
            rowid = c[0].rowid
            d = carray(_cast_int_to_void_p(c[0].descriptor), (1,), dtype=_TVF_DESC_DTYPE)
            ncols = d[0].ncols
            offsets = carray(_cast_int_to_void_p(d[0].col_offsets), (ncols,), dtype=np.int64)
            tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=tags_buf_t)
            widths = carray(_cast_int_to_void_p(d[0].col_widths), (ncols,), dtype=np.int64)
            addr = data_p + rowid * c[0].row_stride + offsets[j]
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
                sqlite3_result_text(ctx, addr, int32(n), _SQLITE_TRANSIENT)
            elif tag == _TAG_BLOB:
                n = _nul_trimmed_len(addr, widths[j])
                sqlite3_result_blob(ctx, addr, int32(n), _SQLITE_TRANSIENT)
            elif tag == _TAG_U:
                scratch = c[0].scratch_p
                n = utf32_to_utf8(addr, widths[j] // 4, scratch)
                sqlite3_result_text(ctx, scratch, int32(n), _SQLITE_TRANSIENT)
            return SQLITE_OK
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error reading tvf column"), -1)
            return SQLITE_ERROR
    return _tvf_xcolumn


def _prepare_fn(fn):
    from numba.extending import is_jitted
    if is_jitted(fn):
        return fn
    if callable(fn):
        return njit(fn)
    raise TypeError("fn must be a callable (plain Python or @njit), got %r" % (fn,))


def _compile_xfilter(stem, arg_tags, out_dtype, fn):
    n_hidden = len(arg_tags)
    arg_decode = _gen_arg_decode(arg_tags)
    fn_call = "_fn(%s)" % ", ".join("a%d" % i for i in range(n_hidden))
    src = _XFILTER_SRC.format(arg_decode=arg_decode, fn_call=fn_call)
    tvf_digest = digest((out_dtype, tuple(arg_tags)), [fn])
    code_txt = "# tvf-digest: %s\n%s" % (tvf_digest, src)
    ns = {**globals(), "_fn": fn, "_N_HIDDEN": n_hidden}
    anchor = _anchor_path(_ANCHOR_SUBDIR, stem, code_txt)
    _materialize_anchor(anchor, code_txt)
    code = compile(code_txt, str(anchor), mode="exec")
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns["_tvf_xfilter_impl"]


def _build_tvf_descriptor(name, arg_types, out_dtype):
    if out_dtype.fields is None or out_dtype.names is None:
        raise TypeError("out_dtype must be a structured numpy dtype, got %r" % (out_dtype,))
    names = list(out_dtype.names)
    subs = [out_dtype.fields[nm][0] for nm in names]
    offs = [out_dtype.fields[nm][1] for nm in names]
    vis_tags = [_col_tag(s, False) for s in subs]
    vis_widths = [int(s.itemsize) for s in subs]

    arg_dtypes = [np.dtype(t) for t in arg_types]
    arg_tags = [_col_tag(dt, False) for dt in arg_dtypes]
    if any(t not in _INT_TAGS and t not in _FLOAT_TAGS for t in arg_tags):
        raise TypeError(
            "register_tvf arg_types must be integer or float scalar dtypes; "
            "string/bytes hidden args are not supported")

    offsets_buf = np.array(offs, dtype=np.int64)
    tags_buf = np.array(vis_tags, dtype=tags_buf_t)
    widths_buf = np.array(vis_widths, dtype=np.int64)
    scratch = max([w + 1 for w, t in zip(vis_widths, vis_tags) if t == _TAG_U], default=0)

    vis_sql = ['"%s" %s' % (n.replace('"', '""'), _SQL_TYPE[t]) for n, t in zip(names, vis_tags)]
    hid_sql = ['"arg%d" %s HIDDEN' % (i, _SQL_TYPE[t]) for i, t in enumerate(arg_tags)]
    schema = ("CREATE TABLE x(%s)" % ", ".join(vis_sql + hid_sql)).encode("utf-8") + b"\x00"

    c = np.zeros(1, _TVF_DESC_DTYPE)
    c["ncols"] = len(names)
    c["n_hidden"] = len(arg_tags)
    c["itemsize"] = int(out_dtype.itemsize)
    c["col_offsets"] = offsets_buf.ctypes.data
    c["col_tags"] = tags_buf.ctypes.data
    c["col_widths"] = widths_buf.ctypes.data
    c["schema_ptr"] = ctypes.cast(ctypes.c_char_p(schema), ctypes.c_void_p).value
    c["scratch_bytes"] = int(scratch)
    return c, offsets_buf, tags_buf, widths_buf, schema, arg_tags


class _TvfHandle:
    """Keeps the per-registration module struct, every cfunc object (SQLite
    stores their addresses), the descriptor buffers, and fn alive. SQLite calls
    the cfuncs by address; if this handle is GC'd they free and SQLite calls
    freed code. The keep-alive lives in the module-level ``_DATA_ANCHOR``,
    released by SQLite via ``xDestroy``."""
    __slots__ = ("_keep",)

    def __init__(self, *objs):
        self._keep = objs


def _stem(name):
    return "tvf_" + "".join(c if c.isalnum() else "_" for c in name)


def register_tvf(db, name, arg_types, out_dtype, fn):
    """Register an eponymous table-valued function backed by a computed array.

    ``SELECT * FROM name(<args>)`` calls ``fn(*args)`` -- a plain Python or
    ``@njit`` callable returning a 1-D numpy structured array of ``out_dtype`` --
    and serves that array's rows. The ``arg_types`` (numpy scalar dtypes; integer
    kinds are read via ``sqlite3_value_int64``, floating kinds via
    ``sqlite3_value_double``) are exposed as trailing HIDDEN columns and must each
    be supplied with an equality value in the query; a call form that leaves a
    hidden arg unbound is rejected with a SQLite constraint error (no rows).

    The registration's keep-alive lives in the module-level ``_DATA_ANCHOR``
    and is released by SQLite via ``xDestroy`` (on connection close
    or re-registration of the same name). ``out_dtype`` is baked
    into the generated allocator (so it caches cross-process); a NaN ``float``
    cell reads back as SQL NULL (SQLite coerces NaN REAL to NULL), as in
    ``register_table``.

    ``fn`` must return a 1-D numpy structured array whose dtype is ``out_dtype``;
    a slice / strided / offset view is handled (the row stride is honored), but a
    return whose *dtype* differs from ``out_dtype`` is read through ``out_dtype``'s
    layout and yields undefined values.
    """
    c, offsets_buf, tags_buf, widths_buf, schema, arg_tags = _build_tvf_descriptor(
        name, arg_types, out_dtype)

    xfilter_impl = _compile_xfilter(_stem(name), arg_tags, out_dtype, _prepare_fn(fn))

    @cfunc(types.int32(types.intp, types.int32, types.intp, types.int32, types.intp), cache=_CACHE)
    def _tvf_xfilter(cur, idx_num, idx_str, argc, argv):
        # If the user fn raises inside xfilter_impl, the @cfunc boundary swallows
        # the exception and returns the zero default (SQLITE_OK), so SQLite sees a
        # successful query with no rows rather than an error. Unlike a UDF, xFilter
        # gets no sqlite3_context, so there is no handle to call sqlite3_result_error.
        xfilter_impl(cur, argc, argv)
        return SQLITE_OK

    xconnect = _make_xconnect()
    xbestindex = _make_xbestindex()
    xdisconnect, xopen, xclose, xnext, xeof, xrowid = _make_static_cfuncs()
    xcolumn = _make_xcolumn()

    module = _Sqlite3Module()
    module.iVersion = 1
    module.xCreate = xconnect.address
    module.xConnect = xconnect.address
    module.xBestIndex = xbestindex.address
    module.xDisconnect = xdisconnect.address
    module.xDestroy = xdisconnect.address
    module.xOpen = xopen.address
    module.xClose = xclose.address
    module.xFilter = _tvf_xfilter.address
    module.xNext = xnext.address
    module.xEof = xeof.address
    module.xColumn = xcolumn.address
    module.xRowid = xrowid.address
    module_p = ctypes.addressof(module)

    handle = _TvfHandle(
        module, c, offsets_buf, tags_buf, widths_buf, schema, fn,
        xfilter_impl, _tvf_xfilter, xconnect, xbestindex, xdisconnect,
        xopen, xclose, xnext, xeof, xrowid, xcolumn)
    _register_with_destroy(db, name, module_p, c.ctypes.data, handle)
