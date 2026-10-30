"""Register a SQLite table-valued function whose rows a Python or jitted function computes for each query.

:func:`register_tvf` registers ``fn`` under a name on a connection. ``SELECT * FROM name(2, 5)`` then calls
``fn(2, 5)`` and returns the rows of the structured array that ``fn`` returns. Each registration generates and
compiles its own ``xFilter`` implementation, the code that calls ``fn``, and builds its own ``sqlite3_module``. The
structs, the function that sets a cell as a column's result, and the registration come from
:mod:`~numbox.core.bindings.sqlite.vtable`.
"""
import ctypes

import numpy as np
from numba import cfunc, types

from numbox.core.bindings.sqlite.constants import (
    SQLITE_OK, SQLITE_ERROR, SQLITE_NOMEM, SQLITE_CONSTRAINT,
    SQLITE_INDEX_CONSTRAINT_EQ,
)
from numbox.core.bindings.sqlite._typemap import _col_tag, _SQL_TYPE, tags_buf_t
from numbox.core.bindings.sqlite.vtable import (
    _Sqlite3Module, _SQLITE3_VTAB_CURSOR_DTYPE, _VTAB_DTYPE, _VTAB_SIZE,
    _IDX_INFO_DTYPE, _CONSTRAINT_DTYPE, _USAGE_DTYPE,
    _emit_cell, _register_with_destroy,
)
from numbox.utils.digest import digest
from numbox.utils.preprocessing import (
    _anchor_path, _anchored_or_uncached, _orphan_anchor_sweep,
)

# The generated code names these. Importing them here puts them in this module's __dict__, from which
# _compile_xfilter seeds the namespace of the generated code.
from numba import carray, njit  # noqa: F401
from numba.core.types import (  # noqa: F401
    int32, int64, float64,
)
from numbox.core.bindings.sqlite.vtable import sqlite3_declare_vtab  # noqa: F401
from numbox.core.bindings.sqlite.exec import sqlite3_malloc, sqlite3_free  # noqa: F401
from numbox.core.bindings.sqlite.value import (  # noqa: F401
    sqlite3_value_int64, sqlite3_value_double,
)
from numbox.core.bindings.sqlite.result import (  # noqa: F401
    sqlite3_result_int64, sqlite3_result_double,
    sqlite3_result_text, sqlite3_result_blob, sqlite3_result_error,
)
from numbox.core.bindings.sqlite._typemap import (  # noqa: F401
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

# The callbacks below compile with the cache option alone, taken from numbox's jit options and treated as enabled when
# the key is absent; no other jit option applies to them. The generated xFilter implementation compiles under all of
# numbox's jit options.
_CACHE = jit_options.get("cache", True)
_SQLITE_TRANSIENT = -1

_ANCHOR_SUBDIR = "numbox-sqlite-tvf"
_orphan_anchor_sweep(_ANCHOR_SUBDIR)


# _TVF_DESC_DTYPE describes the descriptor of a registration, whose address SQLite passes to xConnect as the module's
# client data. xConnect, xBestIndex, xOpen and xColumn read its fields by name through
# carray(ptr, (1,), dtype=_TVF_DESC_DTYPE). The schema
# declares the visible columns of out_dtype first and the n_hidden argument columns after them. col_tags has an entry
# for every column of the schema, the hidden ones included; col_offsets and col_widths cover only the visible ones.
_TVF_DESC_DTYPE = np.dtype([
    ("ncols", "i4"), ("n_hidden", "i4"), ("itemsize", "i8"),
    ("col_offsets", "i8"), ("col_tags", "i8"), ("col_widths", "i8"),
    ("schema_ptr", "i8"), ("scratch_bytes", "i8"),
], align=True)

# _TVF_CUR_DTYPE describes the cursor. mi_p and data_p hold the MemInfo and the data address of the array that fn
# returned for the current filter, and n_rows and row_stride describe that array. scratch_p is the scratch buffer for
# 'U' cells, and args_p is the buffer of decoded arguments. The cursor holds a reference to each array from the moment
# that xFilter stores the array until the next xFilter or xClose releases it, so each array is released once.
_TVF_CUR_DTYPE = np.dtype([
    ("base", _SQLITE3_VTAB_CURSOR_DTYPE), ("descriptor", "i8"), ("rowid", "i8"),
    ("mi_p", "i8"), ("data_p", "i8"), ("n_rows", "i8"), ("row_stride", "i8"), ("scratch_p", "i8"),
    ("args_p", "i8"),
], align=True)
_TVF_CUR_SIZE = _TVF_CUR_DTYPE.itemsize
# _ARG_SLOT is the number of bytes for each hidden argument in the cursor's args_p buffer, which holds each argument as
# the int64 or float64 that xFilter decoded.
_ARG_SLOT = 8

_INT_TAGS = frozenset((_TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64,
                       _TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64, _TAG_BOOL))
_FLOAT_TAGS = frozenset((_TAG_F32, _TAG_F64))


# --- generated-source templates -------------------------------------------
# The template of a registration's xFilter implementation. The code runs in a namespace seeded from this module's
# globals, which hold every helper imported above, with _fn (the caller's function), _N_HIDDEN and jit_options added.
# {arg_decode} and {fn_call} are the only parts that vary with the number of arguments. _compile_xfilter fills them
# from arg_types before it compiles the code, as udf_helpers does for an aggregate.
# The implementation first releases the array of the cursor's previous filter. It then decodes each argument from
# argv, keeps it in the cursor for xColumn, and calls _fn. _incref_meminfo takes a reference to the returned array for
# the cursor, so the array survives the function's return. The intrinsic marks its function so that numba's
# pass that prunes reference counts, which numba 0.66 removed, cannot delete the increment.
# Registrations with different out_dtypes never share compiled code: each compiles from an anchor of its own, named
# after the code and digest((out_dtype, tuple(arg_tags)), [fn]).
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
        # The cursor keeps the value for xColumn, which returns it for the hidden column.
        lines.append("    store_at(c[0].args_p + %d, a%d)" % (_ARG_SLOT * i, i))
    return "\n".join(lines)


# --- per-registration cfunc factories -------------------------------------
# Each call of a factory compiles new cfunc objects for one registration. Their code is the same for every
# registration, because each callback reads the registration's layout from its descriptor. With caching enabled,
# numba therefore loads the compiled code from its cache after the first compilation.
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
            # SQLite zeroes the sqlite3_vtab base after this callback returns and then sets its fields, so the code
            # sets only descriptor, numbox's own member.
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
        # Every hidden argument column needs a usable EQ constraint. Each one is bound to argv in declaration order:
        # argvIndex is the argument's index plus 1, and omit=1 makes SQLite pass the value and drop the constraint
        # that the call syntax created. If an argument is unbound, the callback returns SQLITE_CONSTRAINT, so SQLite
        # rejects the plan and never calls xFilter with too few arguments. If no plan binds every argument, the
        # statement fails to prepare with SQLite's error "no query solution".
        v = carray(_cast_int_to_void_p(vtab), (1,), dtype=_VTAB_DTYPE)
        d = carray(_cast_int_to_void_p(v[0].descriptor), (1,), dtype=_TVF_DESC_DTYPE)
        ii = carray(_cast_int_to_void_p(idx_info), (1,), dtype=_IDX_INFO_DTYPE)
        ncols = d[0].ncols
        n_hidden = d[0].n_hidden
        n_constraint = ii[0].nConstraint
        cons = carray(_cast_int_to_void_p(ii[0].aConstraint), (n_constraint,), dtype=_CONSTRAINT_DTYPE)
        usage = carray(_cast_int_to_void_p(ii[0].aConstraintUsage), (n_constraint,), dtype=_USAGE_DTYPE)

        # Each hidden argument is bound separately, to the first usable EQ constraint on its column. A running count
        # of bound constraints would count a duplicate EQ on one argument, and so miss another argument that is
        # unbound. A 64-bit mask would limit a function to 63 arguments. A second EQ on the same argument, as in
        # f(1) WHERE arg0 = 1, receives no argvIndex, because SQLite refuses a plan in which two constraints share one.
        # It is not omitted either, so SQLite checks it against the hidden column. The scan takes
        # n_hidden * n_constraint steps: microseconds for a few arguments, and about a millisecond at SQLite's
        # default limit of 2000 columns. It runs only when SQLite plans a query.
        for h in range(n_hidden):
            bound = False
            for i in range(n_constraint):
                if (cons[i].usable != 0 and cons[i].op == SQLITE_INDEX_CONSTRAINT_EQ
                        and cons[i].iColumn - ncols == h):
                    usage[i].argvIndex = int32(h + 1)
                    usage[i].omit = 1
                    bound = True
                    break
            if not bound:
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
        args_p = 0
        try:
            v = carray(_cast_int_to_void_p(vtab), (1,), dtype=_VTAB_DTYPE)
            desc = v[0].descriptor
            d = carray(_cast_int_to_void_p(desc), (1,), dtype=_TVF_DESC_DTYPE)
            scratch = d[0].scratch_bytes
            n_hidden = d[0].n_hidden
            cur = sqlite3_malloc(int32(_TVF_CUR_SIZE))
            if cur == 0:
                return SQLITE_NOMEM
            if scratch > 0:
                scratch_p = sqlite3_malloc(int32(scratch))
                if scratch_p == 0:
                    sqlite3_free(cur)
                    return SQLITE_NOMEM
            if n_hidden > 0:
                args_p = sqlite3_malloc(int32(_ARG_SLOT * n_hidden))
                if args_p == 0:
                    sqlite3_free(scratch_p)
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
            c[0].args_p = args_p
            slot = carray(_cast_int_to_void_p(pp_cursor), (1,), dtype=np.intp)
            slot[0] = cur
            return SQLITE_OK
        except Exception:
            sqlite3_free(args_p)
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
            sqlite3_free(c[0].args_p)
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
            tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols + d[0].n_hidden,), dtype=tags_buf_t)
            if j >= ncols:
                # A hidden argument column returns the argument that xFilter decoded for fn, as SQLite's own
                # table-valued functions do.
                arg_p = c[0].args_p + _ARG_SLOT * (j - ncols)
                if tags[j] == _TAG_F32 or tags[j] == _TAG_F64:
                    sqlite3_result_double(ctx, load_unaligned(arg_p, float64))
                else:
                    sqlite3_result_int64(ctx, load_unaligned(arg_p, int64))
                return SQLITE_OK
            offsets = carray(_cast_int_to_void_p(d[0].col_offsets), (ncols,), dtype=np.int64)
            widths = carray(_cast_int_to_void_p(d[0].col_widths), (ncols,), dtype=np.int64)
            addr = data_p + rowid * c[0].row_stride + offsets[j]
            # register_table passes 'S' and BLOB results as SQLITE_STATIC, but every result here is SQLITE_TRANSIENT.
            # It points into the cursor's array, or into its scratch buffer, and the next xFilter releases the array
            # and replaces it, so a static pointer could outlive its memory.
            return _emit_cell(ctx, addr, tags[j], widths[j], c[0].scratch_p, _SQLITE_TRANSIENT)
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
    # The digest covers out_dtype, the argument tags, numba's and numbox's versions, numbox's jit options and the
    # fingerprint of fn. The anchor's name ends in a hash of the code text, which starts with the digest, so an edit
    # that changes the digest compiles new code. If fn reads a value without a fingerprint, digest falls back to a
    # weaker fingerprint. It names the type of such a value instead of the value and omits the module attributes that
    # fn reads, so an edit to such a value or to such an attribute keeps the anchor and the cached code. The utils page
    # describes the case under "If a SQLite function keeps returning old results".
    tvf_digest = digest((out_dtype, tuple(arg_tags)), [fn])
    code_txt = "# tvf-digest: %s\n%s" % (tvf_digest, src)
    anchor = _anchor_path(_ANCHOR_SUBDIR, stem, code_txt)
    ns = {**globals(), "_fn": fn, "_N_HIDDEN": n_hidden,
          "jit_options": _anchored_or_uncached(anchor, code_txt, jit_options)}
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
    # tags_buf holds one tag for each column of the schema, in the order that the schema below declares them: the
    # visible columns, then the hidden arguments.
    tags_buf = np.array(vis_tags + arg_tags, dtype=tags_buf_t)
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
    """The objects that a registration needs while SQLite can call it: its module struct, descriptor, buffers and code.

    The handle holds the ``sqlite3_module`` struct, the descriptor, the buffers that the descriptor points into, the
    schema text, ``fn``, the generated ``xFilter`` implementation and every callback. SQLite and the callbacks read
    the struct, the descriptor and its buffers through their addresses, so these must stay alive. numba keeps a
    cfunc's compiled code after the cfunc object is collected, but it does not document that behavior, so the handle
    holds the callbacks as well. ``_DATA_ANCHOR`` holds the handle until SQLite calls the destructor that the
    registration passed to ``sqlite3_create_module_v2``.
    """
    __slots__ = ("_keep",)

    def __init__(self, *objs):
        self._keep = objs


def _stem(name):
    return "tvf_" + "".join(c if c.isalnum() else "_" for c in name)


def register_tvf(db, name, arg_types, out_dtype, fn):
    """Register ``fn`` on the connection ``db`` as the eponymous table-valued function ``name``.

    ``SELECT * FROM name(a, b)`` calls ``fn(a, b)`` and returns the rows of the array that ``fn`` returns, a 1-D
    NumPy structured array of ``out_dtype``. ``fn`` is a plain Python function, which numbox compiles with ``njit``,
    or a jitted one. ``arg_types`` holds one integer, float or ``bool`` NumPy dtype for each argument. ``fn``
    receives an integer or ``bool`` argument as an int64, and a float argument as a float64. Each argument is also a
    hidden column of the table, after the columns of ``out_dtype``, and the hidden columns are named ``arg0``,
    ``arg1``, ``arg2``, in the order of the arguments. Selecting a hidden column returns the value that ``fn``
    received. A query must supply every argument: one that leaves an argument unbound, such as
    ``SELECT * FROM name``, fails to prepare with SQLite's error "no query solution".

    ``fn`` must return an array whose dtype is ``out_dtype``. A slice, a strided view or an offset view is read
    correctly, because the row stride is honored. An array of another dtype is read through the layout of
    ``out_dtype`` and yields undefined values. A query returns SQL NULL for a NaN float cell, because SQLite converts a
    NaN REAL to NULL, as in :func:`~numbox.core.bindings.sqlite.vtable.register_table`. If ``fn`` raises, the query
    fails with ``SQLITE_ERROR``.

    numba compiles ``fn`` and the generated code that calls it during this call, so numba raises a typing error in
    ``fn`` from this call. The generated code compiles under numbox's jit options, and numbox names its cache after a
    digest of ``out_dtype``, the argument types and ``fn``. If ``fn`` reads a value without a fingerprint, the digest
    misses some edits. The registration can then run code compiled before an edit, as the section "If a SQLite
    function keeps returning old results" of :doc:`numbox.utils` explains.

    numbox keeps the registration's objects, ``fn`` included, until SQLite drops the module, when the connection
    closes or when the name is registered again. Close the connection before the interpreter exits: if Python closes
    a ``sqlite3.Connection`` during its shutdown after a query of the function, the process can end with a
    segmentation fault.

    Raises ``TypeError`` when ``out_dtype`` is not a structured dtype or has a field without a SQLite type, when an
    argument type is not an integer, float or ``bool`` dtype, and when ``fn`` is not callable. Raises
    ``RuntimeError``, with SQLite's message, when SQLite refuses the registration.
    """
    c, offsets_buf, tags_buf, widths_buf, schema, arg_tags = _build_tvf_descriptor(
        name, arg_types, out_dtype)

    xfilter_impl = _compile_xfilter(_stem(name), arg_tags, out_dtype, _prepare_fn(fn))

    # cache=False, unlike every other callback here: this one closes over xfilter_impl, a dispatcher made for each
    # registration, so its cache key never repeats. With cache=True, each registration in each process writes an
    # index entry that no later lookup reads. Three identical processes produced one, two and three overloads, and four
    # registrations in one process produced four, while each of the other callbacks had one. The anchor of
    # xfilter_impl, which _compile_xfilter writes, is what caches this path.
    @cfunc(types.int32(types.intp, types.int32, types.intp, types.int32, types.intp), cache=False)
    def _tvf_xfilter(cur, idx_num, idx_str, argc, argv):
        # The try/except converts an exception from fn into SQLITE_ERROR, xFilter's own return code. Without it, the
        # cfunc boundary would catch the exception and return the zero default, SQLITE_OK, and SQLite would report a
        # successful query with no rows. A try/finally with no except clause does not prevent this: numba 0.65.1
        # fails on its re-raise, and the cfunc boundary catches that error too. xfilter_impl empties the cursor before
        # it calls fn, so the cursor is left consistent and empty when fn raises.
        try:
            xfilter_impl(cur, argc, argv)
        except Exception:
            return SQLITE_ERROR
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
