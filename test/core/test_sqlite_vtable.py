import gc
import os
import subprocess
import sys
import textwrap
from ctypes import addressof, c_int64, cast, c_char_p, string_at

import sqlite3
import pytest
import numpy as np
from numbox.utils.cstrings import c_string
from numbox.core.bindings.sqlite.conn import sqlite3_open, sqlite3_close
from numbox.core.bindings.sqlite.vtable import register_table
from numbox.core.bindings.sqlite.stmt import sqlite3_prepare_v2, sqlite3_step, sqlite3_finalize
from numbox.core.bindings.sqlite.column import (
    sqlite3_column_count, sqlite3_column_type, sqlite3_column_int64, sqlite3_column_double,
    sqlite3_column_text, sqlite3_column_blob, sqlite3_column_bytes,
)
from numbox.core.bindings.sqlite.query import query_to_array
from numbox.utils.pysqlite_bridge import extract_connection_ptr, libraries_coordinated
from numbox.core.bindings.sqlite.vtable import utf32_to_utf8, _nul_trimmed_len
from numbox.core.bindings.sqlite.vtable import _build_descriptor, _TAG_I64, _TAG_F64, _TAG_U
from numbox.utils.lowlevel import array_data_p


def test_imports():
    from numbox.core.bindings.sqlite import vtable as v
    assert hasattr(v.sqlite3_create_module_v2, "py_func")
    assert hasattr(v.sqlite3_declare_vtab, "py_func")
    assert hasattr(v.sqlite3_malloc, "py_func")


def _encode(s, width):
    src = np.zeros(width, dtype=np.uint32)
    for i, ch in enumerate(s):
        src[i] = ord(ch)
    dst = np.zeros(4 * width + 1, dtype=np.uint8)
    n = utf32_to_utf8(array_data_p(src), width, array_data_p(dst))
    return bytes(dst[:n])


def test_utf32_ascii():
    assert _encode("abc", 6) == b"abc"


def test_utf32_multibyte():
    assert _encode("héllo", 6) == "héllo".encode("utf-8")


def test_utf32_emoji_4byte():
    assert _encode("a\U0001F600b", 6) == "a\U0001F600b".encode("utf-8")


def test_utf32_stops_at_nul():
    assert _encode("hi", 6) == b"hi"


def test_utf32_keeps_interior_nul():
    # trailing NUL pad is trimmed; an interior NUL code point is preserved
    assert _encode("a\x00b", 6) == b"a\x00b"


def test_utf32_invalid_codepoint_replacement():
    src = np.array([0x41, 0xD800, 0x110000, 0], dtype=np.uint32)
    dst = np.zeros(64, dtype=np.uint8)
    n = utf32_to_utf8(array_data_p(src), 4, array_data_p(dst))
    assert bytes(dst[:n]) == b"A" + b"\xef\xbf\xbd" + b"\xef\xbf\xbd"


def test_nul_trimmed_len():
    buf = np.frombuffer(b"hi\x00\x00\x00", dtype=np.uint8).copy()
    assert _nul_trimmed_len(array_data_p(buf), 5) == 2


def test_nul_trimmed_len_keeps_interior():
    buf = np.frombuffer(b"a\x00b\x00\x00", dtype=np.uint8).copy()
    assert _nul_trimmed_len(array_data_p(buf), 5) == 3


def test_descriptor_2d_int64():
    a = np.arange(6, dtype=np.int64).reshape(3, 2)
    d = _build_descriptor(a, ["a", "b"], False)
    assert (d.nrows, d.ncols) == (3, 2)
    assert list(d.strides) == [a.strides[0], a.strides[0]]
    assert [b - a.ctypes.data for b in d.bases] == [0, 8]
    assert list(d.tags) == [_TAG_I64, _TAG_I64]
    assert d.schema == b'CREATE TABLE x("a" INTEGER, "b" INTEGER)\x00'


def test_descriptor_columnar_layout():
    from numbox.core.bindings.sqlite.vtable import _build_descriptor_columnar
    a = np.arange(3, dtype=np.int64)
    b = np.arange(3, dtype=np.float64)
    d = _build_descriptor_columnar({"a": a, "b": b}, False)
    assert (d.nrows, d.ncols) == (3, 2)
    assert list(d.tags) == [_TAG_I64, _TAG_F64]
    assert [base for base in d.bases] == [a.ctypes.data, b.ctypes.data]
    assert list(d.strides) == [8, 8]
    assert d.schema == b'CREATE TABLE x("a" INTEGER, "b" REAL)\x00'


def test_descriptor_structured_columns_rename_length_mismatch():
    dt = np.dtype([("x", "i8"), ("y", "f8"), ("z", "i4")])
    a = np.zeros(2, dtype=dt)
    with pytest.raises(ValueError):
        _build_descriptor(a, ["a", "b"], False)


def test_descriptor_structured_mixed():
    dt = np.dtype([("t", "U6"), ("q", "i8"), ("p", "f8")])
    a = np.zeros(2, dtype=dt)
    d = _build_descriptor(a, None, False)
    assert d.ncols == 3
    assert list(d.tags) == [_TAG_U, _TAG_I64, _TAG_F64]
    assert [b - a.ctypes.data for b in d.bases] == [dt.fields["t"][1], dt.fields["q"][1], dt.fields["p"][1]]
    assert d.scratch_bytes == 6 * 4 + 1
    assert d.schema == b'CREATE TABLE x("t" TEXT, "q" INTEGER, "p" REAL)\x00'


def test_descriptor_rejects_bad_shapes():
    with pytest.raises((TypeError, ValueError)):
        _build_descriptor(np.zeros((2, 2, 2), dtype=np.int64), ["a", "b"], False)
    with pytest.raises((TypeError, ValueError)):
        _build_descriptor(np.arange(4, dtype=np.int64), ["a"], False)
    with pytest.raises((TypeError, ValueError)):
        _build_descriptor(np.zeros((2, 3), dtype=np.int64), ["a", "b"], False)
    with pytest.raises((TypeError, ValueError)):
        _build_descriptor(np.empty((2, 2), dtype=object), ["a", "b"], False)
    with pytest.raises((TypeError, ValueError)):
        _build_descriptor(np.empty((3, 0), dtype=np.int64), [], False)


def test_widest_unicode_width_accepted():
    # The per-cursor scratch buffer is its own int32-sized sqlite3_malloc, and
    # numpy caps 'U' itemsize below 2**31, so even the widest constructible
    # unicode column yields a scratch size that fits int32 -- no
    # registration-time guard.
    dt = np.dtype([("u", "U536870911")])
    built = _build_descriptor(np.zeros(0, dtype=dt), None, False)
    assert built.scratch_bytes == 4 * 536870911 + 1
    assert built.scratch_bytes <= 2 ** 31 - 1


def test_descriptor_dtype_itemsize():
    from numbox.core.bindings.sqlite.vtable import _DESC_DTYPE
    assert _DESC_DTYPE.itemsize == 64


def test_index_info_offsets_match_c_abi():
    import ctypes
    from numbox.core.bindings.sqlite.vtable import _IDX_INFO_DTYPE

    class _IndexInfo(ctypes.Structure):
        _fields_ = [
            ("nConstraint", ctypes.c_int),
            ("aConstraint", ctypes.c_void_p),
            ("nOrderBy", ctypes.c_int),
            ("aOrderBy", ctypes.c_void_p),
            ("aConstraintUsage", ctypes.c_void_p),
            ("idxNum", ctypes.c_int),
            ("idxStr", ctypes.c_void_p),
            ("needToFreeIdxStr", ctypes.c_int),
            ("orderByConsumed", ctypes.c_int),
            ("estimatedCost", ctypes.c_double),
            ("estimatedRows", ctypes.c_longlong),
        ]
    assert _IDX_INFO_DTYPE.fields["estimatedCost"][1] == _IndexInfo.estimatedCost.offset
    assert _IDX_INFO_DTYPE.fields["estimatedRows"][1] == _IndexInfo.estimatedRows.offset


def test_xbestindex_sets_cardinality():
    from numbox.core.bindings.sqlite.vtable import (
        _xbestindex, _build_descriptor, _VTAB_DTYPE, _VTAB_SIZE, _IDX_INFO_DTYPE,
    )
    arr = np.arange(12, dtype=np.int64).reshape(4, 3)
    built = _build_descriptor(arr, ["a", "b", "c"], False)
    vtab = np.zeros(_VTAB_SIZE // 8, dtype=np.int64)
    vtab.view(_VTAB_DTYPE)[0]["descriptor"] = built.c.ctypes.data
    ii = np.zeros(_IDX_INFO_DTYPE.itemsize // 8, dtype=np.int64)
    rc = _xbestindex.ctypes(int(vtab.ctypes.data), int(ii.ctypes.data))
    assert rc == 0
    view = ii.view(_IDX_INFO_DTYPE)[0]
    assert int(view["estimatedRows"]) == 4
    assert float(view["estimatedCost"]) == 4.0


def test_xbestindex_estimates_bound_branch():
    # With nbound usable EQ constraints on numeric columns, estimatedRows is the
    # heuristic nrows // (nbound + 1) + 1 and estimatedCost is float(nrows).
    from numbox.core.bindings.sqlite.vtable import (
        _xbestindex, _build_descriptor, _VTAB_DTYPE, _VTAB_SIZE, _IDX_INFO_DTYPE,
        _CONSTRAINT_DTYPE, _USAGE_DTYPE,
    )
    from numbox.core.bindings.sqlite.constants import SQLITE_INDEX_CONSTRAINT_EQ
    from numbox.core.bindings.sqlite.exec import sqlite3_free
    nrows = 10
    arr = np.arange(nrows * 3, dtype=np.int64).reshape(nrows, 3)
    built = _build_descriptor(arr, ["a", "b", "c"], False)
    vtab = np.zeros(_VTAB_SIZE // 8, dtype=np.int64)
    vtab.view(_VTAB_DTYPE)[0]["descriptor"] = built.c.ctypes.data

    nbound = 2
    cons = np.zeros(nbound, dtype=_CONSTRAINT_DTYPE)
    for i in range(nbound):
        cons[i]["iColumn"] = i
        cons[i]["op"] = SQLITE_INDEX_CONSTRAINT_EQ
        cons[i]["usable"] = 1
    usage = np.zeros(nbound, dtype=_USAGE_DTYPE)
    ii = np.zeros(1, dtype=_IDX_INFO_DTYPE)
    ii[0]["nConstraint"] = nbound
    ii[0]["aConstraint"] = cons.ctypes.data
    ii[0]["aConstraintUsage"] = usage.ctypes.data
    rc = _xbestindex.ctypes(int(vtab.ctypes.data), int(ii.ctypes.data))
    assert rc == 0
    view = ii.view(_IDX_INFO_DTYPE)[0]
    assert int(view["idxNum"]) == nbound
    assert usage["argvIndex"].tolist() == [1, 2]
    assert int(view["estimatedRows"]) == nrows // (nbound + 1) + 1
    assert float(view["estimatedCost"]) == float(nrows)
    # xBestIndex handed the serialised (col, op) idxStr to SQLite; free it here
    # since no real connection owns it in this direct-call harness.
    if int(view["needToFreeIdxStr"]):
        sqlite3_free(int(view["idxStr"]))


_SQLITE_ROW = 100
_T_INT, _T_FLOAT, _T_TEXT, _T_BLOB, _T_NULL = 1, 2, 3, 4, 5


def _open_memory():
    db_p = c_int64(0)
    with c_string(":memory:") as name_p:
        rc = sqlite3_open(name_p, addressof(db_p))
    assert rc == 0, rc
    return db_p.value


def _fetchall(db, sql):
    stmt_p = c_int64(0)
    with c_string(sql) as sql_p:
        rc = sqlite3_prepare_v2(db, sql_p, -1, addressof(stmt_p), 0)
    assert rc == 0, (rc, sql)
    stmt = stmt_p.value
    rows = []
    while sqlite3_step(stmt) == _SQLITE_ROW:
        row = []
        for i in range(sqlite3_column_count(stmt)):
            t = sqlite3_column_type(stmt, i)
            if t == _T_INT:
                row.append(sqlite3_column_int64(stmt, i))
            elif t == _T_FLOAT:
                row.append(sqlite3_column_double(stmt, i))
            elif t == _T_TEXT:
                row.append(cast(sqlite3_column_text(stmt, i), c_char_p).value.decode("utf-8"))
            elif t == _T_BLOB:
                n = sqlite3_column_bytes(stmt, i)
                row.append(string_at(sqlite3_column_blob(stmt, i), n) if n else b"")
            else:
                row.append(None)
        rows.append(tuple(row))
    sqlite3_finalize(stmt)
    return rows


def test_int64_table_select_where_order():
    db = _open_memory()
    a = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.int64)
    h = register_table(db, "points", a, columns=["a", "b"])  # noqa: F841
    assert _fetchall(db, "SELECT a, b FROM points WHERE a >= 2 ORDER BY b DESC") == [(3, 30), (2, 20)]
    sqlite3_close(db)


def test_count_and_sum():
    db = _open_memory()
    a = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.int64)
    h = register_table(db, "t", a, columns=["a", "b"])  # noqa: F841
    assert _fetchall(db, "SELECT COUNT(*), SUM(a) FROM t") == [(3, 6)]
    sqlite3_close(db)


def test_quoted_keyword_and_space_columns():
    db = _open_memory()
    a = np.array([[1, 2]], dtype=np.int64)
    h = register_table(db, "t", a, columns=["order", "group by"])  # noqa: F841
    assert _fetchall(db, 'SELECT "order", "group by" FROM t') == [(1, 2)]
    sqlite3_close(db)


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int32, np.int16, np.uint32, np.int8, np.uint8, np.uint16])
def test_numeric_dtype_roundtrip(dt):
    db = _open_memory()
    a = (np.arange(6, dtype=dt).reshape(3, 2) + 1)
    h = register_table(db, "t", a, columns=["a", "b"])  # noqa: F841
    got = _fetchall(db, "SELECT a, b FROM t ORDER BY a")
    exp = [tuple(row) for row in a.tolist()]
    assert got == exp
    sqlite3_close(db)


def test_uint64_roundtrip_and_signed_wrap():
    db = _open_memory()
    a = np.array([[1], [2 ** 63], [2 ** 64 - 1]], dtype=np.uint64)
    h = register_table(db, "t", a, columns=["a"])  # noqa: F841
    # SQLite INTEGER is a signed int64; uint64 values >= 2**63 reinterpret as negative.
    assert _fetchall(db, "SELECT a FROM t") == [(1,), (-(2 ** 63),), (-1,)]
    sqlite3_close(db)


def test_bool_dtype():
    db = _open_memory()
    a = np.array([[True, False], [False, True]], dtype=np.bool_)
    h = register_table(db, "t", a, columns=["a", "b"])  # noqa: F841
    assert _fetchall(db, "SELECT a, b FROM t") == [(1, 0), (0, 1)]
    sqlite3_close(db)


@pytest.mark.parametrize("dt", [np.float64, np.float32])
def test_float_nan_becomes_null_inf_passes_through(dt):
    db = _open_memory()
    a = np.array([[np.nan], [np.inf], [-np.inf], [1.5]], dtype=dt)
    h = register_table(db, "t", a, columns=["x"])  # noqa: F841
    rows = _fetchall(db, "SELECT x FROM t")
    assert rows[0] == (None,)  # SQLite coerces a NaN REAL to SQL NULL
    assert rows[1] == (float("inf"),)
    assert rows[2] == (float("-inf"),)
    assert rows[3] == (1.5,)
    assert _fetchall(db, "SELECT COUNT(*) FROM t WHERE x IS NULL") == [(1,)]
    sqlite3_close(db)


def test_structured_text_and_unicode():
    db = _open_memory()
    dt = np.dtype([("t", "U6"), ("q", "i4"), ("p", "f8"), ("s", "S4")])
    a = np.array([("héllo", 3, 1.5, b"ab"), ("\U0001F600", 7, 2.0, b"cd")], dtype=dt)
    h = register_table(db, "trades", a)  # noqa: F841
    got = _fetchall(db, "SELECT t, q, p, s FROM trades")
    assert got == [("héllo", 3, 1.5, "ab"), ("\U0001F600", 7, 2.0, "cd")]
    sqlite3_close(db)


def test_structured_columns_rename():
    db = _open_memory()
    dt = np.dtype([("x", "i8"), ("y", "f8")])
    a = np.array([(1, 10.5), (2, 20.5), (3, 30.5)], dtype=dt)
    register_table(db, "t", a, columns=["id", "price"])
    assert _fetchall(db, "SELECT id, price FROM t ORDER BY id") == [(1, 10.5), (2, 20.5), (3, 30.5)]
    sqlite3_close(db)


def test_columns_rename_rejects_non_string():
    a = np.array([[1, 2]], dtype=np.int64)
    with pytest.raises(TypeError):
        register_table(0, "t", a, columns=[1, 2])


def test_two_unicode_columns_single_cursor():
    # Both U columns of a row are decoded into the SAME per-cursor scratch
    # buffer, so SQLite must copy the first result before the second xColumn
    # overwrites the scratch -- guards the TRANSIENT choice on the U branch
    # (S/BLOB results are STATIC; flipping U to match corrupts exactly this).
    db = _open_memory()
    dt = np.dtype([("u1", "U4"), ("u2", "U4")])
    a = np.array([("abcd", "wxyz"), ("hi", "yo")], dtype=dt)
    register_table(db, "t", a)
    assert _fetchall(db, "SELECT u1, u2 FROM t") == [("abcd", "wxyz"), ("hi", "yo")]
    sqlite3_close(db)


def test_text_as_blob():
    db = _open_memory()
    dt = np.dtype([("s", "S3")])
    a = np.array([(b"xy",)], dtype=dt)
    h = register_table(db, "t", a, text_as_blob=True)  # noqa: F841
    assert _fetchall(db, "SELECT s FROM t") == [(b"xy",)]
    assert _fetchall(db, "SELECT typeof(s) FROM t") == [("blob",)]
    sqlite3_close(db)


def test_blob_preserves_interior_nul():
    db = _open_memory()
    dt = np.dtype([("s", "S5")])
    a = np.array([(b"a\x00b",)], dtype=dt)  # numpy stores b"a\x00b\x00\x00"
    h = register_table(db, "t", a, text_as_blob=True)  # noqa: F841
    assert _fetchall(db, "SELECT s FROM t") == [(b"a\x00b",)]
    sqlite3_close(db)


def test_fortran_order_matches_c():
    db = _open_memory()
    a = np.asfortranarray(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64))
    h = register_table(db, "t", a, columns=["a", "b"])  # noqa: F841
    assert _fetchall(db, "SELECT a, b FROM t ORDER BY a") == [(1, 2), (3, 4), (5, 6)]
    sqlite3_close(db)


def test_noncontiguous_slice():
    db = _open_memory()
    big = np.arange(40, dtype=np.int64).reshape(8, 5)
    view = big[::2, 1:4]
    h = register_table(db, "t", view, columns=["a", "b", "c"])  # noqa: F841
    assert _fetchall(db, "SELECT a, b, c FROM t") == [tuple(r) for r in view.tolist()]
    sqlite3_close(db)


def test_reversed_rows():
    db = _open_memory()
    a = np.array([[1], [2], [3]], dtype=np.int64)[::-1]
    h = register_table(db, "t", a, columns=["a"])  # noqa: F841
    assert _fetchall(db, "SELECT a FROM t") == [(3,), (2,), (1,)]
    sqlite3_close(db)


def test_packed_unicode_odd_offset():
    db = _open_memory()
    dt = np.dtype([("a", "i1"), ("u", "U4")])
    assert dt.fields["u"][1] == 1
    a = np.array([(1, "wörd")], dtype=dt)
    h = register_table(db, "t", a)  # noqa: F841
    assert _fetchall(db, "SELECT a, u FROM t") == [(1, "wörd")]
    sqlite3_close(db)


def test_empty_table():
    db = _open_memory()
    a = np.empty((0, 2), dtype=np.int64)
    h = register_table(db, "t", a, columns=["a", "b"])  # noqa: F841
    assert _fetchall(db, "SELECT * FROM t") == []
    assert _fetchall(db, "SELECT COUNT(*) FROM t") == [(0,)]
    sqlite3_close(db)


def test_rowid_is_zero_based():
    db = _open_memory()
    a = np.array([[10], [20], [30]], dtype=np.int64)
    h = register_table(db, "t", a, columns=["a"])  # noqa: F841
    assert _fetchall(db, "SELECT rowid, a FROM t") == [(0, 10), (1, 20), (2, 30)]
    sqlite3_close(db)


def test_join_two_tables():
    db = _open_memory()
    a = np.array([[1, 100], [2, 200]], dtype=np.int64)
    b = np.array([[1, 7], [2, 9]], dtype=np.int64)
    h1 = register_table(db, "lhs", a, columns=["id", "v"])  # noqa: F841
    h2 = register_table(db, "rhs", b, columns=["id", "w"])  # noqa: F841
    got = _fetchall(db, "SELECT lhs.v, rhs.w FROM lhs JOIN rhs ON lhs.id = rhs.id ORDER BY lhs.id")
    assert got == [(100, 7), (200, 9)]
    sqlite3_close(db)


def test_join_refilters_inner_cursor_per_outer_row():
    # A nested-loop join re-invokes xFilter on the inner vtable cursor once per
    # outer row, each time with a DIFFERENT bound value on the join key. This
    # exercises the per-xFilter free+realloc of pred_p and the rowid reset; a
    # correct joined result across multiple distinct outer keys demonstrates it.
    db = _open_memory()
    outer = np.array([[1], [2], [3], [2]], dtype=np.int64)
    inner = np.array([[1, 10], [2, 20], [2, 21], [3, 30]], dtype=np.int64)
    h1 = register_table(db, "o", outer, columns=["k"])  # noqa: F841
    h2 = register_table(db, "i", inner, columns=["k", "v"])  # noqa: F841
    got = sorted(_fetchall(db, "SELECT o.k, i.v FROM o JOIN i ON i.k = o.k"))
    exp = sorted((ok, iv) for (ok,) in outer.tolist() for (ik, iv) in inner.tolist() if ik == ok)
    assert got == exp, (got, exp)
    sqlite3_close(db)


def test_self_join_unicode_two_cursors():
    db = _open_memory()
    dt = np.dtype([("k", "i8"), ("u", "U4")])
    a = np.array([(1, "wörd"), (2, "café"), (3, "ab")], dtype=dt)
    h = register_table(db, "t", a)  # noqa: F841
    # a self-join keeps two cursors open on the same vtable at once, so each
    # decodes its 'U' column into its own per-cursor scratch buffer
    got = _fetchall(db, "SELECT a.u, b.u FROM t a JOIN t b ON a.k = b.k ORDER BY a.k")
    assert got == [("wörd", "wörd"), ("café", "café"), ("ab", "ab")]
    sqlite3_close(db)


def test_registration_survives_gc():
    # register_table returns None; the keep-alive is the module-level
    # _DATA_ANCHOR, so the registration must survive gc.collect() with no
    # user-held handle.
    db = _open_memory()
    a = np.array([[5, 6]], dtype=np.int64)
    assert register_table(db, "t", a, columns=["a", "b"]) is None
    gc.collect()
    assert _fetchall(db, "SELECT a, b FROM t") == [(5, 6)]
    sqlite3_close(db)


def test_duplicate_name_replaces():
    db = _open_memory()
    a = np.array([[1]], dtype=np.int64)
    b = np.array([[2]], dtype=np.int64)
    h1 = register_table(db, "dup", a, columns=["x"])  # noqa: F841
    h2 = register_table(db, "dup", b, columns=["x"])  # noqa: F841
    assert _fetchall(db, "SELECT x FROM dup") == [(2,)]
    sqlite3_close(db)


# A self-contained driver: imports register_table, opens :memory:, registers a
# 2-D int64 table, runs a SELECT, and prints the rows. Run twice in fresh
# subprocesses to assert cold/warm cfunc cache reuse (no growth on the warm run).
_DRIVER = textwrap.dedent('''
    from ctypes import addressof, c_int64
    import numpy as np
    from numbox.core.bindings.sqlite.conn import sqlite3_open
    from numbox.core.bindings.sqlite.stmt import sqlite3_prepare_v2, sqlite3_step, sqlite3_finalize
    from numbox.core.bindings.sqlite.column import sqlite3_column_int64
    from numbox.core.bindings.sqlite.constants import SQLITE_ROW
    from numbox.core.bindings.sqlite.vtable import register_table
    from numbox.utils.cstrings import c_string

    db_p = c_int64(0)
    with c_string(":memory:") as n:
        sqlite3_open(n, addressof(db_p))
    db = db_p.value
    a = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.int64)
    h = register_table(db, "t", a, columns=["a", "b"])
    stmt_p = c_int64(0)
    with c_string("SELECT a, b FROM t ORDER BY a") as sp:
        sqlite3_prepare_v2(db, sp, -1, addressof(stmt_p), 0)
    stmt = stmt_p.value
    rows = []
    while sqlite3_step(stmt) == SQLITE_ROW:
        rows.append((sqlite3_column_int64(stmt, 0), sqlite3_column_int64(stmt, 1)))
    sqlite3_finalize(stmt)
    assert rows == [(1, 10), (2, 20), (3, 30)], rows
    print("RESULT", rows)
''')


def _run_vtable_driver(tmp_path, cache_dir):
    script = tmp_path / "vtable_drv.py"
    script.write_text(_DRIVER)
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache_dir))
    out = subprocess.run([sys.executable, str(script)], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")][0]
    return line


def _count_vtable_nbc(cache_dir):
    # Scope to the vtable cfunc/njit caches only, so the no-growth assertion is
    # immune to unrelated bindings whose compile timing differs across runs.
    return sum(1 for _ in cache_dir.rglob("*vtable.*.nbc"))


def test_xprocess_cache_no_growth(tmp_path):
    cache = tmp_path / "nbcache"
    cache.mkdir()
    line1 = _run_vtable_driver(tmp_path, cache)  # cold: compiles + writes cache
    assert line1 == "RESULT [(1, 10), (2, 20), (3, 30)]"
    n_cold = _count_vtable_nbc(cache)
    assert n_cold > 0
    line2 = _run_vtable_driver(tmp_path, cache)  # warm: must reuse, not append
    assert line2 == "RESULT [(1, 10), (2, 20), (3, 30)]"
    assert _count_vtable_nbc(cache) == n_cold, "warm run grew the cache (cache reuse failed)"


def test_pushdown_element_dtype_itemsizes():
    from numbox.core.bindings.sqlite.vtable import (
        _CONSTRAINT_DTYPE, _USAGE_DTYPE, _ORDERBY_DTYPE,
    )
    assert _CONSTRAINT_DTYPE.itemsize == 12
    assert _USAGE_DTYPE.itemsize == 8
    assert _ORDERBY_DTYPE.itemsize == 8
    from numbox.core.bindings.sqlite.vtable import _SPEC_DTYPE
    assert _SPEC_DTYPE.itemsize == 8
    assert _SPEC_DTYPE.names == ("col", "op")
    assert _SPEC_DTYPE.fields["col"][1] == 0
    assert _SPEC_DTYPE.fields["op"][1] == 4


def _select_col0(db, sql):
    """Collect column 0 of every result row as Python ints."""
    return [r[0] for r in _fetchall(db, sql)]


def test_pushdown_eq():
    db = _open_memory()
    a = np.array([[3], [5], [7], [5], [9]], dtype=np.int64)
    h = register_table(db, "t", a, columns=["c"])  # noqa: F841
    assert sorted(_select_col0(db, "SELECT c FROM t WHERE c = 5")) == [5, 5]
    assert _select_col0(db, "SELECT c FROM t WHERE c = 7") == [7]
    assert _select_col0(db, "SELECT c FROM t WHERE c = 100") == []
    sqlite3_close(db)


def test_pushdown_range_matches_fullscan():
    db = _open_memory()
    vals = [3, 5, 7, 5, 9, 1, 5]
    a = np.array([[v] for v in vals], dtype=np.int64)
    h = register_table(db, "t", a, columns=["c"])  # noqa: F841
    preds = {
        ">": lambda v: v > 5,
        ">=": lambda v: v >= 5,
        "<": lambda v: v < 5,
        "<=": lambda v: v <= 5,
    }
    for op, pred in preds.items():
        got = sorted(_select_col0(db, "SELECT c FROM t WHERE c %s 5" % op))
        exp = sorted(v for v in vals if pred(v))
        assert got == exp, (op, got, exp)
    sqlite3_close(db)


def test_pushdown_multi_constraint():
    db = _open_memory()
    rows = [(1, 9), (4, 2), (7, 1), (3, 8), (6, 5), (2, 3), (9, 0)]
    a = np.array(rows, dtype=np.int64)
    h = register_table(db, "t", a, columns=["x", "y"])  # noqa: F841
    got = sorted(
        (r[0], r[1]) for r in _fetchall(db, "SELECT x, y FROM t WHERE x >= 3 AND y < 5")
    )
    exp = sorted((x, y) for (x, y) in rows if x >= 3 and y < 5)
    assert got == exp, (got, exp)
    sqlite3_close(db)


def test_pushdown_float_column():
    db = _open_memory()
    vals = [1.5, 2.5, 5.0, 7.25, 5.0, 0.5]
    a = np.array([[v] for v in vals], dtype=np.float64)
    h = register_table(db, "t", a, columns=["c"])  # noqa: F841
    got = sorted(r[0] for r in _fetchall(db, "SELECT c FROM t WHERE c >= 5.0"))
    exp = sorted(v for v in vals if v >= 5.0)
    assert got == exp, (got, exp)
    eq = sorted(r[0] for r in _fetchall(db, "SELECT c FROM t WHERE c = 5.0"))
    assert eq == [5.0, 5.0]
    sqlite3_close(db)


def test_pushdown_no_constraint_full_scan():
    db = _open_memory()
    a = np.array([[3], [5], [7], [5], [9]], dtype=np.int64)
    h = register_table(db, "t", a, columns=["c"])  # noqa: F841
    assert _select_col0(db, "SELECT c FROM t") == [3, 5, 7, 5, 9]
    sqlite3_close(db)


def test_xdestroy_pops_anchor_on_close():
    from numbox.core.bindings.sqlite import vtable as v
    db = c_int64(0)
    with c_string(":memory:") as p:
        sqlite3_open(p, addressof(db))
    arr = np.array([[1], [2]], dtype=np.int64)
    register_table(db.value, "t", arr, ["c"])
    assert _select_col0(db.value, "SELECT c FROM t") == [1, 2]
    n_before = len(v._DATA_ANCHOR)
    sqlite3_close(db.value)
    assert len(v._DATA_ANCHOR) == n_before - 1


def test_xdestroy_two_tables():
    from numbox.core.bindings.sqlite import vtable as v
    db = c_int64(0)
    with c_string(":memory:") as p:
        sqlite3_open(p, addressof(db))
    h1 = register_table(db.value, "t1", np.array([[1]], np.int64), ["c"])
    h2 = register_table(db.value, "t2", np.array([[2]], np.int64), ["c"])
    n = len(v._DATA_ANCHOR)
    sqlite3_close(db.value)
    assert len(v._DATA_ANCHOR) == n - 2
    del h1, h2


def test_xdestroy_three_tables():
    from numbox.core.bindings.sqlite import vtable as v
    db = c_int64(0)
    with c_string(":memory:") as p:
        sqlite3_open(p, addressof(db))
    h1 = register_table(db.value, "t1", np.array([[1]], np.int64), ["c"])
    h2 = register_table(db.value, "t2", np.array([[2]], np.int64), ["c"])
    h3 = register_table(db.value, "t3", np.array([[3]], np.int64), ["c"])
    n = len(v._DATA_ANCHOR)
    sqlite3_close(db.value)
    assert len(v._DATA_ANCHOR) == n - 3
    del h1, h2, h3


def test_xdestroy_reregister_drops_first():
    from numbox.core.bindings.sqlite import vtable as v
    db = c_int64(0)
    with c_string(":memory:") as p:
        sqlite3_open(p, addressof(db))
    a = np.array([[1]], np.int64)
    b = np.array([[2]], np.int64)
    keys0 = set(v._DATA_ANCHOR)
    register_table(db.value, "t", a, ["c"])
    (key1,) = set(v._DATA_ANCHOR) - keys0  # the first descriptor pointer = its anchor key
    register_table(db.value, "t", b, ["c"])
    # re-registration fires the FIRST descriptor's xDestroy synchronously
    assert key1 not in v._DATA_ANCHOR, "first entry not dropped on re-register"
    (key2,) = set(v._DATA_ANCHOR) - keys0
    assert key2 != key1
    assert _select_col0(db.value, "SELECT c FROM t") == [2]
    n = len(v._DATA_ANCHOR)
    sqlite3_close(db.value)
    assert key2 not in v._DATA_ANCHOR
    assert len(v._DATA_ANCHOR) == n - 1


def test_xdestroy_no_c_free_of_descriptor():
    from numbox.core.bindings.sqlite import vtable as v
    db = c_int64(0)
    with c_string(":memory:") as p:
        sqlite3_open(p, addressof(db))
    arr = np.array([[11], [22], [33]], dtype=np.int64)
    keys0 = set(v._DATA_ANCHOR)
    register_table(db.value, "t", arr, ["c"])
    (key,) = set(v._DATA_ANCHOR) - keys0
    anchor = v._DATA_ANCHOR[key]  # hold the handle past xDestroy
    assert _select_col0(db.value, "SELECT c FROM t") == [11, 22, 33]
    n_before = len(v._DATA_ANCHOR)
    sqlite3_close(db.value)
    assert len(v._DATA_ANCHOR) == n_before - 1
    # xDestroy must NOT have C-freed the numpy-owned descriptor/array: the test
    # still holds arr and the anchored handle, so reading both must not
    # segfault and both must be unchanged.
    assert arr.tolist() == [[11], [22], [33]]
    assert anchor._keep[0].c["nrows"][0] == 3


def test_xdestroy_tvf_pops_anchor_on_close():
    from numbox.core.bindings.sqlite import vtable as v
    from numbox.core.bindings.sqlite.tvf import register_tvf
    from numba import njit

    out = np.dtype([("n", "i8")])

    @njit
    def _series(start, stop):
        o = np.empty(stop - start, out)
        for i in range(stop - start):
            o[i].n = start + i
        return o

    db = c_int64(0)
    with c_string(":memory:") as p:
        sqlite3_open(p, addressof(db))
    assert register_tvf(db.value, "series", (np.int64, np.int64), out, _series) is None
    stmt = c_int64(0)
    with c_string("SELECT n FROM series(2, 5)") as p:
        sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    got = []
    while sqlite3_step(stmt.value) == _SQLITE_ROW:
        got.append(sqlite3_column_int64(stmt.value, 0))
    sqlite3_finalize(stmt.value)
    assert got == [2, 3, 4]
    n_before = len(v._DATA_ANCHOR)
    sqlite3_close(db.value)
    assert len(v._DATA_ANCHOR) == n_before - 1


def test_pushdown_explain_uses_index():
    # SQLite reports the chosen vtable plan as "VIRTUAL TABLE INDEX <idxNum>:".
    # A full scan claims nothing, so idxNum stays 0; a claimed constraint sets a
    # non-zero idxNum. Assert the constrained query picks a non-zero idxNum and
    # the unconstrained query does not.
    db = _open_memory()
    a = np.array([[3], [5], [7]], dtype=np.int64)
    h = register_table(db, "t", a, columns=["c"])  # noqa: F841
    plan = _fetchall(db, "EXPLAIN QUERY PLAN SELECT c FROM t WHERE c = 5")
    text = " ".join(str(field) for row in plan for field in row).upper()
    assert "VIRTUAL TABLE INDEX 1:" in text, text
    full = _fetchall(db, "EXPLAIN QUERY PLAN SELECT c FROM t")
    full_text = " ".join(str(field) for row in full for field in row).upper()
    assert "VIRTUAL TABLE INDEX 0:" in full_text, full_text
    sqlite3_close(db)


def test_text_column_where_not_pushed_down():
    # String/text columns are non-numeric, so xBestIndex never claims a WHERE on
    # them: the plan stays at idxNum 0 (full scan) and the omit=0 re-check filters
    # the surfaced rows -- the full-scan fallback must still return correct rows.
    db = _open_memory()
    dt = np.dtype([("name", "U4"), ("a", "i8")])
    a = np.array([("foo", 1), ("bar", 2), ("foo", 3)], dtype=dt)
    register_table(db, "t", a)
    plan = _fetchall(db, "EXPLAIN QUERY PLAN SELECT a FROM t WHERE name = 'foo'")
    text = " ".join(str(field) for row in plan for field in row).upper()
    assert "INDEX 0" in text, text
    assert sorted(_select_col0(db, "SELECT a FROM t WHERE name = 'foo'")) == [1, 3]
    sqlite3_close(db)


def test_rowid_constraint_not_pushed_down():
    # The 0 <= col < ncols guard in xBestIndex excludes the iColumn == -1 rowid
    # sentinel, so a rowid constraint is never claimed (idxNum stays 0). rowid is
    # the 0-based scan position; the full-scan fallback must still be correct.
    db = _open_memory()
    a = np.array([[10], [20], [30]], dtype=np.int64)
    h = register_table(db, "t", a, columns=["a"])  # noqa: F841
    assert _fetchall(db, "SELECT a FROM t WHERE rowid >= 1 ORDER BY a") == [(20,), (30,)]
    assert _fetchall(db, "SELECT a FROM t WHERE rowid = 2") == [(30,)]
    plan = _fetchall(db, "EXPLAIN QUERY PLAN SELECT a FROM t WHERE rowid >= 1")
    text = " ".join(str(field) for row in plan for field in row).upper()
    assert "INDEX 0" in text, text
    sqlite3_close(db)


@pytest.mark.parametrize("dt", [np.int32, np.uint16, np.float32])
def test_pushdown_smaller_dtypes_match_fullscan(dt):
    # EQ + a range op pushed down on the smaller numeric dtypes, cross-checked
    # against a full-scan (stdlib-equivalent) expectation.
    db = _open_memory()
    vals = [3, 5, 7, 5, 9, 1, 5]
    a = np.array([[v] for v in vals], dtype=dt)
    h = register_table(db, "t", a, columns=["c"])  # noqa: F841
    typed = [dt(v).item() for v in vals]
    assert sorted(_select_col0(db, "SELECT c FROM t WHERE c = 5")) == sorted(v for v in typed if v == 5)
    assert sorted(_select_col0(db, "SELECT c FROM t WHERE c >= 5")) == sorted(v for v in typed if v >= 5)
    sqlite3_close(db)


def test_pushdown_bool_column():
    # A bool column maps to SQLite INTEGER and the cursor reads/compares it, so
    # an eq constraint must be pushed down (non-zero idxNum), not full-scanned.
    db = _open_memory()
    a = np.array([[True], [False], [True], [False], [True]], dtype=np.bool_)
    h = register_table(db, "t", a, columns=["c"])  # noqa: F841
    assert _select_col0(db, "SELECT c FROM t WHERE c = 1") == [1, 1, 1]
    assert _select_col0(db, "SELECT c FROM t WHERE c = 0") == [0, 0]
    plan = _fetchall(db, "EXPLAIN QUERY PLAN SELECT c FROM t WHERE c = 1")
    text = " ".join(str(field) for row in plan for field in row).upper()
    assert "VIRTUAL TABLE INDEX 1:" in text, text
    sqlite3_close(db)


def test_pushdown_bool_range_ops_match_stdlib():
    # Bool pushdown goes through the FLOAT comparison domain (_is_int_tag
    # excludes bool): an int64-truncated RHS would turn c > -0.5 into c > 0
    # and drop rows the omit=0 re-check never sees. Cross-check every range op
    # and a fractional/negative RHS against stdlib sqlite3 on INTEGER 0/1.
    import sqlite3 as pysqlite
    rows = [True, False, True, False, True]
    ref = pysqlite.connect(":memory:")
    ref.execute("CREATE TABLE t(c INTEGER)")
    ref.executemany("INSERT INTO t VALUES (?)", [(int(v),) for v in rows])
    db = _open_memory()
    a = np.array([[v] for v in rows], dtype=np.bool_)
    register_table(db, "t", a, columns=["c"])
    for op in (">", ">=", "<", "<="):
        for rhs in ("-0.5", "0.5", "0", "1"):
            sql = "SELECT c FROM t WHERE c %s %s" % (op, rhs)
            exp = sorted(r[0] for r in ref.execute(sql))
            assert sorted(_select_col0(db, sql)) == exp, sql
    # prove the fractional-RHS range predicate is actually claimed (pushed to
    # xFilter), not just caught by SQLite's omit=0 re-check of surfaced rows
    plan = _fetchall(db, "EXPLAIN QUERY PLAN SELECT c FROM t WHERE c > -0.5")
    text = " ".join(str(field) for row in plan for field in row).upper()
    assert "VIRTUAL TABLE INDEX 1:" in text, text
    ref.close()
    sqlite3_close(db)


def test_pushdown_int64_above_2_53_range():
    db = _open_memory()
    base = 1 << 53
    vals = [base, base + 1, base + 2, base + 3]
    a = np.array([[v] for v in vals], dtype=np.int64)
    h = register_table(db, "t", a, columns=["c"])  # noqa: F841
    threshold = base + 1
    preds = {
        ">": lambda v: v > threshold,
        ">=": lambda v: v >= threshold,
        "<": lambda v: v < threshold,
        "<=": lambda v: v <= threshold,
    }
    for op, pred in preds.items():
        got = sorted(_select_col0(db, "SELECT c FROM t WHERE c %s %d" % (op, threshold)))
        exp = sorted(v for v in vals if pred(v))
        assert got == exp, (op, got, exp)
    # the exact row at 2**53+1 that float64 collapse used to drop under "> base":
    assert sorted(_select_col0(db, "SELECT c FROM t WHERE c > %d" % base)) == sorted(vals[1:])
    sqlite3_close(db)


def test_pushdown_int_column_fractional_rhs_match_stdlib():
    # Int columns constrained by a fractional REAL literal must compare in the
    # float domain: sqlite3_value_int64 would truncate the threshold toward zero
    # (4.5 -> 4, -4.5 -> -4) and drop rows the omit=0 re-check never resurfaces.
    # Cross-check signed int dtypes against stdlib sqlite3 on identical data.
    import sqlite3 as pysqlite
    cases = [
        (np.int8, [3, 4, 5]),
        (np.int8, [-5, -4, -3]),
        (np.int32, [3, 4, 5]),
        (np.int32, [-5, -4, -3]),
        (np.int64, [3, 4, 5]),
        (np.int64, [-5, -4, -3]),
    ]
    for dtype, vals in cases:
        ref = pysqlite.connect(":memory:")
        ref.execute("CREATE TABLE t(c INTEGER)")
        ref.executemany("INSERT INTO t VALUES (?)", [(int(v),) for v in vals])
        db = _open_memory()
        a = np.array([[v] for v in vals], dtype=dtype)
        register_table(db, "t", a, columns=["c"])
        for op in ("<", "<=", ">", ">="):
            for rhs in ("4.5", "-4.5", "0.5", "-0.5"):
                sql = "SELECT c FROM t WHERE c %s %s" % (op, rhs)
                exp = sorted(r[0] for r in ref.execute(sql))
                assert sorted(_select_col0(db, sql)) == exp, (dtype, sql)
        # prove the fractional predicate is actually pushed to xFilter, not just
        # caught by SQLite's omit=0 re-check of surfaced rows
        plan = _fetchall(db, "EXPLAIN QUERY PLAN SELECT c FROM t WHERE c < 4.5")
        text = " ".join(str(field) for row in plan for field in row).upper()
        assert "VIRTUAL TABLE INDEX 1:" in text, text
        ref.close()
        sqlite3_close(db)


def test_pushdown_int64_out_of_range_real_rhs_match_stdlib():
    # An out-of-int64-range REAL literal (~1e23) must compare in the float
    # domain: sqlite3_value_int64 clamps it to INT64_MAX, turning "c < clamp"
    # into "c < INT64_MAX" which wrongly drops the INT64_MAX row.
    import sqlite3 as pysqlite
    INT64_MAX = (1 << 63) - 1
    vals = [-5, 0, 5, INT64_MAX]
    ref = pysqlite.connect(":memory:")
    ref.execute("CREATE TABLE t(c INTEGER)")
    ref.executemany("INSERT INTO t VALUES (?)", [(int(v),) for v in vals])
    db = _open_memory()
    a = np.array([[v] for v in vals], dtype=np.int64)
    register_table(db, "t", a, columns=["c"])
    sql = "SELECT c FROM t WHERE c < 99999999999999999999999"
    exp = sorted(r[0] for r in ref.execute(sql))
    assert sorted(_select_col0(db, sql)) == exp, (sql, exp)
    ref.close()
    sqlite3_close(db)


def test_pushdown_int64_integer_valued_real_rhs_match_stdlib():
    # An int column constrained by an INTEGER-VALUED REAL literal at high
    # magnitude must compare EXACTLY: widening the int64 cell to f64 (the old
    # float-domain path) is lossy above 2**53, so (2**53+1) collapses to 2**53
    # and "c > 2**53.0" wrongly drops the 2**53+1 row. Cross-check against stdlib
    # sqlite3 on identical data, both above +2**53 and below -2**53.
    import sqlite3 as pysqlite
    base = 1 << 53
    cases = [
        ([base, base + 1, base + 2, base + 3, base + 4], ">", "9007199254740992.0"),
        ([base, base + 1, base + 2, base + 3, base + 4], ">=", "9007199254740994.0"),
        ([-base - 4, -base - 3, -base - 2, -base - 1, -base], "<", "-9007199254740992.0"),
        ([-base - 4, -base - 3, -base - 2, -base - 1, -base], "<=", "-9007199254740994.0"),
    ]
    for vals, op, rhs in cases:
        ref = pysqlite.connect(":memory:")
        ref.execute("CREATE TABLE t(c INTEGER)")
        ref.executemany("INSERT INTO t VALUES (?)", [(int(v),) for v in vals])
        db = _open_memory()
        a = np.array([[v] for v in vals], dtype=np.int64)
        register_table(db, "t", a, columns=["c"])
        sql = "SELECT c FROM t WHERE c %s %s" % (op, rhs)
        exp = sorted(r[0] for r in ref.execute(sql))
        assert sorted(_select_col0(db, sql)) == exp, (sql, exp)
        # prove the REAL-RHS predicate is actually pushed to xFilter, not just
        # caught by SQLite's omit=0 re-check of surfaced rows
        plan = _fetchall(db, "EXPLAIN QUERY PLAN " + sql)
        text = " ".join(str(field) for row in plan for field in row).upper()
        assert "VIRTUAL TABLE INDEX 1:" in text, text
        ref.close()
        sqlite3_close(db)


def test_pushdown_uint64_high_magnitude_consistent():
    db = _open_memory()
    vals = [(1 << 63), (1 << 63) + 5, (1 << 63) + 1]
    a = np.array([[v] for v in vals], dtype=np.uint64)
    h = register_table(db, "t", a, columns=["c"])  # noqa: F841
    # xColumn surfaces uint64 as a wrapped int64; the cursor must agree, so a
    # pushdown query returns exactly what a full scan + SQLite re-check returns.
    pushed = sorted(_select_col0(db, "SELECT c FROM t WHERE c > %d" % ((1 << 63) + 0)))
    allrows = sorted(_select_col0(db, "SELECT c FROM t"))
    full = sorted(v for v in allrows if v > ((1 << 63) + 0))
    assert pushed == full, (pushed, full)
    sqlite3_close(db)


def test_xdestroy_deferred_while_statement_open():
    from numbox.core.bindings.sqlite import vtable as v
    from numbox.core.bindings.sqlite.tvf import register_tvf
    from numba import njit
    out = np.dtype([("n", "i8")])

    @njit
    def series(start, stop):
        o = np.empty(stop - start, out)
        for i in range(stop - start):
            o[i].n = start + i
        return o

    db = c_int64(0)
    with c_string(":memory:") as p:
        sqlite3_open(p, addressof(db))
    h = register_tvf(db.value, "series", (np.int64, np.int64), out, series)
    stmt = c_int64(0)
    with c_string("SELECT n FROM series(2, 5)") as p:
        sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    sqlite3_step(stmt.value)  # leave the statement open (not finalized)
    n_before = len(v._DATA_ANCHOR)
    rc = sqlite3_close(db.value)  # sqlite3_close (v1) returns SQLITE_BUSY with an open stmt
    assert rc != 0  # BUSY: close refused, xDestroy NOT fired
    assert len(v._DATA_ANCHOR) == n_before  # anchor entry still present
    sqlite3_finalize(stmt.value)
    assert sqlite3_close(db.value) == 0  # now it closes and fires xDestroy
    assert len(v._DATA_ANCHOR) == n_before - 1
    del h


# ---- columnar (mapping) overload ----

def test_columnar_int64_roundtrip():
    db = _open_memory()
    cols = {"a": np.array([1, 2, 3], dtype=np.int64),
            "b": np.array([10, 20, 30], dtype=np.int64)}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT a, b FROM t ORDER BY a") == [(1, 10), (2, 20), (3, 30)]
    sqlite3_close(db)


def test_columnar_mixed_dtypes():
    db = _open_memory()
    cols = {"i": np.array([1, 2], dtype=np.int32),
            "f": np.array([1.5, 2.5], dtype=np.float64),
            "b": np.array([True, False], dtype=np.bool_)}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT i, f, b FROM t ORDER BY i") == [(1, 1.5, 1), (2, 2.5, 0)]
    sqlite3_close(db)


def test_columnar_pushdown_eq_range():
    db = _open_memory()
    cols = {"a": np.array([1, 2, 3, 4], dtype=np.int64),
            "b": np.array([10, 20, 30, 40], dtype=np.int64)}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT b FROM t WHERE a = 3") == [(30,)]
    assert _fetchall(db, "SELECT b FROM t WHERE a >= 3 ORDER BY a") == [(30,), (40,)]
    sqlite3_close(db)


def test_columnar_unicode_and_bytes():
    db = _open_memory()
    cols = {"u": np.array(["foo", "héllo"], dtype="U6"),
            "s": np.array([b"ab", b"cde"], dtype="S4")}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT u, s FROM t ORDER BY u") == [("foo", "ab"), ("héllo", "cde")]
    sqlite3_close(db)


def test_columnar_text_as_blob():
    db = _open_memory()
    cols = {"s": np.array([b"\xff\xfe", b"\x01\x02"], dtype="S2")}
    register_table(db, "t", cols, text_as_blob=True)
    assert _fetchall(db, "SELECT s FROM t") == [(b"\xff\xfe",), (b"\x01\x02",)]
    sqlite3_close(db)


def test_columnar_uint64_wraps_signed():
    db = _open_memory()
    cols = {"x": np.array([2 ** 63], dtype=np.uint64)}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT x FROM t") == [(-(2 ** 63),)]
    sqlite3_close(db)


def test_columnar_nan_reads_as_null():
    db = _open_memory()
    cols = {"x": np.array([1.0, np.nan, 3.0], dtype=np.float64)}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT x FROM t") == [(1.0,), (None,), (3.0,)]
    sqlite3_close(db)


def test_columnar_join_two_tables():
    db = _open_memory()
    left = {"id": np.array([1, 2, 3], dtype=np.int64),
            "v": np.array([10, 20, 30], dtype=np.int64)}
    right = {"id": np.array([2, 3, 4], dtype=np.int64),
             "w": np.array([200, 300, 400], dtype=np.int64)}
    register_table(db, "lhs", left)
    register_table(db, "rhs", right)
    got = _fetchall(db, "SELECT lhs.id, v, w FROM lhs JOIN rhs ON lhs.id = rhs.id ORDER BY lhs.id")
    assert got == [(2, 20, 200), (3, 30, 300)]
    sqlite3_close(db)


def test_columnar_strided_column():
    db = _open_memory()
    backing = np.array([1, 99, 2, 99, 3, 99], dtype=np.int64)
    strided = backing[::2]            # values 1,2,3; stride 16 bytes; non-contiguous
    assert not strided.flags["C_CONTIGUOUS"]
    cols = {"a": strided, "b": np.array([10, 20, 30], dtype=np.int64)}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT a, b FROM t ORDER BY a") == [(1, 10), (2, 20), (3, 30)]
    sqlite3_close(db)


def test_columnar_strided_string_column():
    db = _open_memory()
    backing = np.array(["aa", "zz", "bb", "zz", "cc", "zz"], dtype="U6")
    strided = backing[::2]            # "aa","bb","cc"; non-contiguous (stride 2*itemsize)
    assert not strided.flags["C_CONTIGUOUS"]
    cols = {"s": strided, "n": np.array([1, 2, 3], dtype=np.int64)}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT s, n FROM t ORDER BY s") == [("aa", 1), ("bb", 2), ("cc", 3)]
    sqlite3_close(db)


def test_columnar_negative_stride_column():
    backing = np.array([10, 20, 30, 40], dtype=np.int64)
    rev = backing[::-1]            # negative stride; rev.ctypes.data is the physical-last element
    assert rev.strides[0] < 0
    db = _open_memory()
    register_table(db, "t", {"a": rev})
    assert _fetchall(db, "SELECT a FROM t") == [(40,), (30,), (20,), (10,)]
    assert _fetchall(db, "SELECT a FROM t WHERE a > 15 ORDER BY a") == [(20,), (30,), (40,)]
    sqlite3_close(db)


def test_columnar_rejects_non_string_key():
    with pytest.raises(TypeError):
        register_table(0, "t", {1: np.array([1, 2], dtype=np.int64)})


def test_columnar_rejects_columns_kwarg():
    with pytest.raises(ValueError):
        register_table(0, "t", {"a": np.array([1], dtype=np.int64)}, columns=["a"])


def test_columnar_rejects_non_array_value():
    with pytest.raises(TypeError):
        register_table(0, "t", {"a": [1, 2, 3]})


def test_columnar_rejects_non_1d_value():
    with pytest.raises(ValueError):
        register_table(0, "t", {"a": np.zeros((2, 2), dtype=np.int64)})


def test_columnar_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        register_table(0, "t", {"a": np.array([1, 2], dtype=np.int64),
                                "b": np.array([1], dtype=np.int64)})


def test_columnar_rejects_empty_mapping():
    with pytest.raises(ValueError):
        register_table(0, "t", {})


def test_columnar_rejects_object_dtype():
    with pytest.raises(TypeError):
        register_table(0, "t", {"a": np.array(["x", "y"], dtype=object)})


def test_columnar_zero_rows():
    db = _open_memory()
    cols = {"a": np.array([], dtype=np.int64),
            "b": np.array([], dtype=np.float64)}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT a, b FROM t") == []
    sqlite3_close(db)


def test_columnar_on_stdlib_connection():
    if not libraries_coordinated():
        pytest.skip("uncoordinated libsqlite3 (see numbox.utils.pysqlite_bridge)")
    conn = sqlite3.connect(":memory:")
    try:
        cols = {"id": np.array([1, 2, 3], dtype=np.int64),
                "px": np.array([100.0, 250.0, 50.0], dtype=np.float64)}
        register_table(extract_connection_ptr(conn), "trades", cols)
        rows = conn.execute("SELECT id FROM trades WHERE px > 100 ORDER BY id").fetchall()
        assert rows == [(2,)]
    finally:
        conn.close()


def test_columnar_query_to_array():
    db = _open_memory()
    cols = {"id": np.array([1, 2, 3], dtype=np.int64),
            "px": np.array([100.0, 250.0, 50.0], dtype=np.float64)}
    register_table(db, "trades", cols)
    out_dtype = np.dtype([("id", "i8"), ("px", "f8")])
    with c_string("SELECT id, px FROM trades WHERE px > 100 ORDER BY id") as sql_p:
        out = query_to_array(db, sql_p, out_dtype)
    assert out.dtype == out_dtype
    assert out["id"].tolist() == [2]
    assert out["px"].tolist() == [250.0]
    sqlite3_close(db)
