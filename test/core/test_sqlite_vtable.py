import ctypes
import gc
import os
import subprocess
import sys
import textwrap
from ctypes import addressof, c_int64, cast, c_char_p, string_at

import pytest
import numpy as np
from numbox.utils.cstrings import c_string
from numbox.core.bindings import (
    sqlite3_open, sqlite3_close, register_table,
    sqlite3_prepare_v2, sqlite3_step, sqlite3_finalize,
    sqlite3_column_count, sqlite3_column_type,
    sqlite3_column_int64, sqlite3_column_double,
    sqlite3_column_text, sqlite3_column_blob, sqlite3_column_bytes,
)
from numbox.core.bindings._sqlite_vtable import utf32_to_utf8, _nul_trimmed_len
from numbox.core.bindings._sqlite_vtable import _build_descriptor, _TAG_I64, _TAG_F64, _TAG_U
from numbox.utils.lowlevel import array_data_p


def test_imports():
    from numbox.core.bindings import _sqlite_vtable as v
    assert hasattr(v.sqlite3_create_module, "py_func")
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
    assert (d.nrows, d.ncols, d.row_stride) == (3, 2, a.strides[0])
    assert list(d.offsets) == [0, 8]
    assert list(d.tags) == [_TAG_I64, _TAG_I64]
    assert d.schema == b'CREATE TABLE x("a" INTEGER, "b" INTEGER)\x00'


def test_descriptor_structured_mixed():
    dt = np.dtype([("t", "U6"), ("q", "i8"), ("p", "f8")])
    a = np.zeros(2, dtype=dt)
    d = _build_descriptor(a, None, False)
    assert d.ncols == 3
    assert list(d.tags) == [_TAG_U, _TAG_I64, _TAG_F64]
    assert list(d.offsets) == [dt.fields["t"][1], dt.fields["q"][1], dt.fields["p"][1]]
    assert d.scratch_bytes == 6 * 4 + 1
    assert d.schema == b'CREATE TABLE x("t" TEXT, "q" INTEGER, "p" REAL)\x00'


def test_descriptor_rejects_bad_shapes():
    import pytest
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


def test_unicode_width_overflow_rejected():
    # itemsize 4*536870906 = 2147483624; + 1 (scratch) + 24 (_CUR_SCRATCH) overflows int32
    dt = np.dtype([("u", "U536870906")])
    a = np.zeros(0, dtype=dt)
    with pytest.raises(ValueError):
        _build_descriptor(a, None, False)


def test_descriptor_offsets_assertion_holds():
    from numbox.core.bindings._sqlite_vtable import _NdarrayTableDescriptor
    assert ctypes.sizeof(_NdarrayTableDescriptor) == 72


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


def test_structured_text_and_unicode():
    db = _open_memory()
    dt = np.dtype([("t", "U6"), ("q", "i4"), ("p", "f8"), ("s", "S4")])
    a = np.array([("héllo", 3, 1.5, b"ab"), ("\U0001F600", 7, 2.0, b"cd")], dtype=dt)
    h = register_table(db, "trades", a)  # noqa: F841
    got = _fetchall(db, "SELECT t, q, p, s FROM trades")
    assert got == [("héllo", 3, 1.5, "ab"), ("\U0001F600", 7, 2.0, "cd")]
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


def test_handle_survives_gc():
    db = _open_memory()
    a = np.array([[5, 6]], dtype=np.int64)
    h = register_table(db, "t", a, columns=["a", "b"])  # noqa: F841
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
    from numbox.core.bindings import (
        sqlite3_open, sqlite3_prepare_v2, sqlite3_step,
        sqlite3_column_int64, sqlite3_finalize)
    from numbox.core.bindings._sqlite_vtable import register_table
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
    while sqlite3_step(stmt) == 100:
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
    return sum(1 for _ in cache_dir.rglob("*_sqlite_vtable*.nbc"))


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
