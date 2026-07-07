from ctypes import addressof, c_int64

import pytest
import numpy as np
from numbox.core.bindings.sqlite._typemap import utf8_to_utf32, utf32_to_utf8
from numbox.core.bindings.sqlite.conn import sqlite3_open, sqlite3_close
from numbox.core.bindings.sqlite.query import query_to_array
from numbox.core.bindings.sqlite.exec import sqlite3_exec
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import array_data_p


def _decode(s_bytes, width_cp):
    src = np.frombuffer(s_bytes, dtype=np.uint8).copy()
    dst = np.zeros(width_cp, dtype=np.uint32)
    n = utf8_to_utf32(array_data_p(src), len(src), array_data_p(dst), width_cp)
    return n, dst


def test_utf8_to_utf32_ascii():
    n, dst = _decode(b"abc", 6)
    assert n == 3 and list(dst[:3]) == [97, 98, 99] and dst[3] == 0


def test_utf8_to_utf32_multibyte():
    n, dst = _decode("héllo".encode("utf-8"), 6)
    assert list(dst[:n]) == [ord(c) for c in "héllo"]


def test_utf8_to_utf32_4byte():
    n, dst = _decode("a\U0001F600b".encode("utf-8"), 6)
    assert [int(x) for x in dst[:n]] == [ord("a"), 0x1F600, ord("b")]


def test_utf8_to_utf32_clamps_to_width():
    n, dst = _decode(b"abcdefgh", 3)
    assert n == 3 and list(dst[:3]) == [97, 98, 99]


def test_utf8_to_utf32_roundtrip():
    s = "abé\U0001F600cd"
    width_cp = 8
    cps = np.array([ord(c) for c in s] + [0] * (width_cp - len(s)), dtype=np.uint32)
    enc = np.zeros(4 * width_cp + 1, dtype=np.uint8)
    nbytes = utf32_to_utf8(array_data_p(cps), width_cp, array_data_p(enc))
    n, dst = _decode(enc[:nbytes].tobytes(), width_cp)
    assert n == len(s)
    assert [int(x) for x in dst[:n]] == [ord(c) for c in s]


def _open_mem():
    db = c_int64(0)
    with c_string(":memory:") as p:
        assert sqlite3_open(p, addressof(db)) == 0
    return db.value


def _exec(db, sql):
    with c_string(sql) as p:
        assert sqlite3_exec(db, p, 0, 0, 0) == 0


def test_query_numeric_roundtrip():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(i INTEGER, x REAL)")
    _exec(db, "INSERT INTO t VALUES (1, 1.5), (2, 2.5), (3, 3.5)")
    dt = np.dtype([("i", "i8"), ("x", "f8")])
    with c_string("SELECT i, x FROM t ORDER BY i") as sql:
        out = query_to_array(db, sql, dt)
    assert out.shape == (3,)
    assert list(out["i"]) == [1, 2, 3]
    assert list(out["x"]) == [1.5, 2.5, 3.5]
    sqlite3_close(db)


def test_query_null_coercion():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(i INTEGER, x REAL)")
    _exec(db, "INSERT INTO t VALUES (NULL, NULL)")
    dt = np.dtype([("i", "i8"), ("x", "f8")])
    with c_string("SELECT i, x FROM t") as sql:
        out = query_to_array(db, sql, dt)
    assert out["i"][0] == 0 and np.isnan(out["x"][0])
    sqlite3_close(db)


def test_query_growth_boundary():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(i INTEGER)")
    _exec(db, "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n<5000) INSERT INTO t SELECT n FROM c")
    dt = np.dtype([("i", "i8")])
    with c_string("SELECT i FROM t ORDER BY i") as sql:
        out = query_to_array(db, sql, dt)
    assert out.shape == (5000,) and out["i"][0] == 1 and out["i"][4999] == 5000
    sqlite3_close(db)


def test_query_text_roundtrip():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(s TEXT)")
    _exec(db, "INSERT INTO t VALUES ('hi'), ('héllo')")
    dt = np.dtype([("s", "U8")])
    with c_string("SELECT s FROM t") as sql:
        out = query_to_array(db, sql, dt)
    assert out["s"][0] == "hi" and out["s"][1] == "héllo"
    sqlite3_close(db)


def test_query_field_count_mismatch_raises():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(a INTEGER, b INTEGER)")
    with c_string("SELECT a, b FROM t") as sql:
        with pytest.raises(ValueError):
            query_to_array(db, sql, np.dtype([("a", "i8")]))
    sqlite3_close(db)


def test_query_bad_sql_raises():
    db = _open_mem()
    with c_string("SELECT FROM") as sql:
        with pytest.raises(RuntimeError) as excinfo:
            query_to_array(db, sql, np.dtype([("a", "i8")]))
    assert "query_to_array failed" in str(excinfo.value)
    assert str(excinfo.value).strip()
    sqlite3_close(db)


def test_query_two_dtypes_no_stale_cache():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(a INTEGER, b INTEGER)")
    _exec(db, "INSERT INTO t VALUES (1, 2)")
    with c_string("SELECT a, b FROM t") as sql:
        o1 = query_to_array(db, sql, np.dtype([("a", "i8"), ("b", "i8")]))
        o2 = query_to_array(db, sql, np.dtype([("a", "i4"), ("b", "i4")]))
    assert o1.dtype.itemsize == 16 and o2.dtype.itemsize == 8
    assert tuple(o2[0]) == (1, 2)
    sqlite3_close(db)


def test_query_empty_result():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(i INTEGER)")
    with c_string("SELECT i FROM t") as sql:
        out = query_to_array(db, sql, np.dtype([("i", "i8")]))
    assert out.shape == (0,)
    sqlite3_close(db)


def test_query_step_error_raises():
    # A step-time error after >=1 row must raise, not return a truncated array.
    # CASE returns plain ints for the first rows, then abs(INT64_MIN) overflows
    # at the last step (verified against the venv sqlite: prepare succeeds, rows
    # emit, then sqlite3_step returns SQLITE_ERROR "integer overflow").
    db = _open_mem()
    _exec(db, "CREATE TABLE t(i INTEGER)")
    _exec(db, "INSERT INTO t VALUES (1), (2), (3)")
    sql_text = "SELECT CASE i WHEN 3 THEN abs(-9223372036854775808) ELSE i END FROM t"
    with c_string(sql_text) as sql:
        with pytest.raises(RuntimeError) as excinfo:
            query_to_array(db, sql, np.dtype([("i", "i8")]))
    assert "query_to_array failed" in str(excinfo.value)
    sqlite3_close(db)


def test_query_packed_misaligned_dtype():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(flag INTEGER, x REAL, s TEXT)")
    _exec(db, "INSERT INTO t VALUES (1, 2.5, 'ok'), (0, -3.25, 'hi')")
    dt = np.dtype([("flag", "i1"), ("x", "f8"), ("s", "U4")])  # x at offset 1, s at offset 9
    assert not dt.isalignedstruct
    with c_string("SELECT flag, x, s FROM t") as sql:
        out = query_to_array(db, sql, dt)
    assert list(out["flag"]) == [1, 0]
    assert list(out["x"]) == [2.5, -3.25]
    assert list(out["s"]) == ["ok", "hi"]
    sqlite3_close(db)


def test_query_blob_into_S_field():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(b BLOB)")
    _exec(db, "INSERT INTO t VALUES (x'00ff10'), (x'4142')")
    with c_string("SELECT b FROM t") as sql:
        out = query_to_array(db, sql, np.dtype([("b", "S3")]))
    # interior bytes preserved up to field width; trailing NUL pad trimmed on read
    assert out["b"][0] == b"\x00\xff\x10"
    assert out["b"][1] == b"AB"
    sqlite3_close(db)


def test_utf8_to_utf32_bad_continuation_is_replacement():
    # 0xE0 starts a 3-byte sequence but the next byte 0x20 is not a continuation
    n, dst = _decode(b"\xe0\x20\x41", 4)
    assert int(dst[0]) == 0xFFFD


def test_utf8_to_utf32_surrogate_is_replacement():
    # CESU-8-style encoding of a high surrogate U+D800 (0xED 0xA0 0x80) is illegal
    n, dst = _decode(b"\xed\xa0\x80", 4)
    assert int(dst[0]) == 0xFFFD


def test_utf8_to_utf32_overlong_is_replacement():
    # overlong 2-byte encoding of '/' (0xC0 0xAF) is illegal
    n, dst = _decode(b"\xc0\xaf", 4)
    assert int(dst[0]) == 0xFFFD


def test_query_xprocess_cache(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap
    driver = textwrap.dedent('''
        from ctypes import addressof, c_int64
        import numpy as np
        from numba import njit
        from numbox.core.bindings.sqlite.conn import sqlite3_open, sqlite3_close
        from numbox.core.bindings.sqlite.exec import sqlite3_exec
        from numbox.core.bindings.sqlite.query import query_to_array
        from numbox.core.bindings.sqlite.stmt import sqlite3_prepare_v2, sqlite3_step, sqlite3_finalize
        from numbox.core.bindings.sqlite.column import sqlite3_column_int64
        from numbox.core.bindings.sqlite.tvf import register_tvf
        from numbox.utils.cstrings import c_string

        OUT = np.dtype([("n", "i8")])

        @njit
        def series(start, stop):
            o = np.empty(stop - start, OUT)
            for i in range(stop - start):
                o[i].n = start + i
            return o

        db_p = c_int64(0)
        with c_string(":memory:") as nm:
            sqlite3_open(nm, addressof(db_p))
        db = db_p.value
        with c_string("CREATE TABLE t(i INTEGER, x REAL)") as p:
            sqlite3_exec(db, p, 0, 0, 0)
        with c_string("INSERT INTO t VALUES (1, 1.5), (2, 2.5)") as p:
            sqlite3_exec(db, p, 0, 0, 0)
        dt = np.dtype([("i", "i8"), ("x", "f8")])
        with c_string("SELECT i, x FROM t ORDER BY i") as s:
            out = query_to_array(db, s, dt)
        q = [int(val) for val in out["i"]]

        h = register_tvf(db, "series", (np.int64, np.int64), OUT, series)
        stmt_p = c_int64(0)
        with c_string("SELECT n FROM series(2, 5)") as sp:
            sqlite3_prepare_v2(db, sp, -1, addressof(stmt_p), 0)
        tvf = []
        while sqlite3_step(stmt_p.value) == 100:
            tvf.append(sqlite3_column_int64(stmt_p.value, 0))
        sqlite3_finalize(stmt_p.value)
        sqlite3_close(db)
        assert q == [1, 2], q
        assert tvf == [2, 3, 4], tvf
        print("RESULT", q, tvf)
    ''')
    script = tmp_path / "query_drv.py"
    script.write_text(driver)
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "nbcache"))
    for _ in range(2):  # cold then warm: cross-process cache reuse must not crash
        out = subprocess.run([sys.executable, str(script)], env=env,
                             capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, out.stderr
        assert "RESULT [1, 2] [2, 3, 4]" in out.stdout


def test_query_empty_text_and_blob():
    # Empty TEXT '' and zero-length BLOB x'' -> SQLite returns a NULL/zero-length
    # pointer with nbytes==0, so _store_cell builds carray(ptr, (0,)) and copies 0
    # bytes. This exercises that path and must yield empty cells without crashing.
    db = _open_mem()
    _exec(db, "CREATE TABLE t(s TEXT, b BLOB, u TEXT)")
    _exec(db, "INSERT INTO t VALUES ('', x'', '')")
    dt = np.dtype([("s", "S4"), ("b", "S4"), ("u", "U4")])
    with c_string("SELECT s, b, u FROM t") as sql:
        out = query_to_array(db, sql, dt)
    assert out.shape == (1,)
    assert out["s"][0] == b"" and out["b"][0] == b"" and out["u"][0] == ""
    sqlite3_close(db)
