"""Tests of the aggregate and window registrations of :mod:`~numbox.core.bindings.sqlite.udf_helpers`.

The tests check the results of ``register_aggregate`` and ``register_window`` for groups of rows, for an empty group,
for plain Python callbacks and for a state with two fields, and the flag that ``deterministic=True`` sets. An exception
in a callback must fail the query: the tests raise one in each callback, read the error message, and count MemInfo
allocations and frees. A registration that SQLite refuses raises ``RuntimeError``. Tests in subprocesses check that a
second process loads the generated code from numba's cache, that a changed literal compiles new code, and that numbox's
jit options can disable the cache.
"""
import gc
from ctypes import addressof, c_char_p, c_int64

import numpy as np
from numba import carray, cfunc, njit, types
from numba.core import types as nb_types
from numba.experimental import structref

from numbox.core.bindings.sqlite.constants import SQLITE_OK, SQLITE_UTF8
from numbox.core.bindings.sqlite.udf_helpers import register_aggregate, register_window
from numbox.core.bindings.sqlite.conn import sqlite3_close, sqlite3_errmsg, sqlite3_open
from numbox.core.bindings.sqlite.udf import sqlite3_create_function_v2, sqlite3_user_data
from numbox.core.bindings.sqlite.exec import sqlite3_exec
from numbox.core.bindings.sqlite.result import sqlite3_result_int, sqlite3_result_int64
from numbox.core.bindings.sqlite.value import sqlite3_value_int64
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import _cast_int_to_void_p


# --- The state type of the aggregates below, a structref with one int64 field, total ---
@structref.register
class SumStateType(nb_types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, nb_types.unliteral(t)) for n, t in fields)


class SumState(structref.StructRefProxy):
    def __new__(cls, total):
        return structref.StructRefProxy.__new__(cls, total)


structref.define_proxy(SumState, SumStateType, ["total"])
sum_state_type = SumStateType([("total", nb_types.int64)])


@njit
def sum_init():
    return SumState(np.int64(0))


@njit
def sum_step(state, ctx, argc, argv_pp):
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    state.total += sqlite3_value_int64(args[0])


@njit
def sum_finalize(state, ctx):
    sqlite3_result_int64(ctx, state.total)


# --- Helpers of the tests ---
def _open_memory():
    db_p = c_int64(0)
    with c_string(":memory:") as name_p:
        assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    return db_p.value


def _make_table(db, values):
    with c_string("CREATE TABLE t(v INTEGER)") as p:
        assert sqlite3_exec(db, p, 0, 0, 0) == SQLITE_OK
    for v in values:
        with c_string("INSERT INTO t VALUES (%d)" % v) as p:
            assert sqlite3_exec(db, p, 0, 0, 0) == SQLITE_OK


def _errmsg(db):
    """Return the message of the most recent error on ``db`` as a ``str``, or ``""`` if SQLite returns no message."""
    p = sqlite3_errmsg(db)
    return c_char_p(p).value.decode("utf-8", "replace") if p else ""


def _read1_int64(db, select_sql):
    """Run ``select_sql``, which passes one value to ``__cap``, and return the value and the cfunc of ``__cap``.

    ``__cap`` writes its argument into a NumPy array at its user data. The caller keeps the cfunc referenced while
    ``__cap`` is registered, as a precaution.
    """
    buf = np.zeros(1, dtype=np.int64)

    @cfunc(types.void(types.intp, types.int32, types.intp))
    def cap_cb(ctx, argc, argv):
        ud = sqlite3_user_data(ctx)
        args = carray(_cast_int_to_void_p(argv), (argc,), dtype=np.intp)
        o = carray(_cast_int_to_void_p(ud), (1,), dtype=np.int64)
        o[0] = sqlite3_value_int64(args[0])
        sqlite3_result_int(ctx, 0)

    with c_string("__cap") as cp:
        assert sqlite3_create_function_v2(
            db, cp, 1, SQLITE_UTF8, buf.ctypes.data, cap_cb.address, 0, 0, 0) == SQLITE_OK
    with c_string(select_sql) as sp:
        assert sqlite3_exec(db, sp, 0, 0, 0) == SQLITE_OK
    return int(buf[0]), cap_cb


def test_aggregate_sum():
    """The aggregate sums 1 to 5 after a garbage collection, although ``register_aggregate`` returns nothing to keep."""
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    register_aggregate(db, "my_sum", 1, sum_state_type, sum_init, sum_step, sum_finalize)
    gc.collect()  # The query below runs after a collection, because numba keeps the callbacks' compiled code.
    val, _cap = _read1_int64(db, "SELECT __cap(my_sum(v)) FROM t")
    sqlite3_close(db)
    assert val == 15


def test_aggregate_empty_group():
    """An aggregate over an empty table returns 0, the result of finalize on a new state from init."""
    db = _open_memory()
    _make_table(db, [])
    register_aggregate(db, "my_sum", 1, sum_state_type, sum_init, sum_step, sum_finalize)
    val, _cap = _read1_int64(db, "SELECT __cap(my_sum(v)) FROM t")
    sqlite3_close(db)
    assert val == 0


def test_aggregate_bad_state_type():
    """``register_aggregate`` raises ``TypeError`` for a ``state_type`` that is not a ``StructRef`` instance type."""
    import pytest
    db = _open_memory()
    with pytest.raises(TypeError):
        register_aggregate(db, "bad", 1, object(), sum_init, sum_step, sum_finalize)
    sqlite3_close(db)


import os                                            # noqa: E402
import subprocess                                    # noqa: E402
import sys                                           # noqa: E402
import textwrap                                      # noqa: E402


# The driver script defines a state type and a sum aggregate in its __main__ and registers the aggregate. It runs
# SELECT cap(f(v)) over the values 1 to 5 and prints the result. {MULT} is the factor of the step function: 1 for the
# sum, and 3 in the test that changes the literal and expects new code.
_DRIVER = textwrap.dedent('''
    from ctypes import addressof, c_int64
    import numpy as np
    from numba import carray, cfunc, njit, types
    from numba.core import types as nb_types
    from numba.experimental import structref
    from numbox.core.bindings.sqlite.constants import SQLITE_OK, SQLITE_UTF8
    from numbox.core.bindings.sqlite.conn import sqlite3_close, sqlite3_open
    from numbox.core.bindings.sqlite.udf import sqlite3_create_function_v2, sqlite3_user_data
    from numbox.core.bindings.sqlite.exec import sqlite3_exec
    from numbox.core.bindings.sqlite.result import sqlite3_result_int, sqlite3_result_int64
    from numbox.core.bindings.sqlite.value import sqlite3_value_int64
    from numbox.core.bindings.sqlite.udf_helpers import register_aggregate
    from numbox.utils.cstrings import c_string
    from numbox.utils.lowlevel import _cast_int_to_void_p

    @structref.register
    class StT(nb_types.StructRef):
        def preprocess_fields(self, fields):
            return tuple((n, nb_types.unliteral(t)) for n, t in fields)
    class St(structref.StructRefProxy):
        def __new__(cls, total):
            return structref.StructRefProxy.__new__(cls, total)
    structref.define_proxy(St, StT, ["total"])
    st = StT([("total", nb_types.int64)])

    @njit
    def s_init():
        return St(np.int64(0))
    @njit
    def s_step(state, ctx, argc, argv_pp):
        a = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
        state.total += {MULT} * sqlite3_value_int64(a[0])
    @njit
    def s_fin(state, ctx):
        sqlite3_result_int64(ctx, state.total)

    db_p = c_int64(0)
    with c_string(":memory:") as n:
        sqlite3_open(n, addressof(db_p))
    db = db_p.value
    with c_string("CREATE TABLE t(v INTEGER)") as p:
        sqlite3_exec(db, p, 0, 0, 0)
    for v in (1, 2, 3, 4, 5):
        with c_string("INSERT INTO t VALUES (%d)" % v) as p:
            sqlite3_exec(db, p, 0, 0, 0)
    register_aggregate(db, "f", 1, st, s_init, s_step, s_fin)
    buf = np.zeros(1, dtype=np.int64)
    @cfunc(types.void(types.intp, types.int32, types.intp))
    def cap(ctx, argc, argv):
        ud = sqlite3_user_data(ctx)
        a = carray(_cast_int_to_void_p(argv), (argc,), dtype=np.intp)
        o = carray(_cast_int_to_void_p(ud), (1,), dtype=np.int64)
        o[0] = sqlite3_value_int64(a[0]); sqlite3_result_int(ctx, 0)
    with c_string("cap") as cp:
        sqlite3_create_function_v2(db, cp, 1, SQLITE_UTF8, buf.ctypes.data, cap.address, 0, 0, 0)
    with c_string("SELECT cap(f(v)) FROM t") as sp:
        sqlite3_exec(db, sp, 0, 0, 0)
    print("RESULT", int(buf[0]))
''')


def _run_driver(tmp_path, cache_dir, mult):
    """Run the driver with ``mult`` in a new process with ``cache_dir`` as NUMBA_CACHE_DIR, and return its sum."""
    script = tmp_path / ("drv_%d.py" % mult)
    script.write_text(_DRIVER.replace("{MULT}", str(mult)))
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache_dir))
    out = subprocess.run([sys.executable, str(script)], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")][0]
    return int(line.split()[1])


def _count_nbc(cache_dir):
    """Return the number of numba data files under ``cache_dir`` for the generated aggregate and window functions."""
    # The anchors of the generated functions have names that start with udaf_ or wudaf_. Counting only their files keeps
    # the test independent of the cache files of other modules.
    return sum(1 for _ in cache_dir.rglob("*udaf*.nbc"))


def test_xprocess_cache_no_growth(tmp_path):
    """A second run of the driver loads the generated functions from numba's cache and adds no data file to it."""
    cache = tmp_path / "nbcache"
    cache.mkdir()
    assert _run_driver(tmp_path, cache, 1) == 15      # The first run compiles the generated functions and caches them.
    n_cold = _count_nbc(cache)
    assert n_cold > 0
    assert _run_driver(tmp_path, cache, 1) == 15      # The second run loads them from the cache.
    assert _count_nbc(cache) == n_cold, "warm run grew the cache (C failure mode)"


def test_invalidation_on_literal_edit(tmp_path):
    """A change of a numeric literal in the step function compiles new code instead of loading the cached code."""
    cache = tmp_path / "nbcache"
    cache.mkdir()
    assert _run_driver(tmp_path, cache, 1) == 15       # The step adds 1 * v, and the sum of 1 to 5 is 15.
    assert _run_driver(tmp_path, cache, 3) == 45       # New code returns 3 * 15 = 45; stale code would return 15.


def test_jit_options_cache_disabled_honored(tmp_path):
    """The generated functions compile under numbox's jit options.

    With ``NUMBOX_JIT_OPTIONS='{"cache": false}'``, the aggregate returns the right sum, and numba writes no data file
    for the generated functions.
    """
    cache = tmp_path / "nbcache_nocache"
    cache.mkdir()
    script = tmp_path / "drv_nocache.py"
    script.write_text(_DRIVER.replace("{MULT}", "1"))
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache),
               NUMBOX_JIT_OPTIONS='{"cache": false}')
    out = subprocess.run([sys.executable, str(script)], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    result = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")][0]
    assert int(result.split()[1]) == 15
    assert _count_nbc(cache) == 0, "cache disabled but a udaf impl .nbc was written"


@njit
def w_inverse(state, ctx, argc, argv_pp):
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    state.total -= sqlite3_value_int64(args[0])


@njit
def w_value(state, ctx):
    sqlite3_result_int64(ctx, state.total)


def _read_window(db, select_sql, nrows):
    """Run ``select_sql``, which passes each row's value to ``__wcap``, and return the values and the cfunc."""
    meta = np.zeros(nrows + 1, dtype=np.int64)  # meta[0] counts the values, and meta[1:] holds them.

    @cfunc(types.void(types.intp, types.int32, types.intp))
    def wcap_cb(ctx, argc, argv):
        ud = sqlite3_user_data(ctx)
        m = carray(_cast_int_to_void_p(ud), (nrows + 1,), dtype=np.int64)
        i = m[0]
        args = carray(_cast_int_to_void_p(argv), (argc,), dtype=np.intp)
        m[1 + i] = sqlite3_value_int64(args[0])
        m[0] = i + 1
        sqlite3_result_int(ctx, 0)

    with c_string("__wcap") as cp:
        assert sqlite3_create_function_v2(
            db, cp, 1, SQLITE_UTF8, meta.ctypes.data, wcap_cb.address, 0, 0, 0) == SQLITE_OK
    with c_string(select_sql) as sp:
        assert sqlite3_exec(db, sp, 0, 0, 0) == SQLITE_OK
    return [int(meta[1 + i]) for i in range(int(meta[0]))], wcap_cb


def test_window_running_sum():
    """A window sum over the current and the preceding row returns 1, 3, 5, 7 and 9 for the values 1 to 5."""
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    register_window(db, "my_wsum", 1, sum_state_type, sum_init, sum_step, w_inverse, w_value, sum_finalize)
    sql = ("SELECT __wcap(my_wsum(v) OVER "
           "(ORDER BY v ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)) "
           "FROM t ORDER BY v")
    vals, _cap = _read_window(db, sql, 5)
    sqlite3_close(db)
    assert vals == [1, 3, 5, 7, 9]


@njit
def sum2_step(state, ctx, argc, argv_pp):  # A body other than sum_step's gives the registration a digest of its own.
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    state.total += 2 * sqlite3_value_int64(args[0])


def test_two_distinct_aggregates_no_collision():
    """Two aggregates that differ only in their step function return their own results, 15 and 30."""
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    register_aggregate(db, "my_sum", 1, sum_state_type, sum_init, sum_step, sum_finalize)
    register_aggregate(db, "my_sum2", 1, sum_state_type, sum_init, sum2_step, sum_finalize)
    v1, _c1 = _read1_int64(db, "SELECT __cap(my_sum(v)) FROM t")
    v2, _c2 = _read1_int64(db, "SELECT __cap(my_sum2(v)) FROM t")
    sqlite3_close(db)
    assert (v1, v2) == (15, 30)


def test_deterministic_flag():
    """An aggregate registered with ``deterministic=True`` returns the right sum."""
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    register_aggregate(db, "my_sum_det", 1, sum_state_type, sum_init, sum_step, sum_finalize, deterministic=True)
    val, _cap = _read1_int64(db, "SELECT __cap(my_sum_det(v)) FROM t")
    sqlite3_close(db)
    assert val == 15


def test_deterministic_flag_ors_bit(monkeypatch):
    """``deterministic=True`` adds ``SQLITE_DETERMINISTIC`` to the flags of the registration, and the default does not.

    The flag affects only how SQLite's query planner treats the function, not an aggregate's value, so a wrapper of
    ``sqlite3_create_function_v2`` records the flags that ``register_aggregate`` passes.
    """
    import numbox.core.bindings.sqlite.udf_helpers as helpers
    real = helpers.sqlite3_create_function_v2
    seen = []

    def spy(db, name_p, n_arg, flags, *rest):
        seen.append(flags)
        return real(db, name_p, n_arg, flags, *rest)

    monkeypatch.setattr(helpers, "sqlite3_create_function_v2", spy)
    db = _open_memory()
    register_aggregate(db, "det_on", 1, sum_state_type, sum_init, sum_step, sum_finalize, deterministic=True)
    register_aggregate(db, "det_off", 1, sum_state_type, sum_init, sum_step, sum_finalize)
    sqlite3_close(db)
    assert seen[0] & helpers.SQLITE_DETERMINISTIC
    assert not (seen[1] & helpers.SQLITE_DETERMINISTIC)


def test_no_meminfo_leak():
    """The generated callbacks release every state: ten runs after the first allocate as many MemInfos as they free."""
    from numba.core.runtime import nrt
    _nrt = nrt._nrt
    if not hasattr(_nrt, "memsys_enable_stats"):
        import pytest
        pytest.skip("NRT allocation stats unavailable")
    _nrt.memsys_enable_stats()
    test_aggregate_sum()          # A first run comes before the measurement, so one-time allocations do not count.
    test_window_running_sum()
    before = nrt.rtsys.get_allocation_stats()
    for _ in range(10):
        test_aggregate_sum()
        test_window_running_sum()
    after = nrt.rtsys.get_allocation_stats()
    allocated = after.mi_alloc - before.mi_alloc
    freed = after.mi_free - before.mi_free
    assert allocated == freed, "meminfo imbalance: %d alloc, %d free" % (allocated, freed)


def test_window_deterministic_flag_ors_bit(monkeypatch):
    """With ``deterministic=True`` only, ``register_window`` adds ``SQLITE_DETERMINISTIC`` to the flags it passes."""
    import numbox.core.bindings.sqlite.udf_helpers as helpers
    real = helpers.sqlite3_create_window_function
    seen = []

    def spy(db, name_p, n_arg, flags, *rest):
        seen.append(flags)
        return real(db, name_p, n_arg, flags, *rest)

    monkeypatch.setattr(helpers, "sqlite3_create_window_function", spy)
    db = _open_memory()
    register_window(db, "wdet_on", 1, sum_state_type, sum_init, sum_step, w_inverse, w_value, sum_finalize, deterministic=True)
    register_window(db, "wdet_off", 1, sum_state_type, sum_init, sum_step, w_inverse, w_value, sum_finalize)
    sqlite3_close(db)
    assert seen[0] & helpers.SQLITE_DETERMINISTIC
    assert not (seen[1] & helpers.SQLITE_DETERMINISTIC)


def test_registration_error_raises(monkeypatch):
    """A registration that SQLite refuses makes ``register_aggregate`` raise ``RuntimeError``.

    The test replaces ``sqlite3_create_function_v2`` with a function that returns 1, ``SQLITE_ERROR``. A refusal by
    SQLite itself would depend on the build, because the largest ``n_arg`` that SQLite accepts,
    ``SQLITE_MAX_FUNCTION_ARG``, is a compile-time option.
    """
    import pytest
    import numbox.core.bindings.sqlite.udf_helpers as helpers
    monkeypatch.setattr(helpers, "sqlite3_create_function_v2",
                        lambda *a, **k: 1)  # 1 is SQLITE_ERROR.
    db = _open_memory()
    with pytest.raises(RuntimeError, match="registration failed"):
        register_aggregate(db, "err", 1, sum_state_type, sum_init, sum_step, sum_finalize)
    sqlite3_close(db)


def test_plain_python_callbacks_accepted():
    """``register_aggregate`` compiles plain Python callbacks with ``njit``, so a caller need not decorate them."""
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])

    def plain_init():
        return SumState(np.int64(0))

    def plain_step(state, ctx, argc, argv_pp):
        args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
        state.total += sqlite3_value_int64(args[0])

    def plain_finalize(state, ctx):
        sqlite3_result_int64(ctx, state.total)

    register_aggregate(db, "plain_sum", 1, sum_state_type, plain_init, plain_step, plain_finalize)
    gc.collect()  # The query runs after a collection, because numba keeps the compiled code of the wrapped callbacks.
    val, _cap = _read1_int64(db, "SELECT __cap(plain_sum(v)) FROM t")
    sqlite3_close(db)
    assert val == 15


def test_non_callable_callback_rejected():
    """``register_aggregate`` raises ``TypeError``, which names the role, for a callback that is not callable."""
    import pytest
    db = _open_memory()
    with pytest.raises(TypeError, match="step must be a callable"):
        register_aggregate(db, "bad_cb", 1, sum_state_type, sum_init, 42, sum_finalize)
    sqlite3_close(db)


# --- A state of two fields, for an aggregate of two arguments ---
@structref.register
class PairStateType(nb_types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, nb_types.unliteral(t)) for n, t in fields)


class PairState(structref.StructRefProxy):
    def __new__(cls, sa, sb):
        return structref.StructRefProxy.__new__(cls, sa, sb)


structref.define_proxy(PairState, PairStateType, ["sa", "sb"])
pair_state_type = PairStateType([("sa", nb_types.int64), ("sb", nb_types.int64)])


@njit
def pair_init():
    return PairState(np.int64(0), np.int64(0))


@njit
def pair_step(state, ctx, argc, argv_pp):
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    state.sa += sqlite3_value_int64(args[0])
    state.sb += sqlite3_value_int64(args[1])


@njit
def pair_finalize(state, ctx):
    sqlite3_result_int64(ctx, state.sa + 10 * state.sb)


def test_multiarg_multifield_state():
    """An aggregate of two arguments updates a state of two fields."""
    db = _open_memory()
    with c_string("CREATE TABLE t2(a INTEGER, b INTEGER)") as p:
        assert sqlite3_exec(db, p, 0, 0, 0) == SQLITE_OK
    for a, b in [(1, 4), (2, 5), (3, 6)]:
        with c_string("INSERT INTO t2 VALUES (%d, %d)" % (a, b)) as p:
            assert sqlite3_exec(db, p, 0, 0, 0) == SQLITE_OK
    register_aggregate(db, "pairsum", 2, pair_state_type, pair_init, pair_step, pair_finalize)
    val, _cap = _read1_int64(db, "SELECT __cap(pairsum(a, b)) FROM t2")
    sqlite3_close(db)
    assert val == 156  # sa is 1 + 2 + 3 = 6 and sb is 4 + 5 + 6 = 15, so the result is 6 + 10 * 15.


@njit
def raising_finalize(state, ctx):
    raise ValueError("boom from finalize")


def test_finalize_exception_surfaces_error_and_no_leak():
    """An exception in finalize fails the query with "error in user finalize callback" and leaks no state.

    The generated xFinal catches the exception, so it still reaches the release of the state.
    """
    from numba.core.runtime import nrt
    _nrt = nrt._nrt
    if not hasattr(_nrt, "memsys_enable_stats"):
        import pytest
        pytest.skip("NRT allocation stats unavailable")

    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    register_aggregate(db, "raise_fin", 1, sum_state_type, sum_init, sum_step, raising_finalize)
    with c_string("SELECT raise_fin(v) FROM t") as sp:
        rc = sqlite3_exec(db, sp, 0, 0, 0)
    assert rc != SQLITE_OK  # SQLite reports the error instead of a wrong result.
    assert "finalize callback" in _errmsg(db)  # The message names the callback.

    _nrt.memsys_enable_stats()
    with c_string("SELECT raise_fin(v) FROM t") as sp:  # This query runs before the measurement.
        sqlite3_exec(db, sp, 0, 0, 0)
    before = nrt.rtsys.get_allocation_stats()
    for _ in range(10):
        with c_string("SELECT raise_fin(v) FROM t") as sp:
            sqlite3_exec(db, sp, 0, 0, 0)
    after = nrt.rtsys.get_allocation_stats()
    sqlite3_close(db)
    allocated = after.mi_alloc - before.mi_alloc
    freed = after.mi_free - before.mi_free
    assert allocated == freed, "leak on raising finalize: %d alloc, %d free" % (allocated, freed)


@njit
def raising_step(state, ctx, argc, argv_pp):
    raise ValueError("boom from step")


def test_step_exception_surfaces_error_and_no_leak():
    """An exception in step fails the query with "error in user step callback" and leaks no state.

    Without the error, the query would succeed with a wrong result. The generated xStep catches the exception, so numba
    still releases the reference of the borrowed state.
    """
    from numba.core.runtime import nrt
    _nrt = nrt._nrt
    if not hasattr(_nrt, "memsys_enable_stats"):
        import pytest
        pytest.skip("NRT allocation stats unavailable")

    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    register_aggregate(db, "raise_step", 1, sum_state_type, sum_init, raising_step, sum_finalize)
    with c_string("SELECT raise_step(v) FROM t") as sp:
        rc = sqlite3_exec(db, sp, 0, 0, 0)
    assert rc != SQLITE_OK  # SQLite reports the error instead of a wrong result.
    assert "step callback" in _errmsg(db)  # The message names the callback.

    _nrt.memsys_enable_stats()
    with c_string("SELECT raise_step(v) FROM t") as sp:  # This query runs before the measurement.
        sqlite3_exec(db, sp, 0, 0, 0)
    before = nrt.rtsys.get_allocation_stats()
    for _ in range(10):
        with c_string("SELECT raise_step(v) FROM t") as sp:
            sqlite3_exec(db, sp, 0, 0, 0)
    after = nrt.rtsys.get_allocation_stats()
    sqlite3_close(db)
    allocated = after.mi_alloc - before.mi_alloc
    freed = after.mi_free - before.mi_free
    assert allocated == freed, "leak on raising step: %d alloc, %d free" % (allocated, freed)


_WIN_SQL = ("SELECT %s(v) OVER (ORDER BY v ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) "
            "FROM t ORDER BY v")


@njit
def raising_value(state, ctx):
    raise ValueError("boom from value")


def test_value_exception_surfaces_error_and_no_leak():
    """An exception in value fails the window query with "error in user value callback" and leaks no state.

    Without the error, the query would return a stale value.
    """
    from numba.core.runtime import nrt
    _nrt = nrt._nrt
    if not hasattr(_nrt, "memsys_enable_stats"):
        import pytest
        pytest.skip("NRT allocation stats unavailable")

    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    register_window(db, "raise_wval", 1, sum_state_type, sum_init, sum_step, w_inverse, raising_value, sum_finalize)
    with c_string(_WIN_SQL % "raise_wval") as sp:
        rc = sqlite3_exec(db, sp, 0, 0, 0)
    assert rc != SQLITE_OK
    assert "value callback" in _errmsg(db)  # The message names the callback.

    _nrt.memsys_enable_stats()
    with c_string(_WIN_SQL % "raise_wval") as sp:  # This query runs before the measurement.
        sqlite3_exec(db, sp, 0, 0, 0)
    before = nrt.rtsys.get_allocation_stats()
    for _ in range(10):
        with c_string(_WIN_SQL % "raise_wval") as sp:
            sqlite3_exec(db, sp, 0, 0, 0)
    after = nrt.rtsys.get_allocation_stats()
    sqlite3_close(db)
    allocated = after.mi_alloc - before.mi_alloc
    freed = after.mi_free - before.mi_free
    assert allocated == freed, "leak on raising value: %d alloc, %d free" % (allocated, freed)


@njit
def raising_inverse(state, ctx, argc, argv_pp):
    raise ValueError("boom from inverse")


def test_inverse_exception_surfaces_error_and_no_leak():
    """An exception in inverse fails the window query and leaks no state.

    SQLite does not document that an error reported in xInverse fails the statement. The leak check is the part of the
    test that does not depend on that behavior.
    """
    from numba.core.runtime import nrt
    _nrt = nrt._nrt
    if not hasattr(_nrt, "memsys_enable_stats"):
        import pytest
        pytest.skip("NRT allocation stats unavailable")

    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    register_window(db, "raise_winv", 1, sum_state_type, sum_init, sum_step, raising_inverse, w_value, sum_finalize)
    with c_string(_WIN_SQL % "raise_winv") as sp:
        rc = sqlite3_exec(db, sp, 0, 0, 0)
    assert rc != SQLITE_OK

    _nrt.memsys_enable_stats()
    with c_string(_WIN_SQL % "raise_winv") as sp:  # This query runs before the measurement.
        sqlite3_exec(db, sp, 0, 0, 0)
    before = nrt.rtsys.get_allocation_stats()
    for _ in range(10):
        with c_string(_WIN_SQL % "raise_winv") as sp:
            sqlite3_exec(db, sp, 0, 0, 0)
    after = nrt.rtsys.get_allocation_stats()
    sqlite3_close(db)
    allocated = after.mi_alloc - before.mi_alloc
    freed = after.mi_free - before.mi_free
    assert allocated == freed, "leak on raising inverse: %d alloc, %d free" % (allocated, freed)


def test_finalize_exception_empty_group_surfaces_error():
    """An exception in finalize fails the query on an empty group too, with "error in user finalize callback".

    On an empty group, no xStep runs, and ``sqlite3_aggregate_context`` returns 0 in xFinal. xFinal then calls finalize
    on a new state from init, and it catches the exception there as well.
    """
    db = _open_memory()
    _make_table(db, [])  # The table is empty, so xStep never runs.
    register_aggregate(db, "raise_fin_empty", 1, sum_state_type, sum_init, sum_step, raising_finalize)
    with c_string("SELECT raise_fin_empty(v) FROM t") as sp:
        rc = sqlite3_exec(db, sp, 0, 0, 0)
    msg = _errmsg(db)
    sqlite3_close(db)
    assert rc != SQLITE_OK
    assert "finalize callback" in msg  # The message names the callback on this path too.


@njit
def init_raise():
    s = SumState(np.int64(0))
    if s.total >= 0:  # The condition always holds; the return after it gives init the state's return type.
        raise ValueError("boom from init")
    return s


def test_init_exception_surfaces_error_no_unraisable():
    """An exception in init fails the query with "error in user init callback" instead of an unraisable exception.

    xStep calls init at the first row of a group, inside a try/except, so the cfunc boundary does not catch the
    exception and print "Exception ignored".
    """
    db = _open_memory()
    _make_table(db, [1, 2, 3])
    register_aggregate(db, "raise_init", 1, sum_state_type, init_raise, sum_step, sum_finalize)
    captured = []
    old_hook = sys.unraisablehook
    sys.unraisablehook = lambda a: captured.append(a)
    try:
        with c_string("SELECT raise_init(v) FROM t") as sp:
            rc = sqlite3_exec(db, sp, 0, 0, 0)
        msg = _errmsg(db)
    finally:
        sys.unraisablehook = old_hook
    sqlite3_close(db)
    assert not captured, "raising init leaked an unraisable (swallowed): %r" % (captured,)
    assert rc != SQLITE_OK
    assert "init callback" in msg  # The message names init, not finalize.


def test_init_exception_empty_group_surfaces_error():
    """An exception in init on an empty group, where xFinal calls init, fails the query.

    The message is "error in user init callback", not the message of finalize.
    """
    db = _open_memory()
    _make_table(db, [])
    register_aggregate(db, "raise_init_empty", 1, sum_state_type, init_raise, sum_step, sum_finalize)
    with c_string("SELECT raise_init_empty(v) FROM t") as sp:
        rc = sqlite3_exec(db, sp, 0, 0, 0)
    msg = _errmsg(db)
    sqlite3_close(db)
    assert rc != SQLITE_OK
    assert "init callback" in msg
