"""Parameter-binding tests for the SQLite buildout."""
from ctypes import addressof, c_int64

import numpy as np
import pytest

from numbox.core.bindings._sqlite_bind import (
    sqlite3_bind_blob,
    sqlite3_bind_double,
    sqlite3_bind_int,
    sqlite3_bind_int64,
    sqlite3_bind_null,
    sqlite3_bind_parameter_count,
    sqlite3_bind_parameter_index,
    sqlite3_bind_parameter_name,
    sqlite3_bind_text,
)
from numbox.core.bindings._sqlite_conn import sqlite3_close, sqlite3_open
from numbox.core.bindings._sqlite_constants import (
    SQLITE_OK,
    SQLITE_RANGE,
    SQLITE_TRANSIENT,
)
from numbox.core.bindings._sqlite_stmt import (
    sqlite3_finalize,
    sqlite3_prepare_v2,
)
from numbox.utils.cstrings import c_string
from test.auxiliary_utils import collect_and_run_tests, str_from_p_as_int


@pytest.fixture
def stmt_three_params():
    """Open :memory: db, prepare 'SELECT ?1, ?2, ?3', yield (db_p, stmt_p),
    teardown finalizes + closes."""
    db_p = c_int64(0)
    with c_string(":memory:") as name_p:
        assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK

    stmt_p = c_int64(0)
    tail_p = c_int64(0)
    with c_string("SELECT ?1, ?2, ?3") as sql_p:
        assert sqlite3_prepare_v2(db_p.value, sql_p, -1,
                                  addressof(stmt_p), addressof(tail_p)) == SQLITE_OK
    yield db_p.value, stmt_p.value
    sqlite3_finalize(stmt_p.value)
    sqlite3_close(db_p.value)


def test_bind_int_returns_ok(stmt_three_params):
    _, stmt_p = stmt_three_params
    assert sqlite3_bind_int(stmt_p, 1, 42) == SQLITE_OK


def test_bind_int64_returns_ok(stmt_three_params):
    _, stmt_p = stmt_three_params
    assert sqlite3_bind_int64(stmt_p, 2, 2**40) == SQLITE_OK


def test_bind_double_returns_ok(stmt_three_params):
    _, stmt_p = stmt_three_params
    assert sqlite3_bind_double(stmt_p, 3, 3.14) == SQLITE_OK


def test_bind_text_transient_returns_ok(stmt_three_params):
    _, stmt_p = stmt_three_params
    with c_string("hello") as text_p:
        assert sqlite3_bind_text(stmt_p, 1, text_p, -1, SQLITE_TRANSIENT) == SQLITE_OK


def test_bind_blob_with_numpy_uint8_returns_ok(stmt_three_params):
    _, stmt_p = stmt_three_params
    buf = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    data_p = buf.ctypes.data
    assert sqlite3_bind_blob(stmt_p, 1, data_p, buf.nbytes, SQLITE_TRANSIENT) == SQLITE_OK


def test_bind_null_returns_ok(stmt_three_params):
    _, stmt_p = stmt_three_params
    assert sqlite3_bind_null(stmt_p, 1) == SQLITE_OK


def test_bind_parameter_count_returns_three(stmt_three_params):
    _, stmt_p = stmt_three_params
    assert sqlite3_bind_parameter_count(stmt_p) == 3


def test_bind_out_of_range_returns_sqlite_range(stmt_three_params):
    _, stmt_p = stmt_three_params
    rc = sqlite3_bind_int(stmt_p, 99, 0)
    assert rc == SQLITE_RANGE


def test_bind_parameter_index_by_name():
    db_p = c_int64(0)
    with c_string(":memory:") as name_p:
        assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    try:
        stmt_p = c_int64(0)
        tail_p = c_int64(0)
        with c_string("SELECT :foo, :bar") as sql_p:
            sqlite3_prepare_v2(db_p.value, sql_p, -1,
                               addressof(stmt_p), addressof(tail_p))
        with c_string(":foo") as foo_p, c_string(":bar") as bar_p:
            assert sqlite3_bind_parameter_index(stmt_p.value, foo_p) == 1
            assert sqlite3_bind_parameter_index(stmt_p.value, bar_p) == 2
        # name lookup round trip (output points into SQLite-owned memory)
        name_back_p = sqlite3_bind_parameter_name(stmt_p.value, 1)
        assert str_from_p_as_int(name_back_p) == ":foo"
        sqlite3_finalize(stmt_p.value)
    finally:
        sqlite3_close(db_p.value)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
