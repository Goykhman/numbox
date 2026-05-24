"""Shared pytest fixtures for SQLite binding tests."""
from ctypes import addressof, c_int64

import pytest

from numbox.core.bindings._sqlite_conn import sqlite3_close, sqlite3_open
from numbox.core.bindings._sqlite_constants import SQLITE_OK
from test.auxiliary_utils import cstr


@pytest.fixture
def memory_db():
    """Open ``:memory:`` via the raw bindings; yield the connection pointer;
    close on teardown."""
    name_buf, name_p = cstr(":memory:")
    db_p = c_int64(0)
    assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    yield db_p.value
    sqlite3_close(db_p.value)
