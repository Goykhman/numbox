"""Shared pytest fixtures for SQLite binding tests."""
from ctypes import addressof, c_int64

import pytest

from numbox.core.bindings._sqlite_conn import sqlite3_close, sqlite3_open
from numbox.core.bindings._sqlite_constants import SQLITE_OK
from numbox.utils.cstrings import c_string


@pytest.fixture
def memory_db():
    """Open ``:memory:`` via the raw bindings; yield the connection pointer;
    close on teardown."""
    db_p = c_int64(0)
    with c_string(":memory:") as name_p:
        assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    yield db_p.value
    sqlite3_close(db_p.value)
