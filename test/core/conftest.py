"""pytest fixtures that the SQLite tests of test/core share."""
from ctypes import addressof, c_int64

import pytest

from numbox.core.bindings.sqlite.conn import sqlite3_close, sqlite3_open
from numbox.core.bindings.sqlite.constants import SQLITE_OK
from numbox.utils.cstrings import c_string


@pytest.fixture
def memory_db():
    """Yield the handle of a connection to a new in-memory database, and close the connection after the test."""
    db_p = c_int64(0)
    with c_string(":memory:") as name_p:
        assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    yield db_p.value
    sqlite3_close(db_p.value)
