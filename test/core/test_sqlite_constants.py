"""Tests of the values of :mod:`~numbox.core.bindings.sqlite.constants`.

The test compares the constraint operators of a virtual table's ``xBestIndex``, the ``SQLITE_INDEX_CONSTRAINT_*``
constants, with their values in SQLite's ``sqlite3.h``. The other constants of the module have no test of their own
here; the tests of the bindings use them.
"""
from test.auxiliary_utils import collect_and_run_tests


def test_index_constraint_ops():
    """The eight operators equal the codes that ``sqlite3.h`` defines for the ``op`` field of a constraint."""
    from numbox.core.bindings.sqlite.constants import (
        SQLITE_INDEX_CONSTRAINT_EQ, SQLITE_INDEX_CONSTRAINT_GT, SQLITE_INDEX_CONSTRAINT_LE,
        SQLITE_INDEX_CONSTRAINT_LT, SQLITE_INDEX_CONSTRAINT_GE, SQLITE_INDEX_CONSTRAINT_NE,
        SQLITE_INDEX_CONSTRAINT_ISNULL, SQLITE_INDEX_CONSTRAINT_IS,
    )
    assert (SQLITE_INDEX_CONSTRAINT_EQ, SQLITE_INDEX_CONSTRAINT_GT, SQLITE_INDEX_CONSTRAINT_LE,
            SQLITE_INDEX_CONSTRAINT_LT, SQLITE_INDEX_CONSTRAINT_GE, SQLITE_INDEX_CONSTRAINT_NE,
            SQLITE_INDEX_CONSTRAINT_ISNULL, SQLITE_INDEX_CONSTRAINT_IS) == (2, 4, 8, 16, 32, 68, 71, 72)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
