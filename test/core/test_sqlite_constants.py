"""Presence + value assertions for the public SQLite constant surface."""
from test.auxiliary_utils import collect_and_run_tests


def test_index_constraint_ops():
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
