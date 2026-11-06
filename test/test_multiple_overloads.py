"""Tests of template instantiation for the structref ``Entity`` of ``test/multiple_overloads.py``.

When numba types a call to ``Entity``, the ``@overload`` of ``Entity`` runs in Python and creates an ``EntityTypeClass``
instance for the type of the attribute ``x1``. Each type of ``x1`` therefore gives a different entity type, before numba
compiles any code for it. The ``calculate`` method is overloaded on ``EntityTypeClass``, so numba compiles one
``calculate`` for each entity type.
"""

from numpy import isclose
from numba.core.types import int16
from test.auxiliary_utils import collect_and_run_tests
from test.multiple_overloads import Entity


def aux(e1_, e2_, e3_, e4_):
    assert isclose(e1_.calculate(), 2.17)
    assert isclose(e2_.calculate(), 2.17)
    assert isclose(e3_.calculate(), 2.17)
    assert isclose(e4_.calculate(), 2.17)


def test_1():
    """``calculate`` returns 2.17 for entities whose ``x1`` is a ``float64``, an ``int16``, an ``int64`` and a string.

    numba compiles four overloads each of ``make_entity``, ``Entity.calculate`` and ``_calculate_``, one for each entity
    type, and caches them when numbox's jit options enable caching.
    """
    e1 = Entity(3.141)
    e2 = Entity(int16(137))  # A plain int is typed as int64, so this int16 value yields a distinct entity type.
    e3 = Entity(2012)
    e4_x1 = "a string"
    e4 = Entity(e4_x1)
    aux(e1, e2, e3, e4)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
