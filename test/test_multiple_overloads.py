""" Demonstrates "compile-time" (in this case, it actually means 'pre-compile' time,
i.e., Python-level orchestration time before numba actually starts its compilation)
template instantiation of the `EntityTypeClass` for different 'EntityType's
determined by different types of the Entity's attribute `x1`. Different `calculate`
methods associated with the same `EntityTypeClass` but overloaded for different Entity
types are then created.
"""

from numpy import isclose
from numba.core.types import int16
from test.multiple_overloads import Entity


def aux(e1_, e2_, e3_, e4_):
    assert isclose(e1_.calculate(), 2.17)
    assert isclose(e2_.calculate(), 2.17)
    assert isclose(e3_.calculate(), 2.17)
    assert isclose(e4_.calculate(), 2.17)


def test_1():
    """ 4 overloads for `make_entity`, `_calculate_` and `Entity.calculate` are created and cached """
    e1 = Entity(3.141)
    e2 = Entity(int16(137))  # default is int64
    e3 = Entity(2012)
    e4_x1 = "a string"
    e4 = Entity(e4_x1)
    aux(e1, e2, e3, e4)
