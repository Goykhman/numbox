"""A cached jitted function, in a module of its own, that calls a method of a struct that ``make_structref`` defines.

``test_make_structref_3`` in ``test/utils/test_highlevel.py`` calls it with a struct that the test defines.
"""
from numba import njit


@njit(cache=True)
def aux_1(s):
    """Return ``s.calculate_1(5, 6)``."""
    return s.calculate_1(5, 6)
