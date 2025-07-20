from numba import njit


@njit(cache=True)
def aux_1(s):
    return s.calculate_1(5, 6)
