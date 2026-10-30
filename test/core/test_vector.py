"""Tests of the growable vector of ``numbox.core.vector.vector``.

The tests cover construction, ``len`` and indexing, ``vector_push`` and ``vector_extend`` with and without growth, and
the cases of a full buffer and of empty vectors. They check that ``make_vector`` returns the same factory for an
element type, and that a capacity of 0 and a call of ``Vector`` in jitted code are refused. One test checks that a
second process can use the accessors that the first process cached.
"""
import numpy
import os
import pytest
import subprocess
import sys
import textwrap

from numba import njit, types as nb_types
from numba.core.errors import NumbaError, TypingError

from numbox.core.vector.vector import _vector_cache, make_vector, Vector, vector_extend, vector_push


Float64Vec, _Float64VecType = make_vector(nb_types.float64)


@njit(cache=True)
def _construction_and_len_probe():
    v = Float64Vec(8)
    return len(v), v.buf.shape[0]


def test_construction_and_len():
    """A cached jitted function creates a vector of capacity 8, whose length is 0 and whose buffer holds 8 elements."""
    size, cap = _construction_and_len_probe()
    assert size == 0
    assert cap == 8


def test_getitem_setitem():
    """``v[i]`` reads the buffer, and ``v[i] = x`` writes it.

    ``len(v)`` returns the size that the test sets, and the assignment does not change it.
    """
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        v = Float64Vec(4)
        v.buf[0] = 10.0
        v.buf[1] = 20.0
        v.size = 2
        v[1] = 99.0
        return v[0], v[1], len(v)

    a, b, n = go()
    assert a == 10.0
    assert b == 99.0
    assert n == 2


def test_factory_caching():
    """``make_vector`` returns the same factory and type for the same element type, stored in ``_vector_cache``."""
    r1 = make_vector(nb_types.float64)
    r2 = make_vector(nb_types.float64)
    assert r1[0] is r2[0]
    assert r1[1] is r2[1]
    assert nb_types.float64.key in _vector_cache


def test_vector_push():
    """Three pushes into a vector of capacity 4 store the elements in order and leave the capacity at 4."""
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        v = Float64Vec(4)
        vector_push(v, 1.0)
        vector_push(v, 2.0)
        vector_push(v, 3.0)
        return v[0], v[1], v[2], len(v), v.buf.shape[0]

    a, b, c, n, cap = go()
    assert (a, b, c) == (1.0, 2.0, 3.0)
    assert n == 3
    assert cap == 4


def test_vector_push_growth():
    """Five pushes into a vector of capacity 2 double its capacity twice, to 8, and keep the elements in order."""
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        v = Float64Vec(2)
        for i in range(5):
            vector_push(v, float(i * 10))
        return v[0], v[1], v[2], v[3], v[4], len(v), v.buf.shape[0]

    vals = go()
    assert vals[:5] == (0.0, 10.0, 20.0, 30.0, 40.0)
    assert vals[5] == 5
    assert vals[6] == 8


def test_vector_extend():
    """Extending a vector of two elements by three elements doubles its capacity from 4 to 8."""
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        a = Float64Vec(4)
        b = Float64Vec(4)
        vector_push(a, 1.0)
        vector_push(a, 2.0)
        vector_push(b, 3.0)
        vector_push(b, 4.0)
        vector_push(b, 5.0)
        vector_extend(a, b)
        return a[0], a[1], a[2], a[3], a[4], len(a), a.buf.shape[0]

    vals = go()
    assert vals[:5] == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert vals[5] == 5
    assert vals[6] == 8


def test_vector_extend_no_growth():
    """Extending a vector by elements that fit keeps its capacity of 8."""
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        a = Float64Vec(8)
        b = Float64Vec(4)
        vector_push(a, 1.0)
        vector_push(b, 2.0)
        vector_push(b, 3.0)
        vector_extend(a, b)
        return a[0], a[1], a[2], len(a), a.buf.shape[0]

    vals = go()
    assert vals[:3] == (1.0, 2.0, 3.0)
    assert vals[3] == 3
    assert vals[4] == 8


def test_vector_push_exact_capacity():
    """Four pushes into a vector of capacity 4 fill the buffer without growing it.

    ``vector_push`` grows the buffer only when the buffer is full before the push.
    """
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        v = Float64Vec(4)
        vector_push(v, 1.0)
        vector_push(v, 2.0)
        vector_push(v, 3.0)
        vector_push(v, 4.0)
        return v[0], v[1], v[2], v[3], len(v), v.buf.shape[0]

    vals = go()
    assert vals[:4] == (1.0, 2.0, 3.0, 4.0)
    assert vals[4] == 4
    assert vals[5] == 4


def test_vector_extend_empty_src():
    """Extending a vector by an empty vector leaves its elements and its capacity unchanged."""
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        a = Float64Vec(4)
        b = Float64Vec(4)
        vector_push(a, 1.0)
        vector_push(a, 2.0)
        vector_extend(a, b)
        return a[0], a[1], len(a), a.buf.shape[0]

    vals = go()
    assert vals[:2] == (1.0, 2.0)
    assert vals[2] == 2
    assert vals[3] == 4


def test_vector_extend_empty_dst():
    """Extending an empty vector copies the elements of the source and keeps the capacity when they fit."""
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        a = Float64Vec(4)
        b = Float64Vec(4)
        vector_push(b, 1.0)
        vector_push(b, 2.0)
        vector_push(b, 3.0)
        vector_extend(a, b)
        return a[0], a[1], a[2], len(a), a.buf.shape[0]

    vals = go()
    assert vals[:3] == (1.0, 2.0, 3.0)
    assert vals[3] == 3
    assert vals[4] == 4


def test_vector_extend_both_empty():
    """Extending an empty vector by an empty vector leaves the length at 0 and the capacity at 4."""
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        a = Float64Vec(4)
        b = Float64Vec(4)
        vector_extend(a, b)
        return len(a), a.buf.shape[0]

    n, cap = go()
    assert n == 0
    assert cap == 4


def test_vector_extend_multiple_doublings():
    """One ``vector_extend`` doubles the capacity three times, from 2 to 16, to fit eleven elements."""
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        a = Float64Vec(2)
        b = Float64Vec(16)
        vector_push(a, 0.0)
        for i in range(10):
            vector_push(b, float(i + 1))
        vector_extend(a, b)
        return a[0], a[1], a[10], len(a), a.buf.shape[0]

    first, second, last, n, cap = go()
    assert first == 0.0
    assert second == 1.0
    assert last == 10.0
    assert n == 11
    assert cap == 16


def test_vector_push_special_floats():
    """NaN and the two infinities are stored and read back unchanged."""
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        v = Float64Vec(4)
        vector_push(v, numpy.nan)
        vector_push(v, numpy.inf)
        vector_push(v, -numpy.inf)
        return v[0], v[1], v[2], len(v)

    a, b, c, n = go()
    assert numpy.isnan(a)
    assert b == numpy.inf
    assert c == -numpy.inf
    assert n == 3


def test_multi_dtype_int64():
    """A vector of int64 elements grows and stores integers in the same way as a vector of float64 elements."""
    Int64Vec, _ = make_vector(nb_types.int64)

    @njit
    def go():
        v = Int64Vec(2)
        vector_push(v, 100)
        vector_push(v, 200)
        vector_push(v, 300)
        return v[0], v[1], v[2], len(v), v.buf.shape[0]

    vals = go()
    assert vals[:3] == (100, 200, 300)
    assert vals[3] == 3
    assert vals[4] == 4


def test_zero_capacity_rejected():
    """A capacity of 0 fails the assertion in ``create``, because a buffer of capacity 0 cannot grow by doubling."""
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        return Float64Vec(0)

    with pytest.raises(AssertionError):
        go()


def test_cache_survives_across_processes(tmp_path):
    """A second process that uses the same cache directory creates a vector and reads its size, as the first did.

    The accessors of ``Vector`` compile with numbox's jit options, which enable caching, so the second process loads
    the accessor that the first compiled. ``VectorTypeClass`` is defined once, at module level, so the vector type of
    the second process matches the type in the cache index. The test checks that both processes exit normally.
    """
    probe = textwrap.dedent("""
        from numba import types as nb_types
        from numbox.core.vector.vector import make_vector
        create, _ = make_vector(nb_types.float64)
        v = create(8)
        print(v.size)
    """)
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path)}

    r1 = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True,
    )
    assert r1.returncode == 0, f"run1 (cold) failed:\n{r1.stderr}"

    r2 = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True,
    )
    assert r2.returncode == 0, f"run2 (warm) failed:\n{r2.stderr}"


def test_vector_ctor_raises_numbaerror_in_njit():
    """Calling ``Vector`` in jitted code fails to compile with an error that names the factory ``make_vector``."""
    def caller():
        return Vector(5)

    with pytest.raises((NumbaError, TypingError), match="make_vector"):
        njit()(caller)()
