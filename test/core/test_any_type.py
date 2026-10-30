"""Tests of ``Any``, which holds one value of any numba type.

The tests store scalars, a string, arrays, a function value and structrefs with ``make_any``, read them back with
``get_as`` and replace them with ``reset``, from Python and from jitted code. They also count the references that an
``Any`` takes to a structref that it holds, and check that ``get_as`` returns the stored object rather than a copy.
"""
import numpy

from numba import float64, int32, int64, njit, typeof
from numba.core import types
from numba.typed import Dict
from numbox.core.any.any_type import AnyType, make_any
from numbox.utils.meminfo import get_nrt_refcount, structref_meminfo
from numbox.utils.highlevel import cres
from test.auxiliary_utils import collect_and_run_tests, deref_int64_intp
from test.common_structrefs import S1, S1Type, S3, S3Type


def test_1():
    """``make_any`` stores a negative integer, and ``get_as`` reads it back as an int64."""
    x = -65
    any1 = make_any(x)
    assert any1.get_as(int64) == x


def test_2():
    """``Any`` values of different types can share a typed dictionary, and ``get_as`` reads each with its own type.

    An array is returned without a copy.
    """
    x1 = 217
    any1 = make_any(x1)
    x2 = 2.17
    any2 = make_any(x2)
    x3 = "a few words"
    any3 = make_any(x3)
    a = numpy.array([1, 2], dtype=numpy.int32)
    any4 = make_any(a)
    d1 = Dict.empty(key_type=types.unicode_type, value_type=AnyType)
    d1['x1'] = any1
    d1['x2'] = any2
    d1['x3'] = any3
    d1['x4'] = any4
    assert d1['x1'].get_as(int64) == x1
    assert (d1['x2'].get_as(float64) - x2) < 1e-15
    assert d1['x3'].get_as(types.unicode_type) == x3
    a1 = d1['x4'].get_as(types.Array(int32, 1, 'C'))
    assert numpy.allclose(a, a1)
    assert a1.ctypes.data == a.ctypes.data


@njit
def aux(x):
    """Return an ``Any`` that holds a new ``S1`` whose field ``x2`` is ``x``, created in jitted code."""
    s1 = S1(81, x, 2.17)
    any1 = make_any(s1)
    return any1


def test_3():
    """An ``Any`` that jitted code creates and returns to Python holds a ``StructRef`` that ``get_as`` reads."""
    x = 137
    val = aux(x).get_as(S1Type)
    assert val.x2 == x


def test_4():
    """An ``Any`` holds a ``cres`` function value, and ``get_as`` returns the same Python object, which can be called.

    The test reads the value from Python and from jitted code.
    """
    aux1_sig = float64(float64)
    aux1_ty = aux1_sig.as_type()

    @cres(aux1_sig)
    def aux1(x):
        return 2 * x

    def run():
        a1 = make_any(aux1)
        f_ = a1.get_as(aux1_ty)
        return f_

    f = run()
    assert id(f) == id(aux1)
    assert abs(f(2.3) - 2 * 2.3) < 1e-15

    f = njit(run)()
    assert id(f) == id(aux1)
    assert abs(f(2.3) - 2 * 2.3) < 1e-15


def test_5():
    """``make_any`` takes a reference to a structref that it stores, and deleting the ``Any`` releases it.

    The ``_Content`` wrapper holds the reference, and the ``Any`` holds the wrapper cast to ``ErasedType``. ``get_as``
    returns a proxy of the same MemInfo, which takes a reference of its own.
    """
    s1 = S1(81, 67, 2.17)
    assert get_nrt_refcount(s1) == 1
    any1 = make_any(s1)
    assert get_nrt_refcount(any1) == 1
    assert get_nrt_refcount(s1) == 2
    any2 = make_any(s1)  # noqa: F841
    assert get_nrt_refcount(s1) == 3
    s1_2 = any1.get_as(S1Type)
    assert get_nrt_refcount(s1) == 4
    assert get_nrt_refcount(s1_2) == 4, "same `MemInfo` as `s1`, see `test_6`"
    del any1
    assert get_nrt_refcount(s1) == 3
    del s1_2
    assert get_nrt_refcount(s1) == 2
    del any2
    assert get_nrt_refcount(s1) == 1


def test_6():
    """``get_as`` returns the structref that ``make_any`` received, with the same MemInfo, not a copy."""
    s1 = S1(81, 67, 2.17)
    s1_meminfo_p = structref_meminfo(s1)[0]
    any1 = make_any(s1)
    s1_2 = any1.get_as(S1Type)
    s1_2_meminfo_p = structref_meminfo(s1_2)[0]
    assert s1_meminfo_p == s1_2_meminfo_p


def test_7():
    """In jitted code, the ``ErasedType`` wrapper in the field ``p`` of an ``Any`` has a reference count of 1."""

    @njit
    def aux():
        s1 = S1(81, 67, 2.17)
        any1 = make_any(s1)
        _content = any1.p
        _conent_meminfo_p = structref_meminfo(_content)[0]
        return deref_int64_intp(_conent_meminfo_p)

    ref_ct_1 = aux()
    assert ref_ct_1 == 1


def test_8():
    """``get_as`` returns the record array that an ``Any`` holds, with the data address of the original array."""
    rec1 = numpy.dtype([("x", numpy.int16), ("y", numpy.float64)])
    a1 = numpy.array([(137, 2.17)], dtype=rec1)
    a1_ty = typeof(a1)
    any1 = make_any(a1)
    assert any1.get_as(a1_ty).ctypes.data == a1.ctypes.data


def test_9():
    """An ``Any`` holds a ``StructRef`` whose field is another ``StructRef``, and ``get_as`` reads the nested fields."""
    s1 = S1(81, 67, 2.17)
    s3 = S3(s1, 2.17)
    any1 = make_any(s3)
    s3_ = any1.get_as(S3Type)
    assert s3_.x1.x1 == 81
    assert s3_.x1.x2 == 67
    assert abs(s3_.x1.x3 - 2.17) < 1e-15
    assert abs(s3_.x2 - 2.17) < 1e-15


def test_10():
    """``reset`` replaces the value of an ``Any`` with a value of another type.

    ``reset`` takes a reference to a structref that it stores, and a later ``reset`` releases that reference once.
    """
    any = make_any(3.14)
    assert abs(any.get_as(float64) - 3.14) < 1e-15
    any.reset("hello")
    assert any.get_as(types.unicode_type) == "hello"
    s1 = S1(123, 432, 2.17)
    assert get_nrt_refcount(s1) == 1
    any.reset(s1)
    assert get_nrt_refcount(s1) == 2, "`any` now holds another ref to `s1` via its `_Content` member"
    assert any.get_as(S1Type).x2 == 432
    any.reset(11)
    assert get_nrt_refcount(s1) == 1, \
        "resetting onto a different value released `any`'s ref to the old `s1` payload exactly once " \
        "(== 2 would be a leak, == 0 a double-free)"


if __name__ == '__main__':
    collect_and_run_tests(__name__)
