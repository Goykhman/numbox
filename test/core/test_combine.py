"""Tests of ``Work.combine`` and of the helpers in ``combine_utils``.

``combine`` copies the data of the requested nodes of a graph into a sheaf, a typed dictionary of ``Any`` values keyed
by node name. ``load_dict_into_array`` copies a sheaf into the first row of a structured array, whose dtype
``make_requested_dtype`` makes. The requests name nodes by ``Work`` node or by name, and one node holds a byte string.
The last test checks that ``combine`` stops once it has found a node for every requested name.
"""
import numpy

from numba import njit, typeof
from numba.core.types import CharSeq, float64
from numpy import isclose

from numbox.core.work.combine_utils import load_dict_into_array, make_requested_dtype, make_sheaf_dict
from numbox.core.work.lowlevel_work_utils import ll_make_work
from numbox.core.work.work import make_work
from numbox.core.work.work_utils import make_work_helper
from numbox.utils.highlevel import cres
from test.auxiliary_utils import collect_and_run_tests


def test_combine_1():
    """``combine`` copies the data of a single node into the sheaf.

    ``load_dict_into_array`` then copies the sheaf into the first row of a structured array.
    """
    w1 = make_work_helper("w1", 3.14)

    requested = {"w1": numpy.float64}
    sheaf = make_sheaf_dict(requested)
    w1.combine(sheaf)

    w1_collected = sheaf["w1"].get_as(typeof(w1.data))
    assert isclose(w1_collected, 3.14)

    collected_dtype = make_requested_dtype(requested)
    collected = numpy.empty(shape=(1,), dtype=collected_dtype)
    load_dict_into_array(collected, sheaf)
    assert isclose(collected[0]["w1"], 3.14)


def test_combine_2():
    """A request names nodes by ``Work`` node or by name, and ``combine`` collects a source and the node itself."""
    w1 = make_work_helper("w1", 1.41)
    w2 = make_work_helper("w2", 1.72)
    w3 = make_work_helper("w3", 0.0, sources=(w1, w2), derive_py=lambda w1_, w2_: w1_ + 2 * w2_)
    w3.calculate()
    requested = (w1, "w3")
    sheaf = make_sheaf_dict(requested)
    w3.combine(sheaf)
    w1_collected = sheaf["w1"].get_as(typeof(w1.data))
    assert isclose(w1_collected, 1.41)
    w3_collected = sheaf["w3"].get_as(typeof(w3.data))
    assert isclose(w3_collected, 1.41 + 2 * 1.72)

    collected_dtype = make_requested_dtype({w1: numpy.float64, "w3": numpy.float64})
    collected = numpy.empty(shape=(1,), dtype=collected_dtype)
    load_dict_into_array(collected, sheaf)
    assert isclose(collected[0]["w1"], 1.41)
    assert isclose(collected[0]["w3"], 1.41 + 2 * 1.72)


w2_t = CharSeq(8)


@njit
def make_w2():
    """Create the node ``w2``, whose data is the byte string ``b"double"`` of numba type ``CharSeq(8)``."""
    return ll_make_work("w2", b"double", (), None, data_ty_ref=w2_t)


derive_w3_sig = float64(float64, float64, w2_t)


@cres(derive_w3_sig)
def derive_w3(w0_, w1_, w2_):
    if w2_ == b"double":
        return w0_ + 2 * w1_
    elif w2_ == b"triple":
        return w0_ + 3 * w1_
    return w1_


def test_combine_3():
    """``combine`` collects a byte string of numba type ``CharSeq(8)`` together with float values.

    ``load_dict_into_array`` writes the byte string to a field of dtype ``|S8``.
    """
    w0 = make_work_helper("w0", 1.72)
    w1 = make_work_helper("w1", 1.41)
    w2 = make_w2()

    w3 = make_work("w3", 0.0, sources=(w0, w1, w2), derive=derive_w3)
    w3.calculate()
    requested = (w0, "w2", w3)
    sheaf = make_sheaf_dict(requested)
    w3.combine(sheaf)
    w0_collected = sheaf["w0"].get_as(typeof(w0.data))
    assert isclose(w0_collected, 1.72)
    w2_collected = sheaf["w2"].get_as(w2_t)
    assert w2_collected == b"double"
    w3_collected = sheaf["w3"].get_as(typeof(w3.data))
    assert isclose(w3_collected, 1.72 + 2 * 1.41)

    collected_dtype = make_requested_dtype({w0: numpy.float64, w2: "|S8", w3: numpy.float64})
    collected = numpy.empty(shape=(1,), dtype=collected_dtype)
    load_dict_into_array(collected, sheaf)
    assert isclose(collected[0]["w0"], 1.72)
    assert collected[0]["w2"] == b"double"
    assert isclose(collected[0]["w3"], 1.72 + 2 * 1.41)


def test_combine_4():
    """``combine`` stops once it has found a node for every requested name.

    The graph holds two nodes named ``w1``. ``combine`` reaches the first through ``w2`` and stops before it reaches the
    second through ``w3``, so the sheaf holds the value of the first.
    """
    w1 = make_work_helper("w1", 1.41)
    w1_pretend = make_work_helper("w1", 1.72)
    w2 = make_work_helper("w2", 0.0, sources=(w1,), derive_py=lambda w1_: w1_)
    w3 = make_work_helper("w3", 0.0, sources=(w1_pretend,), derive_py=lambda w1_: w1_)
    w4 = make_work_helper("w4", 0.0, sources=(w2, w3), derive_py=lambda w2_, w3_: (w2_ + w3_) / 2.0)
    sheaf = make_sheaf_dict(w1)
    w4.combine(sheaf)
    w1_collected = sheaf["w1"].get_as(typeof(w1.data))
    if not isclose(w1_collected, 1.41):
        assert isclose(w1_collected, 1.72), f"something else went wrong... got {w1_collected}"
        raise RuntimeError("didn't stop after locating 'w1' in `w1` via `w2`, went on to `w1_pretend` via `w3`")


if __name__ == "__main__":
    collect_and_run_tests(__name__)
