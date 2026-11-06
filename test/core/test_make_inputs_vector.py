"""A test of ``Work.make_inputs_vector`` on a graph in which two nodes share a source."""
from numbox.core.work.work_utils import make_work_helper


def test_1():
    """The test calls ``make_inputs_vector`` on ``w5``, whose sources ``w3`` and ``w4`` share the source ``w2``.

    The call creates the ``Node`` representations of the sources. The test prints the returned list and asserts
    nothing about it.
    """
    w1 = make_work_helper("w1", 0.0)
    w2 = make_work_helper("w2", 0.0, sources=(w1,), derive_py=lambda w1_: w1_ + 1.41)
    w3 = make_work_helper("w3", 0.0, sources=(w2, ), derive_py=lambda w2_: 2 * w2_)
    w4 = make_work_helper("w4", 0.0, sources=(w2,), derive_py=lambda w2_: 3 * w2_)
    w5 = make_work_helper("w5", 0.0, sources=(w3, w4), derive_py=lambda w3_, w4_: w3_ + w4_)
    print(w5.make_inputs_vector())
