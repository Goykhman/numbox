from numbox.core.work.work_utils import make_work_helper


def test_1():
    w1 = make_work_helper("w1", 0.0)
    w2 = make_work_helper("w2", 0.0, sources=(w1,), derive_py=lambda w1_: w1_ + 1.41)
    w3 = make_work_helper("w3", 0.0, sources=(w2, ), derive_py=lambda w2_: 2 * w2_)
    w4 = make_work_helper("w4", 0.0, sources=(w2,), derive_py=lambda w2_: 3 * w2_)
    w5 = make_work_helper("w5", 0.0, sources=(w3, w4), derive_py=lambda w3_, w4_: w3_ + w4_)
    print(w5.make_inputs_vector())
