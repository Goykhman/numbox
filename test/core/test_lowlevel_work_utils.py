from numba import njit
from numba.core.types import float64, UniTuple
from numba.extending import intrinsic
from numpy import isclose

from numbox.utils.meminfo import _structref_meminfo
from numbox.core.work.node_base import NodeBaseType
from numbox.core.work.lowlevel_work_utils import ll_make_work, create_uniform_inputs
from numbox.core.work.print_tree import make_image
from numbox.utils.highlevel import cres
from test.auxiliary_utils import collect_and_run_tests, _deref_int64_intp


def test_create_uniform_inputs():
    """ Ascertain that reference count of memory regions managed by the uniform
     tuple created by `create_uniform_inputs` agree with expectations. """

    @intrinsic
    def _create_uniform_inputs(typingctx, tup_ty):
        def codegen(context, builder, signature, arguments):
            tup = arguments[0]
            return create_uniform_inputs(context, builder, tup_ty, tup)
        return UniTuple(NodeBaseType, 2)(tup_ty), codegen

    @njit
    def uniform_tup_ref_ct():
        w1 = ll_make_work("w1", 0.0, (), None)
        w1_mi_ = _structref_meminfo(w1)
        w1_payload_p_ = w1_mi_[1]
        w1_mi_ref_ct_ = _deref_int64_intp(w1_mi_[0])

        w2 = ll_make_work("w2", 0.0, (), None)
        w2_mi_ = _structref_meminfo(w2)
        w2_payload_p_ = w2_mi_[1]
        w2_mi_ref_ct_ = _deref_int64_intp(w2_mi_[0])

        het_tup = (w1, w2)
        tup_ = _create_uniform_inputs(het_tup)

        tup_mi_ = _structref_meminfo(tup_)
        tup_payload_p_ = tup_mi_[1]
        tup_mi_ref_ct_ = _deref_int64_intp(tup_mi_[0])

        w1_ = tup_[0]
        w1__mi_ = _structref_meminfo(w1_)
        w1__payload_p_ = w1__mi_[1]
        w1__mi_ref_ct_ = _deref_int64_intp(w1__mi_[0])

        w2_ = tup_[1]
        w2__mi_ = _structref_meminfo(w2_)
        w2__payload_p_ = w2__mi_[1]
        w2__mi_ref_ct_ = _deref_int64_intp(w2__mi_[0])

        return (
            tup_, w1, w2, w1_, w2_,
            tup_mi_ref_ct_, tup_payload_p_, w1_payload_p_, w2_payload_p_, w1__payload_p_, w2__payload_p_,
            w1_mi_ref_ct_, w2_mi_ref_ct_, w1__mi_ref_ct_, w2__mi_ref_ct_
        )

    (
        tup, w1, w2, w1_, w2_,
        tup_mi_ref_ct, tup_payload_p, w1_payload_p, w2_payload_p, w1__payload_p, w2__payload_p,
        w1_mi_ref_ct, w2_mi_ref_ct, w1__mi_ref_ct, w2__mi_ref_ct
    ) = uniform_tup_ref_ct()

    assert w1_mi_ref_ct == 1, f"Got {w1_mi_ref_ct} but just initialized ref count should be 1"
    assert w2_mi_ref_ct == 1, f"Got {w2_mi_ref_ct} but just initialized ref count should be 1"

    assert w1__mi_ref_ct == 3, f"Got {w1_mi_ref_ct} but expected 3 from w1, w1_, tup"
    assert w2__mi_ref_ct == 3, f"Got {w2_mi_ref_ct} but expected 3 from w2, w2_, tup"

    assert w1_payload_p == w1__payload_p, f"Original data should be recovered, got {w1_payload_p, w1__payload_p}"
    assert w2_payload_p == w2__payload_p, f"Original data should be recovered, got {w2_payload_p, w2__payload_p}"

    assert tup_payload_p == w1_payload_p, r"w1 memory region should be aligned with {w1, w2} tuple's"
    assert tup_mi_ref_ct == 3, f"Got {tup_mi_ref_ct} but expected 3 for the memory region pointed at by tup"


def test_make_work_ll():

    @cres(float64())
    def derive_v0():
        return 3.14

    @njit(cache=True)
    def v0_maker(derive_):
        return ll_make_work("v0", 0.0, (), derive_)

    v0 = v0_maker(derive_v0)
    assert v0.data == 0
    assert v0.name == "v0"
    assert v0.inputs == ()
    assert not v0.derived
    v0.calculate()
    assert isclose(v0.data, 3.14)


@cres(float64())
def derive_w1():
    return 3.14


@cres(float64())
def derive_w2():
    return 1.41


@cres(float64(float64, float64))
def derive_w3(w1_, w2_):
    return w1_ * w2_


@cres(float64())
def derive_w4():
    return 1.72


@cres(float64(float64, float64))
def derive_w5(w3_, w4_):
    return w3_ + w4_


@njit(cache=True)
def run_test_work_calculate(derive_w1_, derive_w2_, derive_w3_, derive_w4_, derive_w5_):
    w1 = ll_make_work("w1", 0.0, (), derive_w1_)
    w2 = ll_make_work("w2", 0.0, (), derive_w2_)
    w3 = ll_make_work("w3", 0.0, (w1, w2), derive_w3_)
    w4 = ll_make_work("w4", 0.0, (), derive_w4_)
    w5 = ll_make_work("w5", 0.0, (w3, w4), derive_w5_)
    w1_init_data = w1.data
    w2_init_data = w2.data
    w3_init_data = w3.data
    w4_init_data = w4.data
    w5_init_data = w5.data
    w5.calculate()
    return (
        w1_init_data, w2_init_data, w3_init_data, w4_init_data, w5_init_data,
        w1, w2, w3, w4, w5
    )


def test_make_work_ll_sources():
    (
        w1_init_data, w2_init_data, w3_init_data, w4_init_data, w5_init_data,
        w1, w2, w3, w4, w5
    ) = run_test_work_calculate(derive_w1, derive_w2, derive_w3, derive_w4, derive_w5)
    assert w1_init_data == 0
    assert w2_init_data == 0
    assert w3_init_data == 0
    assert w4_init_data == 0
    assert w5_init_data == 0
    assert abs(w1.data - 3.14) < 1e-15
    assert abs(w2.data - 1.41) < 1e-15
    assert abs(w3.data - w1.data * w2.data) < 1e-15
    assert abs(w4.data - 1.72) < 1e-15
    assert abs(w5.data - (w3.data + w4.data)) < 1e-15


def test_make_work_ll_graph():
    @njit
    def aux():
        w1_ = ll_make_work("first", 0.0, (), None)
        w2_ = ll_make_work("second", 0.0, (w1_,), None)
        w3_ = ll_make_work("third", 0.0, (w2_,), None)
        w4_ = ll_make_work("fourth", 0.0, (), None)
        w5_ = ll_make_work("fifth", 0.0, (w3_, w4_), None)
        return w5_

    w = aux()
    assert w.get_inputs_names() == ["third", "fourth"]
    tree_image = make_image(w)
    tree_image_ref = """
fifth--third---second--first
       |
       fourth"""
    assert tree_image == tree_image_ref


if __name__ == "__main__":
    collect_and_run_tests(__name__)
