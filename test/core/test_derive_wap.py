import gc
import weakref

import numpy
import pytest
from numba import njit, prange
from numba.core.types import float64
from numba.core.types.function_type import CompileResultWAP

from numbox.core.configurations import numba_version
from numbox.core.work.derive_wap import DeriveFunctionType, DeriveWAP, rewrap_derive
from numbox.core.work.work import make_work
from numbox.core.work.work_utils import make_work_helper
from numbox.utils.highlevel import cres


pytestmark = pytest.mark.skipif(
    numba_version < 61, reason="`jit_addr` slot added to `FunctionModel` in numba 0.61"
)


def _raise_when_positive(a):
    if a > 0.0:
        raise ValueError("derive boom")
    return a + 1.0


def test_cres_mints_a_derive_wap():
    """The propagating convention is only reachable through numbox's own type."""
    @cres(float64(float64))
    def compiled(x):
        return x + 1.0

    assert isinstance(compiled, DeriveWAP)
    assert compiled.jit_address != 0
    from numba import typeof
    assert isinstance(typeof(compiled), DeriveFunctionType)


def test_raising_derive_propagates_and_leaves_the_node_recalculable():
    """The whole point: the failure is visible, and it does not poison the node.

    Previously `calculate` returned normally, `data` was zero-filled and `derived`
    was set, so the wrong value was cached permanently.
    """
    source = make_work_helper("source", 5.0)
    node = make_work_helper("node", 99.0, sources=(source,), derive_py=_raise_when_positive)

    with pytest.raises(ValueError, match="derive boom"):
        node.calculate()

    assert node.data == 99.0, "`data` must be untouched by a failed derive"
    assert node.derived == 0, "`derived` must stay unset so the node can be calculated again"


def test_the_original_exception_type_and_message_survive():
    def divide_by_zero(a):
        if a > 0.0:
            return a / 0.0
        return a

    source = make_work_helper("source", 2.0)
    node = make_work_helper(
        "node", 0.0, sources=(source,), derive_py=divide_by_zero,
        jit_options={"error_model": "python"},
    )
    with pytest.raises(ZeroDivisionError):
        node.calculate()


def test_the_node_recalculates_once_the_cause_is_addressed():
    """`derived` staying unset is what makes the retry possible: a cached zero would
    have made the second call a no-op."""
    def raise_on_positive(a):
        if a[0] > 0.0:
            raise ValueError("derive boom")
        return a[0] + 1.0

    source = make_work_helper("source", numpy.array([5.0]))
    node = make_work_helper("node", 99.0, sources=(source,), derive_py=raise_on_positive)

    with pytest.raises(ValueError, match="derive boom"):
        node.calculate()
    assert node.derived == 0

    source.data[0] = -3.0
    node.calculate()
    assert node.data == -2.0
    assert node.derived == 1


def test_unicode_payload_is_readable_after_a_failure():
    """A zero-filled `unicode_type` payload has a NULL data pointer, so reading it
    back from Python used to segfault the interpreter rather than raise."""
    def raise_for_text(a):
        if a > 0.0:
            raise ValueError("no text")
        return "computed"

    source = make_work_helper("source", 1.0)
    node = make_work_helper("node", "initial", sources=(source,), derive_py=raise_for_text)

    with pytest.raises(ValueError, match="no text"):
        node.calculate()

    assert node.data == "initial"


def test_make_work_upgrades_a_foreign_compile_result_wap():
    """A derive built directly against numba carries no callconv entry point, so it
    has to be re-wrapped before it reaches jitted scope."""
    jitted = njit(float64(float64))(_raise_when_positive)
    foreign = CompileResultWAP(jitted.get_compile_result(jitted.nopython_signatures[0]))
    assert not isinstance(foreign, DeriveWAP)

    source = make_work("source", 5.0)
    node = make_work("node", 99.0, sources=(source,), derive=foreign)

    with pytest.raises(ValueError, match="derive boom"):
        node.calculate()
    assert node.data == 99.0
    assert node.derived == 0


def test_the_upgraded_wrapper_outlives_the_call_that_made_it():
    """`py_addr` records the derive's address without taking a reference, so an
    upgrade minted fresh per call would be freed the moment the caller returned and
    every `Work` built from it would point at released memory. The upgrade has to be
    anchored to the object it upgrades."""
    jitted = njit(float64(float64))(_raise_when_positive)
    foreign = CompileResultWAP(jitted.get_compile_result(jitted.nopython_signatures[0]))

    tracker = weakref.ref(rewrap_derive(foreign))
    gc.collect()

    assert tracker() is not None, "the upgraded wrapper was freed while its address was still in use"


def test_upgrading_the_same_derive_twice_yields_the_same_wrapper():
    jitted = njit(float64(float64))(_raise_when_positive)
    foreign = CompileResultWAP(jitted.get_compile_result(jitted.nopython_signatures[0]))
    assert rewrap_derive(foreign) is rewrap_derive(foreign)


def test_a_foreign_derive_survives_a_round_trip_through_jitted_scope():
    """Reading the derive back out dereferences `py_addr`, which is where a freed
    wrapper shows up as a crash rather than as a wrong answer."""
    jitted = njit(float64(float64))(_raise_when_positive)
    foreign = CompileResultWAP(jitted.get_compile_result(jitted.nopython_signatures[0]))

    source = make_work("source", 5.0)
    node = make_work("node", 99.0, sources=(source,), derive=foreign)
    gc.collect()

    @njit
    def read_derive(work_):
        return work_.derive

    assert isinstance(read_derive(node), DeriveWAP)


def test_rewrap_derive_leaves_everything_else_alone():
    assert rewrap_derive(None) is None

    @cres(float64(float64))
    def already(x):
        return x

    assert rewrap_derive(already) is already


def test_array_payload_is_untouched_by_a_failure():
    """An array payload silently became an empty array under the old behaviour,
    which carried no signal at all."""
    def raise_for_array(a):
        if a.shape[0] > 0:
            raise ValueError("no array")
        return a

    initial = numpy.arange(4.0)
    source = make_work_helper("source", numpy.arange(4.0))
    node = make_work_helper("node", initial.copy(), sources=(source,), derive_py=raise_for_array)

    with pytest.raises(ValueError, match="no array"):
        node.calculate()

    assert node.data.shape == (4,)
    assert numpy.array_equal(node.data, initial)


def test_a_parallel_derive_still_propagates():
    """Compiling the derive itself with `parallel` or `nogil` does not change the
    contract, including when the raise is inside its own `prange`."""
    def raise_inside_prange(a):
        acc = 0.0
        for i in prange(4):
            if a > 0.0:
                raise ValueError("parallel boom")
            acc += i
        return a + acc

    source = make_work_helper("source", 5.0)
    node = make_work_helper(
        "node", 99.0, sources=(source,), derive_py=raise_inside_prange,
        jit_options={"parallel": True},
    )
    with pytest.raises(ValueError, match="parallel boom"):
        node.calculate()
    assert node.data == 99.0
    assert node.derived == 0


def test_calculate_inside_a_prange_body_keeps_the_node_intact():
    """What escapes a parallel region is numba's business, and it varies by platform:
    on Linux the failure arrives as numba's `SystemError` carrying the original as
    `__cause__`, while on macOS nothing is raised at all. A derive cannot influence
    either. The invariant that must hold everywhere is that the node is not poisoned,
    so whatever the caller saw, `data` is intact and `derived` is unset."""
    source = make_work_helper("source", 5.0)
    node = make_work_helper("node", 99.0, sources=(source,), derive_py=_raise_when_positive)

    @njit(parallel=True)
    def calculate_in_parallel(work_):
        total = 0.0
        for i in prange(4):
            work_.calculate()
            total += work_.data + i
        return total

    try:
        calculate_in_parallel(node)
    except Exception as escaped:
        original = escaped if isinstance(escaped, ValueError) else escaped.__cause__
        assert isinstance(original, ValueError)
        assert "derive boom" in str(original)

    assert node.data == 99.0
    assert node.derived == 0


def test_a_succeeding_derive_is_unaffected():
    source = make_work_helper("source", -4.0)
    node = make_work_helper("node", 0.0, sources=(source,), derive_py=_raise_when_positive)
    node.calculate()
    assert node.data == -3.0
    assert node.derived == 1


def test_unicode_derive_still_works_when_it_succeeds():
    def make_text(a):
        return "value " + str(a)

    source = make_work_helper("source", -1.0)
    node = make_work_helper("node", "initial", sources=(source,), derive_py=make_text)
    node.calculate()
    assert node.data.startswith("value ")
