import gc
import os
import subprocess
import sys
import textwrap
import weakref

import numpy
import pytest
from numba import njit, prange
from numba.core.types import FunctionType, float64
from numba.core.types.function_type import CompileResultWAP

import numbox.core.work.derive_wap as derive_wap_module
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


def test_a_plain_function_type_derive_propagates_through_the_runtime_branch():
    """A derive field typed as a plain ``FunctionType`` must still propagate when
    the value behind it carries an entry point.

    ``rewrap_derive`` can only upgrade from Python scope, so a jitted caller that
    builds its own ``Work`` fixes the field at whatever type it declared. numba's
    unboxing populates ``jit_addr`` for an njit dispatcher passed as a
    ``FunctionType``-typed argument, so the value can propagate even though the
    numbox-owned type is nowhere in sight, which is why ``_call_derive`` tests the
    slot at run time instead of selecting purely on the field type.

    Deleting that runtime branch passes the rest of the suite: every other
    ``_call_derive`` typing in it is a ``DeriveFunctionType``, so the branch is not
    merely unasserted, it is never compiled.
    """
    @njit(float64(float64))
    def raising(a):
        if a > 0.0:
            raise ValueError("plain boom")
        return a + 1.0

    @njit(float64(float64))
    def succeeding(a):
        return a + 1.0

    @njit(float64(FunctionType(float64(float64))))
    def build_and_calculate(f):
        source = make_work("source", 5.0)
        node = make_work("node", 99.0, sources=(source,), derive=f)
        node.calculate()
        return node.data

    # Same compiled function for both, because the declared argument type is the
    # plain FunctionType rather than either dispatcher.
    assert build_and_calculate(succeeding) == 6.0

    with pytest.raises(ValueError, match="plain boom"):
        build_and_calculate(raising)


def test_a_cres_derive_is_accepted_where_a_plain_function_type_is_declared():
    """``DeriveFunctionType`` has to remain substitutable for the plain
    ``FunctionType`` of the same signature.

    ``typeof`` on a cres derive now yields the numbox-owned type, so an explicitly
    typed ``njit`` that names ``FunctionType`` would stop accepting it without the
    conversion, which is a break in code that predates this change. Disabling
    ``can_convert_to`` passes the whole suite while breaking exactly this call.
    """
    @cres(float64(float64))
    def derive(x):
        return x + 1.0

    assert isinstance(derive, DeriveWAP)

    @njit(float64(FunctionType(float64(float64)), float64))
    def call_through_plain(f, x):
        return f(x)

    assert call_through_plain(derive, 2.0) == 3.0


def test_the_unbox_helper_releases_both_temporaries():
    """Unboxing must not leak a reference per call.

    ``_lower_get_derive_jit_address`` takes two new references, the module
    attribute it calls and the unserialized ``Signature`` it passes, and unboxing
    runs on every call for a derive passed as an argument, so a missing release
    grows without bound. Both are checked separately, because dropping either
    decref alone leaks one per call and the whole suite otherwise passes.

    The module attribute is looked up at run time on each call, so replacing it
    with a delegating spy both intercepts the ``Signature`` numba unserializes
    and gives the attribute probe an object nothing else holds.

    The two probes are not symmetric. Nothing but this unboxer touches the spy,
    so its release is pinned exactly. The ``Signature`` is a single object numba
    memoizes and hands to ``lower_get_wrapper_address`` as well, which numbox
    calls for ``c_addr`` and which never releases it, so one reference per call
    is a floor numba imposes and numbox cannot remove. Failing to release the
    numbox-side reference would double that, which is what the ceiling catches.
    """
    @cres(float64(float64))
    def derive(x):
        return x + 1.0

    @njit
    def call_derive(f, x):
        return f(x)

    call_derive(derive, 1.0)  # compile the unboxer before measuring

    original = derive_wap_module._get_derive_jit_address
    captured, identical = [], []

    def spy(func, sig):
        if captured:
            identical.append(sig is captured[0])
        else:
            captured.append(sig)
        return original(func, sig)

    calls = 200
    derive_wap_module._get_derive_jit_address = spy
    try:
        call_derive(derive, 1.0)  # capture the signature object
        signature_obj = captured[0]
        before_attr = sys.getrefcount(spy)
        before_sig = sys.getrefcount(signature_obj)
        for _ in range(calls):
            call_derive(derive, 1.0)
        attr_delta = sys.getrefcount(spy) - before_attr
        sig_delta = sys.getrefcount(signature_obj) - before_sig
    finally:
        derive_wap_module._get_derive_jit_address = original

    assert identical and all(identical), (
        "numba handed a fresh Signature per call, so the refcount probe below "
        "measures nothing"
    )
    assert attr_delta == 0, (
        f"the module attribute leaked {attr_delta} references over {calls} unboxing "
        f"calls; the `decref(fn)` in _lower_get_derive_jit_address is missing"
    )
    assert sig_delta <= calls, (
        f"the unserialized Signature gained {sig_delta} references over {calls} "
        f"unboxing calls, above the one per call numba's own "
        f"lower_get_wrapper_address contributes on the same memoized object; "
        f"the `decref(sig_obj)` in _lower_get_derive_jit_address is missing"
    )


def _run_derive_probe(probe, env):
    r = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert r.returncode == 0, f"probe failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout.strip()


def test_a_cached_caller_of_a_derive_caches_and_still_propagates(tmp_path):
    """Lowering ``jit_addr`` as a symbol rather than a baked address is what lets
    a caller of a constant-reached derive be cached at all, and the exception
    contract has to survive the cache hit.

    A baked address is a dynamic global, so numba refuses to cache the caller and
    writes nothing. Declaring the entry point symbolically leaves ``c_addr`` and
    ``py_addr`` dead, they are eliminated before numba scans the final module, and
    the caller caches. Reverting the constant lowering to a baked address
    otherwise passes the entire suite.

    Both processes are asserted, because a caller that merely recompiled every
    time would still propagate and would still report no dynamic globals.
    """
    probe = tmp_path / "derive_cache_probe.py"
    probe.write_text(textwrap.dedent('''
        from numba import njit
        from numba.core.types import float64
        from numbox.utils.highlevel import cres

        @cres(float64(float64))
        def const_derive(x):
            if x > 0.0:
                raise ValueError("derive boom")
            return x + 1.0

        @njit(cache=True)
        def uses_const(x):
            return const_derive(x)

        value = uses_const(-2.0)

        propagated = False
        try:
            uses_const(3.0)
        except ValueError:
            propagated = True

        dynamic = any(c.library.has_dynamic_globals for c in uses_const.overloads.values())
        hits = sum(uses_const.stats.cache_hits.values())
        print(f"{value} {dynamic} {hits} {propagated}")
    '''), encoding="utf-8")

    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    first = _run_derive_probe(probe, env)      # cold: compiles and writes the cache
    assert first == "-1.0 False 0 True", (
        f"cold run: expected value -1.0, no dynamic globals, 0 cache hits and a "
        f"propagated exception, got {first!r}. A baked `jit_addr` would report "
        f"dynamic globals and numba would refuse to cache the caller."
    )

    second = _run_derive_probe(probe, env)     # warm: must be served from the cache
    assert second == "-1.0 False 1 True", (
        f"warm run: expected a cache hit that still propagates, got {second!r}"
    )

    written = sorted(p.name for p in (tmp_path / "nbcache").rglob("*.nbi"))
    assert written, "no cache index was written, so the cache-hit count proves nothing"
