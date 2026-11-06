"""Tests for ``numbox.utils.derive_wap``, the function values whose exceptions propagate to jitted callers.

Many tests build ``Work`` nodes whose derive raises. They check that ``calculate`` propagates the exception with its
type and message, and that the failing node keeps its ``data`` with ``derived`` unset. The derives are ``cres`` values,
``CompileResultWAP`` values built with numba directly, ``@njit`` functions and the ``.as_func`` of a ``@proxy``
function, and the nodes are built from Python and from jitted code. Other tests check ``rewrap_derive``, the relation
between ``DeriveFunctionType`` and numba's plain ``FunctionType``, the references that the unboxing releases, and a
case in which the exception is lost. One test runs two subprocesses to check a cached caller that reads a ``cres``
value as a module-level constant. ``test/core/test_proxy.py`` tests the ``.as_func`` of a ``@proxy`` function in more
detail.
"""
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

import numbox.utils.derive_wap as derive_wap_module
from numbox.core.proxy.proxy import proxy
from numbox.utils.derive_wap import DeriveFunctionType, DeriveWAP, rewrap_derive
from numbox.core.work.work import make_work
from numbox.core.work.work_utils import make_work_helper
from numbox.utils.highlevel import cres


def _raise_when_positive(a):
    if a > 0.0:
        raise ValueError("derive boom")
    return a + 1.0


def test_cres_mints_a_derive_wap():
    """``cres`` returns a ``DeriveWAP`` with a nonzero ``jit_address``, and numba types it as ``DeriveFunctionType``.

    Jitted code calls a value of that type through its numba entry point, so an exception raised in it propagates.
    """
    @cres(float64(float64))
    def compiled(x):
        return x + 1.0

    assert isinstance(compiled, DeriveWAP)
    assert compiled.jit_address != 0
    from numba import typeof
    assert isinstance(typeof(compiled), DeriveFunctionType)


def test_raising_derive_propagates_and_leaves_the_node_recalculable():
    """A raising derive makes ``calculate`` raise its exception, and the node keeps its ``data`` with ``derived`` unset.

    The node therefore holds no wrong value, and a later ``calculate`` runs the derive again.
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
    """A later ``calculate`` of a node whose derive raised runs the derive again and succeeds with a new source value.

    The failed call leaves ``derived`` unset, so the later call does not return at once.
    """
    def raise_on_positive(a):
        if a[0] > 0.0:
            raise ValueError("derive boom")
        return a[0] + 1.0

    source = make_work_helper("source", numpy.array([5.0]))
    node = make_work_helper("node", 99.0, sources=(source,), derive_py=raise_on_positive)

    with pytest.raises(ValueError, match="derive boom"):
        node.calculate()
    assert node.derived == 0

    # The data property returns the source's array without a copy, so this assignment changes the source's value.
    source.data[0] = -3.0
    node.calculate()
    assert node.data == -2.0
    assert node.derived == 1


def test_unicode_payload_is_readable_after_a_failure():
    """After a failed ``calculate``, a node with string data still holds its initial string, which Python can read.

    A zero-filled string holds a null pointer, and reading one from Python ends the process with a segmentation fault.
    """
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
    """``make_work`` upgrades a ``CompileResultWAP`` built with numba directly, so its exception propagates.

    numba's ``CompileResultWAP`` records only the C entry point, which discards an exception.
    """
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
    """The wrapper that ``rewrap_derive`` returns stays alive while the caller holds the original value.

    A function value holds the wrapper's address in ``py_addr`` without a reference. ``rewrap_derive`` therefore stores
    the wrapper on the original value, so that no ``Work`` built from the wrapper points to freed memory.
    """
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
    """A jitted function returns the derive of a node built from a ``CompileResultWAP`` as a ``DeriveWAP``.

    Boxing the derive reads the Python object at ``py_addr``, so the call would read freed memory if the wrapper that
    ``make_work`` created had been freed.
    """
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
    """After a failed ``calculate``, a node with array data keeps its array, with the same shape and values.

    A zero-filled array is empty, so a node that stored one would fail the shape check.
    """
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
    """A derive compiled with ``parallel=True`` that raises in its own ``prange`` loop propagates its exception.

    numba runs that loop serially, as the comment below explains, so the test does not cover a loop that runs in
    parallel.
    """
    def raise_inside_prange(a):
        acc = 0.0
        # The raise is a second exit from the prange loop, so numba runs the loop serially and warns with
        # NumbaPerformanceWarning. No test covers a derive whose loop runs in parallel and calls a function that
        # raises. Such a derive returns normally, and calculate stores the incomplete result and sets derived, as the
        # work page describes under "Exceptions in parallel loops".
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
    """A derive that raises while ``calculate`` runs in a parallel loop leaves ``data`` and ``derived`` unchanged.

    What the call from Python raises depends on numba's threading layer and on the thread that ran the failing
    iteration, as the work page describes. An exception that escapes must be the ``ValueError`` or have it as its
    ``__cause__``, as numba's ``SystemError`` does. If nothing escapes, the test checks the node and then skips.
    """
    source = make_work_helper("source", 5.0)
    node = make_work_helper("node", 99.0, sources=(source,), derive_py=_raise_when_positive)

    @njit(parallel=True)
    def calculate_in_parallel(work_):
        total = 0.0
        for i in prange(4):
            work_.calculate()
            total += work_.data + i
        return total

    escaped_something = True
    try:
        calculate_in_parallel(node)
    except Exception as escaped:
        original = escaped if isinstance(escaped, ValueError) else escaped.__cause__
        assert isinstance(original, ValueError)
        assert "derive boom" in str(original)
    else:
        escaped_something = False

    assert node.data == 99.0
    assert node.derived == 0
    if not escaped_something:
        pytest.skip("nothing escaped the parallel region, so only the node invariant was checked")


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
    """A derive of the plain ``FunctionType`` whose value holds a numba entry point propagates its exception.

    The test covers the branch of ``_call_derive`` that tests ``jit_addr`` at run time, which numbox compiles only for a
    derive of the plain ``FunctionType``.
    """
    @njit(float64(float64))
    def raising(a):
        if a > 0.0:
            raise ValueError("plain boom")
        return a + 1.0

    @njit(float64(float64))
    def succeeding(a):
        return a + 1.0

    # The node is built in jitted code, so the type of its derive is the declared plain FunctionType. numba sets
    # jit_addr when it unboxes an @njit function passed as that type.
    @njit(float64(FunctionType(float64(float64))))
    def build_and_calculate(f):
        source = make_work("source", 5.0)
        node = make_work("node", 99.0, sources=(source,), derive=f)
        node.calculate()
        return node.data

    # Both calls run one compiled build_and_calculate, because its signature declares the plain FunctionType rather
    # than the type of either dispatcher.
    assert build_and_calculate(succeeding) == 6.0

    with pytest.raises(ValueError, match="plain boom"):
        build_and_calculate(raising)


def test_the_derive_type_stays_distinct_from_the_plain_function_type():
    """``DeriveFunctionType`` and the plain ``FunctionType`` of one signature are distinct types that compare unequal.

    numba interns types in a cache that would return one object for two equal types. The comment above
    ``DeriveFunctionType`` in ``numbox/utils/derive_wap.py`` describes what fails for each order of creation.
    """
    sig = float64(float64)
    plain = FunctionType(sig)
    derive = DeriveFunctionType(sig)

    assert type(derive) is DeriveFunctionType, (
        "DeriveFunctionType was interned onto another type; numbox's propagating "
        "convention is selected on this type and would be unreachable"
    )
    assert derive is not plain
    assert derive != plain, "an __eq__ making these equal collapses them in numba's type cache"
    assert plain != derive, "the reflected comparison must not collapse them either"


def test_a_container_of_two_derives_still_unifies():
    """A jitted function takes a tuple of two ``cres`` values of one signature and calls both.

    Below numba 0.68.0, numba refuses a tuple that mixes a ``DeriveWAP`` with another kind of function value of the same
    number of arguments, and ``test/core/test_proxy.py`` checks that case. This test checks a tuple of one kind, which
    numba accepts.
    """
    @cres(float64(float64))
    def first(x):
        return x + 1.0

    @cres(float64(float64))
    def second(x):
        return x + 2.0

    @njit
    def use_both(pair, x):
        return pair[0](x) + pair[1](x)

    assert use_both((first, second), 1.0) == 5.0


def test_a_cres_derive_is_accepted_where_a_plain_function_type_is_declared():
    """A ``cres`` value passed from Python to a parameter declared as the plain ``FunctionType`` is accepted and called.

    ``typeof`` returns ``DeriveFunctionType`` for the value, so the call relies on
    ``DeriveFunctionType.can_convert_to``.
    """
    @cres(float64(float64))
    def derive(x):
        return x + 1.0

    assert isinstance(derive, DeriveWAP)

    @njit(float64(FunctionType(float64(float64)), float64))
    def call_through_plain(f, x):
        return f(x)

    assert call_through_plain(derive, 2.0) == 3.0


def test_a_cres_derive_casts_to_plain_function_type_inside_jitted_scope():
    """A jitted function passes a ``cres`` value on to a function whose signature declares the plain ``FunctionType``.

    The cast happens in jitted code and relies on the ``lower_cast`` registration in ``numbox.utils.derive_wap``, which
    the conversion from Python in the test above does not use.
    """
    @cres(float64(float64))
    def derive(x):
        return x + 1.0

    @njit(float64(FunctionType(float64(float64)), float64))
    def declared_plain(f, x):
        return f(x)

    @njit
    def jitted_caller(f, x):
        return declared_plain(f, x)

    assert jitted_caller(derive, 2.0) == 3.0


def test_the_unbox_helper_releases_both_temporaries():
    """Unboxing a ``DeriveWAP`` releases both new references that ``_lower_get_derive_jit_address`` takes.

    The references are the module attribute that the unboxing calls and the ``Signature`` that it passes. numba unboxes
    a derive on every call that passes it as an argument, so a missing release leaks one reference per call.
    """
    @cres(float64(float64))
    def derive(x):
        return x + 1.0

    @njit
    def call_derive(f, x):
        return f(x)

    call_derive(derive, 1.0)  # compiles call_derive and its unboxing before the counts are taken

    # The unboxing looks _get_derive_jit_address up on the module at every call, so the spy below replaces it for the
    # counted calls. The spy records the Signature object that it receives. Only the unboxing takes references to the
    # spy during those calls, so the spy's count measures the release of the module attribute.
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
        call_derive(derive, 1.0)  # records the Signature object in captured
        signature_obj = captured[0]
        before_attr = sys.getrefcount(spy)
        before_sig = sys.getrefcount(signature_obj)
        for _ in range(calls):
            call_derive(derive, 1.0)
        attr_delta = sys.getrefcount(spy) - before_attr
        sig_delta = sys.getrefcount(signature_obj) - before_sig
    finally:
        derive_wap_module._get_derive_jit_address = original

    # The unboxing passes the same memoized Signature object on every call, as the list identical records. numba's
    # own lower_get_wrapper_address, which numbox calls for c_addr, takes a reference to that object and never releases
    # it, so the object's count grows by one per call. A missing release in numbox would add a second reference per
    # call and exceed the bound of one reference per call that the last assertion checks.
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
    """A ``cache=True`` caller that reads a ``cres`` value as a constant is cached, and the exception still propagates.

    The value is a module-level global, which numba reads as a constant. numbox lowers the constant with its numba entry
    point declared as a symbol, so the caller holds no address that changes between processes. The comments in
    ``lower_constant_derive_function_type`` explain why numba then caches the caller. Two subprocesses share one cache
    directory: the first compiles the caller and writes the cache, and the second loads the caller from it.
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

    # The cache directory starts empty. PYTHONPATH holds the sys.path of this process, so the subprocesses import the
    # same numbox.
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    first = _run_derive_probe(probe, env)      # compiles uses_const and writes the cache
    assert first == "-1.0 False 0 True", (
        f"cold run: expected value -1.0, no dynamic globals, 0 cache hits and a "
        f"propagated exception, got {first!r}. A baked `jit_addr` would report "
        f"dynamic globals and numba would refuse to cache the caller."
    )

    # A caller that compiled again in every process would also propagate and report no dynamic globals, so the second
    # process must report a cache hit.
    second = _run_derive_probe(probe, env)     # loads uses_const from the cache
    assert second == "-1.0 False 1 True", (
        f"warm run: expected a cache hit that still propagates, got {second!r}"
    )

    written = sorted(p.name for p in (tmp_path / "nbcache").rglob("*.nbi"))
    assert written, "no cache index was written, so the cache-hit count proves nothing"


def test_a_foreign_wrapper_reached_from_jitted_scope_still_discards():
    """A ``CompileResultWAP`` built with numba directly loses the derive's exception when jitted code builds the node.

    ``calculate`` returns normally, stores a zero-filled value and sets ``derived``, as in each of the three cases that
    the ``Work.calculate`` docstring lists. The test covers the branch of ``_call_derive`` that calls a derive through
    its C entry point.
    """
    jitted = njit(float64(float64))(_raise_when_positive)
    foreign = CompileResultWAP(jitted.get_compile_result(jitted.nopython_signatures[0]))

    @njit
    def build_and_calculate_in_jit(derive_):
        source = make_work("source", 5.0)
        node = make_work("node", 7.5, (source,), derive_)
        node.calculate()
        return node.data, node.derived

    data, derived = build_and_calculate_in_jit(foreign)
    assert data == 0.0, (
        f"expected the discarded-exception zero fill, got {data!r}. A non-zero value here "
        f"means the derive was upgraded after all and this test no longer covers the arm."
    )
    assert derived == 1


def test_a_proxy_binding_reached_from_jitted_scope_propagates():
    """A node that jitted code builds from a ``@proxy`` function's ``.as_func`` propagates the derive's exception.

    In jitted code, ``make_work`` uses the derive unchanged, so the result depends on ``.as_func`` being a
    ``DeriveWAP``. A plain ``CompileResultWAP`` would be called through its C entry point, and ``calculate`` would
    store a zero-filled value. No other test builds a ``Work`` from a ``.as_func``.
    """
    @proxy(float64(float64))
    def raising_binding(x):
        if x > 0.0:
            raise ValueError("proxy boom")
        return x + 1.0

    @njit
    def build_in_jitted_scope(derive_):
        source = make_work("source", 5.0)
        return make_work("node", 99.0, (source,), derive_)

    node = build_in_jitted_scope(raising_binding.as_func)

    with pytest.raises(ValueError, match="proxy boom"):
        node.calculate()

    assert node.data == 99.0, "`data` must be untouched by a failed derive"
    assert node.derived == 0, "`derived` must stay unset so the node can be calculated again"


def test_propagation_through_a_multi_level_chain():
    """A chain of three derived nodes propagates the third node's exception, and the first two keep their results.

    The failing node keeps its ``data`` with ``derived`` unset. The other graphs in this file have two nodes.
    """
    source = make_work_helper("source", -1.0)
    level1 = make_work_helper("level1", 0.0, sources=(source,), derive_py=_raise_when_positive)
    level2 = make_work_helper("level2", 0.0, sources=(level1,), derive_py=_raise_when_positive)
    level3 = make_work_helper("level3", 99.0, sources=(level2,), derive_py=_raise_when_positive)

    with pytest.raises(ValueError, match="derive boom"):
        level3.calculate()

    assert level1.data == 0.0 and level1.derived == 1, "a level that succeeded was rolled back"
    assert level2.data == 1.0 and level2.derived == 1, "a level that succeeded was rolled back"
    assert level3.data == 99.0 and level3.derived == 0, "the failing node was poisoned"
