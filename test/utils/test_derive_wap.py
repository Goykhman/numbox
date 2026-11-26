"""Tests for ``numbox.utils.derive_wap``, the function values whose exceptions propagate to jitted callers.

Many tests build ``Work`` nodes whose derive raises. They check that ``calculate`` propagates the exception with its
type and message, and that the failing node keeps its ``data`` with ``derived`` unset. The derives are ``cres`` values,
``CompileResultWAP`` values built with numba directly, ``@njit`` functions and the ``.as_func`` of a ``@proxy``
function, and the nodes are built from Python and from jitted code. Other tests check ``rewrap_derive``, the relation
between ``DeriveFunctionType`` and numba's plain ``FunctionType``, the references that the unboxing releases, and a
case in which the exception is lost. One test runs two subprocesses to check a cached caller that reads a ``cres``
value as a module-level constant. Further tests check the alias under which ``cres`` publishes a function for such a
caller. They cover what the caller declares, two functions that compute one alias, a value without an alias, and a
caller that numba compiles ahead of time. ``test/core/test_proxy.py`` tests the ``.as_func`` of a ``@proxy`` function
in more detail. ``test/core/test_proxy_cache_stale.py`` tests the repair of a cached caller of a constant after an
edit.
"""
import ctypes
import gc
import os
import re
import subprocess
import sys
import textwrap
import warnings
import weakref
from collections import namedtuple

import numpy
import pytest
from llvmlite import binding as ll
from llvmlite import ir
from numba import njit, objmode, prange, typeof
from numba.core import types
from numba.core.cpu import CPUContext
from numba.core.registry import cpu_target
from numba.core.types import FunctionType, float64
from numba.core.types.function_type import CompileResultWAP

import numbox.utils.derive_wap as derive_wap_module
from numbox.core.proxy.proxy import AliasCollisionWarning, _is_shared_jit_name, proxy
from numbox.utils.derive_wap import DeriveFunctionType, DeriveWAP, rewrap_derive
from numbox.core.work.work import make_work
from numbox.core.work.work_utils import make_work_helper
from numbox.utils.highlevel import cres
from test.auxiliary_utils import open_libm


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


def test_cres_publishes_the_numba_entry_point_under_an_alias():
    """A ``cres`` value carries an alias for its numba entry point, and a caller of the constant declares that alias.

    The alias is registered with llvmlite and resolves to the address that the value recorded. The caller's LLVM IR
    must declare the alias. If the function's library were linked into the caller instead, the IR would hold numba's
    own name for the entry point, and the caller's cached code would import no name that the cache check could test.
    The value has no ``_numbox_proxy_alias``, so the fingerprint of a function that captures the value still comes from
    the value's Python function.
    """
    @cres(float64(float64))
    def tripled(x):
        return 3.0 * x

    alias = tripled._numbox_jit_alias
    assert re.fullmatch(r"numbox_pxy_tripled_[0-9a-f]{16}_jit", alias), alias
    assert ll.address_of_symbol(alias) == tripled.jit_address, "the alias does not resolve to the numba entry point"
    assert not hasattr(tripled, "_numbox_proxy_alias")

    @njit(float64(float64))
    def call_constant(x):
        return tripled(x)

    assert call_constant(2.0) == 6.0
    llvm_ir = next(iter(call_constant.inspect_llvm().values()))
    assert re.search(rf"declare [^\n]*@{alias}\(", llvm_ir), "the caller of the constant does not declare the alias"


def _shape_scalar(x):
    return x + 1.0


def _shape_pair(x, n):
    return (x, x + n)


def _shape_total(a):
    return a.sum()


def _shape_text(s):
    return s + "!"


def _shape_bump(a):
    a[0] += 1.0


@pytest.mark.parametrize("func, sig, args, expected", [
    (_shape_scalar, float64(float64), (2.0,), 3.0),
    (_shape_pair, types.UniTuple(float64, 2)(float64, types.int64), (2.0, 3), (2.0, 5.0)),
    (_shape_total, float64(types.Array(float64, 1, "C")), (numpy.arange(4.0),), 6.0),
    (_shape_text, types.unicode_type(types.unicode_type), ("hey",), "hey!"),
    (_shape_bump, types.void(types.Array(float64, 1, "C")), (numpy.zeros(2),), None),
], ids=["scalar", "tuple", "array", "text", "void"])
def test_a_caller_of_a_constant_declares_the_entry_point_with_the_type_of_its_definition(func, sig, args, expected):
    """The declaration in a caller of a constant has the LLVM function type of the function's definition.

    numbox builds the declaration from the compile result's types and links nothing in, so no linker compares the
    declaration with the definition. A numba release that changed how an entry point receives an argument or returns a
    value could make the two differ, and the call would then read its arguments from the wrong places. The test
    compares the two types for several kinds of argument and return value, and it checks the result of each call.
    """
    value = cres(sig)(func)
    if len(args) == 1:
        @njit
        def call(a):
            return value(a)
    else:
        @njit
        def call(a, b):
            return value(a, b)

    assert call(*args) == expected

    caller_module = ll.parse_assembly(next(iter(call.inspect_llvm().values())))
    body_module = ll.parse_assembly(next(iter(value._numbox_dispatcher.inspect_llvm().values())))
    declared = caller_module.get_function(value._numbox_jit_alias)
    defined = body_module.get_function(value.cres.fndesc.llvm_func_name)
    assert declared.is_declaration and not defined.is_declaration
    assert str(declared.global_value_type) == str(defined.global_value_type)


def _shape_double(n):
    return 2 * n


def _extension_attributes(function):
    """Return, for each argument of an LLVM function, which of the attributes ``signext`` and ``zeroext`` it carries."""
    return [sorted({b"signext", b"zeroext"} & set(argument.attributes)) for argument in function.arguments]


@pytest.mark.skipif(
    not hasattr(CPUContext, "apply_target_attributes"), reason="numba 0.63 has no hook for target attributes"
)
def test_a_caller_of_a_constant_declares_the_entry_point_with_the_target_attributes_of_its_definition(monkeypatch):
    """The declaration in a caller of a constant carries the target-specific attributes of the function's definition.

    Since numba 0.64, ``declare_function`` passes each entry point to the target context's
    ``apply_target_attributes``. On s390x, that hook marks every integer argument and result of fewer than 64 bits with
    ``signext`` or ``zeroext``, because the platform's calling convention requires the caller to extend such a value.
    A caller whose declaration lacks the attribute may pass a 32-bit argument with undefined upper bits. The hook does
    nothing on the platforms that run the tests. The test therefore installs a hook that marks every narrow integer
    argument with ``signext``, and it compares these attributes of the declaration with those of the definition.
    """
    def mark_narrow_integers(self, llvm_func, argtypes=None, restype=None):
        for argument in llvm_func.args:
            if isinstance(argument.type, ir.IntType) and argument.type.width < 64:
                argument.add_attribute("signext")

    monkeypatch.setattr(CPUContext, "apply_target_attributes", mark_narrow_integers)
    value = cres(types.int64(types.int32))(_shape_double)

    @njit
    def call(n):
        return value(n)

    assert call(numpy.int32(21)) == 42

    caller_module = ll.parse_assembly(next(iter(call.inspect_llvm().values())))
    body_module = ll.parse_assembly(next(iter(value._numbox_dispatcher.inspect_llvm().values())))
    declared = caller_module.get_function(value._numbox_jit_alias)
    defined = body_module.get_function(value.cres.fndesc.llvm_func_name)
    assert declared.is_declaration and not defined.is_declaration
    marked = _extension_attributes(defined)
    assert [b"signext"] in marked, "the hook marked nothing, so the comparison below proves nothing"
    assert _extension_attributes(declared) == marked


def test_a_caller_of_a_constant_declares_the_entry_point_with_the_argument_attributes_of_numba_s_declaration():
    """The declaration in a caller of a constant carries the argument attributes of numba's own declaration.

    numba's ``declare_function`` marks the pointer for the result and the pointer for the exception data as
    ``noalias`` and as not captured, and LLVM uses both attributes when it optimizes the caller. numbox builds the
    declaration under the alias without ``declare_function``. The test therefore compares the attributes of that
    declaration with those of a declaration that numba creates for the same entry point.
    """
    value = cres(types.int64(types.int32))(_shape_double)

    @njit
    def call(n):
        return value(n)

    assert call(numpy.int32(21)) == 42

    caller_module = ll.parse_assembly(next(iter(call.inspect_llvm().values())))
    declared = caller_module.get_function(value._numbox_jit_alias)
    numba_module = ir.Module()
    cpu_target.target_context.declare_function(numba_module, value.cres.fndesc)
    expected = ll.parse_assembly(str(numba_module)).get_function(value.cres.fndesc.mangled_name)
    attributes = [sorted(argument.attributes) for argument in expected.arguments]
    assert any(attributes), "numba marked no argument, so the comparison below proves nothing"
    assert [sorted(argument.attributes) for argument in declared.arguments] == attributes


def test_a_function_compiled_twice_by_cres_gets_a_name_of_its_own_and_no_name_is_repointed():
    """A second ``cres`` function that computes an alias gets a name of its own, without a collision warning.

    A module reload compiles each ``cres`` function again, and so does a factory called twice with equal arguments.
    The first name must keep resolving to the first compilation. A caller that reads a value as a constant holds no
    copy of the function, so a name pointed at a second compilation would make a later caller of the first value run
    the second. Both names are retired, because another process may compile the two in the other order. ``cres`` warns
    about none of this at decoration: most ``cres`` functions are never read as constants, and only a cached caller
    that reads one is affected.
    """
    def make():
        @cres(float64(float64))
        def body(x):
            return x + 7.0
        return body

    with warnings.catch_warnings():
        warnings.simplefilter("error", AliasCollisionWarning)
        first = make()
        second = make()

    shared, own = first._numbox_jit_alias, second._numbox_jit_alias
    assert re.fullmatch(re.escape(shared) + r"_\d+", own), (shared, own)
    assert ll.address_of_symbol(shared) == first.jit_address, "the first name was pointed at the second compilation"
    assert ll.address_of_symbol(own) == second.jit_address
    assert _is_shared_jit_name(shared) and _is_shared_jit_name(own), (
        "a name that two functions share still passes the cache check, so a warm caller can be served the wrong one"
    )


def test_one_function_compiled_by_cres_under_two_sets_of_options_claims_two_aliases():
    """One function that ``cres`` compiles under two sets of options has two aliases, and neither is retired.

    The options change the compiled code, so the two compilations are two functions to a caller. With one alias for
    both, the second would receive a name used only in this process, and a cached caller that holds either as a
    constant would compile again in every process.
    """
    def body(x):
        return x / 3.0

    python_model = cres(float64(float64))(body)
    numpy_model = cres(float64(float64), error_model="numpy")(body)
    aliases = {python_model._numbox_jit_alias, numpy_model._numbox_jit_alias}
    assert all(re.fullmatch(r"numbox_pxy_body_[0-9a-f]{16}_jit", alias) for alias in aliases), aliases
    assert len(aliases) == 2, "two sets of options computed one alias"
    assert not any(_is_shared_jit_name(alias) for alias in aliases)


def test_two_functions_that_the_alias_cannot_tell_apart_each_run_their_own_code_as_constants():
    """Two different functions with equal alias inputs each run their own code when read as constants.

    The two tuple classes are both named ``Row`` and hold two floats, so the two signatures print alike, and the two
    accessors have the same code and read no global. Everything that the alias covers is equal, and numbox cannot tell
    the second accessor from the first one compiled again. The fields are in a different order, so the accessors read
    different positions. The caller of the first accessor is compiled after the second accessor exists. If the second
    registration had taken over the first one's name, that caller would return field ``b``. The second accessor has no
    caller here, because numba gives two function types that print alike one type, the first.
    """
    row_ab = namedtuple("Row", ["a", "b"])
    row_ba = namedtuple("Row", ["b", "a"])

    def make(row_type):
        @cres(float64(row_type))
        def get_a(row):
            return row.a
        return get_a

    type_ab, type_ba = typeof(row_ab(1.0, 2.0)), typeof(row_ba(1.0, 2.0))
    assert str(float64(type_ab)) == str(float64(type_ba)), "the scenario needs two signatures that print alike"
    get_a_ab = make(type_ab)
    get_a_ba = make(type_ba)
    assert get_a_ba._numbox_jit_alias.startswith(get_a_ab._numbox_jit_alias + "_"), (
        "the two accessors did not compute one alias, so the test would pass without exercising the rule"
    )

    @njit
    def read_ab(row):
        return get_a_ab(row)

    assert read_ab(row_ab(a=1.0, b=2.0)) == 1.0, "the caller of the first accessor ran the second one"


def test_two_cres_values_over_unfingerprintable_pointers_each_run_their_own_body_as_constants():
    """Two ``cres`` functions that differ only in a value with no fingerprint each run their own code as constants.

    The two functions close over ctypes pointers built from raw addresses, which have no fingerprint, so both compute
    one alias. The second value must carry a name of its own. With the shared name, its caller would run the first
    function and return ``floor(2.5)`` instead of the expected ``ceil(2.5)``. ``cres`` does not warn at decoration.
    """
    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take two distinct symbol addresses from")
    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

    def make(symbol):
        fp = proto(ctypes.cast(getattr(libm, symbol), ctypes.c_void_p).value)

        @cres(float64(float64))
        def body(x):
            return fp(x)
        return body

    with warnings.catch_warnings():
        warnings.simplefilter("error", AliasCollisionWarning)
        floor_value = make("floor")
        ceil_value = make("ceil")

    @njit(float64(float64))
    def const_floor(x):
        return floor_value(x)

    @njit(float64(float64))
    def const_ceil(x):
        return ceil_value(x)

    assert const_floor(2.5) == 2.0
    assert const_ceil(2.5) == 3.0, "the second function was lowered against the first one's entry point"
    assert floor_value._numbox_jit_alias != ceil_value._numbox_jit_alias
    assert _is_shared_jit_name(floor_value._numbox_jit_alias) and _is_shared_jit_name(ceil_value._numbox_jit_alias), (
        "an alias still identifies a function, so a warm caller of the constant can be served the wrong one"
    )


# Two values that jitted code cannot hold and that an object-mode block can read.
_REORDERED_TABLE = numpy.array([(1.0, 10.0), (2.0, 20.0)], dtype=[("a", "f8"), ("b", "f8")])[["b", "a"]]
_HUGE_INT = 7 ** 6000


def test_cres_compiles_a_function_whose_object_mode_block_reads_values_that_numba_cannot_type():
    """``cres`` computes a function's alias from every global that it reads, including one read only in object mode.

    numba does not type such a global, so its value can be one that jitted code could not hold. The first value here
    is a structured array whose fields are not in the order of their offsets, so NumPy defines no ``descr`` for it.
    The second is an int for which ``repr`` raises ``ValueError`` under the interpreter's default digit limit.
    ``cres`` must compute an alias for a function that reads them, and the function must run.
    """
    @cres(float64(float64))
    def lookup(x):
        with objmode(y="float64"):
            row = _REORDERED_TABLE[int(x)]
            y = float(row["a"] + row["b"] + _HUGE_INT % 5)
        return y

    assert lookup._numbox_jit_alias.endswith("_jit")
    assert lookup(1.0) == 2.0 + 20.0 + 1.0


def _ahead_of_time_ir(func):
    """Return the LLVM IR of ``func`` compiled for ``float64(float64)`` as ``numba.pycc`` compiles an exported function.

    The compilation stops at the IR of the finalized library, so it needs no C compiler. The calling test is skipped if
    this numba has no ahead-of-time code generator.
    """
    from numba.core.compiler import Flags, compile_extra
    from numba.core.compiler_lock import global_compiler_lock
    from numba.core.registry import cpu_target

    if not hasattr(cpu_target.target_context, "with_aot_codegen"):
        pytest.skip("this numba has no ahead-of-time code generator")

    flags = Flags()
    flags.no_compile = True
    flags.no_cpython_wrapper = True
    flags.no_cfunc_wrapper = True
    with global_compiler_lock:
        target_context = cpu_target.target_context.with_aot_codegen("aot_probe")
        library = target_context.codegen().create_library("aot_probe")
        compile_extra(
            cpu_target.typing_context, target_context, func, (float64,), float64, flags, locals={}, library=library,
        )
        library.finalize()
        return library.get_llvm_str()


def test_a_module_compiled_ahead_of_time_declares_no_alias():
    """A caller compiled for numba's ahead-of-time target has the constant's code linked in and names no alias.

    numbox registers an alias with llvmlite in the running process. The loader of a module that ``numba.pycc`` compiled
    knows nothing of that table, so a module that declared an alias would fail to import with an undefined symbol.
    """
    @cres(float64(float64))
    def scale(x):
        return x * 2.0

    def twice(x):
        return scale(x) + scale(x)

    llvm_ir = _ahead_of_time_ir(twice)

    assert "twice" in llvm_ir, "the library holds no caller, so the test would pass without exercising the lowering"
    assert "numbox_pxy_" not in llvm_ir, "the ahead-of-time module refers to a name that only this process registers"


def test_an_ahead_of_time_module_that_calls_a_jitted_holder_of_the_constant_declares_the_alias():
    """A jitted function that holds the constant brings the alias into an ahead-of-time module that calls it.

    ``numba.pycc`` compiles only the exported function for its ahead-of-time target. A jitted function that the
    exported function calls is compiled in the same way as for the running process, so it declares the alias, and the
    module that links it in cannot be imported. The pages on ``cres`` and ``@proxy`` state this limit and the way
    around it: the exported function reads the value itself and passes it on as an argument, and the module then names
    no alias.
    """
    @cres(float64(float64))
    def scale(x):
        return x * 2.0

    @njit
    def holder(x):
        return scale(x) + 1.0

    @njit
    def taker(func, x):
        return func(x) + 1.0

    def through_the_holder(x):
        return holder(x)

    def through_an_argument(x):
        return taker(scale, x)

    assert "numbox_pxy_" in _ahead_of_time_ir(through_the_holder)
    assert "numbox_pxy_" not in _ahead_of_time_ir(through_an_argument)


def test_a_rewrapped_wrapper_has_no_alias_and_is_linked_into_a_caller_of_the_constant():
    """A value that ``rewrap_derive`` returns has no alias, and a caller that reads it as a constant links its body in.

    numbox computes an alias only for a function that ``cres`` or ``@proxy`` compiled. For any other ``DeriveWAP``, the
    constant lowering declares numba's own name for the entry point and links the body's library into the caller. The
    call must still return the body's result and propagate its exception, and the caller's IR must name no numbox
    alias.
    """
    jitted = njit(float64(float64))(_raise_when_positive)
    upgraded = rewrap_derive(CompileResultWAP(jitted.get_compile_result(jitted.nopython_signatures[0])))
    assert not hasattr(upgraded, "_numbox_jit_alias")

    @njit(float64(float64))
    def call_constant(x):
        return upgraded(x)

    assert call_constant(-2.0) == -1.0
    with pytest.raises(ValueError, match="derive boom"):
        call_constant(3.0)
    assert "numbox_pxy_" not in next(iter(call_constant.inspect_llvm().values()))


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
