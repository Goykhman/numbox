import importlib
import math
import os
import re
import subprocess
import sys
import textwrap
import threading
import warnings

import numpy as np
import pytest
from numba import float64, njit, typeof
from numba.core.errors import TypingError
from numba.core.types import FunctionType, Omitted
from numba.core.types.function_type import CompileResultWAP

from numbox.core.bindings.errno import errno_get
from numbox.core.bindings.libc import getenv, memcpy
from numbox.core.bindings.call import _call_lib_func
from numbox.core.proxy.proxy import proxy, proxy_if_available, make_proxy_name
from numbox.utils.derive_wap import DeriveFunctionType, DeriveWAP, jit_addr_supported
from numbox.utils.lowlevel import array_data_p, get_unicode_data_p
from test.auxiliary_utils import (
    assert_njit_cache_survives_subprocess_roundtrip,
    collect_and_run_tests,
    open_libm,
)


aux_1_sig = float64(float64)


@proxy(aux_1_sig, jit_options={'cache': True})
def aux_1(x):
    return 3.14 * x


def test_1():
    assert abs(aux_1(2.2) - 3.14 * 2.2) < 1e-15
    llvm_ir = next(iter(aux_1.inspect_llvm().values()))
    assert aux_1.__name__ == make_proxy_name('aux_1')
    if 'numbox_pxy_aux_1' in llvm_ir:
        alias_name = r"double @numbox_pxy_aux_1_\w+\(double"  # noqa: W605
        assert len(re.findall(f"declare {alias_name}", llvm_ir)) == 1
        assert len(re.findall(f"call {alias_name}", llvm_ir)) == 1
    else:
        print(f"LLVM inspection disabled for cached code, {aux_1.__name__}")


aux_2_sig = [float64(float64, float64), float64(float64, Omitted(1.3))]


@proxy(aux_2_sig, jit_options={'cache': True})
def aux_2(x, *, y=1.3):
    return 3.14 * x + y


def test_2():
    assert abs(aux_2(2.2) - (3.14 * 2.2 + 1.3)) < 1e-15
    assert abs(aux_2(2.2, 1.4) - (3.14 * 2.2 + 1.4)) < 1e-15


def _sole_compile_result(dispatcher):
    """Return the single compiled result on a numba dispatcher."""
    sigs = dispatcher.nopython_signatures
    assert len(sigs) == 1, sigs
    return dispatcher.get_compile_result(sigs[0])


def test_proxy_zero_arg_caller_is_cacheable():
    @njit(cache=True)
    def caller():
        return errno_get()
    caller()
    assert not _sole_compile_result(caller).library.has_dynamic_globals


def test_proxy_single_arg_caller_is_cacheable():
    @njit(cache=True)
    def caller(name_p):
        return getenv(name_p)
    caller(get_unicode_data_p("NUMBOX_NONEXISTENT_XYZZY"))
    assert not _sole_compile_result(caller).library.has_dynamic_globals


def test_proxy_multi_arg_caller_is_cacheable():
    @njit(cache=True)
    def caller(dst, src):
        memcpy(array_data_p(dst), array_data_p(src), src.nbytes)
    caller(np.zeros(4, dtype=np.uint8), np.arange(4, dtype=np.uint8))
    assert not _sole_compile_result(caller).library.has_dynamic_globals


def test_proxy_caller_survives_subprocess_round_trip(tmp_path):
    """Real cross-process cache survival test for @proxy-decorated bindings.

    The heuristic tests above (``has_dynamic_globals is False``) only prove
    cache *eligibility*. This test proves cache *correctness*: a caller
    compiled with ``@njit(cache=True)`` against an ``@proxy`` binding
    actually round-trips through the on-disk cache (.nbi/.nbc files), with
    the second process loading the cached IR and producing identical output
    to the cold-cache first process — and neither file is rewritten on the
    warm run (mtimes preserved).

    ``proxy`` declares a process-stable alias for the callee's cfunc wrapper
    (registered via ``llvmlite.binding.add_symbol``) as an extern in the caller's IR module;
    llvmlite's JIT linker resolves the alias per process at cache reload, so
    cached IR survives ASLR across processes without baking in any runtime
    address. See the ``assert_njit_cache_survives_subprocess_roundtrip`` helper
    in ``test/auxiliary_utils.py`` for the full assertion contract, and
    ``test_proxy_referenced_symbol_is_process_stable`` for why the alias (not
    numba's process-local ``v<uid>`` wrapper name) is what keeps concurrently
    built caches consistent.
    """
    assert_njit_cache_survives_subprocess_roundtrip(
        tmp_path,
        probe_source="""
            from numba import njit
            from numbox.core.bindings.errno import errno_get, errno_set

            @njit(cache=True)
            def caller():
                errno_set(42)
                return errno_get()

            v = caller()
            assert v == 42, f"got {v!r}"
            print(v)
        """,
        expected_stdout_lines=["42"],
    )


def test_proxy_referenced_symbol_is_process_stable(tmp_path):
    """A caller must bake the *same* callee symbol regardless of compile order.

    Regression for the concurrent-cache hazard: ``proxy`` references each body's
    cfunc wrapper by a deterministic alias registered via ``llvmlite.binding.add_symbol``,
    not numba's process-local ``v<uid>`` wrapper name. If it regressed to the uid
    name, two processes that compiled a different number of functions first would
    bake different symbols into otherwise-equal cached objects, so a
    concurrently-built shared cache could pair a body defining ``v<Na>`` with a
    caller referencing ``v<Nb>`` and abort on load with
    ``LLVM ERROR: Symbol not found: cfunc...``. We run a probe twice with a
    different number of warm-up compiles and assert the baked callee symbol is
    identical (and is the stable alias).
    """
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent('''
        import sys
        from numba import njit
        from numba.core import types
        from numbox.core.proxy.proxy import proxy

        def d0(x): return x
        def d1(x): return x
        def d2(x): return x
        def d3(x): return x
        def d4(x): return x

        for _f in (d0, d1, d2, d3, d4)[:int(sys.argv[1])]:
            njit(types.int64(types.int64))(_f)(0)

        @proxy(types.int64(types.int64))
        def binding(x):
            return x + 1

        @njit
        def caller(x):
            return binding(x)

        caller(0)
        ir = "\\n".join(caller.inspect_llvm().values())
        toks = set()
        for tok in ir.replace("(", " ").replace(")", " ").replace("*", " ").split():
            if tok.startswith("@") and "numbox_pxy_" in tok:
                toks.add(tok.strip('@"'))
        print("|".join(sorted(toks)))
    '''), encoding="utf-8")

    def _run(prior):
        r = subprocess.run(
            [sys.executable, str(probe), str(prior)],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert r.returncode == 0, f"probe failed (prior={prior}):\n{r.stderr}"
        return r.stdout.strip()

    baseline = _run(0)
    shifted = _run(5)
    assert baseline, "no callee symbol found in caller IR"
    assert baseline == shifted, (
        "@proxy baked a process-dependent callee symbol (concurrent-cache hazard):\n"
        f"  prior=0: {baseline!r}\n  prior=5: {shifted!r}"
    )
    assert "numbox_pxy_" in baseline, f"expected a stable add_symbol alias, got {baseline!r}"


def test_proxy_if_available_present_symbol_returns_real_proxy():
    """When the C symbol is present, ``proxy_if_available`` returns a
    real ``@proxy``-wrapped dispatcher with ``.as_func`` attached, of the
    same numba-version-dependent type ``proxy`` hands out: a ``DeriveWAP``
    where the ``jit_addr`` slot exists, a plain ``CompileResultWAP`` where
    it does not."""
    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    @proxy_if_available(lib, float64(float64), jit_options={"cache": True})
    def cos(x):
        return _call_lib_func("cos", (x,))

    assert hasattr(cos, "as_func")
    if jit_addr_supported():
        assert isinstance(cos.as_func, DeriveWAP)
    else:
        # `isinstance` would pass either way, since `DeriveWAP` subclasses this.
        assert type(cos.as_func) is CompileResultWAP
    assert abs(cos(0.5) - math.cos(0.5)) < 1e-15


def test_proxy_if_available_missing_symbol_returns_stub():
    """When the C symbol is absent, ``proxy_if_available`` returns a
    Python stub that raises ``NotImplementedError`` on call. The stub
    intentionally lacks ``.as_func`` (see helper docstring).

    Stub metadata matches the real ``@proxy`` dispatcher where applicable:
    ``__name__`` is prefixed via :func:`make_proxy_name` (so callers
    that ``repr()`` or log the binding see the same shape regardless
    of whether the symbol was available); ``__qualname__`` and
    ``__doc__`` preserve the user-side function for debugging.
    """
    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    @proxy_if_available(lib, float64(float64))
    def nonexistent_fn(x):
        """Docstring on the stub-target for metadata-preservation check."""
        return x

    assert nonexistent_fn.__name__ == make_proxy_name("nonexistent_fn")
    assert nonexistent_fn.__qualname__.endswith("nonexistent_fn")
    assert nonexistent_fn.__doc__ == "Docstring on the stub-target for metadata-preservation check."
    assert not hasattr(nonexistent_fn, "as_func")
    with pytest.raises(NotImplementedError, match="nonexistent_fn is not available"):
        nonexistent_fn(1.0)


def test_proxy_if_available_missing_symbol_njit_raises_clear_error():
    """When the C symbol is absent, calling the stub from ``@njit`` raises a
    clear ``TypingError`` naming the binding and the missing-symbol cause at
    typing time, not an opaque numba typing failure."""
    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    @proxy_if_available(lib, float64(float64))
    def nonexistent_njit_fn(x):
        return x

    @njit
    def use_it(x):
        return nonexistent_njit_fn(x)

    with pytest.raises(TypingError) as excinfo:
        use_it(1.0)
    message = str(excinfo.value)
    assert "nonexistent_njit_fn is not available" in message
    assert "C symbol missing" in message


def test_proxy_factory_closures_dispatch_to_their_own_body():
    """Same-qualname @proxy closures over different values must not collide.

    A factory ``mk(c)`` produces two ``@proxy`` closures with an identical
    module+qualname+signature but different closure value ``c``. Before the
    alias folded a body fingerprint, both mapped to the same alias and
    llvmlite's last-writer-wins ``add_symbol`` silently rebound BOTH jitted
    callers to the second body (the review reproduced call_p1(10)=30 instead of
    20). The folded fingerprint captures the closure cell, so each body gets its
    own alias and each caller reaches its own body.
    """
    def mk(c):
        @proxy(float64(float64))
        def tmpl(x):
            return c * x
        return tmpl

    p1 = mk(2.0)
    p2 = mk(3.0)

    @njit
    def call_p1(x):
        return p1(x)

    @njit
    def call_p2(x):
        return p2(x)

    assert p1(10.0) == 20.0 and p2(10.0) == 30.0
    assert call_p1(10.0) == 20.0, "jitted caller of p1 reached the wrong body (alias collision)"
    assert call_p2(10.0) == 30.0, "jitted caller of p2 reached the wrong body (alias collision)"


def test_proxy_factory_over_c_symbols_dispatch_distinctly():
    """Factory-made proxies over per-instance C symbol names must not collide.

    The common numbox binding body references the ``@intrinsic``
    ``_call_lib_func``, which the strict fingerprint cannot canonicalize; the
    best-effort fallback still captures the per-instance closure (the C symbol
    name), so two same-qualname bindings over different symbols reach their own
    body. Before that fallback the alias collapsed to bytecode alone (identical
    for both) and a caller of ``cos`` returned ``sin``'s result.
    """
    import numbox.core.bindings.libm  # noqa: F401 - loads libm so cos/sin resolve

    def mk(fname):
        @proxy(float64(float64))
        def libfn(x):
            return _call_lib_func(fname, (x,))
        return libfn

    p_cos = mk("cos")
    p_sin = mk("sin")

    @njit(float64(float64))
    def call_cos(x):
        return p_cos(x)

    @njit(float64(float64))
    def call_sin(x):
        return p_sin(x)

    assert abs(call_cos(1.0) - math.cos(1.0)) < 1e-12, "jitted caller of cos reached the wrong body"
    assert abs(call_sin(1.0) - math.sin(1.0)) < 1e-12, "jitted caller of sin reached the wrong body"


def test_proxy_redefinition_keeps_callers_on_their_own_body():
    """Re-decorating a same module+qualname+sig function with a different body
    must not rebind a stale dispatcher's later-compiled caller to the new body.

    Reproduces the in-process redefinition hazard: with the pre-fix alias a
    caller compiled after the redefinition but calling the FIRST dispatcher got
    the SECOND body (the review reproduced caller_after(1.0)=101.0). The folded
    body fingerprint gives the two bodies distinct aliases.
    """
    @proxy(float64(float64))
    def g(x):
        return x + 1.0

    old_g = g

    @proxy(float64(float64))
    def g(x):  # noqa: F811 - deliberate same-name redefinition
        return x + 100.0

    @njit
    def caller_after(x):
        return old_g(x)

    assert old_g(1.0) == 2.0
    assert g(1.0) == 101.0
    assert caller_after(1.0) == 2.0, "stale dispatcher's caller reached the redefined body"


def test_proxy_absent_binding_caller_gets_loud_diagnostic_not_segfault(tmp_path):
    """A cache=True caller of a proxy_if_available binding that is present when
    the cache is written but absent on reload must fail loudly -- not segfault,
    and not hand back a value.

    numba's caller cache key is callee-blind, so a warm-cached caller cache-hits
    even in a process where ``proxy_if_available`` took the absent (stub) path
    and registered no real body. Before the trap the unresolved extern linked to
    null and the call was a bare SIGSEGV (exit 139, zero diagnostics). The trap
    keeps the alias resolvable, but reaching it is not enough on its own: the
    ``RuntimeError`` it raises is raised inside a ``@cfunc``, where numba swallows
    it and returns zero, so the caller went on to print a wrong answer and exit 0.
    The cache guard now discards an entry whose alias resolves only to a trap, so
    the recompile reaches the stub's typing error instead -- the same outcome a
    cold cache gives.
    """
    pkg = tmp_path / "flip_pkg"
    pkg.mkdir()
    (pkg / "binding_flip.py").write_text(textwrap.dedent('''
        import os
        from numba.core.types import float64
        from numbox.core.proxy.proxy import proxy_if_available

        class FakeLib:
            pass

        lib = FakeLib()
        if os.environ.get("HAVE_SYM") == "1":
            lib.myfn = True

        @proxy_if_available(lib, float64(float64), jit_options={"cache": True})
        def myfn(x):
            return x * 2.0
    '''), encoding="utf-8")
    (pkg / "caller_flip.py").write_text(textwrap.dedent('''
        from numba import njit
        from numba.core.types import float64
        from binding_flip import myfn

        @njit(float64(float64), cache=True)
        def caller(x):
            return myfn(x) + 1.0
    '''), encoding="utf-8")
    (pkg / "run_flip.py").write_text(textwrap.dedent('''
        from caller_flip import caller
        print("RESULT", caller(2.5), flush=True)
    '''), encoding="utf-8")

    cache_dir = tmp_path / "nbcache"
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    env["PYTHONPATH"] = os.pathsep.join([str(pkg), *sys.path])

    # Process 1: symbol available -> alias registered, caller cache written warm.
    env["HAVE_SYM"] = "1"
    r1 = subprocess.run(
        [sys.executable, str(pkg / "run_flip.py")],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert r1.returncode == 0, f"warm-up failed:\n{r1.stderr}"
    assert r1.stdout.strip() == "RESULT 6.0", r1.stdout

    # Process 2: symbol absent -> stub path, alias never registered. The warm
    # caller cache-hits and references the unregistered alias.
    env["HAVE_SYM"] = "0"
    r2 = subprocess.run(
        [sys.executable, str(pkg / "run_flip.py")],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert r2.returncode not in (-11, 139), (
        f"segfault instead of a loud diagnostic (returncode={r2.returncode})\n{r2.stderr}"
    )
    assert "RESULT" not in r2.stdout, (
        f"the absent binding was called and produced a value: {r2.stdout!r}"
    )
    assert "TypingError" in r2.stderr and "myfn" in r2.stderr, (
        f"expected a typing error naming the binding on stderr, got:\n{r2.stderr}"
    )
    assert "not available in the loaded library" in r2.stderr, r2.stderr


def test_proxy_function_above_anchor_line_raises_clear_error(tmp_path):
    # The cache anchor prepends blank lines so the generated @njit lands at the
    # function's co_firstlineno; a function defined above that line can't be
    # anchored (a negative prepend was silently clamped to 0, mis-anchoring it).
    # Decorating such a function must raise a clear error, not mis-anchor.
    mod = tmp_path / "top_proxy_mod.py"
    mod.write_text(
        "from numba import float64\n"
        "from numbox.core.proxy.proxy import proxy\n"
        "@proxy(float64(float64))\n"
        "def top_fn(x):\n"
        "    return 3.14 * x\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        with pytest.raises(ValueError, match="anchor"):
            importlib.import_module("top_proxy_mod")
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("top_proxy_mod", None)


# A proxied body that raises, shared by the exception-semantics tests below.
aux_raises_sig = float64(float64)


@proxy(aux_raises_sig, jit_options={'cache': True})
def aux_raises(x):
    if x > 0.0:
        raise ValueError("proxy boom")
    return x + 1.0


_JIT_ADDR_REASON = "`FunctionModel` has no `jit_addr` slot (numba < 0.61), so `.as_func` stays a plain `CompileResultWAP`"


def test_proxy_as_func_is_a_derive_where_the_jit_addr_slot_exists():
    """``.as_func`` is the handle that carries the propagating calling convention.

    Where ``FunctionModel`` has the ``jit_addr`` slot the binding hands out a ``DeriveWAP``
    typed as ``DeriveFunctionType``, which is what lets a first-class call select numba's own
    calling convention and unwind an exception out of the proxied body. Where the slot is
    absent there is nothing to populate, so the value stays the plain ``CompileResultWAP`` it
    has always been. Both arms are written out rather than skipped, because every numba in the
    supported range runs one of them.

    ``isinstance(..., CompileResultWAP)`` holds either way, which is why the second arm has to
    compare the exact type: ``DeriveWAP`` subclasses it.
    """
    @proxy(float64(float64))
    def scaled(x):
        return 2.0 * x

    as_func = scaled.as_func
    assert isinstance(as_func, CompileResultWAP)
    if jit_addr_supported():
        assert isinstance(as_func, DeriveWAP)
        assert isinstance(typeof(as_func), DeriveFunctionType)
        assert as_func.jit_address != 0, "the callconv entry point is what fills `jit_addr`"
    else:
        assert type(as_func) is CompileResultWAP
        assert type(typeof(as_func)) is FunctionType


@pytest.mark.skipif(not jit_addr_supported(), reason=_JIT_ADDR_REASON)
def test_proxy_as_func_as_an_njit_argument_propagates_a_raising_body():
    """An exception raised inside a proxied body reaches the caller through ``.as_func``.

    The caller's signature is inferred, so numba types the parameter from the value handed in
    and sees ``DeriveFunctionType``. A plain ``CompileResultWAP`` in the same position carries
    no entry point that could unwind, so the call returns a zero-filled ``0.0`` and reports the
    failure only as an unraisable on stderr.
    """
    @njit
    def call_through(f, x):
        return f(x)

    assert call_through(aux_raises.as_func, -3.0) == -2.0
    with pytest.raises(ValueError, match="proxy boom"):
        call_through(aux_raises.as_func, 1.0)


@pytest.mark.skipif(not jit_addr_supported(), reason=_JIT_ADDR_REASON)
def test_proxy_as_func_as_a_jitted_constant_propagates_a_raising_body():
    """The same contract on the constant path, which is lowered separately.

    An argument is typed on the way in and unboxed; a constant is resolved at compile time and
    lowered by ``lower_constant``, so the two reach ``jit_addr`` through different code and one
    can regress while the other stays green. This path is also what the caching behaviour rides
    on: declaring the entry point symbolically rather than baking an address is what lets a
    ``cache=True`` caller of a constant-reached ``.as_func`` be cached at all.
    """
    as_func = aux_raises.as_func

    @njit
    def call_constant(x):
        return as_func(x)

    assert call_constant(-3.0) == -2.0
    with pytest.raises(ValueError, match="proxy boom"):
        call_constant(1.0)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_proxy_dispatcher_call_still_discards_a_raising_body():
    """The binding's other handle keeps its old semantics, on every numba version.

    Calling the dispatcher goes through the proxied body's cfunc wrapper, which numba documents
    as not supporting exceptions: it discards the exception, zero-fills the return value and
    reports the failure only as an unraisable on stderr. So one binding carries two handles that
    disagree, and the asymmetry is worth pinning rather than assuming away.

    The unraisable reaches pytest as a ``PytestUnraisableExceptionWarning``, filtered by the
    marker above, as ``test/core/test_sqlite_tvf.py`` does for its own raising callback. The
    returned value is what is asserted and the warning text is not, because the ``repr`` of the
    context object numba names in it varies by numba version.
    """
    assert aux_raises(-3.0) == -2.0
    assert aux_raises(1.0) == 0.0, "the cfunc wrapper stopped zero-filling a discarded raise"


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_proxy_as_func_declared_as_a_plain_function_type_still_discards():
    """Crossing the Python boundary into a *declared* plain ``FunctionType`` degrades.

    ``DeriveFunctionType.can_convert_to`` permits the conversion, so an explicitly typed ``njit``
    written before the derive type existed keeps accepting a binding, and it documents that the
    value arrives on the C convention: the exception is discarded exactly as it was. The sibling
    suite covers that degradation for a ``cres`` value; a ``@proxy`` binding reaches the same
    declaration through ``.as_func`` and is covered here.

    The type of the value handed in is asserted first because both assertions below hold for a
    plain ``CompileResultWAP`` too: without that check the test would keep passing while
    ``.as_func`` quietly stopped being a ``DeriveWAP``, and ``can_convert_to`` would never be
    entered at all. Where the ``jit_addr`` slot is absent there is no conversion to enter and
    the C convention is all there has ever been, so the check is made only on the other arm.

    Like the dispatcher-call test above, the discarded exception surfaces as an unraisable and
    only the returned value is asserted.
    """
    as_func = aux_raises.as_func
    if jit_addr_supported():
        assert isinstance(as_func, DeriveWAP), (
            f"the value handed in is a {type(as_func).__name__}, so the declared plain "
            f"`FunctionType` never reaches `DeriveFunctionType.can_convert_to`"
        )

    @njit(float64(FunctionType(float64(float64)), float64))
    def call_through_declared_plain(f, x):
        return f(x)

    assert call_through_declared_plain(as_func, -3.0) == -2.0
    assert call_through_declared_plain(as_func, 1.0) == 0.0


@pytest.mark.skipif(not jit_addr_supported(), reason=_JIT_ADDR_REASON)
def test_proxy_as_func_mixed_with_a_numba_native_wrapper_fails_to_unify():
    """Characterization of a narrowing: a heterogeneous tuple of function values is refused.

    numba unifies a tuple's element types in ``unified_function_type`` with a bare
    class-identity comparison, which runs before any of numbox's conversions get a say, so a
    ``DeriveFunctionType`` element beside a plain ``FunctionType`` one trips a message-less
    ``AssertionError`` out of ``numba/core/utils.py``. The same tuple returned a value while
    ``.as_func`` was a plain ``CompileResultWAP``, so this is a real loss of reach and it is
    pinned here rather than left to surprise a caller. Homogeneous tuples are unaffected.

    Why the two types cannot simply be made to compare equal is worked through in
    ``test/utils/test_derive_wap.py::test_the_derive_type_stays_distinct_from_the_plain_function_type``.

    numba's assert carries no message, so the frame it was raised from is checked instead of any
    text. Matching on ``AssertionError`` alone would be satisfied by an unrelated one raised
    anywhere in the same call.
    """
    @njit(float64(float64))
    def native(x):
        return x + 10.0

    foreign = CompileResultWAP(native.get_compile_result(native.nopython_signatures[0]))

    @njit
    def use_pair(pair, x):
        return pair[0](x) + pair[1](x)

    with pytest.raises(AssertionError) as raised:
        use_pair((aux_raises.as_func, foreign), -3.0)
    assert raised.traceback[-1].name == "unified_function_type", (
        f"the refusal moved out of numba's function-type unification: {raised.traceback[-1].name}"
    )


if __name__ == "__main__":
    collect_and_run_tests(__name__)


def test_proxy_jit_flags_dispatch_to_their_own_body():
    """Two decorations of one body differing only in jit flags must not collide.

    The flags govern the machine code ``njit(sig, **jit_options)`` emits, but
    they appear in neither the module+qualname+signature identity nor the body
    fingerprint -- the body source is byte-identical here. Both decorations
    therefore mapped to a single alias and llvmlite's last-writer-wins
    ``add_symbol`` rebound every caller to whichever compiled last: a jitted
    caller of the ``numpy`` error-model binding ran the ``python`` one, so
    ``1.0 / 0.0`` returned 0.0 instead of inf, exit 0, with the
    ``ZeroDivisionError`` swallowed at the cfunc boundary.
    """
    def mk(error_model):
        @proxy(float64(float64), jit_options={"error_model": error_model})
        def recip(x):
            return 1.0 / x
        return recip

    p_np = mk("numpy")
    p_py = mk("python")

    assert p_np._numbox_proxy_alias != p_py._numbox_proxy_alias, (
        "jit flags absent from the alias -- the two bodies collide"
    )

    @njit
    def call_np(x):
        return p_np(x)

    assert math.isinf(call_np(0.0)), (
        "jitted caller of the numpy error-model binding reached the python one"
    )


def _make_extfn_binding(symbol):
    """A factory binding, the shape whose captured value the fingerprint has to see."""
    from numba.core.types import ExternalFunction, float64 as nb_f8

    fn = ExternalFunction(symbol, nb_f8(nb_f8))

    @proxy(float64(float64))
    def body(x):
        return fn(x)
    return body


def test_two_factory_bindings_over_different_symbols_get_different_aliases():
    """Two bodies a factory made over different C symbols must not share one alias.

    They share a code object and a qualname, so the alias can only tell them apart by the
    captured value. When it could not, both minted one alias, ``ll.add_symbol`` took the
    second registration, and every caller compiled against the first was silently rebound to
    the second body: a wrong number, decided by construction order, with no diagnostic.
    """
    floor_b = _make_extfn_binding("floor")
    ceil_b = _make_extfn_binding("ceil")
    assert floor_b._numbox_proxy_alias != ceil_b._numbox_proxy_alias, (
        "two bodies over different C symbols collapsed onto one alias"
    )

    @njit(float64(float64))
    def call_floor(x):
        return floor_b(x)

    @njit(float64(float64))
    def call_ceil(x):
        return ceil_b(x)

    assert call_floor(2.5) == 2.0, "the floor binding ran another body"
    assert call_ceil(2.5) == 3.0, "the ceil binding ran another body"


def test_bodies_the_fingerprint_cannot_separate_are_detected_rather_than_rebound():
    """The residual class: a captured value with no process-stable identity.

    A ctypes pointer built from a raw address has no library and no symbol name, so two of
    them differ only by an address that moves between processes and no fingerprint that has
    to mean the same thing next run can separate them. The alias is therefore allowed to
    collide, and the collision is caught instead: the second body is published under a
    process-local name of its own so both stay callable and correct, and the shared name is
    retired so no warm caller is served against whichever body holds it today.
    """
    import ctypes
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take two distinct symbol addresses from")
    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

    def make(addr):
        fp = proto(addr)

        @proxy(float64(float64))
        def body(x):
            return fp(x)
        return body

    first = make(ctypes.cast(libm.floor, ctypes.c_void_p).value)
    with pytest.warns(AliasCollisionWarning):
        second = make(ctypes.cast(libm.ceil, ctypes.c_void_p).value)

    assert first._numbox_proxy_alias != second._numbox_proxy_alias, (
        "the second body was handed a name the first already holds"
    )
    assert first._numbox_proxy_alias in _ABSENT_ALIASES, (
        "the shared name still identifies a body, so a warm caller can be served the wrong one"
    )

    @njit(float64(float64))
    def call_first(x):
        return first(x)

    @njit(float64(float64))
    def call_second(x):
        return second(x)

    assert call_first(2.5) == 2.0
    assert call_second(2.5) == 3.0


def test_a_retired_alias_stays_retired_when_its_body_is_registered_again():
    """Retirement has to outlive the body that lost the name, not just the moment it lost it.

    The collision above retires the shared name because another process may build the two
    bodies in the other order, and a later registration of either of them does not change
    that. Taking the alias back out of ``_ABSENT_ALIASES`` there hands a warm ``cache=True``
    caller back the entry the retirement exists to discard, and which of the two bodies that
    entry was compiled against is decided by a construction order this process cannot see.
    Every other retirement assertion in this file reads the set at the moment of publication;
    this is the one that reads it after the same body comes round a second time.

    The two assertions before the last one are what stop this holding by construction. A
    re-registration only reaches the branch under test when it is read as the same body: it
    has to come back under the shared name rather than a ``_c<N>`` one of its own, and it has
    to warn about nothing. An implementation that instead read it as a fresh collision would
    also leave the alias retired, and without those two the test would pass while never
    reaching the branch it is about. What it fails against is the implementation that clears
    ``_ABSENT_ALIASES`` unconditionally on that branch.
    """
    import ctypes
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take two distinct symbol addresses from")
    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

    def make(addr):
        fp = proto(addr)

        @proxy(float64(float64))
        def body(x):
            return fp(x)
        return body

    floor_addr = ctypes.cast(libm.floor, ctypes.c_void_p).value
    first = make(floor_addr)
    alias = first._numbox_proxy_alias
    with pytest.warns(AliasCollisionWarning):
        second = make(ctypes.cast(libm.ceil, ctypes.c_void_p).value)
    assert alias in _ABSENT_ALIASES, (
        "the collision left the shared name identifying a body, so there is nothing here to keep retired"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", AliasCollisionWarning)
        again = make(floor_addr)

    assert again._numbox_proxy_alias == alias, (
        "the body that already holds the shared name was handed a second name, so this never "
        "reached the branch that reads a registration as the same body"
    )
    assert alias in _ABSENT_ALIASES, (
        "registering the first body again took the shared name back out of _ABSENT_ALIASES; the "
        "name still does not say which of the two bodies a caller cached in another process was "
        "compiled against, and that caller now loads instead of recompiling"
    )

    @njit(float64(float64))
    def call_again(x):
        return again(x)

    @njit(float64(float64))
    def call_the_other(x):
        return second(x)

    assert call_again(2.5) == 2.0, "the re-registered body ran another body"
    assert call_the_other(2.5) == 3.0, (
        "re-pointing the shared name rebound the body that took a name of its own"
    )


def _make_scaling_binding(scale):
    """A factory whose body captures only values the fingerprint canonicalizes."""
    @proxy(float64(float64))
    def scaled(x):
        return scale * x
    return scaled


def _make_intrinsic_binding():
    """A factory whose body reads a global the fingerprint cannot canonicalize.

    ``_call_lib_func`` is an ``@intrinsic``, which has no canonical form, so this body
    fingerprints through the best-effort walker and its digest carries a placeholder
    where the intrinsic was. That placeholder is the only place two different bodies can
    still meet on one alias, so it is the path the re-registration check has to get right.
    """
    @proxy(float64(float64))
    def flr(x):
        return _call_lib_func("floor", (x,))
    return flr


def test_one_body_registered_twice_keeps_its_alias():
    """Registering the same body again is not a collision, and must not be read as one.

    A factory called twice with equal arguments builds one body twice, and so does
    reloading a module. Both reach the alias the first registration already holds, but
    both name the same code, so re-pointing it is correct and is what happened before the
    collision check existed. Reading it as a collision retires the alias, which discards
    every warm caller of it in this and every later process, for nothing.
    """
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    with warnings.catch_warnings():
        warnings.simplefilter("error", AliasCollisionWarning)
        first = _make_scaling_binding(2.0)
        second = _make_scaling_binding(2.0)

    assert first._numbox_proxy_alias == second._numbox_proxy_alias, (
        "one body registered twice was handed two different aliases"
    )
    assert first._numbox_proxy_alias not in _ABSENT_ALIASES, (
        "one body registered twice retired its own alias, so every warm caller recompiles"
    )

    @njit(float64(float64))
    def call(x):
        return second(x)

    assert call(3.0) == 6.0


def test_a_body_over_an_opaque_value_registered_twice_keeps_its_alias():
    """The same, for the fingerprint path that carries a placeholder.

    Every shipped numbox binding takes this path, because every one of them reads the
    ``@intrinsic`` ``_call_lib_func``. Comparing the two registrations by the address
    they compiled to reports a collision for all of them -- one ``importlib.reload`` of
    a bindings module retired all 42 libm aliases -- so the values behind the
    placeholders are what has to be compared instead.
    """
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    with warnings.catch_warnings():
        warnings.simplefilter("error", AliasCollisionWarning)
        first = _make_intrinsic_binding()
        second = _make_intrinsic_binding()

    assert first._numbox_proxy_alias == second._numbox_proxy_alias, (
        "one body registered twice was handed two different aliases"
    )
    assert first._numbox_proxy_alias not in _ABSENT_ALIASES, (
        "one body registered twice retired its own alias, so every warm caller recompiles"
    )

    @njit(float64(float64))
    def call(x):
        return second(x)

    assert call(2.5) == 2.0


def test_a_trap_keeps_its_hands_off_an_alias_a_present_binding_holds():
    """The absent-symbol trap must not take an alias a live body already answers to.

    One definition bound over two candidate library handles, the first carrying the symbol
    and the second not, reaches one alias: ``_stable_cfunc_alias`` keys on the body,
    signature and jit_options, and not on the handle. Registering the trap there anyway
    replaced a body that returns the right answer with a cfunc that raises inside its own
    wrapper, which swallows the exception and hands the caller a zeroed return -- measured
    as ``6.2e-310`` where ``1.0`` was correct, with no warning of any kind. That is the
    silent wrong answer this registry exists to stop, reached from the other direction.

    The body wins the alias, because a caller compiled in this process is calling a binding
    that is present. A process that only ever sees the absent handle registers the trap
    against an empty name and keeps the diagnostic it was written for.
    """
    from llvmlite import binding as ll
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    class _WithoutTheSymbol:
        """A second candidate handle, exporting nothing."""

    def bind(handle):
        @proxy_if_available(handle, float64(float64))
        def cos(x):
            return _call_lib_func("cos", (x,))
        return cos

    present = bind(lib)
    alias = present._numbox_proxy_alias
    address_before = ll.address_of_symbol(alias)

    with pytest.warns(AliasCollisionWarning, match="no trap was installed"):
        bind(_WithoutTheSymbol())

    assert ll.address_of_symbol(alias) == address_before, (
        "the trap took an alias a present binding already answers to"
    )
    assert alias not in _ABSENT_ALIASES, (
        "the alias was retired even though the body holding it is callable here, so every "
        "warm caller of a working binding recompiles for nothing"
    )

    @njit(float64(float64))
    def call_present(x):
        return present(x)

    assert abs(call_present(0.5) - math.cos(0.5)) < 1e-15, (
        "a caller compiled after the absent handle was bound reached the trap, not the body"
    )


def test_a_body_reaching_a_traps_alias_is_told_it_found_a_trap():
    """The collision message has to name what actually holds the alias.

    A trap records a witness no body can match, so a body reaching its alias always takes
    the collision branch. Reporting that as "two @proxy bodies fingerprint alike" describes
    a second body that does not exist, and sends anyone diagnosing a cold cache looking for
    a factory that is not there.

    The action is right and does not change. ``_stale_proxy_aliases`` treats membership of
    ``_ABSENT_ALIASES`` as stale whether or not the symbol still resolves, so keeping the
    alias retired is what stops a warm ``cache=True`` caller being served the trap's
    swallowed return in a process where the binding really is missing.
    """
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    class _WithoutTheSymbol:
        """A first candidate handle, exporting nothing."""

    def bind(handle):
        @proxy_if_available(handle, float64(float64))
        def cos(x):
            return _call_lib_func("cos", (x,))
        return cos

    bind(_WithoutTheSymbol())
    with pytest.warns(AliasCollisionWarning, match="absent-binding trap") as caught:
        present = bind(lib)

    assert "fingerprint alike" not in str(caught[0].message), (
        "the trap case reports a second body that does not exist"
    )
    assert present._numbox_proxy_alias.rsplit("_c", 1)[0] in _ABSENT_ALIASES, (
        "the alias a trap answers to was handed to a body, so a warm caller in a process "
        "without the binding is served the trap's swallowed return instead of its error"
    )
    assert abs(present(0.5) - math.cos(0.5)) < 1e-15


def test_open_libm_hands_back_a_usable_handle():
    """Pin the helper's contract on whatever platform this is running on.

    Twelve tests skip when ``open_libm`` returns ``None``, four of them the only
    coverage of the ctypes-pointer fingerprint path, so a platform where it quietly
    stopped finding a math library would give that up and still report green.
    Asserting it here makes that a failure instead, and on Windows it is what notices
    if neither ``msvcrt`` nor ``ucrtbase`` can supply the symbols.

    This checks the platform, not the gate. Anything reaching these assertions has
    already satisfied ``open_libm``'s own check, so a gate narrowed to a subset would
    still pass here on any runtime that exports all three regardless. What the gate
    admits is pinned by ``test_open_libm_refuses_a_handle_a_caller_would_reach_past``
    below.

    This lives here rather than beside the helper in ``test/auxiliary_utils.py``,
    because that filename does not match pytest's ``python_files`` and nothing in it is
    collected by a plain run.
    """
    lib = open_libm()
    assert lib is not None, "no usable math library on this platform, so twelve tests skip"
    assert hasattr(lib, "ceil") and hasattr(lib, "cos") and hasattr(lib, "floor")
    assert open_libm() is not lib, "handles are shared, so a two-handle comparison is vacuous"


def test_open_libm_refuses_a_handle_a_caller_would_reach_past(monkeypatch):
    """A gate narrower than what the callers bind admits a library they then fail on.

    ``open_libm`` returns ``None`` so a platform without a usable C runtime skips
    rather than fails. That only holds while the gate covers each symbol the callers
    need: four of them bind ``cos``, so a runtime carrying ``ceil`` and ``floor`` but
    not ``cos`` has to come back as ``None`` and skip them, rather than come back as a
    handle and fail them.

    Every runtime in the matrix exports all three, so a narrowed gate would go unnoticed
    by the rest of the suite. Each symbol is hidden in turn from a handle that is
    otherwise real, which keeps the check on the gate rather than on the platform.
    """
    import ctypes

    from numbox.core.bindings import utils

    class _Hiding:
        def __init__(self, lib, hidden):
            self._lib = lib
            self._hidden = hidden

        def __getattr__(self, name):
            if name == self._hidden:
                raise AttributeError(name)
            return getattr(self._lib, name)

    real = open_libm()
    if real is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    for hidden in ("ceil", "cos", "floor"):
        crippled = _Hiding(real, hidden)
        if utils.platform_ == "Windows":
            monkeypatch.setattr(ctypes, "CDLL", lambda name, _h=crippled: _h)
        else:
            monkeypatch.setattr(utils, "load_lib_path", lambda path, _h=crippled: _h)
        assert open_libm() is None, f"a handle without {hidden} was admitted"


def test_one_symbol_reached_through_two_prototypes_is_not_one_body():
    """A ctypes pointer's address is not the body, and reading it as one is silent.

    Two prototypes over one symbol are two calls. numba types and lowers a call from the
    pointer's ``restype`` and ``argtypes``, so the ``c_double`` prototype compiles a call
    returning a double and the ``c_float`` one a call returning a float, and the two bodies
    are different machine code. They fingerprint alike, because a pointer built from a raw
    address has no process-stable identity to fold and both bodies collapse onto one
    placeholder, so both reach one alias. Telling the two registrations apart by the address
    alone reads that as one body registered twice, and that branch re-points the alias and
    warns about nothing, so the second body silently takes the symbol every caller of the
    first was compiled against.

    The last assertion is the one that fails against that implementation: a caller compiled
    after the second registration returned 0.0 where 2.0 is right, exit 0, no warning of any
    kind. The assertions before it hold the premise it needs. The caller compiled *before*
    the second registration is right either way, since it was linked before any rebinding,
    which is what makes the pair of callers a measurement rather than a coincidence; and the
    registered address is what separates a second name from a re-pointed symbol, so that a
    fix handing out a name while still re-pointing the old one does not pass.
    """
    import ctypes

    from llvmlite import binding as ll
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take a symbol address from")
    addr = ctypes.cast(libm.floor, ctypes.c_void_p).value

    def make(fp):
        @proxy(float64(float64))
        def body(x):
            return fp(x)
        return body

    first = make(ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(addr))
    alias = first._numbox_proxy_alias
    address_before = ll.address_of_symbol(alias)

    @njit(float64(float64))
    def call_before(x):
        return first(x)

    assert call_before(2.5) == 2.0, "the first body was not calling floor to begin with"

    with pytest.warns(AliasCollisionWarning, match="fingerprint alike"):
        second = make(ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float)(addr))

    assert second._numbox_proxy_alias != alias, (
        "the second prototype was handed the name the first is published under"
    )
    assert alias in _ABSENT_ALIASES, (
        "the shared name still identifies a body, so a warm caller can be served the wrong one"
    )
    assert ll.address_of_symbol(alias) == address_before, (
        "the second body re-pointed the symbol the first is published under"
    )

    @njit(float64(float64))
    def call_after(x):
        return first(x)

    assert call_after(2.5) == 2.0, (
        "a caller compiled after the second registration reached the other prototype's body"
    )


def test_a_body_that_lost_a_collision_does_not_serve_a_warm_caller(tmp_path):
    """The name the losing body is given says nothing about the body, so no warm caller may load on it.

    It is minted from a count of what this process has published, so in a process that builds
    the colliding pair in the other order that same name belongs to the other body. The two
    processes here share one numba cache and build the pair in opposite orders. The first
    caches a caller of the ceil binding, which loses the collision there and is published
    under the minted name; the second builds ceil first, so ceil keeps the shared alias and
    the minted name is where floor ended up. numba's cache key is callee-blind, so the warm
    caller cache-hits, and before the minted name was retired it computed ``floor(2.5)`` and
    printed 2.0 where 3.0 is right, exit 0, no warning. Retiring the minted name alongside
    the shared one is what discards that entry.

    The ``STALE`` count is what stops this passing by construction. A second process that
    never found the cached entry would print 3.0 as well, by recompiling for some reason of
    numba's own, so it has to report that it found a cached entry and threw it away. The
    first process reporting none is the other half: a cache that was stale on the way in
    would make the second run's discard say nothing about the order.
    """
    if open_libm() is None:
        pytest.skip("No math library discoverable to take two distinct symbol addresses from")

    pkg = tmp_path / "order_pkg"
    pkg.mkdir()
    (pkg / "colliding_pair.py").write_text(textwrap.dedent('''
        import ctypes
        import os

        from numba.core.types import float64

        from numbox.core.proxy.proxy import proxy
        from test.auxiliary_utils import open_libm

        libm = open_libm()
        proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

        def make(symbol):
            fp = proto(ctypes.cast(getattr(libm, symbol), ctypes.c_void_p).value)

            @proxy(float64(float64))
            def body(x):
                return fp(x)
            return body

        if os.environ["ORDER"] == "floor_first":
            floor_binding = make("floor")
            ceil_binding = make("ceil")
        else:
            ceil_binding = make("ceil")
            floor_binding = make("floor")
    '''), encoding="utf-8")
    (pkg / "caller_pair.py").write_text(textwrap.dedent('''
        from numba import njit
        from numba.core.types import float64

        from colliding_pair import ceil_binding

        @njit(float64(float64), cache=True)
        def caller(x):
            return ceil_binding(x)
    '''), encoding="utf-8")
    (pkg / "run_pair.py").write_text(textwrap.dedent('''
        import warnings

        from numbox.core.proxy.proxy import StaleProxyCacheWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import caller_pair
        print("STALE", sum(issubclass(w.category, StaleProxyCacheWarning) for w in caught), flush=True)
        print("RESULT", caller_pair.caller(2.5), flush=True)
    '''), encoding="utf-8")

    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join([str(pkg), *sys.path])

    def run(order):
        env["ORDER"] = order
        done = subprocess.run(
            [sys.executable, str(pkg / "run_pair.py")],
            capture_output=True, text=True, encoding="utf-8", env=env,
        )
        assert done.returncode == 0, f"{order} failed:\n{done.stderr}"
        return done.stdout

    cold = run("floor_first")
    assert "STALE 0" in cold, f"the cold run discarded a cache entry it should not have:\n{cold}"
    assert "RESULT 3.0" in cold, f"the caller of the ceil binding did not run ceil:\n{cold}"

    warm = run("ceil_first")
    assert "RESULT 3.0" in warm, (
        f"a warm caller of the ceil binding was served against the body the minted name "
        f"happens to point at in this process:\n{warm}"
    )
    assert "STALE 0" not in warm, (
        f"the warm run reported no discarded entry, so the right answer here says nothing "
        f"about the order it was cached in -- it can be had by never writing the cache at "
        f"all:\n{warm}"
    )


def test_a_body_deriving_from_a_collision_does_not_serve_a_warm_caller(tmp_path):
    """Retiring the two colliding names is not enough: what captures one of them inherits the order.

    The sibling test above covers a caller of a colliding body. This is the same failure one
    composition step out, where the protection used to stop. A second ``@proxy`` body that
    merely *calls* a colliding binding captures its dispatcher, and folding that dispatcher by
    the alias it happens to hold makes the derived body's own alias a function of the order the
    process built the pair in: the wrapper over ceil folds the minted name in a process that
    built floor first, and the wrapper over floor folds it in a process that built ceil first,
    so the two wrappers exchange names between the two runs. Neither wrapper name is retired,
    because each is freshly minted and resolves, so unlike the colliding pair the derived body
    keeps its cross-process cache. The warm caller then cache-hits, numba's cache key being
    callee-blind, and computes ``floor(2.5)`` where ``ceil(2.5)`` is right: 2.0, exit 0, and the
    only warning is the collision one, which says callers of both recompile in every process.

    Refusing to fold a retired alias is what closes it. The two wrappers become
    indistinguishable to the fingerprint rather than distinguishable by the wrong thing, so they
    collide with each other, and that collision retires their names the way the first one
    retired the binding names.

    The ``STALE`` counts carry the same weight as in the sibling: the warm run has to report
    that it found a cached entry and discarded it, or the right answer is available by never
    having written the cache at all.
    """
    if open_libm() is None:
        pytest.skip("No math library discoverable to take two distinct symbol addresses from")

    pkg = tmp_path / "derived_pkg"
    pkg.mkdir()
    (pkg / "colliding_pair.py").write_text(textwrap.dedent('''
        import ctypes
        import os

        from numba.core.types import float64

        from numbox.core.proxy.proxy import proxy
        from test.auxiliary_utils import open_libm

        libm = open_libm()
        proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

        def make(symbol):
            fp = proto(ctypes.cast(getattr(libm, symbol), ctypes.c_void_p).value)

            @proxy(float64(float64))
            def body(x):
                return fp(x)
            return body

        if os.environ["ORDER"] == "floor_first":
            floor_binding = make("floor")
            ceil_binding = make("ceil")
        else:
            ceil_binding = make("ceil")
            floor_binding = make("floor")
    '''), encoding="utf-8")
    (pkg / "wrapped_pair.py").write_text(textwrap.dedent('''
        from numba.core.types import float64

        from numbox.core.proxy.proxy import proxy

        from colliding_pair import ceil_binding, floor_binding

        def make_caller(binding):
            @proxy(float64(float64))
            def caller(x):
                return binding(x)
            return caller

        wrapped_floor = make_caller(floor_binding)
        wrapped_ceil = make_caller(ceil_binding)
    '''), encoding="utf-8")
    (pkg / "caller_pair.py").write_text(textwrap.dedent('''
        from numba import njit
        from numba.core.types import float64

        from wrapped_pair import wrapped_ceil

        @njit(float64(float64), cache=True)
        def caller(x):
            return wrapped_ceil(x)
    '''), encoding="utf-8")
    (pkg / "run_pair.py").write_text(textwrap.dedent('''
        import warnings

        from numbox.core.proxy.proxy import StaleProxyCacheWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import caller_pair
        print("STALE", sum(issubclass(w.category, StaleProxyCacheWarning) for w in caught), flush=True)
        print("RESULT", caller_pair.caller(2.5), flush=True)
    '''), encoding="utf-8")

    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join([str(pkg), *sys.path])

    def run(order):
        env["ORDER"] = order
        done = subprocess.run(
            [sys.executable, str(pkg / "run_pair.py")],
            capture_output=True, text=True, encoding="utf-8", env=env,
        )
        assert done.returncode == 0, f"{order} failed:\n{done.stderr}"
        return done.stdout

    cold = run("floor_first")
    assert "STALE 0" in cold, f"the cold run discarded a cache entry it should not have:\n{cold}"
    assert "RESULT 3.0" in cold, f"the caller of the ceil wrapper did not run ceil:\n{cold}"

    warm = run("ceil_first")
    assert "RESULT 3.0" in warm, (
        f"a warm caller of the wrapper around the ceil binding was served against the wrapper "
        f"around floor, because the wrapper's own alias followed the order its dependency was "
        f"built in:\n{warm}"
    )
    assert "STALE 0" not in warm, (
        f"the warm run reported no discarded entry, so the right answer here says nothing "
        f"about the order it was cached in -- it can be had by never writing the cache at "
        f"all:\n{warm}"
    )


class _AliasLockProbe:
    """Asks, from a second thread, whether the alias registry lock is free at this instant.

    A ``threading.Lock`` is not reentrant and the thread being probed is the one holding it,
    so the question cannot be asked from there. The timeout is what makes the answer a fact
    rather than a deadlock, and every probe costs it once when the lock is held as it should
    be. Disarmed until the publication under test, so that the registrations leading up to it
    are not paid for.
    """

    def __init__(self, lock):
        self.lock = lock
        self.armed = False
        self.excluded = {}

    def at(self, where):
        if not self.armed:
            return
        taken = []
        waiter = threading.Thread(target=lambda: taken.append(self.lock.acquire(timeout=0.25)))
        waiter.start()
        waiter.join()
        if taken[0]:
            self.lock.release()
        self.excluded[where] = not taken[0]


class _EntryCountingLock:
    """The alias lock, counting how many times the region it guards was entered.

    Finding the lock held at two instants does not say the two are inside one critical
    section: a lock taken and released separately around each of them answers a probe
    identically, and leaves a gap between them wide enough for another thread to mint the
    same name. The count is what tells those apart, because a publication that releases the
    lock in the middle enters more than once.
    """

    def __init__(self, lock):
        self.lock = lock
        self.entries = 0

    def __enter__(self):
        self.lock.acquire()
        self.entries += 1
        return self

    def __exit__(self, *exc_info):
        self.lock.release()
        return False


class _ProbingLl:
    """``llvmlite.binding`` as ``proxy.py`` reaches it, probing the symbol registration."""

    def __init__(self, real, probe, where):
        self.real = real
        self.probe = probe
        self.where = where

    def __getattr__(self, name):
        return getattr(self.real, name)

    def add_symbol(self, name, address):
        self.probe.at(self.where)
        return self.real.add_symbol(name, address)


def test_a_colliding_body_takes_its_name_under_the_registry_lock(monkeypatch):
    """The name a losing body is given is read out of the registry and written back into it.

    ``len(_PUBLISHED_ALIASES)`` decides the name and the store that follows is what takes it,
    so two threads that both read the count before either stores mint one name for two
    bodies. ``ll.add_symbol`` keeps whichever wrote last and the other body's own name then
    resolves to it, which is the silent rebinding the registry exists to stop.
    ``_publish_cfunc_alias`` is reached from plain Python at decoration time, after numba's
    compiler lock has been released, so nothing else serializes it.

    The window is narrow and which thread loses it is timing, so what is pinned here is the
    exclusion rather than a race: while one publisher is inside the region, no other thread
    can be. That takes two measurements. The region is probed at both ends, before the
    comparison that decides the branch and at the ``add_symbol`` that follows the mint,
    because a lock around only the read or only the write leaves the count and the store on
    opposite sides of it and mints duplicates just the same. And the lock is counted, because
    both ends being inside *a* critical section is also true of a lock dropped and retaken in
    between, which reopens the window it was added to close. Those, and taking no lock at all,
    are what this rules out. The probe runs on a second thread with a timeout, because the
    lock is not reentrant and the publishing thread is holding it.
    """
    import ctypes

    from llvmlite import binding as ll

    import numbox.core.proxy.proxy as proxy_module
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ALIAS_LOCK

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take two distinct symbol addresses from")
    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

    probe = _AliasLockProbe(_ALIAS_LOCK)
    real_match = proxy_module._opaque_values_match

    def spy_match(old, new):
        probe.at("while the branch was being decided")
        return real_match(old, new)

    monkeypatch.setattr(proxy_module, "_opaque_values_match", spy_match)
    monkeypatch.setattr(
        proxy_module, "ll", _ProbingLl(ll, probe, "while the minted name was being registered")
    )

    def make(addr):
        fp = proto(addr)

        @proxy(float64(float64))
        def body(x):
            return fp(x)
        return body

    first = make(ctypes.cast(libm.floor, ctypes.c_void_p).value)
    counted = _EntryCountingLock(_ALIAS_LOCK)
    monkeypatch.setattr(proxy_module, "_ALIAS_LOCK", counted)
    probe.armed = True
    try:
        with pytest.warns(AliasCollisionWarning):
            second = make(ctypes.cast(libm.ceil, ctypes.c_void_p).value)
    finally:
        probe.armed = False

    assert first._numbox_proxy_alias != second._numbox_proxy_alias, (
        "the second body did not take a name of its own, so it never minted one"
    )
    assert sorted(probe.excluded) == [
        "while the branch was being decided", "while the minted name was being registered"
    ], f"the publish region was not reached at both ends: {sorted(probe.excluded)}"
    for where, held in probe.excluded.items():
        assert held, f"another thread could enter the publish region {where}"
    assert counted.entries == 1, (
        f"the publication entered the alias lock {counted.entries} times, so the read that "
        f"decides the name and the store that takes it are not held apart by one critical "
        f"section; another thread can hold the lock in the gap and mint the same name"
    )


class _TrapCompileHeld:
    """Stands in for ``cfunc``, holding the absent side inside its trap compile.

    ``proxy.py`` compiles a cfunc only for the trap, so this is enough to stop an absent
    registration between the read that decides it and the write that acts on it, and to let
    a body publish there on purpose rather than by luck. The name is checked anyway, so that
    a future use on the present side cannot block the test against itself.
    """

    def __init__(self, real_cfunc):
        self.real_cfunc = real_cfunc
        self.compiling = threading.Event()
        self.published = threading.Event()

    def __call__(self, sig):
        def compile_it(fn):
            if fn.__name__ == "_trap":
                self.compiling.set()
                assert self.published.wait(60), "the present side never published"
            return self.real_cfunc(sig)(fn)
        return compile_it


def test_a_trap_does_not_take_an_alias_a_body_won_while_the_trap_was_compiling(monkeypatch):
    """The trap's decision that no body holds the alias has to still be true when it writes.

    ``_register_absent_alias_trap`` reads ``_PUBLISHED_ALIASES`` to decide whether a body
    already answers, and then compiles a cfunc before it writes. That compile is the widest
    check-then-act in the module -- around 100 ms against a publication of around 40 ms -- so a
    body publishing on another thread lands inside it routinely rather than rarely, and lands
    after the decision that said there was none. The trap raises inside a cfunc wrapper, which
    swallows the exception and hands back a zeroed return, so the caller gets ``6.2e-310``
    where the library's answer was right, with no warning of any kind: the same silent wrong
    answer :func:`_publish_cfunc_alias` exists to stop, reached from the other direction.

    Blocked rather than raced, so this is a witness for the interleaving and not a
    probability. The absent side is held inside its compile until the present side has
    published, which is the ordering the stagger produces on its own.
    """
    from llvmlite import binding as ll

    import numbox.core.proxy.proxy as proxy_module
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    class _WithoutTheSymbol:
        """A second candidate handle, exporting nothing."""

    def bind(handle):
        @proxy_if_available(handle, float64(float64))
        def floor(x):
            return _call_lib_func("floor", (x,))
        return floor

    held = _TrapCompileHeld(proxy_module.cfunc)
    monkeypatch.setattr(proxy_module, "cfunc", held)
    present = {}
    absent_warnings = []

    def register_present():
        assert held.compiling.wait(60), "the absent side never reached its compile"
        present["binding"] = bind(lib)
        held.published.set()

    def register_absent():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bind(_WithoutTheSymbol())
        absent_warnings.extend(w.category for w in caught)

    absent = threading.Thread(target=register_absent)
    live = threading.Thread(target=register_present)
    absent.start()
    live.start()
    live.join(120)
    held.published.set()
    absent.join(120)
    assert not (absent.is_alive() or live.is_alive()), "a registration thread did not finish"
    assert "binding" in present, "the present side did not finish its registration"
    binding = present["binding"]
    alias = binding._numbox_proxy_alias

    @njit(float64(float64))
    def call_present(x):
        return binding(x)

    assert abs(call_present(2.5) - 2.0) < 1e-15, (
        f"a caller compiled after the race reached the trap, not the body: "
        f"{call_present(2.5)!r}"
    )
    assert alias not in _ABSENT_ALIASES, (
        "the alias of a body that is callable here was retired by a trap that lost to it, so "
        "every warm caller of a working binding recompiles for nothing"
    )
    assert ll.address_of_symbol(alias) == proxy_module._PUBLISHED_ALIASES[alias][0], (
        "the symbol and the registry disagree about what answers to the alias"
    )
    assert AliasCollisionWarning in absent_warnings, (
        f"the trap gave up its alias without saying so: {absent_warnings}"
    )


def test_each_body_that_loses_a_collision_takes_a_name_no_other_body_holds():
    """A minted name is only a name if the next body to lose does not take it too.

    The name comes from ``len(_PUBLISHED_ALIASES)`` and is registered with ``ll.add_symbol``,
    which is last-writer-wins and silent. Two losing bodies handed one name is therefore the
    same wrong answer as two bodies on one alias, just one level down, and it is the count's
    growing on every publication that stops it -- a constant, or a per-alias counter that
    forgot to advance, would read exactly the same at the one collision the other tests
    exercise. Three bodies on one alias is the smallest arrangement where that shows.
    """
    import ctypes

    from llvmlite import binding as ll

    from numbox.core.proxy.proxy import AliasCollisionWarning

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take three distinct symbol addresses from")
    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

    def make(symbol):
        fp = proto(ctypes.cast(getattr(libm, symbol), ctypes.c_void_p).value)

        @proxy(float64(float64))
        def three_way(x):
            return fp(x)
        return three_way

    first = make("floor")
    with pytest.warns(AliasCollisionWarning):
        second = make("ceil")
    with pytest.warns(AliasCollisionWarning):
        third = make("cos")

    aliases = [b._numbox_proxy_alias for b in (first, second, third)]
    assert len(set(aliases)) == 3, (
        f"two bodies were published under one name, so whichever registered last holds it and "
        f"the other's callers reach the wrong body: {aliases}"
    )

    @njit(float64(float64))
    def call_second(x):
        return second(x)

    @njit(float64(float64))
    def call_third(x):
        return third(x)

    assert abs(call_second(2.5) - 3.0) < 1e-15, (
        f"the second body's name reached another body: {call_second(2.5)!r}"
    )
    assert abs(call_third(2.5) - math.cos(2.5)) < 1e-15, (
        f"the third body's name reached another body: {call_third(2.5)!r}"
    )
    assert len({ll.address_of_symbol(a) for a in aliases}) == 3, (
        "three names, fewer than three addresses: one of them was overwritten by add_symbol"
    )
