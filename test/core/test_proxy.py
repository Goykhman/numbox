"""Tests of ``@proxy`` and ``proxy_if_available``: calls, aliases, collisions, traps and exceptions.

The tests check that a proxied function returns its body's result and that a jitted caller calls the body's alias
instead of embedding a copy. numba must be able to cache such a caller. The alias must be the same in every process,
and it must differ for bodies that differ in code, captured values or jit options. If the fingerprint cannot separate
two bodies, numbox publishes the second under a name of its own, retires the shared alias and warns. Other tests cover
the stub and the trap of an absent C symbol, the lock around the alias registry, exceptions raised in a body, and
``open_libm``, the helper that opens the math library. The repair of stale cached callers is tested in
``test_proxy_cache_stale.py``.
"""
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
from numbox.core.configurations import numba_version
from numbox.core.proxy.proxy import proxy, proxy_if_available, make_proxy_name
from numbox.utils.derive_wap import DeriveFunctionType, DeriveWAP
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
    """``aux_1`` returns the body's result, and the IR of its dispatcher ``__aux_1`` declares and calls the alias.

    The wrapper's LLVM IR must hold one declaration of the alias and one call to it. If numba loaded the wrapper from
    the cache, numba cannot show the IR, and the test checks only the result and the name.
    """
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
    """A proxied function with a keyword-only default argument accepts a call that omits it and one that passes it.

    The second signature types the omitted ``y`` as ``Omitted(1.3)``. The wrapper that ``@proxy`` generates takes every
    parameter as positional-or-keyword, so the call ``aux_2(2.2, 1.4)`` passes ``y`` by position.
    """
    assert abs(aux_2(2.2) - (3.14 * 2.2 + 1.3)) < 1e-15
    assert abs(aux_2(2.2, 1.4) - (3.14 * 2.2 + 1.4)) < 1e-15


def test_proxy_dispatcher_carries_the_functions_docstring():
    """The dispatcher that ``@proxy`` returns, and its ``py_func``, have the decorated function's docstring.

    ``help()`` and the rendered API read the docstring from the dispatcher. The test also checks a libc binding:
    ``getenv`` has the docstring of its body.
    """

    @proxy(float64(float64))
    def documented(x):
        """Return twice ``x``."""
        return 2.0 * x

    assert documented.__doc__ == "Return twice ``x``."
    assert documented.py_func.__doc__ == "Return twice ``x``."
    assert documented(1.5) == 3.0
    body = getenv.as_func._numbox_dispatcher.py_func
    assert body.__doc__ and getenv.__doc__ == body.__doc__


def _sole_compile_result(dispatcher):
    """Return the compile result of a dispatcher that numba has compiled for exactly one signature."""
    sigs = dispatcher.nopython_signatures
    assert len(sigs) == 1, sigs
    return dispatcher.get_compile_result(sigs[0])


def test_proxy_zero_arg_caller_is_cacheable():
    """A cached caller of ``errno_get``, a binding without arguments, has no dynamic globals.

    numba refuses to cache code with dynamic globals, the addresses that differ in every process. The next two tests
    check the same for a binding with one argument and for one with three.
    """
    @njit(cache=True)
    def caller():
        return errno_get()
    caller()
    assert not _sole_compile_result(caller).library.has_dynamic_globals


def test_proxy_single_arg_caller_is_cacheable():
    """A cached caller of ``getenv``, a binding with one argument, has no dynamic globals."""
    @njit(cache=True)
    def caller(name_p):
        return getenv(name_p)
    caller(get_unicode_data_p("NUMBOX_NONEXISTENT_XYZZY"))
    assert not _sole_compile_result(caller).library.has_dynamic_globals


def test_proxy_multi_arg_caller_is_cacheable():
    """A cached caller of ``memcpy``, a binding with three arguments, has no dynamic globals."""
    @njit(cache=True)
    def caller(dst, src):
        memcpy(array_data_p(dst), array_data_p(src), src.nbytes)
    caller(np.zeros(4, dtype=np.uint8), np.arange(4, dtype=np.uint8))
    assert not _sole_compile_result(caller).library.has_dynamic_globals


def test_proxy_caller_survives_subprocess_round_trip(tmp_path):
    """A cached caller of an ``@proxy`` binding loads from numba's cache in a second process and gives the same output.

    The tests above check only that numba accepts such a caller for caching. This test runs the caller in two
    processes that share a cache, and the second run must print the same result without adding, removing or rewriting
    a cache file. The caller's code declares the binding's alias, which each process registers with
    ``llvmlite.binding.add_symbol``. llvmlite's linker resolves the alias when numba loads the cached code, so the code
    holds no address that differs between processes. ``assert_njit_cache_survives_subprocess_roundtrip`` in
    ``test/auxiliary_utils.py`` states the checks in full, and ``test_proxy_referenced_symbol_is_process_stable`` shows
    why callers refer to the alias and not to numba's own name for the body.
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
    """A caller refers to the same callee symbol however many functions the process compiled before it.

    ``proxy`` registers each body under an alias with ``llvmlite.binding.add_symbol``, and the alias is the same in
    every process. numba's own name for the body's C entry point holds a per-process counter, ``v<N>``. If callers
    referred to that name, two processes that compiled different numbers of functions first would write different
    symbols into otherwise equal cache entries. A cache that both processes filled could then pair a body that defines
    one name with a caller that refers to the other, and the load would abort with ``LLVM ERROR: Symbol not found``.
    The probe runs twice: first with no function compiled before the caller, then with five functions compiled before
    it. The test asserts that the caller refers to the same alias both times.
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
    """If the C symbol is present, ``proxy_if_available`` returns what ``proxy`` returns.

    The result calls the C function, and its ``.as_func`` is a ``DeriveWAP``.
    """
    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    @proxy_if_available(lib, float64(float64), jit_options={"cache": True})
    def cos(x):
        return _call_lib_func("cos", (x,))

    assert hasattr(cos, "as_func")
    assert isinstance(cos.as_func, DeriveWAP)
    assert abs(cos(0.5) - math.cos(0.5)) < 1e-15


def test_proxy_if_available_missing_symbol_returns_stub():
    """If the C symbol is absent, ``proxy_if_available`` returns a stub whose call raises ``NotImplementedError``.

    The stub has no ``.as_func``, because a function value needs a compiled body. Its ``__name__`` is the name that
    :func:`make_proxy_name` gives the dispatcher of a present binding, so a log or a ``repr`` shows the same name
    whether the symbol is present or absent. Its ``__qualname__`` and ``__doc__`` are those of the decorated function.
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
    """If the C symbol is absent, a jitted call of the stub fails with a ``TypingError`` that names the missing symbol.

    The message names the binding and says that its C symbol is missing. Without the stub's ``@overload``, numba would
    report an untyped global that names the binding but not the reason.
    """
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
    """Closures of one factory over different values have different aliases, and each caller reaches its own body.

    The closures share a module, a qualified name and a signature, so only the body's fingerprint, which holds the
    closure cell, separates their aliases. If they shared an alias, the second registration with
    ``llvmlite.binding.add_symbol`` would replace the first, and the caller of the first closure would run the second
    body.
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
    """One factory's bindings over different C symbol names have different aliases, so each caller reaches its own body.

    Each body calls the ``@intrinsic`` ``_call_lib_func``, which the strict fingerprint cannot canonicalize. The
    best-effort fingerprint still holds the closure cell with the symbol name. A fingerprint of the bytecode alone would
    be the same for both bodies, and the caller of ``cos`` would return the result of ``sin``.
    """
    import numbox.core.bindings.libm  # noqa: F401 - the import loads libm, so the symbols cos and sin resolve

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
    """A redefinition with a different body does not change what a caller of the first dispatcher runs.

    The two functions share a module, a qualified name and a signature. Their fingerprints differ, so they have
    different aliases, and a caller of the first dispatcher compiled after the redefinition still runs the first body.
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
    """A cached caller of a binding whose C symbol disappears fails with a ``TypingError``.

    The process does not crash, and the caller returns no value. numba's cache key for the caller does not cover its
    callees, so the second process finds the cached caller. Its code refers to the binding's alias. In that process,
    ``proxy_if_available`` returns the stub and registers a trap under the alias. Without the trap, the unresolved
    symbol would crash the process without a message. The trap raises inside a ``@cfunc``, where numba discards the
    exception and returns zero, so a call through it would print a wrong result and exit with status 0. The cache check
    therefore discards an entry that refers to a trap's alias, and the recompile fails with the stub's ``TypingError``,
    as it does with an empty cache.
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

    # The first process has the symbol, so it publishes the binding's alias and caches the caller.
    env["HAVE_SYM"] = "1"
    r1 = subprocess.run(
        [sys.executable, str(pkg / "run_flip.py")],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert r1.returncode == 0, f"warm-up failed:\n{r1.stderr}"
    assert r1.stdout.strip() == "RESULT 6.0", r1.stdout

    # The second process lacks the symbol, so proxy_if_available returns the stub and registers a trap under the alias.
    # The cached caller refers to that alias, and the cache check discards it.
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
    """A function whose decorator line is above line 8 of its file fails at decoration with a ``ValueError``.

    ``proxy`` compiles the generated wrapper under the function's file name, with blank lines prepended, so that the
    wrapper's ``@njit`` line is the function's first line. Seven lines of the wrapper's source precede that ``@njit``
    line, so the function's first line cannot be above line 8. The module here puts the decorator on line 3.
    """
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


# A proxied body that raises for a positive argument. The exception tests below share it.
aux_raises_sig = float64(float64)


@proxy(aux_raises_sig, jit_options={'cache': True})
def aux_raises(x):
    if x > 0.0:
        raise ValueError("proxy boom")
    return x + 1.0


def test_proxy_as_func_is_a_derive_wap():
    """``.as_func`` is a ``DeriveWAP`` whose numba type is ``DeriveFunctionType`` and whose numba entry point is set.

    A call through a value of that type takes the numba entry point, so an exception raised in the body propagates.
    ``DeriveWAP`` subclasses ``CompileResultWAP``, so the test asserts the subclass, the type and the entry point's
    address, not only the base class.
    """
    @proxy(float64(float64))
    def scaled(x):
        return 2.0 * x

    as_func = scaled.as_func
    assert isinstance(as_func, CompileResultWAP)
    assert isinstance(as_func, DeriveWAP)
    assert isinstance(typeof(as_func), DeriveFunctionType)
    assert as_func.jit_address != 0, "the callconv entry point is what fills `jit_addr`"


def test_proxy_as_func_as_an_njit_argument_propagates_a_raising_body():
    """An exception raised in the body propagates out of a jitted call through ``.as_func`` passed as an argument.

    numba infers the parameter's type from the value, ``DeriveFunctionType``, so the call takes the body's numba entry
    point. A plain ``CompileResultWAP`` in the same position is called through the C entry point: the call returns
    ``0.0``, and numba writes the exception to stderr as unraisable.
    """
    @njit
    def call_through(f, x):
        return f(x)

    assert call_through(aux_raises.as_func, -3.0) == -2.0
    with pytest.raises(ValueError, match="proxy boom"):
        call_through(aux_raises.as_func, 1.0)


def test_proxy_as_func_as_a_jitted_constant_propagates_a_raising_body():
    """An exception raised in the body also propagates when the jitted caller reads ``.as_func`` as a constant.

    numba unboxes an argument at the call, but it lowers a constant at compile time with ``lower_constant``. The two
    paths set the numba entry point in different code, so a defect can affect one path and not the other. The constant
    path also decides caching: the lowering declares the entry point as a symbol instead of writing its address into
    the code. numba can therefore cache a ``cache=True`` caller of the constant.
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
    """A call through the dispatcher returns ``0.0`` when the body raises, on every numba version.

    The call enters the body through its C entry point, the ``@cfunc`` wrapper, which cannot propagate a Python
    exception. numba writes the exception to stderr as unraisable, and the call returns a zero of the return type.
    ``.as_func`` propagates the same exception, so the two ways to call one binding behave differently, and the test
    checks that difference.

    pytest reports the unraisable exception as a ``PytestUnraisableExceptionWarning``, which the marker filters, as
    ``test/core/test_sqlite_tvf.py`` does for its raising callback. The test asserts the returned value and not the
    warning text, because the text holds a ``repr`` that differs between numba versions.
    """
    assert aux_raises(-3.0) == -2.0
    assert aux_raises(1.0) == 0.0, "the cfunc wrapper stopped zero-filling a discarded raise"


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_proxy_as_func_declared_as_a_plain_function_type_still_discards():
    """If Python passes ``.as_func`` into a parameter declared as a plain ``FunctionType``, the exception is discarded.

    numba unboxes the value as the plain type, which sets no numba entry point, so the call takes the C entry point and
    returns ``0.0``. ``DeriveFunctionType.can_convert_to`` permits the conversion, so a jitted function whose signature
    declares a plain ``FunctionType`` parameter still accepts a binding's ``.as_func``.
    ``test/utils/test_derive_wap.py`` covers the same conversion for a ``cres`` value.

    The test first asserts that ``.as_func`` is a ``DeriveWAP``. Both later assertions also hold for a plain
    ``CompileResultWAP``, which never reaches ``can_convert_to``, so without the first assertion the test could pass
    even if ``.as_func`` were not a ``DeriveWAP``. As in the dispatcher test above, the discarded exception is reported
    as unraisable, and the test asserts only the returned value.
    """
    as_func = aux_raises.as_func
    assert isinstance(as_func, DeriveWAP), (
        f"the value handed in is a {type(as_func).__name__}, so the declared plain "
        f"`FunctionType` never reaches `DeriveFunctionType.can_convert_to`"
    )

    @njit(float64(FunctionType(float64(float64)), float64))
    def call_through_declared_plain(f, x):
        return f(x)

    assert call_through_declared_plain(as_func, -3.0) == -2.0
    assert call_through_declared_plain(as_func, 1.0) == 0.0


def test_proxy_as_func_mixed_with_a_numba_native_wrapper_fails_to_unify_for_numba_lt_68_and_works_for_numba_gte_68():
    """Below numba 0.68.0, a tuple of ``.as_func`` and a plain ``CompileResultWAP`` fails; numba 0.68.0 accepts it.

    numba below 0.68.0 unifies the element types of a tuple in ``unified_function_type``, in ``numba/core/utils.py``.
    That function asserts that the function types are equal, before any conversion that numbox defines can apply, so a
    ``DeriveFunctionType`` beside a plain ``FunctionType`` fails with an ``AssertionError`` that has no message. A
    tuple whose elements have one type is not affected. The reason that the two types must not compare equal is given
    in ``test/utils/test_derive_wap.py::test_the_derive_type_stays_distinct_from_the_plain_function_type``.

    The assertion has no message, so the test checks the name of the function that raised it. An ``AssertionError``
    alone could come from anywhere in the call.
    """
    @njit(float64(float64))
    def native(x):
        return x + 10.0

    foreign = CompileResultWAP(native.get_compile_result(native.nopython_signatures[0]))

    @njit
    def use_pair(pair, x):
        return pair[0](x) + pair[1](x)

    if numba_version >= 68:
        assert use_pair((aux_raises.as_func, foreign), -3.0) == 5
    else:
        with pytest.raises(AssertionError) as raised:
            use_pair((aux_raises.as_func, foreign), -3.0)
        assert raised.traceback[-1].name == "unified_function_type", (
            f"the refusal moved out of numba's function-type unification: {raised.traceback[-1].name}"
        )


if __name__ == "__main__":
    collect_and_run_tests(__name__)


def test_proxy_jit_flags_dispatch_to_their_own_body():
    """Two decorations of one body that differ only in ``error_model`` have different aliases.

    The jit options change the machine code but not the module, the qualified name, the signature or the body's
    fingerprint, so the alias hashes the options as well. With one alias for both, the later registration with
    ``add_symbol`` would replace the earlier one, and every caller would call the body compiled last. A caller of the
    ``numpy`` binding would then run the ``python`` one, whose C entry point discards the ``ZeroDivisionError`` and
    returns 0.0 instead of inf.
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
    """Return a proxied body that calls the C function ``symbol`` through a captured ``ExternalFunction``."""
    from numba.core.types import ExternalFunction, float64 as nb_f8

    fn = ExternalFunction(symbol, nb_f8(nb_f8))

    @proxy(float64(float64))
    def body(x):
        return fn(x)
    return body


def test_two_factory_bindings_over_different_symbols_get_different_aliases():
    """Two bodies that a factory makes over different ``ExternalFunction`` symbols have different aliases.

    The bodies share a code object and a qualified name, so only the captured value separates their aliases. With one
    alias for both, the second registration with ``add_symbol`` would replace the first, and the callers of the first
    body would return the second body's result without a warning.
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
    """Bodies that differ only in a ctypes pointer built from a raw address compute one alias, and numbox detects it.

    Such a pointer names no library and no symbol, and its address differs between processes, so no fingerprint that
    is the same in every process can separate the two bodies. numbox publishes the second body under a name of its
    own, valid only in this process, and warns with ``AliasCollisionWarning``. It also retires the shared alias, so the
    cache check discards a cached caller of either body, which could otherwise run the wrong body. Each caller in this
    process reaches its own body.
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
    """A retired alias stays in ``_ABSENT_ALIASES`` when the body that holds it registers again.

    numbox retires the alias of a collision because another process may build the two bodies in the other order, and a
    later registration does not change that. If the registration removed the alias from the set, a ``cache=True``
    caller cached in another process would load again, although the alias does not identify which of the two bodies
    the caller was compiled against. The other retirement tests read the set at the first publication; this test reads
    it after the same body registers again.

    The re-registration runs with ``AliasCollisionWarning`` as an error and must return the shared alias, not a name
    of the form ``<alias>_c<N>``. These two checks show that it took the branch for one body registered twice, which
    is the branch under test. An implementation that treated it as a new collision would also leave the alias
    retired, so without them the test could pass without reaching that branch. The test fails against an
    implementation that removes the alias from ``_ABSENT_ALIASES`` on that branch.
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
    """Return a binding whose body captures only values that the strict fingerprint canonicalizes."""
    @proxy(float64(float64))
    def scaled(x):
        return scale * x
    return scaled


def _make_intrinsic_binding():
    """Return a binding whose body reads a global that the strict fingerprint cannot canonicalize.

    ``_call_lib_func`` is an ``@intrinsic`` and has no canonical form, so the best-effort walker fingerprints the body
    with a placeholder in its place. Two different bodies can compute one alias only through such placeholders, so the
    re-registration check must handle this path correctly.
    """
    @proxy(float64(float64))
    def flr(x):
        return _call_lib_func("floor", (x,))
    return flr


def test_one_body_registered_twice_keeps_its_alias():
    """A body registered twice keeps its alias, without a warning and without retiring the alias.

    A factory called twice with equal arguments builds one body twice, and so does a module reload. Both registrations
    compute the alias that the first one published and compile the same code, so pointing the alias at the new address
    is correct. If numbox treated the second registration as a collision, it would retire the alias, and the cache
    check would discard every cached caller of the binding in every later process.
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
    """A body with a placeholder in its fingerprint, registered twice, also keeps its alias.

    Every numbox binding takes this path, because each one reads an ``@intrinsic``, most of them ``_call_lib_func``.
    Two compilations of one body have different addresses, so a comparison of addresses would report a collision for
    every binding, and one ``importlib.reload`` of ``numbox.core.bindings.libm`` would retire all 42 of its aliases.
    numbox therefore compares the values behind the placeholders.
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
    """The trap of an absent binding does not replace a body that is already published under the same alias.

    One definition bound over two library handles, the first with the symbol and the second without, computes one
    alias, because the alias covers the body, the signature and the jit options but not the handle. A trap under that
    alias would raise inside its ``@cfunc`` wrapper, which discards the exception and returns zero. Every call through
    the alias would then return a wrong result without a warning. ``_PUBLISHED_ALIASES`` exists to prevent that.
    The trap is not registered: the body remains published under the alias, and numbox warns with
    ``AliasCollisionWarning``, because a caller compiled in this process calls a binding that is present. A process
    that binds only the absent handle registers the trap.
    """
    from llvmlite import binding as ll
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    class _WithoutTheSymbol:
        """A second library handle, without the C symbol."""

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
    """If a body computes the alias of a trap, the ``AliasCollisionWarning`` names the trap, not a second body.

    A trap is recorded with a witness that no body matches, so such a body always takes the collision branch and
    receives a name of its own. A message about two colliding bodies would describe a body that does not exist. A
    reader who investigates a cold cache would then look for a factory that is not there.

    The alias stays retired: ``_stale_proxy_aliases`` treats an alias in ``_ABSENT_ALIASES`` as stale even if the
    symbol resolves. In a process without the binding, the cache check therefore discards a cached caller of the alias.
    A call through the alias would run the trap and return zero.
    """
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    class _WithoutTheSymbol:
        """A first library handle, without the C symbol."""

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
    """``open_libm`` returns a handle with ``ceil``, ``cos`` and ``floor`` on this platform, and a new one per call.

    The tests that bind these symbols skip when ``open_libm`` returns ``None``, among them the tests of bodies that
    capture a ctypes pointer built from a raw address. If ``open_libm`` stopped finding a math library on some
    platform, those tests would skip there and the suite would still pass, so this test makes that case a failure. On
    Windows, it fails if neither ``msvcrt`` nor ``ucrtbase`` provides the symbols.

    The test checks the platform, not the symbol test in ``open_libm``: a handle that reaches these assertions has
    already passed it. ``test_open_libm_refuses_a_handle_a_caller_would_reach_past`` below checks the symbol test.
    This test is in this file, not beside the helper in ``test/auxiliary_utils.py``, because that file's name does not
    match pytest's ``python_files`` pattern, so pytest does not collect it.
    """
    lib = open_libm()
    assert lib is not None, "no usable math library on this platform, so twelve tests skip"
    assert hasattr(lib, "ceil") and hasattr(lib, "cos") and hasattr(lib, "floor")
    assert open_libm() is not lib, "handles are shared, so a two-handle comparison is vacuous"


def test_open_libm_refuses_a_handle_a_caller_would_reach_past(monkeypatch):
    """``open_libm`` returns ``None`` for a library that lacks any one of ``ceil``, ``cos`` and ``floor``.

    The callers skip on ``None``, so a platform without a usable C runtime skips those tests instead of failing them.
    That holds only if ``open_libm`` tests every symbol that the callers bind: a library with ``ceil`` and ``floor``
    but without ``cos`` must give ``None``, because several callers bind ``cos``. Every runtime in the test matrix
    exports all three symbols, so the rest of the suite cannot detect an ``open_libm`` that tests fewer symbols. This
    test hides each symbol in turn from a real handle, so it checks ``open_libm`` and not the platform.
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
    """Two ctypes prototypes over one symbol address are two bodies, and the second receives a name of its own.

    numba types and lowers a call from the pointer's ``restype`` and ``argtypes``, so the ``c_double`` prototype
    compiles a call that returns a double and the ``c_float`` prototype one that returns a float. The two bodies are
    different machine code. Their fingerprints are equal, because a pointer built from a raw address has no
    fingerprint, and both bodies hold one placeholder for it. An implementation that compared only the addresses would
    treat the second registration as the same body. It would point the alias at the second body without a warning,
    and a caller of the first body compiled afterwards would run the second.

    The last assertion checks that caller, and the assertions before it establish its premise. The caller compiled
    before the second registration returns 2.0 in either case, because numba linked it before any change, so the pair
    of callers shows the effect of the registration. The alias must still resolve to the first body's address, so a
    fix that gives the second body a new name but still points the alias at it fails.
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
    """A cached caller of a collision's losing body is discarded in a process that builds the pair in the other order.

    numbox names the losing body after the number of aliases that this process has published, so the same name can
    belong to the other body in another process. The two processes here share one numba cache and build the pair in
    opposite orders. The first caches a caller of the ceil binding, which loses the collision there and receives the
    process-local name. The second builds ceil first, so the ceil binding is published under the shared alias and the
    floor binding receives the process-local name. numba's cache key does not cover the callees, so the second process
    finds the cached caller. numbox retires the process-local name together with the shared alias, so the cache check
    discards the entry, and the caller computes ``ceil(2.5)``. Without the retirement, it would compute ``floor(2.5)``
    and print 2.0 instead of 3.0, without a warning.

    The ``STALE`` counts show that the result comes from the discard. The second process must report that it found and
    discarded a cached entry, because a process that compiled the caller again for another reason would also print
    3.0. The first process must report no discard. If the cache were already stale before the test, the discard in the
    second process would not depend on the order.
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
    """A cached caller of a proxied body that calls a colliding binding is also discarded when the order changes.

    The previous test covers a caller of a colliding body, and this test covers a caller of a proxied body that calls a
    colliding binding. ``wrapped_ceil`` and ``wrapped_floor`` are such bodies: they call the colliding bindings and
    capture their dispatchers. If the fingerprint folded such a dispatcher by its alias, the alias of the calling body
    would depend on the order in which the process built the pair. ``wrapped_ceil`` would fold the process-local name
    in a process that built floor first, and ``wrapped_floor`` would fold it in a process that built ceil first. The
    two would then exchange aliases between the runs. Neither alias would be retired, so the second process would load
    the cached caller and compute ``floor(2.5)`` instead of ``ceil(2.5)``. The only warning would be the
    ``AliasCollisionWarning`` about the two bindings.

    The fingerprint walker therefore does not fold a retired alias. The two calling bodies then have equal
    fingerprints and collide with each other, and that collision retires their aliases, as the first collision
    retired the aliases of the bindings. The ``STALE`` counts have the same role as in the previous test.
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
    """A probe that records, from a second thread, whether the alias lock is held at a point of the publication.

    A ``threading.Lock`` is not reentrant, and the publishing thread holds the lock, so the probe cannot try to
    acquire it from that thread. The second thread tries with a timeout of 0.25 seconds, so an attempt on a held lock
    times out instead of deadlocking, and each probe of a held lock costs the timeout once. The probe stays disarmed
    until the publication under test, so the registrations before it do not pay that cost.
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
    """A wrapper of the alias lock that counts how many times a thread acquires it.

    A probe that finds the lock held at two points cannot distinguish one critical section from two. A lock acquired
    and released around each point separately gives the same answers, and between the two points another thread could
    assign the same name. The count separates the cases, because a publication that releases the lock between the two
    points acquires it twice.
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
    """A replacement for ``llvmlite.binding`` in ``proxy.py`` that probes the lock at each ``add_symbol`` call.

    Every other attribute comes from the real module.
    """

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
    """The losing body of a collision reads the count of aliases and publishes its new name in one critical section.

    ``len(_PUBLISHED_ALIASES)`` decides the name, and the store that follows takes it. If two threads read the count
    before either stored its name, they would assign one name to two bodies. The second registration with
    ``add_symbol`` would replace the first, and the callers of the first body would run the second without a warning.
    ``_publish_cfunc_alias`` runs in plain Python at decoration, after numba has released its compiler lock, so
    nothing else serializes it.

    The interval between the read and the store is short, and timing decides which thread loses, so the test checks
    the exclusion instead of provoking a race. It probes the lock at both ends of the region: at the comparison that
    decides the branch, and at the ``add_symbol`` call that registers the new name. A lock around only the read or only
    the write would leave the count and the store outside one critical section. The test also counts the acquisitions,
    because a lock released and acquired again between the two ends would pass both probes and leave the interval
    unprotected. The test fails against those two implementations and against one that takes no lock. The probe runs
    on a second thread with a timeout, because the lock is not reentrant and the publishing thread holds it.
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
    """A replacement for ``cfunc`` in ``proxy.py`` that holds the trap's compile until the present binding publishes.

    ``proxy.py`` compiles a cfunc only for the trap, so the replacement stops the absent registration between the read
    that decides it and the write that depends on it. The present binding therefore publishes in that interval on
    every run, not only when the timing allows it. The replacement holds only a function named ``_trap``, so a cfunc
    that ``proxy.py`` might compile for a present binding could not block the test.
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
    """A trap does not take an alias that a body published while the trap was compiling.

    ``_register_absent_alias_trap`` reads ``_PUBLISHED_ALIASES`` to decide whether a body is published under the
    alias, and then compiles a cfunc before it writes. That compile is the longest interval in the module between a
    check and the write that depends on it, so another thread can publish a body during the compile, after the read
    that found no body. The trap raises inside a cfunc wrapper, which discards the exception and returns zero. A
    caller of the alias would then receive a wrong result without a warning, the same failure that
    :func:`_publish_cfunc_alias` prevents for two bodies. The test blocks the absent side instead of relying on a race,
    so the interleaving happens on every run: the absent side waits inside its compile until the present side has
    published.
    """
    from llvmlite import binding as ll

    import numbox.core.proxy.proxy as proxy_module
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    class _WithoutTheSymbol:
        """A second library handle, without the C symbol."""

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
    """Each body that loses a collision receives a name that no other body holds.

    The name comes from ``len(_PUBLISHED_ALIASES)`` and is registered with ``add_symbol``, which replaces an earlier
    registration without a warning. Two losing bodies with one name would therefore give the same wrong result as two
    bodies with one alias. The count grows with every publication, which prevents that. A constant suffix, or a
    counter per alias that did not advance, would give the same name at the single collision that the other tests
    create. Three bodies with one alias are the smallest case that shows the difference.
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
