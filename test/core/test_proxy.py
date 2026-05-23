import math
import re

import numpy as np
import pytest
from numba import float64, njit
from numba.core.types import Omitted
from numba.core.types.function_type import CompileResultWAP

from numbox.core.bindings import errno_get, getenv, memcpy
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.utils import load_lib_path, platform_
from numbox.core.proxy.proxy import proxy, proxy_if_available, make_proxy_name
from numbox.utils.lowlevel import array_data_p, get_unicode_data_p
from test.auxiliary_utils import (
    assert_njit_cache_survives_subprocess_roundtrip,
    collect_and_run_tests,
)


aux_1_sig = float64(float64)


@proxy(aux_1_sig, jit_options={'cache': True})
def aux_1(x):
    return 3.14 * x


def test_1():
    assert abs(aux_1(2.2) - 3.14 * 2.2) < 1e-15
    llvm_ir = next(iter(aux_1.inspect_llvm().values()))
    assert aux_1.__name__ == make_proxy_name('aux_1')
    if '@cfunc.' in llvm_ir:
        cfunc_name = r"double @cfunc\.\w+aux_1\w+\(double"  # noqa: W605
        assert len(re.findall(f"declare {cfunc_name}", llvm_ir)) == 1
        assert len(re.findall(f"call {cfunc_name}", llvm_ir)) == 1
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

    ``proxy`` declares the callee's ``llvm_cfunc_wrapper_name`` as an extern
    in the caller's IR module; llvmlite's JIT linker resolves the symbol per
    process at cache reload, so cached IR survives ASLR across processes
    without baking in any runtime address. See the
    ``assert_njit_cache_survives_subprocess_roundtrip`` helper in
    ``test/auxiliary_utils.py`` for the full assertion contract.
    """
    assert_njit_cache_survives_subprocess_roundtrip(
        tmp_path,
        probe_source="""
            from numba import njit
            from numbox.core.bindings import errno_get, errno_set

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


def _locate_libm():
    """Find a math/libc library with at least the ``cos`` symbol."""
    if platform_ == "Windows":
        from ctypes.util import find_msvcrt
        return find_msvcrt()
    from ctypes.util import find_library
    return find_library("m")


def test_proxy_if_available_present_symbol_returns_real_proxy():
    """When the C symbol is present, ``proxy_if_available`` returns a
    real ``@proxy``-wrapped dispatcher with ``.as_func`` attached."""
    lib_path = _locate_libm()
    if lib_path is None:
        pytest.skip("No suitable math/C runtime library discoverable")
    lib = load_lib_path(lib_path)

    @proxy_if_available(lib, float64(float64), jit_options={"cache": True})
    def cos(x):
        return _call_lib_func("cos", (x,))

    assert hasattr(cos, "as_func")
    assert isinstance(cos.as_func, CompileResultWAP)
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
    lib_path = _locate_libm()
    if lib_path is None:
        pytest.skip("No suitable math/C runtime library discoverable")
    lib = load_lib_path(lib_path)

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


if __name__ == "__main__":
    collect_and_run_tests(__name__)
