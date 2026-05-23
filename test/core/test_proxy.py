import re

import numpy as np
from numba import float64, njit
from numba.core.types import Omitted

from numbox.core.bindings import errno_get, getenv, memcpy
from numbox.core.proxy.proxy import proxy, make_proxy_name
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


if __name__ == "__main__":
    collect_and_run_tests(__name__)
