import platform
import sys
import pytest


def test_abi_imports():
    from numbox.core.bindings import abi
    assert hasattr(abi, "_emit_byval_call")
    assert hasattr(abi, "_call_lib_func_byval")
    assert hasattr(abi, "_call_lib_func_struct_in")
    assert hasattr(abi, "_call_lib_func_struct_out")
    assert hasattr(abi, "_is_win")
    assert hasattr(abi, "_is_sysv_x86_64")


@pytest.mark.skipif(sys.platform == "win32", reason="SysV path only")
def test_sysv_platform_flags():
    from numbox.core.bindings import abi
    assert abi._is_win is False
    if platform.machine() == "x86_64":
        assert abi._is_sysv_x86_64 is True


# NOTE: Codegen correctness for the three @intrinsic helpers
# (_call_lib_func_byval, _call_lib_func_struct_in, _call_lib_func_struct_out)
# is exercised end-to-end by numbduck's test_ducklib.py — that suite calls
# real DuckDB C-API functions whose signatures take and return small structs.
# A standalone codegen smoke test here would need a controlled C library
# with ≤16-byte struct-by-value entry points; without one, numba's eager
# symbol resolution at @njit compile time aborts on fake symbols.
