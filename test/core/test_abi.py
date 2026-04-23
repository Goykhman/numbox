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
