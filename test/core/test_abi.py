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


def test_struct_bytes_supports_all_struct_types():
    """The struct-size helper used by the ABI intrinsics handles every
    numba struct-shaped type: Tuple, UniTuple, NamedTuple (via .types),
    and Record (via .size)."""
    import collections
    from numba.core import types
    from numbox.core.bindings.abi import _struct_bytes

    assert _struct_bytes(
        types.Tuple([types.int32, types.int32, types.int64]), "t") == 16
    assert _struct_bytes(
        types.UniTuple(types.int32, 4), "t") == 16

    MyNT = collections.namedtuple("MyNT", ["a", "b"])
    assert _struct_bytes(
        types.NamedTuple([types.int32, types.int64], MyNT), "t") == 12

    rec = types.Record.make_c_struct([("a", types.int32), ("b", types.int64)])
    assert _struct_bytes(rec, "t") == 16  # 4 + 4 pad + 8


def test_struct_bytes_rejects_non_struct_type():
    """Scalar or otherwise non-struct types raise a clean TypingError."""
    from numba.core import types
    from numba.core.errors import TypingError
    from numbox.core.bindings.abi import _struct_bytes

    with pytest.raises(TypingError, match="struct-shaped type"):
        _struct_bytes(types.int32, "_call_lib_func_struct_in")
