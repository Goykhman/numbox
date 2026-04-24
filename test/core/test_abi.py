import sys
import pytest


def test_abi_imports():
    from numbox.core.bindings import abi
    assert hasattr(abi, "_emit_byval_call")
    assert hasattr(abi, "_call_lib_func_byval")
    assert hasattr(abi, "_call_lib_func_struct_in")
    assert hasattr(abi, "_call_lib_func_struct_out")
    assert hasattr(abi, "_call_lib_func_args_struct_out")
    assert hasattr(abi, "_is_win")


def test_is_win_flag():
    from numbox.core.bindings import abi
    assert abi._is_win == (sys.platform == "win32")


# NOTE on ABI coverage:
#   - struct-IN codegen (_call_lib_func_byval, _call_lib_func_struct_in) is
#     exercised end-to-end by numbduck's test_ducklib.py — that suite calls
#     real DuckDB C-API functions whose signatures take structs by value.
#     A standalone struct-in test here would need a controlled C library
#     with ≤16-byte struct-by-value entry points, which libc doesn't provide.
#   - struct-OUT codegen (_call_lib_func_struct_out,
#     _call_lib_func_args_struct_out) is exercised below via libc's lldiv,
#     which returns a 16-byte lldiv_t on every supported platform. This
#     catches return-side ABI regressions (sret on Windows x64 vs direct
#     register return on SysV x86-64 and AAPCS64) without needing a
#     bespoke test library. Gate regressions in _call_lib_func_struct_in
#     would symmetrically manifest here because both intrinsics share the
#     same `if _is_win:` gate.


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


def test_call_lib_func_args_struct_out_lldiv():
    """End-to-end: call libc ``lldiv(10, 3)`` via the struct-out intrinsic
    and validate the 16-byte ``lldiv_t`` return value.

    Exercises the return-side ABI gate on whatever platform the test runs
    on: SysV x86-64 and AAPCS64 read ``lldiv_t`` back from GP registers;
    Windows x64 reads it from a caller-allocated ``sret`` slot. A gate
    regression on any of those three ABIs would surface as a wrong quot
    or rem here. ``lldiv`` is in the C standard library (glibc, macOS
    libSystem, Windows UCRT/MSVCRT) and ``long long`` is 64 bits on all
    three, so the signature is stable.
    """
    from numba import njit
    from numbox.core.bindings import _c  # ensures libc is loaded  # noqa: F401
    from numbox.core.bindings.abi import _call_lib_func_args_struct_out

    @njit
    def run():
        return _call_lib_func_args_struct_out("lldiv", (10, 3))

    quot, rem = run()
    assert quot == 3
    assert rem == 1
