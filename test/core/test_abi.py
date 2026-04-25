import pytest


def test_abi_imports():
    """The surviving symbols are present and the retired ones are gone."""
    from numbox.core.bindings import abi

    assert hasattr(abi, "_emit_byval_call")
    assert hasattr(abi, "_call_lib_func_byval")
    assert hasattr(abi, "_struct_bytes")
    assert hasattr(abi, "_classify")
    assert hasattr(abi, "_current_platform")

    for retired in (
        "_call_lib_func_struct_in",
        "_call_lib_func_struct_out",
        "_call_lib_func_args_struct_out",
        "_is_win",
    ):
        assert not hasattr(abi, retired), (
            f"{retired} should have been removed"
        )


def test_struct_bytes_supports_all_struct_types():
    """The struct-size helper used by the ABI codegen handles every
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
        _struct_bytes(types.int32, "_call_lib_func_byval")


def test_call_lib_func_lldiv_via_unified():
    """End-to-end: call libc ``lldiv(10, 3)`` via the unified intrinsic
    and validate the 16-byte ``lldiv_t`` return value.

    Exercises the return-side ABI path on whatever platform the test
    runs on: SysV x86-64 and AAPCS64 read ``lldiv_t`` back from GP
    registers; Windows x64 reads it from a caller-allocated ``sret``
    slot. A regression on any of those three ABIs surfaces as a wrong
    quot or rem here.
    """
    from numba import njit
    from numbox.core.bindings import _c  # ensures libc is loaded  # noqa: F401
    from numbox.core.bindings.call import _call_lib_func

    @njit
    def run():
        return _call_lib_func("lldiv", (10, 3))

    quot, rem = run()
    assert quot == 3
    assert rem == 1
