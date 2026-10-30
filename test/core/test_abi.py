"""Tests of the calling-convention rules in numbox.core.bindings.abi and of the C calls that _call_lib_func emits.

The ``abi`` tests check the size and the field offsets of a struct written as a numba tuple or record. They also check
the class of a value and the class of each 8-byte half of a 16-byte struct, which the SysV x86-64 ABI calls an
eightbyte. The ``_call_lib_func`` tests compile calls of C functions that ``ctypes`` callbacks implement, or of stubs
that the tests never run. They check the values that reach a callback, or the declaration of the call in the LLVM IR:
the ``byval`` and ``sret`` attributes, the ``{i64, i64}`` repack and the pointer to a copy. Some tests apply to one
convention, SysV x86-64, AAPCS64 or Windows x64, and skip on the others. ``test_math.py`` and ``test_bindings.py``
test the results of the bindings themselves.
"""
import pytest


def test_abi_imports():
    """Check that ``abi`` defines its classification helpers and that ``call`` defines its two intrinsics.

    The tests below import these private names, and numbduck imports ``_call_lib_func``.
    """
    from numbox.core.bindings import abi, call

    assert hasattr(abi, "_struct_bytes")
    assert hasattr(abi, "_classify")
    assert hasattr(abi, "_classify_eightbytes")
    assert hasattr(abi, "_is_canonical_int64_pair_layout")
    assert hasattr(abi, "_EIGHTBYTE_CLASS_INTEGER")
    assert hasattr(abi, "_EIGHTBYTE_CLASS_SSE")
    assert hasattr(abi, "_current_platform")
    assert hasattr(call, "_call_lib_func")
    assert hasattr(call, "_call_lib_func_byval")


def test_classify_eightbytes_int_int_non_canonical():
    """Check that ``Tuple([int32, int32, int64])``, the layout of DuckDB's ``duckdb_interval``, has two INTEGER halves.

    The low half holds the two ``int32`` fields, at offsets 0 and 4, and the high half holds the ``int64``.
    """
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _classify_eightbytes,
    )

    ty = types.Tuple([types.int32, types.int32, types.int64])
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_int_int_canonical_pair():
    """Check that ``UniTuple(int64, 2)``, whose LLVM type is already ``{i64, i64}``, has two INTEGER halves."""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _classify_eightbytes,
    )

    ty = types.UniTuple(types.int64, 2)
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_four_i32():
    """Check that ``UniTuple(int32, 4)``, with fields at offsets 0, 4, 8 and 12, has two INTEGER halves.

    Its LLVM type is ``[4 x i32]``, not ``{i64, i64}``, so ``_call_lib_func`` repacks it on SysV x86-64 and AAPCS64.
    """
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _classify_eightbytes,
    )

    ty = types.UniTuple(types.int32, 4)
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_sse_sse():
    """Check that ``UniTuple(float64, 2)`` has two SSE halves.

    SysV x86-64 passes such a struct in XMM0 and XMM1, so a repack to ``{i64, i64}``, which LLVM passes in
    general-purpose registers, would be wrong.
    """
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_SSE, _classify_eightbytes,
    )

    ty = types.UniTuple(types.float64, 2)
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_SSE, _EIGHTBYTE_CLASS_SSE,
    )


def test_classify_eightbytes_sse_int():
    """Check that ``Tuple([float32, float32, int64])`` has an SSE low half and an INTEGER high half.

    The low half holds the two ``float32`` fields, and the high half holds the ``int64``.
    """
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_SSE,
        _classify_eightbytes,
    )

    ty = types.Tuple([types.float32, types.float32, types.int64])
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_SSE, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_mixed_lo_eightbyte_is_integer():
    """Check that a half that holds an integer and a float is INTEGER, by the SysV merge rule.

    In ``Tuple([int32, float32, int64])``, the low half holds an ``int32`` and a ``float32``, so C passes it in a
    general-purpose register and not in an SSE register. GCC passes that half of the matching C struct in ``%rdi``.
    """
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _classify_eightbytes,
    )

    ty = types.Tuple([types.int32, types.float32, types.int64])
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_mixed_both_eightbytes_integer():
    """Check that ``Tuple([float32, int32, float32, int32])`` has two INTEGER halves.

    Each half holds a ``float32`` and an ``int32``, so the SysV merge rule makes it INTEGER. GCC passes the two halves
    of the matching C struct in the general-purpose registers ``%rdi`` and ``%rsi``.
    """
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _classify_eightbytes,
    )

    ty = types.Tuple([types.float32, types.int32, types.float32, types.int32])
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_mixed_int_float_eightbyte_triggers_repack():
    """Check that ``_needs_int_int_eightbyte_repack`` asks for the repack of a struct whose halves mix ints and floats.

    By the SysV merge rule, each such half is INTEGER, so ``_call_lib_func`` passes the struct as ``{i64, i64}`` on
    SysV x86-64 and AAPCS64. Without the repack, LLVM passes each field in a register of its own. On x86-64 Linux, a C
    function that takes ``(int32, float32, int64)`` then reads 0.0 for the ``float32`` of ``(7, 2.5, 1000000)``. A
    struct of two ``float64`` fields is not repacked. The test skips on Windows x64, which passes a 16-byte struct as
    a pointer to a copy.
    """
    from numba.core import types
    from numbox.core.bindings.call import (
        _needs_int_int_eightbyte_repack, _current_platform,
    )

    plat = _current_platform()
    if plat not in ("sysv_x86_64", "aapcs64"):
        pytest.skip(f"eightbyte repack is SysV/AAPCS64-only (plat={plat})")
    for ty in (
        types.Tuple([types.int32, types.float32, types.int64]),
        types.Tuple([types.float32, types.int32, types.float32, types.int32]),
    ):
        assert _needs_int_int_eightbyte_repack(ty, plat), ty
    # A 16-byte struct of floats alone, a homogeneous floating-point aggregate in AAPCS64's terms, is not repacked.
    assert not _needs_int_int_eightbyte_repack(
        types.UniTuple(types.float64, 2), plat)


def test_classify_padded_tuple_double_is_aligned_not_spanning():
    """Check that a ``float64`` after an ``int32`` starts at offset 8, so it never spans the boundary between halves.

    ``Tuple([int32, float64, int32])`` has 4 bytes of padding after the first ``int32``, and the struct is 24 bytes.
    ``_classify`` therefore returns LARGE, and the struct never reaches the classification of halves. A scalar field
    at its natural alignment never spans the 8-byte boundary.
    """
    from numba.core import types
    from numbox.core.bindings.abi import (
        _struct_bytes, _iter_struct_fields, _classify, _CLASS_STRUCT_LARGE,
    )

    ty = types.Tuple([types.int32, types.float64, types.int32])
    assert _struct_bytes(ty, "t") == 24
    # The float64 starts at offset 8 and lies wholly in the high half, not in the bytes from 4 to 12.
    assert list(_iter_struct_fields(ty, "t")) == [(0, 4, False), (8, 8, True), (16, 4, False)]
    assert _classify(ty) == _CLASS_STRUCT_LARGE


def test_classify_eightbytes_record_with_padding():
    """Check that a 16-byte record of an ``int32`` at offset 0 and an ``int64`` at offset 8 has two INTEGER halves.

    ``Record.make_c_struct`` places 4 bytes of padding between the two fields, as a C compiler does.
    """
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _classify_eightbytes,
    )

    ty = types.Record.make_c_struct([("a", types.int32), ("b", types.int64)])
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_rejects_non_16b():
    """Check that ``_classify_eightbytes`` raises ``TypingError`` for a struct of 24 or 8 bytes.

    The function classifies two 8-byte halves, so it accepts only a 16-byte struct.
    """
    from numba.core import types
    from numba.core.errors import TypingError
    from numbox.core.bindings.abi import _classify_eightbytes

    with pytest.raises(TypingError, match="16-byte"):
        _classify_eightbytes(types.UniTuple(types.int64, 3))  # 24B
    with pytest.raises(TypingError, match="16-byte"):
        _classify_eightbytes(types.UniTuple(types.int32, 2))  # 8B


def test_classify_eightbytes_rejects_non_struct():
    """Check that ``_classify_eightbytes`` raises ``TypingError`` for a scalar type."""
    from numba.core import types
    from numba.core.errors import TypingError
    from numbox.core.bindings.abi import _classify_eightbytes

    with pytest.raises(TypingError, match="struct-shaped"):
        _classify_eightbytes(types.int64)


def test_is_canonical_int64_pair_layout_true_cases():
    """Check that two 64-bit integer fields form the canonical layout, which needs no repack.

    ``UniTuple(int64, 2)``, ``Tuple([int64, int64])`` and ``Tuple([uint64, intp])`` all have the LLVM type
    ``{i64, i64}`` that the repack would produce.
    """
    from numba.core import types
    from numbox.core.bindings.abi import _is_canonical_int64_pair_layout

    assert _is_canonical_int64_pair_layout(types.UniTuple(types.int64, 2))
    assert _is_canonical_int64_pair_layout(
        types.Tuple([types.int64, types.int64]))
    assert _is_canonical_int64_pair_layout(
        types.Tuple([types.uint64, types.intp]))


def test_is_canonical_int64_pair_layout_false_cases():
    """Check that a layout other than two 64-bit integer fields at offsets 0 and 8 is not canonical.

    The cases are ``{i32, i32, i64}``, the ``duckdb_interval`` layout that needs the repack, four ``int32`` fields,
    two ``float64`` fields and a 24-byte tuple.
    """
    from numba.core import types
    from numbox.core.bindings.abi import _is_canonical_int64_pair_layout

    assert not _is_canonical_int64_pair_layout(
        types.Tuple([types.int32, types.int32, types.int64]))
    assert not _is_canonical_int64_pair_layout(
        types.UniTuple(types.int32, 4))
    assert not _is_canonical_int64_pair_layout(
        types.UniTuple(types.float64, 2))
    assert not _is_canonical_int64_pair_layout(
        types.UniTuple(types.int64, 3))


def test_is_canonical_int64_pair_layout_record():
    """Check that a record of two ``int64`` fields qualifies as canonical and that one with an ``int32`` does not."""
    from numba.core import types
    from numbox.core.bindings.abi import _is_canonical_int64_pair_layout

    rec_canonical = types.Record.make_c_struct([
        ("a", types.int64), ("b", types.int64)])
    assert _is_canonical_int64_pair_layout(rec_canonical)

    rec_non_canonical = types.Record.make_c_struct([
        ("a", types.int32), ("b", types.int64)])
    assert not _is_canonical_int64_pair_layout(rec_non_canonical)


def test_struct_bytes_supports_all_struct_types():
    """Check that ``_struct_bytes`` gives the C size of a ``Tuple``, a ``UniTuple``, a ``NamedTuple`` and a ``Record``.

    The function computes a tuple's size from the C layout of its field types, and it takes a record's size from the
    record's ``size`` attribute.
    """
    import collections
    from numba.core import types
    from numbox.core.bindings.abi import _struct_bytes

    assert _struct_bytes(
        types.Tuple([types.int32, types.int32, types.int64]), "t") == 16
    assert _struct_bytes(
        types.UniTuple(types.int32, 4), "t") == 16

    MyNT = collections.namedtuple("MyNT", ["a", "b"])
    assert _struct_bytes(
        types.NamedTuple([types.int32, types.int64], MyNT), "t") == 16  # 4 + 4 pad + 8

    rec = types.Record.make_c_struct([("a", types.int32), ("b", types.int64)])
    assert _struct_bytes(rec, "t") == 16  # 4 + 4 pad + 8


def test_struct_bytes_tuple_matches_record_with_padding():
    """Check that a tuple's size includes C's padding and equals the size of the record with the same fields.

    numba places each field of a tuple at its natural alignment, as a C compiler does, and ``Record.make_c_struct``
    lays out the record by the same rules.
    """
    from numba.core import types
    from numbox.core.bindings.abi import _struct_bytes

    for fields in (
        [types.int32, types.int64],          # 4 + 4 pad + 8 = 16
        [types.int32, types.float64],         # 4 + 4 pad + 8 = 16
        [types.int32, types.float64, types.int32],  # 4 + 4 pad + 8 + 4 pad = 24
        [types.int8, types.int64],            # 1 + 7 pad + 8 = 16
    ):
        rec = types.Record.make_c_struct(
            [(chr(ord("a") + i), t) for i, t in enumerate(fields)])
        assert _struct_bytes(types.Tuple(fields), "t") == _struct_bytes(rec, "t")


def test_classify_padded_tuple_is_struct_large():
    """Check that ``Tuple([int32, float64, int32])`` is LARGE, the class whose results return through ``sret``.

    The widths of the fields add up to 16 bytes, but with C's padding the struct is 24 bytes: an ``int32``, 4 bytes of
    padding, a ``float64``, an ``int32`` and 4 more bytes of padding.
    """
    from numba.core import types
    from numbox.core.bindings.abi import _classify, _CLASS_STRUCT_LARGE

    ty = types.Tuple([types.int32, types.float64, types.int32])
    assert _classify(ty) == _CLASS_STRUCT_LARGE


def test_classify_eightbytes_int_double_honors_padding():
    """Check that ``Tuple([int32, float64])`` has an INTEGER low half and an SSE high half.

    The ``float64`` starts at offset 8, after 4 bytes of padding, so it lies wholly in the high half.
    """
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_SSE, _classify_eightbytes,
    )

    ty = types.Tuple([types.int32, types.float64])
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_SSE,
    )


def test_struct_bytes_rejects_non_struct_type():
    """Check that ``_struct_bytes`` raises ``TypingError`` for a type that is not a struct, such as ``int32``."""
    from numba.core import types
    from numba.core.errors import TypingError
    from numbox.core.bindings.abi import _struct_bytes

    with pytest.raises(TypingError, match="struct-shaped type"):
        _struct_bytes(types.int32, "_call_lib_func_byval")


def test_struct_bytes_rejects_tuple_with_non_bitwidth_field():
    """Check that a tuple field of no fixed bit width, such as a ``voidptr``, raises ``TypingError``.

    The error is a ``TypingError``, not the ``AttributeError`` that a missing ``bitwidth`` attribute would raise.
    """
    from numba.core import types
    from numba.core.errors import TypingError
    from numbox.core.bindings.abi import _struct_bytes

    with pytest.raises(TypingError, match="bitwidth"):
        _struct_bytes(types.Tuple([types.intp, types.voidptr]), "_call_lib_func_byval")


def test_call_lib_func_lldiv_via_unified():
    """Check that ``_call_lib_func`` returns libc's ``lldiv(10, 3)``, a 16-byte ``lldiv_t``, as the tuple ``(3, 1)``.

    The result comes back in two general-purpose registers on SysV x86-64 and AAPCS64, and through ``sret``, a result
    slot that the caller allocates, on Windows x64. An error in the return path of any of the three conventions gives
    a wrong quotient or remainder.
    """
    from numba import njit
    from numbox.core.bindings.call import _call_lib_func

    @njit
    def run():
        return _call_lib_func("lldiv", (10, 3))

    quot, rem = run()
    assert quot == 3
    assert rem == 1


def test_call_lib_func_scalar_args_unchanged():
    """Check that ``_call_lib_func`` passes a scalar argument and returns a scalar result: ``cos(0.0)`` is 1.0.

    numbox's bindings pass only numbers and pointers, so an error in this path would break every binding.
    """
    from numba import njit
    from numbox.core.bindings.call import _call_lib_func

    @njit
    def run():
        return _call_lib_func("cos", (0.0,))

    assert run() == 1.0


def test_call_lib_func_scalar_arg_auto_wrapped():
    """Check that ``_call_lib_func`` takes a single argument without a tuple.

    ``_call_lib_func("cos", 0.0)`` returns the same value as ``_call_lib_func("cos", (0.0,))``.
    """
    from numba import njit
    from numbox.core.bindings.call import _call_lib_func

    @njit
    def run():
        return _call_lib_func("cos", 0.0)

    assert run() == 1.0


def _register_test_symbol(name):
    """Register a ``ctypes`` stub that returns 0 under ``name``, so that LLVM finds the symbol, and return the stub.

    The tests that call this helper compile a call of ``name`` and read the LLVM IR or expect a refusal; none of them
    runs the stub. The caller keeps the returned object alive while the symbol is registered, because ``ctypes``
    frees the code of a callback when its object is collected.
    """
    import ctypes
    import llvmlite.binding as ll

    @ctypes.CFUNCTYPE(ctypes.c_int32)
    def _stub():
        return 0

    addr = ctypes.cast(_stub, ctypes.c_void_p).value
    ll.add_symbol(name, addr)
    return _stub


@pytest.fixture
def patch_signature():
    """Yield ``register(name, sig)``, which adds or replaces an entry of the ``signatures`` table for one test.

    At teardown, the fixture restores each entry that ``register`` replaced and removes each entry that it added.
    """
    from numbox.core.bindings.signatures import signatures

    sentinel = object()
    saved = []

    def register(name, sig):
        saved.append((name, signatures.get(name, sentinel)))
        signatures[name] = sig

    yield register

    for name, prev in saved:
        if prev is sentinel:
            del signatures[name]
        else:
            signatures[name] = prev


def _platform_str():
    """Return the calling convention of the host for the skip conditions below, or ``"unknown"`` on another host.

    ``_current_platform`` raises ``RuntimeError`` on a host that ``abi`` does not support, and the skip conditions
    run when pytest imports this module.
    """
    from numbox.core.bindings.abi import _current_platform
    try:
        return _current_platform()
    except RuntimeError:
        return "unknown"


@pytest.mark.skipif(
    _platform_str() != "sysv_x86_64",
    reason="byval + optnone + noinline are SysV x86-64 specific",
)
def test_call_lib_func_byval_attribute_in_ir_for_large_struct(patch_signature):
    """Check the attributes of a call that passes a 24-byte struct on SysV x86-64: ``byval``, ``optnone``, ``noinline``.

    The argument gets ``byval``, so LLVM copies the struct onto the stack, as the convention requires. The function
    that makes the call gets ``optnone`` and ``noinline``, which keep LLVM's optimizer from removing the stack copy
    before C reads it. The test compiles the call and reads the IR; it never runs the C function.
    """
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_byval_large_24b"
    keepalive = _register_test_symbol(name)
    big_struct = nb_types.UniTuple(nb_types.int64, 3)
    patch_signature(name, nb_types.int32(big_struct))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    run.compile((nb_types.UniTuple(nb_types.int64, 3),))
    ir_text = list(run.inspect_llvm().values())[0]

    assert "byval(" in ir_text, (
        "expected 'byval(' attribute on >16B struct arg on SysV x86-64;\n"
        f"IR was:\n{ir_text}"
    )
    assert "optnone" in ir_text, (
        "expected 'optnone' on enclosing function on SysV x86-64"
    )
    assert "noinline" in ir_text, (
        "expected 'noinline' on enclosing function on SysV x86-64"
    )
    del keepalive


@pytest.mark.skipif(
    _platform_str() != "sysv_x86_64",
    reason="≤16B-struct passing differs across ABIs; SysV-specific check",
)
def test_call_lib_func_no_byval_attribute_for_small_struct(patch_signature):
    """Check that a 16-byte struct argument on SysV x86-64 is passed by value, without ``byval`` or ``optnone``.

    The function that makes the call is therefore optimized as usual. The test reads only the IR.
    """
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_byval_small_16b"
    keepalive = _register_test_symbol(name)
    small_struct = nb_types.UniTuple(nb_types.int64, 2)
    patch_signature(name, nb_types.int32(small_struct))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    run.compile((nb_types.UniTuple(nb_types.int64, 2),))
    ir_text = list(run.inspect_llvm().values())[0]

    assert "byval(" not in ir_text, (
        "did not expect 'byval(' on ≤16B struct arg on SysV x86-64;\n"
        f"IR was:\n{ir_text}"
    )
    assert "optnone" not in ir_text, (
        "did not expect 'optnone' on enclosing function for ≤16B struct"
    )
    del keepalive


@pytest.mark.skipif(
    _platform_str() != "win_x64",
    reason="Windows-x64-specific 1/2/4/8-byte register-passing rule",
)
def test_call_lib_func_8byte_struct_arg_on_windows_passes_by_value(patch_signature):
    """Check that an 8-byte struct argument on Windows x64 is declared as a value, not as a pointer to a copy.

    The Windows x64 convention passes a struct of 1, 2, 4 or 8 bytes in a register. The test reads only the
    declaration in the IR. Whether C receives the fields intact has not been measured on Windows.
    """
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_win_pass_8b"
    keepalive = _register_test_symbol(name)
    eight_byte_struct = nb_types.UniTuple(nb_types.int32, 2)
    patch_signature(name, nb_types.int32(eight_byte_struct))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    run.compile((nb_types.UniTuple(nb_types.int32, 2),))
    ir_text = list(run.inspect_llvm().values())[0]

    declare_line = next(
        (line for line in ir_text.splitlines() if name in line and "declare" in line),
        None,
    )
    assert declare_line is not None, (
        f"could not find declare line for {name} in IR:\n{ir_text}"
    )
    assert "*" not in declare_line.split("(")[1], (
        f"expected struct-by-value (no pointer) on Windows for 8B arg; "
        f"declare line was:\n{declare_line}"
    )
    del keepalive


@pytest.mark.skipif(
    _platform_str() != "win_x64",
    reason="Windows-x64-specific 1/2/4/8-byte register-return rule",
)
def test_call_lib_func_8byte_struct_return_on_windows_no_sret(patch_signature):
    """Check that an 8-byte struct result on Windows x64 is declared as a value, without ``sret``.

    The Windows x64 convention returns a struct of 1, 2, 4 or 8 bytes in RAX, and a struct of any other size through
    ``sret``. The test reads only the declaration in the IR.
    """
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_win_ret_8b"
    keepalive = _register_test_symbol(name)
    eight_byte_struct = nb_types.UniTuple(nb_types.int32, 2)
    patch_signature(name, eight_byte_struct(nb_types.int32))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    run.compile((nb_types.int32,))
    ir_text = list(run.inspect_llvm().values())[0]

    declare_line = next(
        (line for line in ir_text.splitlines() if "declare" in line and name in line),
        None,
    )
    assert declare_line is not None, (
        f"could not find declare line for {name} in IR:\n{ir_text}"
    )
    assert "sret" not in declare_line, (
        f"did not expect 'sret' on 8B struct return on Windows x64; "
        f"declare line was:\n{declare_line}"
    )
    del keepalive


def test_call_lib_func_undefined_signature_raises():
    """Check that ``_call_lib_func`` refuses a name that LLVM finds but the ``signatures`` table lacks.

    The test registers a stub under the name, so that the check for the symbol passes, and it expects an error whose
    message contains "Undefined signature" when numba compiles the call.
    """
    from numba import njit
    from numba.core.errors import TypingError
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_no_sig_unified"
    keepalive = _register_test_symbol(name)

    @njit
    def run():
        return _call_lib_func(name, (0.0,))

    with pytest.raises((ValueError, TypingError), match="Undefined signature"):
        run()
    del keepalive


def test_call_lib_func_int_int_struct_arg_round_trip(patch_signature):
    """Check that a 16-byte ``{i32, i32, i64}`` struct, the ``duckdb_interval`` layout, reaches a C callback intact.

    ``_call_lib_func`` repacks the struct to ``{i64, i64}`` on SysV x86-64 and AAPCS64, so that C receives each 8-byte
    half in one general-purpose register. Without the repack, LLVM passes each field in a register of its own. A C
    function on x86-64 Linux then receives ``(7, 0, 11)`` for the values ``(7, 11, 1000000)``. On Windows x64, which
    passes in registers only a struct of 1, 2, 4 or 8 bytes, ``_call_lib_func`` passes a pointer to a copy.
    """
    import ctypes
    import llvmlite.binding as ll
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    class _IntervalC(ctypes.Structure):
        _fields_ = [
            ("a", ctypes.c_int32),
            ("b", ctypes.c_int32),
            ("c", ctypes.c_int64),
        ]

    received = {}

    @ctypes.CFUNCTYPE(ctypes.c_int64, _IntervalC)
    def echo(s):
        received["a"] = s.a
        received["b"] = s.b
        received["c"] = s.c
        return s.a + s.b + s.c

    name = "numbox_test_int_int_eightbyte_round_trip"
    addr = ctypes.cast(echo, ctypes.c_void_p).value
    ll.add_symbol(name, addr)
    arg_struct = nb_types.Tuple(
        [nb_types.int32, nb_types.int32, nb_types.int64])
    patch_signature(name, nb_types.int64(arg_struct))

    @njit(nb_types.int64(nb_types.int32, nb_types.int32, nb_types.int64))
    def run(a, b, c):
        return _call_lib_func(name, ((a, b, c),))

    result = run(7, 11, 1_000_000)

    assert received == {"a": 7, "b": 11, "c": 1_000_000}, (
        f"second i32 field was dropped by SysV by-value lowering: {received}"
    )
    assert result == 7 + 11 + 1_000_000
    del echo  # echo stays referenced until after the jitted call.


@pytest.mark.skipif(
    _platform_str() not in ("sysv_x86_64", "aapcs64"),
    reason="return-side eightbyte repack only kicks in on SysV / AAPCS64",
)
def test_call_lib_func_int_int_return_uses_i64_pair_in_ir(patch_signature):
    """Check that a 16-byte ``{i32, i32, i64}`` result is declared as ``{i64, i64}`` on SysV x86-64 and AAPCS64.

    C returns each 8-byte half of such a struct in one general-purpose register, and LLVM reads a ``{i64, i64}``
    result from the same two registers. The generated code then copies the value back into the struct's own LLVM
    type. The test reads only the IR, because a ``ctypes`` callback cannot return a struct: ``CFUNCTYPE`` with a
    ``Structure`` result raises ``TypeError: invalid result type for callback function``. No test therefore runs
    ``_repack_from_i64_pair`` against C. ``test_call_lib_func_int_int_struct_arg_round_trip`` runs the argument side,
    ``_repack_to_i64_pair``.
    """
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_int_int_eightbyte_return_ir"
    keepalive = _register_test_symbol(name)
    ret_struct = nb_types.Tuple(
        [nb_types.int32, nb_types.int32, nb_types.int64])
    patch_signature(name, ret_struct(nb_types.int32))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    run.compile((nb_types.int32,))
    ir_text = list(run.inspect_llvm().values())[0]
    declare_line = next(
        (line for line in ir_text.splitlines()
         if "declare" in line and name in line),
        None,
    )
    assert declare_line is not None, (
        f"could not find declare line for {name} in IR:\n{ir_text}"
    )
    # The return type is the text between "declare" and the function's name. The repack makes it { i64, i64 }; the
    # struct's own type, { i32, i32, i64 }, would mean that the repack was not applied.
    decl_prefix = declare_line.split('@' + name)[0]
    assert "i64, i64" in decl_prefix, (
        f"expected return type to be repacked to '{{i64, i64}}'; "
        f"declare line was:\n{declare_line}"
    )
    assert "i32, i32, i64" not in decl_prefix, (
        f"return type still shows non-canonical '{{i32, i32, i64}}' "
        f"-- repack not applied; declare line was:\n{declare_line}"
    )
    del keepalive


def test_call_lib_func_four_i32_struct_arg_round_trip(patch_signature):
    """Check that a 16-byte ``UniTuple(int32, 4)`` struct, of LLVM type ``[4 x i32]``, reaches a C callback intact.

    ``_repack_to_i64_pair`` stores the value in a ``{i64, i64}`` slot, which is aligned to 8 bytes, and loads it back
    as ``{i64, i64}``. A slot of the struct's own type would be aligned to 4 bytes only, and the 8-byte-aligned load
    from it would be undefined behavior in LLVM. This path runs on SysV x86-64 and AAPCS64; Windows x64 passes the
    struct as a pointer to a copy. Without the repack, a C function on x86-64 Linux receives ``(101, 0, 202, 0)`` for
    the values ``(101, 202, 303, 404)``.
    """
    import ctypes
    import llvmlite.binding as ll
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    class _Quad32C(ctypes.Structure):
        _fields_ = [
            ("a", ctypes.c_int32),
            ("b", ctypes.c_int32),
            ("c", ctypes.c_int32),
            ("d", ctypes.c_int32),
        ]

    received = {}

    @ctypes.CFUNCTYPE(ctypes.c_int64, _Quad32C)
    def echo(s):
        received["a"] = s.a
        received["b"] = s.b
        received["c"] = s.c
        received["d"] = s.d
        return s.a + s.b + s.c + s.d

    name = "numbox_test_four_i32_round_trip"
    addr = ctypes.cast(echo, ctypes.c_void_p).value
    ll.add_symbol(name, addr)
    arg_struct = nb_types.UniTuple(nb_types.int32, 4)
    patch_signature(name, nb_types.int64(arg_struct))

    @njit(nb_types.int64(
        nb_types.int32, nb_types.int32, nb_types.int32, nb_types.int32))
    def run(a, b, c, d):
        return _call_lib_func(name, ((a, b, c, d),))

    result = run(101, 202, 303, 404)

    assert received == {"a": 101, "b": 202, "c": 303, "d": 404}, (
        f"4×i32 fields scrambled: {received}"
    )
    assert result == 101 + 202 + 303 + 404
    del echo  # echo stays referenced until after the jitted call.


@pytest.mark.skipif(
    _platform_str() != "sysv_x86_64",
    reason="repack-skip rules are SysV x86-64 specific",
)
def test_call_lib_func_sse_eightbyte_arg_not_repacked(patch_signature):
    """Check that a 16-byte struct of two ``float64`` fields is not repacked to ``{i64, i64}`` on SysV x86-64.

    The convention passes such a struct in XMM0 and XMM1, while LLVM would pass ``{i64, i64}`` in the general-purpose
    registers RDI and RSI. The declaration in the IR therefore keeps the ``double`` fields.
    """
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_sse_pair_no_repack"
    keepalive = _register_test_symbol(name)
    sse_struct = nb_types.UniTuple(nb_types.float64, 2)
    patch_signature(name, nb_types.int32(sse_struct))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    run.compile((sse_struct,))
    ir_text = list(run.inspect_llvm().values())[0]
    declare_line = next(
        (line for line in ir_text.splitlines()
         if "declare" in line and name in line),
        None,
    )
    assert declare_line is not None, (
        f"could not find declare line for {name} in IR:\n{ir_text}"
    )
    assert "double" in declare_line, (
        f"expected SSE pair to keep its double-typed arg in declare "
        f"(numba lowers UniTuple(float64, 2) to '[2 x double]'); "
        f"got:\n{declare_line}"
    )
    assert "i64" not in declare_line, (
        f"SSE pair must not be repacked to a 64-bit-integer-typed arg "
        f"(would force GP registers instead of XMM); declare line "
        f"was:\n{declare_line}"
    )
    del keepalive


def test_call_lib_func_canonical_int64_pair_round_trip(patch_signature):
    """Check that ``UniTuple(int64, 2)``, which needs no repack, reaches a C callback intact on every convention.

    ``_needs_int_int_eightbyte_repack`` skips this layout, because LLVM already passes two ``int64`` fields in two
    general-purpose registers. The test covers that by-value path on SysV x86-64 and AAPCS64, and the pointer to a
    copy on Windows x64. ``test_call_lib_func_lldiv_via_unified`` covers a result of the same layout. numbduck passes
    DuckDB's ``duckdb_hugeint`` and ``duckdb_uhugeint`` as tuples of two 64-bit integers, which take this path.
    """
    import ctypes
    import llvmlite.binding as ll
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    class _PairC(ctypes.Structure):
        _fields_ = [("lo", ctypes.c_int64), ("hi", ctypes.c_int64)]

    received = {}

    @ctypes.CFUNCTYPE(ctypes.c_int64, _PairC)
    def echo(s):
        received["lo"] = s.lo
        received["hi"] = s.hi
        return s.hi

    name = "numbox_test_canonical_i64_pair_round_trip"
    addr = ctypes.cast(echo, ctypes.c_void_p).value
    ll.add_symbol(name, addr)
    arg_struct = nb_types.UniTuple(nb_types.int64, 2)
    patch_signature(name, nb_types.int64(arg_struct))

    @njit(nb_types.int64(nb_types.int64, nb_types.int64))
    def run(lo, hi):
        return _call_lib_func(name, ((lo, hi),))

    result = run(0x0123456789ABCDEF, -42)

    assert received == {"lo": 0x0123456789ABCDEF, "hi": -42}
    assert result == -42
    del echo  # echo stays referenced until after the jitted call.


def test_call_lib_func_byval_undefined_signature_raises():
    """Check that ``_call_lib_func_byval`` refuses a name that the ``signatures`` table lacks.

    The error message contains "Undefined signature".
    """
    from numba import njit, types as nb_types
    from numba.core.errors import TypingError
    from numbox.core.bindings.call import _call_lib_func_byval

    name = "numbox_test_no_sig_byval"
    keepalive = _register_test_symbol(name)
    arg_struct = nb_types.UniTuple(nb_types.int64, 2)

    @njit
    def run(x):
        return _call_lib_func_byval(name, x)

    with pytest.raises((ValueError, TypingError), match="Undefined signature"):
        run.compile((arg_struct,))
    del keepalive


def test_call_lib_func_missing_llvm_symbol_raises(patch_signature):
    """Check that ``_call_lib_func`` refuses a name that the ``signatures`` table has but LLVM cannot find.

    The error message contains "unavailable in the LLVM context".
    """
    from numba import njit, types as nb_types
    from numba.core.errors import TypingError
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_no_llvm_symbol"
    patch_signature(name, nb_types.float64(nb_types.float64))

    @njit
    def run():
        return _call_lib_func(name, (0.0,))

    with pytest.raises((RuntimeError, TypingError), match="unavailable in the LLVM context"):
        run()


def test_call_lib_func_large_return_uses_sret_in_ir(patch_signature):
    """Check that a struct result over 16 bytes is declared as ``void``, with an ``sret`` first argument.

    The caller allocates the result and passes its address as a hidden first argument. ``_call_lib_func`` declares
    the call this way on SysV x86-64, AAPCS64 and Windows x64, which all return a struct over 16 bytes through memory.
    """
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_large_ret_sret_ir"
    keepalive = _register_test_symbol(name)
    big_ret = nb_types.UniTuple(nb_types.int64, 3)
    patch_signature(name, big_ret(nb_types.int32))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    run.compile((nb_types.int32,))
    ir_text = list(run.inspect_llvm().values())[0]
    declare_line = next(
        (line for line in ir_text.splitlines()
         if "declare" in line and name in line),
        None,
    )
    assert declare_line is not None, (
        f"could not find declare line for {name} in IR:\n{ir_text}"
    )
    assert "declare void" in declare_line, (
        f"expected 'declare void' for sret return; got:\n{declare_line}"
    )
    assert "sret(" in declare_line, (
        f"expected 'sret(' attribute on hidden first arg; got:\n{declare_line}"
    )
    del keepalive


@pytest.mark.skipif(
    _platform_str() not in ("sysv_x86_64", "win_x64"),
    reason=(
        "Round-trip relies on a Python ctypes callback receiving the "
        "sret hidden first arg in the same register as a normal first "
        "arg. True on SysV x86-64 (RDI) and Windows x64 (RCX). On "
        "AAPCS64 the sret pointer goes in x8 (indirect-result-location "
        "register), separate from x0 — ctypes thunks read args in "
        "x0/x1 order, so the buffer pointer never reaches the callback "
        "and the test segfaults. The IR-only tests still pin sret "
        "lowering on every platform; numbduck's real C consumers "
        "(duckdb_get_decimal/_varint) follow the AAPCS64 ABI directly "
        "and exercise this path on AAPCS64 in their own CI."
    ),
)
def test_call_lib_func_large_return_round_trip_unituple_24b(patch_signature):
    """Check that a 24-byte ``UniTuple(int64, 3)`` result returns intact through ``sret``.

    The ``ctypes`` callback has the signature of the call after the ``sret`` lowering, ``void(_BigC *, int64)``, and
    writes the three fields through the pointer to the result that the caller allocated. The test runs on SysV x86-64
    and Windows x64 and skips on AAPCS64; the skip reason explains why.
    """
    import ctypes
    import llvmlite.binding as ll
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    class _BigC(ctypes.Structure):
        _fields_ = [
            ("a", ctypes.c_int64),
            ("b", ctypes.c_int64),
            ("c", ctypes.c_int64),
        ]

    received = {}

    @ctypes.CFUNCTYPE(None, ctypes.POINTER(_BigC), ctypes.c_int64)
    def echo(out_p, x):
        out_p[0].a = x
        out_p[0].b = x + 1
        out_p[0].c = x + 2
        received["x"] = x

    name = "numbox_test_large_ret_unituple_3xi64"
    addr = ctypes.cast(echo, ctypes.c_void_p).value
    ll.add_symbol(name, addr)
    ret_struct = nb_types.UniTuple(nb_types.int64, 3)
    patch_signature(name, ret_struct(nb_types.int64))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    a, b, c = run(100)
    assert received == {"x": 100}
    assert (a, b, c) == (100, 101, 102)
    del echo  # echo stays referenced until after the jitted call.


def test_call_lib_func_large_return_record_rejected(patch_signature):
    """Check that ``_call_lib_func`` refuses a ``Record`` result over 16 bytes with a ``TypingError``.

    numba represents a record value as a pointer to its bytes, ``[N x i8]*``. A record result in an ``sret`` slot on
    the stack would therefore give the caller a pointer into the frame of a function that has returned. A tuple
    result of the same layout works, as the round-trip and IR tests above show.
    """
    from numba import njit, types as nb_types
    from numba.core.errors import TypingError
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_large_ret_record_rejected"
    keepalive = _register_test_symbol(name)
    rec_ty = nb_types.Record.make_c_struct([
        ("a", nb_types.int32),
        ("b", nb_types.int64),
        ("c", nb_types.int64),
    ])
    patch_signature(name, rec_ty(nb_types.int32))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    with pytest.raises(TypingError, match="Record returns >16 bytes"):
        run.compile((nb_types.int32,))
    del keepalive


@pytest.mark.skipif(
    _platform_str() not in ("sysv_x86_64", "win_x64"),
    reason=(
        "Same AAPCS64 sret/x8 ctypes-callback mismatch documented on "
        "the UniTuple round-trip. IR-only coverage of the LARGE-return "
        "path on AAPCS64 lives in the *_uses_sret_in_ir tests."
    ),
)
def test_call_lib_func_large_return_round_trip_varint_shaped(patch_signature):
    """Check that ``Tuple([intp, uint64, int8])``, numbduck's result type for ``duckdb_get_varint``, returns intact.

    numba lays out the tuple as a C compiler does: fields at offsets 0, 8 and 16, then 7 bytes of padding, 24 bytes in
    all. ``_call_lib_func`` therefore returns it through ``sret``. The ``ctypes`` structure sets ``_pack_ = 1`` and is
    17 bytes long, without the padding, but its fields have the same offsets, so the callback writes each field where
    the caller reads it. The test covers a large result whose fields differ in size and whose last field is 1 byte.
    """
    import ctypes
    import llvmlite.binding as ll
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    class _VarintC(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ("a", ctypes.c_int64),
            ("b", ctypes.c_uint64),
            ("c", ctypes.c_int8),
        ]
    assert ctypes.sizeof(_VarintC) == 17

    received = {}

    @ctypes.CFUNCTYPE(None, ctypes.POINTER(_VarintC), ctypes.c_int32)
    def echo(out_p, x):
        out_p[0].a = -x
        out_p[0].b = x * 7
        out_p[0].c = x % 128
        received["x"] = x

    name = "numbox_test_large_ret_varint_shaped"
    addr = ctypes.cast(echo, ctypes.c_void_p).value
    ll.add_symbol(name, addr)
    ret_struct = nb_types.Tuple([nb_types.intp, nb_types.uint64, nb_types.int8])
    patch_signature(name, ret_struct(nb_types.int32))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    a, b, c = run(13)
    assert received == {"x": 13}
    assert (a, b, c) == (-13, 91, 13)
    del echo  # echo stays referenced until after the jitted call.


def test_call_lib_func_large_return_decimal_shaped_uses_sret_in_ir(patch_signature):
    """Check that a ``Tuple([uint8, uint8, uint64, int64])`` result is declared as ``void``, with ``sret``.

    The tuple is numbduck's result type for ``duckdb_get_decimal``. numba places the two ``uint8`` fields at offsets 0
    and 1, the ``uint64`` at 8 and the ``int64`` at 16, as a C compiler does, so the tuple is 24 bytes, and
    ``_classify`` returns LARGE. The test reads only the IR;
    ``test_call_lib_func_large_return_round_trip_varint_shaped`` runs a large result through a C callback.
    """
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_large_ret_decimal_shape_ir"
    keepalive = _register_test_symbol(name)
    decimal_ret = nb_types.Tuple([
        nb_types.uint8, nb_types.uint8, nb_types.uint64, nb_types.int64])
    patch_signature(name, decimal_ret(nb_types.int32))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    run.compile((nb_types.int32,))
    ir_text = list(run.inspect_llvm().values())[0]
    declare_line = next(
        (line for line in ir_text.splitlines()
         if "declare" in line and name in line),
        None,
    )
    assert declare_line is not None, (
        f"could not find declare line for {name} in IR:\n{ir_text}"
    )
    assert "declare void" in declare_line, (
        f"expected 'declare void' for sret return; got:\n{declare_line}"
    )
    assert "sret(" in declare_line, (
        f"expected 'sret(' on hidden first arg; got:\n{declare_line}"
    )
    del keepalive
