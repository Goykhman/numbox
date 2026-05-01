import pytest


def test_abi_imports():
    """Public helper symbols are exported by their respective modules."""
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
    """`Tuple([int32, int32, int64])` (the layout of duckdb_interval) has
    two pure-INTEGER eightbytes. The lo eightbyte holds two i32 fields
    (offsets 0, 4); the hi eightbyte holds one i64 (offset 8)."""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _classify_eightbytes,
    )

    ty = types.Tuple([types.int32, types.int32, types.int64])
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_int_int_canonical_pair():
    """`UniTuple(int64, 2)` has the canonical `{i64, i64}` layout —
    INT/INT eightbytes, no repack needed."""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _classify_eightbytes,
    )

    ty = types.UniTuple(types.int64, 2)
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_four_i32():
    """`UniTuple(int32, 4)` has fields at offsets 0/4/8/12 — both
    eightbytes are pure INTEGER, but the LLVM type isn't `{i64, i64}`."""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _classify_eightbytes,
    )

    ty = types.UniTuple(types.int32, 4)
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_sse_sse():
    """`UniTuple(float64, 2)` has SSE eightbytes — lowered to XMM0/XMM1
    on SysV x86-64. Repack to `{i64, i64}` would be wrong."""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_SSE, _classify_eightbytes,
    )

    ty = types.UniTuple(types.float64, 2)
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_SSE, _EIGHTBYTE_CLASS_SSE,
    )


def test_classify_eightbytes_sse_int():
    """`Tuple([float32, float32, int64])` has SSE in lo (two f32s),
    INT in hi (one i64)."""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_SSE,
        _classify_eightbytes,
    )

    ty = types.Tuple([types.float32, types.float32, types.int64])
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_SSE, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_mixed_lo_eightbyte_is_sse():
    """SysV rule: if any field in an eightbyte is SSE, the whole
    eightbyte is SSE. `Tuple([int32, float32, int64])` has int+float
    in lo → lo eightbyte is SSE."""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_SSE,
        _classify_eightbytes,
    )

    ty = types.Tuple([types.int32, types.float32, types.int64])
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_SSE, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_field_spans_eightbyte_boundary():
    """An SSE field that straddles the 8-byte boundary makes BOTH
    eightbytes SSE. ``Tuple([int32, float64, int32])`` has the
    ``float64`` at offsets [4, 12) — touching both lo and hi
    eightbytes — so the result is SSE/SSE per the SysV any-SSE-wins
    rule, not SSE/INT. (Bytes 0–3 are i32 INTEGER but the SSE field
    touching bytes 4–7 makes lo SSE; bytes 8–11 are part of the same
    SSE field, making hi SSE; bytes 12–15 are i32 INTEGER but hi is
    already SSE.)"""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_SSE, _classify_eightbytes,
    )

    ty = types.Tuple([types.int32, types.float64, types.int32])
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_SSE, _EIGHTBYTE_CLASS_SSE,
    )


def test_classify_eightbytes_record_with_padding():
    """`Record.make_c_struct([(a, int32), (b, int64)])` is 16B with a
    4-byte gap (i32@0, pad, i64@8). Both eightbytes are pure INTEGER."""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _classify_eightbytes,
    )

    ty = types.Record.make_c_struct([("a", types.int32), ("b", types.int64)])
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_rejects_non_16b():
    """The classifier is only meaningful for 16-byte aggregates (the size
    where SysV may pass two eightbytes by-value in registers). Non-16B
    inputs raise a clean `TypingError`."""
    from numba.core import types
    from numba.core.errors import TypingError
    from numbox.core.bindings.abi import _classify_eightbytes

    with pytest.raises(TypingError, match="16-byte"):
        _classify_eightbytes(types.UniTuple(types.int64, 3))  # 24B
    with pytest.raises(TypingError, match="16-byte"):
        _classify_eightbytes(types.UniTuple(types.int32, 2))  # 8B


def test_classify_eightbytes_rejects_non_struct():
    """Scalar (non-struct) types raise a clean `TypingError`."""
    from numba.core import types
    from numba.core.errors import TypingError
    from numbox.core.bindings.abi import _classify_eightbytes

    with pytest.raises(TypingError, match="struct-shaped"):
        _classify_eightbytes(types.int64)


def test_is_canonical_int64_pair_layout_true_cases():
    """`UniTuple(int64, 2)`, `Tuple([int64, int64])`, and `Tuple([uint64,
    intp])` all lower to LLVM `{i64, i64}` — no repack needed."""
    from numba.core import types
    from numbox.core.bindings.abi import _is_canonical_int64_pair_layout

    assert _is_canonical_int64_pair_layout(types.UniTuple(types.int64, 2))
    assert _is_canonical_int64_pair_layout(
        types.Tuple([types.int64, types.int64]))
    assert _is_canonical_int64_pair_layout(
        types.Tuple([types.uint64, types.intp]))


def test_is_canonical_int64_pair_layout_false_cases():
    """Anything not exactly two 64-bit integer fields at offsets 0/8 is
    non-canonical — including `{i32, i32, i64}` (the duckdb_interval
    layout that needs repack), `{i32 × 4}`, `{f64, f64}`, and 24-byte
    aggregates."""
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
    """`Record.make_c_struct([(a, int64), (b, int64)])` lowers to
    `{i64, i64}` — canonical. With smaller fields the LLVM type is
    different and the helper returns False."""
    from numba.core import types
    from numbox.core.bindings.abi import _is_canonical_int64_pair_layout

    rec_canonical = types.Record.make_c_struct([
        ("a", types.int64), ("b", types.int64)])
    assert _is_canonical_int64_pair_layout(rec_canonical)

    rec_non_canonical = types.Record.make_c_struct([
        ("a", types.int32), ("b", types.int64)])
    assert not _is_canonical_int64_pair_layout(rec_non_canonical)


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
    from numbox.core.bindings.call import _call_lib_func

    @njit
    def run():
        return _call_lib_func("lldiv", (10, 3))

    quot, rem = run()
    assert quot == 3
    assert rem == 1


def test_call_lib_func_scalar_args_unchanged():
    """Regression guard: scalar args + scalar return path goes through
    `_call_lib_func` unchanged from the pre-unification behavior.

    `cos(0.0)` from libm returns `1.0`. If the rewrite of `_call_lib_func`
    broke the scalar path that math / c / sqlite bindings depend on,
    this fails with an LLVM IR error or a wrong return value.
    """
    from numba import njit
    from numbox.core.bindings.call import _call_lib_func

    @njit
    def run():
        return _call_lib_func("cos", (0.0,))

    assert run() == 1.0


def test_call_lib_func_scalar_arg_auto_wrapped():
    """A single non-tuple arg is auto-wrapped into a 1-tuple at the
    intrinsic boundary, so `_call_lib_func("cos", 0.0)` is equivalent
    to `_call_lib_func("cos", (0.0,))`.
    """
    from numba import njit
    from numbox.core.bindings.call import _call_lib_func

    @njit
    def run():
        return _call_lib_func("cos", 0.0)

    assert run() == 1.0


def _register_test_symbol(name):
    """Register a no-op address under ``name`` so ``ll.address_of_symbol``
    finds something for the IR-inspection tests. The body is never
    executed — the tests only inspect the LLVM IR emitted at compile
    time. Returns the ctypes wrapper, which the caller must keep alive
    for the symbol to remain valid.
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
    """Add a temporary entry to ``signatures`` and remove it after.

    Yields a function ``register(name, sig)`` that the test calls to
    install a fake signature. The fixture undoes the install on teardown,
    even if the install replaced an existing entry.
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
    """On SysV x86-64, a 24-byte struct arg is lowered with ``byval``
    on the LLVM parameter and ``optnone`` + ``noinline`` on the
    enclosing function. The actual C function is never called — the
    test only inspects the IR emitted by numba.
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
    """On SysV x86-64, a ≤16B struct arg is passed by value in
    registers; LLVM lowers without a ``byval`` attribute and without
    forcing ``optnone`` / ``noinline``.
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
    """On Windows x64, an 8-byte struct arg is passed by value in
    registers (1/2/4/8-byte aggregates take the register-passing path).
    The LLVM IR should NOT alloca + pass-by-pointer this case.
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
    """On Windows x64, an 8-byte struct return goes directly in RAX —
    no ``sret`` slot, no void return. Sizes outside {1, 2, 4, 8} use
    sret; this test pins the small-size special case.
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
    """`_call_lib_func` raises when the function name has an LLVM symbol
    but is missing from the `signatures` dict.
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
    """Round-trip a 16-byte ``{i32, i32, i64}`` struct (the
    ``duckdb_interval`` shape) through ``_call_lib_func`` on every
    supported ABI.

    On SysV x86-64 the by-value path requires the new INT/INT
    eightbyte repack — without it, llvmlite drops fields. On Windows
    x64, 16B falls outside the ``{1, 2, 4, 8}`` register-passable
    set so the call goes via alloca + pointer-pass. On AAPCS64 the
    by-value path passes the struct in ``X0`` / ``X1`` directly.

    If this test fails on AAPCS64 (ubuntu-arm or macOS-ARM64), that
    indicates a remaining ABI/lowering issue despite the existing
    INT/INT eightbyte repack handling for that platform.
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
    del echo  # keepalive


@pytest.mark.skipif(
    _platform_str() not in ("sysv_x86_64", "aapcs64"),
    reason="return-side eightbyte repack only kicks in on SysV / AAPCS64",
)
def test_call_lib_func_int_int_return_uses_i64_pair_in_ir(patch_signature):
    """For a 16B non-canonical INT/INT return on SysV x86-64 / AAPCS64,
    the LLVM IR must declare the C function as returning ``{i64, i64}``,
    not ``{i32, i32, i64}`` — sidestepping llvmlite's eightbyte-
    packing gap on the return side. The numba-side return value is
    then unpacked back to the original LLVM type via memory bitcast.

    IR-only check rather than a true round-trip because Python ctypes
    ``CFUNCTYPE`` callbacks cannot return ``Structure`` by value
    ("invalid result type for callback function") — there's no clean
    way to author a Python-side producer that returns a ``{i32, i32,
    i64}`` struct via the platform ABI for a JIT'd caller to consume.
    The arg-side counterpart
    (``test_call_lib_func_int_int_struct_arg_round_trip``) already
    provides end-to-end validation of the symmetric
    ``_repack_to_i64_pair`` + ``_repack_from_i64_pair`` machinery.
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
    # The return type appears between `declare` and the function name.
    # We want `{ i64, i64 }` (the canonical eightbyte pair), NOT
    # `{ i32, i32, i64 }` (which would be llvmlite emitting the
    # non-canonical layout that drops fields).
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
    """Round-trip a 16-byte ``UniTuple(int32, 4)`` struct (LLVM type
    ``[4 x i32]``, 4-byte alignment) through ``_call_lib_func`` on
    every supported ABI.

    On SysV x86-64 / AAPCS64 this exercises the alignment-safe repack
    path: ``_repack_to_i64_pair`` allocates a well-aligned ``{i64, i64}``
    slot rather than an under-aligned ``[4 x i32]`` slot, so the
    final ``{i64, i64}`` load doesn't trip undefined behavior on
    stricter targets / optimization levels. On Windows the 16B path
    goes via alloca + pointer-pass.
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
    del echo  # keepalive


@pytest.mark.skipif(
    _platform_str() != "sysv_x86_64",
    reason="repack-skip rules are SysV x86-64 specific",
)
def test_call_lib_func_sse_eightbyte_arg_not_repacked(patch_signature):
    """A 16B ``{double, double}`` arg has SSE eightbytes — passed in
    XMM0/XMM1 on SysV x86-64. It must NOT be repacked to ``{i64, i64}``
    (which would force GP registers RDI/RSI), so the LLVM declare line
    keeps the ``double, double`` shape rather than ``i64, i64``.
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
    """A canonical 16B ``UniTuple(int64, 2)`` round-trips correctly
    through ``_call_lib_func`` on every supported ABI — regression
    guard that the canonical-skip in
    ``_needs_int_int_eightbyte_repack`` doesn't break the by-value
    path on SysV x86-64 / AAPCS64 or the alloca + pointer-pass path
    on Windows x64. This is the arg-side complement to the existing
    ``test_call_lib_func_lldiv_via_unified`` (which exercises the
    return side of canonical 16B). Numbduck's ``duckdb_hugeint`` and
    ``duckdb_uhugeint`` bind wrappers will rely on this path.
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
    del echo  # keepalive


def test_call_lib_func_byval_undefined_signature_raises():
    """`_call_lib_func_byval` raises when the function name is missing
    from the `signatures` dict.
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
    """`_call_lib_func` raises when the function name is in the
    `signatures` dict but has no LLVM symbol registered.
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
    """A >16-byte struct return is lowered as ``sret``: the LLVM declare
    line for the C symbol must be ``void`` returning, with the hidden
    first arg flagged ``sret`` (caller-allocated buffer pointer). Same
    pattern on every platform — SysV x86-64, AAPCS64, and Windows x64
    all use indirect-result-location for >16-byte aggregates.
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
    """Round-trip a 24-byte ``UniTuple(int64, 3)`` return through
    ``_call_lib_func``. The test C callback is declared as
    ``void(_BigC*, int64)`` -- exactly the sret-lowered shape -- and
    writes three i64 fields into the caller-allocated buffer. Verifies
    the full sret round-trip on SysV x86-64 and Windows x64: alloca,
    hidden first arg, callee write-through-pointer, caller load.
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
    del echo  # keepalive


def test_call_lib_func_large_return_record_rejected(patch_signature):
    """A >16-byte ``Record`` return is explicitly rejected with
    ``TypingError``. Numba's ``RecordModel`` represents values as raw
    ``[N x i8]*`` pointers, so the natural stack-alloca sret slot
    would dangle after the ``@njit`` function returns and Python-side
    boxing would dereference freed memory. Safe support needs NRT-
    allocated storage hooked into numba's record-ownership model -- a
    larger lift than this PR takes on, with no current consumer to
    validate against (numbduck uses ``Tuple``, not ``Record``).
    Tuple/UniTuple returns with the same byte layout work end-to-end
    via the round-trip and ``uses_sret_in_ir`` tests above.
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
    """Round-trip a 17-byte varint-shaped ``Tuple([intp, uint64, int8])``
    return -- the layout numbduck uses for ``duckdb_get_varint``. Numba
    ``BaseTuple`` is bit-packed (8+8+1=17B no padding), so the matching
    ctypes side uses ``_pack_=1`` to suppress the trailing alignment
    padding a default C struct would add (which would round to 24B).
    Pins the LARGE-return path on a non-uniform struct ending in a
    1-byte field.
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
    del echo  # keepalive


def test_call_lib_func_large_return_decimal_shaped_uses_sret_in_ir(patch_signature):
    """An 18-byte decimal-shaped ``Tuple([uint8, uint8, uint64, int64])``
    return -- the layout numbduck uses for ``duckdb_get_decimal`` -- is
    classified as ``_CLASS_STRUCT_LARGE`` (size > 16) and lowered via
    sret. IR-only check; numbduck's bit-packed tuple layout doesn't
    match any standard C struct alignment for this shape, so a real
    round-trip would require ``_pack_=1``. The varint test already
    exercises the value-path; this test just pins the codegen for the
    decimal shape numbduck specifically depends on.
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
