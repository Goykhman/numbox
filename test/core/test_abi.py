import pytest


def test_abi_imports():
    """Public helper symbols are exported by their respective modules."""
    from numbox.core.bindings import abi, call

    assert hasattr(abi, "_struct_bytes")
    assert hasattr(abi, "_classify")
    assert hasattr(abi, "_current_platform")
    assert hasattr(call, "_call_lib_func")
    assert hasattr(call, "_call_lib_func_byval")


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
