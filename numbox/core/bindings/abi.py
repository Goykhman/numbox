"""ABI codegen primitives shared by numba bindings.

Platform identification (``_current_platform``), struct-shape
classification (``_classify``), struct-size measurement
(``_struct_bytes``), and the unconditional ``func(T*)`` byval helper
(``_call_lib_func_byval``) — used by ``numbox.core.bindings.call.
_call_lib_func`` to dispatch C-function calls per platform and per
struct shape.

The per-platform ABI dispatch table — Windows x64 (which passes >8B
aggregates via caller-allocated pointers and returns them via
``sret``) vs SysV x86-64 / AAPCS64 (which pass and return ≤16B
aggregates directly in GP registers, with a ``byval`` + ``optnone``
+ ``noinline`` idiom for >16B by-value args on SysV x86-64) — lives
in ``call.py`` itself.

References:
    https://github.com/numba/llvmlite/issues/300#issuecomment-327235846
    https://github.com/llvm/llvm-project/issues/85417
    https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention
    https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst
"""
import platform
import sys

from llvmlite import ir
from llvmlite.ir import FunctionType
from numba.core import types as nb_types
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.extending import intrinsic

from numbox.core.bindings.signatures import signatures


_PLATFORM_WIN_X64 = "win_x64"
_PLATFORM_SYSV_X86_64 = "sysv_x86_64"
_PLATFORM_AAPCS64 = "aapcs64"


def _current_platform():
    """Identify the C calling convention for the current host.

    Returns one of ``_PLATFORM_WIN_X64``, ``_PLATFORM_SYSV_X86_64``,
    ``_PLATFORM_AAPCS64``. Used by the ABI-aware codegen in
    ``numbox.core.bindings.call._call_lib_func`` to pick the right
    struct-passing convention. Raises ``RuntimeError`` on unsupported
    ``(sys.platform, platform.machine())`` combinations rather than
    silently misclassifying — Windows ARM64 (`platform.machine() ==
    "ARM64"`) is unsupported and would otherwise default to the wrong
    ABI dispatch.
    """
    machine = platform.machine().lower()
    if sys.platform == "win32":
        if machine in ("x86_64", "amd64"):
            return _PLATFORM_WIN_X64
        raise RuntimeError(
            f"Unsupported Windows architecture for ABI dispatch: "
            f"{platform.machine()} (only x86_64 / AMD64 supported)"
        )
    if machine in ("x86_64", "amd64"):
        return _PLATFORM_SYSV_X86_64
    if machine in ("arm64", "aarch64"):
        return _PLATFORM_AAPCS64
    raise RuntimeError(
        f"Unsupported platform for ABI dispatch: "
        f"{sys.platform}/{platform.machine()}"
    )


_CLASS_SCALAR = "scalar"
_CLASS_STRUCT_SMALL = "struct_small"
_CLASS_STRUCT_LARGE = "struct_large"


_WIN_REGISTER_PASSABLE_SIZES = (1, 2, 4, 8)


def _is_windows_register_passable(struct_bytes):
    """Whether a struct of size ``struct_bytes`` is passed/returned in
    registers on the Windows x64 ABI.

    Windows x64 passes aggregates of size 1, 2, 4, or 8 bytes directly
    (in integer registers) and returns them in RAX. Other sizes (3, 5,
    6, 7, or anything > 8) go via caller-allocated pointer for args and
    via ``sret`` for returns.
    """
    return struct_bytes in _WIN_REGISTER_PASSABLE_SIZES


def _classify(ty):
    """Classify a numba type for ABI dispatch.

    Returns one of:

    - ``_CLASS_SCALAR`` — any non-struct numba type (e.g. ``int32``,
      ``float64``, pointers represented as ``intp``).
    - ``_CLASS_STRUCT_SMALL`` — ``Record`` / ``BaseTuple`` of size
      <= 16 bytes; passed by value on register-passing ABIs.
    - ``_CLASS_STRUCT_LARGE`` — ``Record`` / ``BaseTuple`` of size
      > 16 bytes; passed by pointer with ``byval`` on SysV x86-64,
      by pointer (no special attribute) on other ABIs.
    """
    if not isinstance(ty, (nb_types.Record, nb_types.BaseTuple)):
        return _CLASS_SCALAR
    size = _struct_bytes(ty, "_classify")
    return _CLASS_STRUCT_SMALL if size <= 16 else _CLASS_STRUCT_LARGE


def _struct_bytes(ty, fn_name):
    """Compute struct size in bytes for a numba struct-shaped type.

    Supports ``types.Record`` (via ``.size``) and ``types.BaseTuple``
    subclasses — ``types.Tuple``, ``types.UniTuple``, and
    ``types.NamedTuple`` — via ``.types`` (a tuple of scalar field
    types with ``.bitwidth``). Anything else raises a clean
    ``TypingError`` naming the caller so a misuse doesn't surface as a
    cryptic ``AttributeError`` / ``KeyError``.
    """
    if isinstance(ty, nb_types.Record):
        return ty.size
    if isinstance(ty, nb_types.BaseTuple):
        return sum(t.bitwidth for t in ty.types) // 8
    raise TypingError(
        f"{fn_name}: expected a struct-shaped type (Record, Tuple, "
        f"UniTuple, or NamedTuple), got {ty!r}."
    )


def _emit_byval_call(builder, arg, arg_ll_ty, ret_type, func_name):
    """Emit IR to pass a struct by pointer: alloca, store, call via pointer."""
    stack_p = builder.alloca(arg_ll_ty)
    builder.store(arg, stack_p)
    func_ty_ll = FunctionType(ret_type, [arg_ll_ty.as_pointer()])
    func_p = get_or_insert_function(builder.module, func_ty_ll, func_name)
    return builder.call(func_p, [stack_p])


@intrinsic(prefer_literal=True)
def _call_lib_func_byval(typingctx, func_name_ty, arg_ty):
    """Pass ``arg`` to a C function by pointer on all platforms.

    Used when the C signature takes a pointer to a struct and the caller
    holds the struct as a value; the intrinsic allocates a stack slot,
    stores the value, and passes the slot's address.
    """
    func_name = func_name_ty.literal_value
    func_sig = signatures.get(func_name, None)
    if func_sig is None:
        raise ValueError(f"Undefined signature for {func_name}")

    def codegen(context, builder, signature, arguments):
        _, arg = arguments
        arg_ll_ty = context.get_value_type(arg_ty)
        ret_type = context.get_value_type(signature.return_type)
        return _emit_byval_call(
            builder, arg, arg_ll_ty, ret_type, func_name)

    sig = func_sig.return_type(func_name_ty, arg_ty)
    return sig, codegen
