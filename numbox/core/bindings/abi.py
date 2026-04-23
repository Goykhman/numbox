"""Struct-by-value ABI codegen helpers for numba bindings.

LLVM's JIT treats ABI lowering as a frontend responsibility — it won't
insert the right calling convention for struct args/returns by itself.
These helpers generate the appropriate IR for SysV x86-64 and Windows.

References:
    https://github.com/numba/llvmlite/issues/300#issuecomment-327235846
    https://github.com/llvm/llvm-project/issues/85417
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


_is_win = sys.platform == "win32"
_is_sysv_x86_64 = platform.machine() == "x86_64" and not _is_win


def _resolve_sig(func_name):
    func_sig = signatures.get(func_name, None)
    if func_sig is None:
        raise ValueError(f"Undefined signature for {func_name}")
    return func_sig


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


def _emit_byval_call(builder, context, arg, arg_ll_ty, ret_type, func_name):
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
    holds the struct as a value (e.g. ``duckdb_result *``).
    """
    func_name = func_name_ty.literal_value
    func_sig = _resolve_sig(func_name)

    def codegen(context, builder, signature, arguments):
        _, arg = arguments
        arg_ll_ty = context.get_value_type(arg_ty)
        ret_type = context.get_value_type(signature.return_type)
        return _emit_byval_call(
            builder, context, arg, arg_ll_ty, ret_type, func_name)

    sig = func_sig.return_type(func_name_ty, arg_ty)
    return sig, codegen


@intrinsic(prefer_literal=True)
def _call_lib_func_struct_in(typingctx, func_name_ty, arg_ty):
    """Pass a ≤16-byte struct: by value on SysV x86-64, by pointer elsewhere.

    Only SysV x86-64 has been validated for direct register passing of small
    structs through LLVM's JIT path. On Windows and any other platform
    (including aarch64), fall back to the always-correct by-pointer path
    via ``_emit_byval_call``.
    """
    func_name = func_name_ty.literal_value
    func_sig = _resolve_sig(func_name)
    struct_bytes = _struct_bytes(arg_ty, "_call_lib_func_struct_in")
    if struct_bytes > 16:
        raise TypingError(
            f"_call_lib_func_struct_in: struct too large for by-value "
            f"passing ({struct_bytes} bytes > 16)"
        )

    def codegen(context, builder, signature, arguments):
        _, arg = arguments
        arg_ll_ty = context.get_value_type(arg_ty)
        ret_type = context.get_value_type(signature.return_type)
        if not _is_sysv_x86_64:
            return _emit_byval_call(
                builder, context, arg, arg_ll_ty, ret_type, func_name)
        func_ty_ll = FunctionType(ret_type, [arg_ll_ty])
        func_p = get_or_insert_function(
            builder.module, func_ty_ll, func_name)
        return builder.call(func_p, [arg])

    sig = func_sig.return_type(func_name_ty, arg_ty)
    return sig, codegen


@intrinsic(prefer_literal=True)
def _call_lib_func_struct_out(typingctx, func_name_ty, arg_ty):
    """Return a ≤16-byte struct: by value on SysV x86-64, via sret elsewhere.

    Only SysV x86-64 has been validated for direct register-returning of small
    structs through LLVM's JIT path. On Windows and any other platform, fall
    back to the always-correct sret pattern (hidden first pointer arg).
    """
    func_name = func_name_ty.literal_value
    func_sig = _resolve_sig(func_name)
    ret_ty = func_sig.return_type
    struct_bytes = _struct_bytes(ret_ty, "_call_lib_func_struct_out")
    if struct_bytes > 16:
        raise TypingError(
            f"_call_lib_func_struct_out: return struct too large for "
            f"by-value return ({struct_bytes} bytes > 16)"
        )

    def codegen(context, builder, signature, arguments):
        _, arg = arguments
        ret_ll_ty = context.get_value_type(signature.return_type)
        if not _is_sysv_x86_64:
            sret_p = builder.alloca(ret_ll_ty)
            func_ty_ll = FunctionType(
                ir.VoidType(),
                [ret_ll_ty.as_pointer(), arg.type]
            )
            func_p = get_or_insert_function(
                builder.module, func_ty_ll, func_name)
            func_p.args[0].add_attribute("sret")
            builder.call(func_p, [sret_p, arg])
            return builder.load(sret_p)
        func_ty_ll = FunctionType(ret_ll_ty, [arg.type])
        func_p = get_or_insert_function(
            builder.module, func_ty_ll, func_name)
        return builder.call(func_p, [arg])

    sig = func_sig.return_type(func_name_ty, arg_ty)
    return sig, codegen
