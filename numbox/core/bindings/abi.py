"""Struct-by-value ABI codegen helpers for numba bindings.

LLVM's JIT treats ABI lowering as a frontend responsibility — it won't
insert the right calling convention for struct args/returns by itself.
These helpers generate the appropriate IR for the two ABI families that
matter for numba bindings: the Windows x64 ABI, which passes aggregates
>8 bytes via caller-allocated pointers and returns them via ``sret``;
and the register-passing ABIs (SysV x86-64 and AAPCS64), which pass and
return ≤16-byte aggregates directly in GP registers.

References:
    https://github.com/numba/llvmlite/issues/300#issuecomment-327235846
    https://github.com/llvm/llvm-project/issues/85417
    https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention
    https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst
"""
import sys

from llvmlite import ir
from llvmlite.ir import FunctionType
from numba.core import types as nb_types
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.extending import intrinsic

from numbox.core.bindings.signatures import signatures


_is_win = sys.platform == "win32"


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
    holds the struct as a value; the intrinsic allocates a stack slot,
    stores the value, and passes the slot's address.
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
    """Pass a ≤16-byte struct: by pointer on Windows x64, by value elsewhere.

    Windows x64 passes aggregates >8 bytes via a caller-allocated pointer.
    SysV x86-64 and AAPCS64 both pass ≤16-byte composites directly in GP
    registers; LLVM's JIT lowers a struct-typed argument to the correct
    register assignment per target. So only Windows takes the by-pointer
    path via ``_emit_byval_call``; every other platform passes directly.
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
        if _is_win:
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
    """Return a ≤16-byte struct: via sret on Windows x64, by value elsewhere.

    Windows x64 returns aggregates >8 bytes via ``sret`` (hidden first
    pointer arg, void return). SysV x86-64 and AAPCS64 both return
    ≤16-byte composites directly in GP registers. So only Windows takes
    the sret path; every other platform returns the struct directly.
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
        if _is_win:
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


@intrinsic(prefer_literal=True)
def _call_lib_func_args_struct_out(typingctx, func_name_ty, args_ty):
    """Call ``ret_struct func(scalars...)`` returning a ≤16-byte struct.

    Mirrors ``_call_lib_func_struct_out`` return-side ABI gating (sret on
    Windows x64, by value elsewhere) but takes a tuple of scalar args
    instead of a single struct arg — for libc/libm functions like
    ``lldiv(long long, long long) -> lldiv_t`` that exercise return-side
    ABI gating without needing a library with struct-by-value entry points.
    """
    func_name = func_name_ty.literal_value
    func_sig = _resolve_sig(func_name)
    ret_ty = func_sig.return_type
    struct_bytes = _struct_bytes(ret_ty, "_call_lib_func_args_struct_out")
    if struct_bytes > 16:
        raise TypingError(
            f"_call_lib_func_args_struct_out: return struct too large for "
            f"by-value return ({struct_bytes} bytes > 16)"
        )

    def codegen(context, builder, signature, arguments):
        _, args_tuple = arguments
        ret_ll_ty = context.get_value_type(signature.return_type)
        args_ll = []
        for arg_ind, _ in enumerate(args_ty):
            args_ll.append(builder.extract_value(args_tuple, arg_ind))
        arg_ll_tys = [a.type for a in args_ll]
        if _is_win:
            sret_p = builder.alloca(ret_ll_ty)
            func_ty_ll = FunctionType(
                ir.VoidType(),
                [ret_ll_ty.as_pointer()] + arg_ll_tys
            )
            func_p = get_or_insert_function(
                builder.module, func_ty_ll, func_name)
            func_p.args[0].add_attribute("sret")
            builder.call(func_p, [sret_p] + args_ll)
            return builder.load(sret_p)
        func_ty_ll = FunctionType(ret_ll_ty, arg_ll_tys)
        func_p = get_or_insert_function(
            builder.module, func_ty_ll, func_name)
        return builder.call(func_p, args_ll)

    sig = func_sig.return_type(func_name_ty, args_ty)
    return sig, codegen
