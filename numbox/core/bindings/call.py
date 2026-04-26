import llvmlite.binding as ll
from llvmlite import ir as llir

from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import BaseTuple, NoneType
from numba.extending import intrinsic

from numbox.core.bindings.abi import (
    _CLASS_SCALAR, _CLASS_STRUCT_SMALL, _CLASS_STRUCT_LARGE,
    _PLATFORM_AAPCS64, _PLATFORM_SYSV_X86_64, _PLATFORM_WIN_X64,
    _classify, _current_platform, _is_windows_register_passable,
    _struct_bytes,
)
from numbox.core.bindings.signatures import signatures


@intrinsic(prefer_literal=True)
def _call_lib_func(typingctx, func_name_ty, args_ty=NoneType):
    """Call a C library function with ABI-correct argument and return passing.

    The C function name is resolved from numbox's ``signatures`` dict.
    Each arg in ``args_ty`` and the resolved return type are classified
    as scalar / struct <= 16 bytes / struct >16 bytes, then lowered to
    LLVM IR per the host's calling convention:

    - **Scalar args / returns** -- passed and returned directly.
    - **<=16-byte struct args** -- by value on SysV x86-64 and AAPCS64
      (LLVM's frontend lowers to register passing); on Windows x64,
      sizes 1/2/4/8 are passed by value in registers (per the Windows
      x64 ABI) and other sizes go by pointer (alloca + store + pass-
      pointer).
    - **>16-byte struct args** -- by pointer on every platform; on SysV
      x86-64 the ``byval`` attribute is added to the LLVM arg and the
      enclosing function gets ``optnone`` + ``noinline`` so the LLVM
      optimizer does not elide the caller-side stack copy before the
      callee reads it. See:
      https://github.com/numba/llvmlite/issues/300#issuecomment-327235846
    - **<=16-byte struct returns** -- direct on SysV x86-64 and AAPCS64;
      on Windows x64, sizes 1/2/4/8 return in RAX (direct) and other
      sizes use ``sret`` (caller-allocated hidden first arg, void
      return).
    - **>16-byte struct returns** -- raise ``TypingError``. No consumer
      currently needs this; add it when one does.

    For C signatures of form ``func(T*)`` (pointer to struct) rather
    than ``func(T)`` lowered to a byval pointer by the ABI, use the
    sibling ``_call_lib_func_byval`` intrinsic in this module instead.
    The numba type system can't disambiguate ``T`` from ``T*``; the
    caller picks the intrinsic based on what the C header declares.
    """
    func_name = func_name_ty.literal_value
    func_p_as_int = ll.address_of_symbol(func_name)
    if func_p_as_int is None:
        raise RuntimeError(f"{func_name} is unavailable in the LLVM context")
    func_sig = signatures.get(func_name, None)
    if func_sig is None:
        raise ValueError(f"Undefined signature for {func_name}")

    ret_ty = func_sig.return_type
    ret_class = _classify(ret_ty)
    if ret_class == _CLASS_STRUCT_LARGE:
        raise TypingError(
            f"_call_lib_func: return struct >16 bytes is unsupported "
            f"({func_name})"
        )

    if args_ty == NoneType:
        arg_types = ()
        arg_classes = ()
    elif isinstance(args_ty, BaseTuple):
        arg_types = tuple(args_ty)
        arg_classes = tuple(_classify(at) for at in arg_types)
    else:
        arg_types = (args_ty,)
        arg_classes = (_classify(args_ty),)

    plat = _current_platform()
    use_sret = (
        ret_class == _CLASS_STRUCT_SMALL
        and plat == _PLATFORM_WIN_X64
        and not _is_windows_register_passable(_struct_bytes(ret_ty, "_call_lib_func"))
    )

    def codegen(context, builder, signature, arguments):
        if args_ty == NoneType:
            arg_vals = ()
        elif isinstance(args_ty, BaseTuple):
            _, args_pack = arguments
            arg_vals = tuple(
                builder.extract_value(args_pack, i)
                for i in range(len(arg_types))
            )
        else:
            _, scalar_arg = arguments
            arg_vals = (scalar_arg,)

        ret_ll_ty = context.get_value_type(ret_ty)

        ll_arg_tys = []
        ll_arg_vals = []
        byval_arg_indices = []

        if use_sret:
            sret_ptr = builder.alloca(ret_ll_ty)
            ll_arg_tys.append(ret_ll_ty.as_pointer())
            ll_arg_vals.append(sret_ptr)
        else:
            sret_ptr = None

        for arg_ty, arg_cls, val in zip(arg_types, arg_classes, arg_vals):
            arg_ll_ty = context.get_value_type(arg_ty)
            if arg_cls == _CLASS_SCALAR:
                ll_arg_tys.append(arg_ll_ty)
                ll_arg_vals.append(val)
                continue
            pass_by_value = arg_cls == _CLASS_STRUCT_SMALL and (
                plat in (_PLATFORM_SYSV_X86_64, _PLATFORM_AAPCS64)
                or (
                    plat == _PLATFORM_WIN_X64
                    and _is_windows_register_passable(
                        _struct_bytes(arg_ty, "_call_lib_func")
                    )
                )
            )
            if pass_by_value:
                ll_arg_tys.append(arg_ll_ty)
                ll_arg_vals.append(val)
                continue
            stack_p = builder.alloca(arg_ll_ty)
            builder.store(val, stack_p)
            ll_arg_tys.append(arg_ll_ty.as_pointer())
            ll_arg_vals.append(stack_p)
            if arg_cls == _CLASS_STRUCT_LARGE and plat == _PLATFORM_SYSV_X86_64:
                byval_arg_indices.append(len(ll_arg_vals) - 1)

        if use_sret:
            func_ll_ty = llir.FunctionType(llir.VoidType(), ll_arg_tys)
        else:
            func_ll_ty = llir.FunctionType(ret_ll_ty, ll_arg_tys)
        func_p = get_or_insert_function(builder.module, func_ll_ty, func_name)

        if use_sret:
            func_p.args[0].add_attribute("sret")
        for idx in byval_arg_indices:
            func_p.args[idx].add_attribute("byval")
        if byval_arg_indices:
            builder.function.attributes.add("optnone")
            builder.function.attributes.add("noinline")

        if use_sret:
            builder.call(func_p, ll_arg_vals)
            return builder.load(sret_ptr)
        return builder.call(func_p, ll_arg_vals)

    sig = ret_ty(func_name_ty, args_ty)
    return sig, codegen


def _emit_byval_call(builder, arg, arg_ll_ty, ret_type, func_name):
    """Emit IR to pass a struct by pointer: alloca, store, call via pointer."""
    stack_p = builder.alloca(arg_ll_ty)
    builder.store(arg, stack_p)
    func_ty_ll = llir.FunctionType(ret_type, [arg_ll_ty.as_pointer()])
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
