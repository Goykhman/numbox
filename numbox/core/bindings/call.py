import llvmlite.binding as ll
from llvmlite import ir as llir

from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import BaseTuple, NoneType, Record
from numba.extending import intrinsic

from numbox.core.bindings.abi import (
    _CLASS_SCALAR, _CLASS_STRUCT_SMALL, _CLASS_STRUCT_LARGE,
    _EIGHTBYTE_CLASS_INTEGER,
    _PLATFORM_AAPCS64, _PLATFORM_SYSV_X86_64, _PLATFORM_WIN_X64,
    _classify, _classify_eightbytes, _current_platform,
    _is_canonical_int64_pair_layout, _is_windows_register_passable,
    _struct_bytes,
)
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import extract_literal_str


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
      pointer). On SysV x86-64 and AAPCS64, 16-byte aggregates whose
      eightbytes are pure-INTEGER but whose LLVM type isn't already
      ``{i64, i64}`` (e.g. ``{i32, i32, i64}`` -- duckdb_interval) are
      repacked via memory bitcast before the call -- working around
      llvmlite not modelling the eightbyte-packing rule, which
      otherwise drops fields.
    - **>16-byte struct args** -- by pointer on every platform; on SysV
      x86-64 the ``byval`` attribute is added to the LLVM arg and the
      enclosing function gets ``optnone`` + ``noinline`` so the LLVM
      optimizer does not elide the caller-side stack copy before the
      callee reads it. See:
      https://github.com/numba/llvmlite/issues/300#issuecomment-327235846
    - **<=16-byte struct returns** -- direct on SysV x86-64 and AAPCS64;
      on Windows x64, sizes 1/2/4/8 return in RAX (direct) and other
      sizes use ``sret`` (caller-allocated hidden first arg, void
      return). 16-byte non-canonical INT/INT returns get the same
      eightbyte repack on SysV / AAPCS64: the LLVM call is declared
      to return ``{i64, i64}`` and the result is unpacked back to
      the original LLVM type via memory bitcast.
    - **>16-byte struct returns** -- ``sret`` (caller-allocated hidden
      first arg, void return) on every platform. SysV x86-64 / AAPCS64
      / Windows x64 all use indirect-result-location for this size
      class; the codegen path is shared with the Windows ``<=16-byte``
      non-register-passable case. ``Record`` returns are explicitly
      rejected (``TypingError``): numba's ``RecordModel`` represents
      values as raw pointers, so a stack-alloca sret slot would dangle
      after the ``@njit`` function returns; safe support needs an NRT-
      allocated buffer + integration with numba's record-ownership
      model. Add when a consumer needs it.

    For C signatures of form ``func(T*)`` (pointer to struct) rather
    than ``func(T)`` lowered to a byval pointer by the ABI, use the
    sibling ``_call_lib_func_byval`` intrinsic in this module instead.
    The numba type system can't disambiguate ``T`` from ``T*``; the
    caller picks the intrinsic based on what the C header declares.
    """
    func_name = extract_literal_str("_call_lib_func", func_name_ty, field="func_name")
    func_p_as_int = ll.address_of_symbol(func_name)
    if func_p_as_int is None:
        raise TypingError(f"{func_name} is unavailable in the LLVM context")
    func_sig = signatures.get(func_name, None)
    if func_sig is None:
        raise TypingError(f"Undefined signature for {func_name}")

    ret_ty = func_sig.return_type
    ret_class = _classify(ret_ty)
    if ret_class == _CLASS_STRUCT_LARGE and isinstance(ret_ty, Record):
        raise TypingError(
            f"_call_lib_func: Record returns >16 bytes are not yet "
            f"supported ({func_name}). Numba's RecordModel represents "
            f"values as raw pointers, so the natural stack-alloca sret "
            f"slot would dangle after the @njit function returns. Safe "
            f"support needs an NRT-allocated buffer hooked into numba's "
            f"record-ownership model. Use a Tuple/UniTuple return shape "
            f"instead, or open an issue if you need Record."
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
        ret_class == _CLASS_STRUCT_LARGE
        or (
            ret_class == _CLASS_STRUCT_SMALL
            and plat == _PLATFORM_WIN_X64
            and not _is_windows_register_passable(
                _struct_bytes(ret_ty, "_call_lib_func"))
        )
    )
    needs_ret_repack = (
        ret_class == _CLASS_STRUCT_SMALL
        and not use_sret
        and _needs_int_int_eightbyte_repack(ret_ty, plat)
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
                if _needs_int_int_eightbyte_repack(arg_ty, plat):
                    val, arg_ll_ty = _repack_to_i64_pair(
                        builder, val, arg_ll_ty)
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
        elif needs_ret_repack:
            i64x2_ll_ty = llir.LiteralStructType(
                [llir.IntType(64), llir.IntType(64)])
            func_ll_ty = llir.FunctionType(i64x2_ll_ty, ll_arg_tys)
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
        result = builder.call(func_p, ll_arg_vals)
        if needs_ret_repack:
            return _repack_from_i64_pair(builder, result, ret_ll_ty)
        return result

    sig = ret_ty(func_name_ty, args_ty)
    return sig, codegen


def _emit_byval_call(builder, arg, arg_ll_ty, ret_type, func_name):
    """Emit IR to pass a struct by pointer: alloca, store, call via pointer."""
    stack_p = builder.alloca(arg_ll_ty)
    builder.store(arg, stack_p)
    func_ty_ll = llir.FunctionType(ret_type, [arg_ll_ty.as_pointer()])
    func_p = get_or_insert_function(builder.module, func_ty_ll, func_name)
    return builder.call(func_p, [stack_p])


def _needs_int_int_eightbyte_repack(ty, plat):
    """Whether a 16-byte by-value struct (arg or return) needs repack
    to ``{i64, i64}`` before the call on the host's ABI.

    Both SysV x86-64 and AAPCS64 are affected: llvmlite's small-struct
    register lowering drops fields when both eightbytes are pure-
    INTEGER but the LLVM type isn't already the canonical
    ``{i64, i64}`` shape (e.g. ``{i32, i32, i64}`` — the duckdb_interval
    layout — loses fields). Windows x64 is unaffected because 16-byte
    aggregates fall outside the ``{1, 2, 4, 8}`` register-passable
    set, so the call goes via alloca + pointer-pass and avoids the
    register-coercion path entirely.
    """
    if plat not in (_PLATFORM_SYSV_X86_64, _PLATFORM_AAPCS64):
        return False
    if _struct_bytes(ty, "_call_lib_func") != 16:
        return False
    if _is_canonical_int64_pair_layout(ty):
        return False
    return _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def _repack_to_i64_pair(builder, val, orig_ll_ty):
    """Repack a 16-byte struct value to ``{i64, i64}`` via memory bitcast.

    Allocates a well-aligned ``{i64, i64}`` slot (8-byte alignment),
    bitcasts it to ``orig_ll_ty*`` for the store, then loads the same
    bytes back as ``{i64, i64}``. Storing into the bitcast pointer is
    safe because the slot is over-aligned relative to ``orig_ll_ty``;
    the subsequent ``{i64, i64}`` load is well-aligned, avoiding the
    UB that an ``alloca(orig_ll_ty)`` slot (e.g. 4-byte aligned for
    ``[4 x i32]``) would expose. Both eightbytes must already be
    pure-INTEGER (caller checks via ``_needs_int_int_eightbyte_repack``);
    the bitcast is a pure byte reinterpretation, no zext/shift
    gymnastics required. Returns ``(repacked_val, repacked_ll_ty)``
    for the caller to substitute into the call site.
    See: https://github.com/numba/llvmlite/issues/300#issuecomment-327235846
    """
    i64x2_ll_ty = llir.LiteralStructType(
        [llir.IntType(64), llir.IntType(64)])
    slot = builder.alloca(i64x2_ll_ty)
    slot_as_orig = builder.bitcast(slot, orig_ll_ty.as_pointer())
    builder.store(val, slot_as_orig)
    return builder.load(slot), i64x2_ll_ty


def _repack_from_i64_pair(builder, i64_pair_val, target_ll_ty):
    """Inverse of ``_repack_to_i64_pair``: convert a ``{i64, i64}``
    return value back into ``target_ll_ty``.

    Allocates a well-aligned ``{i64, i64}`` slot, stores the call
    result, bitcasts the slot to ``target_ll_ty*``, and loads. Used
    for the return-side eightbyte repack: when the C function returns
    a 16-byte INT/INT struct that isn't canonical ``{i64, i64}``,
    llvmlite has the same field-dropping issue on the return as on
    the arg side -- so we declare the LLVM call returning
    ``{i64, i64}`` and unpack it here.
    """
    i64x2_ll_ty = llir.LiteralStructType(
        [llir.IntType(64), llir.IntType(64)])
    slot = builder.alloca(i64x2_ll_ty)
    builder.store(i64_pair_val, slot)
    slot_as_target = builder.bitcast(slot, target_ll_ty.as_pointer())
    return builder.load(slot_as_target)


@intrinsic(prefer_literal=True)
def _call_lib_func_byval(typingctx, func_name_ty, arg_ty):
    """Pass ``arg`` to a C function by pointer on all platforms.

    Used when the C signature takes a pointer to a struct and the caller
    holds the struct as a value; the intrinsic allocates a stack slot,
    stores the value, and passes the slot's address.
    """
    func_name = extract_literal_str("_call_lib_func_byval", func_name_ty, field="func_name")
    func_sig = signatures.get(func_name, None)
    if func_sig is None:
        raise TypingError(f"Undefined signature for {func_name}")

    def codegen(context, builder, signature, arguments):
        _, arg = arguments
        arg_ll_ty = context.get_value_type(arg_ty)
        ret_type = context.get_value_type(signature.return_type)
        return _emit_byval_call(
            builder, arg, arg_ll_ty, ret_type, func_name)

    sig = func_sig.return_type(func_name_ty, arg_ty)
    return sig, codegen
