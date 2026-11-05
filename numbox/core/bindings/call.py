"""The internal helper through which the libc, libm and SQLite bindings call C functions.

``_call_lib_func`` is a numba intrinsic that compiles a call of a C function by its name. It takes the result type from
:mod:`numbox.core.bindings.signatures`, checks that LLVM can find the name, and passes each value by the host's calling
convention, which :mod:`numbox.core.bindings.abi` classifies. numbox's bindings call it from functions decorated with
:func:`~numbox.core.proxy.proxy.proxy`, so that numba has converted the arguments to the table's types first.
``_call_lib_func_byval`` passes a struct through a pointer to a copy, for a C function that takes a pointer to a
struct. The section "Passing a struct" of the bindings page says which structs reach C intact.
"""
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
    """Call the C function named by a string literal, with its arguments in a tuple, and return the C result.

    ``func_name_ty`` is the C function's name, which must be a string literal. ``args_ty`` is a tuple of the arguments,
    which is empty or absent for a function that takes none; a single argument may also be passed without a tuple. The
    ``signatures`` table supplies the result type, and each argument keeps the numba type that the caller passes. Call
    the helper from a function compiled for the table's signature, as numbox's proxied bindings are, so that numba has
    converted the arguments to the C types.

    Numbers and pointers are passed unchanged. A struct, written as a numba tuple, is passed by the host's C convention,
    within the limits that the comment above ``numbox.core.bindings.abi._classify`` lists. Declare no ``Record`` result:
    one larger than 16 bytes raises ``TypingError``, and a smaller one crashes the process when the call runs. Use
    ``_call_lib_func_byval`` for a C function that takes a pointer to a struct.

    numba raises ``TypingError`` when it compiles the call if LLVM cannot find the name, if the table has no entry for
    it, or if a tuple field has no fixed bit width.
    """
    func_name = extract_literal_str("_call_lib_func", func_name_ty, field="func_name")
    # The address is used only to check that LLVM can resolve the name. The generated code calls the function by name
    # and never uses this address, so the JIT linker resolves the name again for a cached caller in a later process,
    # where the address differs.
    func_p_as_int = ll.address_of_symbol(func_name)
    if func_p_as_int is None:
        raise TypingError(f"{func_name} is unavailable in the LLVM context")
    func_sig = signatures.get(func_name, None)
    if func_sig is None:
        raise TypingError(f"Undefined signature for {func_name}")

    # Only the result type comes from the signatures table. Each argument is lowered with the numba type that the
    # caller passes (arg_types below), without conversion. Call this helper only from a function compiled for the
    # table's signature, as numbox's proxied bindings are, so that numba converts the arguments first. A direct call
    # with other types passes them unconverted: given the tuple (11, 22, 33), whose fields numba types as int64, a C
    # function whose entry takes the struct (int32, int32, int64) receives wrong values.
    ret_ty = func_sig.return_type
    ret_class = _classify(ret_ty)
    # numba represents a Record value as a pointer to its data, so a Record result in an sret slot on the stack would
    # give the caller a pointer into the stack frame of a function that has returned. A Record result of 16 bytes or
    # less passes this check: the call is then declared to return a pointer where C returns the struct's bytes, and the
    # process crashes.
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
    # A result larger than 16 bytes is returned through sret, and so is a smaller one on Windows x64 if its size is not
    # 1, 2, 4 or 8. With sret, the caller allocates the result and passes its address as a hidden first argument, and
    # the call is declared to return void.
    use_sret = (
        ret_class == _CLASS_STRUCT_LARGE
        or (
            ret_class == _CLASS_STRUCT_SMALL
            and plat == _PLATFORM_WIN_X64
            and not _is_windows_register_passable(
                _struct_bytes(ret_ty, "_call_lib_func"))
        )
    )
    # A 16-byte result with two INTEGER halves needs the same repack as an argument: the call is declared to return
    # {i64, i64}, and the codegen copies the value back into the result's own LLVM type.
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
            # The comment above abi._classify lists which struct shapes reach C intact on each path below.
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
            # Any other struct is passed as a pointer to a copy on the stack of the function that makes the call. On
            # SysV x86-64, the code below also marks the pointer to a struct over 16 bytes with the byval attribute.
            # LLVM then copies the struct onto the stack, as the C convention requires, and C receives the copy instead
            # of a pointer.
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
        # optnone and noinline keep LLVM's optimizer from removing the stack copy before C reads it
        # (https://github.com/numba/llvmlite/issues/300#issuecomment-327235846). The cost is that the whole function
        # that makes the call is compiled without optimization and is never inlined.
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
    """Emit a call of ``func_name`` that passes ``arg`` as a pointer to a copy on the stack, and return its result.

    The pointer has no ``byval`` attribute, so C receives the address of the copy.
    """
    stack_p = builder.alloca(arg_ll_ty)
    builder.store(arg, stack_p)
    func_ty_ll = llir.FunctionType(ret_type, [arg_ll_ty.as_pointer()])
    func_p = get_or_insert_function(builder.module, func_ty_ll, func_name)
    return builder.call(func_p, [stack_p])


def _needs_int_int_eightbyte_repack(ty, plat):
    """Return True if a 16-byte struct, passed or returned by value, must be repacked to ``{i64, i64}`` on ``plat``.

    The repack applies on SysV x86-64 and AAPCS64 to a 16-byte struct whose two halves are both INTEGER class and whose
    layout is not already two 64-bit integers, such as ``{i32, i32, i64}``, the layout of DuckDB's
    ``duckdb_interval``. Windows x64 never needs it, because it passes a 16-byte struct through a pointer.
    """
    if plat not in (_PLATFORM_SYSV_X86_64, _PLATFORM_AAPCS64):
        return False
    if _struct_bytes(ty, "_call_lib_func") != 16:
        return False
    if _is_canonical_int64_pair_layout(ty):
        return False
    # LLVM passes each field of an aggregate in a register of its own, while C packs the fields of each 8-byte half
    # into one register. If the struct is passed as {i32, i32, i64}, C therefore does not receive the two i32 fields as
    # one half. If it is passed as {i64, i64}, each half is passed in the register where C reads it.
    return _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def _repack_to_i64_pair(builder, val, orig_ll_ty):
    """Return ``(value, type)``: ``val``, a 16-byte struct, reinterpreted as ``{i64, i64}``, and that LLVM type.

    The bytes remain unchanged, so both halves must be INTEGER class, as ``_needs_int_int_eightbyte_repack`` checks
    before the call. The caller substitutes the returned value and type for the original ones in the call. See
    https://github.com/numba/llvmlite/issues/300#issuecomment-327235846 for the reason that LLVM needs the repack.
    """
    i64x2_ll_ty = llir.LiteralStructType(
        [llir.IntType(64), llir.IntType(64)])
    # The slot is an {i64, i64}, aligned to 8 bytes, and the store writes val through a pointer of val's own type. The
    # {i64, i64} load that reads the bytes back is then aligned. A slot of val's type, such as a 4-byte-aligned
    # [4 x i32], would make that load misaligned, which is undefined behavior in LLVM. The bitcast only reinterprets
    # the bytes, so no extension or shift is needed.
    slot = builder.alloca(i64x2_ll_ty)
    slot_as_orig = builder.bitcast(slot, orig_ll_ty.as_pointer())
    builder.store(val, slot_as_orig)
    return builder.load(slot), i64x2_ll_ty


def _repack_from_i64_pair(builder, i64_pair_val, target_ll_ty):
    """Return ``i64_pair_val``, a ``{i64, i64}`` call result, reinterpreted as ``target_ll_ty``.

    This is the inverse of ``_repack_to_i64_pair``, for a 16-byte result with two INTEGER halves. LLVM would read each
    field of such a result from a register of its own, while C returns each 8-byte half in one register, so
    ``_call_lib_func`` declares the call to return ``{i64, i64}`` and converts the value here.
    """
    i64x2_ll_ty = llir.LiteralStructType(
        [llir.IntType(64), llir.IntType(64)])
    # The aligned {i64, i64} slot holds the call's result, and the load reads the same bytes as the target type.
    slot = builder.alloca(i64x2_ll_ty)
    builder.store(i64_pair_val, slot)
    slot_as_target = builder.bitcast(slot, target_ll_ty.as_pointer())
    return builder.load(slot_as_target)


@intrinsic(prefer_literal=True)
def _call_lib_func_byval(typingctx, func_name_ty, arg_ty):
    """Call the C function named by a string literal with a pointer to a copy of ``arg``, and return the C result.

    Use it for a C function that takes a pointer to a struct, ``T *``, when the caller holds the struct as a value.
    numba's types cannot distinguish ``T`` from ``T *``, so the caller chooses this helper or ``_call_lib_func`` by the
    C declaration. The copy is on the stack of the function that makes the call, so C must not keep the pointer after
    that function returns. The result type comes from the ``signatures`` table, and numba raises ``TypingError`` if the
    table has no entry for the name. Unlike ``_call_lib_func``, this helper does not check that LLVM can find the name.
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
