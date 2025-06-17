import llvmlite.binding as ll

from numba.core.cgutils import get_or_insert_function
from numba.core.types import FunctionType, NoneType, Tuple, UniTuple
from numba.extending import intrinsic

from numbox.core.bindings.signatures import signatures
from numbox.utils.lowlevel import get_ll_func_sig


def _call_lib_func_res(context, builder, func_ty, func_name, func_args):
    func_ty_ll = get_ll_func_sig(context, func_ty)
    func_p = get_or_insert_function(builder.module, func_ty_ll, func_name)
    res = builder.call(func_p, func_args)
    return res


@intrinsic(prefer_literal=True)
def _call_lib_func(typingctx, func_name_ty, args_ty=NoneType):
    func_name = func_name_ty.literal_value
    func_p_as_int = ll.address_of_symbol(func_name)
    if func_p_as_int is None:
        raise RuntimeError(f"{func_name} is unavailable in the LLVM context")
    func_sig = signatures.get(func_name, None)
    if func_sig is None:
        raise ValueError(f"Undefined signature for {func_name}")
    func_ty = FunctionType(func_sig)

    if args_ty == NoneType:
        def codegen(context, builder, signature, arguments):
            return _call_lib_func_res(context, builder, func_ty, func_name, ())
    else:
        def codegen(context, builder, signature, arguments):
            _, args = arguments
            func_args = []
            for arg_ind, arg_ty in enumerate(args_ty):
                arg = builder.extract_value(args, arg_ind)
                func_args.append(arg)
            return _call_lib_func_res(context, builder, func_ty, func_name, func_args)
        assert isinstance(args_ty, (Tuple, UniTuple))
    sig = func_ty.signature.return_type(func_name_ty, args_ty)
    return sig, codegen
