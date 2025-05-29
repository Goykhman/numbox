from llvmlite import ir
from llvmlite.ir.builder import IRBuilder
from numba import njit
from numba.core.base import BaseContext
from numba.core.cgutils import int32_t
from numba.core.types import FunctionType, StructRef, TypeRef
from numba.extending import intrinsic

from numbox.utils.highlevel import determine_field_index


@intrinsic
def _cast(typingctx, source_class, dest_ty_ref: TypeRef):
    dest_ty = dest_ty_ref.instance_type
    sig = dest_ty(source_class, dest_ty_ref)

    def codegen(context: BaseContext, builder, signature, args):
        source_ty_ll = context.get_value_type(source_class)
        dest_ty_ll = context.get_value_type(dest_ty)
        val = context.cast(builder, args[0], source_ty_ll, dest_ty_ll)
        context.nrt.incref(builder, dest_ty, val)
        return val
    return sig, codegen


@njit
def cast(source, dest_ty):
    """ Cast `source` to the type `dest_ty` """
    return _cast(source, dest_ty)


@intrinsic
def _deref(typingctx, p_class, ty_ref: TypeRef):
    ty = ty_ref.instance_type
    sig = ty(p_class, ty_ref)

    def codegen(context: BaseContext, builder, signature, args):
        ty_ll = context.get_value_type(ty)
        p = args[0]
        _, meminfo_p = context.nrt.get_meminfos(builder, p_class, p)[0]
        payload_p = context.nrt.meminfo_data(builder, meminfo_p)
        x_as_ty_p = builder.bitcast(payload_p, ty_ll.as_pointer())
        val = builder.load(x_as_ty_p)
        context.nrt.incref(builder, ty, val)
        return val
    return sig, codegen


@njit
def deref(p, ty):
    """ Dereference payload of structref `p` as type `ty` """
    return _deref(p, ty)


def extract_data_member(
    context: BaseContext,
    builder: IRBuilder,
    struct_fe_ty: StructRef,
    struct_obj,
    member_name: str
):
    """ For the given struct object of the given numba (front-end) type extract
     data member with the given name (must be literal, available at compile time) """
    payload_ty = struct_fe_ty.get_data_type()
    meminfo = context.nrt.get_meminfos(builder, struct_fe_ty, struct_obj)[0]
    _, meminfo_p = meminfo
    payload_p = context.nrt.meminfo_data(builder, meminfo_p)
    payload_ty_ll = context.get_data_type(payload_ty)
    payload_ty_p_ll = payload_ty_ll.as_pointer()
    payload_p = builder.bitcast(payload_p, payload_ty_p_ll)
    member_ind = determine_field_index(struct_fe_ty, member_name)
    data_p = builder.gep(payload_p, (int32_t(0), int32_t(member_ind)))
    data = builder.load(data_p)
    return data


def get_func_p_from_func_struct(builder: IRBuilder, func_struct):
    """ Extract void* function pointer from the low-level FunctionType structure """
    func_raw_p_ind = 0
    return builder.extract_value(func_struct, func_raw_p_ind)


def get_ll_func_sig(context: BaseContext, func_ty: FunctionType):
    func_sig = func_ty.signature
    return ir.FunctionType(
        context.get_data_type(func_sig.return_type),
        [context.get_data_type(arg) for arg in func_sig.args]
    )
