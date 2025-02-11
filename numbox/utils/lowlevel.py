from numba import njit
from numba.core.base import BaseContext
from numba.extending import intrinsic


@intrinsic
def _cast(typingctx, source_class, dest_ty_class):
    dest_ty = dest_ty_class.instance_type
    sig = dest_ty(source_class, dest_ty_class)

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
def _deref(typingctx, p_class, ty_class):
    ty = ty_class.instance_type
    sig = ty(p_class, ty_class)

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
