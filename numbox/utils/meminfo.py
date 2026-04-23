import ctypes

import numba

from llvmlite import ir as llir
from numba import types
from numba.core import cgutils
from numba.core.extending import intrinsic


return_tuple_type = types.UniTuple(types.intp, 2)


@intrinsic
def _structref_meminfo(typingctx, s_type):

    def codegen(context, builder, signature, args):
        s = args[0]
        meminfo = context.nrt.get_meminfos(builder, s_type, s)[0]
        type_data, meminfo_p = meminfo

        meminfo_p_as_int = builder.ptrtoint(meminfo_p, context.get_data_type(numba.intp))
        data_p = context.nrt.meminfo_data(builder, meminfo_p)
        data_p_as_int = builder.ptrtoint(data_p, context.get_data_type(numba.intp))

        return context.make_tuple(builder, return_tuple_type, (meminfo_p_as_int, data_p_as_int))

    sig = return_tuple_type(s_type)
    return sig, codegen


@numba.njit
def structref_meminfo(s):
    """
    :param s: instance of StructRef
    :return: tuple of pointers (as 64-bit integers) to meminfo member of the StructRef and its data payload.
    """
    return _structref_meminfo(s)


def get_nrt_refcount(s):
    """
    :param s: instance of an object with payload wrapped in a meminfo
    :return: NRT reference count, the one which is controllable by nrt context's incref/decref

    Leverages meminfo memory layout::

        struct MemInfo {
            size_t            refct;
            NRT_dtor_function dtor;
            void              *dtor_info;
            void              *data;
            size_t            size;    /* only used for NRT allocated memory */
            NRT_ExternalAllocator *external_allocator;
        };

    see `nrt <https://github.com/numba/numba/blob/main/numba/core/runtime/nrt.cpp#L17>`_.
    """
    meminfo_p, data_p = structref_meminfo(s)
    return ctypes.c_int64.from_address(meminfo_p).value


_MI_TY = types.MemInfoPointer(types.voidptr)


@intrinsic
def _incref_meminfo(typingctx, p_ty):
    """Incref a MemInfo at ``intp`` via NRT."""
    sig = types.void(p_ty)

    def codegen(context, builder, signature, args):
        mi_ll_ty = context.get_value_type(_MI_TY)
        meminfo = builder.inttoptr(args[0], mi_ll_ty)
        context.nrt.incref(builder, _MI_TY, meminfo)
    return sig, codegen


@intrinsic
def _release_meminfo(typingctx, p_ty):
    """Decref a MemInfo at ``intp`` via ``NRT_MemInfo_release``.

    Can't use ``context.nrt.decref()`` here -- ``removerefctpass`` strips
    ``NRT_decref`` when the function signature has no NRT-tracked types.
    ``NRT_MemInfo_release`` does the same atomic decref + dtor call AND
    causes ``_legalize()`` to bail out of the rewrite, protecting the
    whole function.
    """
    sig = types.void(p_ty)

    def codegen(context, builder, signature, args):
        ptr_ty = llir.IntType(8).as_pointer()
        fnty = llir.FunctionType(llir.VoidType(), [ptr_ty])
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, "NRT_MemInfo_release")
        meminfo = builder.inttoptr(args[0], ptr_ty)
        builder.call(fn, [meminfo])
    return sig, codegen


@intrinsic
def _deref_structref_raw_ptr(typingctx, struct_type_ref, p_ty):
    """Reconstruct a structref value from an ``intp`` MemInfo pointer."""
    inst_type = struct_type_ref.instance_type
    sig = inst_type(struct_type_ref, p_ty)

    def codegen(context, builder, signature, args):
        p_val = args[1]
        mi_ll_ty = context.get_value_type(_MI_TY)
        meminfo = builder.inttoptr(p_val, mi_ll_ty)
        st = cgutils.create_struct_proxy(inst_type)(context, builder)
        st.meminfo = meminfo
        return st._getvalue()
    return sig, codegen


@numba.njit
def borrow_structref(struct_type, p):
    """Incref + reconstruct a structref from an ``intp`` MemInfo pointer.

    The caller receives a live structref that participates in normal NRT
    refcount. Net-zero for the external owner if the caller's scope exits
    without additional actions (local decref on drop balances the incref).
    """
    _incref_meminfo(p)
    return _deref_structref_raw_ptr(struct_type, p)


@numba.njit
def export_meminfo(s):
    """Export a structref as an ``intp`` MemInfo pointer with a +1 incref.

    The returned ``intp`` keeps the allocation alive until the caller
    balances it with ``_release_meminfo``.
    """
    meminfo_p, _ = structref_meminfo(s)
    _incref_meminfo(meminfo_p)
    return meminfo_p
