import ctypes

import numba

from numba import types
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


def get_nrt_refcount(meminfo_p):
    """
    :param meminfo_p: meminfo pointer
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
    meminfo_p, data_p = structref_meminfo(meminfo_p)
    return ctypes.c_int64.from_address(meminfo_p).value
