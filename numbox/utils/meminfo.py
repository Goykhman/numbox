"""Helpers that pass a structref through an integer address and read the reference count of its MemInfo.

A MemInfo is numba's reference-counted header for the memory of a structref or an array. :func:`export_meminfo`
returns the address of a structref's MemInfo as an ``intp`` and takes a reference, :func:`borrow_structref` builds a
structref from such an address, and :func:`release_meminfo` releases the reference. :func:`structref_meminfo` returns
the addresses of a structref's MemInfo and data, and :func:`get_nrt_refcount` reads the reference count.
"""
import ctypes

import numba

from llvmlite import ir as llir
from numba import types
from numba.core import cgutils
from numba.core.errors import TypingError
from numba.core.extending import intrinsic


return_tuple_type = types.UniTuple(types.intp, 2)


@intrinsic
def _structref_meminfo(typingctx, s_type):

    def codegen(context, builder, signature, args):
        """Emit the conversion of the MemInfo pointer and its data pointer to ``intp``, and return both as a tuple."""
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
    """Return the addresses of the MemInfo of the structref ``s`` and of its data, as a tuple of two ``intp``.

    The addresses take no reference, so they are valid only while ``s`` is alive. numba creates a MemInfo for the call
    alone when Python passes a NumPy array, so for an array the first address is no longer valid when the call returns.
    """
    return _structref_meminfo(s)


def get_nrt_refcount(s):
    """Return the reference count of the MemInfo of the structref ``s``, which numba's ``incref`` and ``decref`` change.

    The function reads the first field, ``refct``, of numba's ``MemInfo`` structure from the address that
    :func:`structref_meminfo` returns, so Python code can call it but jitted code cannot. The count is not meaningful
    for a NumPy array that Python passes in, because numba creates a MemInfo for the call alone. The layout of the
    structure is::

        struct MemInfo {
            size_t            refct;
            NRT_dtor_function dtor;
            void              *dtor_info;
            void              *data;
            size_t            size;    /* only used for NRT allocated memory */
            NRT_ExternalAllocator *external_allocator;
        };

    The structure is defined in numba's
    `nrt.cpp <https://github.com/numba/numba/blob/main/numba/core/runtime/nrt.cpp#L17>`_.
    """
    meminfo_p, data_p = structref_meminfo(s)
    return ctypes.c_int64.from_address(meminfo_p).value


_MI_TY = types.MemInfoPointer(types.voidptr)


def _require_intp(p_ty, fn_name):
    """Raise ``TypingError``, naming ``fn_name``, unless the type ``p_ty`` of an address argument is ``types.intp``.

    An address held in a narrower integer type, such as ``int32``, has already lost its upper bits on a 64-bit host, so
    converting it with ``inttoptr`` would produce a wrong pointer.
    """
    if p_ty != types.intp:
        raise TypingError(f"{fn_name} expects types.intp, got {p_ty}")


@intrinsic
def _incref_meminfo(typingctx, p_ty):
    """Increment the reference count of the MemInfo at the ``intp`` address ``p``, in jitted code.

    The ``removerefctpass`` of numba 0.65 and earlier, a pass that deletes the reference count changes of a function
    that seems not to need them, does not remove this increment.
    """
    _require_intp(p_ty, "_incref_meminfo")
    sig = types.void(p_ty)

    def codegen(context, builder, signature, args):
        """Emit an ``NRT_incref`` of the MemInfo, with the marker that prevents ``removerefctpass`` from deleting it."""
        mi_ll_ty = context.get_value_type(_MI_TY)
        meminfo = builder.inttoptr(args[0], mi_ll_ty)
        context.nrt.incref(builder, _MI_TY, meminfo)
        # numba 0.65 and earlier run removerefctpass on a function that seems not to need reference counting: its
        # result, its arguments other than arrays and the results of its calls hold no type that numba counts
        # references for. export_meminfo called with an array is such a function. The pass deletes every NRT_incref
        # and NRT_decref call in that function by name, including this increment. _release_meminfo makes _legalize()
        # return False by calling an NRT function outside the pass's list of accepted functions, but numba registers
        # no such function for an increment. The code therefore adds the enclosing function to the named metadata
        # numba_args_may_always_need_nrt, numba's own marker for a function whose references outlive the call. For
        # that function's module, _legalize() then returns False, and the pass does not run. The marker can only stop
        # the pass, so it changes nothing where another check already stops it. numba lowers each function into a
        # module of its own, so the check for a caller of the enclosing function never finds the marker. numba 0.66
        # removed the pass together with every use of the marker, so nothing in numba 0.66 or later reads it.
        module = builder.module
        nmd = module.add_named_metadata("numba_args_may_always_need_nrt")
        nmd.add(module.add_metadata([builder.function]))
    return sig, codegen


@intrinsic
def _release_meminfo(typingctx, p_ty):
    """Decrement the reference count of the MemInfo at the ``intp`` address ``p``, in jitted code.

    The decrement calls ``NRT_MemInfo_release``, which frees the memory when the count reaches zero.
    """
    _require_intp(p_ty, "_release_meminfo")
    sig = types.void(p_ty)

    def codegen(context, builder, signature, args):
        """Emit a call of ``NRT_MemInfo_release`` on the MemInfo at the address."""
        # The code calls NRT_MemInfo_release rather than context.nrt.decref. A signature such as void(intp) holds no
        # type that numba counts references for, so the removerefctpass of numba 0.65 and earlier would delete a plain
        # NRT_decref. NRT_MemInfo_release is not among the functions that the pass accepts, NRT_incref and NRT_decref,
        # so _legalize() returns False and the pass does not run on the function.
        ptr_ty = llir.IntType(8).as_pointer()
        fnty = llir.FunctionType(llir.VoidType(), [ptr_ty])
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, "NRT_MemInfo_release")
        meminfo = builder.inttoptr(args[0], ptr_ty)
        builder.call(fn, [meminfo])
    return sig, codegen


@intrinsic
def _deref_structref_raw_ptr(typingctx, struct_type_ref, p_ty):
    """Return a structref of the type ``struct_type_ref`` over the MemInfo at the address ``p``, in jitted code.

    The reference count does not change, so the caller must first take the reference that the returned structref holds,
    as :func:`borrow_structref` does.
    """
    _require_intp(p_ty, "_deref_structref_raw_ptr")
    inst_type = struct_type_ref.instance_type
    sig = inst_type(struct_type_ref, p_ty)

    def codegen(context, builder, signature, args):
        """Emit a structref whose ``meminfo`` field is the address, converted to a pointer."""
        p_val = args[1]
        mi_ll_ty = context.get_value_type(_MI_TY)
        meminfo = builder.inttoptr(p_val, mi_ll_ty)
        st = cgutils.create_struct_proxy(inst_type)(context, builder)
        st.meminfo = meminfo
        return st._getvalue()
    return sig, codegen


@numba.njit
def borrow_structref(struct_type, p):
    """Return a structref of the type ``struct_type`` over the MemInfo at the address ``p``.

    Python code and jitted code can call it. The structref holds a reference of its own, which numba releases when the
    structref is no longer used, so a borrow leaves the reference count unchanged. ``p`` must come from
    :func:`export_meminfo` and must not have been released yet. ``struct_type`` must be the type of the exported
    structref, which is not checked.
    """
    _incref_meminfo(p)
    return _deref_structref_raw_ptr(struct_type, p)


@numba.njit
def export_meminfo(s):
    """Return the address of the MemInfo of ``s`` as an ``intp``, and take a reference that keeps ``s`` alive.

    Python code and jitted code can call it. ``s`` is a structref or a NumPy array. Pass each address to
    :func:`release_meminfo` exactly once, and do not use it afterward, because the memory may then be freed.
    """
    meminfo_p, _ = structref_meminfo(s)
    _incref_meminfo(meminfo_p)
    return meminfo_p


@numba.njit
def release_meminfo(p):
    """Release the reference that :func:`export_meminfo` took for the address ``p``.

    Python code and jitted code can call it. The memory is freed if no other reference remains, so ``p`` must not be
    used afterward.
    """
    _release_meminfo(p)
