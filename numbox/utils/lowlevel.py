from llvmlite import ir
from llvmlite.ir.builder import IRBuilder
from numba import carray, njit
from numba.core.base import BaseContext
from numba.core.cgutils import int32_t, intp_t, is_not_null as cgutils_is_not_null, pack_array, voidptr_t
from numba.experimental.jitclass.base import imp_dtor
from numba.experimental.structref import _Utils
from numba.core.types import (
    boolean, FunctionType, intp, StructRef, TypeRef, Tuple, char,
    UnicodeType, unicode_type, uintp, UniTuple, voidptr
)
from numba.core.typing.context import Context
from numba.extending import intrinsic

from numbox.core.configurations import default_jit_options, MAX_STR_LENGTH
from numbox.utils.void_type import VoidType
from numbox.utils.highlevel import determine_field_index


@intrinsic
def _cast(typingctx: Context, source_ty, dest_ty_ref: TypeRef):
    dest_ty = dest_ty_ref.instance_type
    sig = dest_ty(source_ty, dest_ty_ref)

    def codegen(context: BaseContext, builder, signature, args):
        source_ty_ll = context.get_data_type(source_ty)
        dest_ty_ll = context.get_data_type(dest_ty)
        val = context.cast(builder, args[0], source_ty_ll, dest_ty_ll)
        context.nrt.incref(builder, dest_ty, val)
        return val
    return sig, codegen


@njit(**default_jit_options)
def cast(source, dest_ty):
    """ Cast `source` to the type `dest_ty` """
    return _cast(source, dest_ty)


@intrinsic
def _cast_int_to_void_p(typingctx, p_ty):
    """ Cast 64-bit integer to void pointer type """
    def codegen(context, builder, signature, arguments):
        return builder.inttoptr(arguments[0], voidptr_t)
    return voidptr(p_ty), codegen


@intrinsic
def _deref_payload(typingctx: Context, struct_ty, ty_ref: TypeRef):
    ty = ty_ref.instance_type
    sig = ty(struct_ty, ty_ref)

    def codegen(context: BaseContext, builder, signature, args):
        ty_ll = context.get_data_type(ty)
        struct_ = args[0]
        _, meminfo_p = context.nrt.get_meminfos(builder, struct_ty, struct_)[0]
        payload_p = context.nrt.meminfo_data(builder, meminfo_p)
        x_as_ty_p = builder.bitcast(payload_p, ty_ll.as_pointer())
        val = builder.load(x_as_ty_p)
        context.nrt.incref(builder, ty, val)
        return val
    return sig, codegen


@njit(**default_jit_options)
def deref_payload(p, ty):
    """ Dereference payload of structref `p` as type `ty` """
    return _deref_payload(p, ty)


def extract_struct_member(
    context: BaseContext,
    builder: IRBuilder,
    struct_fe_ty: StructRef,
    struct_obj,
    member_name: str,
    incref: bool = False
):
    """ For the given struct object of the given numba (front-end) type extract
    member with the given name (must be literal, available at compile time) """
    member_ty = struct_fe_ty.field_dict[member_name]
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
    if incref:
        context.nrt.incref(builder, member_ty, data)
    return data


def get_func_p_from_func_struct(builder: IRBuilder, func_struct):
    """ Extract void* function pointer from the low-level FunctionType structure """
    func_raw_p_ind = 0
    return builder.extract_value(func_struct, func_raw_p_ind)


@intrinsic
def _get_func_p_as_int_from_func_struct(typingctx, func_ty):
    def codegen(context, builder, signature, args):
        return builder.ptrtoint(get_func_p_from_func_struct(builder, args[0]), intp_t)
    return intp(func_ty), codegen


@njit(**default_jit_options)
def get_func_p_as_int_from_func_struct(func_):
    return _get_func_p_as_int_from_func_struct(func_)


@intrinsic
def _get_func_tuple(typingctx, func_ty):
    return_tuple_type = UniTuple(intp, 3)
    sig = return_tuple_type(func_ty)

    def codegen(context, builder, signature, args):
        func_struct = args[0]
        lst = []
        for i in range(3):
            lst.append(builder.ptrtoint(builder.extract_value(func_struct, i), intp_t))
        return context.make_tuple(builder, return_tuple_type, lst)
    return sig, codegen


@njit(**default_jit_options)
def get_func_tuple(func):
    return _get_func_tuple(func)


def get_ll_func_sig(context: BaseContext, func_ty: FunctionType):
    func_sig = func_ty.signature
    arg_types = []
    for arg in func_sig.args:
        arg_types.append(context.get_data_type(arg))
    return ir.FunctionType(context.get_data_type(func_sig.return_type), arg_types)


@njit(unicode_type(intp), **default_jit_options)
def get_str_from_p_as_int(p):
    """ Given pointer to null-terminated array of characters as an integer `p`,
    return unicode string object copying the original string's data """
    void_p = _cast_int_to_void_p(p)
    s = ""
    mem_view = carray(void_p, shape=(MAX_STR_LENGTH,), dtype=char)
    for char_as_code_p in mem_view:
        if char_as_code_p == 0:
            return s
        s += chr(char_as_code_p)
    return s


@intrinsic
def _get_unicode_data_p(typingctx, str_ty):
    def codegen(context, builder, signature, arguments):
        str_ = arguments[0]
        meminfo = context.nrt.get_meminfos(builder, unicode_type, str_)[0]
        _, meminfo_p = meminfo
        payload_p = context.nrt.meminfo_data(builder, meminfo_p)
        return builder.ptrtoint(payload_p, context.get_data_type(intp))
    assert isinstance(str_ty, UnicodeType)
    return intp(str_ty), codegen


@njit(**default_jit_options)
def get_unicode_data_p(s):
    """ Given Python unicode string, return pointer to its data payload,
    array of null-terminated characters.
    See https://github.com/numba/numba/blob/release0.61/numba/cpython/unicode.py#L83
    """
    return _get_unicode_data_p(s)


@intrinsic
def is_not_null(typingctx, struct_ty):
    def codegen(context, builder, signature, arguments):
        struct_ = arguments[0]
        meminfo = context.nrt.get_meminfos(builder, struct_ty, struct_)[0]
        type_data, meminfo_p = meminfo
        return cgutils_is_not_null(builder, meminfo_p)
    return boolean(struct_ty), codegen


def _new(context, builder, struct_ty_):
    """
    This is essentially
     `numba new <https://github.com/numba/numba/blob/release0.61/numba/experimental/structref.py#L301>`_

    Returns structref (structure with one member being pointer to its meminfo) and the pointer to its payload.
    """
    model = context.data_model_manager[struct_ty_.get_data_type()]
    alloc_type = model.get_value_type()
    alloc_size = context.get_abi_sizeof(alloc_type)
    meminfo = context.nrt.meminfo_alloc_dtor(
        builder,
        context.get_constant(uintp, alloc_size),
        imp_dtor(context, builder.module, struct_ty_),
    )
    data_pointer = context.nrt.meminfo_data(builder, meminfo)
    data_pointer = builder.bitcast(data_pointer, alloc_type.as_pointer())
    # builder.store(get_null_value(alloc_type), data_pointer)
    work_ = context.make_helper(builder, struct_ty_)
    work_.meminfo = meminfo
    return work_._getvalue(), data_pointer


def populate_structref(
    context, builder, signature, structref_type_, structref_, args, ordered_args_names, decref_old=False
):
    """ Store `args` with the corresponding ordered names `ordered_args_names`
    in structref with type `structref_type_` and payload at `data_pointer`.

    Based on numba.experimental.structref::define_attributes::struct_setattr_impl

    Do not call `decref_old` when populating a newly-created structref, as there's nothing to decref there.
    """
    field_dict = structref_type_.field_dict

    utils = _Utils(context, builder, structref_type_)
    dataval = utils.get_data_struct(structref_)

    for arg_i, (arg_name, arg) in enumerate(zip(ordered_args_names, args)):
        arg_ty = signature.args[arg_i]
        field_type = field_dict[arg_name]
        casted = context.cast(builder, arg, arg_ty, field_type)
        old_value = getattr(dataval, arg_name)
        context.nrt.incref(builder, arg_ty, casted)
        if decref_old:
            context.nrt.decref(builder, arg_ty, old_value)
        setattr(dataval, arg_name, casted)


@intrinsic
def _tuple_of_struct_ptrs_as_int(typingctx: Context, tup_ty: Tuple):
    """ Prefer using `_uniformize_tuple_of_structs` instead, if possible at all,
     to outsource smart pointer memory management to StructRef erased/void type. """
    tup_size = tup_ty.count

    def codegen(context, builder, signature, arguments):
        array_values = []
        tup = arguments[0]
        for tup_ind in range(tup_size):
            tup_item = builder.extract_value(tup, tup_ind)
            tup_item_meminfo = context.nrt.get_meminfos(builder, tup_ty[tup_ind], tup_item)[0]
            _, tup_item_meminfo_p = tup_item_meminfo
            tup_item_p_as_int = builder.ptrtoint(tup_item_meminfo_p, context.get_data_type(intp))
            array_values.append(tup_item_p_as_int)
        return pack_array(builder, array_values)
    ret_type = UniTuple(intp, tup_size)
    sig = ret_type(tup_ty)
    return sig, codegen


@njit(**default_jit_options)
def tuple_of_struct_ptrs_as_int(tup):
    return _tuple_of_struct_ptrs_as_int(tup)


@intrinsic
def _uniformize_tuple_of_structs(typingctx: Context, tup_ty: Tuple, uniform_ty_ref: TypeRef):
    tup_size = tup_ty.count
    for tup_ind in range(tup_size):
        item_ty = tup_ty[tup_ind]
        assert isinstance(item_ty, StructRef), "Only Tuple of StructRef supported"
    uniform_ty = uniform_ty_ref.instance_type
    ret_type = UniTuple(uniform_ty, tup_size)

    def codegen(context, builder, signature, arguments):
        array_values = []
        tup = arguments[0]
        for tup_ind in range(tup_size):
            tup_item = builder.extract_value(tup, tup_ind)
            source_ty_ll = context.get_data_type(tup_ty[tup_ind])
            dest_ty_ll = context.get_data_type(uniform_ty)
            val = context.cast(builder, tup_item, source_ty_ll, dest_ty_ll)
            context.nrt.incref(builder, uniform_ty, val)
            array_values.append(val)
        return pack_array(builder, array_values, context.get_data_type(uniform_ty))
    sig = ret_type(tup_ty, uniform_ty_ref)
    return sig, codegen


@njit(**default_jit_options)
def uniformize_tuple_of_structs(tup, uniform_ty=VoidType):
    return _uniformize_tuple_of_structs(tup, uniform_ty)
