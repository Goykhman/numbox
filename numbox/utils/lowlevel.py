"""Helpers for raw memory in jitted code: addresses held as integers, loads and stores, and casts between structrefs.

Most helpers are jitted functions that Python code can call too, usually built on an ``@intrinsic`` of the same name
with a leading underscore. An address is an ``intp``, the integer type that numbox's bindings take for a pointer. The
helpers check neither that an address points to live memory nor that the memory has the layout of the given type;
the caller must ensure both for as long as the address is used. The functions that take a context and a builder,
such as :func:`extract_struct_member`, are for code inside an ``@intrinsic``.

None of the jitted helpers has a fingerprint, because each one calls an ``@intrinsic`` or reads a numba type. If a
struct method, a ``compile_kernel`` formula or a derive calls one of them by its name, the code that numbox generates
for it compiles without an on-disk cache. ``digest`` computes the weaker fingerprint that the utils page describes for
a SQLite callback that calls one of them.
"""
from llvmlite import ir
from llvmlite.ir.builder import IRBuilder
from numba import carray, njit
from numba.core.base import BaseContext
from numba.core.cgutils import int32_t, intp_t, is_not_null as cgutils_is_not_null, pack_array, voidptr_t
from numba.experimental.jitclass.base import imp_dtor
from numba.experimental.structref import _Utils
from numba.core.types import (
    boolean, FunctionType, intp, StructRef, TypeRef, Tuple, uint8,
    UnicodeType, unicode_type, uintp, UniTuple, void, voidptr
)
from numba.core.types.misc import unliteral
from numba.core.errors import TypingError
from numba.core.typing.context import Context
from numba.extending import intrinsic

from numbox.core.configurations import function_struct_size, jit_options, MAX_STR_LENGTH
from numbox.core.configurations import numba_version  # noqa: F401  re-exported for existing importers
from numbox.utils.void_type import VoidType
from numbox.utils.highlevel import determine_field_index


@intrinsic
def _cast(typingctx: Context, source_ty, dest_ty_ref: TypeRef):
    dest_ty = dest_ty_ref.instance_type
    sig = dest_ty(source_ty, dest_ty_ref)

    def codegen(context: BaseContext, builder, signature, args):
        """Emit the cast of the structref to the destination type, with a new reference to the result."""
        source_ty_ll = context.get_data_type(source_ty)
        dest_ty_ll = context.get_data_type(dest_ty)
        val = context.cast(builder, args[0], source_ty_ll, dest_ty_ll)
        context.nrt.incref(builder, dest_ty, val)
        return val
    return sig, codegen


@njit(**jit_options)
def cast(source, dest_ty):
    """Return the structref ``source`` as a value of the structref type ``dest_ty``, over the same memory.

    The data is not converted, so the fields of ``dest_ty`` must match the leading fields of the memory that ``source``
    refers to; a type with no fields, such as ``VoidType``, always matches. The result holds a reference of its own.
    """
    return _cast(source, dest_ty)


@intrinsic
def _cast_int_to_void_p(typingctx, p_ty):
    """Return the integer address ``p`` as a ``voidptr``, in jitted code."""
    def codegen(context, builder, signature, arguments):
        """Emit the ``inttoptr`` of the integer to ``i8*``."""
        return builder.inttoptr(arguments[0], voidptr_t)
    return voidptr(p_ty), codegen


@intrinsic
def _cast_void_p_to_int(typingctx, p_ty):
    """Return the pointer ``p`` as an ``intp`` address, in jitted code."""
    def codegen(context, builder, signature, args):
        """Emit the ``ptrtoint`` of the pointer to ``intp``."""
        return builder.ptrtoint(args[0], intp_t)
    return intp(p_ty), codegen


@intrinsic
def _load_at(typingctx: Context, p_ty, ty_ref: TypeRef):
    if unliteral(p_ty) not in (intp, uintp):
        raise TypingError(
            f"load_at: pointer argument must be intp or uintp, got {p_ty}"
        )
    ty = ty_ref.instance_type
    sig = ty(p_ty, ty_ref)

    def codegen(context: BaseContext, builder, signature, args):
        """Emit the ``inttoptr`` of the address and a load of the value at it."""
        ty_ll = context.get_data_type(ty)
        ptr = builder.inttoptr(args[0], ty_ll.as_pointer())
        return builder.load(ptr)
    return sig, codegen


@njit(**jit_options)
def load_at(p, ty):
    """Return the value of the numba type ``ty`` stored at the address ``p``, an ``intp`` or a ``uintp``.

    The caller must ensure that ``p`` points to live memory that holds a value in the LLVM layout of ``ty``, aligned
    for that type; :func:`load_unaligned` reads from an address that may be misaligned. An address of another integer
    type raises ``TypingError``.
    """
    return _load_at(p, ty)


@intrinsic
def _store_at(typingctx: Context, p_ty, v_ty):
    if unliteral(p_ty) not in (intp, uintp):
        raise TypingError(
            f"store_at: pointer argument must be intp or uintp, got {p_ty}"
        )
    sig = void(p_ty, v_ty)

    def codegen(context: BaseContext, builder, signature, args):
        """Emit the ``inttoptr`` of the address and a store of the value at it."""
        p, v = args
        ty_ll = context.get_data_type(v_ty)
        ptr = builder.inttoptr(p, ty_ll.as_pointer())
        builder.store(v, ptr)
    return sig, codegen


@njit(**jit_options)
def store_at(p, v):
    """Store ``v`` at the address ``p``, an ``intp`` or a ``uintp``, in the LLVM layout of the numba type of ``v``.

    The numba type of ``v`` determines how many bytes are written. A Python ``int`` is an ``int64``, so the function
    writes eight bytes for it. To write fewer, convert ``v`` first: ``store_at(p, int32(value))`` writes four bytes.
    The caller must ensure that ``p`` points to writable memory with room for the value, aligned for its type;
    :func:`store_unaligned` writes to an address that may be misaligned. An address of another integer type raises
    ``TypingError``.
    """
    return _store_at(p, v)


@intrinsic
def _store_unaligned(typingctx: Context, p_ty, v_ty):
    if unliteral(p_ty) not in (intp, uintp):
        raise TypingError(
            f"store_unaligned: pointer argument must be intp or uintp, got {p_ty}"
        )
    sig = void(p_ty, v_ty)

    def codegen(context: BaseContext, builder, signature, args):
        """Emit the ``inttoptr`` of the address and a store with an alignment of 1."""
        p, v = args
        ty_ll = context.get_data_type(v_ty)
        ptr = builder.inttoptr(p, ty_ll.as_pointer())
        builder.store(v, ptr, align=1)
    return sig, codegen


@njit(**jit_options)
def store_unaligned(p, v):
    """Store ``v`` at the address ``p`` as :func:`store_at` does, without assuming any alignment.

    The store is correct at a misaligned address, such as a field of a packed structured dtype, where
    :func:`store_at` has undefined behavior.
    """
    return _store_unaligned(p, v)


@intrinsic
def _load_unaligned(typingctx: Context, p_ty, ty_ref: TypeRef):
    if unliteral(p_ty) not in (intp, uintp):
        raise TypingError(
            f"load_unaligned: pointer argument must be intp or uintp, got {p_ty}"
        )
    ty = ty_ref.instance_type
    sig = ty(p_ty, ty_ref)

    def codegen(context: BaseContext, builder, signature, args):
        """Emit the ``inttoptr`` of the address and a load with an alignment of 1."""
        ty_ll = context.get_data_type(ty)
        ptr = builder.inttoptr(args[0], ty_ll.as_pointer())
        return builder.load(ptr, align=1)
    return sig, codegen


@njit(**jit_options)
def load_unaligned(p, ty):
    """Return the value of the type ``ty`` at the address ``p`` as :func:`load_at` does, without assuming any alignment.

    The load is correct at a misaligned address, such as a field of a packed structured dtype, where :func:`load_at`
    has undefined behavior.
    """
    return _load_unaligned(p, ty)


@intrinsic
def _deref_payload(typingctx: Context, struct_ty, ty_ref: TypeRef):
    ty = ty_ref.instance_type
    sig = ty(struct_ty, ty_ref)

    def codegen(context: BaseContext, builder, signature, args):
        """Emit a load of the structref's data as the type, with a new reference to the loaded value."""
        ty_ll = context.get_data_type(ty)
        struct_ = args[0]
        _, meminfo_p = context.nrt.get_meminfos(builder, struct_ty, struct_)[0]
        payload_p = context.nrt.meminfo_data(builder, meminfo_p)
        x_as_ty_p = builder.bitcast(payload_p, ty_ll.as_pointer())
        val = builder.load(x_as_ty_p)
        context.nrt.incref(builder, ty, val)
        return val
    return sig, codegen


@njit(**jit_options)
def deref_payload(p, ty):
    """Return the data of the structref ``p`` read as the numba type ``ty``, with a reference of its own.

    The data is read from its start without conversion, so ``ty`` must match the layout of the data or of its leading
    part, such as the type of the structref's first field.
    """
    return _deref_payload(p, ty)


def extract_struct_member(
    context: BaseContext,
    builder: IRBuilder,
    struct_fe_ty: StructRef,
    struct_obj,
    member_name: str,
    incref: bool = False
):
    """Emit a load of the field ``member_name`` of the structref ``struct_obj``, of the numba type ``struct_fe_ty``.

    The function is for code inside an ``@intrinsic``. ``member_name`` is a Python string known when the code is
    generated, and a name that is not a field raises ``KeyError``. The loaded value is borrowed from the structref and
    must not outlive it. If ``incref`` is True, it holds a new reference instead, as a value that an intrinsic returns
    must hold.
    """
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
    """Emit the extraction of the C entry point, the first of the three addresses, from a first-class function value.

    The function is for code inside an ``@intrinsic``. ``func_struct`` is the LLVM value of the function value, and the
    result is an ``i8*``.
    """
    func_raw_p_ind = 0
    return builder.extract_value(func_struct, func_raw_p_ind)


@intrinsic
def _get_func_p_as_int_from_func_struct(typingctx, func_ty):
    def codegen(context, builder, signature, args):
        """Emit the ``ptrtoint`` of the value's C entry point to ``intp``."""
        return builder.ptrtoint(get_func_p_from_func_struct(builder, args[0]), intp_t)
    return intp(func_ty), codegen


@njit(**jit_options)
def get_func_p_as_int_from_func_struct(func_):
    """Return the address of the C entry point of the first-class function value ``func_``, as an ``intp``."""
    return _get_func_p_as_int_from_func_struct(func_)


@intrinsic
def _get_func_tuple(typingctx, func_ty):
    return_tuple_type = UniTuple(intp, function_struct_size)
    sig = return_tuple_type(func_ty)

    def codegen(context, builder, signature, args):
        """Emit the ``ptrtoint`` of each address in the function value, and return the results as a tuple."""
        func_struct = args[0]
        lst = []
        for i in range(function_struct_size):
            lst.append(builder.ptrtoint(builder.extract_value(func_struct, i), intp_t))
        return context.make_tuple(builder, return_tuple_type, lst)
    return sig, codegen


@njit(**jit_options)
def get_func_tuple(func):
    """Return the three addresses that the first-class function value ``func`` holds, as a tuple of ``intp``.

    The addresses are its C entry point, its Python object and its numba entry point. The numba entry point is 0 if the
    value has none, as for a ``@cfunc``; :mod:`numbox.utils.derive_wap` describes the two entry points.
    """
    return _get_func_tuple(func)


def get_ll_func_sig(context: BaseContext, func_ty: FunctionType):
    """Return the LLVM function type whose return and argument types are the data types of the signature of ``func_ty``.

    The function is for code inside an ``@intrinsic``. numbox casts the C entry point of a first-class function value
    to this type before it calls the entry point.
    """
    func_sig = func_ty.signature
    arg_types = []
    for arg in func_sig.args:
        arg_types.append(context.get_data_type(arg))
    return ir.FunctionType(context.get_data_type(func_sig.return_type), arg_types)


@njit(unicode_type(intp), **jit_options)
def get_str_from_p_as_int(p):
    """Return a copy of the NUL-terminated C string at the address ``p`` as a ``str``, with one character per byte.

    Each byte becomes the character of the same code, 0 to 255, as Latin-1 decoding gives. The UTF-8 bytes of a
    non-ASCII text are therefore returned undecoded; for a returned string ``s``,
    ``s.encode("latin-1").decode("utf-8")`` gives the text. The function reads at most ``MAX_STR_LENGTH`` bytes,
    ``2**31 - 1``, and returns what it has read if it finds no NUL before that.
    """
    void_p = _cast_int_to_void_p(p)
    s = ""
    mem_view = carray(void_p, shape=(MAX_STR_LENGTH,), dtype=uint8)
    for char_as_code_p in mem_view:
        if char_as_code_p == 0:
            return s
        s += chr(char_as_code_p)
    return s


@intrinsic
def _get_unicode_data_p(typingctx, str_ty):
    def codegen(context, builder, signature, arguments):
        """Emit the extraction of the string's data pointer, as an ``intp``."""
        str_ = arguments[0]
        # The data pointer is field 0 of numba's unicode structure:
        # https://github.com/numba/numba/blob/0.64.0/numba/cpython/unicode.py#L86
        data_p_ind = 0
        data_p = builder.extract_value(str_, data_p_ind)
        return builder.ptrtoint(data_p, context.get_data_type(intp))
    # numba types some string expressions as optional, such as the product "A" * n, whose type is
    # OptionalType(unicode_type). Such an argument fails this assert with a bare AssertionError.
    assert isinstance(str_ty, UnicodeType)
    return intp(str_ty), codegen


@njit(**jit_options)
def get_unicode_data_p(s):
    """Return the address of the character data of the string ``s``, as an ``intp``, without a copy.

    The string stores one, two or four bytes per character, as its widest character needs, followed by a NUL, so the
    bytes are the text's UTF-8 only for ASCII text; :func:`numbox.utils.cstrings.c_string` makes a UTF-8 copy. The
    address is valid only while ``s`` is alive. A string that numba types as optional, such as ``"A" * n``, raises a
    bare ``AssertionError``. numba's unicode structure is defined in
    https://github.com/numba/numba/blob/release0.61/numba/cpython/unicode.py#L83.
    """
    return _get_unicode_data_p(s)


@intrinsic
def is_not_null(typingctx, struct_ty):
    """Return whether a structref has a MemInfo, that is, whether its MemInfo pointer is not NULL, in jitted code."""
    def codegen(context, builder, signature, arguments):
        """Emit the comparison of the structref's MemInfo pointer with NULL."""
        struct_ = arguments[0]
        meminfo = context.nrt.get_meminfos(builder, struct_ty, struct_)[0]
        type_data, meminfo_p = meminfo
        return cgutils_is_not_null(builder, meminfo_p)
    return boolean(struct_ty), codegen


def _new(context, builder, struct_ty_):
    """Emit the allocation of a new structref of the type ``struct_ty_``, and return it with a pointer to its data.

    The structref is a structure whose one member is the pointer to its MemInfo. The code follows numba's
    `new <https://github.com/numba/numba/blob/release0.61/numba/experimental/structref.py#L301>`_, except that the data
    is not zeroed, so the caller must store every field before the structref is used.
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
    # numba's new zeroes the data at this point with builder.store(get_null_value(alloc_type), data_pointer). _new
    # leaves the data uninitialized, so the caller must store every field before the structref is used, and
    # populate_structref must not release the old values.
    work_ = context.make_helper(builder, struct_ty_)
    work_.meminfo = meminfo
    return work_._getvalue(), data_pointer


def populate_structref(
    context, builder, signature, structref_type_, structref_, args, ordered_args_names, decref_old=False
):
    """Emit the stores of ``args`` into the fields named by ``ordered_args_names`` of the structref ``structref_``.

    The function is for code inside an ``@intrinsic``. ``structref_type_`` is the structref's numba type, and each
    argument is cast from its type in ``signature.args`` to the type of its field, which takes a new reference to it.
    With ``decref_old=True``, the old value of each field is released after the new value is referenced. Leave it False
    for a structref that ``_new`` has just allocated, because its fields hold no values to release. The code follows
    numba's ``struct_setattr_impl`` in ``numba.experimental.structref.define_attributes``.
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
    """Return the MemInfo addresses of the structrefs in a tuple, as a tuple of ``intp``, in jitted code.

    The addresses take no reference. Prefer :func:`uniformize_tuple_of_structs` where possible: its result holds
    references, so numba keeps the structrefs alive and releases them.
    """
    tup_size = tup_ty.count

    def codegen(context, builder, signature, arguments):
        """Emit the ``ptrtoint`` of each element's MemInfo pointer, and return the results as a tuple."""
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


@njit(**jit_options)
def tuple_of_struct_ptrs_as_int(tup):
    """Return the MemInfo addresses of the structrefs in the tuple ``tup``, as a tuple of ``intp``.

    The addresses take no reference, so they are valid only while the structrefs are alive.
    """
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
        """Emit the cast of each element to the uniform type, with a new reference, and return the casts as a tuple."""
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


@njit(**jit_options)
def uniformize_tuple_of_structs(tup, uniform_ty=VoidType):
    """Return the structrefs of ``tup`` as a tuple of the one structref type ``uniform_ty``, by default ``VoidType``.

    The function casts each element as :func:`cast` does, over the same memory, and each result holds a reference of
    its own. An element that is not a structref raises ``AssertionError`` when the function is compiled.
    """
    return _uniformize_tuple_of_structs(tup, uniform_ty)


@njit
def array_data_p(arr):
    """Return the address of the data of the NumPy array ``arr`` as an ``intp``, the type of a pointer in the bindings.

    Python code and jitted code can call it. The address takes no reference, so it is valid only while ``arr`` is
    alive. If ``NUMBA_DISABLE_JIT=1`` is set, the function runs as Python and returns a ``numpy.int64``, which ctypes
    functions such as ``ctypes.memmove`` reject.
    """
    # numba types arr.ctypes.data as uintp. The conversion gives the signed intp that numbox's binding signatures use
    # for a pointer.
    return intp(arr.ctypes.data)
