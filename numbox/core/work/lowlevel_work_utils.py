from llvmlite import ir
from numba.core.cgutils import int32_t, pack_array
from numba.core.types import boolean, FunctionType, NoneType, StructRef, unicode_type, UniTuple
from numba.extending import intrinsic

from numbox.core.work.work import _verify_signature, WorkTypeClass
from numbox.core.work.node_base import NodeBaseType, node_base_attributes
from numbox.utils.lowlevel import _new, populate_structref


def create_uniform_inputs(context, builder, tup_ty, tup, inputs_ty):
    tup_size = tup_ty.count
    for tup_ind in range(tup_size):
        item_ty = tup_ty[tup_ind]
        assert isinstance(item_ty, StructRef), "Only Tuple of StructRef supported"
    array_values = []
    for tup_ind in range(tup_size):
        tup_item = builder.extract_value(tup, tup_ind)
        source_ty_ll = context.get_data_type(tup_ty[tup_ind])
        dest_ty_ll = context.get_data_type(NodeBaseType)
        val = context.cast(builder, tup_item, source_ty_ll, dest_ty_ll)
        context.nrt.incref(builder, NodeBaseType, val)
        array_values.append(val)
    inputs = pack_array(builder, array_values, context.get_data_type(NodeBaseType))
    context.nrt.incref(builder, inputs_ty, inputs)
    return inputs


def store_inputs(context, builder, sources_ty, sources, data_pointer, inputs_ty):
    inputs = create_uniform_inputs(context, builder, sources_ty, sources, inputs_ty)
    inputs_p = builder.gep(data_pointer, (int32_t(0), int32_t(1)))
    builder.store(inputs, inputs_p)


def store_derived(builder, data_pointer):
    derived_p = builder.gep(data_pointer, (int32_t(0), int32_t(5)))
    builder.store(ir.IntType(1)(0), derived_p)


@intrinsic(prefer_literal=False)
def ll_make_work(typingctx, name_ty, data_ty, sources_ty, derive_ty):
    """
    Purely intrinsic work constructor, alternative to overloaded
    `numbox.core.work.work.Work`.

    Substantially more efficient in memory use, cache disk space, and
    compilation time for inlining multiple `Work` instantiations inside
    jitted context (e.g., in large-graph applications).

    (Alternatively, one can try `inline="always"` for the `make_work`
    which might save memory and cache disk space demand but significantly
    lengthens compilation time.)
    """
    assert isinstance(derive_ty, (FunctionType, NoneType)), f"""Either None or Compile Result supported,
not CPUDispatcher, got {derive_ty}, of type {type(derive_ty)}"""
    inputs_ty = UniTuple(NodeBaseType, sources_ty.count)
    work_attributes_ = node_base_attributes + [
        ("inputs", inputs_ty),
        ("data", data_ty),
        ("sources", sources_ty),
        ("derive", derive_ty),
        ("derived", boolean),
    ]
    _verify_signature(data_ty, sources_ty, derive_ty)
    work_type_ = WorkTypeClass(work_attributes_)
    args_names = ["name", "data", "sources", "derive"]

    def codegen(context, builder, signature, args):
        work_value, data_pointer = _new(context, builder, work_type_)
        populate_structref(context, builder, work_type_, args, data_pointer, args_names)
        if len(sources_ty) > 0:
            store_inputs(context, builder, sources_ty, args[2], data_pointer, inputs_ty)
        store_derived(builder, data_pointer)
        return work_value
    return work_type_(name_ty, data_ty, sources_ty, derive_ty), codegen
