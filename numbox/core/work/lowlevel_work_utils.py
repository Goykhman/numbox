from numba.core.cgutils import get_null_value, int8_t, int32_t, pack_array
from numba.core.types import FunctionType, int8, NoneType, StructRef, TypeRef, UniTuple
from numba.experimental.structref import register
from numba.extending import intrinsic

from numbox.core.work.node import NodeType
from numbox.core.work.node_base import NodeBaseType, node_base_attributes
from numbox.utils.highlevel import determine_field_index
from numbox.utils.lowlevel import _new, populate_structref


def derive_ty_error(derive_ty):
    return f"Either None or Compile Result supported, not CPUDispatcher, got {derive_ty}, of type {type(derive_ty)}"


def _verify_signature(data_ty, sources_ty, derive_ty):
    args_ty = []
    for source_ind in range(sources_ty.count):
        source_ty = sources_ty[source_ind]
        source_data_ty = source_ty.field_dict["data"]
        args_ty.append(source_data_ty)
    derive_sig = data_ty(*args_ty)
    if isinstance(derive_ty, FunctionType):
        if derive_ty.signature != derive_sig:
            raise ValueError(
                f"""Signatures do not match, derive defines {derive_ty.signature}
but data and sources imply {derive_sig}"""
            )
    else:
        assert isinstance(derive_ty, NoneType), derive_ty_error(derive_ty)


@register
class WorkTypeClass(StructRef):
    pass


def _create_work_type(data_ty, sources_ty, derive_ty, inputs_ty):
    """
    Dynamically create `WorkType`, depending on the type of `data`, `sources`, and `derive`.

    Different instances of `Work` accommodate various types of data they might contain,
    various heterogeneous types of other instances of `Work` it might depend on,
    and custom `derive` function objects used to calculate the instance's `data` depending
    on the `data` of its `sources`.

    (`name`,) tuple initializes the :obj:`numbox.core.node_base.NodeBase`
    header of the composition.
    """
    assert isinstance(derive_ty, (FunctionType, NoneType)), derive_ty_error(derive_ty)
    work_attributes_ = node_base_attributes + [
        ("inputs", inputs_ty),
        ("data", data_ty),
        ("sources", sources_ty),
        ("derive", derive_ty),
        ("derived", int8),
        ("node", NodeType)
    ]
    _verify_signature(data_ty, sources_ty, derive_ty)
    work_type_ = WorkTypeClass(work_attributes_)
    return work_type_


def create_uniform_inputs(context, builder, tup_ty, tup):
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
    return inputs


def store_inputs(context, builder, sources_ty, sources, data_pointer, inputs_index):
    inputs = create_uniform_inputs(context, builder, sources_ty, sources)
    inputs_p = builder.gep(data_pointer, (int32_t(0), int32_t(inputs_index)))
    builder.store(inputs, inputs_p)


def store_derived(builder, data_pointer, derived_index):
    derived_p = builder.gep(data_pointer, (int32_t(0), int32_t(derived_index)))
    builder.store(int8_t(0), derived_p)


def store_node(context, builder, data_pointer, node_index):
    node_p = builder.gep(data_pointer, (int32_t(0), int32_t(node_index)))
    builder.store(get_null_value(context.get_value_type(NodeType)), node_p)


def ensure_work_boxing():
    """
    this will trigger `define_boxing` in the imported `numbox.core.work.work`;
    resolves circular import and prepares boxing to Python object of
    the low-level created work types.
    """
    from numbox.core.work.work import Work  # noqa: F401


@intrinsic(prefer_literal=False)
def ll_make_work(typingctx, name_ty, data_ty, sources_ty, derive_ty, data_ty_ref: TypeRef = NoneType):
    """
    Purely intrinsic work constructor.

    Substantially more efficient in memory use, cache disk space, and
    compilation time for inlining multiple `Work` instantiations inside
    jitted context (e.g., in large-graph applications).

    Alternative `make_work`-like constructors with `inline="always"`
    might save memory and cache disk space demand (when the inlining
    directive is actually heeded by numba engine) but significantly
    lengthen the compilation time.
    """
    if data_ty_ref != NoneType:
        data_ty = data_ty_ref.instance_type
    _verify_signature(data_ty, sources_ty, derive_ty)
    inputs_ty = UniTuple(NodeBaseType, sources_ty.count)
    work_type_ = _create_work_type(data_ty, sources_ty, derive_ty, inputs_ty)
    ensure_work_boxing()
    ordered_args_names = ["name", "data", "sources", "derive"]

    def codegen(context, builder, signature, args):
        work_value, data_pointer = _new(context, builder, work_type_)
        populate_structref(context, builder, signature, work_type_, work_value, args[:4], ordered_args_names)
        if len(sources_ty) > 0:
            store_inputs(
                context, builder, sources_ty, args[2], data_pointer,
                determine_field_index(work_type_, "inputs")
            )
        store_derived(builder, data_pointer, determine_field_index(work_type_, "derived"))
        store_node(context, builder, data_pointer, determine_field_index(work_type_, "node"))
        return work_value
    return work_type_(name_ty, data_ty, sources_ty, derive_ty, data_ty_ref), codegen
