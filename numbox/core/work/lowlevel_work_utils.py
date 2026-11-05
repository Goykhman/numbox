"""The numba implementation of ``Work`` nodes: their type class :class:`WorkTypeClass`, the intrinsic constructor
:func:`ll_make_work`, and the code that fills the fields of a new node. :mod:`numbox.core.work.work` defines the Python
class ``Work`` and its jitted methods for the types that this module creates.
"""
from numba.core.cgutils import get_null_value, int8_t, int32_t, pack_array
from numba.core.types import FunctionType, int8, NoneType, StructRef, TypeRef, UniTuple
from numba.experimental.structref import register
from numba.extending import intrinsic

from numbox.core.work.node import NodeType
from numbox.core.work.node_base import NodeBaseType, node_base_attributes
from numbox.utils.highlevel import determine_field_index
from numbox.utils.lowlevel import _new, populate_structref


def derive_ty_error(derive_ty):
    """Return the message for a derive type that is neither a first-class function type nor ``None``."""
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
    """The numba type class of ``Work`` nodes, with one type for each combination of data, source and derive types."""
    pass


def _create_work_type(data_ty, sources_ty, derive_ty, inputs_ty):
    """
    Return the numba type of a ``Work`` node with the given types of data, sources, derive and inputs.

    Each combination of these types is a type of its own, so the nodes of one graph can hold data of different types,
    depend on sources of different types and have derives of different signatures. The payload begins with the
    ``name`` field of :obj:`numbox.core.work.node_base.NodeBase`, so a ``Work`` node can be cast to ``NodeBaseType``.
    ``derive_ty`` must be a ``FunctionType`` or ``NoneType``, and a signature that does not match ``data_ty`` and the
    sources' data types raises ``ValueError``.
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
    """Emit code that casts each node of ``tup`` to ``NodeBaseType`` and packs the results into an array value.

    The emitted code increments the reference count of each cast value. Every element type of ``tup_ty`` must be a
    struct type, which an assertion checks.
    """
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
    """Emit code that stores the sources, cast to ``NodeBaseType``, in the ``inputs`` field of a new node's payload."""
    inputs = create_uniform_inputs(context, builder, sources_ty, sources)
    inputs_p = builder.gep(data_pointer, (int32_t(0), int32_t(inputs_index)))
    builder.store(inputs, inputs_p)


def store_derived(builder, data_pointer, derived_index):
    """Emit code that stores 0 in the ``derived`` field of a new node's payload."""
    derived_p = builder.gep(data_pointer, (int32_t(0), int32_t(derived_index)))
    builder.store(int8_t(0), derived_p)


def store_node(context, builder, data_pointer, node_index):
    """Emit code that stores a null ``NodeType`` in the ``node`` field of a new node, for ``as_node`` to fill."""
    node_p = builder.gep(data_pointer, (int32_t(0), int32_t(node_index)))
    builder.store(get_null_value(context.get_value_type(NodeType)), node_p)


def ensure_work_boxing():
    """
    Import :mod:`numbox.core.work.work`, which registers with numba how to box a ``Work`` node into a Python object.

    The import runs when this function is called, because ``numbox.core.work.work`` imports this module, and an import
    at the top of this module would be circular.
    """
    from numbox.core.work.work import Work  # noqa: F401


@intrinsic(prefer_literal=False)
def ll_make_work(typingctx, name_ty, data_ty, sources_ty, derive_ty, data_ty_ref: TypeRef = NoneType):
    """
    Create a ``Work`` node in jitted code, with the construction emitted directly into the calling function.

    The arguments and checks are those of :func:`~numbox.core.work.work.make_work`, without the upgrade of a
    ``CompileResultWAP``: a ``derive`` whose signature does not match the types of ``data`` and of the sources' data
    raises ``ValueError``. The optional ``data_ty_ref``, a numba type, sets the node's data type in place of the type
    of ``data``, and numba converts ``data`` to it. The ``derived`` field of the new node is 0. Calling the intrinsic
    from Python raises ``NotImplementedError``.

    ``ll_make_work`` uses much less memory, cache disk space and compilation time than ``make_work`` when one jitted
    function creates many nodes, for example to build a large graph. A ``make_work``-like constructor declared with
    ``inline="always"`` might also save memory and disk space if numba applies the directive, but it would take much
    longer to compile.
    """
    if data_ty_ref != NoneType:
        data_ty = data_ty_ref.instance_type
    _verify_signature(data_ty, sources_ty, derive_ty)
    inputs_ty = UniTuple(NodeBaseType, sources_ty.count)
    work_type_ = _create_work_type(data_ty, sources_ty, derive_ty, inputs_ty)
    ensure_work_boxing()
    ordered_args_names = ["name", "data", "sources", "derive"]

    def codegen(context, builder, signature, args):
        """Emit the allocation of the node and the stores that fill its fields."""
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
