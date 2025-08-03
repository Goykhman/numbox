from inspect import getfile, getmodule
from io import StringIO
from numba import njit
from numba.core.errors import NumbaError
from numba.core.types import (
    boolean, DictType, FunctionType, Literal, NoneType, Tuple, unicode_type, UnicodeType
)
from numba.core.typing.context import Context
from numba.experimental.structref import define_boxing, new
from numba.extending import intrinsic, overload, overload_method
from numba.typed.typeddict import Dict
from numba.typed.typedlist import List

from numbox.core.any.erased_type import ErasedType
from numbox.core.configurations import default_jit_options
from numbox.core.work.lowlevel_work_utils import ll_make_work, WorkTypeClass
from numbox.core.work.node import NodeType
from numbox.core.work.node_base import NodeBase, NodeBaseType
from numbox.utils.lowlevel import (
    extract_struct_member, _cast, _get_func_tuple, is_not_null, get_func_p_from_func_struct, get_ll_func_sig
)


def _file_anchor():
    raise NotImplementedError


deleted_work_ctor_error = "Use `make_work` instead"


class Work(NodeBase):
    """
    Structure describing a unit of work.

    Instances of this class can be connected in a graph with other `Work` instances.

    Attributes
    ----------
    name : str
        Name of the structure instance.
    inputs : UniTuple[NodeBaseType]
        Uniform tuple of `Work.sources`, cast as `NodeBaseType`.
    data : Any
        Scalar or array data payload contained in (and calculated by) this structure.
    sources : Tuple[Work, ...]
        Heterogeneous tuple of `Work` instances that this `Work` instance depends on.
    derive : FunctionType
        Function of the signature determined by the data types of `sources` and `data`.
    derived : int8
        Flag indicating whether the `data` has already been calculated.
    node : NodeType
        Work as Node, with its sources in a List.

    (`name`, ) attributes of the `Work` structure payload are
    homogeneously typed across all instances of `Work` and accommodate
    cast-ability to the :obj:`numbox.core.node_base.NodeBase` base of `NodeBaseType`.
    """
    def __new__(cls, *args, **kws):
        raise NotImplementedError(deleted_work_ctor_error)

    @property
    @njit(**default_jit_options)
    def inputs(self):
        return self.inputs

    @property
    @njit(**default_jit_options)
    def data(self):
        return self.data

    @property
    @njit(**default_jit_options)
    def sources(self):
        return self.sources

    @property
    @njit(**default_jit_options)
    def derive(self):
        return _get_func_tuple(self.derive)

    @property
    @njit(**default_jit_options)
    def derived(self):
        return self.derived

    @njit(**default_jit_options)
    def as_node(self):
        return self.as_node()

    @njit(**default_jit_options)
    def calculate(self):
        return self.calculate()

    @njit(**default_jit_options)
    def load(self, data):
        return self.load(data)

    @njit(**default_jit_options)
    def combine(self, data):
        return self.combine(data)

    @njit(**default_jit_options)
    def get_input(self, i):
        return self.get_input(i)

    def get_inputs_names(self):
        return list(self._get_inputs_names())

    @njit(**default_jit_options)
    def _get_inputs_names(self):
        return self.get_inputs_names()

    @njit(**default_jit_options)
    def make_inputs_vector(self):
        return self.make_inputs_vector()

    def all_inputs_names(self):
        return list(self._all_inputs_names())

    @njit(**default_jit_options)
    def _all_inputs_names(self):
        return self.all_inputs_names()

    def all_end_nodes(self):
        return list(self._all_end_nodes())

    @njit(**default_jit_options)
    def _all_end_nodes(self):
        return self.all_end_nodes()

    @njit(**default_jit_options)
    def depends_on(self, obj_):
        return self.depends_on(obj_)


define_boxing(WorkTypeClass, Work)


def _deleted_work_ctor(*args, **kwargs):
    raise NumbaError(deleted_work_ctor_error)


overload(Work, jit_options=default_jit_options)(_deleted_work_ctor)


@njit(**default_jit_options)
def make_work(name, data, sources=(), derive=None):
    return ll_make_work(name, data, sources, derive)


@intrinsic
def _call_derive(typingctx: Context, derive_ty: FunctionType, sources_ty: Tuple):
    def codegen(context, builder, signature, arguments):
        derive_struct, sources = arguments
        derive_args = []
        for source_ind, source_ty in enumerate(sources_ty):
            source = builder.extract_value(sources, source_ind)
            data = extract_struct_member(context, builder, source_ty, source, "data")
            derive_args.append(data)
        derive_p_raw = get_func_p_from_func_struct(builder, derive_struct)
        derive_ty_ll = get_ll_func_sig(context, derive_ty)
        derive_p = builder.bitcast(derive_p_raw, derive_ty_ll.as_pointer())
        res = builder.call(derive_p, derive_args)
        return res
    sig = derive_ty.signature.return_type(derive_ty, sources_ty)
    return sig, codegen


_source_getter_registry = {}


def _make_source_getter(source_ind):
    return f"""
@intrinsic
def _get_source_{source_ind}(typingctx: Context, sources_ty: Tuple):
    def codegen(context, builder, signature, arguments):
        sources = arguments[0]
        val = builder.extract_value(sources, {source_ind})
        context.nrt.incref(builder, sources_ty[{source_ind}], val)
        return val
    sig = sources_ty[{source_ind}](sources_ty)
    return sig, codegen
"""


def _make_calculate_code(num_sources):
    code_txt = StringIO()
    code_txt.write("""
def _calculate_(work_):
    if work_.derived:
        return""")
    if num_sources > 0:
        code_txt.write("""
    sources = work_.sources""")
        for source_ind_ in range(num_sources):
            code_txt.write(f"""
    source_{source_ind_} = _get_source_{source_ind_}(sources)
    if not source_{source_ind_}.derived:
        source_{source_ind_}.calculate()""")
    code_txt.write("""
    v = _call_derive(work_.derive, work_.sources)
    work_.derived = True
    work_.data = v
""")
    return code_txt.getvalue()


_calculate_registry = {}


def ensure_presence_of_source_getters_in_ns(num_sources_, ns_):
    for source_i in range(num_sources_):
        source_getter_code_txt = _make_source_getter(source_i)
        source_getter_code = compile(source_getter_code_txt, getfile(_file_anchor), mode="exec")
        exec(source_getter_code, ns_)
        _source_getter_registry[source_i] = True


@overload_method(WorkTypeClass, "calculate", strict=False, jit_options=default_jit_options)
def ol_calculate(self_ty):
    derive_ty = self_ty.field_dict["derive"]
    if isinstance(derive_ty, NoneType):
        def _(self):
            self.derived = True
        return _

    sources_ty = self_ty.field_dict["sources"]
    num_sources = sources_ty.count
    _calculate = _calculate_registry.get(num_sources, None)
    if _calculate is not None:
        return _calculate
    ns = getmodule(_file_anchor).__dict__
    ensure_presence_of_source_getters_in_ns(num_sources, ns)
    code_txt = _make_calculate_code(num_sources)
    code = compile(code_txt, getfile(_file_anchor), mode="exec")
    exec(code, ns)
    _calculate = ns["_calculate_"]
    _calculate_registry[num_sources] = _calculate
    return _calculate


@intrinsic
def _cast_to_work_data(typingctx, work_ty, data_as_erased_ty: ErasedType):
    data_ty = work_ty.field_dict["data"]
    sig = data_ty(work_ty, data_as_erased_ty)

    def codegen(context, builder, signature, arguments):
        data_as_erased = arguments[1]
        data_ty_ll = context.get_data_type(data_ty)
        _, meminfo_p = context.nrt.get_meminfos(builder, data_as_erased_ty, data_as_erased)[0]
        payload_p = context.nrt.meminfo_data(builder, meminfo_p)
        x_as_ty_p = builder.bitcast(payload_p, data_ty_ll.as_pointer())
        val = builder.load(x_as_ty_p)
        context.nrt.incref(builder, data_ty, val)
        return val
    return sig, codegen


def _make_loader_code(num_sources):
    code_txt = StringIO()
    code_txt.write("""
def _loader_(work_, data_):
    reset = False
    work_name = work_.name
    if work_name in data_:
        work_.data = _cast_to_work_data(work_, data_[work_name].p)
        reset = True""")
    if num_sources > 0:
        code_txt.write("""
    sources = work_.sources""")
        for source_ind_ in range(num_sources):
            code_txt.write(f"""
    source_{source_ind_} = _get_source_{source_ind_}(sources)
    reset_source = source_{source_ind_}.load(data_)
    reset = reset or reset_source
""")
    code_txt.write("""
    work_.derived = work_.derived and not reset
    return reset
""")
    return code_txt.getvalue()


_loader_registry = {}


@overload_method(WorkTypeClass, "load", strict=False, jit_options=default_jit_options)
def ol_load(work_ty, data_ty: DictType):
    """ Load `data` into the graph with the root node `work`.
     Data is provided as dictionary mapping node name to `Any` type
     containing erased payload `p` of the `work.data` type. """
    sources_ty = work_ty.field_dict["sources"]
    num_sources = sources_ty.count
    _loader = _loader_registry.get(num_sources, None)
    if _loader is not None:
        return _loader
    ns = getmodule(_file_anchor).__dict__
    ensure_presence_of_source_getters_in_ns(num_sources, ns)
    code_txt = _make_loader_code(num_sources)
    code = compile(code_txt, getfile(_file_anchor), mode="exec")
    exec(code, ns)
    _loader = ns["_loader_"]
    _loader_registry[num_sources] = _loader
    return _loader


def _make_combine_code(num_sources):
    code_txt = StringIO()
    code_txt.write("""
def _combine_(work_, data_, harvested_=None):
    if harvested_ is None:
        harvested_ = Dict.empty(key_type=unicode_type, value_type=boolean)
    if len(harvested_) == len(data_):
        return
    work_name = work_.name
    if work_name in data_:
        harvested_[work_name] = True
        data_[work_name].reset(work_.data)""")
    if num_sources > 0:
        code_txt.write("""
    sources = work_.sources""")
        for source_ind_ in range(num_sources):
            code_txt.write(f"""
    source_{source_ind_} = _get_source_{source_ind_}(sources)
    source_{source_ind_}.combine(data_, harvested_)
""")
    code_txt.write("""
    return
""")
    return code_txt.getvalue()


_combine_registry = {}


@overload_method(WorkTypeClass, "combine", strict=False, jit_options=default_jit_options)
def ol_combine(work_ty, data_ty: DictType, harvested_ty=NoneType):
    """ Harvest nodes data from the graph with the root node `work`.
     `data` is provided as dictionary mapping node name to `Any` type
     containing erased payload `p` to be reset to `data`. """
    sources_ty = work_ty.field_dict["sources"]
    num_sources = sources_ty.count
    _combine = _combine_registry.get(num_sources, None)
    if _combine is not None:
        return _combine
    ns = {**getmodule(_file_anchor).__dict__, **{"boolean": boolean, "Dict": Dict}}
    ensure_presence_of_source_getters_in_ns(num_sources, ns)
    code_txt = _make_combine_code(num_sources)
    code = compile(code_txt, getfile(_file_anchor), mode="exec")
    exec(code, ns)
    _combine = ns["_combine_"]
    _combine_registry[num_sources] = _combine
    return _combine


@overload_method(WorkTypeClass, "get_input", strict=False, jit_options=default_jit_options)
def ol_get_input(self_ty, i_ty):
    def _(self, i):
        num_inputs = len(self.inputs)
        if i >= num_inputs:
            raise NumbaError(f"Requested input {i} while the node has {num_inputs} inputs")
        inputs_vector = self.make_inputs_vector()
        return _cast(inputs_vector[i], NodeType)
    return _


@overload_method(WorkTypeClass, "get_inputs_names", strict=False, jit_options=default_jit_options)
def ol_get_inputs_names(self_ty):
    def _(self):
        names_ = List.empty_list(unicode_type)
        for s in self.inputs:
            names_.append(s.name)
        return names_
    return _


def _make_inputs_vector_code(num_sources):
    code_txt = StringIO()
    code_txt.write("""
def _inputs_vector_(work_):
    node = work_.node
    if is_not_null(node):
        return node.inputs
    inputs_vector = List.empty_list(NodeBaseType)""")
    if num_sources > 0:
        code_txt.write("""
    sources = work_.sources""")
        for source_ind_ in range(num_sources):
            code_txt.write(f"""
    source_{source_ind_} = _get_source_{source_ind_}(sources)
    node = source_{source_ind_}.as_node()
    inputs_vector.append(_cast(node, NodeBaseType))
""")
    code_txt.write("""
    return inputs_vector""")
    return code_txt.getvalue()


_inputs_vector_registry = {}


@overload_method(WorkTypeClass, "make_inputs_vector", strict=False, jit_options=default_jit_options)
def ol_make_inputs_vector(self_ty):
    sources_ty = self_ty.field_dict["sources"]
    num_sources = sources_ty.count
    if num_sources == 0:
        def _(self):
            inputs_vector = List.empty_list(NodeBaseType)
            return inputs_vector
        return _
    _inputs_vector = _inputs_vector_registry.get(num_sources, None)
    if _inputs_vector is not None:
        return _inputs_vector
    ns = {**getmodule(_file_anchor).__dict__, **{"new": new}}
    ensure_presence_of_source_getters_in_ns(num_sources, ns)
    code_txt = _make_inputs_vector_code(num_sources)
    code = compile(code_txt, getfile(_file_anchor), mode="exec")
    exec(code, ns)
    _inputs_vector = ns["_inputs_vector_"]
    _inputs_vector_registry[num_sources] = _inputs_vector
    return _inputs_vector


@overload_method(WorkTypeClass, "as_node", strict=False, jit_options=default_jit_options)
def ol_as_node(self_ty):
    def _(self):
        node = self.node
        if is_not_null(node):
            return node
        node = new(NodeType)
        node.name = self.name
        node.inputs = self.make_inputs_vector()
        self.node = node
        return node
    return _


@overload_method(WorkTypeClass, "all_inputs_names", strict=False, jit_options=default_jit_options)
def ol_all_inputs_names(self_ty):
    def _(self):
        node = self.as_node()
        return node.all_inputs_names()
    return _


@overload_method(WorkTypeClass, "all_end_nodes", strict=False, jit_options=default_jit_options)
def ol_all_end_nodes(self_ty):
    def _(self):
        node = self.as_node()
        return node.all_end_nodes()
    return _


@overload_method(WorkTypeClass, "depends_on", strict=False, jit_options=default_jit_options)
def ol_depends_on(self_ty, obj_ty):
    if isinstance(obj_ty, (Literal, UnicodeType,)):
        def _(self, name_):
            node = self.as_node()
            return name_ in node.all_inputs_names()
    else:
        def _(self, obj_):
            node = self.as_node()
            return obj_.name in node.all_inputs_names()
        assert isinstance(obj_ty, WorkTypeClass), f"Cannot handle {obj_ty}"
    return _
