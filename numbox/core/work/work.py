from inspect import getfile, getmodule
from io import StringIO
from numba import njit
from numba.core.types import boolean, DictType, FunctionType, NoneType, Tuple
from numba.core.typing.context import Context
from numba.experimental.structref import define_boxing, new, register
from numba.extending import intrinsic, overload, overload_method
from numba.typed.typedlist import List

from numbox.core.any.erased_type import ErasedType
from numbox.core.configurations import default_jit_options
from numbox.core.work.node import Node, NodeTypeClass, node_attributes
from numbox.utils.lowlevel import (
    extract_struct_member, get_func_p_from_func_struct, get_ll_func_sig,
    _uniformize_tuple_of_structs
)


@register
class WorkTypeClass(NodeTypeClass):
    pass


class Work(Node):
    """
    Structure describing a unit of work.

    Instances of this class can be connected in a graph with other `Work` instances.

    Attributes
    ----------
    name : str
        Name of the structure instance.
    inputs : List[ErasedType]
        Uniform list of `Work.sources`, cast as `ErasedType`.
    data : Any
        Scalar or array data payload contained in (and calculated by) this structure.
    sources : Tuple[Work, ...]
        Heterogeneous tuple of `Work` instances that this `Work` instance depends on.
    derive : FunctionType
        Function of the signature determined by the data types of `sources` and `data`.
    derived : bool
        Flag indicating whether the `data` has already been calculated.

    (`name`, `inputs`) attributes of the `Work` structure payload are
    homogeneously typed across all instances of `Work` and accommodate
    cast-ability to the :obj:`numbox.core.node.Node` base of `NodeType`.
    """
    def __new__(cls, *args, **kws):
        return make_work(*args, **kws)

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
    def derived(self):
        return self.derived

    @njit(**default_jit_options)
    def calculate(self):
        return self.calculate()

    @njit(**default_jit_options)
    def load(self, data):
        return self.load(data)


define_boxing(WorkTypeClass, Work)


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


@overload(Work, strict=False, jit_options=default_jit_options)
def ol_work(name_ty, data_ty, sources_ty, derive_ty):
    """
    Dynamically create `WorkType`, depending on the type of `data`, `sources`, and `derive`.

    Different instances of `Work` accommodate various types of data they might contain,
    various heterogeneous types of other instances of `Work` it might depend on,
    and custom `derive` function objects used to calculate the instance's `data` depending
    on the `data` of its `sources`.

    (`name`, `data`) initialize the :obj:`numbox.core.node.Node` header of the composition.
    """
    assert isinstance(derive_ty, (FunctionType, NoneType)), f"""Either None or Compile Result supported,
not CPUDispatcher, got {derive_ty}, of type {type(derive_ty)}"""
    work_attributes_ = node_attributes + [
        ("data", data_ty),
        ("sources", sources_ty),
        ("derive", derive_ty),
        ("derived", boolean)
    ]
    _verify_signature(data_ty, sources_ty, derive_ty)
    work_type_ = WorkTypeClass(work_attributes_)

    def work_constructor(name_, data_, sources_, derive_):
        uniform_inputs_tuple = _uniformize_tuple_of_structs(sources_, ErasedType)
        uniform_inputs = List.empty_list(ErasedType)
        for s in uniform_inputs_tuple:
            uniform_inputs.append(s)
        w = new(work_type_)
        w.name = name_
        w.inputs = uniform_inputs
        w.data = data_
        w.sources = sources_
        w.derive = derive_
        w.derived = False
        return w
    return work_constructor


def _make_work(*_, **__):
    raise NotImplementedError


@overload(_make_work, strict=False, jit_options=default_jit_options)
def ol_make_work(name_ty, data_ty, sources_ty, derive_ty_):
    def _(name_, data_, sources_, derive_):
        return Work(name_, data_, sources_, derive_)
    return _


@njit(**default_jit_options)
def make_work(name, data, sources=(), derive=None):
    return _make_work(name, data, sources, derive)


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
    for source_ind_ in range(num_sources):
        code_txt.write(_make_source_getter(source_ind_))
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
    source_{source_ind_}.calculate()""")
    code_txt.write("""
    v = _call_derive(work_.derive, work_.sources)
    work_.derived = True
    work_.data = v
""")
    return code_txt.getvalue()


_calculate_registry = {}


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
    if _calculate is None:
        code_txt = _make_calculate_code(num_sources)
        ns = getmodule(_make_work).__dict__
        code = compile(code_txt, getfile(_make_work), mode="exec")
        exec(code, ns)
        _calculate = ns["_calculate_"]
        _calculate_registry[num_sources] = _calculate
    return _calculate


@intrinsic
def _cast_to_work_data(typingctx, work_ty, data_as_any_ty: ErasedType):
    data_ty = work_ty.field_dict["data"]
    sig = data_ty(work_ty, data_as_any_ty)

    def codegen(context, builder, signature, arguments):
        data_as_any = arguments[1]
        ty_ll = context.get_data_type(data_ty)
        _, meminfo_p = context.nrt.get_meminfos(builder, data_as_any_ty, data_as_any)[0]
        payload_p = context.nrt.meminfo_data(builder, meminfo_p)
        x_as_ty_p = builder.bitcast(payload_p, ty_ll.as_pointer())
        val = builder.load(x_as_ty_p)
        context.nrt.incref(builder, data_ty, val)
        return val
    return sig, codegen


def _make_loader_code(num_sources):
    code_txt = StringIO()
    for source_ind_ in range(num_sources):
        code_txt.write(_make_source_getter(source_ind_))
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


def _load_graph(*args):
    raise NotImplementedError("Not to be called from Python scope")


@overload_method(WorkTypeClass, "load", strict=False, jit_options=default_jit_options)
def ol_load(work_ty, data_ty: DictType):
    """ Load `data` into the graph with the root node `work`.
     Data is provided as dictionary mapping node name to `Any` type
     containing erased payload `p` of the `work.data` type. """
    sources_ty = work_ty.field_dict["sources"]
    num_sources = sources_ty.count
    _loader = _loader_registry.get(num_sources, None)
    if _loader is None:
        code_txt = _make_loader_code(num_sources)
        ns = getmodule(_load_graph).__dict__
        code = compile(code_txt, getfile(_load_graph), mode="exec")
        exec(code, ns)
        _loader = ns["_loader_"]
        _loader_registry[num_sources] = _loader
    return _loader
