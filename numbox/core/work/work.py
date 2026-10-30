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

from numba.core import cgutils

from numbox.core.any.erased_type import ErasedType
from numbox.core.configurations import jit_options
from numbox.core.work.lowlevel_work_utils import ll_make_work, WorkTypeClass
from numbox.core.work.node import NodeType
from numbox.core.work.node_base import NodeBase, NodeBaseType
from numbox.utils.derive_wap import DeriveFunctionType, rewrap_derive
from numbox.utils.lowlevel import (
    extract_struct_member, _cast, _get_func_tuple, is_not_null, get_func_p_from_func_struct, get_ll_func_sig
)


def _file_anchor():
    raise NotImplementedError


deleted_work_ctor_error = "Use `make_work` instead"


class Work(NodeBase):
    """
    A node of a jitted calculation graph: a value, the nodes it is computed from, and the function that computes it.

    Nodes are created with :func:`make_work`, with ``ll_make_work`` in jitted code, or by
    :func:`~numbox.core.work.builder.make_graph`. Calling ``Work`` raises ``NotImplementedError`` in Python and fails
    to compile in jitted code. The attributes below are the fields that jitted code reads; the properties of this class
    read most of them from Python.

    Attributes
    ----------
    name : str
        The node's name.
    inputs : UniTuple[NodeBaseType]
        The sources, cast to ``NodeBaseType``.
    data : Any
        The node's value. Its numba type is set when the node is created, and ``calculate`` stores the derive's result
        in it.
    sources : Tuple[Work, ...]
        The nodes whose ``data`` the derive reads, in a tuple that can hold ``Work`` nodes of different types.
    derive : FunctionType
        The function that computes ``data`` from the ``data`` of the sources, or ``None`` for a node without one. Its
        signature is the type of ``data`` over the types of the sources' ``data``.
        :meth:`~numbox.core.work.work.Work.calculate` describes what happens when it raises.
    derived : int8
        1 once ``data`` has been calculated, otherwise 0.
    node : NodeType
        The node's :class:`~numbox.core.work.node.Node` representation, which ``as_node`` creates on its first call;
        null until then.

    The ``name`` field comes first in every ``Work`` payload and has the same type in all of them, so a ``Work`` node
    can be cast to ``NodeBaseType``, the type of :obj:`numbox.core.work.node_base.NodeBase`.
    """
    def __new__(cls, *args, **kws):
        raise NotImplementedError(deleted_work_ctor_error)

    # Each jitted method compiles with numbox's jit options, so numba caches it by default. numba compiles and caches
    # it once for each Work type it is called on. A node whose data is an @njit function has a type that contains a
    # dispatcher. In every new process, numba finds no entry for such a type in the cache index and adds one. The cache
    # files of these methods, and of the overloads below, therefore grow by one entry per process. No entry is ever
    # stale.
    @property
    @njit(**jit_options)
    def inputs(self):
        """The sources cast to ``NodeBaseType``, as a tuple of :class:`~numbox.core.work.node_base.NodeBase` proxies."""
        return self.inputs

    @property
    @njit(**jit_options)
    def data(self):
        """The node's value. An array is returned without a copy, so writing to it changes the node's value."""
        return self.data

    @property
    @njit(**jit_options)
    def sources(self):
        """The nodes this node is computed from, as a tuple of ``Work`` nodes."""
        return self.sources

    @property
    @njit(**jit_options)
    def derive(self):
        """The derive's addresses, as a tuple of three integers.

        The integers are the derive's C entry point, the address of its Python object and numba's entry point. The
        third is 0 when the derive has no numba entry point. In that case the derive's exception does not propagate
        out of ``calculate``. Reading this property on a node without a derive raises ``TypeError``.
        """
        return _get_func_tuple(self.derive)

    @property
    @njit(**jit_options)
    def derived(self):
        """1 once the node's ``data`` has been calculated, otherwise 0."""
        return self.derived

    @njit(**jit_options)
    def as_node(self):
        """Return the node's :class:`~numbox.core.work.node.Node` representation, creating it on the first call.

        The first call also creates a representation for each source that lacks one, and these representations become
        the inputs of the new ``Node``. Later calls return the stored ``Node``.
        """
        return self.as_node()

    @njit(**jit_options)
    def calculate(self):
        """Calculate the node's ``data``, after calculating each source that is not yet calculated.

        The call returns at once for a node that is already calculated, and only marks a node without a derive as
        calculated. Otherwise ``calculate`` calls the derive on the ``data`` of the sources, stores the result in
        ``data`` and sets ``derived``. If the derive raises, the exception propagates with its type and message,
        ``data`` keeps its value and ``derived`` stays unset, so a later call runs the derive again. numba's error path
        keeps the references it took. A node whose derive has failed therefore stays in memory, with everything it
        references, until the process ends. Another failure of the same node adds no further reference.

        In three cases the derive's exception does not propagate. The first is a derive passed from Python into a
        jitted parameter declared as numba's plain ``FunctionType``. The second is a ``CompileResultWAP`` built with
        numba directly that reaches jitted code, where ``make_work`` cannot upgrade it. The third is a ``@cfunc``. In
        each case ``calculate`` stores a zero-filled result and sets ``derived``. The ``.as_func`` value of a
        ``@proxy`` function is not such a ``CompileResultWAP``, and its exception propagates. The documentation of
        ``numbox.core.work`` describes these cases under "When the exception is lost".
        """
        return self.calculate()

    @njit(**jit_options)
    def load(self, data):
        """Set the nodes' ``data`` from the dictionary ``data``, in this node's graph, and return whether any was set.

        The graph is this node and the nodes it depends on. ``data`` is a typed dictionary from node names to
        :class:`~numbox.core.any.any_type.Any` values. ``load`` reads each value as the node's data type, with no
        conversion, so a value must have that type. It sets ``derived`` back to 0 on each node it loads and on each
        node between that node and this one, so the next ``calculate`` computes them again. A loaded node that has a
        derive is computed again as well, and its loaded value is replaced.
        """
        return self.load(data)

    @njit(**jit_options)
    def combine(self, data):
        """Copy the ``data`` of the named nodes of this node's graph into the dictionary ``data``, under their names.

        ``data`` is a typed dictionary from node names to :class:`~numbox.core.any.any_type.Any` values, such as the
        sheaf that :func:`~numbox.core.work.combine_utils.make_sheaf_dict` returns. The walk covers this node and the
        nodes it depends on, depth first, and stops once it has found a node for every name. A name that no node of the
        graph has keeps its value.
        """
        return self.combine(data)

    @njit(**jit_options)
    def get_input(self, i):
        """Return the source at index ``i`` as its :class:`~numbox.core.work.node.Node` representation.

        The call creates the representations of the sources that lack one. A negative index, or one past the last
        source, raises ``NumbaError``.
        """
        return self.get_input(i)

    def get_inputs_names(self):
        """Return the names of the sources, as a list."""
        return list(self._get_inputs_names())

    @njit(**jit_options)
    def _get_inputs_names(self):
        return self.get_inputs_names()

    @njit(**jit_options)
    def make_inputs_vector(self):
        """Return the :class:`~numbox.core.work.node.Node` representations of the sources, as a typed list.

        The list holds ``NodeBaseType`` values. If ``as_node`` has created this node's own representation, the list is
        that representation's list of inputs. Otherwise it is a new list, and the representations of the sources that
        lack one are created.
        """
        return self.make_inputs_vector()

    def all_inputs_names(self):
        """Return the names of all the nodes this node depends on, each once, in depth-first order.

        The call reads the graph through ``as_node``, which creates the node's ``Node`` representation if needed.
        """
        return list(self._all_inputs_names())

    @njit(**jit_options)
    def _all_inputs_names(self):
        return self.all_inputs_names()

    def all_end_nodes(self):
        """Return the names of the nodes without sources that this node depends on, each once.

        The call reads the graph through ``as_node``, which creates the node's ``Node`` representation if needed.
        """
        return list(self._all_end_nodes())

    @njit(**jit_options)
    def _all_end_nodes(self):
        return self.all_end_nodes()

    @njit(**jit_options)
    def depends_on(self, obj_):
        """Return whether this node depends on ``obj_``, a node name or a ``Work`` node, directly or through others."""
        return self.depends_on(obj_)


define_boxing(WorkTypeClass, Work)


def _deleted_work_ctor(*args, **kwargs):
    raise NumbaError(deleted_work_ctor_error)


overload(Work, jit_options=jit_options)(_deleted_work_ctor)


@njit(**jit_options)
def _make_work_jit(name, data, sources=(), derive=None):
    return ll_make_work(name, data, sources, derive)


def make_work(name, data, sources=(), derive=None):
    """Create a ``Work`` node, from Python or from jitted code.

    ``data`` is the node's initial value and sets its numba type. ``sources`` is a tuple of ``Work`` nodes, and
    ``derive`` computes the node's value from their ``data``. ``derive`` is ``None`` or a function value compiled for
    one signature, the type of ``data`` over the types of the sources' ``data``, such as a
    :func:`~numbox.utils.highlevel.cres` function. A ``derive`` of another signature raises ``ValueError``, and for a
    lazily compiled ``@njit`` function an assertion fails.

    Called from Python, ``make_work`` upgrades a ``CompileResultWAP`` built with numba directly to a
    :class:`~numbox.utils.derive_wap.DeriveWAP` with :func:`~numbox.utils.derive_wap.rewrap_derive`, so that an
    exception the derive raises propagates out of ``calculate``. The node then holds the upgraded wrapper. Called from
    jitted code, ``make_work`` uses ``derive`` unchanged. The value's numba type is already fixed in jitted code, and
    the upgrade depends on the value's Python class, which jitted code cannot access.

    ``make_work.py_func`` is the Python function from which numba compiles the constructor that ``make_work`` calls.
    Calling it from Python raises ``NotImplementedError``, because that function calls the ``ll_make_work`` intrinsic.
    Unlike ``make_work``, that function does not upgrade ``derive``.
    """
    # rewrap_derive creates a DeriveWAP, stores it on the caller's CompileResultWAP and returns it, so the node holds
    # that wrapper. When a jitted function returns the node's derive, Python receives the DeriveWAP, not the caller's
    # object.
    return _make_work_jit(name, data, sources, rewrap_derive(derive))


@overload(make_work, strict=False, jit_options=jit_options)
def ol_make_work(name, data, sources=(), derive=None):
    """Implement ``make_work`` in jitted code with ``ll_make_work``, using ``derive`` unchanged.

    A ``CompileResultWAP`` built with numba directly is not upgraded here, so its exception does not propagate.
    """
    def _(name, data, sources=(), derive=None):
        return ll_make_work(name, data, sources, derive)
    return _


# Code that treats make_work as a dispatcher reads its py_func attribute, so make_work.py_func is set to the py_func
# of the jitted constructor.
make_work.py_func = _make_work_jit.py_func


@intrinsic
def _call_derive(typingctx: Context, derive_ty: FunctionType, sources_ty: Tuple):
    """Call a node's derive on the ``data`` of its sources and return the result.

    An exception that the derive raises propagates to the caller only if the derive's value holds numba's entry point,
    its ``jit_addr``. A ``DeriveFunctionType`` value always holds it. A value of the plain ``FunctionType`` holds it
    only if numba populated it, as numba does when it unboxes an ``@njit`` function passed as an argument of that
    type. A value without it is called through its C entry point, which discards the exception and returns a
    zero-filled result.
    """
    fsig = derive_ty.signature

    def codegen(context, builder, signature, arguments):
        """Emit the call of the derive on the sources' ``data`` and return its result."""
        derive_struct, sources = arguments
        derive_args = []
        for source_ind, source_ty in enumerate(sources_ty):
            source = builder.extract_value(sources, source_ind)
            data = extract_struct_member(context, builder, source_ty, source, "data")
            derive_args.append(data)

        def emit_propagating_call(jit_addr):
            """Call the derive through numba's entry point ``jit_addr``, which propagates an exception."""
            func_ty = context.call_conv.get_function_type(fsig.return_type, fsig.args)
            derive_p = builder.bitcast(jit_addr, func_ty.as_pointer())
            status, res = context.call_conv.call_function(
                builder, derive_p, fsig.return_type, fsig.args, derive_args
            )
            with cgutils.if_unlikely(builder, status.is_error):
                context.call_conv.return_status_propagate(builder, status)
            return res

        def emit_c_call():
            """Call the derive through its C entry point, which discards an exception, and return the result."""
            derive_p_raw = get_func_p_from_func_struct(builder, derive_struct)
            derive_ty_ll = get_ll_func_sig(context, derive_ty)
            derive_p = builder.bitcast(derive_p_raw, derive_ty_ll.as_pointer())
            return builder.call(derive_p, derive_args)

        jit_addr = cgutils.create_struct_proxy(derive_ty)(
            context, builder, value=derive_struct).jit_addr
        # numbox fills jit_addr whenever it unboxes or lowers a DeriveFunctionType value, so the type alone selects
        # the propagating call, with no test at run time.
        if isinstance(derive_ty, DeriveFunctionType):
            return emit_propagating_call(jit_addr)

        # A plain FunctionType value holds jit_addr only if numba populated it, so the slot is tested at run time.
        res_slot = cgutils.alloca_once(builder, context.get_value_type(fsig.return_type))
        with builder.if_else(cgutils.is_null(builder, jit_addr), likely=False) as (null, populated):
            with null:
                # Only the C entry point remains. This branch is taken in the three cases in which the derive's
                # exception does not propagate. They are a derive passed from Python into a parameter declared as the
                # plain FunctionType, a CompileResultWAP built with numba directly that reached jitted code, and a
                # @cfunc. The C entry point discards the exception (numba issue 8246,
                # https://github.com/numba/numba/issues/8246), so calculate stores the zero-filled result.
                builder.store(emit_c_call(), res_slot)
            with populated:
                builder.store(emit_propagating_call(jit_addr), res_slot)
        return builder.load(res_slot)
    sig = fsig.return_type(derive_ty, sources_ty)
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


# The generated _calculate_ sets derived only after _call_derive returns. A derive whose exception propagates thus
# leaves derived unset and data unchanged, and a later calculate runs the derive again.
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
    """Define ``_get_source_{i}`` in ``ns_`` for each ``i`` below ``num_sources_`` that ``ns_`` lacks.

    ``_get_source_{i}`` is an intrinsic that returns element ``i`` of a node's ``sources`` tuple. ``ns_`` is the module
    namespace or a copy of it that ``_codegen_method`` extends, so the check reads ``ns_`` itself.
    """
    for source_i in range(num_sources_):
        if f"_get_source_{source_i}" in ns_:
            continue
        source_getter_code_txt = _make_source_getter(source_i)
        source_getter_code = compile(source_getter_code_txt, getfile(_file_anchor), mode="exec")
        exec(source_getter_code, ns_)  # nosec B102 - JIT codegen of internal source


def _codegen_method(num_sources, registry, make_code, exec_name, ns_extras=None):
    """Return the generated implementation of a ``Work`` method for a node with ``num_sources`` sources.

    The first call for a number of sources renders the code with ``make_code`` and runs it in the module namespace, or
    in a copy of that namespace extended by ``ns_extras``. The code must define ``exec_name``, and ``registry`` stores
    that function for later calls.
    """
    impl = registry.get(num_sources, None)
    if impl is not None:
        return impl
    ns = getmodule(_file_anchor).__dict__
    if ns_extras is not None:
        ns = {**ns, **ns_extras}
    ensure_presence_of_source_getters_in_ns(num_sources, ns)
    code = compile(make_code(num_sources), getfile(_file_anchor), mode="exec")
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    impl = ns[exec_name]
    registry[num_sources] = impl
    return impl


@overload_method(WorkTypeClass, "calculate", strict=False, jit_options=jit_options)
def ol_calculate(self_ty):
    """Implement ``Work.calculate()`` in jitted code: calculate the sources not yet calculated, then the node itself.

    A jitted caller can catch the derive's exception only with a bare ``except:`` or with ``except Exception:``.
    """
    derive_ty = self_ty.field_dict["derive"]
    if isinstance(derive_ty, NoneType):
        def _(self):
            self.derived = True
        return _

    num_sources = self_ty.field_dict["sources"].count
    return _codegen_method(num_sources, _calculate_registry, _make_calculate_code, "_calculate_")


# This intrinsic reads the payload of an Any value as the node's data type, with no check and no conversion.
@intrinsic
def _cast_to_work_data(typingctx, work_ty, data_as_erased_ty: ErasedType):
    data_ty = work_ty.field_dict["data"]
    sig = data_ty(work_ty, data_as_erased_ty)

    def codegen(context, builder, signature, arguments):
        """Emit a load of the erased payload as the node's data type, with a new reference."""
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


@overload_method(WorkTypeClass, "load", strict=False, jit_options=jit_options)
def ol_load(work_ty, data_ty: DictType):
    """Implement ``Work.load(data)`` in jitted code: set the ``data`` of the named nodes from ``Any`` values.

    Each value's erased payload ``p`` must hold the node's data type, because ``load`` reads it without a conversion.
    """
    num_sources = work_ty.field_dict["sources"].count
    return _codegen_method(num_sources, _loader_registry, _make_loader_code, "_loader_")


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


@overload_method(WorkTypeClass, "combine", strict=False, jit_options=jit_options)
def ol_combine(work_ty, data_ty: DictType, harvested_ty=NoneType):
    """Implement ``Work.combine(data)`` in jitted code: store the ``data`` of the named nodes in ``Any`` values.

    ``combine`` resets the erased payload ``p`` of each value to the node's ``data``. The optional ``harvested``
    dictionary records the names found so far, and ``combine`` passes it to the ``combine`` call of each source.
    """
    num_sources = work_ty.field_dict["sources"].count
    return _codegen_method(num_sources, _combine_registry, _make_combine_code, "_combine_",
                           {"boolean": boolean, "Dict": Dict})


@overload_method(WorkTypeClass, "get_input", strict=False, jit_options=jit_options)
def ol_get_input(self_ty, i_ty):
    """Implement ``Work.get_input(i)`` in jitted code: return the source at index ``i`` as a ``Node``.

    A negative index, or one past the last source, raises ``NumbaError``.
    """
    def _(self, i):
        num_inputs = len(self.inputs)
        if i < 0 or i >= num_inputs:
            raise NumbaError(f"Requested input {i} while the node has {num_inputs} inputs")
        inputs_vector = self.make_inputs_vector()
        return _cast(inputs_vector[i], NodeType)
    return _


@overload_method(WorkTypeClass, "get_inputs_names", strict=False, jit_options=jit_options)
def ol_get_inputs_names(self_ty):
    """Implement ``Work.get_inputs_names()`` in jitted code, returning the names as a typed list."""
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


@overload_method(WorkTypeClass, "make_inputs_vector", strict=False, jit_options=jit_options)
def ol_make_inputs_vector(self_ty):
    """Implement ``Work.make_inputs_vector()`` in jitted code, with code generated for the node's number of sources."""
    num_sources = self_ty.field_dict["sources"].count
    if num_sources == 0:
        def _(self):
            inputs_vector = List.empty_list(NodeBaseType)
            return inputs_vector
        return _
    return _codegen_method(num_sources, _inputs_vector_registry, _make_inputs_vector_code, "_inputs_vector_",
                           {"new": new})


@overload_method(WorkTypeClass, "as_node", strict=False, jit_options=jit_options)
def ol_as_node(self_ty):
    """Implement ``Work.as_node()`` in jitted code, creating the ``Node`` representation on the first call."""
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


@overload_method(WorkTypeClass, "all_inputs_names", strict=False, jit_options=jit_options)
def ol_all_inputs_names(self_ty):
    """Implement ``Work.all_inputs_names()`` in jitted code, through the node's ``Node`` representation."""
    def _(self):
        node = self.as_node()
        return node.all_inputs_names()
    return _


@overload_method(WorkTypeClass, "all_end_nodes", strict=False, jit_options=jit_options)
def ol_all_end_nodes(self_ty):
    """Implement ``Work.all_end_nodes()`` in jitted code, through the node's ``Node`` representation."""
    def _(self):
        node = self.as_node()
        return node.all_end_nodes()
    return _


@overload_method(WorkTypeClass, "depends_on", strict=False, jit_options=jit_options)
def ol_depends_on(self_ty, obj_ty):
    """Implement ``Work.depends_on(obj)`` in jitted code, for a name or a ``Work`` node.

    For an argument of another type, an assertion fails when numba compiles the call.
    """
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
