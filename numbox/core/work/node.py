from numba import njit, typeof
from numba.core.errors import NumbaError
from numba.core.types import ListType, Literal, unicode_type, UnicodeType
from numba.typed.typedlist import List
from numba.experimental.structref import define_boxing, new, register
from numba.extending import overload, overload_method

from numbox.core.configurations import jit_options
from numbox.core.work.node_base import NodeBase, NodeBaseType, NodeBaseTypeClass, node_base_attributes
from numbox.utils.lowlevel import cast, _cast, _uniformize_tuple_of_structs


class Node(NodeBase):
    """A node of a directed acyclic graph that holds its name and the nodes it depends on, its inputs.

    :func:`make_node` creates a node, from Python or from jitted code, and so does ``Node(name, inputs)``. The inputs
    are stored in a typed list of ``NodeBaseType``. ``all_inputs_names``, ``all_end_nodes`` and ``depends_on`` walk the
    graph by reading each input as a ``Node``, and ``get_input`` returns an input as a ``Node``. For an input of another
    type, such as a :class:`~numbox.core.work.work.Work` node, these methods read its payload with the wrong layout,
    and the process can crash. For a graph of ``Work`` nodes, use the ``Node`` representations that
    :meth:`Work.as_node <numbox.core.work.work.Work.as_node>` creates.
    """
    def __new__(cls, name, inputs):
        return make_node(name, inputs)

    @property
    @njit(**jit_options)
    def inputs(self):
        """The inputs, as a typed list of :class:`~numbox.core.work.node_base.NodeBase` proxies."""
        return self.inputs

    def get_input(self, i):
        """Return the input at index ``i`` as a ``Node``.

        A negative index, or one past the last input, raises ``NumbaError``.
        """
        # _get_input returns the input as NodeBaseType, and the cast reads it as a Node, the type of this node.
        return cast(self._get_input(i), typeof(self))

    @njit(**jit_options)
    def _get_input(self, i):
        return self.get_input(i)

    def get_inputs_names(self):
        """Return the names of the inputs, as a list."""
        return list(self._get_inputs_names())

    @njit(**jit_options)
    def _get_inputs_names(self):
        return self.get_inputs_names()

    def all_inputs_names(self):
        """Return the names of all the nodes this node depends on, each once, in depth-first order."""
        return list(self._all_inputs_names())

    @njit(**jit_options)
    def _all_inputs_names(self):
        return self.all_inputs_names()

    def all_end_nodes(self):
        """Return the names of the nodes without inputs that this node depends on, each once."""
        return list(self._all_end_nodes())

    @njit(**jit_options)
    def _all_end_nodes(self):
        return self.all_end_nodes()

    @njit(**jit_options)
    def depends_on(self, obj_):
        """Return whether this node depends on ``obj_``, a node name or a ``Node``, directly or through other nodes."""
        return self.depends_on(obj_)


@register
class NodeTypeClass(NodeBaseTypeClass):
    """The numba type class of ``NodeType``, the type of every ``Node``."""
    pass


define_boxing(NodeTypeClass, Node)
node_attributes = node_base_attributes + [
    ("inputs", ListType(NodeBaseType))
]
NodeType = NodeTypeClass(node_attributes)


@overload(Node, strict=False, jit_options=jit_options)
def ol_node(name_ty, inputs_ty):
    """Implement ``Node(name, inputs)`` in jitted code, casting the tuple of nodes ``inputs`` to ``NodeBaseType``.

    Each element of ``inputs`` must be a struct node. For an element of another type, an assertion fails when numba
    compiles the call.
    """
    def node_constructor(name, inputs):
        """Create the node and copy its inputs into a typed list."""
        uniform_inputs_tuple = _uniformize_tuple_of_structs(inputs, NodeBaseType)
        uniform_inputs = List.empty_list(NodeBaseType)
        for s in uniform_inputs_tuple:
            uniform_inputs.append(s)
        node = new(NodeType)
        node.name = name
        node.inputs = uniform_inputs
        return node
    return node_constructor


@overload_method(NodeTypeClass, "get_input", strict=False, jit_options=jit_options)
def ol_get_input(self_ty, i_ty):
    """Implement ``Node.get_input(i)`` in jitted code: return the input at index ``i`` as ``NodeBaseType``.

    A negative index, or one past the last input, raises ``NumbaError``.
    """
    def _(self, i):
        num_inputs = len(self.inputs)
        if i < 0 or i >= num_inputs:
            raise NumbaError(f"Requested input {i} while the node has {num_inputs} inputs")
        return self.inputs[i]
    return _


@overload_method(NodeTypeClass, "get_inputs_names", strict=False, jit_options=jit_options)
def ol_get_inputs_names(self_ty):
    """Implement ``Node.get_inputs_names()`` in jitted code, returning the names as a typed list."""
    def _(self):
        names_ = List.empty_list(unicode_type)
        for s in self.inputs:
            names_.append(s.name)
        return names_
    return _


@njit(**jit_options)
def _all_input_names(node_, names_):
    node = _cast(node_, NodeType)
    for i in range(len(node.inputs)):
        input_ = node.get_input(i)
        name_ = input_.name
        if name_ not in names_:
            names_.append(name_)
        _all_input_names(input_, names_)


@overload_method(NodeTypeClass, "all_inputs_names", strict=False, jit_options=jit_options)
def ol_all_inputs_names(self_ty):
    """Implement ``Node.all_inputs_names()`` in jitted code, returning the names as a typed list."""
    def _(self):
        names = List.empty_list(unicode_type)
        for i in range(len(self.inputs)):
            input_node = self.get_input(i)
            name_ = input_node.name
            if name_ not in names:
                names.append(name_)
            _all_input_names(input_node, names)
        return names
    return _


@njit(**jit_options)
def _all_end_nodes(node_, names_):
    node = _cast(node_, NodeType)
    for i in range(len(node.inputs)):
        input_ = node.get_input(i)
        name_ = input_.name
        if name_ not in names_ and len(_cast(input_, NodeType).inputs) == 0:
            names_.append(name_)
        _all_end_nodes(input_, names_)


@overload_method(NodeTypeClass, "all_end_nodes", strict=False, jit_options=jit_options)
def ol_all_end_nodes(self_ty):
    """Implement ``Node.all_end_nodes()`` in jitted code, returning the names as a typed list."""
    def _(self):
        names = List.empty_list(unicode_type)
        for i in range(len(self.inputs)):
            input_ = self.get_input(i)
            name_ = input_.name
            if name_ not in names and len(_cast(input_, NodeType).inputs) == 0:
                names.append(name_)
            _all_end_nodes(input_, names)
        return names
    return _


@overload_method(NodeTypeClass, "depends_on", strict=False, jit_options=jit_options)
def ol_depends_on(self_ty, obj_ty):
    """Implement ``Node.depends_on(obj)`` in jitted code, for a name or a ``Node``.

    For an argument of another type, an assertion fails when numba compiles the call.
    """
    if isinstance(obj_ty, (Literal, UnicodeType,)):
        def _(self, name_):
            return name_ in self.all_inputs_names()
    else:
        assert isinstance(obj_ty, NodeTypeClass), f"Cannot handle {obj_ty}"

        def _(self, obj_):
            return obj_.name in self.all_inputs_names()
    return _


@njit(**jit_options)
def make_node(name, inputs=()):
    """Create a ``Node`` named ``name`` whose inputs are the nodes ``inputs``, from Python or from jitted code.

    ``inputs`` is a tuple of ``Node`` nodes. The :class:`Node` class explains why a node of another type must not be an
    input.
    """
    return Node(name, inputs)
