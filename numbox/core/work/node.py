from numba import njit
from numba.core.errors import NumbaError
from numba.core.types import ListType, Literal, unicode_type, UnicodeType
from numbox.core.work.node_base import NodeBaseType, NodeBase, NodeBaseTypeClass, node_base_attributes
from numba.typed.typedlist import List
from numba.experimental.structref import define_boxing, new, register
from numba.extending import overload, overload_method

from numbox.core.configurations import default_jit_options
from numbox.utils.lowlevel import _cast, _uniformize_tuple_of_structs


class Node(NodeBase):
    def __new__(cls, name, inputs):
        return make_node(name, inputs)

    @njit(**default_jit_options)
    def get_input(self, i):
        return self.get_input(i)

    @property
    @njit(**default_jit_options)
    def name(self):
        return self.name

    @property
    def inputs(self):
        return tuple(self._inputs())

    @njit(**default_jit_options)
    def _inputs(self):
        return self.get_inputs()

    def all_inputs_names(self):
        return list(self._all_inputs_names())

    @njit(**default_jit_options)
    def _all_inputs_names(self):
        return self.all_inputs_names()

    @njit(**default_jit_options)
    def depends_on(self, name_):
        return self.depends_on(name_)


@register
class NodeTypeClass(NodeBaseTypeClass):
    pass


define_boxing(NodeTypeClass, Node)
node_attributes = node_base_attributes + [
    ("inputs", ListType(NodeBaseType))
]
NodeType = NodeTypeClass(node_attributes)


@overload(Node, strict=False, jit_options=default_jit_options)
def ol_node(name_ty, inputs_ty):
    def node_constructor(name, inputs):
        uniform_inputs_tuple = _uniformize_tuple_of_structs(inputs, NodeBaseType)
        uniform_inputs = List.empty_list(NodeBaseType)
        for s in uniform_inputs_tuple:
            uniform_inputs.append(s)
        node = new(NodeType)
        node.name = name
        node.inputs = uniform_inputs
        return node
    return node_constructor


@overload_method(NodeTypeClass, "get_input", strict=False, jit_options=default_jit_options)
def ol_get_input(self_ty, i_ty):
    def _(self, i):
        num_inputs = len(self.inputs)
        if i >= num_inputs:
            raise NumbaError(f"Requested input {i} while the node has {num_inputs} inputs")
        return _cast(self.inputs[i], NodeType)
    return _


@overload_method(NodeTypeClass, "get_inputs", strict=False, jit_options=default_jit_options)
def ol_get_inputs(self_ty):
    def _(self):
        inputs_list = List.empty_list(NodeType)
        for s in self.inputs:
            inputs_list.append(_cast(s, NodeType))
        return inputs_list
    return _


@njit
def _all_inputs_names(node_):
    return node_.all_inputs_names()


@njit(**default_jit_options)
def _all_input_names(node, names_):
    for input_ in _cast(node, NodeType).get_inputs():
        name_ = input_.name
        if name_ not in names_:
            names_.append(name_)
        _all_input_names(input_, names_)


@overload_method(NodeTypeClass, "all_inputs_names", strict=False, jit_options=default_jit_options)
def ol_all_inputs_names(self_ty):
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


@overload_method(NodeTypeClass, "depends_on", strict=False, jit_options=default_jit_options)
def ol_depends_on(self_ty, name_ty):
    if isinstance(name_ty, (Literal, UnicodeType,)):
        def _(self, name_):
            return name_ in self.all_inputs_names()
    else:
        assert isinstance(name_ty, NodeTypeClass), f"Cannot handle {name_ty}"

        def _(self, name_):
            return name_.name in self.all_inputs_names()
    return _


@njit(**default_jit_options)
def make_node(name, inputs=()):
    return Node(name, inputs)
