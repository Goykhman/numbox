from numba import njit, typeof
from numba.core.errors import NumbaError
from numba.core.types import ListType, Literal, unicode_type, UnicodeType
from numba.typed.typedlist import List
from numba.experimental.structref import define_boxing, new, register
from numba.extending import overload, overload_method

from numbox.core.configurations import default_jit_options
from numbox.core.work.node_base import NodeBase, NodeBaseType, NodeBaseTypeClass, node_base_attributes
from numbox.utils.lowlevel import cast, _cast, _uniformize_tuple_of_structs


class Node(NodeBase):
    def __new__(cls, name, inputs):
        return make_node(name, inputs)

    @property
    @njit(**default_jit_options)
    def inputs(self):
        return self.inputs

    def get_input(self, i):
        return cast(self._get_input(i), typeof(self))

    @njit(**default_jit_options)
    def _get_input(self, i):
        return self.get_input(i)

    def get_inputs_names(self):
        return list(self._get_inputs_names())

    @njit(**default_jit_options)
    def _get_inputs_names(self):
        return self.get_inputs_names()

    def all_inputs_names(self):
        return list(self._all_inputs_names())

    @njit(**default_jit_options)
    def _all_inputs_names(self):
        return self.all_inputs_names()

    @njit(**default_jit_options)
    def depends_on(self, obj_):
        return self.depends_on(obj_)


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
        return self.inputs[i]
    return _


@overload_method(NodeTypeClass, "get_inputs_names", strict=False, jit_options=default_jit_options)
def ol_get_inputs_names(self_ty):
    def _(self):
        names_ = List.empty_list(unicode_type)
        for s in self.inputs:
            names_.append(s.name)
        return names_
    return _


@njit(**default_jit_options)
def _all_input_names(node_, names_):
    node = _cast(node_, NodeType)
    for i in range(len(node.inputs)):
        input_ = node.get_input(i)
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
def ol_depends_on(self_ty, obj_ty):
    if isinstance(obj_ty, (Literal, UnicodeType,)):
        def _(self, name_):
            return name_ in self.all_inputs_names()
    else:
        assert isinstance(obj_ty, NodeTypeClass), f"Cannot handle {obj_ty}"

        def _(self, obj_):
            return obj_.name in self.all_inputs_names()
    return _


@njit(**default_jit_options)
def make_node(name, inputs=()):
    return Node(name, inputs)
