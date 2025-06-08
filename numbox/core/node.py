from numba import njit
from numba.core.cgutils import add_global_variable
from numba.core.errors import NumbaError
from numba.core.types import DictType, ListType, StructRef, unicode_type, void
from numba.typed.typeddict import Dict
from numba.typed.typedlist import List
from numba.experimental.structref import define_boxing, new, register, StructRefProxy
from numba.extending import intrinsic, overload, overload_method

from numbox.core.configurations import default_jit_options
from numbox.core.erased_type import ErasedType
from numbox.utils.lowlevel import _cast, _uniformize_tuple_of_structs


class Node(StructRefProxy):
    def __new__(cls, name, inputs):
        return make_node(name, inputs)

    @njit(**default_jit_options)
    def get_input(self, i):
        return self.get_input(i)

    @njit(**default_jit_options)
    def get_inputs_names(self):
        return self.get_inputs_names()

    @property
    @njit(**default_jit_options)
    def name(self):
        return self.name

    @property
    @njit(**default_jit_options)
    def inputs(self):
        return self.inputs

    def all_inputs_names(self):
        return list(self._all_inputs_names())

    @njit(**default_jit_options)
    def _all_inputs_names(self):
        return self.all_inputs_names()

    def __str__(self):
        return self.name


@register
class NodeTypeClass(StructRef):
    pass


define_boxing(NodeTypeClass, Node)
node_attributes = [
    ("name", unicode_type),
    ("inputs", ListType(ErasedType))
]
NodeType = NodeTypeClass(node_attributes)


@overload(Node, strict=False, jit_options=default_jit_options)
def ol_node(name_ty, inputs_ty):
    def node_constructor(name, inputs):
        uniform_inputs_tuple = _uniformize_tuple_of_structs(inputs, ErasedType)
        uniform_inputs = List.empty_list(ErasedType)
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


@overload_method(NodeTypeClass, "get_inputs_names", strict=False, jit_options=default_jit_options)
def ol_get_inputs_names(self_ty):
    def _(self):
        return [_cast(s, NodeType).name for s in self.inputs]
    return _


@njit
def _all_inputs_names(node_):
    return node_.all_inputs_names()


@njit(**default_jit_options)
def _all_input_names(node, names_):
    for i in range(len(node.inputs)):
        input_ = node.get_input(i)
        names_.append(input_.name)
        _all_input_names(input_, names_)


@overload_method(NodeTypeClass, "all_inputs_names", strict=False, jit_options=default_jit_options)
def ol_all_inputs_names(self_ty):
    def _(self):
        names = List.empty_list(unicode_type)
        for i in range(len(self.inputs)):
            input_node = self.get_input(i)
            name = input_node.name
            names.append(name)
            _all_input_names(input_node, names)
        return names
    return _


@njit(**default_jit_options)
def make_node(name, inputs=()):
    return Node(name, inputs)
