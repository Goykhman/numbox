from numba import njit
from numba.core.errors import NumbaError
from numba.core.types import ListType, StructRef, unicode_type
from numba.typed.typedlist import List
from numba.experimental.structref import define_boxing, new, register, StructRefProxy
from numba.extending import overload, overload_method

from numbox.core.configurations import default_jit_options
from numbox.core.erased_type import ErasedType
from numbox.utils.lowlevel import cast, _uniformize_tuple_of_structs


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
        return cast(self.inputs[i], NodeType)
    return _


@overload_method(NodeTypeClass, "get_inputs_names", strict=False, jit_options=default_jit_options)
def ol_get_inputs_names(self_ty):
    def _(self):
        return [cast(s, NodeType).name for s in self.inputs]
    return _


@njit(**default_jit_options)
def make_node(name, inputs=()):
    return Node(name, inputs)
