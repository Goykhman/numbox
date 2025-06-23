from numba import njit
from numba.core.types import StructRef, unicode_type
from numba.experimental.structref import define_boxing, new, register, StructRefProxy
from numba.extending import overload, overload_method
from numba.typed import List

from numbox.core.configurations import default_jit_options


class NodeBase(StructRefProxy):
    def __new__(cls, name):
        return make_node_base(name)

    @property
    @njit(**default_jit_options)
    def name(self):
        return self.name

    def get_inputs_names(self):
        return list(self._get_inputs_names())

    @njit(**default_jit_options)
    def _get_inputs_names(self):
        return self.get_inputs_names()

    def __str__(self):
        return self.name


@register
class NodeBaseTypeClass(StructRef):
    pass


define_boxing(NodeBaseTypeClass, NodeBase)
node_base_attributes = [
    ("name", unicode_type),
]
NodeBaseType = NodeBaseTypeClass(node_base_attributes)


@overload(NodeBase, strict=False, jit_options=default_jit_options)
def ol_node_base(name_ty):
    def node_base_constructor(name):
        node = new(NodeBaseType)
        return node
    return node_base_constructor


@njit(**default_jit_options)
def make_node_base(name):
    return NodeBase(name)


@overload_method(NodeBaseTypeClass, "get_inputs_names", strict=False, jit_options=default_jit_options)
def ol_get_inputs_names(self_ty):
    def _(self):
        names_ = List.empty_list(unicode_type)
        for s in self.inputs:
            names_.append(s.name)
        return names_
    return _
