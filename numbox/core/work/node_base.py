from numba import njit
from numba.core.errors import NumbaError
from numba.core.types import StructRef, unicode_type
from numba.experimental.structref import define_boxing, register, StructRefProxy
from numba.extending import overload

from numbox.core.configurations import default_jit_options


deleted_node_base_ctor_error = "Intended for inheritance."


class NodeBase(StructRefProxy):
    def __new__(cls, name):
        raise NotImplementedError(deleted_node_base_ctor_error)

    @property
    @njit(**default_jit_options)
    def name(self):
        return self.name

    def __str__(self):
        return self.name


@register
class NodeBaseTypeClass(StructRef):
    pass


define_boxing(NodeBaseTypeClass, NodeBase)
node_base_attributes = [
    ("name", unicode_type)
]
NodeBaseType = NodeBaseTypeClass(node_base_attributes)


def _node_base_deleted_ctor():
    raise NumbaError(deleted_node_base_ctor_error)


overload(NodeBase, jit_options=default_jit_options)(_node_base_deleted_ctor)
