from numba import njit
from numba.core.errors import NumbaError
from numba.core.types import StructRef, unicode_type
from numba.experimental.structref import define_boxing, register, StructRefProxy
from numba.extending import overload

from numbox.core.configurations import jit_options


deleted_node_base_ctor_error = "Intended for inheritance."


class NodeBase(StructRefProxy):
    """A node whose only field is its name: the Python proxy of ``NodeBaseType`` and the base class of Node and Work.

    ``NodeBaseType`` has one field, ``name``. The payloads of :class:`~numbox.core.work.node.Node` and
    :class:`~numbox.core.work.work.Work` nodes begin with that field, so a node of either kind can be cast to
    ``NodeBaseType``, as the inputs of a ``Node`` are cast. Calling ``NodeBase`` raises ``NotImplementedError`` in
    Python and fails to compile in jitted code.
    """
    def __new__(cls, name):
        raise NotImplementedError(deleted_node_base_ctor_error)

    @property
    @njit(**jit_options)
    def name(self):
        """The node's name."""
        return self.name

    def __str__(self):
        return self.name


@register
class NodeBaseTypeClass(StructRef):
    """The numba type class of ``NodeBaseType``, the type of a node whose only field is its name."""
    pass


define_boxing(NodeBaseTypeClass, NodeBase)
node_base_attributes = [
    ("name", unicode_type)
]
NodeBaseType = NodeBaseTypeClass(node_base_attributes)


def _node_base_deleted_ctor():
    raise NumbaError(deleted_node_base_ctor_error)


overload(NodeBase, jit_options=jit_options)(_node_base_deleted_ctor)
