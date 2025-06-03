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
    def __new__(cls, name, sources):
        return make_node(name, sources)

    @njit(**default_jit_options)
    def get_source(self, i):
        return self.get_source(i)

    @njit(**default_jit_options)
    def get_sources_names(self):
        return self.get_sources_names()

    @property
    @njit(**default_jit_options)
    def name(self):
        return self.name

    @property
    @njit(**default_jit_options)
    def sources(self):
        return self.sources


@register
class NodeTypeClass(StructRef):
    pass


define_boxing(NodeTypeClass, Node)
node_attributes = [
    ("name", unicode_type),
    ("sources", ListType(ErasedType))
]
NodeType = NodeTypeClass(node_attributes)


@overload(Node, strict=False, jit_options=default_jit_options)
def ol_node(name_ty, sources_ty):
    def node_constructor(name, sources):
        uniform_sources_tuple = _uniformize_tuple_of_structs(sources, ErasedType)
        uniform_sources = List.empty_list(ErasedType)
        for s in uniform_sources_tuple:
            uniform_sources.append(s)
        node = new(NodeType)
        node.name = name
        node.sources = uniform_sources
        return node
    return node_constructor


@overload_method(NodeTypeClass, "get_source", strict=False, jit_options=default_jit_options)
def ol_get_source(self_ty, i_ty):
    def _(self, i):
        num_sources = len(self.sources)
        if i >= num_sources:
            raise NumbaError(f"Requested source {i} while the node has {num_sources} sources")
        return cast(self.sources[i], NodeType)
    return _


@overload_method(NodeTypeClass, "get_sources_names", strict=False, jit_options=default_jit_options)
def ol_get_sources_names(self_ty):
    def _(self):
        return [cast(s, NodeType).name for s in self.sources]
    return _


@njit(**default_jit_options)
def make_node(name, sources=()):
    return Node(name, sources)
