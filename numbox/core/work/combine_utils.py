import numpy

from io import StringIO
from inspect import getfile, getmodule
from numba import njit
from numba.core.types import Record, unicode_type
from numba.extending import overload
from numba.typed.typeddict import Dict
from typing import Collection

from numbox.core.any.any_type import AnyType, make_any
from numbox.core.configurations import default_jit_options
from numbox.core.work.node import Node
from numbox.core.work.work import Work


RequestedTy = Node | str | Work


def add_one(node, sheaf):
    node_name = getattr(node, "name", node)
    assert isinstance(node_name, str), f"Expected node name as string, got {node_name} of type {type(node_name)}"
    sheaf[node_name] = make_any(None)


def make_sheaf_dict(requested: RequestedTy | Collection[RequestedTy]):
    sheaf = Dict.empty(key_type=unicode_type, value_type=AnyType)
    if isinstance(requested, RequestedTy):
        add_one(requested, sheaf)
    else:
        for node in requested:
            add_one(node, sheaf)
    return sheaf


def _load_dict_into_array(*args):
    raise NotImplementedError


def _make_load_to_array_code(record_fields):
    code_txt = StringIO()
    fields_names = list(record_fields.keys())
    for field_ind, field_name in enumerate(fields_names):
        code_txt.write(f"""
{field_name}_ty = record_fields["{field_name}"].type
""")
    code_txt.write("""
def _load_to_array_(array_, sheaf_):""")
    for field_ind, field_name in enumerate(fields_names):
        code_txt.write(f"""
    array_[0].{field_name} = sheaf_["{field_name}"].get_as({field_name}_ty)""")
    code_txt.write("""
    return""")
    return code_txt.getvalue()


@overload(_load_dict_into_array, strict=False, jit_options=default_jit_options)
def ol_load_array_row_into_dict(array_ty, sheaf_ty):
    record_ty: Record = array_ty.dtype
    record_fields = record_ty.fields
    code_txt = _make_load_to_array_code(record_fields)
    ns = {**getmodule(_load_dict_into_array).__dict__, **{"record_fields": record_fields}}
    code = compile(code_txt, getfile(_load_dict_into_array), mode="exec")
    exec(code, ns)
    _load_to_array = ns["_load_to_array_"]
    return _load_to_array


@njit(**default_jit_options)
def load_dict_into_array(array, sheaf):
    return _load_dict_into_array(array, sheaf)


def make_requested_dtype(requested: dict):
    return numpy.dtype([(getattr(node, "name", node), ty) for node, ty in requested.items()], align=True)
