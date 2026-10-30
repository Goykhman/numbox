"""Collect the values of a graph's nodes. :func:`make_sheaf_dict` makes a sheaf, a typed dictionary of ``Any`` values
keyed by node name, for ``Work.combine`` to fill. :func:`load_dict_into_array` copies a sheaf into the first row of a
NumPy structured array, whose dtype :func:`make_requested_dtype` makes.
"""
import numpy

from io import StringIO
from inspect import getfile, getmodule
from numba import njit
from numba.core.types import Record, unicode_type
from numba.extending import overload
from numba.typed.typeddict import Dict
from typing import Collection

from numbox.core.any.any_type import AnyType, make_any
from numbox.core.configurations import jit_options
from numbox.core.work.node import Node
from numbox.core.work.work import Work


RequestedTy = Node | str | Work


def add_one(node, sheaf):
    """Add an entry for ``node``, a node or a node name, to ``sheaf``, with an ``Any`` value that holds ``None``.

    If the name of ``node`` is not a string, an assertion fails.
    """
    node_name = getattr(node, "name", node)
    assert isinstance(node_name, str), f"Expected node name as string, got {node_name} of type {type(node_name)}"
    sheaf[node_name] = make_any(None)


def make_sheaf_dict(requested: RequestedTy | Collection[RequestedTy]):
    """Return a sheaf, a typed dictionary from the name of each requested node to an ``Any`` value that holds ``None``.

    ``requested`` is a node, a node name, or a collection of them. ``Work.combine`` replaces each value with the
    ``data`` of the node of that name.
    """
    sheaf = Dict.empty(key_type=unicode_type, value_type=AnyType)
    if isinstance(requested, RequestedTy):
        add_one(requested, sheaf)
    else:
        for node in requested:
            add_one(node, sheaf)
    return sheaf


# This stub has an implementation for jitted code only, the overload below; a call from Python raises
# NotImplementedError.
def _load_dict_into_array(*args):
    raise NotImplementedError


# The generated code has one statement per field of the record, because jitted code can access a record field only
# through a name that is known at compile time.
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


@overload(_load_dict_into_array, strict=False, jit_options=jit_options)
def ol_load_array_row_into_dict(array_ty, sheaf_ty):
    """Implement ``load_dict_into_array`` in jitted code: set each field of the array's first row from the sheaf.

    Each field's name must be a key of the sheaf, and the value stored under that key must hold the field's type,
    because ``get_as`` reads the value without a conversion.
    """
    record_ty: Record = array_ty.dtype
    record_fields = record_ty.fields
    code_txt = _make_load_to_array_code(record_fields)
    ns = {**getmodule(_load_dict_into_array).__dict__, **{"record_fields": record_fields}}
    code = compile(code_txt, getfile(_load_dict_into_array), mode="exec")
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    _load_to_array = ns["_load_to_array_"]
    return _load_to_array


@njit(**jit_options)
def load_dict_into_array(array, sheaf):
    """Copy the values of ``sheaf`` into the first row of the structured array ``array``, one field per node name.

    :func:`make_requested_dtype` makes a matching dtype, and each value must hold the type of its field.
    """
    return _load_dict_into_array(array, sheaf)


def make_requested_dtype(requested: dict):
    """Return an aligned NumPy structured dtype with one field for each requested node, named after the node.

    ``requested`` maps each node, or node name, to the dtype of its field.
    """
    return numpy.dtype([(getattr(node, "name", node), ty) for node, ty in requested.items()], align=True)
