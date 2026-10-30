"""Load the fields of a NumPy structured array into a graph. Called from jitted code,
:func:`load_array_row_into_dict` stores the fields of one row in a typed dictionary of ``Any`` values keyed by field
name, the dictionary that ``Work.load`` reads. :func:`np_struct_member_type` gives the numba type of a field.
"""
import numpy

from inspect import getfile, getmodule
from io import StringIO
from numba import from_dtype
from numba.core.types import Record, unicode_type
from numba.extending import overload

from numbox.core.configurations import jit_options
from numbox.core.any.any_type import AnyType, _make_any


def load_array_row_into_dict(*args):
    """Store each field of one row of a structured array, under the field's name, in a dictionary of ``Any`` values.

    Call it from jitted code as ``load_array_row_into_dict(array, row_index, loader_dict)``. ``loader_dict`` is a typed
    dictionary from strings to ``Any`` values, which ``Work.load`` reads. Called from Python, the function raises
    ``NotImplementedError``.
    """
    raise NotImplementedError


# The generated code has one statement per field of the record, because jitted code reads a record field only under
# a name known at compile time.
def _make_load_to_dict_code(fields_names):
    code_txt = StringIO()
    code_txt.write("""
def _load_to_dict_(array_, row_ind_, loader_dict_):
    row = array_[row_ind_]""")
    for field_ind, field_name in enumerate(fields_names):
        code_txt.write(f"""
    val = row.{field_name}
    loader_dict_["{field_name}"] = _make_any(val)
    """)
    code_txt.write("""
    return""")
    return code_txt.getvalue()


@overload(load_array_row_into_dict, strict=False, jit_options=jit_options)
def ol_load_array_row_into_dict(array_ty, row_ind_ty, loader_dict_ty):
    """Implement ``load_array_row_into_dict`` in jitted code, with a statement for each field of the array's record."""
    record_ty: Record = array_ty.dtype
    record_fields = record_ty.fields
    fields_names = list(record_fields.keys())
    code_txt = _make_load_to_dict_code(fields_names)
    ns = {**getmodule(load_array_row_into_dict).__dict__, **{
        "AnyType": AnyType, "_make_any": _make_any, "unicode_type": unicode_type
    }}
    code = compile(code_txt, getfile(load_array_row_into_dict), mode="exec")
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    _load_to_dict = ns["_load_to_dict_"]
    return _load_to_dict


def np_struct_member_type(data_ty_: numpy.dtypes.VoidDType, member_name):
    """Return the numba type of the field ``member_name`` of the NumPy structured dtype ``data_ty_``."""
    return from_dtype(data_ty_).fields[member_name].type
