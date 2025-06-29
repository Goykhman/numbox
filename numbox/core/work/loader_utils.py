import numpy

from inspect import getfile, getmodule
from io import StringIO
from numba import from_dtype
from numba.core.types import Record, unicode_type
from numba.extending import overload

from numbox.core.configurations import default_jit_options
from numbox.core.any.any_type import AnyType, _make_any


def load_array_row_into_dict(*args):
    raise NotImplementedError


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


@overload(load_array_row_into_dict, strict=False, jit_options=default_jit_options)
def ol_load_array_row_into_dict(array_ty, row_ind_ty, loader_dict_ty):
    record_ty: Record = array_ty.dtype
    record_fields = record_ty.fields
    fields_names = list(record_fields.keys())
    code_txt = _make_load_to_dict_code(fields_names)
    ns = {**getmodule(load_array_row_into_dict).__dict__, **{
        "AnyType": AnyType, "_make_any": _make_any, "unicode_type": unicode_type
    }}
    code = compile(code_txt, getfile(load_array_row_into_dict), mode="exec")
    exec(code, ns)
    _load_to_dict = ns["_load_to_dict_"]
    return _load_to_dict


def np_struct_member_type(data_ty_: numpy.dtypes.VoidDType, member_name):
    return from_dtype(data_ty_).fields[member_name].type
