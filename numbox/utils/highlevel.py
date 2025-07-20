import hashlib
import re
from inspect import getfile, getmodule, getsource
from io import StringIO
from numba import njit
from numba.core.itanium_mangler import mangle_type_or_value
from numba.core.types import Type
from numba.core.types.functions import Dispatcher
from numba.core.types.function_type import CompileResultWAP
from numba.core.typing.templates import Signature
from numba.experimental.function_type import FunctionType
from numba.experimental.structref import define_boxing, new, StructRefProxy
from numba.extending import overload, overload_method
from textwrap import dedent, indent
from typing import Callable, Dict, Iterable, Optional

from numbox.core.configurations import default_jit_options
from numbox.utils.standard import make_params_strings


def _file_anchor():
    raise NotImplementedError


def cres(sig, **kwargs):
    """ Returns Python proxy to `FunctionType` rather than `CPUDispatcher` returned by `njit` """
    if not isinstance(sig, Signature):
        raise ValueError(f"Expected a single signature, found {sig} of type {type(sig)}")

    def _(func):
        func_jit = njit(sig, **kwargs)(func)
        sigs = func_jit.nopython_signatures
        assert len(sigs) == 1, f"Ambiguous signature, {sigs}"
        func_cres = func_jit.get_compile_result(sigs[0])
        cres_wap = CompileResultWAP(func_cres)
        return cres_wap
    return _


def determine_field_index(struct_ty, field_name):
    for i_, field_pair in enumerate(struct_ty._fields):
        if field_pair[0] == field_name:
            return i_
    raise ValueError(f"{field_name} not in {struct_ty}")


def hash_type(ty: Type):
    mangled_ty = mangle_type_or_value(ty)
    return hashlib.sha256(mangled_ty.encode("utf-8")).hexdigest()


def make_structref_code_txt(
    struct_name: str,
    struct_fields: Iterable[str] | Dict[str, Type],
    struct_type_class: type | Type,
    struct_methods: Optional[Dict[str, Callable]] = None,
    jit_options: Optional[dict] = None
):
    if isinstance(struct_fields, dict):
        struct_fields, fields_types = list(struct_fields.keys()), list(struct_fields.values())
    else:
        assert isinstance(struct_fields, (list, tuple)), struct_fields
        fields_types = None
    struct_fields_str = ", ".join([field for field in struct_fields])
    make_name = f"make_{struct_name.lower()}"
    new_returns = f"{make_name}({struct_fields_str})"
    repr_str = f"f'{struct_name}(" + ", ".join([f"{field}={{self.{field}}}" for field in struct_fields]) + ")'"
    code_txt = StringIO()
    code_txt.write(f"""
class {struct_name}(StructRefProxy):
    def __new__(cls, {struct_fields_str}):
        return {new_returns}

    def __repr__(self):
        return {repr_str}
""")
    for field in struct_fields:
        code_txt.write(f"""
    @property
    @njit(**jit_options)
    def {field}(self):
        return self.{field}
""")
    methods_code_txt = StringIO()
    if struct_methods is not None:
        assert isinstance(struct_methods, dict), f"""
    Expected dictionary of methods names to callable, got {struct_methods}"""
        for method_name, method in struct_methods.items():
            params_str, names_params_str = make_params_strings(method)
            names_params_lst = names_params_str.split(", ")
            self_name = names_params_lst[0]
            names_params_str_wo_self = ", ".join(names_params_lst[1:])
            method_source = dedent(getsource(method))
            method_hash = hashlib.sha256(method_source.encode("utf-8")).hexdigest()
            code_txt.write(f"""
    def {method_name}({params_str}):
        return {self_name}.{method_name}_{method_hash}({names_params_str_wo_self})
    
    @njit(**jit_options)
    def {method_name}_{method_hash}({params_str}):
        return {self_name}.{method_name}({names_params_str_wo_self})
""")
            method_header = re.findall(r"^\s*def\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*:[^\n]*", method_source, re.MULTILINE)
            assert len(method_header) == 1, method_header
            method_name, params_str_ = method_header[0]
            assert params_str == params_str_, (params_str, params_str_)
            method_source = re.sub(r"\bdef\s+([a-zA-Z_]\w*)\b", f"def _", method_source)
            methods_code_txt.write(f"""
@overload_method({struct_type_class.__name__}, "{method_name}", jit_options=jit_options)
def ol_{method_name}({params_str}):
{indent(method_source, "    ")}
    return _
""")
    code_txt.write(f"""
define_boxing({struct_type_class.__name__}, {struct_name})
""")
    struct_type_name = f"{struct_name}Type"
    struct_fields_ty_str = ", ".join([f"{field}_ty" for field in struct_fields])
    struct_type_code_block = ""
    if fields_types is None:
        struct_type_code_block = f"""fields_types = [{struct_fields_ty_str}]
    fields_and_their_types = list(zip(fields, fields_types))    
    {struct_name}Type = {struct_type_class.__name__}(fields_and_their_types)        
"""
    else:
        code_txt.write(f"""
fields_and_their_types = list(zip(fields, fields_types))    
{struct_name}Type = {struct_type_class.__name__}(fields_and_their_types)
""")
    ctor_code_block = "\n".join([f"        struct_.{field} = {field}" for field in struct_fields])
    code_txt.write(f"""
@overload({struct_name}, strict=False, jit_options=jit_options)
def ol_{struct_name.lower()}({struct_fields_ty_str}):
    {struct_type_code_block}
    def ctor({struct_fields_str}):
        struct_ = new({struct_type_name})
{ctor_code_block}
        return struct_
    return ctor
""")
    if fields_types is not None:
        code_txt.write(f"""
{make_name}_sig = {struct_name}Type(*fields_types)
""")
    else:
        code_txt.write(f"""
{make_name}_sig = None
""")
    code_txt.write(f"""
@njit({make_name}_sig, **jit_options)
def {make_name}({struct_fields_str}):
    return {struct_name}({struct_fields_str})
""")
    code_txt = code_txt.getvalue() + methods_code_txt.getvalue()
    return code_txt, fields_types


def make_structref(
    struct_name: str,
    struct_fields: Iterable[str] | Dict[str, Type],
    struct_type_class: type | Type,
    *,
    struct_methods: Optional[Dict[str, Callable]] = None,
    jit_options: Optional[dict] = None
):
    """
    Makes structure type with `struct_name` and `struct_fields` from the StructRef type class.

    A unique `struct_type_class` for each structref needs to be provided.
    If caching of code that will be using the created struct type is desired,
    these type class(es) need/s to be defined in a python module that is *not* executed.
    (Same requirement is also to observed even when the full definition of StructRef
    is entirely hard-coded rather than created dynamically.)

    In particular, that's why `struct_type_class` cannot be incorporated into
    the dynamic compile / exec routine here.

    Dictionary of methods to be bound to the created structref can be provided as well.
    Struct methods will get inlined into the caller if numba deems it to be optimal
    (even if `jit_options` says otherwise), therefore changing the methods code
    without poking the jitted caller can result in a stale cache - when the latter is
    cached. This is not an exclusive limitation of a dynamic structref creation via
    this function and is equally true when the structref definition is coded explicitly.
    """
    code_txt, fields_types = make_structref_code_txt(
        struct_name, struct_fields, struct_type_class, struct_methods, jit_options
    )
    if jit_options is None:
        jit_options = default_jit_options
    ns = {
        **getmodule(_file_anchor).__dict__,
        **{
            "fields": struct_fields,
            "fields_types": fields_types,
            "define_boxing": define_boxing,
            "jit_options": jit_options,
            "new": new,
            "njit": njit,
            "overload_method": overload_method,
            "StructRefProxy": StructRefProxy,
            struct_type_class.__name__: struct_type_class
        }
    }
    code = compile(code_txt, getfile(_file_anchor), mode="exec")
    exec(code, ns)
    return ns[struct_name]


def prune_type(ty):
    if isinstance(ty, Dispatcher):
        sigs = ty.get_call_signatures()[0]
        assert len(sigs) == 1, f"Ambiguous signature, {sigs}"
        ty = FunctionType(sigs[0])
    return ty
