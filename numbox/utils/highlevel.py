import hashlib
from inspect import getfile, getmodule
from io import StringIO
from numba import njit
from numba.core.itanium_mangler import mangle_type_or_value
from numba.core.types import Type
from numba.core.types.functions import Dispatcher
from numba.core.types.function_type import CompileResultWAP
from numba.core.typing.templates import Signature
from numba.experimental.function_type import FunctionType
from numba.experimental.structref import define_proxy, StructRefProxy

from numbox.core.configurations import default_jit_options


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
    return hashlib.sha256(mangled_ty.encode('utf-8')).hexdigest()


def make_structref(name, fields, struct_type_class, jit_options=None):
    """
    Makes structure type with `name` and `fields` from the StructRef type class.
    A unique `struct_type_class` for each structref needs to be provided.
    If caching of code that will be using the created struct type is desired,
    these type class(es) need/s to be defined in a python module that is _not_ executed.
    (Same requirement is also to observed even when the full definition of StructRef
    is entirely hard-coded rather than created dynamically.)
    In particular, that's why `struct_type_class` cannot be incorporated into
    the dynamic compile / exec routine here.
    """
    if jit_options is None:
        jit_options = default_jit_options
    code_txt = StringIO()
    params = ", ".join([field for field in fields])
    code_txt.write(f"""
class {name}(StructRefProxy):
    def __new__(cls, {params}):
        return super().__new__(cls, {params})""")
    for field in fields:
        code_txt.write(f"""
    @property
    @njit(**jit_options)
    def {field}(self):
        return self.{field}""")
    code_txt = code_txt.getvalue()
    ns = {
        **getmodule(_file_anchor).__dict__,
        **{
            "jit_options": jit_options,
            "njit": njit,
            "StructRefProxy": StructRefProxy,
        }
    }
    code = compile(code_txt, getfile(_file_anchor), mode="exec")
    exec(code, ns)
    define_proxy(ns[name], struct_type_class, fields)
    return ns[name]


def prune_type(ty):
    if isinstance(ty, Dispatcher):
        sigs = ty.get_call_signatures()[0]
        assert len(sigs) == 1, f"Ambiguous signature, {sigs}"
        ty = FunctionType(sigs[0])
    return ty
