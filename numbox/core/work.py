from numba import njit
from numba.core.types import FunctionType, NoneType, StructRef, Tuple, unicode_type
from numba.core.typing.context import Context
from numba.experimental.structref import define_boxing, new, register, StructRefProxy
from numba.extending import intrinsic, overload, overload_method

from numbox.core.configurations import default_jit_options
from numbox.utils.lowlevel import (
    extract_struct_member, get_func_p_from_func_struct, get_ll_func_sig
)


@register
class WorkTypeClass(StructRef):
    pass


class Work(StructRefProxy):
    def __new__(cls, *args, **kws):
        return make_work(*args, **kws)

    @property
    @njit(**default_jit_options)
    def name(self):
        return self.name

    @property
    @njit(**default_jit_options)
    def data(self):
        return self.data

    @property
    @njit(**default_jit_options)
    def sources(self):
        return self.sources

    @njit(**default_jit_options)
    def calculate(self):
        return self.calculate()


define_boxing(WorkTypeClass, Work)


@overload(Work, strict=False, jit_options=default_jit_options)
def ol_work(name_ty, data_ty, sources_ty, derive_ty):
    work_attributes_ = [
        ("name", unicode_type),
        ("data", data_ty),
        ("sources", Tuple(sources_ty)),
        ("derive", derive_ty),
    ]
    assert isinstance(derive_ty, (FunctionType, NoneType)), f"""Either None or Compile Result supported,
not CPUDispatcher, got {derive_ty}, of type {type(derive_ty)}"""
    work_type_ = WorkTypeClass(work_attributes_)

    def work_constructor(name_, data_, sources_, derive_):
        w = new(work_type_)
        w.name = name_
        w.data = data_
        w.sources = sources_
        w.derive = derive_
        return w
    return work_constructor


def _make_work(*_, **__):
    raise NotImplementedError


@overload(_make_work, strict=False, jit_options=default_jit_options)
def ol_make_work(name_ty, data_ty, sources_ty, derive_ty_):
    if isinstance(sources_ty, NoneType):
        def _(name_, data_, sources_, derive_):
            return Work(name_, data_, (), derive_)
    else:
        def _(name_, data_, sources_, derive_):
            return Work(name_, data_, sources_, derive_)
    return _


@njit(**default_jit_options)
def make_work(name, data, sources=None, derive=None):
    return _make_work(name, data, sources, derive)


@intrinsic
def _call_function(typingctx: Context, func_ty: FunctionType, args_ty: Tuple):
    def codegen(context, builder, signature, arguments):
        func_struct, args = arguments
        derive_args = []
        for arg_ind, arg_ty in enumerate(args_ty):
            arg_struct = builder.extract_value(args, arg_ind)
            data = extract_struct_member(context, builder, args_ty[arg_ind], arg_struct, "data")
            derive_args.append(data)
        derive_p_raw = get_func_p_from_func_struct(builder, func_struct)
        func_ty_ll = get_ll_func_sig(context, func_ty)
        derive_p = builder.bitcast(derive_p_raw, func_ty_ll.as_pointer())
        res = builder.call(derive_p, derive_args)
        return res
    sig = func_ty.signature.return_type(func_ty, args_ty)
    return sig, codegen


@overload_method(WorkTypeClass, "calculate", strict=False, jit_options=default_jit_options)
def ol_calculate(self_class):
    def _(self):
        v = _call_function(self.derive, self.sources)
        self.data = v
    return _
