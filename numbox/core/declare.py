import inspect
from llvmlite import ir  # noqa: F401
from numba import njit
from numba.core import cgutils  # noqa: F401
from numba.core.typing.templates import Signature
from numba.extending import intrinsic  # noqa: F401
from types import FunctionType as PyFunctionType
from typing import Optional


def make_swap_name(name):
    return f'__{name}'


def declare(sig, jit_options: Optional[dict] = None):
    assert isinstance(sig, Signature)
    jit_options = isinstance(jit_options, dict) and jit_options or {}

    def wrap(func):
        assert isinstance(func, PyFunctionType)
        func_jit = njit(sig, **jit_options)(func)
        llvm_cfunc_wrapper_name = func_jit.get_compile_result(sig).fndesc.llvm_cfunc_wrapper_name
        func_args_str = ', '.join(inspect.getfullargspec(func).args)
        func_swap_name = make_swap_name(func.__name__)
        code_txt = f"""
@intrinsic
def _{func_swap_name}(typingctx, {func_args_str}):
    def codegen(context, builder, signature, args):
        func_ty_ll = ir.FunctionType(
            context.get_data_type(sig.return_type),
            [context.get_data_type(arg) for arg in sig.args]
        )
        f = cgutils.get_or_insert_function(builder.module, func_ty_ll, "{llvm_cfunc_wrapper_name}")
        return builder.call(f, args)
    return sig, codegen

@njit(sig, **jit_opts)
def {func_swap_name}({func_args_str}):
    return _{func_swap_name}({func_args_str})
"""
        ns = {
            **inspect.getmodule(func).__dict__,
            **{'cgutils': cgutils, 'intrinsic': intrinsic, 'ir': ir, 'jit_opts': jit_options, 'njit': njit, 'sig': sig}
        }
        if ns.get(func_swap_name) is not None:
            raise ValueError(f"Name {func_swap_name} in module {inspect.getmodule(func)} is reserved")
        code = compile(code_txt, inspect.getfile(func), mode='exec')
        exec(code, ns)
        return ns[func_swap_name]
    return wrap
