import inspect
from llvmlite import ir  # noqa: F401
from numba import njit
from numba.core import cgutils  # noqa: F401
from numba.core.typing.templates import Signature
from numba.extending import intrinsic  # noqa: F401
from types import FunctionType as PyFunctionType
from typing import List, Optional, Tuple


def make_proxy_name(name):
    return f'__{name}'


def make_params_strings(func):
    func_params = inspect.signature(func).parameters
    func_params_str = ', '.join(
        [k if v.default == inspect._empty else f'{k}={v.default}' for k, v in func_params.items()]
    )
    func_names_params_str = ', '.join(func_params.keys())
    return func_params_str, func_names_params_str


def proxy(sig, jit_options: Optional[dict] = None):
    main_sig = isinstance(sig, Signature) and sig or isinstance(sig, (List, Tuple)) and sig[0]
    jit_options = isinstance(jit_options, dict) and jit_options or {}

    def wrap(func):
        assert isinstance(func, PyFunctionType)
        func_jit = njit(sig, **jit_options)(func)
        llvm_cfunc_wrapper_name = func_jit.get_compile_result(main_sig).fndesc.llvm_cfunc_wrapper_name
        func_args_str, func_names_args_str = make_params_strings(func)
        func_proxy_name = make_proxy_name(func.__name__)
        code_txt = f"""
@intrinsic
def _{func_proxy_name}(typingctx, {func_names_args_str}):
    def codegen(context, builder, signature, args):
        func_ty_ll = ir.FunctionType(
            context.get_data_type(main_sig.return_type),
            [context.get_data_type(arg) for arg in main_sig.args]
        )
        f = cgutils.get_or_insert_function(builder.module, func_ty_ll, "{llvm_cfunc_wrapper_name}")
        return builder.call(f, args)
    return main_sig, codegen

@njit(sig, **jit_opts)
def {func_proxy_name}({func_args_str}):
    return _{func_proxy_name}({func_names_args_str})
"""
        ns = {
            **inspect.getmodule(func).__dict__,
            **{
                'cgutils': cgutils, 'intrinsic': intrinsic, 'ir': ir, 'jit_opts': jit_options, 'njit': njit,
                'sig': sig, 'main_sig': main_sig
            }
        }
        if ns.get(func_proxy_name) is not None:
            raise ValueError(f"Name {func_proxy_name} in module {inspect.getmodule(func)} is reserved")
        code = compile(code_txt, inspect.getfile(func), mode='exec')
        exec(code, ns)
        return ns[func_proxy_name]
    return wrap
