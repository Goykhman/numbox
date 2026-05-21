import inspect
from llvmlite import ir  # noqa: F401
from numba import njit
from numba.core import cgutils  # noqa: F401
from numba.core.types.function_type import CompileResultWAP
from numba.core.typing.templates import Signature
from numba.extending import intrinsic  # noqa: F401
from types import FunctionType as PyFunctionType
from typing import List, Optional, Tuple

from numbox.utils.standard import make_params_strings


def make_proxy_name(name):
    return f'__{name}'


def proxy(sig, jit_options: Optional[dict] = None):
    """ Create a proxy for the decorated function `func` with the given signature(s) `sig`.

    The original function `func` will be eagerly JIT-compiled with the given signature(s).
    A proxy with the name `func_proxy_name` will be created to call `func` in the LLVM scope.
    The original function's variable will be bound to the proxy, i.e., calling the decorated
    function will call the proxy.

    The proxy is a JIT-compiled wrap that invokes the intrinsic that *declares* the `func`
    and calls it with the original arguments. Declaration instructions are relatively cheap
    to statically link into (potential) caller's LLVM code, which is the main motivation behind
    this decorator.

    Machine code for `func` can be cached when so specified in `jit_options`, in which case its
    JIT-compilation will load the `func` into the LLVM scope. Caching option is the other major
    motivation for this decorator, without the need to cache one can avoid static linking
    of the callee's LLVM code into the caller's by simply ignoring the former.

    In case when more than one signature is provided as the `sig` parameter, it is assumed
    that the first signature is the 'main' one while the other ones are supplied to
    allow for the `Omitted` types with default values for (some of) the parameters.

    The returned dispatcher also exposes ``.as_func``: a ``CompileResultWAP``
    for the main signature. Cacheable as a called jitted function (via the
    dispatcher); passable as a function-type argument (via ``.as_func``).

    See tests for some examples of the use cases.
    """
    main_sig = isinstance(sig, Signature) and sig or isinstance(sig, (List, Tuple)) and sig[0]
    jit_options = isinstance(jit_options, dict) and jit_options or {}
    jit_opts = jit_options.copy()
    jit_opts.update(jit_opts, inline='always')

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
                'cgutils': cgutils, 'intrinsic': intrinsic, 'ir': ir, 'jit_opts': jit_opts, 'njit': njit,
                'sig': sig, 'main_sig': main_sig
            }
        }
        if ns.get(func_proxy_name) is not None:
            raise ValueError(f"Name {func_proxy_name} in module {inspect.getmodule(func)} is reserved")
        # Anchor the wrapper at func's source file: prepend blank lines so the
        # wrapper's @njit decorator lands at func.__code__.co_firstlineno (the
        # user's @proxy decorator line). See docs/numbox.core.proxy.rst —
        # section "Cache-anchor mechanism" — for the design rationale + the
        # findsource-finds-@-in-docstring hazard this avoids.
        code_lines = code_txt.split('\n')
        njit_lineno_in_txt = next(
            i + 1 for i, line in enumerate(code_lines) if line.startswith('@njit(')
        )
        prepend = max(0, func.__code__.co_firstlineno - njit_lineno_in_txt)
        prefixed = '\n' * prepend + code_txt
        code = compile(prefixed, inspect.getfile(func), mode='exec')
        exec(code, ns)
        dispatcher = ns[func_proxy_name]
        dispatcher.as_func = CompileResultWAP(func_jit.get_compile_result(main_sig))
        return dispatcher
    return wrap
