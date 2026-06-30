import hashlib
import inspect
from llvmlite import binding as ll
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


def _stable_cfunc_alias(func, main_sig):
    """Deterministic, process-stable LLVM symbol name for ``func``'s cfunc wrapper.

    numba mangles the wrapper name (``fndesc.llvm_cfunc_wrapper_name``) with a
    process-local unique-id abi-tag (``v<N>`` from ``FunctionIdentity.unique_id``,
    an ``itertools.count``). That tag is not part of the numba cache key, so two
    processes give the same function different wrapper names. Persisting a
    reference to that name — in this proxy, or in any ``cache=True`` caller that
    inlines the ``inline='always'`` proxy — lets concurrently-built caches pair a
    body object defining ``v<Na>`` with a caller referencing ``v<Nb>``, which
    aborts on load with ``LLVM ERROR: Symbol not found: cfunc...``. Referencing
    this deterministic alias instead (resolved per-process via ``add_symbol``,
    the same mechanism ``pysqlite_bridge`` uses) keeps cached references valid
    across processes.
    """
    raw = f"{func.__module__ or ''}.{func.__qualname__}.{main_sig}".encode("utf-8")
    safe_name = "".join(c if c.isascii() and c.isalnum() else "_" for c in func.__name__)
    return f"numbox_pxy_{safe_name}_{hashlib.sha256(raw).hexdigest()[:16]}"


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
        cres = func_jit.get_compile_result(main_sig)
        # Register a process-stable alias for the body's cfunc wrapper and reference
        # that instead of numba's process-local ``v<uid>`` name (see _stable_cfunc_alias).
        cfunc_alias = _stable_cfunc_alias(func, main_sig)
        ll.add_symbol(cfunc_alias, cres.library.get_pointer_to_function(cres.fndesc.llvm_cfunc_wrapper_name))
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
        f = cgutils.get_or_insert_function(builder.module, func_ty_ll, "{cfunc_alias}")
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
        co_firstlineno = func.__code__.co_firstlineno
        if co_firstlineno < njit_lineno_in_txt:
            raise ValueError(
                f"@proxy function {func.__name__!r} is defined at line {co_firstlineno} of "
                f"{inspect.getfile(func)}, above the cache anchor's minimum line "
                f"{njit_lineno_in_txt}; the generated @njit cannot be anchored to its "
                f"co_firstlineno (a negative prepend would mis-anchor it). Move the "
                f"function further down in the file."
            )
        prepend = co_firstlineno - njit_lineno_in_txt
        prefixed = '\n' * prepend + code_txt
        code = compile(prefixed, inspect.getfile(func), mode='exec')
        exec(code, ns)  # nosec B102 - JIT codegen of internal source
        dispatcher = ns[func_proxy_name]
        dispatcher.as_func = CompileResultWAP(cres)
        return dispatcher
    return wrap


def proxy_if_available(lib, sig, jit_options: Optional[dict] = None):
    """Like ``proxy(sig, jit_options=...)``, but stubs out the wrapper if
    the C symbol matching ``func.__name__`` is absent from ``lib``.

    Use for binding sets that target multiple library versions where some
    symbols only exist in newer releases. Callers get a stub that raises
    ``NotImplementedError`` instead of a confusing LLVM link error at call
    time. Parallel to ``cres_if_available`` in :mod:`numbox.utils.highlevel`.

    The stub does NOT expose ``.as_func`` — a function-value handle is
    meaningless without an underlying jitted body, and a stub one would
    have to either raise on attribute access (ugly) or pretend to be a
    function value (worse). Callers that pass ``.as_func`` to function-
    type arguments must guard the access::

        if hasattr(my_binding, "as_func"):
            use(my_binding.as_func)
    """
    def _(func):
        if hasattr(lib, func.__name__):
            return proxy(sig, jit_options=jit_options)(func)

        def stub(*args, **_kwargs):
            raise NotImplementedError(f"{func.__name__} is not available")
        stub.__name__ = make_proxy_name(func.__name__)
        stub.__qualname__ = func.__qualname__
        stub.__doc__ = func.__doc__
        return stub
    return _
