"""Stdio handles (stdout/stderr/stdin) callable from @njit code.

Uses extern-symbol references in LLVM IR so cache=True remains correct
under ASLR. Linux and macOS expose the handles as data symbols (Linux:
``stdout``/``stderr``/``stdin``; macOS: ``__stdoutp``/``__stderrp``/
``__stdinp`` — what the libc headers' stdio macros expand to). Windows
exposes them via an accessor function (``__acrt_iob_func``).

Windows requires UCRT (Universal C Runtime), bundled with Windows 10
and later. Older Windows versions exposed FILE* via per-MSVC-version
symbols (``_iob``, ``__iob_func``) and are not supported.
"""
from llvmlite import ir as llir
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import intp
from numba.extending import intrinsic

from numbox.core.bindings.utils import extract_literal_str, load_lib, platform_
from numbox.core.proxy.proxy import proxy


__all__ = ["stdout", "stderr", "stdin"]


load_lib("c")


_DATA_SYMBOL_BY_NAME = {
    "Linux": {"stdout": "stdout", "stderr": "stderr", "stdin": "stdin"},
    "Darwin": {"stdout": "__stdoutp", "stderr": "__stderrp", "stdin": "__stdinp"},
}
_WIN_IOB_INDEX = {"stdin": 0, "stdout": 1, "stderr": 2}


def _get_or_insert_global(module, ll_ty, name):
    try:
        return module.get_global(name)
    except KeyError:
        gv = llir.GlobalVariable(module, ll_ty, name=name)
        gv.linkage = "external"
        return gv


@intrinsic(prefer_literal=True)
def _stdio_handle(typingctx, name_ty):
    name = extract_literal_str("_stdio_handle", name_ty, field="name")
    if name not in ("stdout", "stderr", "stdin"):
        raise TypingError(
            f"_stdio_handle: name must be one of stdout/stderr/stdin, got {name!r}"
        )
    if platform_ not in ("Linux", "Darwin", "Windows"):
        raise TypingError(
            f"_stdio_handle: unsupported platform {platform_!r}")

    def codegen(context, builder, signature, arguments):
        intp_ll = context.get_value_type(intp)
        ptr_ll = llir.IntType(8).as_pointer()
        if platform_ in ("Linux", "Darwin"):
            sym = _DATA_SYMBOL_BY_NAME[platform_][name]
            gv = _get_or_insert_global(builder.module, ptr_ll, sym)
            file_ptr = builder.load(gv)
            return builder.ptrtoint(file_ptr, intp_ll)
        # platform_ == "Windows" (guarded at typing time above)
        func_ty = llir.FunctionType(ptr_ll, [llir.IntType(32)])
        func_p = get_or_insert_function(
            builder.module, func_ty, "__acrt_iob_func")
        idx = llir.Constant(llir.IntType(32), _WIN_IOB_INDEX[name])
        file_ptr = builder.call(func_p, [idx])
        return builder.ptrtoint(file_ptr, intp_ll)

    sig = intp(name_ty)
    return sig, codegen


@proxy(intp(), jit_options={"cache": True})
def stdout():
    """Return the current process's stdout FILE* as intp. See module docstring."""
    return _stdio_handle("stdout")


@proxy(intp(), jit_options={"cache": True})
def stderr():
    """Return the current process's stderr FILE* as intp. See module docstring."""
    return _stdio_handle("stderr")


@proxy(intp(), jit_options={"cache": True})
def stdin():
    """Return the current process's stdin FILE* as intp. See module docstring."""
    return _stdio_handle("stdin")
