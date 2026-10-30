"""The C standard streams, ``stdout``, ``stderr`` and ``stdin``, as ``FILE *`` values for jitted code and Python.

:func:`stdout`, :func:`stderr` and :func:`stdin` return the running process's three streams as ``intp`` addresses,
which C functions such as ``fputs``, ``fflush`` and ``fprintf`` take. Each call reads the address in the running
process, so a cached caller never uses an address from the process that compiled it. On Windows, the functions need the
Universal C Runtime (UCRT) of Windows 10 and later. Older Windows versions name the streams ``_iob`` or ``__iob_func``,
differently in each Visual C++ runtime, and numbox does not support them.
"""
from llvmlite import ir as llir
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import intp
from numba.extending import intrinsic

from numbox.core.bindings.utils import extract_literal_str, load_lib, platform_
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy


__all__ = ["stdout", "stderr", "stdin"]


load_lib("c")


# On Linux and macOS, the C library exports each stream as a data symbol that holds its FILE *: stdout, stderr and
# stdin on Linux, and on macOS __stdoutp, __stderrp and __stdinp, the names that its stdio.h macros expand to. On
# Windows, the UCRT returns the streams from the function __acrt_iob_func, at the indexes in _WIN_IOB_INDEX.
_DATA_SYMBOL_BY_NAME = {
    "Linux": {"stdout": "stdout", "stderr": "stderr", "stdin": "stdin"},
    "Darwin": {"stdout": "__stdoutp", "stderr": "__stderrp", "stdin": "__stdinp"},
}
_WIN_IOB_INDEX = {"stdin": 0, "stdout": 1, "stderr": 2}


def _get_or_insert_global(module, ll_ty, name):
    """Return the global ``name`` of the LLVM ``module``, declared as an external of type ``ll_ty`` if absent."""
    try:
        return module.get_global(name)
    except KeyError:
        gv = llir.GlobalVariable(module, ll_ty, name=name)
        gv.linkage = "external"
        return gv


@intrinsic(prefer_literal=True)
def _stdio_handle(typingctx, name_ty):
    """Return the ``FILE *`` of the standard stream named by a string literal, as an ``intp``.

    The name is ``"stdout"``, ``"stderr"`` or ``"stdin"``. numba raises ``TypingError`` when it compiles the call for
    any other name, and on a platform other than Linux, macOS and Windows.
    """
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
        # The IR names the symbol as an external global and loads the FILE * from it when the code runs. An address
        # written into the IR would be the compiling process's, which ASLR makes wrong in any later process that loads
        # the cached code.
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


@proxy(intp(), jit_options=jit_options)
def stdout():
    """Return the running process's C stream ``stdout``, a ``FILE *``, as an ``intp``.

    C reference: `stdout(3) <https://man7.org/linux/man-pages/man3/stdout.3.html>`_.

    Each call reads the address in the running process. C buffers this stream apart from Python's ``sys.stdout``, so
    flush both, with ``fflush(stdout())`` and ``sys.stdout.flush()``, to keep jitted output in order with Python's.
    """
    return _stdio_handle("stdout")


@proxy(intp(), jit_options=jit_options)
def stderr():
    """Return the running process's C stream ``stderr``, a ``FILE *``, as an ``intp``.

    C reference: `stderr(3) <https://man7.org/linux/man-pages/man3/stderr.3.html>`_.

    Each call reads the address in the running process. C writes ``stderr`` without a buffer on Linux. On Windows, C
    buffers a ``stderr`` that goes to a file or a pipe, so call ``fflush(stderr())`` there.
    """
    return _stdio_handle("stderr")


@proxy(intp(), jit_options=jit_options)
def stdin():
    """Return the running process's C stream ``stdin``, a ``FILE *``, as an ``intp``.

    C reference: `stdin(3) <https://man7.org/linux/man-pages/man3/stdin.3.html>`_.

    Each call reads the address in the running process.
    """
    return _stdio_handle("stdin")
