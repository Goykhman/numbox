from llvmlite import ir as llir
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import int32, int64, intp, void
from numba.extending import intrinsic

from numbox.core.bindings.utils import platform_, load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy
from numbox.utils.lowlevel import load_at, store_at


__all__ = ["errno_get", "errno_set"]


load_lib("c")


# Each C library returns the address of the calling thread's errno from a function of its own.
_ERRNO_ACCESSOR = {
    "Linux": "__errno_location",
    "Darwin": "__error",
    "Windows": "_errno",
}


@intrinsic
def _errno_ptr(typingctx):
    """Return the address of the calling thread's ``errno`` as an ``intp``.

    The intrinsic calls the C library's accessor: ``__errno_location`` on Linux, ``__error`` on macOS and ``_errno`` on
    Windows. numba raises ``TypingError`` when it compiles the call on any other platform.
    """
    sym = _ERRNO_ACCESSOR.get(platform_)
    if sym is None:
        raise TypingError(
            f"_errno_ptr: unsupported platform {platform_!r}")

    def codegen(context, builder, signature, arguments):
        intp_ll = context.get_value_type(intp)
        i32_ptr = llir.IntType(32).as_pointer()
        # The declaration carries no readnone or memory(none) attribute, although glibc declares __errno_location with
        # __attribute_const__, which LLVM expresses as readnone. With that attribute, LLVM could move the call out of a
        # loop, or use one result across other calls. The address belongs to the thread that computed it, so code that
        # runs on several threads, such as the body of a parallel loop, must look it up on the thread that uses it.
        # Because LLVM treats the call as opaque, the code calls the accessor at every errno access, on the thread that
        # makes the access.
        func_ty = llir.FunctionType(i32_ptr, [])
        func_p = get_or_insert_function(builder.module, func_ty, sym)
        ptr = builder.call(func_p, [])
        return builder.ptrtoint(ptr, intp_ll)
    return intp(), codegen


@proxy(int32(), jit_options=jit_options)
def errno_get():
    """Return the calling thread's ``errno`` as an ``int32``.

    C reference: `errno(3) <https://man7.org/linux/man-pages/man3/errno.3.html>`_.

    Each call looks up the errno of the thread that runs it, so an iteration of a ``prange`` loop reads the errno of
    its own thread. A Python caller reads the errno that a jitted function set on the same thread. After a parallel
    loop, the calling thread's errno depends on numba's threading layer. Under OpenMP, the calling thread runs some
    iterations itself, and its errno holds the value that those iterations set. Under the workqueue layer, the calling
    thread runs no iterations, and its errno is unchanged.
    """
    return load_at(_errno_ptr(), int32)


@proxy(void(int64), jit_options=jit_options)
def errno_set(v):
    """Set the calling thread's ``errno`` to ``v``, narrowed to ``int32``.

    C reference: `errno(3) <https://man7.org/linux/man-pages/man3/errno.3.html>`_.

    ``v`` is an ``int64``, Python's integer width, and the binding stores its low 32 bits, the width of C's
    ``int errno``: ``2**32 + 5`` is stored as 5.
    """
    store_at(_errno_ptr(), int32(v))
