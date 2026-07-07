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


_ERRNO_ACCESSOR = {
    "Linux": "__errno_location",
    "Darwin": "__error",
    "Windows": "_errno",
}


@intrinsic
def _errno_ptr(typingctx):
    sym = _ERRNO_ACCESSOR.get(platform_)
    if sym is None:
        raise TypingError(
            f"_errno_ptr: unsupported platform {platform_!r}")

    def codegen(context, builder, signature, arguments):
        intp_ll = context.get_value_type(intp)
        i32_ptr = llir.IntType(32).as_pointer()
        # NOTE: deliberately NOT setting readnone/memory(none) on this
        # declaration, even though glibc declares __errno_location with
        # __attribute_const__ (which maps to LLVM's readnone). Setting the
        # attribute would allow LLVM to hoist the call out of loops or
        # across function-call boundaries — fatal for correctness when the
        # OS thread changes (e.g., across @njit(parallel=True) workers).
        # An opaque/side-effecting call is what we need here.
        func_ty = llir.FunctionType(i32_ptr, [])
        func_p = get_or_insert_function(builder.module, func_ty, sym)
        ptr = builder.call(func_p, [])
        return builder.ptrtoint(ptr, intp_ll)
    return intp(), codegen


@proxy(int32(), jit_options=jit_options)
def errno_get():
    """Return the current thread's errno as int32.

    Re-resolves the per-thread errno location on every call: on
    @njit(parallel=True) workers, the accessor returns that worker's
    errno. A Python caller observes errno set inside a normal @njit
    function (same OS thread), but not errno set inside a parallel
    region's worker (different OS thread).
    """
    return load_at(_errno_ptr(), int32)


@proxy(void(int64), jit_options=jit_options)
def errno_set(v):
    """Set the current thread's errno to v.

    Accepts int64 (Python's default integer width); the value is
    narrowed to int32 before being stored at the per-thread errno
    location, matching C's ``int errno``.
    """
    store_at(_errno_ptr(), int32(v))
