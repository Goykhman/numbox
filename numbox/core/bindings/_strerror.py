"""Thread-safe ``strerror_safe(errnum, buf, buflen)`` callable from @njit.

Supported libcs (other Linux libcs are not supported):

- **glibc** — `__xpg_strerror_r
  <https://codebrowser.dev/glibc/glibc/string/xpg-strerror.c.html>`_ (POSIX
  XSI form, present on glibc 2.3.4+ which shipped in 2004)
- **musl** — also ``__xpg_strerror_r``: musl declares ``strerror_r`` as the
  POSIX XSI form and exports ``__xpg_strerror_r`` as a `weak alias
  <https://git.musl-libc.org/cgit/musl/tree/src/string/strerror_r.c>`_ to
  the same implementation, so the same symbol resolves on glibc and musl
- **macOS** — `strerror_r
  <https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/strerror_r.3.html>`_
  (POSIX form)
- **Windows** — `strerror_s
  <https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/strerror-s-strerror-s-wcserror-s-wcserror-s>`_
  with reordered args (buffer, size, errnum)

On glibc, plain ``strerror_r`` is the GNU form (returns ``char *``) and would
not match this module's POSIX-shaped IR. The Linux symbol selector
unconditionally picks ``__xpg_strerror_r`` and never the GNU form. A
``strerror_r`` fallback remains in the selector as defense-in-depth in case
a future libc drops the ``__xpg_strerror_r`` symbol, but the fallback is
unreachable on every libc currently supported (verified by the Alpine
``musl_symbol_check`` CI canary).
"""
import llvmlite.binding as ll
from llvmlite import ir as llir
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import int32, int64, intp
from numba.extending import intrinsic

from numbox.core.bindings.utils import intp_ll_type, platform_, load_lib
from numbox.core.proxy.proxy import proxy


__all__ = ["strerror_safe"]


load_lib("c")


def _select_posix_symbol():
    if platform_ == "Linux":
        if ll.address_of_symbol("__xpg_strerror_r") is not None:
            return "__xpg_strerror_r"
        # Fallback: defense-in-depth in case a future libc drops the
        # __xpg_strerror_r symbol. Currently unreachable — glibc 2.3.4+
        # exports it directly, and musl exports it as a weak alias to its
        # own (POSIX-form) strerror_r. On glibc, plain strerror_r is the
        # GNU char*-returning form and would NOT match our IR signature;
        # we never want to land here on glibc.
        return "strerror_r"
    if platform_ == "Darwin":
        return "strerror_r"
    return None


@intrinsic
def _strerror_safe(typingctx, errnum_ty, buf_ty, buflen_ty):
    # The underlying libc signatures (strerror_r, __xpg_strerror_r,
    # strerror_s) all take a 32-bit int for errnum. Enforce int32 at
    # typing time so a future caller that drops the explicit cast in
    # the public strerror_safe wrapper gets a clean TypingError instead
    # of an opaque IR-lowering type-mismatch.
    if errnum_ty != int32:
        raise TypingError(
            f"_strerror_safe: errnum must be int32, got {errnum_ty!r}")
    if platform_ == "Windows":
        sym = "strerror_s"
    else:
        sym = _select_posix_symbol()
        if sym is None:
            raise TypingError(
                f"_strerror_safe: unsupported platform {platform_!r}")

    def codegen(context, builder, signature, arguments):
        errnum, buf_p, buflen = arguments
        i32 = llir.IntType(32)
        i8p = llir.IntType(8).as_pointer()
        size_t_ll = intp_ll_type(context)
        buf = builder.inttoptr(buf_p, i8p)
        if platform_ == "Windows":
            func_ty = llir.FunctionType(i32, [i8p, size_t_ll, i32])
            func_p = get_or_insert_function(builder.module, func_ty, sym)
            return builder.call(func_p, [buf, buflen, errnum])
        func_ty = llir.FunctionType(i32, [i32, i8p, size_t_ll])
        func_p = get_or_insert_function(builder.module, func_ty, sym)
        return builder.call(func_p, [errnum, buf, buflen])
    return int32(errnum_ty, buf_ty, buflen_ty), codegen


def _render_ir_for_probe():
    """Render the IR _strerror_safe would emit for a probe call.

    Used by the IR-inspection test (test_strerror_safe.py) to verify
    that when ll.address_of_symbol("__xpg_strerror_r") returns None,
    the chosen symbol is strerror_r and not __xpg_strerror_r. Bypasses
    end-to-end execution: direct text inspection is the safe verification.

    Uses ``intp_ll_type(None)`` — the same shared helper the real codegen
    uses with a ``context``, so both paths derive the size_t LLVM width
    from one place. numba locks intp's lowering to ``IntType(intp.bitwidth)``,
    so the context'd and contextless paths produce identical types.
    """
    module = llir.Module(name="probe")
    i32 = llir.IntType(32)
    i8p = llir.IntType(8).as_pointer()
    sym = _select_posix_symbol()
    func_ty = llir.FunctionType(i32, [i32, i8p, intp_ll_type()])
    get_or_insert_function(module, func_ty, sym)
    return str(module)


@proxy(int32(int64, intp, intp), jit_options={"cache": True})
def strerror_safe(errnum, buf, buflen):
    """Write the error message for errnum into buf (length buflen).

    Returns 0 on success, positive errno (ERANGE on short buffer,
    EINVAL on unknown errnum) on failure. Thread-safe on all supported
    platforms. Cross-platform dispatch happens at lowering time:
    __xpg_strerror_r on glibc, strerror_r on musl / macOS, strerror_s
    on Windows (with arg reorder).
    """
    return _strerror_safe(int32(errnum), buf, buflen)
