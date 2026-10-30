"""A thread-safe message for an error number, for jitted code and Python.

:func:`strerror_safe` writes the message for an error number into a buffer that the caller provides. It calls the
thread-safe form of ``strerror`` that the platform's C library exports, chosen when numba compiles the call:

- **glibc**: `__xpg_strerror_r <https://codebrowser.dev/glibc/glibc/string/xpg-strerror.c.html>`_, the POSIX (XSI)
  form, which glibc exports from version 2.3.4, released in 2004;
- **musl**: also ``__xpg_strerror_r``, which musl exports as a `weak alias
  <https://git.musl-libc.org/cgit/musl/tree/src/string/strerror_r.c>`_ of its ``strerror_r``, itself the POSIX form;
- **macOS**: `strerror_r <https://developer.apple.com/library/archive/documentation/System/Conceptual/
  ManPages_iPhoneOS/man3/strerror_r.3.html>`_, the POSIX form;
- **Windows**: `strerror_s
  <https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/strerror-s-strerror-s-wcserror-s-wcserror-s>`_,
  which takes its arguments in the order buffer, size, error number.

numbox supports no other C library on Linux. glibc's plain ``strerror_r`` is the GNU form, which returns a ``char *``
and does not match the POSIX-shaped call that this module emits. The module therefore calls ``__xpg_strerror_r`` on
Linux whenever LLVM finds it, as it does with glibc and musl, and falls back to ``strerror_r`` only for a C library that
lacks the symbol.
"""
import llvmlite.binding as ll
from llvmlite import ir as llir
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import int32, int64, intp
from numba.extending import intrinsic

from numbox.core.bindings.utils import intp_ll_type, platform_, load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy


__all__ = ["strerror_safe"]


load_lib("c")


def _select_posix_symbol():
    """Return the name of the POSIX-form ``strerror_r`` that the C library exports on Linux or macOS, else None.

    The name is ``__xpg_strerror_r`` on Linux if LLVM finds that symbol, and ``strerror_r`` on Linux otherwise and on
    macOS. Windows and any other platform give None.
    """
    if platform_ == "Linux":
        if ll.address_of_symbol("__xpg_strerror_r") is not None:
            return "__xpg_strerror_r"
        # Neither glibc nor musl reaches this line: glibc exports __xpg_strerror_r from 2.3.4, and musl exports it as a
        # weak alias of its POSIX-form strerror_r. The fallback is for a C library without the symbol. On glibc, plain
        # strerror_r is the GNU form, which returns a char * and does not match the declared i32 result.
        return "strerror_r"
    if platform_ == "Darwin":
        return "strerror_r"
    return None


@intrinsic
def _strerror_safe(typingctx, errnum_ty, buf_ty, buflen_ty):
    """Call the platform's thread-safe ``strerror`` with ``(errnum, buf, buflen)``, and return its ``int32`` result.

    ``errnum`` must be an ``int32``, and numba raises ``TypingError`` for any other type when it compiles the call, as
    it does on a platform that ``_select_posix_symbol`` does not know. On Windows the call is
    ``strerror_s(buf, buflen, errnum)``.
    """
    # strerror_r, __xpg_strerror_r and strerror_s all take errnum as a 32-bit int. The check refuses another type when
    # numba types the call, so a caller that passes an int64, as strerror_safe would without its int32 conversion, gets
    # this TypingError instead of a type mismatch in the generated IR.
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
    """Return the text of an LLVM module that declares the POSIX symbol that ``_strerror_safe`` would call.

    ``test_strerror_safe.py`` makes ``ll.address_of_symbol("__xpg_strerror_r")`` return None and checks that the
    declared symbol is then ``strerror_r``. The test reads the text instead of making the call, because on glibc the
    plain ``strerror_r`` is the GNU form, which the POSIX-shaped declaration does not match. The ``size_t`` parameter
    comes from ``intp_ll_type()``, the helper that the codegen calls with its context, and the two give the same type.
    """
    module = llir.Module(name="probe")
    i32 = llir.IntType(32)
    i8p = llir.IntType(8).as_pointer()
    sym = _select_posix_symbol()
    func_ty = llir.FunctionType(i32, [i32, i8p, intp_ll_type()])
    get_or_insert_function(module, func_ty, sym)
    return str(module)


@proxy(int32(int64, intp, intp), jit_options=jit_options)
def strerror_safe(errnum, buf, buflen):
    """Write the message for the error number ``errnum`` into the buffer at ``buf``, and return 0 or an error number.

    C reference: `strerror_r(3) <https://man7.org/linux/man-pages/man3/strerror_r.3.html>`_.

    ``buf`` is the ``intp`` address of ``buflen`` bytes that the caller keeps alive during the call, and the message
    ends with a NUL. The binding narrows ``errnum`` to ``int32``, the width of C's ``int``. With glibc, an unknown
    number returns ``EINVAL`` (22) and writes ``Unknown error`` followed by the number, and a buffer too short for the
    message returns ``ERANGE`` (34) and writes as much of the message as fits. The function keeps no buffer of its own,
    so threads can call it at the same time. On Linux it calls ``__xpg_strerror_r``, which glibc and musl export, on
    macOS ``strerror_r``, and on Windows ``strerror_s``, with the arguments in that function's order.
    """
    return _strerror_safe(int32(errnum), buf, buflen)
