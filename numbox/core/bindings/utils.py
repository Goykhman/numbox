from ctypes import CDLL
from ctypes.util import find_library
from platform import system

from llvmlite import ir as llir
from numba.core.types import intp


platform_ = system()


def intp_ll_type(context=None):
    """LLVM integer type matching numba's ``intp`` on the current platform.

    Pass the numba codegen ``context`` when available so the type is
    derived via ``context.get_value_type(intp)`` — the canonical pattern
    for platform-dependent widths (size_t, ssize_t, ptrdiff_t) in
    intrinsics. When called outside codegen (test helpers, IR-rendering
    probes), pass ``None`` for the locked-in fallback
    ``llir.IntType(intp.bitwidth)``; numba's intp lowering is locked to
    the same bitwidth, so the two paths produce identical LLVM types.
    """
    if context is not None:
        return context.get_value_type(intp)
    return llir.IntType(intp.bitwidth)


def load_lib(name):
    """ Load library `libname` in global symbol mode.
     `find_library` is a relatively basic utility that
     mostly just prefixes `lib` and suffixes extension.
     When adding (custom) libraries to the global symbol
     scope, consider setting `DYLD_LIBRARY_PATH`."""
    if platform_ in ("Darwin", "Linux"):
        from os import RTLD_GLOBAL

        lib_path = find_library(name)
        _ = CDLL(lib_path, mode=RTLD_GLOBAL)
    elif platform_ == "Windows":
        from ctypes.util import find_msvcrt
        if name in ("c", "m"):
            lib_path = find_msvcrt()
            if lib_path is not None:
                _ = CDLL(lib_path, winmode=0)
            else:
                import ctypes
                _ = ctypes.cdll.msvcrt
        else:
            lib_path = find_library(name)
            if lib_path is None:
                raise RuntimeError(f"Could not find shared library for {name}")
            _ = CDLL(lib_path, winmode=0)
    else:
        raise RuntimeError(f"Platform {platform_} is not supported, yet.")


def load_lib_path(path):
    """Load a shared library by ``ctypes.CDLL``-acceptable identifier.

    Accepts any string ``CDLL`` accepts — an absolute path, a soname
    (e.g. ``libm.so.6`` as returned by ``ctypes.util.find_library``), or
    a bare filename resolvable by the loader. Linux/Darwin use
    ``RTLD_GLOBAL`` so symbols reach LLVM's JIT; Windows uses
    ``winmode=0``. Unlike ``load_lib(name)``, the handle is returned so
    callers can check symbol presence with ``hasattr``.
    """
    if platform_ in ("Darwin", "Linux"):
        from os import RTLD_GLOBAL
        return CDLL(path, mode=RTLD_GLOBAL)
    if platform_ == "Windows":
        return CDLL(path, winmode=0)
    raise RuntimeError(f"Platform {platform_} is not supported, yet.")
