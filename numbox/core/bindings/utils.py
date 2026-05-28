from ctypes import CDLL
from ctypes.util import find_library
from platform import system

from llvmlite import ir as llir
from numba.core.errors import TypingError
from numba.core.types import Literal, intp


platform_ = system()


def extract_literal_str(binding_name, ty, *, field="argument"):
    """Extract the Python str value of a ``Literal[str]`` type, or raise
    a clean ``TypingError`` naming the binding and the field.

    Used by intrinsics that require a compile-time string (e.g. printf
    format strings, libc function names, stdio handle names). ``field``
    labels the offending argument in the error message.
    """
    if not isinstance(ty, Literal):
        raise TypingError(
            f"{binding_name}: {field} must be a literal str, got {ty!r}"
        )
    val = ty.literal_value
    if not isinstance(val, str):
        raise TypingError(
            f"{binding_name}: {field} must be a Python str, got {type(val).__name__}"
        )
    return val


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


def _windows_bundled_dll_path(name):
    """Best-effort: find a DLL bundled with the Python distribution on Windows.

    Tries (in order):
    - <sys.prefix>/DLLs/<name>.dll (CPython, also catches non-venv installs)
    - <sys.base_prefix>/DLLs/<name>.dll (venv -> base Python)
    - <sys.base_prefix>/Library/bin/<name>.dll (conda layout)

    Returns the absolute path of the first existing candidate, or None if no
    bundled DLL is found.
    """
    import os
    import sys
    dirs = [
        os.path.join(sys.prefix, "DLLs"),
        os.path.join(sys.base_prefix, "DLLs"),
        os.path.join(sys.base_prefix, "Library", "bin"),
    ]
    for d in dirs:
        candidate = os.path.join(d, f"{name}.dll")
        if os.path.exists(candidate):
            return candidate
    return None


def _resolve_lib_path(name):
    """Resolve a library name to a CDLL-loadable path.

    Per-platform logic:
    - Linux / Darwin: ctypes.util.find_library(name)
    - Windows: for "c"/"m", find_msvcrt(); otherwise prefer
      _windows_bundled_dll_path(name) (the Python-distribution-bundled
      DLL in <prefix>/DLLs/ or <prefix>/Library/bin/) over PATH-based
      find_library(name). PATH lookup goes last because PATH on Windows
      may contain third-party-shipped copies of common DLLs that are
      statically configured for that tool's internal use and AV on
      external calls. The motivating example: GitHub Actions
      windows-latest runners ship AWS CLI v2, whose
      C:\\Program Files\\Amazon\\AWSCLIV2\\sqlite3.dll is on PATH,
      exports the full SQLite symbol surface, but writes to NULL inside
      sqlite3_open when called from a process other than aws.exe.
      CPython's bundled sqlite3.dll is the canonical SQLite for the
      Python ecosystem on Windows, so prefer it whenever it's present.

    Returns the path string, or None if no path can be resolved.
    """
    if platform_ in ("Darwin", "Linux"):
        return find_library(name)
    if platform_ == "Windows":
        from ctypes.util import find_msvcrt
        if name in ("c", "m"):
            return find_msvcrt()
        bundled = _windows_bundled_dll_path(name)
        if name == "sqlite3":
            # _sqlite3.pyd is dynamically linked to <prefix>/DLLs/sqlite3.dll,
            # so that DLL is a hard prerequisite of every working Python on
            # Windows. Never fall back to find_library — third-party tools
            # may ship statically-configured sqlite3.dll copies that AV on
            # external callers (AWS CLI v2 is the motivating example).
            return bundled
        if bundled is not None:
            return bundled
        return find_library(name)
    return None


_loaded_libs = {}


def load_lib(name):
    """Load library ``name`` in global symbol mode and return the cached
    CDLL handle. Loads on first call; subsequent calls return the same
    handle from ``_loaded_libs``.

    ``find_library`` is a relatively basic utility that mostly just
    prefixes ``lib`` and suffixes the platform extension. When binding
    user-compiled shared libraries that aren't on the system loader's
    default search path, consider setting ``DYLD_LIBRARY_PATH`` (macOS)
    or ``LD_LIBRARY_PATH`` (Linux) before the first ``load_lib`` call;
    the loader consults those env vars before its built-in search. For
    full control over the path, use :func:`load_lib_path` instead.

    Caching pins the handle for the process lifetime so modules can
    share it without cross-module imports. It also matters because
    ``ctypes.CDLL.__del__`` calls ``dlclose`` / ``FreeLibrary``; without
    retention, the OS-level reference count drops to zero on return and
    the library can be unloaded — invalidating any extern-ref symbols
    LLVM's JIT linker already resolved into module IR. Returning the
    handle also enables ``proxy_if_available`` to query symbol presence
    via ``hasattr(handle, func_name)``.
    """
    handle = _loaded_libs.get(name)
    if handle is None:
        handle = _load_lib_with_handle(name)
        _loaded_libs[name] = handle
    return handle


def _load_lib_with_handle(name):
    """Internal: resolve ``name`` via :func:`_resolve_lib_path` and load
    the resulting library in global symbol mode, returning the CDLL
    handle. Use :func:`load_lib` instead — it caches.
    """
    path = _resolve_lib_path(name)
    if path is None:
        # Preserve the historical Windows c/m fallback (msvcrt via ctypes.cdll).
        if platform_ == "Windows" and name in ("c", "m"):
            import ctypes
            return ctypes.cdll.msvcrt
        raise RuntimeError(f"Could not find shared library for {name}")
    if platform_ in ("Darwin", "Linux"):
        from os import RTLD_GLOBAL
        return CDLL(path, mode=RTLD_GLOBAL)
    if platform_ == "Windows":
        return CDLL(path, winmode=0)
    raise RuntimeError(f"Platform {platform_} is not supported, yet.")


def load_lib_path(path):
    """Load a shared library by ``ctypes.CDLL``-acceptable identifier
    and return the handle (uncached).

    Accepts any string ``CDLL`` accepts — an absolute path, a soname
    (e.g. ``libm.so.6`` as returned by ``ctypes.util.find_library``), or
    a bare filename resolvable by the loader. Linux/Darwin use
    ``RTLD_GLOBAL`` so symbols reach LLVM's JIT; Windows uses
    ``winmode=0``. Prefer :func:`load_lib` when the library is
    referenced by name and across multiple modules in the same process.
    """
    if platform_ in ("Darwin", "Linux"):
        from os import RTLD_GLOBAL
        return CDLL(path, mode=RTLD_GLOBAL)
    if platform_ == "Windows":
        return CDLL(path, winmode=0)
    raise RuntimeError(f"Platform {platform_} is not supported, yet.")
