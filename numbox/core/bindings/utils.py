"""Library loading and small helpers for the binding modules.

:func:`load_lib` and :func:`load_lib_path` load a C library so that LLVM finds its symbols when it links jitted code
into the process. :func:`extract_literal_str` reads the value of a numba string literal, such as a format string or a C
function's name, and :func:`intp_ll_type` returns the LLVM type that numbox uses for ``size_t`` and pointers.
``platform_`` holds the name of the operating system, as ``platform.system()`` returns it.
"""
from ctypes import CDLL
from ctypes.util import find_library
from platform import system

from llvmlite import ir as llir
from numba.core.errors import TypingError
from numba.core.types import Literal, intp


platform_ = system()


def extract_literal_str(binding_name, ty, *, field="argument"):
    """Return the ``str`` value of a numba string literal type.

    Intrinsics and overloads use it for an argument that numba must know when it compiles the call, such as a format
    string, a C function's name or a standard stream's name. It raises ``TypingError`` if ``ty`` is not a literal or if
    its value is not a ``str``, and the message names ``binding_name`` and the argument, ``field``.
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
    """Return the LLVM integer type of numba's ``intp``, which numbox uses for ``size_t`` and pointers.

    With a codegen ``context``, the type is ``context.get_value_type(intp)``, which is how an intrinsic derives a width
    that depends on the platform, such as that of ``size_t``, ``ssize_t`` or ``ptrdiff_t``. Without one, as in a test or
    an IR probe, it is ``llir.IntType(intp.bitwidth)``. numba lowers ``intp`` to an integer of that same width, so both
    calls give the same type.
    """
    if context is not None:
        return context.get_value_type(intp)
    return llir.IntType(intp.bitwidth)


def _windows_bundled_dll_path(name):
    """Return the path of the DLL ``name`` that the Python installation bundles on Windows, or None.

    The function checks three paths in order and returns the first that exists: ``<sys.prefix>/DLLs/<name>.dll``,
    CPython's DLL directory for an installation outside a virtual environment; ``<sys.base_prefix>/DLLs/<name>.dll``,
    the base installation of a virtual environment; and ``<sys.base_prefix>/Library/bin/<name>.dll``, conda's layout.
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
    """Return a path or a name that ``ctypes.CDLL`` can load for the library ``name``, or None if none is found.

    On Linux and macOS the result is ``ctypes.util.find_library(name)``. On Windows, the result for ``"c"`` and ``"m"``
    is ``find_msvcrt()``. For ``"sqlite3"``, the result is only the DLL that Python bundles, from
    ``_windows_bundled_dll_path``. For any other name, the function prefers that DLL to one that ``find_library`` finds
    on the ``PATH``.

    The function prefers the bundled DLL because a directory on the ``PATH`` can hold a copy of a common DLL that
    another tool installs for its own use, and such a copy can crash an outside caller. On GitHub's windows-latest
    runners, AWS CLI v2 puts ``C:\\Program Files\\Amazon\\AWSCLIV2\\sqlite3.dll`` on the ``PATH``. That DLL exports the
    full SQLite API, but it writes through a null pointer inside ``sqlite3_open`` when a process other than ``aws.exe``
    calls it. CPython's bundled ``sqlite3.dll`` is the SQLite that Python's own ``sqlite3`` module uses on Windows.
    """
    if platform_ in ("Darwin", "Linux"):
        return find_library(name)
    if platform_ == "Windows":
        from ctypes.util import find_msvcrt
        if name in ("c", "m"):
            return find_msvcrt()
        bundled = _windows_bundled_dll_path(name)
        if name == "sqlite3":
            # _sqlite3.pyd links to <prefix>/DLLs/sqlite3.dll, so every Python on Windows whose sqlite3 module works
            # has that DLL. find_library is never consulted for sqlite3, because a copy on the PATH can belong to
            # another tool and crash an outside caller, as AWS CLI v2's copy does.
            return bundled
        if bundled is not None:
            return bundled
        return find_library(name)
    return None


_loaded_libs = {}


def load_lib(name):
    """Return the ``ctypes.CDLL`` handle of the library whose short name is ``name``, loading it at the first call.

    The first call loads the library so that LLVM finds its symbols, and later calls with the same name return the same
    handle from the module's cache, ``_loaded_libs``. Each binding module calls it for its own library, and the modules
    share the handle without importing one another. A caller can test for a symbol with ``hasattr(handle, name)``, as
    :func:`~numbox.core.proxy.proxy.proxy_if_available` does. ctypes never unloads a library, so the library stays
    loaded for the life of the process.

    On Linux and macOS, ``ctypes.util.find_library`` finds the file by the short name, such as ``"m"`` for the C math
    library, in the system's library search and not in the Python installation. The file is then opened in global symbol
    mode, ``RTLD_GLOBAL``. On Windows, the function loads the C runtime ``msvcrt.dll`` for ``"c"`` and ``"m"``, and only
    the ``sqlite3.dll`` bundled with Python for ``"sqlite3"``. For any other name, it prefers a DLL bundled with Python
    to one on the ``PATH``.

    If no file is found, the function raises ``RuntimeError: Could not find shared library for <name>``; pass the file's
    path to :func:`load_lib_path` instead. On Linux, ``find_library`` also searches the directories in
    ``LD_LIBRARY_PATH``, but it finds a library there only if the library was built with a SONAME, and the system's
    loader reads the variable only when the process starts. On macOS, ``find_library`` searches the directories in
    ``DYLD_LIBRARY_PATH`` as ``os.environ`` holds the variable at the call.
    """
    handle = _loaded_libs.get(name)
    if handle is None:
        handle = _load_lib_with_handle(name)
        _loaded_libs[name] = handle
    return handle


def _load_lib_with_handle(name):
    """Find the library ``name`` with ``_resolve_lib_path``, open it for LLVM to find its symbols, and return a handle.

    The function raises ``RuntimeError`` if it finds no file, and it does not cache the handle; :func:`load_lib` does.
    """
    path = _resolve_lib_path(name)
    if path is None:
        # find_msvcrt() returns None on a Python built with Visual Studio 2015 or later, as every supported CPython
        # is, so on Windows "c" and "m" load msvcrt.dll through ctypes.cdll.
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
    """Load the library at ``path`` so that LLVM finds its symbols, and return a new ``ctypes.CDLL`` handle.

    ``path`` is any string that ``ctypes.CDLL`` accepts: an absolute path, a soname such as ``libm.so.6``, which
    ``ctypes.util.find_library`` returns, or a file name that the system's loader resolves. The function opens the
    library with ``RTLD_GLOBAL`` on Linux and macOS and with ``winmode=0`` on Windows, and it raises ``RuntimeError``
    on any other platform. Each call returns a new handle; use :func:`load_lib` for a library that several modules
    load by its short name.
    """
    if platform_ in ("Darwin", "Linux"):
        from os import RTLD_GLOBAL
        return CDLL(path, mode=RTLD_GLOBAL)
    if platform_ == "Windows":
        return CDLL(path, winmode=0)
    raise RuntimeError(f"Platform {platform_} is not supported, yet.")
