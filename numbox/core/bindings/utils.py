from ctypes.util import find_library
from platform import system


platform_ = system()


def load_lib(name):
    """ Load library `libname` in global symbol mode.
     `find_library` is a relatively basic utility that
     mostly just prefixes `lib` and suffixes extension.
     When adding (custom) libraries to the global symbol
     scope, consider setting `DYLD_LIBRARY_PATH`."""
    if platform_ in ("Darwin", "Linux"):
        from os import RTLD_GLOBAL
        from ctypes import CDLL

        lib_path = find_library(name)
        _ = CDLL(lib_path, mode=RTLD_GLOBAL)
    elif platform_ == "Windows":
        from ctypes import WinDLL, cdll

        if name in ("c", "m"):
            _ = cdll.msvcrt
        else:
            lib_path = find_library(name)
            if lib_path is None:
                raise RuntimeError(f"Could not find shared library for {name}")
            _ = WinDLL(lib_path, winmode=0)
    else:
        raise RuntimeError(f"Platform {platform_} is not supported, yet.")
