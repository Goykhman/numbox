from ctypes import CDLL
from ctypes.util import find_library
from os import RTLD_GLOBAL


def load_lib(name):
    """ Load library `libname` in global symbol mode.
     `find_library` is a relatively basic utility that
     mostly just prefixes `lib` and suffixes extension.
     When adding (custom) libraries to the global symbol
     scope, consider setting `DYLD_LIBRARY_PATH`."""
    lib_path = find_library(name)
    _ = CDLL(lib_path, mode=RTLD_GLOBAL)
