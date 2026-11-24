"""Fingerprints of Python functions and values: texts that describe them and are the same in every process.

numbox hashes fingerprints into the names of the code that it generates and caches, so that a change to a user's
function produces new code. ``_fingerprint_function`` folds a function's module and qualified name, and its code
objects with their bytecode, constants and names. It also folds the function's default values and closure values, the
values of the globals that it reads, and the data and dispatchers that it reads as module attributes. It descends into
the functions and dispatchers among those values, and it stops at a cycle. Two functions with the same source but
different captured values therefore have different fingerprints, whereas a hash of the bare code object would be the
same for both. ``_canon_value`` returns the text for one value and raises ``_Unfingerprintable`` for a value that has
no text that means the same thing in another process. The caller decides how to proceed in that case: some callers
compile the code without a cache, and ``@proxy`` and ``digest`` name the code with
``_fingerprint_function_best_effort``, which never raises. ``make_structref``, ``compile_kernel``, the work builder,
``@proxy`` and ``digest`` use the module, which also resolves the jit options that their names fold.

With ``fold_named``, the walk also folds a function or a class that a module attribute holds. A function that numba
compiles from the function's own content, such as a ``@register_jitable`` function, gets the text that ``_canon_value``
gives it. A function or a class that numba compiles from an implementation of its own, such as ``numpy.sin``, gets the
name that ``_named_value_text`` gives it. The callers that hash the text only into the name of cached code pass
``fold_named``. ``@proxy`` and ``cres`` hash the text into an alias and do not pass it. ``_fingerprint_module_attrs``
gives the reason.

``_type_identity`` and ``_signature_identity`` return the texts for a numba type and a numba signature.
"""
import ctypes
import dis
import hashlib
import os
import sys
import sysconfig

from functools import lru_cache
from importlib.machinery import ModuleSpec
from types import BuiltinFunctionType, CodeType, FunctionType, MethodType, ModuleType
from typing import Any, Optional, Sequence

import numpy as np

from numba.core.dispatcher import Dispatcher
from numba.core.itanium_mangler import mangle_type_or_value
from numba.core.types import ExternalFunction, Type
from numba.core.types.function_type import CompileResultWAP
from numba.core.types.functions import Dispatcher as DispatcherType
from numba.core.typing.templates import Signature
from numba.np.ufunc.dufunc import DUFunc
from numba.np.ufunc.gufunc import GUFunc
from numba.stencils.stencil import StencilFunc

from numbox.core.cache_key import _ADDRESS
from numbox.core.configurations import jit_options as _default_jit_options


class _Unfingerprintable(Exception):
    """The error for a value that has no fingerprint, no text that means the same thing in every process.

    The caller decides how to proceed.
    """


def _safe_repr(obj: object) -> str:
    """Return ``repr(obj)``, or ``<TypeName repr-failed>`` if ``repr`` raises.

    The fallbacks for a value without a fingerprint use it. ``compile_kernel``, for example, names a formula without a
    fingerprint by its ``repr`` and compiles it without a cache; a ``__repr__`` that raises must not make
    ``compile_kernel`` fail.
    """
    try:
        return repr(obj)
    except Exception:  # noqa: BLE001 - fallback must not crash on a raising __repr__
        return f"<{type(obj).__name__} repr-failed>"


def _dtype_key(dtype: np.dtype) -> str:
    """Return a text for the NumPy dtype ``dtype`` that preserves the layout of a structured dtype.

    For a plain dtype the text is ``dtype.str``. ``dtype.str`` gives ``|V<itemsize>`` for every structured dtype and
    drops the names, types, order and offsets of the fields, so two arrays of one size and different layouts would have
    the same text. The text of a structured dtype therefore holds its ``str``, its ``descr`` and its itemsize.

    NumPy defines no ``descr`` for a dtype whose fields overlap or are not in the order of their offsets, such as the
    dtype of ``array[["b", "a"]]``. The ``str`` of such a dtype lists the names, the formats and the offsets of the
    fields, so the text holds a fixed word in place of the ``descr``.
    """
    if dtype.names is not None:
        try:
            descr = dtype.descr
        except ValueError:
            descr = "undefined"
        return f"struct({dtype!s};descr={descr};itemsize={dtype.itemsize})"
    return dtype.str


# The @proxy aliases that no longer identify the body they were created for. numbox.core.proxy.proxy adds them. The
# set is defined here, because proxy imports this module, so that _canon_value can read it without importing proxy.
# It holds both aliases of a collision, the shared alias and the process-local alias assigned to the second body, and
# the alias under which an absent-binding trap is registered.
#
# _canon_value does not fold a retired alias, so that a collision does not make the caches of other bodies wrong. The
# order in which a process builds two colliding bodies determines the alias assigned to each body. A third body that
# captures one of them would fold proxy(<alias>), and in a process that built the pair in the other order, that alias
# names the other body. The colliding bodies are protected, because their own aliases are retired and the cache guard
# discards a cached caller of a retired alias. A body derived from one of them is not protected. It is published under
# a new alias that resolves and is in no set of retired aliases, so its cache entries in other processes can be wrong,
# and the collision would affect every body that captures a colliding one. Because the retired alias is not folded, the
# walk of the derived body reaches the captured dispatcher, which has no fingerprint. The derived body is then compared
# with _opaque_values_match, like any other pair that the fingerprint cannot separate, and its own alias is retired in
# turn.
#
# A derived body fingerprinted before the collision that retires its dependency is not covered. It folded an alias
# that still identified a body, and it is already published under an alias of its own. Covering it would require
# recording the aliases that each fingerprint folded and retiring their dependents transitively.
_ABSENT_ALIASES: set[str] = set()


# repr raises ValueError for an int of more decimal digits than sys.get_int_max_str_digits() allows. A process can set
# that limit as low as 640 digits, and an int of at most 2048 bits has at most 617. _canon_value writes a larger int
# with hex, which has no limit, so that the text of an int does not depend on the limit of the process.
_MAX_REPR_INT_BITS = 2048


def _canon_value(value: Any, seen: set[int], fold_named: bool = False) -> str:
    if isinstance(value, int) and value.bit_length() > _MAX_REPR_INT_BITS:
        return hex(value)
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return repr(value)
    if isinstance(value, np.generic):
        return f"npscalar({_dtype_key(value.dtype)};{value.tobytes().hex()})"
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise _Unfingerprintable(f"object-dtype ndarray {value.dtype.str}")
        data = np.ascontiguousarray(value)
        try:
            raw = hashlib.sha256(data.tobytes()).hexdigest()
        except (ValueError, TypeError) as e:
            raise _Unfingerprintable(f"unhashable ndarray {value.dtype.str}") from e
        return f"ndarray({_dtype_key(data.dtype)};{value.shape};{raw})"
    if isinstance(value, (tuple, list)):
        return f"{type(value).__name__}[" + ",".join(_canon_value(v, seen, fold_named) for v in value) + "]"
    if isinstance(value, (set, frozenset)):
        return f"{type(value).__name__}[" + ",".join(sorted(_canon_value(v, seen, fold_named) for v in value)) + "]"
    if isinstance(value, dict):
        items = sorted((_canon_value(k, seen, fold_named), _canon_value(v, seen, fold_named)) for k, v in value.items())
        return "dict[" + ",".join(f"{k}:{v}" for k, v in items) + "]"
    if isinstance(value, ModuleType):
        return f"module({value.__name__})"
    if isinstance(value, Dispatcher):
        proxy_alias = getattr(value, "_numbox_proxy_alias", None)
        if proxy_alias is not None and proxy_alias not in _ABSENT_ALIASES:
            # A @proxy function: its alias is a hash of the body's fingerprint, the signature and the jit options, so it
            # identifies all three. A walk of the function's wrapper would reach the @intrinsic that the wrapper calls,
            # which has no fingerprint. A retired alias identifies nothing, so the function falls through to that walk.
            return f"proxy({proxy_alias})"
        topts = _canon_value(dict(getattr(value, "targetoptions", {}) or {}), seen)
        declared = _declared_types_text(value) if fold_named else ""
        return f"dispatcher({_fingerprint_function(value.py_func, seen, fold_named)};{topts}{declared})"
    if isinstance(value, DUFunc):
        # A @vectorize function: numba links its compiled loop and compiles in the globals and closure values of the
        # scalar body, so the text folds that body with the options it compiles under.
        wrapped = getattr(value, "__wrapped__", None)
        topts = _canon_value(dict(getattr(value, "targetoptions", {}) or {}), seen)
        body = _fingerprint_function(wrapped, seen, fold_named) if isinstance(wrapped, FunctionType) else _safe_repr(wrapped)
        declared = _declared_types_text(value) if fold_named else ""
        return f"dufunc({body};{topts}{declared})"
    if isinstance(value, FunctionType):
        # A plain Python function. If numba compiles calls of it through an @overload, the text folds the Python
        # function, not the overload's implementation, so an edit to the implementation leaves the text unchanged.
        return f"function({_fingerprint_function(value, seen, fold_named)})"
    if isinstance(value, ExternalFunction):
        # numba's own key for this type, the symbol and the signature. Both are strings and types, so the text is the
        # same in the next process, and it separates two bindings that a factory made for different C symbols.
        return f"externalfunction({value.symbol};{value.sig})"
    if isinstance(value, CompileResultWAP):
        # A first-class function value. numbox tags the two kinds that it creates, cres values and the .as_func of a
        # @proxy function, with what identifies them across processes. The wrapper itself holds only an address and a
        # mangled name with a per-process counter. A value built with numba directly has no tag and falls through to
        # the raise of _Unfingerprintable at the end of the function.
        proxy_alias = getattr(value, "_numbox_proxy_alias", None)
        if proxy_alias is not None and proxy_alias not in _ABSENT_ALIASES:
            return f"proxyfunc({proxy_alias})"
        py_func = getattr(value, "_numbox_py_func", None)
        if py_func is not None:
            return f"cresfunc({_fingerprint_function(py_func, seen, fold_named)};{value.cres.signature})"
    if isinstance(value, ctypes._CFuncPtr):
        return f"cfuncptr({_ctypes_func_key(value)})"
    # Any other value has no fingerprint, among them a builtin function, a ufunc, a numba type, a class and a
    # CompileResultWAP that numbox did not tag.
    raise _Unfingerprintable(type(value).__name__)


# The type of a NumPy function that is not a ufunc, such as numpy.sum. NumPy 1.25 and later define a type for such a
# function. An earlier NumPy defines it as a plain Python function, and this name is then the function type.
_ARRAY_FUNCTION_TYPE = type(np.sum)

# These are the top-level packages whose functions and classes numba compiles from an implementation that numba or
# numbox supplies. numba does not compile the value's own body. numba discards a cache index that another version of
# numba wrote. numbox's version is part of the key of every entry that numbox caches. An entry that one implementation
# compiled is therefore never loaded for another. The standard library is another such package, and
# _in_supplied_package recognizes it by its location.
_SUPPLIED_PACKAGES = frozenset({"numpy", "numba", "numbox"})


def _standard_library_folders() -> frozenset[str]:
    """Return the directories that hold the top-level modules of the interpreter's standard library."""
    folders = set()
    for key in ("stdlib", "platstdlib"):
        root = sysconfig.get_path(key)
        if root:
            root = os.path.normcase(os.path.realpath(root))
            folders.update((root, os.path.join(root, "lib-dynload")))
    return frozenset(folders)


_STANDARD_LIBRARY_FOLDERS = _standard_library_folders()


@lru_cache(maxsize=None)
def _is_standard_library_file(origin: str, top: str) -> bool:
    """Return whether ``origin`` is the file of the standard library's top-level module or package named ``top``."""
    folder, file_name = os.path.split(os.path.normcase(os.path.realpath(origin)))
    if file_name.partition(".")[0] == "__init__" and os.path.basename(folder) == os.path.normcase(top):
        folder = os.path.dirname(folder)
    return folder in _STANDARD_LIBRARY_FOLDERS


def _in_supplied_package(module_name: str) -> bool:
    """Return whether the module named ``module_name`` belongs to NumPy, numba, numbox or the standard library.

    Only the first part of the name is tested. It names the standard library if ``sys.stdlib_module_names`` lists it
    and the loaded module of that name is built in, frozen, or a file in the interpreter's standard-library directory.
    ``sys.stdlib_module_names`` lists the standard library without its test packages and without ``__main__``.

    The name alone is not enough, because a user's module can have the name of a standard-library module, such as
    ``code`` or ``statistics``. Python then loads the user's module under that name. Its file is in another directory,
    so it does not belong to the standard library, and numbox folds its functions by their content. A standard-library
    module that Python loads from another place does not belong either, as in an application that PyInstaller bundled.
    Its Python functions are then folded by their content as well.
    """
    top = module_name.partition(".")[0]
    if top in _SUPPLIED_PACKAGES:
        return True
    if top not in sys.stdlib_module_names:
        return False
    module = sys.modules.get(top)
    spec = vars(module).get("__spec__") if type(module) is ModuleType else None
    origin = spec.origin if type(spec) is ModuleSpec else None
    if type(origin) is not str:
        return False
    return origin in ("built-in", "frozen") or _is_standard_library_file(origin, top)


def _home_of(described: Any) -> Optional[str]:
    """Return the name of the module that is expected to hold ``described`` under the value's own name, or None.

    For a function or a class, that is its ``__module__``. A function of ``numpy.random`` or of ``random``, such as
    ``numpy.random.normal`` or ``random.gauss``, is a method bound to a generator object that the module creates. For
    such a method, that is the module of the class of that object, which is ``numpy.random.mtrand`` or ``random``. The
    same rule applies to a bound builtin method, which records no module itself. The class must have the metaclass
    ``type``, so the lookup runs no code of another metaclass. A ``__module__`` that is not exactly a ``str`` gives
    None.

    The function does not test that the module holds the value. ``_named_value_text`` tests that.
    """
    kind = type(described)
    home = None if kind is MethodType else getattr(described, "__module__", None)
    if home is None and (kind is MethodType or kind is BuiltinFunctionType):
        owner_class = type(described.__self__)
        if type(owner_class) is type:
            home = owner_class.__module__
    return home if type(home) is str else None


def _is_supplied(value: Any) -> bool:
    """Return whether ``value`` is a function or a class of NumPy, numba, numbox or the standard library.

    ``value`` qualifies if it meets two conditions. It is a plain Python function, a method bound to an object, a
    NumPy function that is not a ufunc, or a class whose type is ``type``. ``_in_supplied_package`` accepts the module
    that ``_home_of`` returns for it. ``numpy.sum``, ``numpy.ones``, ``numpy.float32``, ``numba.carray``,
    ``heapq.nlargest``, ``random.gauss`` and ``float`` qualify. numba compiles a call of such a value from an
    implementation that numba or numbox supplies, so the value's own body does not affect the compiled code. A
    function or a class of any other package does not qualify, and neither does a class with another metaclass, such
    as an enum.

    The tests are on the exact type, as in ``_named_value_text``.
    """
    kind = type(value)
    if kind is not FunctionType and kind is not MethodType and kind is not _ARRAY_FUNCTION_TYPE and kind is not type:
        return False
    home = _home_of(value)
    return home is not None and _in_supplied_package(home)


def _named_value_text(value: Any) -> Optional[str]:
    """Return the text that names a function or a class by its module and name, or None.

    numba compiles a call of a NumPy ufunc, a builtin function, or a function or class that ``_is_supplied`` accepts by
    the value's identity, and the module and the name identify the value in every process. A ufunc has a name if NumPy
    holds it under the ufunc's own name, such as ``numpy.sin``. Any other value has one if the module that
    ``_home_of`` returns for it holds it under the value's own name, such as ``math.sqrt``, ``len``, ``numpy.sum``,
    ``numpy.float32`` or ``numpy.random.normal``. The lookup reads the module's ``__dict__`` and compares by identity.
    A ufunc that the user built, a method of an object that the user created, and a value that was replaced in its
    module therefore have no name, and neither has a value of any other type. The tests are on the exact type, so the
    lookup reads no ``__name__`` from an object that only reports one of these types as its ``__class__``, and a
    ``__module__`` of a subclass of ``str`` is not hashed. The lookup likewise accepts only an entry of
    ``sys.modules`` whose type is exactly the module type. A module that ``importlib.util.LazyLoader`` has not loaded
    yet has another type, and reading its ``__dict__`` would run the module.

    A numba dispatcher is named after its Python function, by the same rule, if ``_is_supplied`` accepts that function
    and the module holds the dispatcher under the function's name, as ``numbox.utils.lowlevel`` holds
    ``array_data_p``. ``_fingerprint_module_attrs`` uses that name only for a dispatcher that has no fingerprint.
    """
    if type(value) is np.ufunc:
        if vars(np).get(value.__name__) is value:
            return f"ufunc(numpy.{value.__name__})"
        return None
    described = value.py_func if isinstance(value, Dispatcher) else value
    if type(described) is BuiltinFunctionType:
        tag = "builtin"
    elif _is_supplied(described):
        tag = "class" if type(described) is type else "named"
    else:
        return None
    home, name = _home_of(described), getattr(described, "__name__", None)
    if home is None or type(name) is not str:
        return None
    module = sys.modules.get(home)
    if type(module) is ModuleType and vars(module).get(name) is value:
        return f"{tag}({home}.{name})"
    return None


def _is_held_by_its_own_package(holder: Any, value: Any) -> bool:
    """Return whether the module named ``holder`` belongs to the top-level package of ``value``.

    ``value`` is a function, a class or a dispatcher that ``_named_value_text`` names, or the name of a module. A
    package holds its own functions and classes in the same way in every process that imports it, as ``numpy.add``,
    ``numpy.sum``, ``numpy.strings.add`` and ``math.sqrt`` show, so such an attribute adds nothing to a fingerprint.
    ``_fingerprint_module_attrs`` applies the same test to a submodule that its own package holds, such as
    ``numpy.linalg``, and its docstring gives the reason. The top-level package is the part of a dotted name before the
    first dot, so ``numpy_extras.ops`` is not a module of ``numpy``.
    """
    if type(value) is str:
        home = value
    elif type(value) is np.ufunc:
        home = "numpy"
    else:
        home = _home_of(value.py_func if isinstance(value, Dispatcher) else value)
    return type(holder) is str and home is not None and holder.partition(".")[0] == home.partition(".")[0]


def _process_stable(described: Any) -> Any:
    """Return ``described``, a numba type or a numba signature, or raise ``_Unfingerprintable`` for an unstable text.

    numba's text for some types names an object of the process by its address. The type of a jitclass instance, a
    deferred type and a type that holds a dispatcher are such types. ``_type_identity`` hashes that text, so no other
    process computes the same digest. Code named after such a digest would compile in every process and leave new
    cache files each time. The caller therefore gets no fingerprint and compiles without a cache.
    """
    if _ADDRESS.search(str(described)):
        raise _Unfingerprintable(f"{described} names an object by its address")
    return described


def _held_type_text(value: Any) -> str:
    """Return the text for a numba type or a NumPy dtype object that a module attribute holds.

    numba compiles such a value in as the type that it stands for, as in ``helpers.T(x)`` or
    ``np.empty(n, helpers.DT)``. The text of a numba type holds the digest that ``_type_identity`` returns for it. The
    text of a dtype object holds the text that ``_dtype_key`` returns for it.
    """
    if isinstance(value, np.dtype):
        return f"npdtype({_dtype_key(value)})"
    return f"numbatype({_type_identity(_process_stable(value))[0]})"


def _declared_types_text(value: Any) -> str:
    """Return the text for the types that a dispatcher or a ``@vectorize`` function was declared with, or ``""``.

    numba compiles a call of a dispatcher from the dispatcher's Python function, its options and the types of the
    arguments. Two declarations add types of their own, and both change the code of a caller without changing the
    Python function. The ``locals`` option gives a type to a local variable, and the text holds the type of each
    entry. A dispatcher that was declared with signatures compiles only those, and numba converts the arguments of a
    call to the types of the signature that fits best, so the text holds the signatures of such a dispatcher. A
    ``@vectorize`` function that was declared with signatures has only those loops, and the text holds NumPy's type
    string of each loop, such as ``d->f``.

    A dispatcher or a ``@vectorize`` function that still compiles on demand has the signatures of the calls made so
    far. They differ from process to process, and a call always compiles for its own argument types, so the text
    holds none of them. The text is then empty unless the dispatcher has ``locals``, and the text of a dispatcher
    without either declaration is the one that it had before numbox folded these types.

    Raises ``_Unfingerprintable`` for a declared type that ``_process_stable`` rejects.
    """
    if isinstance(value, DUFunc):
        return ";loops=[" + ",".join(value.types) + "]" if getattr(value, "_frozen", False) else ""
    text = ""
    local_types = getattr(value, "locals", None) or {}
    if local_types:
        entries = (f"{name}:{_type_identity(_process_stable(local_types[name]))[0]}" for name in sorted(local_types))
        text += ";locals=[" + ",".join(entries) + "]"
    if not getattr(value, "_can_compile", True):
        signatures = (_signature_identity(_process_stable(sig))[0] for sig in value.nopython_signatures)
        text += ";signatures=[" + ",".join(signatures) + "]"
    return text


def _ctypes_func_key(value: Any) -> str:
    """Return the identity of a ctypes function pointer: its library's name, its symbol and its call shape.

    A pointer that is reached as an attribute of a loaded library carries the library and the symbol name, and both
    mean the same thing in the next process. The text holds the string with which the library was opened, not the file
    that the loader resolved. Two handles on one library, opened under a soname and under an absolute path, therefore
    give different texts. The difference can cause a recompile but never a wrong result; for a ``@proxy`` body, the
    cache guard discards a cached caller of the old alias when numba loads that caller.

    ``_Unfingerprintable`` is raised for a pointer without a library or a symbol name, because nothing else about it
    means the same thing in the next run. A pointer built from a raw address is the common case: two such pointers
    differ only in an address, which ASLR randomizes in each process. A prototype built over a library, as in
    ``proto(("floor", lib))``, and a Windows lookup by ordinal raise as well, because ctypes names neither. A text that
    held the address would differ in every process, which the fingerprint exists to prevent, so the function raises and
    the caller decides how to treat such a pointer. The call shape comes from :func:`_ctypes_call_shape`.
    """
    name = getattr(value, "__name__", None)
    objects = getattr(value, "_objects", None)
    lib = objects.get("0") if isinstance(objects, dict) else None
    # Only a library has _name. For a Python callback, _objects["0"] is its CThunkObject, which has none; folding None
    # would assign the same text to every callback with the same __name__.
    lib_name = getattr(lib, "_name", None)
    if name is None or not isinstance(lib_name, str) or not lib_name:
        raise _Unfingerprintable("ctypes function pointer with no library and symbol name")
    return f"{lib_name}:{name};{_ctypes_call_shape(value)}"


def _ctypes_call_shape(value: Any) -> str:
    """Return the call shape of a ctypes function pointer: its result type, argument types and calling convention.

    numba types and lowers a call through the pointer from these three, so two pointers that agree on the symbol or
    the address but differ in them describe two calls and compile to two bodies. Every part is a type name or a flag
    word, so the text means the same thing in the next process. :func:`_ctypes_func_key` folds it with the symbol.
    :func:`_opaque_values_match` compares it, with the address, for a pointer that has no process-stable identity, to
    determine whether the same address is called in the same way. An unset ``argtypes`` gives ``<unset>`` and an empty
    one gives an empty list, because numba rejects a call through the first and accepts a call through the second. The
    flags are folded because numba selects the calling convention from them on Windows. A symbol reached through
    ``CDLL`` and the same symbol reached through ``WinDLL`` describe two different calls, and an alias shared by both
    would pair a caller with a body compiled for the wrong convention.
    """
    restype = getattr(getattr(value, "restype", None), "__name__", None)
    argtypes = getattr(value, "argtypes", None)
    args = "<unset>" if argtypes is None else ",".join(getattr(a, "__name__", "?") for a in argtypes)
    return f"{restype}({args});flags={getattr(type(value), '_flags_', None)}"


def _fingerprint_codeobj(code: CodeType, seen: set[int]) -> str:
    # co_consts holds the docstring as well, so an edit to the docstring alone changes the fingerprint.
    consts = ",".join(
        _fingerprint_codeobj(c, seen) if isinstance(c, CodeType) else _canon_value(c, seen)
        for c in code.co_consts
    )
    return (
        f"code({code.co_code.hex()};flags={code.co_flags};argc={code.co_argcount};"
        f"kwonly={code.co_kwonlyargcount};names={','.join(code.co_names)};consts=[{consts}])"
    )


def _referenced_global_names(code: CodeType) -> set[str]:
    names = set(code.co_names)
    for c in code.co_consts:
        if isinstance(c, CodeType):
            names |= _referenced_global_names(c)
    return names


# The opcodes that read, write or delete a name in the global namespace. An attribute access (LOAD_ATTR, LOAD_METHOD,
# STORE_ATTR) and an import use other opcodes, although their names are in co_names too.
_GLOBAL_NAME_OPS = frozenset({
    "LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL",
    "LOAD_NAME", "STORE_NAME", "DELETE_NAME",
})


def _loaded_global_names(code: CodeType) -> set[str]:
    """Return the names that ``code`` and its nested code objects read, write or delete in the global namespace.

    The names come from the opcodes, which tell a ``LOAD_GLOBAL c1`` from a ``LOAD_ATTR c1``. ``co_names``, which
    :func:`_referenced_global_names` reads, also holds attribute names: ``a.c1`` and ``cfg.SCALE`` put ``c1`` and
    ``SCALE`` in ``co_names`` through ``LOAD_ATTR``, though neither is a global. A caller that folded the values of the
    globals named in ``co_names`` would fold a module global that only shares its name with a record field or an
    attribute. That would change the fingerprint when only that global changes or, if the value has no fingerprint,
    make the fingerprint fail; the work builder would then compile the derive without a cache. Module attributes such as
    ``cfg.SCALE`` are still folded, through the full ``co_names``, and ``cfg`` itself is a global here.
    """
    names: set[str] = set()
    for instr in dis.get_instructions(code):
        if instr.opname in _GLOBAL_NAME_OPS and isinstance(instr.argval, str):
            names.add(instr.argval)
    for c in code.co_consts:
        if isinstance(c, CodeType):
            names |= _loaded_global_names(c)
    return names


# The types of the module attributes whose values the fingerprint folds as data. These are the values that numba
# compiles into the code as constants. _fingerprint_module_attrs says what else it folds: a dispatcher, a @vectorize
# function and, with fold_named, the functions and classes that a module attribute holds.
_FROZEN_DATA_TYPES = (bool, int, float, complex, str, bytes,
                      np.generic, np.ndarray, tuple, list, set, frozenset, dict)


def _has_fingerprint(value: Any, seen: set[int]) -> bool:
    """Return True if ``_canon_value`` gives ``value`` a text without ``fold_named``."""
    try:
        _canon_value(value, seen)
    except (_Unfingerprintable, RecursionError):
        return False
    return True


def _fingerprint_module_attrs(referenced: set[str], namespace: dict, seen: set[int],
                              fold_named: bool = False) -> str:
    """Return the text for the module attributes that a function reads, such as ``cfg.SCALE``, or an empty string.

    ``_fingerprint_function`` folds the globals that a function reads, but not a value that it reads through a module,
    as in ``import cfg`` and ``cfg.SCALE``, the chained ``cfg.sub.SCALE``, or a module captured in a closure. The module
    itself gives ``module(<name>)``, and ``SCALE`` is not a global, yet numba compiles its value into the code. Without
    this text, a change of the value would leave the fingerprint unchanged.

    ``referenced`` holds the names in the function's ``co_names`` and in those of its nested code. ``namespace`` maps
    the names through which the function reaches a module to their values: its globals, its closure variables and,
    with ``fold_named``, its default arguments. The function walks each module among those values, and each submodule
    reached through a referenced name. It folds every referenced name of such a module whose value is data or a
    dispatcher: ``None``, a number, a string, bytes, a NumPy scalar or array, a tuple, list, set or dict, a dispatcher
    or a ``@vectorize`` function. Without ``fold_named``, the function leaves out any other value, and any of these
    values that has no fingerprint. A value that a module provides only through a module-level ``__getattr__`` is
    never folded; numba still compiles it in, so the cache must be cleared after it changes.

    With ``fold_named``, the function folds three more kinds of value.

    The first kind is a value that numba compiles from the value's own content: a plain Python function, which numba
    compiles from its body if it is ``@register_jitable``, a ``cres`` value, an ``ExternalFunction`` and a ctypes
    function pointer. Such a value gets the text that ``_canon_value`` gives it when a function reads it as a global.
    If it has no such text, the function raises ``_Unfingerprintable``, as a global read does. A dispatcher and a
    ``@vectorize`` function follow the same rule: without ``fold_named`` one that has no fingerprint is left out, and
    with ``fold_named`` it raises. A ``@guvectorize`` function and a ``@stencil`` function never have a fingerprint.
    Without ``fold_named`` they are left out, and with it they raise. A struct method, a formula or a derive then
    compiles without a cache. Its code is therefore never loaded from a cache after ``helpers.op``, or the function
    that it holds, has changed. ``digest`` cannot disable caching. It computes the text again without ``fold_named``,
    which leaves such a value out, so code named after a digest is still loaded after such a change.

    One kind of dispatcher without a fingerprint does not raise: a jitted function of NumPy, numba, numbox or the
    standard library that ``_named_value_text`` names, such as ``numbox.utils.lowlevel.array_data_p``, which calls an
    ``@intrinsic``. It is folded by that name, as a value of the second kind, so code that calls it through its
    module stays cached.

    The second kind is a value that numba compiles from an implementation of its own: a NumPy ufunc, a builtin
    function, and a function or a class that ``_is_supplied`` accepts, such as ``numpy.sum`` or ``numpy.float32``. It
    is folded by the name that ``_named_value_text`` gives it, as ``helpers.op=ufunc(numpy.sin)``. ``helpers.op`` can
    hold ``numpy.sin`` in one process and ``numpy.cos`` in the next, and without the name both have the same text. In
    the same way, the function folds the name of a module that an attribute holds, which gives
    ``helpers.lib=module(math)``. An attribute of a module in the value's own top-level package, such as ``numpy.add``
    or ``numpy.strings.add``, is not folded. It is the same in every process, and folding it would make the text
    depend on which submodules are loaded. A submodule that its own package holds, such as ``numpy.linalg`` or
    ``numpy.strings``, is not folded either. A package holds a submodule only in a process that has loaded it, so
    folding it would make the text depend on the loaded submodules as well.

    The third kind is a numba type or a NumPy dtype object, such as ``numba.float32`` or ``np.dtype("float32")``.
    numba compiles in the type that it stands for, and the function folds the text that ``_held_type_text`` gives it,
    as ``helpers.T=numbatype(...)``. A module of NumPy, numba, numbox or the standard library holds the same type
    under a name in every process, so ``numba.float64`` read through ``numba`` is not folded. A numba type that
    ``_process_stable`` rejects raises ``_Unfingerprintable``.

    The walk matches names and does not check which attributes the code reads. It folds an attribute of a walked
    module whenever the function's code uses the attribute's name on any object. A function that calls
    ``values.sum()`` while a module that it reads holds a function named ``sum`` folds that function too. The
    fingerprint then changes with a function that the code never calls, and the code compiles without a cache if that
    function has no fingerprint. Neither case loads stale code.

    Some values are left out with ``fold_named`` too. These are an ``@intrinsic``, a class that ``_is_supplied`` does
    not accept, such as an enum or a jitclass, and a function without a name, such as a method of an object that the
    user created. Data without a fingerprint, such as a tuple that holds a numba type, is left out as well, and so is
    a value of any kind that this docstring does not name. A change of such a value leaves the fingerprint unchanged.
    A fingerprint also never holds the implementation of an ``@overload``.

    Data can hold a function, as a tuple of dispatchers does. Such data can have a text without ``fold_named`` and
    none with it, because one of its functions calls a helper without a fingerprint through a module. The function
    raises ``_Unfingerprintable`` for such data, as it does for an attribute that holds that function itself.

    ``compile_kernel``, ``make_structref``, the work builder and ``digest`` pass ``fold_named``, because they hash the
    text only into the name of cached code. The work builder also names the wrapper that it generates for a derive
    after that text, so the alias of such a wrapper changes with that text. ``@proxy`` and ``cres`` hash the text of a
    user's function into an alias and do not pass ``fold_named``. Two bodies that differ only in such an attribute
    then compute one alias, and the rules for a shared alias apply to them. numbox reports a collision if the bodies
    also differ in a value that it can compare, and it does not load a cached caller that holds either function as a
    constant. If each body had an alias of its own, those rules would not apply. numba would then load code for one
    body that it compiled for the other, because its key for that code holds neither the attribute nor the alias.
    """
    attrs: list[str] = []
    mods_seen: set[int] = set()

    def _walk(prefix: str, mod: ModuleType) -> None:
        # A module reached twice, through a cycle of imports, is walked once.
        if id(mod) in mods_seen:
            return
        mods_seen.add(id(mod))
        mod_dict = getattr(mod, "__dict__", {})
        holder = mod_dict["__name__"] if "__name__" in mod_dict else None
        for attr in sorted(referenced):
            # The walk tests membership in the module's __dict__ and never calls getattr. getattr would run a
            # module-level __getattr__ (PEP 562), such as numpy's, for every name of co_names that the module lacks,
            # which imports submodules and emits deprecation warnings.
            if attr not in mod_dict:
                continue
            value = mod_dict[attr]
            if isinstance(value, ModuleType):
                # helpers.lib can hold math in one process and cmath in the next, and the walk of either module folds
                # nothing for lib.sqrt. The walk therefore folds the name of the module, as the fingerprint does for a
                # global that holds a module. It reads the name from the module's __dict__, as it reads every other
                # attribute.
                if fold_named:
                    held_dict = getattr(value, "__dict__", {})
                    held = held_dict["__name__"] if "__name__" in held_dict else None
                    if type(held) is str and not _is_held_by_its_own_package(holder, held):
                        attrs.append(f"{prefix}.{attr}=module({held})")
                _walk(f"{prefix}.{attr}", value)
                continue
            # The tests are identity tests. A membership test would compare the types with ==, which calls the
            # __eq__ of a metaclass.
            if fold_named and (type(value) is np.ufunc or type(value) is BuiltinFunctionType or _is_supplied(value)):
                named = _named_value_text(value)
                if named is not None and not _is_held_by_its_own_package(holder, value):
                    attrs.append(f"{prefix}.{attr}={named}")
                continue
            if fold_named and (type(value) is FunctionType or isinstance(
                    value, (Dispatcher, DUFunc, GUFunc, StencilFunc, CompileResultWAP, ExternalFunction, ctypes._CFuncPtr))):
                # numba compiles a call of such a value from the value's own content, so the walk folds the value as
                # a global read does. The walk raises for a value without a fingerprint. A jitted function that
                # numbox or numba supplies is the exception: it has a name.
                try:
                    canon = _canon_value(value, seen, fold_named)
                except (_Unfingerprintable, RecursionError):
                    canon = _named_value_text(value) if isinstance(value, Dispatcher) else None
                    if canon is None:
                        raise
                    if _is_held_by_its_own_package(holder, value):
                        continue
                attrs.append(f"{prefix}.{attr}={canon}")
                continue
            if fold_named and isinstance(value, (Type, np.dtype)):
                # numba compiles in the type that such a value stands for. A module of NumPy, numba, numbox or the
                # standard library holds the same type under a name in every process, so the walk folds only a type
                # that another module holds.
                if type(holder) is not str or not _in_supplied_package(holder):
                    attrs.append(f"{prefix}.{attr}={_held_type_text(value)}")
                continue
            if not (value is None or isinstance(value, _FROZEN_DATA_TYPES)
                    or isinstance(value, (Dispatcher, DUFunc))):
                continue
            try:
                canon = _canon_value(value, seen, fold_named)
            except (_Unfingerprintable, RecursionError):
                # Data without a fingerprint, such as a tuple that holds a numba type, is left out. Data that has one
                # without fold_named holds a function that only fold_named leaves without a fingerprint, such as a
                # dispatcher that calls a helper without one through a module. The walk raises for such data, as it
                # does for an attribute that holds that function itself.
                if fold_named and _has_fingerprint(value, seen):
                    raise
                continue
            attrs.append(f"{prefix}.{attr}={canon}")

    for name in sorted(namespace):
        val = namespace[name]
        if isinstance(val, ModuleType):
            _walk(name, val)
    return f";module_attrs=[{';'.join(sorted(attrs))}]" if attrs else ""


def _fingerprint_function(func: FunctionType, seen: set[int], fold_named: bool = False) -> str:
    # A function that the walk has already entered, through recursion or a cycle, is folded by its qualified name.
    if id(func) in seen:
        return f"recursive({func.__qualname__})"
    seen = seen | {id(func)}
    code = func.__code__
    cells = []
    closure_vals = {}
    for name, cell in zip(code.co_freevars, func.__closure__ or ()):
        try:
            contents = cell.cell_contents
        except ValueError as e:
            raise _Unfingerprintable("empty closure cell") from e
        cells.append(f"{name}={_canon_value(contents, seen, fold_named)}")
        closure_vals[name] = contents
    # The values of globals are folded for the names that the opcodes read as globals. Module attributes are looked up
    # for every name in co_names, which includes attribute names such as SCALE in cfg.SCALE.
    referenced = _referenced_global_names(code)
    global_names = _loaded_global_names(code)
    hashed_globals = []
    for name in sorted(global_names):
        if name in func.__globals__:
            hashed_globals.append(f"{name}={_canon_value(func.__globals__[name], seen, fold_named)}")
    modattr_ns = {n: func.__globals__[n] for n in referenced if n in func.__globals__}
    if fold_named:
        # The function reads a module that it takes as a default argument through the parameter's name, as m.K in
        # "def f(y, m=helpers)". A parameter hides a global of the same name, so its value replaces the global's.
        defaults = func.__defaults__ or ()
        modattr_ns.update(zip(code.co_varnames[code.co_argcount - len(defaults):code.co_argcount], defaults))
        modattr_ns.update(func.__kwdefaults__ or {})
    modattr_ns.update(closure_vals)
    module_attrs = _fingerprint_module_attrs(referenced, modattr_ns, seen, fold_named)
    return (
        f"func({func.__module__}:{func.__qualname__};{_fingerprint_codeobj(code, seen)};"
        f"defaults={_canon_value(func.__defaults__ or (), seen, fold_named)};"
        f"kwdefaults={_canon_value(func.__kwdefaults__ or {}, seen, fold_named)};"
        f"closure=[{';'.join(cells)}];globals=[{';'.join(hashed_globals)}]{module_attrs})"
    )


def _opaque_values_match(old: Sequence[Any], new: Sequence[Any]) -> bool:
    """Return whether two lists of values without a fingerprint hold the same values, pair by pair, in this process.

    The lists come from two calls of :func:`_fingerprint_function_best_effort` that gave the same fingerprint. The two
    bodies then differ, if at all, only in these values, so equal values mean one body, and different values mean two
    bodies that the fingerprint cannot separate. Two values match if they are the same object, if they compare equal,
    or, for two ctypes function pointers, if they hold the same address and have the same call shape. The comparison
    has to hold only within one process, so unlike the fingerprint it may compare addresses.
    """
    if len(old) != len(new):
        return False
    for a, b in zip(old, new):
        # The two values are usually the same object: a reloaded module binds the same @intrinsic object again, and a
        # factory called twice captures the same object. Equality is tried next, for values that define it.
        if a is b:
            continue
        # A ctypes function pointer defines no equality beyond identity, so the address that it holds is compared
        # together with its call shape. The address alone does not identify the body: one symbol reached through two
        # prototypes describes two calls that numba types and lowers differently. If the two pointers were taken as one
        # value, the second body would receive the alias of the first body, and no collision would be reported.
        # _ctypes_call_shape is also what the fingerprint folds for a pointer that it can name, so both paths separate
        # two pointers by the same rule.
        if isinstance(a, ctypes._CFuncPtr) and isinstance(b, ctypes._CFuncPtr):
            if (ctypes.cast(a, ctypes.c_void_p).value == ctypes.cast(b, ctypes.c_void_p).value
                    and _ctypes_call_shape(a) == _ctypes_call_shape(b)):
                continue
            return False
        try:
            if bool(a == b):
                continue
        except Exception:  # nosec B112 - an object that cannot answer == is not a match
            pass
        return False
    return True


def _fingerprint_function_best_effort(func: FunctionType, opaque: Optional[list] = None) -> str:
    """Return a fingerprint of ``func`` that never raises: a placeholder stands for each value without a fingerprint.

    ``_fingerprint_function`` raises at the first value without a fingerprint, so a body that only reads such a
    global, such as an ``@intrinsic``, has no fingerprint. This function substitutes ``<opaque:TypeName>`` for each
    such value. Bodies that differ in a constant, a closure value, a default or a global that has a fingerprint
    therefore still have different fingerprints. Every placeholder comes from a type name, never from an address, so
    the result is the same in every process. Unlike ``_fingerprint_function``, it omits the module attributes that
    the function reads, so a change to one does not change the result.

    ``@proxy`` names a body without a fingerprint with this function, and ``digest`` hashes such a function with it.
    ``make_structref`` and the work builder use it too, but they compile such code without a cache. Two different
    bodies that differ only in values without a fingerprint have the same result. If ``opaque`` is a list, the values
    replaced by a placeholder are appended to it in the order of the walk. A caller with two equal fingerprints can then
    call :func:`_opaque_values_match` to determine whether they come from one body or from two that the fingerprint
    cannot separate.
    """
    seen: set[int] = set()

    def _canon(value: Any) -> str:
        try:
            return _canon_value(value, seen)
        except (_Unfingerprintable, RecursionError):
            if opaque is not None:
                opaque.append(value)
            return f"<opaque:{type(value).__name__}>"

    def _canon_const(const: object) -> str:
        if isinstance(const, CodeType):
            inner = ",".join(_canon_const(c) for c in const.co_consts)
            return f"code({const.co_code.hex()};consts=[{inner}])"
        return _canon(const)

    code = func.__code__
    consts = ",".join(_canon_const(c) for c in code.co_consts)
    cells = []
    for name, cell in zip(code.co_freevars, func.__closure__ or ()):
        try:
            contents = cell.cell_contents
        except ValueError:
            cells.append(f"{name}=<empty-cell>")
            continue
        cells.append(f"{name}={_canon(contents)}")
    hashed_globals = [
        f"{name}={_canon(func.__globals__[name])}"
        for name in sorted(_loaded_global_names(code))
        if name in func.__globals__
    ]
    # Unlike _fingerprint_function, this walk does not call _fingerprint_module_attrs, so the module attributes that
    # the function reads are not folded. If digest uses this fingerprint for the callback of a SQLite registration, the
    # name of the registration's anchor does not change when such an attribute changes, and the registration loads the
    # code compiled before the change.
    return (
        f"besteffort({func.__module__}:{func.__qualname__};"
        f"code({code.co_code.hex()};flags={code.co_flags};argc={code.co_argcount};"
        f"kwonly={code.co_kwonlyargcount};names={','.join(code.co_names)};consts=[{consts}]);"
        f"defaults={_canon(func.__defaults__ or ())};"
        f"kwdefaults={_canon(func.__kwdefaults__ or {})};"
        f"closure=[{';'.join(cells)}];globals=[{';'.join(hashed_globals)}])"
    )


# ---- numba types and signatures ----

def _type_identity(ty: Type) -> tuple[str, bool]:
    """Return ``(digest, cacheable)`` for the numba type ``ty``: a SHA-256 that is the same in every process, and
    whether code typed with ``ty`` may be cached.

    A dispatcher type, or a container type that holds one, is not cacheable. A Record type is hashed by its ``str``, a
    dispatcher type by the fingerprint of its Python function, and any other type by numba's mangled name for it.
    """
    cacheable = True
    ty_str = str(ty)
    if isinstance(ty, DispatcherType):
        # A dispatcher type holds the address of its Python function, which differs from one process to the next. A
        # signature that holds one misses numba's on-disk index in every process, and numba appends a new data entry
        # each time, so the cache files grow without bound.
        cacheable = False
        try:
            body = "dispatcher(" + _fingerprint_function(ty.dispatcher.py_func, set()) + ")"
        except (_Unfingerprintable, RecursionError):
            body = "dispatcher-besteffort(" + _fingerprint_function_best_effort(ty.dispatcher.py_func) + ")"
    elif "Record(" in ty_str:
        # numba mangles a Record type with its _code, a counter of the order in which the process creates types, so
        # the mangled name differs between a fresh process and one that has created other types first. The str of a
        # Record describes its fields, so the str is hashed instead.
        body = "structural(" + ty_str + ")"
    else:
        body = mangle_type_or_value(ty)
    if cacheable and "CPUDispatcher(" in ty_str:
        # A dispatcher inside a container type, such as a tuple, makes the type uncacheable as well.
        cacheable = False
    return hashlib.sha256(body.encode("utf-8")).hexdigest(), cacheable


def _signature_identity(sig: Signature) -> tuple[str, bool]:
    """Return ``(text, cacheable)`` for the numba signature ``sig``: the digests of its return and argument types, and
    whether all of them are cacheable, as :func:`_type_identity` returns them."""
    ret_hash, ret_ok = _type_identity(sig.return_type)
    arg_ids = [_type_identity(a) for a in sig.args]
    text = ret_hash + "(" + ",".join(h for h, _ok in arg_ids) + ")"
    cacheable = ret_ok and all(ok for _h, ok in arg_ids)
    return text, cacheable


# ---- Effective jit flags ----
#
# The compile_kernel digest and the make_graph kernel fingerprint fold the jit options, so that a change of an option
# changes the name of the compiled code. The two functions below are defined here rather than in compile_kernel, so that
# builder can use them without importing compile_kernel, whose import sweeps orphaned anchors. The NUMBA_* settings
# that those names fold come from numbox.core.cache_key.

def _effective_flags(jit_options: dict | None) -> dict:
    """Return numbox's default jit options updated with ``jit_options``, without ``cache``.

    ``cache`` determines only whether the code is written to disk, not what numba emits, so a name that folded it would
    give identical binaries different names. Every other option, such as ``fastmath``, ``error_model`` or
    ``boundscheck``, can change the code and is kept.
    """
    opts = {**_default_jit_options, **(jit_options or {})}
    return {k: v for k, v in opts.items() if k != "cache"}


def _flags_canon(flags: dict) -> tuple[str, bool]:
    """Return ``(text, canonical)`` for the jit options ``flags``.

    ``canonical`` is False if the value of an option has no fingerprint, such as a ``pipeline_class`` or ``locals``
    that hold numba types; the text is then the ``repr`` of the sorted items. The caller must then compile without a
    cache, because a name that does not include an option cannot distinguish the binaries that the option produces.
    """
    try:
        return _canon_value(flags, set()), True
    except (_Unfingerprintable, RecursionError):
        return repr(sorted(flags.items(), key=repr)), False
