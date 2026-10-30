"""Fingerprints of Python functions and values: texts that describe them and are the same in every process.

numbox hashes fingerprints into the names of the code that it generates and caches, so that a change to a user's
function produces new code. ``_fingerprint_function`` folds a function's module and qualified name, and its code
objects with their bytecode, constants and names. It also folds the function's default values and closure values, the
values of the globals that it reads, and the data and dispatchers that it reads as module attributes. It descends into
the functions and dispatchers among those values, and it stops at a cycle. Two functions with the same source but
different captured values therefore get different fingerprints, which a hash of the bare code object would not give.
``_canon_value`` gives the text for one value and raises ``_Unfingerprintable`` for a value that has no text that
means the same thing in another process. Each caller decides what to do then: some compile the code without a cache,
and ``@proxy`` and ``digest`` name it with ``_fingerprint_function_best_effort``, which never raises.
``make_structref``, ``compile_kernel``, the work builder, ``@proxy`` and ``digest`` use the module, which also
resolves the jit options that their names fold.
"""
import ctypes
import dis
import hashlib

from types import CodeType, FunctionType, ModuleType
from typing import Any, Optional, Sequence

import numpy as np

from numba.core.dispatcher import Dispatcher
from numba.core.types import ExternalFunction
from numba.core.types.function_type import CompileResultWAP
from numba.np.ufunc.dufunc import DUFunc

from numbox.core.configurations import jit_options as _default_jit_options


class _Unfingerprintable(Exception):
    """The error for a value that has no fingerprint, no text that means the same thing in every process.

    The caller decides what to do then.
    """


def _safe_repr(obj: object) -> str:
    """Return ``repr(obj)``, or ``<TypeName repr-failed>`` if ``repr`` raises.

    The fallbacks for a value without a fingerprint use it. ``compile_kernel``, for example, names a formula without a
    fingerprint by its ``repr`` and compiles it without a cache; a ``__repr__`` that raises must not make that fail.
    """
    try:
        return repr(obj)
    except Exception:  # noqa: BLE001 - fallback must not crash on a raising __repr__
        return f"<{type(obj).__name__} repr-failed>"


def _dtype_key(dtype: np.dtype) -> str:
    """Return a text for the NumPy dtype ``dtype`` that keeps the layout of a structured dtype.

    For a plain dtype the text is ``dtype.str``. ``dtype.str`` gives ``|V<itemsize>`` for every structured dtype and
    drops the names, types, order and offsets of the fields, so two arrays of one size and different layouts would get
    one text. The text of a structured dtype therefore holds its ``str``, its ``descr`` and its itemsize.
    """
    if dtype.names is not None:
        return f"struct({dtype!s};descr={dtype.descr};itemsize={dtype.itemsize})"
    return dtype.str


# The @proxy aliases that no longer identify the body they were created for. numbox.core.proxy.proxy adds them. The
# set is defined here, because proxy imports this module, so that _canon_value can read it without importing proxy.
# It holds both aliases of a collision, the shared alias and the process-local one that the second body gets, and the
# alias that an absent-binding trap answers to.
#
# _canon_value does not fold a retired alias, and that keeps a collision from spreading. The order in which a process
# builds two colliding bodies decides the alias that each body gets. A third body that captures one of them would fold
# proxy(<alias>), and in a process that built the pair in the other order, that alias names the other body. The
# colliding bodies are protected, because their own aliases are retired and the cache guard discards a cached caller
# of a retired alias. A body derived from one of them is not. It is published under a new alias that resolves and is
# in no set of retired aliases, so its cache entries in other processes can be wrong, and the collision would affect
# every body that captures a colliding one. Because the retired alias is not folded, the walk of the derived body
# reaches the captured dispatcher, which has no fingerprint. The derived body is then compared with
# _opaque_values_match, like any other pair that the fingerprint cannot separate, and its own alias is retired in turn.
#
# A derived body fingerprinted before the collision that retires its dependency is not covered. It folded an alias
# that still identified a body, and it is already published under an alias of its own. Covering it would require
# recording the aliases that each fingerprint folded and retiring their dependents transitively.
_ABSENT_ALIASES: set[str] = set()


def _canon_value(value: Any, seen: set[int]) -> str:
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
        return f"{type(value).__name__}[" + ",".join(_canon_value(v, seen) for v in value) + "]"
    if isinstance(value, (set, frozenset)):
        return f"{type(value).__name__}[" + ",".join(sorted(_canon_value(v, seen) for v in value)) + "]"
    if isinstance(value, dict):
        items = sorted((_canon_value(k, seen), _canon_value(v, seen)) for k, v in value.items())
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
        return f"dispatcher({_fingerprint_function(value.py_func, seen)};{topts})"
    if isinstance(value, DUFunc):
        # A @vectorize function: numba links its compiled loop and compiles in the globals and closure values of the
        # scalar body, so the text folds that body with the options it compiles under.
        wrapped = getattr(value, "__wrapped__", None)
        topts = _canon_value(dict(getattr(value, "targetoptions", {}) or {}), seen)
        body = _fingerprint_function(wrapped, seen) if isinstance(wrapped, FunctionType) else _safe_repr(wrapped)
        return f"dufunc({body};{topts})"
    if isinstance(value, FunctionType):
        # A plain Python function. If numba compiles calls of it through an @overload, the text folds the Python
        # function, not the overload's implementation, so an edit to the implementation leaves the text unchanged.
        return f"function({_fingerprint_function(value, seen)})"
    if isinstance(value, ExternalFunction):
        # numba's own key for this type, the symbol and the signature. Both are strings and types, so the text is the
        # same in the next process, and it separates two bindings that a factory made for different C symbols.
        return f"externalfunction({value.symbol};{value.sig})"
    if isinstance(value, CompileResultWAP):
        # A first-class function value. numbox tags the two kinds that it creates, cres values and the .as_func of a
        # @proxy function, with what identifies them across processes. The wrapper itself holds only an address and a
        # mangled name with a per-process counter. A value built with numba directly has no tag and falls through to
        # the _Unfingerprintable at the end.
        proxy_alias = getattr(value, "_numbox_proxy_alias", None)
        if proxy_alias is not None and proxy_alias not in _ABSENT_ALIASES:
            return f"proxyfunc({proxy_alias})"
        py_func = getattr(value, "_numbox_py_func", None)
        if py_func is not None:
            return f"cresfunc({_fingerprint_function(py_func, seen)};{value.cres.signature})"
    if isinstance(value, ctypes._CFuncPtr):
        return f"cfuncptr({_ctypes_func_key(value)})"
    # Any other value has no fingerprint, among them a builtin function, a ufunc, a numba type, a class and a
    # CompileResultWAP that numbox did not tag.
    raise _Unfingerprintable(type(value).__name__)


def _ctypes_func_key(value: Any) -> str:
    """Return the identity of a ctypes function pointer: its library's name, its symbol and its call shape.

    A pointer that is reached as an attribute of a loaded library carries the library and the symbol name, and both
    mean the same thing in the next process. The text holds the string with which the library was opened, not the file
    that the loader resolved. Two handles on one library, opened under a soname and under an absolute path, therefore
    give different texts. The difference costs at most a recompile, never a wrong result; for a ``@proxy`` body, the
    cache guard discards a cached caller of the old alias when it loads.

    ``_Unfingerprintable`` is raised for a pointer without a library or a symbol name, because nothing else about it
    means the same thing in the next run. A pointer built from a raw address is the common case: two of them differ
    only by an address that ASLR moves. A prototype built over a library, as in ``proto(("floor", lib))``, and a
    Windows lookup by ordinal raise as well, because ctypes names neither. A text with the address would differ in
    every process, which the fingerprint exists to avoid, so the caller decides what to do with such a pointer. The
    call shape comes from :func:`_ctypes_call_shape`.
    """
    name = getattr(value, "__name__", None)
    objects = getattr(value, "_objects", None)
    lib = objects.get("0") if isinstance(objects, dict) else None
    # Only a library has _name. For a Python callback, _objects["0"] is its CThunkObject, which has none; folding None
    # would give every callback with the same __name__ the same text.
    lib_name = getattr(lib, "_name", None)
    if name is None or not isinstance(lib_name, str) or not lib_name:
        raise _Unfingerprintable("ctypes function pointer with no library and symbol name")
    return f"{lib_name}:{name};{_ctypes_call_shape(value)}"


def _ctypes_call_shape(value: Any) -> str:
    """Return the call shape of a ctypes function pointer: its result type, argument types and calling convention.

    numba types and lowers a call through the pointer from these three, so two pointers that agree on the symbol or
    the address but differ in them describe two calls and compile to two bodies. Every part is a type name or a flag
    word, so the text means the same thing in the next process. :func:`_ctypes_func_key` folds it with the symbol.
    :func:`_opaque_values_match` compares it, with the address, for a pointer that has no process-stable identity;
    there the shape says whether the same address is called the same way. An unset ``argtypes`` gives ``<unset>`` and
    an empty one an empty list, because numba rejects a call through the first and accepts the second. The flags are
    folded because numba selects the calling convention from them on Windows. A symbol reached through ``CDLL`` and the
    same symbol reached through ``WinDLL`` are two calls, and one alias for both would pair a caller with a body of the
    wrong convention.
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
    attribute. That would change the fingerprint for no reason or, if the value has no fingerprint, make the
    fingerprint fail; the work builder would then compile the derive without a cache. Module attributes such as
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


# The types of the module attributes whose values the fingerprint folds. These are the data that numba compiles into
# the code as constants. A dispatcher attribute is folded as well, because numba links the dispatcher's code and
# compiles in its globals and closure values, as for a dispatcher read as a global. numba resolves a ufunc, a plain
# function or a type by identity or by its own lowering, so the fingerprint leaves those out. A @register_jitable
# helper is a plain function, and an @intrinsic is neither data nor a dispatcher. An edit to either, called as
# module.helper(), therefore leaves the fingerprint unchanged.
_FROZEN_DATA_TYPES = (bool, int, float, complex, str, bytes,
                      np.generic, np.ndarray, tuple, list, set, frozenset, dict)


def _fingerprint_module_attrs(referenced: set[str], namespace: dict, seen: set[int]) -> str:
    """Return the text for the module attributes that a function reads, such as ``cfg.SCALE``, or an empty string.

    ``_fingerprint_function`` folds the globals that a function reads, but not a value that it reads through a module,
    as in ``import cfg`` and ``cfg.SCALE``, the chained ``cfg.sub.SCALE``, or a module captured in a closure. The module
    itself gives ``module(<name>)``, and ``SCALE`` is not a global, yet numba compiles its value into the code. Without
    this text, a change of the value would leave the fingerprint unchanged.

    ``referenced`` holds the names in the function's ``co_names`` and in those of its nested code. ``namespace`` maps
    the function's globals and closure variables to their values. The function walks each module among those values,
    and each submodule reached through a referenced name. It folds every referenced name of such a module whose value is
    data or a dispatcher: ``None``, a number, a string, bytes, a NumPy scalar or array, a tuple, list, set or dict, a
    dispatcher or a ``@vectorize`` function. It leaves out any other value, such as a plain function, a ufunc or a
    class, and any value without a fingerprint. A formula that reads such a value stays cacheable, and the walk does
    not descend into a package such as numpy. A value that a module provides only through a module-level
    ``__getattr__`` is not folded; numba still compiles it in, so the cache must be cleared after it changes.
    """
    attrs: list[str] = []
    mods_seen: set[int] = set()

    def _walk(prefix: str, mod: ModuleType) -> None:
        # A module reached twice, through a cycle of imports, is walked once.
        if id(mod) in mods_seen:
            return
        mods_seen.add(id(mod))
        mod_dict = getattr(mod, "__dict__", {})
        for attr in sorted(referenced):
            # The walk tests membership in the module's __dict__ and never calls getattr. getattr would run a
            # module-level __getattr__ (PEP 562), such as numpy's, for every name of co_names that the module lacks,
            # which imports submodules and emits deprecation warnings.
            if attr not in mod_dict:
                continue
            value = mod_dict[attr]
            if isinstance(value, ModuleType):
                _walk(f"{prefix}.{attr}", value)
                continue
            if not (value is None or isinstance(value, _FROZEN_DATA_TYPES)
                    or isinstance(value, (Dispatcher, DUFunc))):
                continue
            try:
                canon = _canon_value(value, seen)
            except (_Unfingerprintable, RecursionError):
                continue
            attrs.append(f"{prefix}.{attr}={canon}")

    for name in sorted(namespace):
        val = namespace[name]
        if isinstance(val, ModuleType):
            _walk(name, val)
    return f";module_attrs=[{';'.join(sorted(attrs))}]" if attrs else ""


def _fingerprint_function(func: FunctionType, seen: set[int]) -> str:
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
        cells.append(f"{name}={_canon_value(contents, seen)}")
        closure_vals[name] = contents
    # The values of globals are folded for the names that the opcodes read as globals. Module attributes are looked up
    # for every name in co_names, which includes attribute names such as SCALE in cfg.SCALE.
    referenced = _referenced_global_names(code)
    global_names = _loaded_global_names(code)
    hashed_globals = []
    for name in sorted(global_names):
        if name in func.__globals__:
            hashed_globals.append(f"{name}={_canon_value(func.__globals__[name], seen)}")
    modattr_ns = {n: func.__globals__[n] for n in referenced if n in func.__globals__}
    modattr_ns.update(closure_vals)
    module_attrs = _fingerprint_module_attrs(referenced, modattr_ns, seen)
    return (
        f"func({func.__module__}:{func.__qualname__};{_fingerprint_codeobj(code, seen)};"
        f"defaults={_canon_value(func.__defaults__ or (), seen)};"
        f"kwdefaults={_canon_value(func.__kwdefaults__ or {}, seen)};"
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
        # Identity decides the usual case: a reloaded module binds the same @intrinsic object again, and a factory
        # called twice captures the same one. Equality is tried next, for values that define it.
        if a is b:
            continue
        # A ctypes function pointer defines neither, so the address it holds is compared together with its call shape.
        # The address alone does not identify the body: one symbol reached through two prototypes is two calls that
        # numba types and lowers differently. Taken as one value, the second body would silently get the alias under
        # which the first was published, and no collision would be reported. _ctypes_call_shape is also what the
        # fingerprint folds for a pointer that it can name, so both paths agree on what makes two pointers two calls.
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
    global, such as an ``@intrinsic``, gets no fingerprint at all. This function puts ``<opaque:TypeName>`` in place of
    each such value instead. Bodies that differ in a constant, a closure value, a default or a global that has a
    fingerprint therefore still get different fingerprints. Every placeholder comes from a type name, never from an
    address, so the result is the same in every process. Unlike ``_fingerprint_function``, it leaves out the module
    attributes that the function reads, so a change to one leaves the result unchanged.

    ``@proxy`` names a body without a fingerprint with this function, and ``digest`` hashes such a function with it.
    ``make_structref`` and the work builder use it too, but they compile such code without a cache. A placeholder is
    where two different bodies can still get one fingerprint. If ``opaque`` is a list, the values that got a
    placeholder are appended to it in the order of the walk. A caller with two equal fingerprints can then ask
    :func:`_opaque_values_match` whether they come from one body or from two that the fingerprint cannot separate.
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
    # the function reads are not folded. A SQLite registration whose callback gets this fingerprint from digest
    # therefore keeps its anchor's name after such an attribute changes, and it loads the code compiled before.
    return (
        f"besteffort({func.__module__}:{func.__qualname__};"
        f"code({code.co_code.hex()};flags={code.co_flags};argc={code.co_argcount};"
        f"kwonly={code.co_kwonlyargcount};names={','.join(code.co_names)};consts=[{consts}]);"
        f"defaults={_canon(func.__defaults__ or ())};"
        f"kwdefaults={_canon(func.__kwdefaults__ or {})};"
        f"closure=[{';'.join(cells)}];globals=[{';'.join(hashed_globals)}])"
    )


# ---- Effective jit flags ----
#
# The compile_kernel digest and the make_graph kernel fingerprint fold the jit options, so that a change of an option
# changes the name of the compiled code. The two functions below live here rather than in compile_kernel, so that
# builder can use them without importing compile_kernel, whose import sweeps orphaned anchors. The NUMBA_* settings
# that those names fold come from numbox.core.cache_key.

def _effective_flags(jit_options: dict | None) -> dict:
    """Return numbox's default jit options updated with ``jit_options``, without ``cache``.

    ``cache`` decides only whether the code is written to disk, not what numba emits, so a name that folded it would
    separate identical binaries. Every other option, such as ``fastmath``, ``error_model`` or ``boundscheck``, can
    change the code and is kept.
    """
    opts = {**_default_jit_options, **(jit_options or {})}
    return {k: v for k, v in opts.items() if k != "cache"}


def _flags_canon(flags: dict) -> tuple[str, bool]:
    """Return ``(text, canonical)`` for the jit options ``flags``.

    ``canonical`` is False if the value of an option has no fingerprint, such as a ``pipeline_class`` or ``locals``
    that hold numba types; the text is then the ``repr`` of the sorted items. The caller must then compile without a
    cache, because a name that cannot see an option cannot keep apart the binaries that the option produces.
    """
    try:
        return _canon_value(flags, set()), True
    except (_Unfingerprintable, RecursionError):
        return repr(sorted(flags.items(), key=repr)), False
