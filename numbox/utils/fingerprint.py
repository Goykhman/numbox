"""Content fingerprinting of Python values and functions for cache keys.

A value canonicalizer (`_canon_value`) and a deep function fingerprint
(`_fingerprint_function`) that capture everything numba freezes into a compiled
artifact: code-object bytecode/consts/names, default arguments, closure-cell
values, and the values of referenced module-level globals (recursing into helper
functions and dispatchers, with cycle protection). Stronger than hashing the
bare code object -- two functions with identical source but different captured
closure/global values fingerprint differently. Shared by
`numbox.core.variable.compile_kernel` (kernel cache digest) and
`numbox.utils.digest` (SQLite UDAF cache key).

`_Unfingerprintable` is raised for any value with no canonical form; callers
decide how to degrade (compile_kernel marks the kernel uncached, digest falls
back to cloudpickle of the code object).
"""
import ctypes
import dis
import hashlib

from types import CodeType, FunctionType, ModuleType
from typing import Any, Optional, Sequence

import numpy as np

from numba.core import config as numba_config
from numba.core.dispatcher import Dispatcher
from numba.core.types import ExternalFunction
from numba.core.types.function_type import CompileResultWAP
from numba.np.ufunc.dufunc import DUFunc

from numbox.core.configurations import jit_options as _default_jit_options


class _Unfingerprintable(Exception):
    """A value with no canonical fingerprint; the caller decides how to degrade."""


def _safe_repr(obj: object) -> str:
    """``repr(obj)`` that never raises -- the fingerprint fallback must always
    yield a string so an un-fingerprintable formula degrades to uncached rather
    than crashing when its ``__repr__`` itself raises."""
    try:
        return repr(obj)
    except Exception:  # noqa: BLE001 - fallback must not crash on a raising __repr__
        return f"<{type(obj).__name__} repr-failed>"


def _dtype_key(dtype: np.dtype) -> str:
    """Canonical string for a numpy dtype that preserves structured layout.

    ``dtype.str`` collapses any structured dtype to ``|V<itemsize>`` -- field
    names, types, order and offsets are all erased, so two same-byte different-
    layout arrays would collide. For a structured dtype fold its full
    descriptor and itemsize instead."""
    if dtype.names is not None:
        return f"struct({dtype!s};descr={dtype.descr};itemsize={dtype.itemsize})"
    return dtype.str


# Aliases that have stopped identifying the body they were minted for. Written by
# `numbox.core.proxy.proxy`, which imports this module: the set lives on this side so the
# walker can consult it without importing that one back. It holds both halves of a
# collision, the shared name and the process-local one the newcomer is given, and the name
# an absent-binding trap answers to.
#
# `_canon_value` refuses to fold one, and that is what keeps a collision from spreading.
# Which of two colliding bodies ends up with which name is decided by the order this process
# built them in, so a third body that captures one and folds `proxy(<alias>)` gets a
# fingerprint that names the other body in a process that built the pair the other way round.
# The colliding bodies are themselves protected, because their own names are retired and the
# cache guard discards a warm caller of a retired name. A body derived from one of them is
# not: it is published under a fresh name that resolves and sits in no retirement set, so it
# keeps a cross-process cache that is now wrong, which is how a collision escapes one
# composition step out. Refusing to fold drops the derived body onto the ordinary walk, where
# the dispatcher it captured has no canonical form, so it meets `_opaque_values_match` like
# any other pair the fingerprint cannot separate and its own alias is retired in turn.
#
# A derived body fingerprinted before the collision that retires its dependency is not
# reached: it folded a name that still identified a body and is already published under one
# of its own. Closing that would mean recording which aliases each fingerprint consumed and
# retiring dependents transitively.
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
            # A @proxy binding: its content-addressed alias fully identifies the
            # body and signature. Walking its wrapper would recurse into the
            # @intrinsic it calls, which has no canonical form. A retired alias
            # identifies nothing, so it falls through to that walk instead.
            return f"proxy({proxy_alias})"
        topts = _canon_value(dict(getattr(value, "targetoptions", {}) or {}), seen)
        return f"dispatcher({_fingerprint_function(value.py_func, seen)};{topts})"
    if isinstance(value, DUFunc):
        # @vectorize: numba links its compiled loop and freezes the scalar body's
        # captured globals/closure, so fold that body plus its targetoptions.
        wrapped = getattr(value, "__wrapped__", None)
        topts = _canon_value(dict(getattr(value, "targetoptions", {}) or {}), seen)
        body = _fingerprint_function(wrapped, seen) if isinstance(wrapped, FunctionType) else _safe_repr(wrapped)
        return f"dufunc({body};{topts})"
    if isinstance(value, FunctionType):
        return f"function({_fingerprint_function(value, seen)})"
    if isinstance(value, ExternalFunction):
        # numba's own key for this type, (symbol, signature): strings and types,
        # so it is the same in the next process and it separates two bindings a
        # factory made over different C symbols.
        return f"externalfunction({value.symbol};{value.sig})"
    if isinstance(value, CompileResultWAP):
        # A first-class function value. numbox tags the two it mints with what identifies
        # them across processes, because the wrapper itself carries only a compiled address
        # and a mangled name holding a per-process counter. Untagged ones -- built directly
        # against numba -- keep falling through to the placeholder below.
        proxy_alias = getattr(value, "_numbox_proxy_alias", None)
        if proxy_alias is not None and proxy_alias not in _ABSENT_ALIASES:
            return f"proxyfunc({proxy_alias})"
        py_func = getattr(value, "_numbox_py_func", None)
        if py_func is not None:
            return f"cresfunc({_fingerprint_function(py_func, seen)};{value.cres.signature})"
    if isinstance(value, ctypes._CFuncPtr):
        return f"cfuncptr({_ctypes_func_key(value)})"
    raise _Unfingerprintable(type(value).__name__)


def _ctypes_func_key(value: Any) -> str:
    """Identity of a ctypes function pointer: the library it was opened under, and the symbol.

    A pointer reached by attribute access on a loaded library carries both, and both
    survive into the next process. What is folded is the *string* the library was opened
    with rather than the library the loader resolved it to, so two handles on one library
    opened under a soname and under an absolute path canonicalize differently. That is
    over-discrimination: it costs a recompile the load-time guard heals, never a wrong
    answer.

    A pointer with no symbol name to fold carries no identity that means the same thing
    next run. One built from a raw address is the common case, and two of those differ
    only by an address ASLR moves; a prototype built over a library, ``proto(("floor",
    lib))``, and a Windows lookup by ordinal are the same problem for a different reason,
    since ctypes names neither. Folding an address would buy discrimination at the cost of
    the process-stability the whole fingerprint exists for, so all three raise and the
    caller decides what to do with a value it cannot tell apart from another.

    The prototype over the symbol is folded alongside it, by :func:`_ctypes_call_shape`.
    """
    name = getattr(value, "__name__", None)
    objects = getattr(value, "_objects", None)
    lib = objects.get("0") if isinstance(objects, dict) else None
    # Only a library carries ``_name``. A Python callback's ``_objects["0"]`` is its
    # CThunkObject, which does not, and folding the None would collapse every callback
    # sharing a ``__name__`` onto one key.
    lib_name = getattr(lib, "_name", None)
    if name is None or not isinstance(lib_name, str) or not lib_name:
        raise _Unfingerprintable("ctypes function pointer with no library and symbol name")
    return f"{lib_name}:{name};{_ctypes_call_shape(value)}"


def _ctypes_call_shape(value: Any) -> str:
    """The call a ctypes function pointer describes, apart from what it points at.

    ``restype``, ``argtypes`` and the calling convention are what numba types the call from
    and lowers it against, so two pointers agreeing on the symbol or the address but not on
    these describe two different calls and compile to two different bodies. Every part of it
    is a type name or a flag word, so it means the same thing in the next process and
    :func:`_ctypes_func_key` can fold it; a pointer that key refuses for want of a
    process-stable identity is compared on it by :func:`_opaque_values_match` instead, where
    an address is admissible and this is what says whether the address is being called the
    same way.

    An unset ``argtypes`` and an empty one are different: numba rejects a call through the
    first and accepts the second, so they must not fold together.

    The calling convention is folded because numba selects the call's ABI from it on win32:
    a symbol reached through ``CDLL`` and the same symbol reached through ``WinDLL`` are two
    different calls, and sharing one alias would pair a caller with a body at the wrong
    convention.
    """
    restype = getattr(getattr(value, "restype", None), "__name__", None)
    argtypes = getattr(value, "argtypes", None)
    args = "<unset>" if argtypes is None else ",".join(getattr(a, "__name__", "?") for a in argtypes)
    return f"{restype}({args});flags={getattr(type(value), '_flags_', None)}"


def _fingerprint_codeobj(code: CodeType, seen: set[int]) -> str:
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


# The bytecode ops that name the *global/module namespace*, as opposed to an
# attribute (``LOAD_ATTR``/``LOAD_METHOD``/``STORE_ATTR``) or an import target.
_GLOBAL_NAME_OPS = frozenset({
    "LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL",
    "LOAD_NAME", "STORE_NAME", "DELETE_NAME",
})


def _loaded_global_names(code: CodeType) -> set[str]:
    """Names this code touches in the *global* namespace, from the instruction stream.

    ``co_names`` (used by :func:`_referenced_global_names`) pools global reads with
    attribute names: ``a.c1`` and ``cfg.SCALE`` both put ``c1`` / ``SCALE`` in
    ``co_names`` via ``LOAD_ATTR``, though neither is a global. Deciding which
    globals' *values* to fold off ``co_names`` therefore folds -- or, if the value
    has no canonical form, chokes on -- a module global that merely shares a name with
    a record field or a chained attribute, over-invalidating the fingerprint (or, for
    the builder's derive path, forcing the address-bearing fallback and unbounded cache
    growth). Reading the opcodes tells a ``LOAD_GLOBAL c1`` from a ``LOAD_ATTR c1``.
    Recurses into nested code objects (comprehensions, lambdas). ``cfg.SCALE``-style
    module-attribute folding stays keyed on the full ``co_names`` set, so a module
    attribute still re-keys; only the base ``cfg`` is a global here, and it is caught.
    """
    names: set[str] = set()
    for instr in dis.get_instructions(code):
        if instr.opname in _GLOBAL_NAME_OPS and isinstance(instr.argval, str):
            names.add(instr.argval)
    for c in code.co_consts:
        if isinstance(c, CodeType):
            names |= _loaded_global_names(c)
    return names


# The value types numba freezes into a compiled artifact as constants -- the
# data module attributes whose value must enter the fingerprint. A Dispatcher
# module attribute is folded too (numba links it and freezes its own captured
# globals/closure, exactly as for a direct global dispatcher). A ufunc, plain
# function, or type is a stable reference numba resolves by identity (or by its
# own builtin lowering), so it is skipped.
_FROZEN_DATA_TYPES = (bool, int, float, complex, str, bytes,
                      np.generic, np.ndarray, tuple, list, set, frozenset, dict)


def _fingerprint_module_attrs(referenced: set[str], namespace: dict, seen: set[int]) -> str:
    """Fold the values of ``<module>.<attr>`` reads that numba freezes as constants.

    ``_fingerprint_function`` folds referenced globals present in ``__globals__``,
    but a value read through module indirection (``import cfg; cfg.SCALE``, or the
    chained ``cfg.sub.SCALE``, or a closure-captured module) is invisible: the
    module canonicalizes to ``module(<name>)`` and the leaf ``SCALE`` is not itself
    a global. numba bakes the leaf value into the binary, so a later change would be
    served stale. ``namespace`` maps referenced globals AND closure free-variables
    to their values; for each module reachable there, fold every referenced name in
    its ``__dict__`` that is frozen data or a Dispatcher (whose own captured
    globals/closure numba freezes when it links the call -- folded via
    ``_canon_value`` exactly as a direct global dispatcher is), recursing into
    referenced submodules (with module-id cycle protection). Ufuncs, plain
    functions and types are stable references numba resolves by identity, so they
    are skipped (the formula stays cacheable, no recursion into e.g. numpy).
    Scanning is by ``__dict__`` membership (never ``getattr``) so no module
    ``__getattr__`` is triggered -- resolving referenced names via ``getattr``
    would fire an unrelated module's PEP 562 ``__getattr__`` (numpy's, say) on
    every co-name collision, importing submodules and emitting deprecation
    warnings. The cost is a documented limitation: a value a module exposes ONLY
    via PEP 562 ``__getattr__`` is not folded (numba freezes it, so change it and
    clear the cache)."""
    attrs: list[str] = []
    mods_seen: set[int] = set()

    def _walk(prefix: str, mod: ModuleType) -> None:
        if id(mod) in mods_seen:
            return
        mods_seen.add(id(mod))
        mod_dict = getattr(mod, "__dict__", {})
        for attr in sorted(referenced):
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
    """Whether two runs of the best-effort walker stood on the same un-canonical values.

    Two bodies that fingerprint alike are the same body except in whatever the walker
    could not canonicalize, so comparing exactly those values answers whether they are
    the same body. The comparison only has to hold within one process, which is why it
    may do what the fingerprint may not and look at an address.

    Identity settles the ordinary case: reloading a module rebinds the same
    ``@intrinsic`` object, and a factory called twice closes over the same one.
    Equality is tried next for values that define it. A ctypes function pointer defines
    neither, so compare the address it holds together with the call the prototype over it
    describes. The address alone is not the body: one symbol reached through two prototypes
    is two calls numba types and lowers differently, so reading those as one value hands the
    second body the alias the first was published under, silently and with no collision to
    report. :func:`_ctypes_call_shape` is what the fingerprint itself folds for a pointer it
    can name, so the two paths agree on what makes two pointers two calls.
    """
    if len(old) != len(new):
        return False
    for a, b in zip(old, new):
        if a is b:
            continue
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
    """Best-effort function fingerprint that never raises.

    ``_fingerprint_function`` raises ``_Unfingerprintable`` on the *first* value
    with no canonical form and discards everything computed so far, so a body
    that merely *references* an un-canonicalizable global (e.g. an
    ``@intrinsic``) collapses to nothing usable. This variant substitutes a
    stable ``<opaque:type>`` placeholder for any such value instead, so bodies
    differing only in a captured constant, closure cell, default, or a
    canonicalizable global still fingerprint distinctly. For consumers that need
    a discriminating identifier even for un-fingerprintable bodies -- the
    ``@proxy`` cfunc alias -- rather than the degrade-to-uncached path that
    ``_fingerprint_function``'s raising contract serves (``compile_kernel``,
    ``digest``). Process-stable: every placeholder is derived from a type name,
    never an address.

    A placeholder is where two different bodies can still meet on one fingerprint. Pass
    ``opaque`` to collect the values that got one, in walk order, so a caller holding two
    equal fingerprints can ask :func:`_opaque_values_match` whether they came from one
    body or from two the fingerprint could not separate.
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
# Shared by every content-addressed cache key that must re-key when the resolved
# jit flags change: the `compile_kernel` digest and the `make_graph` kernel
# fingerprint. Kept here rather than in `compile_kernel` so `builder` can reuse it
# without importing that module (whose import runs an orphan-anchor sweep).

_CODEGEN_ENV_KNOBS = (
    "BOUNDSCHECK", "LOOP_VECTORIZE", "SLP_VECTORIZE", "ENABLE_AVX", "DISABLE_INTEL_SVML",
)


def _effective_flags(jit_options: dict | None) -> dict:
    """The non-`cache` jit flags, resolved against the numbox defaults.

    `cache` is excluded deliberately: whether an artifact is written to disk does
    not change the emitted binary, so folding it would split the key needlessly.
    Every other flag (`fastmath`, `error_model`, `boundscheck`, ...) does change
    the binary and must re-key.
    """
    opts = {**_default_jit_options, **(jit_options or {})}
    return {k: v for k, v in opts.items() if k != "cache"}


def _codegen_env_canon() -> str:
    """Canonical string for the process's result-affecting numba codegen env knobs.

    These change the emitted binary but live in neither the jit flags nor numba's
    own on-disk cache key, and they override even an explicit jit flag at lowering,
    so they must enter a digest directly, read at compile time.
    """
    return repr([(k, getattr(numba_config, k, None)) for k in _CODEGEN_ENV_KNOBS])


def _flags_canon(flags: dict) -> tuple[str, bool]:
    """Canonical string for the effective jit flags, and whether it is a true
    canonicalization (``False`` if a flag value had no canonical form, e.g. a
    ``pipeline_class`` or numba-typed ``locals``). A ``False`` here means the
    caller must degrade to uncached: a key that cannot see a flag cannot protect
    the binary that flag produced."""
    try:
        return _canon_value(flags, set()), True
    except (_Unfingerprintable, RecursionError):
        return repr(sorted(flags.items(), key=repr)), False
