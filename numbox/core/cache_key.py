"""numbox's part of numba's on-disk cache key.

numba keys a cached function on its signature, the target machine, and hashes of its bytecode and closure. That key
leaves out two things that shape the binary: the jit flags the function compiled under, and the ``NUMBA_*`` settings
that change the code numba emits. Change one of them between two runs and the second run loads the binary the first
run built, with no warning. numbox adds both, with its own version, for every function it compiles and for any
function decorated with :func:`keyed_njit` or :func:`keyed_cfunc`; numba keeps its own key for everything else.

What the key holds, past numba's own:

- **numbox's version.** numbox's ``@overload`` and ``@intrinsic`` code generators shape the code of whatever calls
  them, and their version stands in for them.
- **The flags.** The decorator kind (``njit`` or ``cfunc``), the flags numba compiles under and the ``locals`` given
  at decoration. The flags are the decorator's options as numba parses them when a compile starts, its defaults
  filled in. Where the options leave ``nrt``, ``fastmath``, ``error_model`` or ``forceinline`` unset, numba takes them
  from the function whose compile calls this one, and so does the key.
- **The settings.** Each of :data:`KEYED_SETTINGS`, as ``numba.config`` holds it when numba looks the function up.

What it leaves out, as numba's key does: the values of the globals a function reads, and the code of the functions it
calls. numba checks a cached function against the file that defines it, so an edit to that file misses. A changed
global or an edited callee in another file does not, and the old binary is served. Clear the cache after such a
change, or pass the value as an argument. An edit to numbox itself in a development checkout is such a change too:
numbox's version stays the same, so clear the cache after it.

A function whose flags the key cannot hold compiles without an on-disk cache, and an :class:`UncachedWarning` names
what the key could not hold. That is a ``parallel`` dict, which numba empties as it parses it, a pipeline class other
than numba's default, or a function given as the ``inline`` cost model. It is also a type in ``locals`` that numba
names after an object's address.

The settings are read as they stand. Set a ``NUMBA_*`` variable before Python starts. numba reads
``NUMBA_DISABLE_INTEL_SVML`` as it is imported, and from 0.66 on ``NUMBA_DEBUG_NRT`` and
``NUMBA_DEBUG_NRT_STACK_LIMIT`` as well. It reads some other settings once, when the first jit decoration builds its
code generator. It reads ``os.environ`` again only as a compile starts, after the key is read. Any setting but those
three can also be changed in code before the first jit decoration.
"""
import contextlib
import functools
import re
import threading
import warnings

from types import FunctionType

import numpy as np

from numba import cfunc, njit
from numba.core import caching, config as numba_config
from numba.core.ccallback import CFunc
from numba.core.compiler import Compiler, Flags
from numba.core.cpu_options import FastMathOptions, InlineOptions, ParallelOptions
from numba.core.dispatcher import Dispatcher
from numba.core.types import Type as NumbaType

import numbox


#: The ``NUMBA_*`` settings that change the code numba emits, each with the ``numba.config`` attribute numba reads it
#: into. A survey of numba 0.63 to 0.68, each minor at its latest patch, classified them. The survey compiled a fixed
#: corpus under every setting flipped and compared the emitted LLVM IR and assembly. A setting is keyed if flipping it
#: changes the code, speed-only and debug-information settings included. ``NUMBA_DUMP_ASSEMBLY``, ``NUMBA_DEBUG`` and
#: ``NUMBA_DIFF_IR`` change the code only as a side effect of the dumps they ask for, and are keyed all the same. Three
#: change the code only beside another setting, so a flip of one at a time cannot show them. They were found in
#: numba's source: ``NUMBA_DEBUG_NRT_STACK_LIMIT`` under ``NUMBA_DEBUG_NRT``, and ``NUMBA_FULL_TRACEBACKS`` and
#: ``NUMBA_DEVELOPER_MODE`` (its default) in a bounds check. ``NUMBA_DISABLE_INTEL_SVML`` changes the code only where
#: Intel's SVML library is installed, which the survey's machines lacked: numba then calls it for vectorized math.
KEYED_SETTINGS = {
    "NUMBA_BOUNDSCHECK": "BOUNDSCHECK",
    "NUMBA_OPT": "OPT",
    "NUMBA_LOOP_VECTORIZE": "LOOP_VECTORIZE",
    "NUMBA_SLP_VECTORIZE": "SLP_VECTORIZE",
    "NUMBA_ENABLE_AVX": "ENABLE_AVX",
    "NUMBA_DEBUGINFO": "DEBUGINFO_DEFAULT",
    "NUMBA_ENABLE_PROFILING": "ENABLE_PROFILING",
    "NUMBA_EXTEND_VARIABLE_LIFETIMES": "EXTEND_VARIABLE_LIFETIMES",
    "NUMBA_LLVM_REFPRUNE_PASS": "LLVM_REFPRUNE_PASS",
    "NUMBA_LLVM_REFPRUNE_FLAGS": "LLVM_REFPRUNE_FLAGS",
    "NUMBA_DEBUG_NRT": "DEBUG_NRT",
    "NUMBA_DEBUG_NRT_STACK_LIMIT": "DEBUG_NRT_STACK_LIMIT",
    "NUMBA_FULL_TRACEBACKS": "FULL_TRACEBACKS",
    "NUMBA_DEVELOPER_MODE": "DEVELOPER_MODE",
    "NUMBA_DEBUG_JIT": "DEBUG_JIT",
    "NUMBA_DEBUG_ARRAY_OPT": "DEBUG_ARRAY_OPT",
    "NUMBA_DEBUG_ARRAY_OPT_RUNTIME": "DEBUG_ARRAY_OPT_RUNTIME",
    "NUMBA_PARFOR_MAX_TUPLE_SIZE": "PARFOR_MAX_TUPLE_SIZE",
    "NUMBA_GDB_BINARY": "GDB_BINARY",
    "NUMBA_DUMP_ASSEMBLY": "DUMP_ASSEMBLY",
    "NUMBA_DEBUG": "DEBUG",
    "NUMBA_DIFF_IR": "DIFF_IR",
    "NUMBA_DISABLE_INTEL_SVML": "DISABLE_INTEL_SVML",
}

_IN_NUMBAS_KEY = "in numba's own key, through the target machine"
_NO_BINARY = "produces no binary"
_NO_CHANGE = "changes no emitted code"
_LOCATION = "chooses where the cache lives, not what goes in it"

#: Every other ``NUMBA_*`` setting numba 0.63 to 0.68 reads, with the reason it stays out of the key. A test fails when
#: a numba release reads a setting that is in neither table, in ``config.py`` or through ``os.environ`` elsewhere.
UNKEYED_SETTINGS = {
    "NUMBA_CPU_NAME": _IN_NUMBAS_KEY,
    "NUMBA_CPU_FEATURES": _IN_NUMBAS_KEY,
    "NUMBA_DISABLE_JIT": _NO_BINARY,
    "NUMBA_USE_LEGACY_TYPE_SYSTEM": _NO_BINARY,
    "NUMBA_CACHE_DIR": _LOCATION,
    "NUMBA_CACHE_LOCATOR_CLASSES": _LOCATION,
    **dict.fromkeys((
        "NUMBA_ALWAYS_WARN_UNINIT_VAR", "NUMBA_CAPTURED_ERRORS", "NUMBA_CHROME_TRACE", "NUMBA_COLOR_SCHEME",
        "NUMBA_DEBUG_ARRAY_OPT_STATS", "NUMBA_DEBUG_CACHE", "NUMBA_DEBUG_FRONTEND", "NUMBA_DEBUG_INLINE_CLOSURE",
        "NUMBA_DEBUG_PRINT_AFTER", "NUMBA_DEBUG_PRINT_BEFORE", "NUMBA_DEBUG_PRINT_WRAP",
        "NUMBA_DEBUG_TYPEINFER", "NUMBA_DISABLE_CUDA", "NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING",
        "NUMBA_DISABLE_PERFORMANCE_WARNINGS", "NUMBA_DISABLE_TYPEINFER_FAIL_CACHE", "NUMBA_DUMP_ANNOTATION",
        "NUMBA_DUMP_BYTECODE", "NUMBA_DUMP_CFG", "NUMBA_DUMP_FUNC_OPT", "NUMBA_DUMP_HTML", "NUMBA_DUMP_IR",
        "NUMBA_DUMP_LLVM", "NUMBA_DUMP_OPTIMIZED", "NUMBA_DUMP_SSA", "NUMBA_DYLD_WORKAROUND", "NUMBA_ENABLE_CUDASIM",
        "NUMBA_ENABLE_SYS_MONITORING", "NUMBA_FORCE_CUDA_CC", "NUMBA_FUNCTION_CACHE_SIZE", "NUMBA_HIGHLIGHT_DUMPS",
        "NUMBA_JIT_COVERAGE", "NUMBA_LLVM_PASS_TIMINGS", "NUMBA_NRT_STATS", "NUMBA_NUM_THREADS",
        "NUMBA_PARALLEL_DIAGNOSTICS", "NUMBA_SHOW_HELP", "NUMBA_THREADING_LAYER", "NUMBA_THREADING_LAYER_PRIORITY",
        "NUMBA_TRACE", "NUMBA_USE_LLVMLITE_MEMORY_MANAGER",
        "NUMBA_CUDA_ARRAY_INTERFACE_SYNC", "NUMBA_CUDA_DEBUGINFO", "NUMBA_CUDA_DEFAULT_PTX_CC", "NUMBA_CUDA_DRIVER",
        "NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY", "NUMBA_CUDA_INCLUDE_PATH", "NUMBA_CUDA_LOG_API_ARGS",
        "NUMBA_CUDA_LOG_LEVEL", "NUMBA_CUDA_LOG_SIZE", "NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS",
        "NUMBA_CUDA_MAX_PENDING_DEALLOCS_COUNT", "NUMBA_CUDA_MAX_PENDING_DEALLOCS_RATIO", "NUMBA_CUDA_MEMORY_MANAGER",
        "NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM", "NUMBA_CUDA_USE_NVIDIA_BINDING", "NUMBA_CUDA_VERBOSE_JIT_LOG",
        "NUMBA_CUDA_WARN_ON_IMPLICIT_COPY",
    ), _NO_CHANGE),
}


_WHY = "numbox's cache key cannot hold it reliably, so the function is not cached."


class UncachedWarning(RuntimeWarning):
    """A function numbox keys is compiled without an on-disk cache, because the key cannot hold one of its flags.

    That is a ``parallel`` dict, which numba empties as it parses it; a pipeline class other than numba's default; a
    function given as the ``inline`` cost model; or a type in ``locals`` that numba names after an object's address.
    The warning names the function and what the key could not hold.
    """


class _Unkeyable(Exception):
    """Raised inside the key computation for a flag the key cannot hold; the cache wrappers turn it into a miss."""


# ---- The settings ----

def _setting_value(value) -> str:
    """The canonical text of one setting's value.

    numba's ``NUMBA_OPT`` is an ``int`` that keeps the value it was given, ``max`` among them, in ``_raw_value``; a
    bool and an int that agree mean the same to numba, and are put the same way.
    """
    raw = getattr(value, "_raw_value", None)
    if raw is not None:
        value = raw
    if isinstance(value, int):
        value = int(value)
    return repr(value)


def _current_settings() -> tuple:
    """Each keyed setting and its value, as ``numba.config`` holds it now."""
    return tuple((env, _setting_value(getattr(numba_config, attr, None))) for env, attr in KEYED_SETTINGS.items())


def _settings_canon() -> str:
    """The settings part of the key as it stands.

    Shared with the content-addressed names of the kernels :mod:`numbox.core.variable.compile_kernel` generates and
    the derives :mod:`numbox.core.work.builder` anchors, which fold it into their digests.
    """
    return repr(list(_current_settings()))


# ---- The flags ----

#: How numba's text for a type names an object of this process by its address: a jitclass's types, a deferred type, and
#: the default repr of a function or dispatcher inside a type.
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+|jitclass\.[\w.]*#[0-9a-fA-F]+|DeferredType#\d+")


def _type_text(value, where) -> str:
    """A type given in ``locals``, as the key holds it: numba's text for it, unless that text names an address."""
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, NumbaType):
        text = str(value)
        if _ADDRESS.search(text):
            raise _Unkeyable(f"{where} is the numba type {text}, which names an object by its address")
        return f"numbatype({type(value).__module__}.{type(value).__qualname__}:{text})"
    raise _Unkeyable(f"{where} is of type {type(value).__name__}, which has no stable form")


def _flag_value_text(value, where) -> str:
    """One of the flags numba compiles under, as the key holds it: plain values as they are, a set of fast-math flags in
    order, the parallel options one by one, or the inlining choice."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return repr(value)
    if isinstance(value, FastMathOptions):
        return "fastmath[" + ",".join(sorted(value.flags)) + "]"
    if isinstance(value, ParallelOptions):
        return "parallel[" + ",".join(f"{k}={_flag_value_text(v, where)}" for k, v in value._get_values().items()) + "]"
    if isinstance(value, InlineOptions) and isinstance(value.value, str):
        return f"inline[{value.value!r}]"
    if isinstance(value, InlineOptions):
        raise _Unkeyable(f"{where} is a cost model, a function the key does not hold")
    raise _Unkeyable(f"{where} is of type {type(value).__name__}, which has no stable form")


def _flags_text(kind, compiler) -> str:
    """The decorator kind, the flags numba compiles under and the ``locals`` given at decoration.

    The flags come from numba's own parse of the decorator's options, with the changes numba's compiler makes to them
    next, a C callback's among them. Called inside the compile of a function that calls this one, the parse takes the
    flags the options leave unset from that compile, as numba's does. For a variable
    given in ``locals`` that it renames, numba's SSA pass adds each new name, the variable's name and a number after a
    dot, typed as the variable. Those are left out, since the given type decides them. A name with a dot whose part
    before the last dot is not given is the caller's own, and stays.
    """
    options = compiler.targetoptions
    if isinstance(options.get("parallel"), dict):
        raise _Unkeyable("the jit flag parallel is a dict, which numba empties as it parses it")
    if compiler.pipeline_class is not Compiler:
        pipeline = getattr(compiler.pipeline_class, "__qualname__", compiler.pipeline_class)
        raise _Unkeyable(f"the pipeline class is {pipeline}, not numba's default")
    given = compiler.locals or {}
    locals_ = sorted((name, ty) for name, ty in given.items() if name.rpartition(".")[0] not in given)
    flags = Flags()
    compiler.targetdescr.options.parse_as_flags(flags, options)
    flags = compiler._customize_flags(flags)
    values = ";".join(
        f"{name}={_flag_value_text(value, f'the jit flag {name}')}" for name, value in sorted(flags.values().items())
    )
    types = ",".join(f"{name}:{_type_text(ty, 'a type in locals')}" for name, ty in locals_)
    return f"kind={kind};flags=[{values}];locals=[{types}]"


def _key_part(keyed) -> tuple:
    """numbox's part of the key for the next compile: its version, the flags and the settings."""
    kind, compiler = keyed
    return (numbox.__version__, _flags_text(kind, compiler), _settings_canon())


def explain_cache_key(func) -> dict:
    """The parts numbox adds to ``func``'s cache key, as they stand for a direct call's next compile in this process.

    ``func`` is a dispatcher or ``@cfunc`` whose cache numbox keys, what ``cres`` returns, or a ``@proxy`` function's
    ``.as_func``: a function numbox compiled, or one decorated with :func:`keyed_njit` or :func:`keyed_cfunc` with
    ``cache=True``. The result maps ``"numbox"`` to numbox's version, ``"flags"`` to the flags as the key holds them,
    and ``"settings"`` to each keyed setting's value. Two calls that disagree on a part are two different keys.
    """
    func = getattr(func, "_numbox_dispatcher", func)
    cache = getattr(func, "_cache", None)
    keyed = getattr(cache, "_numbox_keyed", None)
    if keyed is None:
        raise ValueError(
            f"{func!r} has no cache numbox keys: it is cached under numba's key alone, or not cached; decorate it "
            "with keyed_njit or keyed_cfunc and cache=True"
        )
    try:
        flags = _flags_text(*keyed)
    except _Unkeyable as exc:
        raise ValueError(f"{func!r} compiles without an on-disk cache: {exc}") from None
    return {"numbox": numbox.__version__, "flags": flags, "settings": dict(_current_settings())}


# ---- The opt-in ----

_keying = threading.local()


@contextlib.contextmanager
def _keyed(func):
    """Mark the cache numba sets up for ``func`` while this is open, so numbox keys it."""
    stack = getattr(_keying, "funcs", None)
    if stack is None:
        stack = _keying.funcs = []
    stack.append(func)
    try:
        yield
    finally:
        stack.pop()


def _keys(py_func) -> bool:
    """Whether numbox keys the cache of ``py_func``: numbox's own code, or a function being decorated by numbox."""
    module = getattr(py_func, "__module__", None) or ""
    if module == "numbox" or module.startswith("numbox."):
        return True
    return any(f is py_func for f in getattr(_keying, "funcs", ()))


def keyed_njit(*args, **kwargs):
    """``numba.njit``, taking its arguments unchanged, with numbox's cache key on the function it decorates.

    With ``cache=True`` the function is cached under numba's key plus numbox's part of it: see the module documentation
    for what that holds. The key is set up as numba sets the cache up, at decoration, so it applies to an eager compile
    (a signature given) as to a lazy one. Without ``cache=True`` this is ``numba.njit``.
    """
    func = args[0] if args else kwargs.get("signature_or_function")
    if isinstance(func, FunctionType):
        with _keyed(func):
            return njit(*args, **kwargs)
    decorator = njit(*args, **kwargs)

    def wrap(func):
        with _keyed(func):
            return decorator(func)
    return wrap


def keyed_cfunc(*args, **kwargs):
    """``numba.cfunc``, taking its arguments unchanged, with numbox's cache key on the function it decorates.

    With ``cache=True`` the C callback is cached under numba's key plus numbox's part of it, which tells it apart from
    an ``njit`` of the same function as well. Without ``cache=True`` this is ``numba.cfunc``.
    """
    decorator = cfunc(*args, **kwargs)

    def wrap(func):
        with _keyed(func):
            return decorator(func)
    return wrap


# ---- The hooks ----

_uncached_reported = set()


def _qualname(cache) -> str:
    py_func = getattr(cache, "_py_func", None)
    return getattr(py_func, "__qualname__", None) or getattr(cache, "_name", "?")


def _warn_uncached(cache, exc):
    report = (getattr(cache, "_name", None), str(exc))
    if report in _uncached_reported:
        return
    _uncached_reported.add(report)
    warnings.warn(
        f"numbox: {_qualname(cache)} is compiled without an on-disk cache: {exc}. {_WHY}",
        UncachedWarning, stacklevel=4,
    )


def _wrap_index_key(original):
    @functools.wraps(original)
    def _index_key(self, sig, codegen):
        key = original(self, sig, codegen)
        keyed = getattr(self, "_numbox_keyed", None)
        return key if keyed is None else key + (_key_part(keyed),)
    return _index_key


def _wrap_load_overload(original):
    @functools.wraps(original)
    def load_overload(self, sig, target_context):
        try:
            return original(self, sig, target_context)
        except _Unkeyable as exc:
            _warn_uncached(self, exc)
            return None
    return load_overload


def _wrap_save_overload(original):
    @functools.wraps(original)
    def save_overload(self, sig, data):
        try:
            original(self, sig, data)
        except _Unkeyable as exc:
            _warn_uncached(self, exc)
    return save_overload


def _wrap_enable_caching(original, kind, py_func_of):
    @functools.wraps(original)
    def enable_caching(self):
        original(self)
        if _keys(py_func_of(self)):
            self._cache._numbox_keyed = (kind, self._compiler)
    return enable_caching


def _install():
    """Wrap numba's cache and the two decorators' cache set-up; once per process.

    Every lookup happens before the first assignment, so a numba that lacks one of them leaves no half-installed key.
    """
    if getattr(caching, "_numbox_cache_key", False):
        return
    wrapped = [
        (caching.Cache, "_index_key", _wrap_index_key(caching.Cache._index_key)),
        (caching.Cache, "load_overload", _wrap_load_overload(caching.Cache.load_overload)),
        (caching.Cache, "save_overload", _wrap_save_overload(caching.Cache.save_overload)),
        (Dispatcher, "enable_caching", _wrap_enable_caching(Dispatcher.enable_caching, "njit", lambda d: d.py_func)),
        (CFunc, "enable_caching", _wrap_enable_caching(CFunc.enable_caching, "cfunc", lambda c: c._pyfunc)),
    ]
    caching._numbox_cache_key = True
    for owner, name, replacement in wrapped:
        setattr(owner, name, replacement)


try:
    _install()
except Exception as _exc:
    warnings.warn(
        f"numbox: could not add its part to numba's cache key ({_exc!r}); what numbox compiles is cached under "
        "numba's key alone",
        RuntimeWarning,
    )
