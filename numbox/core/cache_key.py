"""numbox's part of numba's on-disk cache key, and the decorators that add it to a function's key.

numba caches a compiled function under a key that leaves out the jit options and most of the ``NUMBA_*`` settings
that change the emitted code. numbox appends a part of its own to that key for each function that it compiles with
caching enabled: numbox's version, the jit options that numba compiles under, and the settings in
:data:`KEYED_SETTINGS`. :func:`numbox_njit` and :func:`numbox_cfunc` add the same part to the functions you decorate,
and :func:`explain_cache_key` returns it. numbox installs the part by wrapping numba's cache functions once per
process, when ``numbox.core`` is first imported.
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
#: into. A survey of numba 0.63 to 0.68, each minor version at its latest patch release, classified them. The survey
#: compiled a fixed corpus with each setting changed in turn and compared the emitted LLVM IR and assembly. A setting
#: that numba's own key leaves out is keyed if and only if changing it changes the code, even if only speed or
#: debugging information changes. ``NUMBA_DUMP_ASSEMBLY``, ``NUMBA_DEBUG`` and ``NUMBA_DIFF_IR`` change the code only
#: as a side effect of the dumps they request, and are keyed as well. Three settings change the code only in
#: combination with another setting or flag, so a survey that changes one setting at a time cannot reveal them. numba's
#: source shows them. ``NUMBA_DEBUG_NRT_STACK_LIMIT``, which numba reads from 0.66 on, changes the code only under
#: ``NUMBA_DEBUG_NRT``. ``NUMBA_FULL_TRACEBACKS`` and ``NUMBA_DEVELOPER_MODE`` (its default) change it only where
#: bounds checking is enabled. ``NUMBA_DISABLE_INTEL_SVML`` changes the code only where Intel's SVML library is
#: installed, which the survey's machines lacked; numba then calls SVML for vectorized math.
KEYED_SETTINGS = {
    "NUMBA_BOUNDSCHECK": "BOUNDSCHECK",
    "NUMBA_OPT": "OPT",
    "NUMBA_LOOP_VECTORIZE": "LOOP_VECTORIZE",
    "NUMBA_SLP_VECTORIZE": "SLP_VECTORIZE",
    # numba's own key already reflects this setting. Unless NUMBA_CPU_FEATURES is set, numba removes the AVX features
    # from its target machine's CPU features when the setting is 0, and its key holds those features
    # (get_host_cpu_features and CPUCodegen.magic_tuple in numba.core.codegen). Keying it here is redundant.
    "NUMBA_ENABLE_AVX": "ENABLE_AVX",
    "NUMBA_DEBUGINFO": "DEBUGINFO_DEFAULT",
    "NUMBA_ENABLE_PROFILING": "ENABLE_PROFILING",
    "NUMBA_EXTEND_VARIABLE_LIFETIMES": "EXTEND_VARIABLE_LIFETIMES",
    "NUMBA_LLVM_REFPRUNE_PASS": "LLVM_REFPRUNE_PASS",
    "NUMBA_LLVM_REFPRUNE_FLAGS": "LLVM_REFPRUNE_FLAGS",
    # numba 0.66 and later choose the type of the NRT incref and decref calls from these two settings when numba is
    # imported (print_nrt_stack in numba.core.runtime.nrtdynmod). A change of either setting inside a running process
    # therefore changes the key, while those calls keep the type chosen at import.
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

# test_every_setting_numba_reads_is_classified fails when a numba release reads a setting in neither table.
#: Every other ``NUMBA_*`` setting that numba 0.63 to 0.68 reads, with the reason it is not keyed.
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
    """Warning that numbox compiles a function without an on-disk cache, because its key cannot hold a jit option.

    One of these options causes it:

    - A ``parallel`` dict, which numba empties as it parses it.
    - A pipeline class other than numba's default.
    - A function given as the ``inline`` option, which numba calls as a cost model.
    - A type in ``locals`` whose name numba builds from an object's address.
    - Any other value that the key has no stable text for.

    numba then compiles the function again in every process. numbox warns once per process for each function and
    option, and the warning names both.
    """


class _Unkeyable(Exception):
    """Exception that the key computation raises for a jit option that the key cannot hold.

    The cache wrappers treat it as a cache miss, save nothing, and warn with :class:`UncachedWarning`.
    """


# ---- The settings ----

def _setting_value(value) -> str:
    """Return the text that the key holds for one setting's value.

    numba's ``NUMBA_OPT`` is an ``int`` that keeps the value it was given, such as ``max``, in ``_raw_value``, and the
    text is that value. A bool and an int that numba treats as equal have the same text.
    """
    raw = getattr(value, "_raw_value", None)
    if raw is not None:
        value = raw
    if isinstance(value, int):
        value = int(value)
    return repr(value)


def _current_settings() -> tuple:
    """Return each keyed setting with the text of its current value in ``numba.config``."""
    # The values come from numba.config, not from os.environ. numba copies the NUMBA_* environment variables into
    # numba.config when it is imported, together with the settings of a .numba_config.yaml file in the working
    # directory if there is one. It copies them again when a compile starts, if one of them changed:
    # CompilerBase.__init__ calls config.reload_config, which runs _EnvReloader.update.
    return tuple((env, _setting_value(getattr(numba_config, attr, None))) for env, attr in KEYED_SETTINGS.items())


def _settings_canon() -> str:
    """Return the settings part of the key, with the current values, as one string.

    :mod:`numbox.core.variable.compile_kernel` and :mod:`numbox.core.work.builder` also include it in the digests
    that name the kernels and the derive functions they generate.
    """
    return repr(list(_current_settings()))


# ---- The flags ----

#: How numba's text for a type names an object of this process by its address: a jitclass's types, a deferred type, and
#: the default repr of a function or dispatcher inside a type.
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+|jitclass\.[\w.]*#[0-9a-fA-F]+|DeferredType#\d+")


def _type_text(value, where) -> str:
    """Return the text that the key holds for a type given in ``locals``: numba's text for the type.

    Raises :class:`_Unkeyable` if that text names an object by its address, or if ``value`` is neither a string nor a
    numba type. ``where`` describes the value in the message.
    """
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, NumbaType):
        text = str(value)
        if _ADDRESS.search(text):
            raise _Unkeyable(f"{where} is the numba type {text}, which names an object by its address")
        return f"numbatype({type(value).__module__}.{type(value).__qualname__}:{text})"
    raise _Unkeyable(f"{where} is of type {type(value).__name__}, which has no stable form")


def _flag_value_text(value, where) -> str:
    """Return the text that the key holds for the value of one flag that numba compiles under.

    The key writes a plain value as its ``repr``, a set of fast-math flags in sorted order, each parallel option
    separately, and an inlining choice as its string. Raises :class:`_Unkeyable` for a function given as the inlining
    cost model and for a value of any other type. ``where`` describes the value in the message.
    """
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
    """Return the flags part of the key: the decorator kind, the flags numba compiles under and the ``locals`` types.

    Raises :class:`_Unkeyable` for a ``parallel`` dict, for a pipeline class other than numba's default, and for a
    flag or a type that the key cannot hold.
    """
    options = compiler.targetoptions
    # numba empties a parallel dict as it parses it, so a later compile given the same dict would use numba's defaults.
    if isinstance(options.get("parallel"), dict):
        raise _Unkeyable("the jit flag parallel is a dict, which numba empties as it parses it")
    pipeline = compiler.pipeline_class
    # A pipeline class is code, and the key does not hold code.
    if pipeline is not Compiler:
        name = pipeline.__qualname__ if isinstance(pipeline, type) else f"an instance of {type(pipeline).__qualname__}"
        raise _Unkeyable(f"the pipeline class is {name}, not numba's default")
    given = compiler.locals or {}
    # numba's SSA pass adds a name for each variable in locals that it renames: the variable's name, a dot and a
    # number, typed as the variable. The key leaves those names out, because the given type determines them. A name
    # with a dot whose part before the last dot is not in locals is the caller's own, and the key keeps it.
    locals_ = sorted((name, ty) for name, ty in given.items() if name.rpartition(".")[0] not in given)
    flags = Flags()
    # The flags come from numba's own parse of the options, followed by the changes that numba's compiler makes to
    # them, including the changes for a C callback. When numba compiles the function inside a caller's compile, the
    # parse takes nrt, fastmath, error_model and forceinline from the caller wherever the options leave them unset.
    compiler.targetdescr.options.parse_as_flags(flags, options)
    flags = compiler._customize_flags(flags)
    values = ";".join(
        f"{name}={_flag_value_text(value, f'the jit flag {name}')}" for name, value in sorted(flags.values().items())
    )
    types = ",".join(f"{name}:{_type_text(ty, 'a type in locals')}" for name, ty in locals_)
    return f"kind={kind};flags=[{values}];locals=[{types}]"


def _key_part(keyed) -> tuple:
    """Return numbox's part of the key: numbox's version, the flags text and the settings text."""
    kind, compiler = keyed
    return (numbox.__version__, _flags_text(kind, compiler), _settings_canon())


def explain_cache_key(func) -> dict:
    """Return numbox's part of ``func``'s cache key, with the values that the next cache lookup in this process uses.

    ``func`` is a dispatcher or a ``@cfunc`` whose cache numbox keys, the object that ``cres`` returns, or the
    ``.as_func`` of a ``@proxy`` function. numbox keys the functions that it compiles with caching enabled, and the
    functions decorated with :func:`numbox_njit` or :func:`numbox_cfunc` and ``cache=True``. The result maps
    ``"numbox"`` to numbox's version, ``"flags"`` to the flags as the key holds them, and ``"settings"`` to the text of
    each keyed setting's value. The flags are those of a call from Python; a jitted caller can compile the function
    under other options. Two results that differ belong to two different keys.

    Raises ``ValueError`` if numbox does not key ``func``'s cache, or if numbox compiles ``func`` without an on-disk
    cache.
    """
    func = getattr(func, "_numbox_dispatcher", func)
    cache = getattr(func, "_cache", None)
    keyed = getattr(cache, "_numbox_keyed", None)
    if keyed is None:
        raise ValueError(
            f"{func!r} has no cache numbox keys: it is cached under numba's key alone, or not cached; decorate it "
            "with numbox_njit or numbox_cfunc and cache=True"
        )
    try:
        flags = _flags_text(*keyed)
    except _Unkeyable as exc:
        raise ValueError(f"{func!r} compiles without an on-disk cache: {exc}") from None
    return {"numbox": numbox.__version__, "flags": flags, "settings": dict(_current_settings())}


# ---- The opt-in ----

# Each thread keeps its own stack of marks, so decorations on two threads cannot pop each other's marks.
_keying = threading.local()


@contextlib.contextmanager
def _keyed(func):
    """Mark ``func`` while the context is active, so that numbox keys the cache that numba sets up for it."""
    stack = getattr(_keying, "funcs", None)
    if stack is None:
        stack = _keying.funcs = []
    stack.append(func)
    try:
        yield
    finally:
        stack.pop()


def _keys(py_func) -> bool:
    """Return whether numbox keys the cache of ``py_func``: a numbox module's function, or one numbox is decorating."""
    module = getattr(py_func, "__module__", None) or ""
    if module == "numbox" or module.startswith("numbox."):
        return True
    return any(f is py_func for f in getattr(_keying, "funcs", ()))


def numbox_njit(*args, **kwargs):
    """Decorate a function as ``numba.njit`` does, and add numbox's part to the key under which numba caches it.

    The decorator takes numba's arguments unchanged, including signatures and options. With ``cache=True``, numba
    caches the function under its own key with numbox's part appended: numbox's version, the jit options and the
    settings in :data:`KEYED_SETTINGS`. The part applies to an eager compile, for which the decorator is given a
    signature, and to a lazy compile at the first call. Without ``cache=True``, the decorator behaves exactly as
    ``numba.njit``.
    """
    func = args[0] if args else kwargs.get("signature_or_function")
    # numba sets up the function's cache inside the decorator call, so the function is marked for the length of that
    # call: here for the form numbox_njit(func, ...), and in wrap for the form numbox_njit(...)(func).
    if isinstance(func, FunctionType):
        with _keyed(func):
            return njit(*args, **kwargs)
    decorator = njit(*args, **kwargs)

    def wrap(func):
        with _keyed(func):
            return decorator(func)
    return wrap


def numbox_cfunc(*args, **kwargs):
    """Decorate a function as ``numba.cfunc`` does, and add numbox's part to the key under which numba caches it.

    The decorator takes numba's arguments unchanged. With ``cache=True``, numbox's part also holds the decorator kind,
    so numba caches the C callback apart from a :func:`numbox_njit` of the same function. Under numba's key alone, an
    eager ``numba.njit`` given the same signature as a numba ``Signature`` object shares the callback's entry. A call to
    such an njit after it loaded the callback's entry crashes the process with a segmentation fault. Without
    ``cache=True``, the decorator behaves exactly as ``numba.cfunc``.
    """
    decorator = cfunc(*args, **kwargs)

    def wrap(func):
        with _keyed(func):
            return decorator(func)
    return wrap


# ---- The hooks ----

# Each pair of a cache's name and a reason is reported once per process.
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
        # numba computes the key twice for each compile: at the lookup before it (Cache._load_overload) and at the
        # save after it (Cache._save_overload). The compile in between can refresh numba.config from os.environ,
        # because CompilerBase.__init__ calls config.reload_config, so the saved key holds the refreshed values. The
        # code may not match them. numba parsed the compile's flags before the refresh
        # (_FunctionCompiler._compile_core), and it built the target machine, with NUMBA_OPT and NUMBA_ENABLE_AVX, at
        # the first decoration in the process (CPUCodegen._init). A setting changed inside a running process can
        # therefore be saved under a key that does not describe the code.
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
        # numba calls enable_caching at decoration if cache=True. A cache that numbox keys records the decorator kind
        # and the compiler, from which _flags_text reads the options at each lookup and save.
        if _keys(py_func_of(self)):
            self._cache._numbox_keyed = (kind, self._compiler)
    return enable_caching


def _install():
    """Wrap numba's cache methods and the cache set-up of numba's two decorators, once per process.

    If the install fails, numba is left unchanged, and a later fresh import of this module tries again.
    """
    # A reload of this module returns here, and the installed hooks keep working, because importlib.reload reuses the
    # module's namespace and the hooks read _keying from it. A fresh import after the module was removed from
    # sys.modules also returns here. Its decorators then mark functions in a new _keying that the installed hooks never
    # read, so those functions are cached under numba's key alone, while numbox's own functions remain keyed.
    if getattr(caching, "_numbox_cache_key", False):
        return
    # Every attribute is looked up before the first assignment, so on a numba version that lacks one of them, nothing
    # is installed.
    wrapped = [
        (caching.Cache, "_index_key", _wrap_index_key(caching.Cache._index_key)),
        (caching.Cache, "load_overload", _wrap_load_overload(caching.Cache.load_overload)),
        (caching.Cache, "save_overload", _wrap_save_overload(caching.Cache.save_overload)),
        (Dispatcher, "enable_caching", _wrap_enable_caching(Dispatcher.enable_caching, "njit", lambda d: d.py_func)),
        # The kind keeps a C callback's entries apart from an njit's. numba keys an eager njit by the signature as it
        # is given (Dispatcher.compile). It keys a C callback by the Signature that it builds (CFunc.__init__). The two
        # therefore share entries if the njit is given a Signature object. numba compiles a C callback without the
        # CPython wrapper (no_cpython_wrapper in _CFuncCompiler._customize_flags), so a call to an njit that loaded
        # the callback's entry crashes the process.
        (CFunc, "enable_caching", _wrap_enable_caching(CFunc.enable_caching, "cfunc", lambda c: c._pyfunc)),
    ]
    applied = []
    try:
        for owner, name, replacement in wrapped:
            original = getattr(owner, name)
            setattr(owner, name, replacement)
            applied.append((owner, name, original))
        # The marker is set after the last assignment. An assignment that fails, including the marker's assignment,
        # restores the attributes assigned before it.
        caching._numbox_cache_key = True
    except BaseException:
        for owner, name, original in reversed(applied):
            setattr(owner, name, original)
        raise


try:
    _install()
except Exception as _exc:
    warnings.warn(
        f"numbox: could not add its part to numba's cache key ({_exc!r}); what numbox compiles is cached under "
        "numba's key alone",
        RuntimeWarning,
    )
