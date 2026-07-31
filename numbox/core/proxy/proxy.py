import hashlib
import inspect
import struct
import warnings
from llvmlite import binding as ll
from llvmlite import ir
from numba import cfunc, njit
from numba.core import cgutils
from numba.core.errors import TypingError
from numba.core.types.function_type import CompileResultWAP
from numba.core.typing.templates import Signature
from numba.extending import intrinsic, overload
from types import FunctionType as PyFunctionType
from typing import List, Optional, Tuple

from numbox.core.configurations import _PROXY_CACHE_STRICT_ENV, _strict_cache_mode
from numbox.utils.fingerprint import (
    _Unfingerprintable, _fingerprint_function, _fingerprint_function_best_effort,
)
from numbox.utils.standard import make_params_strings


_ALIAS_PREFIX = "numbox_pxy_"


def make_proxy_name(name):
    return f'__{name}'


def _body_fingerprint(func):
    """Content fingerprint of ``func``'s body, for alias disambiguation.

    Reuses the deep walker so bytecode, constants, default arguments, closure
    cell values and referenced-global values all count. The common numbox
    binding body references the ``@intrinsic`` ``_call_lib_func``, which has no
    canonical form, so the strict walker raises; the best-effort walker then
    still captures the constants/closure/defaults/globals it *can* canonicalize
    (substituting an opaque type placeholder for the rest). Two bodies that
    differ only in a captured value -- a factory over per-instance C symbol
    names, or a literal-only redefinition -- therefore get distinct aliases,
    not one collapsed to bytecode alone.
    """
    try:
        return _fingerprint_function(func, set())
    except (_Unfingerprintable, RecursionError):
        return _fingerprint_function_best_effort(func)


def _stable_cfunc_alias(func, main_sig, jit_options=None):
    """Deterministic, process-stable LLVM symbol name for ``func``'s cfunc wrapper.

    numba mangles the wrapper name (``fndesc.llvm_cfunc_wrapper_name``) with a
    process-local unique-id abi-tag (``v<N>`` from ``FunctionIdentity.unique_id``,
    an ``itertools.count``). That tag is not part of the numba cache key, so two
    processes give the same function different wrapper names. Persisting a
    reference to that name — in this proxy, or in any ``cache=True`` caller that
    inlines the ``inline='always'`` proxy — lets concurrently-built caches pair a
    body object defining ``v<Na>`` with a caller referencing ``v<Nb>``, which
    aborts on load with ``LLVM ERROR: Symbol not found: cfunc...``. Referencing
    this deterministic alias instead (resolved per-process via
    ``llvmlite.binding.add_symbol``) keeps cached references valid
    across processes.

    The alias folds a body fingerprint alongside ``module + qualname + sig``:
    without it, two different bodies sharing one identity (factory-made
    same-qualname closures, in-process redefinition, or fork twins on a shared
    cache) collapse onto one alias, and the last-writer-wins ``add_symbol``
    silently rebinds callers to the wrong body. Folding the body gives each body
    its own alias. A consequence is that a body/signature change renames the
    alias, so a warm ``cache=True`` caller in another file (numba's cache key is
    callee-blind) references a symbol this process never registered. The load-time
    guard heals this automatically -- it discards that one cache entry and
    recompiles it, warning once with :class:`StaleProxyCacheWarning`; no manual
    cache clearing is needed (set ``NUMBOX_PROXY_CACHE_STRICT`` to raise
    :class:`StaleProxyCacheError` before the heal instead). The one variant that
    leaves the alias unchanged, an absent ``proxy_if_available`` binding, is
    covered by a diagnostic trap (see ``_register_absent_alias_trap``); the guard
    treats such an alias as stale too, so a warm caller reaches the same clean
    typing error a cold cache gives rather than the trap's swallowed-in-a-cfunc
    zero.

    The resolved ``jit_options`` are folded too. They govern the machine code
    ``njit(sig, **jit_options)`` emits, but appear nowhere in the identity above,
    so two decorations of one body differing only in a flag -- ``error_model``,
    ``fastmath``, ``boundscheck`` -- collapsed onto a single alias and
    last-writer-wins ``add_symbol`` sent every caller to whichever compiled last.
    That is silent and total: a caller of the ``numpy`` error-model binding ran
    the ``python`` one, so a division by zero returned 0.0 instead of inf with the
    exception swallowed at the cfunc boundary.
    """
    raw = (
        f"{func.__module__ or ''}.{func.__qualname__}.{main_sig}."
        f"{_body_fingerprint(func)}."
        f"{sorted((jit_options or {}).items(), key=repr)!r}"
    ).encode("utf-8")
    safe_name = "".join(c if c.isascii() and c.isalnum() else "_" for c in func.__name__)
    return f"{_ALIAS_PREFIX}{safe_name}_{hashlib.sha256(raw).hexdigest()[:16]}"


def _call_proxied_alias(context, builder, main_sig, alias_name, args):
    """Emit a call to the proxied body's process-stable cfunc alias.

    Factored out of the generated codegen so the exec'd template stays short:
    every line above its ``@njit`` raises the minimum source line a ``@proxy``
    function may occupy (the ``co_firstlineno`` cache anchor).
    """
    func_ty_ll = ir.FunctionType(
        context.get_data_type(main_sig.return_type),
        [context.get_data_type(arg) for arg in main_sig.args],
    )
    f = cgutils.get_or_insert_function(builder.module, func_ty_ll, alias_name)
    return builder.call(f, args)


# The cfunc traps below must outlive the add_symbol registrations that publish
# their addresses, so keep a process-lifetime reference to each.
_PROXY_TRAP_KEEPALIVE = []

# Aliases currently standing for an ABSENT binding. A trap registration makes the alias resolve without
# there being a body behind it, so presence alone stops meaning "safe to call" -- see _stale_proxy_aliases.
_ABSENT_ALIASES = set()


def _register_absent_alias_trap(func, main_sig, jit_options=None):
    """Register a diagnostic trap under the alias ``proxy(sig)(func)`` would use.

    A ``cache=True`` caller compiled when the binding was present bakes that
    alias into its cached object; numba's cache key is callee-blind, so the
    caller cache-hits even in a process where ``proxy_if_available`` took the
    absent (stub) path and registered no real body. With no symbol there the
    baked extern call is a null-pointer jump -- a bare SIGSEGV with zero
    diagnostics. Registering a cfunc that raises a named ``RuntimeError`` under
    the same alias turns that into a clear message on stderr instead. The trap
    matches ``main_sig``'s arity with plain positional parameters (the cfunc
    wrapper the caller invokes is positional, even for ``Omitted`` bindings).
    """
    alias = _stable_cfunc_alias(func, main_sig, jit_options)
    msg = (
        f"numbox @proxy binding {func.__name__!r} is not available in the loaded library "
        f"(C symbol missing), but a cache=True caller compiled when it was present is "
        f"calling it -- clear the numba cache (NUMBA_CACHE_DIR) and rebuild."
    )
    trap_params = ", ".join(f"_a{i}" for i in range(len(main_sig.args)))
    trap_ns = {"cfunc": cfunc, "main_sig": main_sig, "_trap_msg": msg}
    exec(  # nosec B102 - internal codegen of a fixed-shape trap body
        f"def _trap({trap_params}):\n"
        f"    raise RuntimeError(_trap_msg)\n"
        f"_trap_cfunc = cfunc(main_sig)(_trap)\n",
        trap_ns,
    )
    _PROXY_TRAP_KEEPALIVE.append(trap_ns["_trap_cfunc"])
    _ABSENT_ALIASES.add(alias)
    ll.add_symbol(alias, trap_ns["_trap_cfunc"].address)


def proxy(sig, jit_options: Optional[dict] = None):
    """ Create a proxy for the decorated function `func` with the given signature(s) `sig`.

    The original function `func` will be eagerly JIT-compiled with the given signature(s).
    A proxy with the name `func_proxy_name` will be created to call `func` in the LLVM scope.
    The original function's variable will be bound to the proxy, i.e., calling the decorated
    function will call the proxy.

    The proxy is a JIT-compiled wrap that invokes the intrinsic that *declares* the `func`
    and calls it with the original arguments. Declaration instructions are relatively cheap
    to statically link into (potential) caller's LLVM code, which is the main motivation behind
    this decorator.

    Machine code for `func` can be cached when so specified in `jit_options`, in which case its
    JIT-compilation will load the `func` into the LLVM scope. Caching option is the other major
    motivation for this decorator, without the need to cache one can avoid static linking
    of the callee's LLVM code into the caller's by simply ignoring the former.

    In case when more than one signature is provided as the `sig` parameter, it is assumed
    that the first signature is the 'main' one while the other ones are supplied to
    allow for the `Omitted` types with default values for (some of) the parameters.

    The returned dispatcher also exposes ``.as_func``: a ``CompileResultWAP``
    for the main signature. Cacheable as a called jitted function (via the
    dispatcher); passable as a function-type argument (via ``.as_func``).

    See tests for some examples of the use cases.
    """
    main_sig = isinstance(sig, Signature) and sig or isinstance(sig, (List, Tuple)) and sig[0]
    jit_options = isinstance(jit_options, dict) and jit_options or {}
    jit_opts = jit_options.copy()
    jit_opts.update(jit_opts, inline='always')

    def wrap(func):
        assert isinstance(func, PyFunctionType)
        func_jit = njit(sig, **jit_options)(func)
        cres = func_jit.get_compile_result(main_sig)
        # Register a process-stable alias for the body's cfunc wrapper and reference
        # that instead of numba's process-local ``v<uid>`` name (see _stable_cfunc_alias).
        cfunc_alias = _stable_cfunc_alias(func, main_sig, jit_options)
        _ABSENT_ALIASES.discard(cfunc_alias)
        ll.add_symbol(cfunc_alias, cres.library.get_pointer_to_function(cres.fndesc.llvm_cfunc_wrapper_name))
        func_args_str, func_names_args_str = make_params_strings(func)
        func_proxy_name = make_proxy_name(func.__name__)
        # The alias resolution lives in _call_proxied_alias so this generated
        # source stays short: every line above the @njit raises the minimum source
        # line a @proxy function may occupy (the co_firstlineno cache anchor below).
        code_txt = f"""
@intrinsic
def _{func_proxy_name}(typingctx, {func_names_args_str}):
    def codegen(context, builder, signature, args):
        return _call_proxied_alias(context, builder, main_sig, "{cfunc_alias}", args)
    return main_sig, codegen

@njit(sig, **jit_opts)
def {func_proxy_name}({func_args_str}):
    return _{func_proxy_name}({func_names_args_str})
"""
        ns = {
            **inspect.getmodule(func).__dict__,
            **{
                'cgutils': cgutils, 'intrinsic': intrinsic, 'ir': ir, 'jit_opts': jit_opts, 'njit': njit,
                'sig': sig, 'main_sig': main_sig, '_call_proxied_alias': _call_proxied_alias
            }
        }
        if ns.get(func_proxy_name) is not None:
            raise ValueError(f"Name {func_proxy_name} in module {inspect.getmodule(func)} is reserved")
        # Anchor the wrapper at func's source file: prepend blank lines so the
        # wrapper's @njit decorator lands at func.__code__.co_firstlineno (the
        # user's @proxy decorator line). See docs/numbox.core.proxy.rst —
        # section "Cache-anchor mechanism" — for the design rationale + the
        # findsource-finds-@-in-docstring hazard this avoids.
        code_lines = code_txt.split('\n')
        njit_lineno_in_txt = next(
            i + 1 for i, line in enumerate(code_lines) if line.startswith('@njit(')
        )
        co_firstlineno = func.__code__.co_firstlineno
        if co_firstlineno < njit_lineno_in_txt:
            raise ValueError(
                f"@proxy function {func.__name__!r} is defined at line {co_firstlineno} of "
                f"{inspect.getfile(func)}, above the cache anchor's minimum line "
                f"{njit_lineno_in_txt}; the generated @njit cannot be anchored to its "
                f"co_firstlineno (a negative prepend would mis-anchor it). Move the "
                f"function further down in the file."
            )
        prepend = co_firstlineno - njit_lineno_in_txt
        prefixed = '\n' * prepend + code_txt
        code = compile(prefixed, inspect.getfile(func), mode='exec')
        exec(code, ns)  # nosec B102 - JIT codegen of internal source
        dispatcher = ns[func_proxy_name]
        dispatcher.as_func = CompileResultWAP(cres)
        # Tag the dispatcher with its process-stable alias so the fingerprint
        # walker can identify a @proxy binding by that alias instead of recursing
        # into its wrapper's @intrinsic (which has no canonical form) -- otherwise
        # a callback that calls a proxied binding is un-fingerprintable and its
        # cache digest silently degrades to a coarse fallback.
        dispatcher._numbox_proxy_alias = cfunc_alias
        return dispatcher
    return wrap


def proxy_if_available(lib, sig, jit_options: Optional[dict] = None):
    """Like ``proxy(sig, jit_options=...)``, but stubs out the wrapper if
    the C symbol matching ``func.__name__`` is absent from ``lib``.

    Use for binding sets that target multiple library versions where some
    symbols only exist in newer releases. Python callers get a stub that raises
    ``NotImplementedError`` instead of a confusing LLVM link error at call
    time; ``@njit`` callers get a ``TypingError`` naming the binding and the
    missing-symbol cause at typing time (the untyped stub would otherwise
    surface an untyped-global failure that names the binding but not why it
    is unusable). Parallel to ``cres_if_available`` in
    :mod:`numbox.utils.highlevel`.

    The stub does NOT expose ``.as_func`` — a function-value handle is
    meaningless without an underlying jitted body, and a stub one would
    have to either raise on attribute access (ugly) or pretend to be a
    function value (worse). Callers that pass ``.as_func`` to function-
    type arguments must guard the access::

        if hasattr(my_binding, "as_func"):
            use(my_binding.as_func)
    """
    def _(func):
        if hasattr(lib, func.__name__):
            return proxy(sig, jit_options=jit_options)(func)

        name = func.__name__
        main_sig = sig if isinstance(sig, Signature) else sig[0]
        # A warm cache=True caller from a process where the binding WAS present
        # will cache-hit and call the alias; register a loud trap under it so that
        # is a named error rather than a null-pointer SIGSEGV. See the helper.
        _register_absent_alias_trap(func, main_sig, jit_options)

        def stub(*args, **_kwargs):
            raise NotImplementedError(f"{name} is not available")
        stub.__name__ = make_proxy_name(name)
        stub.__qualname__ = func.__qualname__
        stub.__doc__ = func.__doc__

        # The stub is untyped, so a bare @njit call to it fails typing with an
        # untyped-global error that names the binding but not why it is unusable.
        # Register an @overload that raises a clear, named TypingError instead.
        @overload(stub)
        def _unavailable(*args, **kwargs):
            raise TypingError(
                f"{name} is not available in the loaded library "
                f"(C symbol missing)"
            )
        return stub
    return _


class StaleProxyCacheWarning(RuntimeWarning):
    """A cached numba function referenced a ``@proxy`` alias this process never registered, so the entry was
    discarded and recompiled.

    To make it fatal instead, filter it programmatically --
    ``warnings.filterwarnings("error", category=StaleProxyCacheWarning)`` -- or, under pytest, with
    ``-W error::numbox.core.proxy.proxy.StaleProxyCacheWarning``. The interpreter's own ``-W`` cannot name this
    class: CLI warning filters are resolved before the import system can reach numbox, so the category is rejected
    as an invalid module name. The blanket ``-W error::RuntimeWarning`` does work.
    """


class UnvalidatedProxyCacheWarning(RuntimeWarning):
    """A cached payload could not be read, so it was loaded without being checked for stale ``@proxy``
    aliases. Reported once per process: the condition is a property of the environment, not of one entry."""


class StaleProxyCacheError(RuntimeError):
    """Strict mode found a stale ``@proxy`` alias and refused to heal it.

    Raised in place of :class:`StaleProxyCacheWarning` when ``NUMBOX_PROXY_CACHE_STRICT`` is set. The load
    is aborted *before* the discard-and-recompile, so the stale entry is left on disk for inspection rather
    than being replaced -- the opposite of the default heal, which silently overwrites it. Clear the numba
    cache (``NUMBA_CACHE_DIR``) or unset the knob to recover.
    """


class UnvalidatedProxyCacheError(RuntimeError):
    """Strict mode could not read a cache payload and refused to load it unchecked.

    Raised in place of :class:`UnvalidatedProxyCacheWarning` when ``NUMBOX_PROXY_CACHE_STRICT`` is set and
    unpacking the payload raises -- a shape a future numba might introduce, or a malformed object. The
    default degrades this to loading unchecked; strict mode makes it loud so a validation gap cannot hide.
    A well-formed object that the reader parses cleanly but finds no stale alias in is *not* a gap and is
    loaded normally, in strict mode as by default: it cannot be told apart from a healthy object that merely
    carries the alias prefix in a string constant.
    """


_ELF_MAGIC = b"\x7fELF"
# 32- and 64-bit, both byte orders, and both widths of the fat/universal container.
_MACHO_MAGICS = frozenset((0xFEEDFACE, 0xFEEDFACF, 0xCEFAEDFE, 0xCFFAEDFE,
                           0xCAFEBABE, 0xBEBAFECA, 0xCAFEBABF, 0xBFBAFECA))
_unvalidated_reported = False


def _elf_undefined_symbols(obj):
    """Names of the undefined (imported) symbols of an ELF64 little-endian relocatable object.

    A 32-bit or big-endian ELF is left unread and yields nothing, the same contract ``_undefined_symbols``
    documents for a container format it cannot parse.
    """
    und = set()
    if len(obj) < 0x40 or obj[4] != 2 or obj[5] != 1:
        return und
    (e_shoff,) = struct.unpack_from("<Q", obj, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", obj, 0x3A)
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        (sh_type,) = struct.unpack_from("<I", obj, sh + 4)
        if sh_type != 2:  # SHT_SYMTAB
            continue
        sym_off, sym_size = struct.unpack_from("<QQ", obj, sh + 24)
        (sh_link,) = struct.unpack_from("<I", obj, sh + 40)
        (sh_entsize,) = struct.unpack_from("<Q", obj, sh + 56)
        (str_off,) = struct.unpack_from("<Q", obj, e_shoff + sh_link * e_shentsize + 24)
        for j in range(sym_size // sh_entsize):
            (st_name,) = struct.unpack_from("<I", obj, sym_off + j * sh_entsize)
            (st_shndx,) = struct.unpack_from("<H", obj, sym_off + j * sh_entsize + 6)
            if st_shndx == 0 and st_name:  # SHN_UNDEF and named
                end = obj.index(b"\x00", str_off + st_name)
                und.add(obj[str_off + st_name:end].decode("utf-8", "replace"))
    return und


def _macho_undefined_symbols(obj):
    """Names of the undefined external symbols of a 64-bit little-endian Mach-O relocatable object.

    Mach-O prefixes C symbols with an underscore, and it is stripped here so the names come back as numbox
    registered them. A reader that skipped that step would test ``_numbox_pxy_...`` against the prefix, match
    nothing, and quietly pass every stale object -- the failure mode this whole guard exists to prevent, in the
    one place it would look like success. Other widths, byte orders and the fat container are left unread, as
    in the ELF reader.
    """
    und = set()
    if len(obj) < 32 or struct.unpack_from("<I", obj, 0)[0] != 0xFEEDFACF:
        return und
    (ncmds,) = struct.unpack_from("<I", obj, 16)
    off = 32
    for _ in range(ncmds):
        if off + 8 > len(obj):
            break
        cmd, cmdsize = struct.unpack_from("<II", obj, off)
        # Both bounds are about a corrupt or foreign object, not a real one: cmdsize is what advances the
        # walk, so a zero would spin here forever, and this runs on every cache load. Fail-open cannot
        # rescue a hang the way it rescues an exception.
        if cmdsize < 8 or off + cmdsize > len(obj):
            break
        if cmd == 0x2:  # LC_SYMTAB
            symoff, nsyms, stroff = struct.unpack_from("<III", obj, off + 8)
            nsyms = min(nsyms, max(0, len(obj) - symoff) // 16)
            for i in range(nsyms):
                n_strx, n_type = struct.unpack_from("<IB", obj, symoff + i * 16)
                if n_type & 0xE0:  # N_STAB: a debugging entry, never a link-time import
                    continue
                if (n_type & 0x0E) or not (n_type & 0x01):  # not N_UNDF, or not N_EXT
                    continue
                end = obj.index(b"\x00", stroff + n_strx)
                name = obj[stroff + n_strx:end].decode("utf-8", "replace")
                und.add(name[1:] if name.startswith("_") else name)
        off += cmdsize
    return und


def _coff_undefined_symbols(obj):
    """COFF/PE (Windows). Not read yet -- see ``_undefined_symbols`` for what an empty result means here."""
    return set()


def _undefined_symbols(object_code):
    """Undefined (imported) symbol names of one relocatable object, dispatched on its container format.

    A format this cannot read yields an empty set, which degrades the caller to "nothing to check" rather than
    to a false verdict: an unreadable object is loaded exactly as it was before this guard existed. COFF is the
    fallback because a COFF object begins with a machine-type word rather than a distinctive magic.
    """
    if object_code[:4] == _ELF_MAGIC:
        return _elf_undefined_symbols(object_code)
    if len(object_code) >= 4 and struct.unpack_from("<I", object_code, 0)[0] in _MACHO_MAGICS:
        return _macho_undefined_symbols(object_code)
    return _coff_undefined_symbols(object_code)


def _stale_proxy_aliases(payload, libdata_of_payload):
    """``@proxy`` aliases a serialized code library references that this process cannot correctly call.

    A non-empty result means the payload must not be loaded. It is *every* stale alias the object imports,
    not the first -- an object that calls two edited proxies carries two unregistered names, and the caller
    joins the whole list into one warning, so the diagnostic never undercounts within an object. (What the
    guard cannot enumerate is stale entries never loaded: a proxied body whose own source changed recompiles
    fresh, overwriting its cache file, and is never handed to ``rebuild`` -- correctly drawing no warning.)
    ``ll.address_of_symbol`` is nearly an exact oracle: these names live only in llvmlite's explicit-symbol
    map, and because the alias encodes the body, signature and jit options, a resolving alias is normally the
    right body to call. The exception is an alias standing for an absent ``proxy_if_available`` binding, which
    resolves to a trap holding no body at all. Loading that object is worse than discarding it -- the trap's
    error is raised inside a ``@cfunc``, where numba swallows it and returns zero, so the caller silently
    computes on a wrong value, while discarding yields the same clean typing error a cold cache gives.

    The payload is unpacked inside the handler rather than by the caller, because its shape is precisely the
    thing a future numba might change.

    Under ``NUMBOX_PROXY_CACHE_STRICT`` a payload that cannot be *read* raises :class:`UnvalidatedProxyCacheError`
    instead of degrading to unchecked. A well-formed object the reader parses cleanly is not forced to raise
    even when it yields no stale alias: an object that carries the prefix only as a string constant parses
    exactly like one that carries it as an unresolved import would if the reader went blind, so there is no
    way to escalate the second without also aborting the first -- a healthy, entirely non-proxy cached
    function. Reader blindness is caught where it can be told apart, in CI, by a test that asserts the reader
    parses the objects numba actually emits; it is not guessed at here on live loads.
    """
    strict = _strict_cache_mode()
    try:
        _name, kind, data = libdata_of_payload(payload)
        if kind != "object":
            return []
        object_code = data[0]
        if _ALIAS_PREFIX.encode() not in object_code:
            return []  # fast path: the alias string appears nowhere in the object
        return sorted(
            s for s in _undefined_symbols(object_code)
            if s.startswith(_ALIAS_PREFIX) and (s in _ABSENT_ALIASES or not ll.address_of_symbol(s))
        )
    except Exception as exc:
        # Fail open: this runs on EVERY numba cache load in the process, so a surprise -- a future numba
        # changing the payload shape, a malformed object -- must degrade to "no validation", never to
        # breaking unrelated caching. Under the strict knob a caller has asked for the opposite: surface it.
        if strict:
            raise UnvalidatedProxyCacheError(
                f"numbox: could not check a numba cache payload for stale @proxy aliases ({exc!r}); "
                f"{_PROXY_CACHE_STRICT_ENV} is set, so the load was aborted rather than proceeding unchecked"
            ) from exc
        # Say so once, though: abandoning the check silently is indistinguishable from passing it, and what
        # it silently restores is the crash this guard exists to prevent.
        global _unvalidated_reported
        if not _unvalidated_reported:
            _unvalidated_reported = True
            warnings.warn(
                f"numbox: could not check a numba cache payload for stale @proxy aliases ({exc!r}); it was "
                "loaded unchecked. Later payloads in this process are checked, but not reported again.",
                UnvalidatedProxyCacheWarning, stacklevel=3,
            )
        return []


def _guarded_rebuild(orig_rebuild, libdata_of_payload):
    """Wrap a numba ``CacheImpl.rebuild`` so a payload is validated while it is still plain bytes.

    This is the last point upstream of the execution engine. Once ``unserialize_library`` runs, the object's
    externals are resolved in one batch and a single unregistered name zeroes every relocation in it, which is
    unrecoverable and kills the process without a diagnostic. Returning ``None`` is numba's own contract for a
    cache miss, so the stale entry is discarded and recompiled in place. Under
    ``NUMBOX_PROXY_CACHE_STRICT`` the discard is refused and :class:`StaleProxyCacheError` is raised instead,
    before the ``None`` return, so the stale entry stays on disk for inspection.
    """
    def rebuild(self, target_context, payload):
        stale = _stale_proxy_aliases(payload, libdata_of_payload)
        if stale:
            filename_base = getattr(self, 'filename_base', '?')
            if _strict_cache_mode():
                raise StaleProxyCacheError(
                    f"numbox: {filename_base} references @proxy cfunc alias(es) this process cannot call "
                    f"({', '.join(stale)}); {_PROXY_CACHE_STRICT_ENV} is set, so the stale entry was left on disk for "
                    "inspection instead of being discarded and recompiled"
                )
            warnings.warn(
                "numbox: discarding a stale numba cache entry and recompiling: "
                f"{filename_base} references @proxy cfunc alias(es) this process "
                "cannot call (the proxied body, signature or jit_options changed since the caller was "
                "cached, or the binding is no longer available): "
                + ", ".join(stale),
                StaleProxyCacheWarning,
                stacklevel=3,
            )
            return None
        return orig_rebuild(self, target_context, payload)
    return rebuild


def _install_cache_alias_guard():
    """Validate every numba cache load in this process against the registered ``@proxy`` aliases.

    Ordering is guaranteed without any coordination: a caller can only have been compiled against a ``@proxy``
    binding whose module it imports, and importing that module imports this one. A cached function that
    references no alias may load before numbox exists and needs no validation.

    Every lookup happens before the first assignment, so a numba that does not offer one of these classes leaves
    the process with no guard rather than half of one.
    """
    from numba.core import caching
    if getattr(caching, "_numbox_proxy_alias_guard", False):
        return
    wrapped = [
        (caching.CompileResultCacheImpl, _guarded_rebuild(caching.CompileResultCacheImpl.rebuild,
                                                          lambda payload: payload[0])),
        (caching.CodeLibraryCacheImpl, _guarded_rebuild(caching.CodeLibraryCacheImpl.rebuild,
                                                        lambda payload: payload)),
    ]
    caching._numbox_proxy_alias_guard = True
    for impl, rebuild in wrapped:
        impl.rebuild = rebuild


try:
    _install_cache_alias_guard()
except Exception:
    # Fail open, as the per-load validator does. Losing the guard costs protection against a stale cache
    # entry; letting an install-time surprise escape would cost the whole library its importability.
    pass
