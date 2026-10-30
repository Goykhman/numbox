"""A decorator that compiles a function once and lets jitted callers call that copy instead of embedding their own.

:func:`proxy` compiles the decorated function at import. A jitted caller of the result holds a declaration of the
compiled function and a call to it, not a copy of its code. :func:`proxy_if_available` does the same for a binding to a
C symbol that the loaded library may lack. numbox registers each compiled function under an alias that is the same in
every process. It also checks the entries of numba's CPU cache as numba loads them. If an entry refers to an alias
that this process cannot call, numbox discards the entry, and numba compiles the caller again. The warning and error
classes report what the check finds, and :class:`AliasCollisionWarning` reports two functions that computed the same
alias.
"""
import hashlib
import inspect
import struct
import threading
import warnings
from llvmlite import binding as ll
from llvmlite import ir
from numba import cfunc, njit
from numba.core import cgutils
from numba.core.errors import TypingError
from numba.core.typing.templates import Signature
from numba.extending import intrinsic, overload
from types import FunctionType as PyFunctionType
from typing import List, Optional, Tuple

from numbox.core.cache_key import _keyed, numbox_njit
from numbox.core.configurations import _PROXY_CACHE_STRICT_ENV, _strict_cache_mode
from numbox.utils.derive_wap import DeriveWAP
from numbox.utils.fingerprint import (
    _ABSENT_ALIASES, _Unfingerprintable, _fingerprint_function, _fingerprint_function_best_effort,
    _opaque_values_match,
)
from numbox.utils.standard import make_params_strings


_ALIAS_PREFIX = "numbox_pxy_"


def make_proxy_name(name):
    """Return the name of the wrapper that :func:`proxy` generates for a function named ``name``.

    The name is ``name`` with two leading underscores. The dispatcher that :func:`proxy` returns has this name, and so
    does the stub that :func:`proxy_if_available` returns for an absent symbol. numba's error messages use this name,
    not the function's own.
    """
    return f'__{name}'


def _body_fingerprint(func, opaque=None):
    """Return a fingerprint of ``func``'s body that is the same in every process.

    The fingerprint covers the bytecode, the constants, the default arguments, the values of the closure cells and the
    values of the globals that the body reads. The docstring is one of the constants, so an edit to the docstring
    alone changes the fingerprint. A value that has no form stable across processes enters as a placeholder for its
    type. If ``opaque`` is a list, each such value is appended to it. :func:`_publish_cfunc_alias` compares those
    values to distinguish a second body from the same body registered twice.
    """
    # The strict walker raises on the first value that has no canonical form. The body of a binding usually calls one
    # such value, the @intrinsic _call_lib_func. The best-effort walker then folds every value that it can
    # canonicalize, and it folds a placeholder for each other value. Two bodies that differ only in a captured value
    # therefore still have different fingerprints: closures that a factory makes over different C symbol names, or a
    # redefinition that changes only a literal. Unlike the strict walker, the best-effort walker does not fold the
    # module attributes that the body reads, such as cfg.SCALE.
    try:
        return _fingerprint_function(func, set())
    except (_Unfingerprintable, RecursionError):
        return _fingerprint_function_best_effort(func, opaque)


def _stable_cfunc_alias(func, main_sig, jit_options=None, opaque=None):
    """Return the alias of ``func``'s compiled body: an LLVM symbol name that is the same in every process.

    The alias is ``numbox_pxy_``, then the function's name with each character other than an ASCII letter or digit
    replaced by an underscore, then an underscore and the first 16 hex digits of a SHA-256 hash. The hash covers the
    module, the qualified name, the main signature, the fingerprint from :func:`_body_fingerprint` and the resolved
    ``jit_options``. A change to any of them renames the alias, and the cache check then discards each cached caller
    that refers to the old alias. The alias of an absent :func:`proxy_if_available` binding is the one that the present
    binding would have. ``opaque`` is passed on to :func:`_body_fingerprint`.
    """
    # numba names the body's C entry point with a tag from a per-process counter, the "v<N>" of
    # FunctionIdentity.unique_id. numba's cache key leaves the tag out, so one function can have a different name in
    # each process. If the cached wrapper and the cached callers that inline it held numba's name, a cache that two
    # processes built could pair a body that defines one name with a caller that refers to the other. The load would
    # then abort with "LLVM ERROR: Symbol not found". Callers therefore refer to this alias, which each process
    # registers for its own compiled body with llvmlite.binding.add_symbol.
    #
    # The fingerprint separates bodies that share a module, a qualified name and a signature: closures that a factory
    # makes under one qualified name, a redefinition in one process, or processes related by fork() that share a cache.
    # If such bodies shared an alias, every caller would call the body registered last, because add_symbol replaces an
    # earlier registration of the same name.
    #
    # The jit options are hashed because they change the machine code. Without them, two decorations of one body that
    # differ only in error_model, fastmath or boundscheck would share an alias, and every caller would call the body
    # compiled last. A caller of the binding compiled under the numpy error model could then call the body compiled
    # under the python model. Under the python model, a division by zero raises, so the C entry point would discard the
    # exception and return 0.0 instead of inf.
    raw = (
        f"{func.__module__ or ''}.{func.__qualname__}.{main_sig}."
        f"{_body_fingerprint(func, opaque)}."
        f"{sorted((jit_options or {}).items(), key=repr)!r}"
    ).encode("utf-8")
    safe_name = "".join(c if c.isascii() and c.isalnum() else "_" for c in func.__name__)
    return f"{_ALIAS_PREFIX}{safe_name}_{hashlib.sha256(raw).hexdigest()[:16]}"


def _call_proxied_alias(context, builder, main_sig, alias_name, args):
    """Declare the body's alias in the module being built, and emit a call to it with ``args``.

    The declaration takes its return type and its argument types from ``main_sig``.
    """
    func_ty_ll = ir.FunctionType(
        context.get_data_type(main_sig.return_type),
        [context.get_data_type(arg) for arg in main_sig.args],
    )
    f = cgutils.get_or_insert_function(builder.module, func_ty_ll, alias_name)
    return builder.call(f, args)


# This list holds a reference to each trap cfunc below for the life of the process. numba does not unload compiled code
# when a CFunc object is collected, so the address that add_symbol publishes remains valid without the reference. The
# list is a precaution against a numba release that would unload such code.
_PROXY_TRAP_KEEPALIVE = []

# _ABSENT_ALIASES, imported above from numbox.utils.fingerprint, holds the aliases that no longer identify one body.
# They are the alias of an absent binding, under which a trap is registered, and both names involved in a collision.
# A trap's alias resolves to the trap, not to a proxied body, so an alias that resolves is not necessarily safe to call
# (see _stale_proxy_aliases). The walker in numbox.utils.fingerprint reads the set, because it must not fold a retired
# alias into the fingerprint of a body that captured it. The set is defined in that module, which this module already
# imports, so the walker needs no import of this module. This module only adds to the set and never rebinds it, so
# both modules see the same set.

# _PUBLISHED_ALIASES maps each alias that this process published to the address published under it and to a witness,
# the values that the body's fingerprint could not canonicalize. With the witness, _publish_cfunc_alias distinguishes
# a second body that computes the same alias from the same body registered again. A second body would otherwise rebind
# every caller compiled against the first. A trap is recorded with the witness None, which no body matches.
_PUBLISHED_ALIASES = {}


# _ALIAS_LOCK makes two read-modify-write sequences on _PUBLISHED_ALIASES and _ABSENT_ALIASES atomic. Both run in plain
# Python at decoration, after numba has released its compiler lock, so nothing else serializes them.
# _publish_cfunc_alias names a colliding body after the number of published aliases and then publishes that name.
# Without the lock, two threads that both counted before either published would assign the same name to two bodies.
# _register_absent_alias_trap decides on a read of _PUBLISHED_ALIASES and writes the trap's alias several statements
# later. Without the lock, a body published between that read and the write would be overwritten by the trap.
#
# The trap's cfunc compiles between that decision and the write. The trap therefore reads the registry again under the
# lock instead of holding the lock during the compile. Only the read that decides and the writes that depend on it need
# to be atomic. The compile does not need to be atomic.
_ALIAS_LOCK = threading.Lock()


def _a_body_answers(alias):
    """Return whether a proxied body, not a trap, is published under ``alias`` in this process."""
    published, witness = _PUBLISHED_ALIASES.get(alias, (None, None))
    return published is not None and witness is not None


class AliasCollisionWarning(RuntimeWarning):
    """Warning that two different proxied functions computed the same alias, so that the alias identifies neither.

    numbox publishes the second function under a name of its own, used only in this process, and treats both names as
    stale in cache entries. Cached callers of either function therefore compile again in every process. The usual cause
    is a captured value that differs between processes, such as a ctypes function built from a raw address, which
    numbox leaves out of the alias. Access the C function as an attribute of the loaded library instead. The warning
    also reports a :func:`proxy_if_available` binding that is absent in one decoration and present in another in the
    same process. In that case, the message says which body a call through the alias runs.
    """


def _publish_cfunc_alias(alias, address, opaque):
    """Publish ``address`` under ``alias``, and return the name that callers must refer to.

    If nothing is published under ``alias`` in this process, or the body published under it has the same ``opaque``
    values, the function publishes ``address`` under ``alias`` and returns ``alias``. If another body or a trap is
    published under it, the function publishes ``address`` under a new name used only in this process and returns that
    name. It also adds both names to ``_ABSENT_ALIASES`` and warns with :class:`AliasCollisionWarning`.
    """
    with _ALIAS_LOCK:
        published, witness = _PUBLISHED_ALIASES.get(alias, (None, None))
        # llvmlite.binding.add_symbol replaces the address of a symbol for the whole process without a warning, so the
        # last registration of a name takes effect. A second body published under the alias of the first would
        # therefore rebind every caller compiled against the first. Those callers would then return wrong results
        # without any message. The same body is often registered again, after a module reload or when a factory is
        # called twice with equal arguments, and replacing the address is then correct. Two bodies with equal
        # fingerprints can differ only in their opaque values, so equal opaque values mean one body compiled twice.
        if published is None or (witness is not None and _opaque_values_match(witness, opaque)):
            _PUBLISHED_ALIASES[alias] = (address, tuple(opaque))
            ll.add_symbol(alias, address)
            return alias
        # The two bodies differ only in values that have no form stable across processes, such as a ctypes function
        # built from a raw address (see _ctypes_func_key). The second body is published under a name of its own, so both
        # bodies remain callable in this process. The shared alias no longer identifies a body, because another process
        # may build the two bodies in the other order. It is therefore retired: the cache check discards a cached
        # caller of either body instead of loading it against whichever body is published under the alias. Both bodies
        # lose cross-process caching as a result.
        #
        # The name of the second body is retired too. It is made from the number of aliases that this process has
        # published, so it does not identify the body either. The same body receives another name in a process that
        # published a different number of aliases first. A process that builds the pair in the other order assigns
        # this name to the other body. No name derived from the content can separate the two bodies, because their
        # fingerprints are equal. The name is therefore valid only for callers compiled in this process, and the cache
        # check discards a cached caller that refers to it.
        #
        # A retirement lasts for the life of the process, because its cause is a property of the two bodies, not of the
        # order in which this process built them. If the body published under the alias registers again, the alias is
        # published again with the new address. If the other body registers again, it receives another new name.
        # Another process may still build the two in the other order. The branch above therefore does not change
        # _ABSENT_ALIASES, and no code removes an alias from the set. A first publication needs no removal, because
        # every alias in the set already has an entry in _PUBLISHED_ALIASES.
        #
        # A trap can also be published under the alias, with a witness that no body matches. A body that computes a
        # trap's alias therefore receives a name of its own, and the alias stays retired. A cached caller of that alias
        # was compiled against the binding that is absent here, so a call through the alias must run the trap, not this
        # body.
        _ABSENT_ALIASES.add(alias)
        distinct = f"{alias}_c{len(_PUBLISHED_ALIASES)}"
        _ABSENT_ALIASES.add(distinct)
        _PUBLISHED_ALIASES[distinct] = (address, tuple(opaque))
        ll.add_symbol(distinct, address)
    # When a trap is published under the alias, the message names the trap instead of reporting a second body that
    # does not exist.
    if witness is None:
        reason = (
            f"an absent-binding trap already answers to the alias {alias}, so this body is "
            f"published as {distinct} instead and the alias stays retired. A warm caller that "
            f"reaches that name has to find the trap and its named error, because the process "
            f"that cached it may be one where the binding is missing."
        )
    else:
        reason = (
            f"two @proxy bodies fingerprint alike and both reached the alias {alias}; it names "
            f"neither of them now, so callers of both recompile in every process. This one is "
            f"published as {distinct} instead. A body whose only difference is a value with no "
            f"process-stable identity, such as a ctypes pointer built from a raw address, "
            f"cannot be told apart by a fingerprint that has to mean the same thing next run."
        )
    warnings.warn(reason, AliasCollisionWarning, stacklevel=3)
    return distinct


def _register_absent_alias_trap(func, main_sig, jit_options=None):
    """Register a trap under the alias that :func:`proxy` computes for ``func``, whose C symbol is absent.

    The trap is a ``@cfunc`` of ``main_sig`` that raises a ``RuntimeError`` naming the binding. A caller cached in a
    process where the symbol was present refers to this alias, because numba's cache key does not cover the callees.
    The cache check discards such a caller before numba loads it, because the alias is in ``_ABSENT_ALIASES``. A caller
    calls the trap only if the check did not inspect its entry. Without the trap, the entry would refer to an
    unregistered symbol, and the process would crash without a message. With the trap, the call prints the error to
    stderr and returns zero.

    If a proxied body is already published under the alias in this process, the function registers no trap and warns
    with :class:`AliasCollisionWarning`.
    """
    alias = _stable_cfunc_alias(func, main_sig, jit_options)
    # If a body is already published under this alias in this process, no trap replaces it. The trap raises inside a
    # cfunc wrapper, which discards the exception and returns zero. A trap registered under the alias of a working body
    # would therefore make every call through the alias return zero, without any message. _PUBLISHED_ALIASES exists to
    # prevent that failure. A call through the alias in this process then runs the body of a binding that is present.
    # If the binding is absent in a process, no body is published under its alias there. The function then registers
    # the trap.
    #
    # The registry is read twice, and the second read decides whether the trap is registered. A compile is slow enough
    # that another thread routinely publishes a body while the trap compiles. The first read happened before
    # the compile began. The trap is therefore compiled first. The registry is then read again under _ALIAS_LOCK,
    # together with the writes that depend on that read. A trap is dropped if a body was published under its alias
    # during the compile. The first read only lets the function skip the compile if a body is already published under
    # the alias, because the second read would then discard the trap.
    yielded_to_body = _a_body_answers(alias)
    if not yielded_to_body:
        msg = (
            f"numbox @proxy binding {func.__name__!r} is not available in the loaded library "
            f"(C symbol missing), but a cache=True caller compiled when it was present is "
            f"calling it -- clear the numba cache (NUMBA_CACHE_DIR) and rebuild."
        )
        # The trap takes plain positional parameters, one for each argument of main_sig, because a caller passes
        # positional arguments to the cfunc wrapper, even for a binding with Omitted signatures.
        trap_params = ", ".join(f"_a{i}" for i in range(len(main_sig.args)))
        trap_ns = {"cfunc": cfunc, "main_sig": main_sig, "_trap_msg": msg}
        exec(  # nosec B102 - internal codegen of a fixed-shape trap body
            f"def _trap({trap_params}):\n"
            f"    raise RuntimeError(_trap_msg)\n"
            f"_trap_cfunc = cfunc(main_sig)(_trap)\n",
            trap_ns,
        )
        with _ALIAS_LOCK:
            yielded_to_body = _a_body_answers(alias)
            if not yielded_to_body:
                _PROXY_TRAP_KEEPALIVE.append(trap_ns["_trap_cfunc"])
                _ABSENT_ALIASES.add(alias)
                ll.add_symbol(alias, trap_ns["_trap_cfunc"].address)
                # The witness None matches no body. A body that computes this alias later is therefore detected, instead
                # of being published under the trap's alias without a warning.
                _PUBLISHED_ALIASES[alias] = (trap_ns["_trap_cfunc"].address, None)
    if yielded_to_body:
        warnings.warn(
            f"the @proxy binding {func.__name__!r} is absent from this library but a body is already "
            f"published under the alias {alias} in this process, so the alias keeps naming that body "
            f"and no trap was installed. A caller compiled here calls the body that is present.",
            AliasCollisionWarning,
            stacklevel=3,
        )


def proxy(sig, jit_options: Optional[dict] = None):
    """Compile the decorated function once, at import, so that jitted callers call it instead of embedding a copy.

    ``sig`` is a numba signature such as ``float64(float64)``, or a list of signatures for a function with default
    arguments. The first signature in a list is the main signature. Each further signature describes a call that omits
    trailing defaulted parameters, and it types each omitted parameter as numba's ``Omitted`` of its default value. A
    signature given as a string fails with an ``AttributeError``. ``jit_options`` is passed to ``njit`` for the
    function and for the wrapper that numbox generates around it. A value that is not a ``dict`` is ignored. The
    function is cached only if ``jit_options`` enables caching, as ``{"cache": True}`` and
    ``numbox.core.configurations.jit_options`` do.

    The decorator returns the dispatcher of the wrapper. The dispatcher's name is ``__`` followed by the function's
    name, and its docstring is the function's docstring. A call through the dispatcher, from Python or from jitted
    code, enters the compiled function through its C entry point, under an alias that numbox registers in every
    process. If the function raises, such a call does not propagate the exception: Python prints it to stderr, and the
    call returns a zero of the return type. The wrapper takes every parameter as positional-or-keyword, so a
    keyword-only parameter can also be passed by position.

    The dispatcher's ``.as_func`` is the compiled function as a first-class function value, a
    :class:`~numbox.utils.derive_wap.DeriveWAP`. It has only the main signature, so a call through it passes every
    parameter. Read it in Python and pass it to a jitted function as an argument, because jitted code cannot read it
    from the dispatcher. An exception raised in the function propagates out of a call through ``.as_func``, unless
    Python passes the value into a parameter that the jitted function declares as a plain ``FunctionType``.

    numbox computes the alias from the function's module, qualified name, code and docstring, the values it captures,
    the main signature and ``jit_options``. An edit to any of these renames the alias. numbox then discards each cached
    caller that refers to the old alias, and numba compiles the caller again. numbox warns once for each discarded entry
    with :class:`StaleProxyCacheWarning`, or raises :class:`StaleProxyCacheError` in the strict mode that
    ``NUMBOX_PROXY_CACHE_STRICT`` enables. A jitted caller that reads ``.as_func`` from a module-level variable embeds a
    copy of the function instead, and the check cannot see that copy. After an edit, such a cached caller can continue
    to run the old code without a warning, so delete its cache files after every edit. If the C symbol of a
    :func:`proxy_if_available` binding is missing, such a caller can return the missing binding's result or crash with
    a segmentation fault.

    Decoration raises a ``ValueError`` if the function's first decorator line is above line 8 of its file, or if its
    module defines a global named ``__`` followed by the function's name. A function that takes ``*args`` or
    ``**kwargs`` cannot be proxied: with a signature for its other parameters, decoration fails with numba's
    signature-mismatch ``TypeError``. Under ``NUMBA_DISABLE_JIT=1``, decoration fails with an ``AttributeError``. The
    documentation page ``numbox.core.proxy`` shows examples.
    """
    # A signature given as a string is neither a Signature nor a list, so main_sig is False, and
    # get_compile_result(main_sig) in wrap raises an AttributeError.
    main_sig = isinstance(sig, Signature) and sig or isinstance(sig, (List, Tuple)) and sig[0]
    jit_options = isinstance(jit_options, dict) and jit_options or {}
    jit_opts = jit_options.copy()
    # The wrapper is compiled with the function's jit options and inline="always", so numba inlines the wrapper into
    # each jitted caller.
    jit_opts.update(jit_opts, inline='always')

    def wrap(func):
        assert isinstance(func, PyFunctionType)
        # The function belongs to the caller's module, not to numbox. It is therefore marked, so that numbox keys its
        # cache (see numbox.core.cache_key).
        with _keyed(func):
            func_jit = njit(sig, **jit_options)(func)
        # Under NUMBA_DISABLE_JIT=1, njit returns the Python function itself, which has no get_compile_result, so the
        # decoration fails here with an AttributeError.
        cres = func_jit.get_compile_result(main_sig)
        # Callers call the body through an alias that is the same in every process, not through numba's name for the
        # body's C entry point. That name contains a tag from a per-process counter (see _stable_cfunc_alias).
        opaque = []
        cfunc_alias = _publish_cfunc_alias(
            _stable_cfunc_alias(func, main_sig, jit_options, opaque),
            cres.library.get_pointer_to_function(cres.fndesc.llvm_cfunc_wrapper_name),
            opaque,
        )
        # make_params_strings refuses *args and **kwargs with a ValueError, but the compile above runs first. numba's
        # signature-mismatch TypeError therefore comes first, and this check is reached only if sig also types the star
        # parameter and the body compiles.
        func_args_str, func_names_args_str = make_params_strings(func)
        func_proxy_name = make_proxy_name(func.__name__)
        # The template defines an @intrinsic and an @njit wrapper that calls it. The intrinsic declares the body's alias
        # in the caller's LLVM module and calls it with the caller's arguments. numba inlines the wrapper into each
        # jitted caller, so the caller's code holds that declaration and call instead of a copy of the body. A
        # declaration and a call are cheap to link into every caller. _call_proxied_alias emits the call outside the
        # template, because each line above the template's @njit raises the lowest line at which a @proxy function may
        # start (see the anchor below).
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
        # In this namespace, the template's njit is numbox_njit, so numbox keys the wrapper's cache as it keys the
        # body's. The template still contains the text "@njit(", which the anchor code below searches for.
        ns = {
            **inspect.getmodule(func).__dict__,
            **{
                'cgutils': cgutils, 'intrinsic': intrinsic, 'ir': ir, 'jit_opts': jit_opts, 'njit': numbox_njit,
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
        # When numba builds a dispatcher, functools.update_wrapper copies the docstring of its py_func, which for the
        # wrapper is None. Both are then assigned the function's docstring, so that help() and autodoc show it. numba
        # copies the docstring again if it rebuilds the dispatcher from py_func. The wrapper's code objects remain
        # unchanged.
        dispatcher.py_func.__doc__ = func.__doc__
        dispatcher.__doc__ = func.__doc__
        dispatcher.as_func = DeriveWAP(cres)
        # The fingerprint walker in numbox.utils.fingerprint folds a proxied function by this alias. Without the tag,
        # it would walk into the wrapper's @intrinsic, which has no canonical form, and the digest of a callback that
        # calls a proxied function would fall back to a coarser form without any warning.
        dispatcher._numbox_proxy_alias = cfunc_alias
        # .as_func carries the alias as well. A function value holds only a compiled address and a mangled name with a
        # per-process counter. Without the tag, the strict walker could not fold the value, and the best-effort walker
        # would fold only its type name, which every first-class function value in the process shares.
        dispatcher.as_func._numbox_proxy_alias = cfunc_alias
        # explain_cache_key reads the cache of the body's dispatcher through this attribute.
        dispatcher.as_func._numbox_dispatcher = func_jit
        return dispatcher
    return wrap


def proxy_if_available(lib, sig, jit_options: Optional[dict] = None):
    """Apply :func:`proxy` if ``lib`` has a C symbol named after the decorated function, and return a stub if not.

    The decorated function must have the C symbol's name, because the check is ``hasattr(lib, name)``. This suits a
    binding to a symbol that only newer versions of a library provide. If the symbol is present, the result is what
    :func:`proxy` returns, ``.as_func`` included. If it is absent, the result is a stub with the name that
    :func:`make_proxy_name` returns, and with the function's qualified name and docstring. A call to the stub from
    Python raises ``NotImplementedError``. A jitted caller of the stub fails to compile with a ``TypingError`` that
    names the binding and says that its C symbol is missing. :func:`~numbox.utils.highlevel.cres_if_available` makes
    the same check for :func:`~numbox.utils.highlevel.cres`.

    The stub has no ``.as_func``, so code that passes the function as a value checks for the attribute first::

        if hasattr(my_binding, "as_func"):
            use(my_binding.as_func)

    A caller cached in a process where the symbol was present refers to the binding's alias. Where the symbol is
    absent, numbox discards that cache entry with a :class:`StaleProxyCacheWarning`, and the recompilation fails with
    the same ``TypingError``.
    """
    def _(func):
        if hasattr(lib, func.__name__):
            return proxy(sig, jit_options=jit_options)(func)

        name = func.__name__
        main_sig = sig if isinstance(sig, Signature) else sig[0]
        # A caller cached in a process where the binding was present refers to its alias. The cache check discards such
        # a caller. If the check could not inspect the entry, the call runs this trap, which prints a named error.
        # Without the trap, the process would crash on an unregistered symbol (see _register_absent_alias_trap).
        _register_absent_alias_trap(func, main_sig, jit_options)

        # If proxy compiled a body whose C symbol is missing, LLVM would abort the process at decoration with "LLVM
        # ERROR: Symbol not found". The decorator therefore returns this stub for an absent symbol. The stub has no
        # .as_func, because a function value needs a compiled body.
        def stub(*args, **_kwargs):
            raise NotImplementedError(f"{name} is not available")
        stub.__name__ = make_proxy_name(name)
        stub.__qualname__ = func.__qualname__
        stub.__doc__ = func.__doc__

        # numba cannot type the stub, a plain Python function, so a jitted call to it would fail with an untyped-global
        # error that names the binding but does not state the reason. The @overload below raises a TypingError that
        # names the missing C symbol instead.
        @overload(stub)
        def _unavailable(*args, **kwargs):
            raise TypingError(
                f"{name} is not available in the loaded library "
                f"(C symbol missing)"
            )
        return stub
    return _


class StaleProxyCacheWarning(RuntimeWarning):
    """Warning that numbox discarded a cached function because it refers to an alias that this process cannot call.

    numba then compiles the function again and saves the new code over the stale entry. An alias is stale if the
    proxied function, its signature or its ``jit_options`` changed after the caller was cached, if a
    :func:`proxy_if_available` binding is absent in this process, or if an alias collision retired the alias. To make
    the warning an error, call ``warnings.filterwarnings("error", category=StaleProxyCacheWarning)``, or run pytest
    with ``-W error::numbox.core.proxy.proxy.StaleProxyCacheWarning``. The interpreter's own ``-W`` option and
    ``PYTHONWARNINGS`` cannot name this class. Python reads them at startup, before site-packages is on the module
    path, so the class cannot be imported, and Python ignores the filter. ``-W error::RuntimeWarning`` works, because
    the class derives from ``RuntimeWarning``.
    """


class UnvalidatedProxyCacheWarning(RuntimeWarning):
    """Warning that numbox could not read a numba cache entry and loaded it without checking it for stale aliases.

    numbox warns once per process, because the cause, such as a numba release that changed the layout of cache
    entries, usually affects every entry. In the strict mode that ``NUMBOX_PROXY_CACHE_STRICT`` enables, numbox raises
    :class:`UnvalidatedProxyCacheError` instead.
    """


class StaleProxyCacheError(RuntimeError):
    """Error that strict mode raises in place of :class:`StaleProxyCacheWarning` for a cache entry with a stale alias.

    ``NUMBOX_PROXY_CACHE_STRICT`` enables strict mode. numbox raises the error before it discards the entry, so the
    stale entry stays on disk for inspection. By default, numbox discards the entry with the warning, and numba
    overwrites it with the code it compiles again. To recover, delete the caller's numba cache files, or unset
    ``NUMBOX_PROXY_CACHE_STRICT``.
    """


class UnvalidatedProxyCacheError(RuntimeError):
    """Error that strict mode raises in place of :class:`UnvalidatedProxyCacheWarning` for an entry it cannot read.

    ``NUMBOX_PROXY_CACHE_STRICT`` enables strict mode. Strict mode refuses to load an entry that numbox could not check,
    so that a gap in the check becomes visible. The cause can be a numba release that changed the layout of cache
    entries, or a damaged object. An entry that numbox reads without error, and in which it finds no stale alias, loads
    normally in strict mode too. A reader that missed the entry's imports could produce the same result, but numbox
    cannot distinguish that case from a valid entry that holds the alias prefix only in a string constant.
    """


_ELF_MAGIC = b"\x7fELF"
# 32- and 64-bit, both byte orders, and both widths of the fat/universal container.
_MACHO_MAGICS = frozenset((0xFEEDFACE, 0xFEEDFACF, 0xCEFAEDFE, 0xCFFAEDFE,
                           0xCAFEBABE, 0xBEBAFECA, 0xCAFEBABF, 0xBFBAFECA))
_unvalidated_reported = False


def _elf_undefined_symbols(obj):
    """Return the names of the undefined symbols of a 64-bit little-endian ELF relocatable object.

    A 32-bit or big-endian ELF object yields an empty set, the result that :func:`_undefined_symbols` returns for any
    format that it cannot read.
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
    """Return the names of the undefined external symbols of a 64-bit little-endian Mach-O relocatable object.

    Mach-O prefixes every C symbol with an underscore. The function returns the names without that underscore, as
    numbox registered them. Other widths, other byte orders and the universal container yield an empty set, as in
    :func:`_elf_undefined_symbols`.
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
        # These bounds matter only for a corrupt or foreign object. cmdsize advances the walk, so a zero would loop
        # forever, and this reader runs on every cache load. The caller's exception handler catches an exception, but it
        # cannot interrupt an endless loop.
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
                # Without the strip, no name would start with _ALIAS_PREFIX, and the check would pass every stale
                # object while it appeared to succeed.
                und.add(name[1:] if name.startswith("_") else name)
        off += cmdsize
    return und


def _coff_undefined_symbols(obj):
    """Return an empty set, because this module does not read COFF objects.

    An empty set means that there is nothing to check (see :func:`_undefined_symbols`).
    """
    # COFF is the object format of Windows, but numba's JIT emits ELF objects there as well.
    # test_the_reader_recognises_the_objects_numba_actually_caches checks on every CI platform that the reader parses
    # the objects that numba caches.
    return set()


def _undefined_symbols(object_code):
    """Return the names of the undefined symbols of one relocatable object, read according to its container format.

    A format that this module cannot read yields an empty set. The caller then finds no stale alias and loads the
    object as it would without the check. An object in an unreadable format therefore never causes a discard.
    """
    if object_code[:4] == _ELF_MAGIC:
        return _elf_undefined_symbols(object_code)
    if len(object_code) >= 4 and struct.unpack_from("<I", object_code, 0)[0] in _MACHO_MAGICS:
        return _macho_undefined_symbols(object_code)
    # COFF is the fallback, because a COFF object begins with a machine-type word, not with a distinctive magic number.
    return _coff_undefined_symbols(object_code)


def _stale_proxy_aliases(payload, libdata_of_payload):
    """Return the sorted ``@proxy`` aliases that a serialized code library imports and that this process cannot call.

    A non-empty result means that numba must not load the payload. The result holds every stale alias that the object
    imports, so the warning names them all. An alias is stale if this process has not registered it, or if it is in
    ``_ABSENT_ALIASES``. ``libdata_of_payload`` extracts the library data from the payload. Only entries that numba
    loads are passed to this function. An entry that numba compiles again because its own source changed produces no
    warning.

    If reading the payload raises, the result is empty, and numbox warns once per process with
    :class:`UnvalidatedProxyCacheWarning`. In strict mode, numbox raises :class:`UnvalidatedProxyCacheError` instead.
    An object that parses without error and imports no stale alias passes, in strict mode too.
    """
    strict = _strict_cache_mode()
    try:
        # The payload is unpacked here, inside the handler, because its shape is what a numba release is most likely
        # to change.
        _name, kind, data = libdata_of_payload(payload)
        if kind != "object":
            return []
        object_code = data[0]
        if _ALIAS_PREFIX.encode() not in object_code:
            return []  # fast path: the alias string appears nowhere in the object
        # The aliases exist only in llvmlite's table of symbols registered with add_symbol, and an alias encodes the
        # body, the signature and the jit options. An alias that resolves therefore normally names the right body. The
        # exception is an alias in _ABSENT_ALIASES. An absent binding's alias resolves to a trap. The trap raises inside
        # a @cfunc, where numba discards the error and returns zero, so the call would return zero instead of a result.
        # A name that a collision retired resolves to a body, but another process may resolve the same name to a
        # different body. If the entry is discarded, the caller of an absent binding is compiled again and fails with
        # the TypingError that an empty cache produces.
        #
        # Strict mode does not raise when the object parses and imports no stale alias. If a reader failed to see the
        # imports of a stale object, the stale object would be indistinguishable from a valid object that holds the
        # prefix only in a string constant. An error raised for the one would therefore also abort the load of the
        # other. test_the_reader_recognises_the_objects_numba_actually_caches checks instead that the reader parses
        # the objects numba emits on each platform.
        return sorted(
            s for s in _undefined_symbols(object_code)
            if s.startswith(_ALIAS_PREFIX) and (s in _ABSENT_ALIASES or not ll.address_of_symbol(s))
        )
    except Exception as exc:
        # This handler runs on every cache load of the two wrapped cache implementations. An unexpected error, from a
        # numba release that changed the payload's shape or from a malformed object, therefore leaves the entry
        # unchecked instead of breaking unrelated caching. In strict mode, the handler raises an error.
        if strict:
            raise UnvalidatedProxyCacheError(
                f"numbox: could not check a numba cache payload for stale @proxy aliases ({exc!r}); "
                f"{_PROXY_CACHE_STRICT_ENV} is set, so the load was aborted rather than proceeding unchecked"
            ) from exc
        # numbox reports the skipped check once per process with this warning. A skip without a warning would be
        # indistinguishable from a passed check. It would also allow a stale entry to crash the process, which the
        # check exists to prevent.
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
    """Return a replacement for numba's ``CacheImpl.rebuild`` that checks a payload for stale aliases before the load.

    If the payload refers to a stale alias, the replacement warns with :class:`StaleProxyCacheWarning` and returns
    ``None``, numba's result for a cache miss. numba then compiles the function again and saves the new code over the
    entry. In strict mode, the replacement raises :class:`StaleProxyCacheError` instead, and the entry stays on disk.
    Otherwise it returns what ``orig_rebuild`` returns.
    """
    def rebuild(self, target_context, payload):
        # The check runs while the payload is still bytes, the last point before numba's execution engine receives it.
        # Once unserialize_library runs, LLVM's runtime linker resolves the object's external symbols in one batch, and
        # a single unregistered name zeroes every relocation in the object. The process then crashes without a message,
        # and no handler can recover from that.
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
    """Install the stale-alias check on the ``rebuild`` of numba's two core cache implementations, once per process.

    The two implementations are ``CompileResultCacheImpl`` and ``CodeLibraryCacheImpl``, and the check then covers
    every cache load through them in this process. A cached caller of a proxied function imports that function's
    module, which imports this one, so the check is installed before such a caller loads. A cached function that refers
    to no alias may load before this module is imported, and it needs no check. numba's ``CUDACacheImpl`` derives from
    ``CacheImpl`` directly, so CUDA cache loads are not checked. If numba lacks one of the two classes, nothing is
    installed.
    """
    from numba.core import caching
    # A second call, after a reload of this module, returns here, so the methods are not wrapped twice.
    if getattr(caching, "_numbox_proxy_alias_guard", False):
        return
    # Every attribute is looked up before the first assignment. If numba lacks one of the classes, the lookup raises
    # before any assignment, so the check is never installed on one class alone.
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
    # If the installation fails, the module still imports, without the check and without a warning. A failure thus
    # leaves caching on, as the check keeps an entry that it cannot read. Without the check, a stale cache entry can
    # crash the process. An exception here would instead make numbox fail to import.
    pass
