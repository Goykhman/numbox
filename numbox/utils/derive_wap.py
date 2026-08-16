"""First-class function values whose exceptions propagate out of the call.

The mechanism
-------------

numba lowers a first-class ``FunctionType`` call two ways and picks between them on
the function model's ``jit_addr`` slot. A populated slot selects the numba calling
convention plus ``return_status_propagate``, so an exception unwinds into the caller.
An empty slot selects the C wrapper, which numba documents as not supporting
exceptions: it discards the exception, zero-fills the return value and reports the
failure only as an unraisable on stderr.

numba fills the slot for a ``Dispatcher`` and leaves it empty for everything else,
``_get_jit_address`` returning 0. A bare compile result therefore arrives at a call
site with no entry point that could carry an exception out.

:class:`DeriveWAP` captures the compile result's numba-callconv entry point, and
:class:`DeriveFunctionType` populates ``jit_addr`` from it on both unboxing and
constant lowering. A value of this type carries a usable entry point wherever it
goes.

Using it
--------

:func:`numbox.utils.highlevel.cres` mints these values, so a `cres`-compiled function
is already one::

    @cres(float64(float64))
    def f(x):
        if x < 0:
            raise ValueError("negative")
        return x

Filling the slot is necessary but not sufficient: the *call site* decides which
convention it emits, and one that reads the wrapper address unconditionally gets the
C wrapper however the slot is filled. A consumer opts in by reading ``jit_addr`` and
emitting ``call_conv.call_function`` plus ``return_status_propagate`` when it is
non-null; numba's own version of that is
``numba.core.lowering.Lower.__call_first_class_function_pointer``.

``numbox.core.work.work._call_derive`` is the worked example in this repository, and
:doc:`numbox.core.work` walks through it. It selects the propagating convention at
compile time for a :class:`DeriveFunctionType`, and tests the slot at runtime for a
plain ``FunctionType``, which numba does populate for an njit dispatcher passed as a
``FunctionType``-typed argument.

Limits
------

:meth:`DeriveFunctionType.can_convert_to` documents the one direction that does not
survive: a value handed from Python into a parameter *declared* as a plain
``FunctionType`` degrades to the C convention on the way in.

Where numba has no such slot :func:`jit_addr_supported` is false, the whole mechanism
is inert and ``cres`` returns a plain ``CompileResultWAP``.

Standing on numba
-----------------

The decorators below are numba's public extension API, save for ``lower_constant``,
which ``numba.extending`` does not re-export. What they register against is not:
``FunctionModel``, ``CompileResultWAP``, ``Conversion``, ``box_function_type`` and
``lower_get_wrapper_address`` all sit outside ``numba.extending``, and the constant
lowering drives ``context.declare_function`` and ``context.active_code_library``
directly. What does hold is that no numba internals are patched: nothing here
replaces numba behaviour, it only registers against it.
"""
from numba.core import cgutils, types
from numba.core.imputils import lower_cast, lower_constant
from numba.core.typeconv import Conversion
from numba.core.types.function_type import CompileResultWAP, FunctionType
from numba.core.typing.typeof import typeof_impl
from numba.experimental.function_type import (
    FunctionModel, box_function_type, lower_get_wrapper_address,
)
from numba.extending import NativeValue, box, register_model, unbox

from numbox.core.configurations import function_struct_has


__all__ = ["DeriveFunctionType", "DeriveWAP", "jit_addr_supported", "rewrap_derive"]


#: Where :func:`rewrap_derive` parks the upgraded wrapper on the object it upgrades,
#: so that the address stored in ``py_addr`` stays backed for as long as the caller
#: holds the original.
_UPGRADED_ATTR = "_numbox_derive_wap"


def jit_addr_supported() -> bool:
    """Whether the running numba exposes the ``jit_addr`` slot.

    The slot was added to ``FunctionModel`` in numba 0.61. Where it is absent the
    mechanism has nothing to populate, so numbox leaves first-class calls to numba
    rather than shipping a half-installed variant.
    """
    return function_struct_has("jit_addr")


class DeriveFunctionType(FunctionType):
    """First-class function type whose values always carry a populated ``jit_addr``.

    Kept distinct from ``FunctionType`` so that ``_call_derive`` can select the
    propagating calling convention at compile time, with no runtime branch, for
    every derive numbox itself compiled.
    """

    def can_convert_to(self, typingctx, other):
        """Permit passing a :class:`DeriveWAP` where a plain ``FunctionType`` of
        the same signature is declared.

        Such a call site keeps working unchanged, which is the point, but it
        keeps its *old* behaviour too: crossing the Python boundary into a
        parameter declared as a plain ``FunctionType`` degrades the value to the
        C convention, so a `derive` supplied that way still discards its
        exception. Reaching the same declared type by a cast within jitted scope
        does not degrade it, because :func:`lower_cast_derive_to_function_type`
        is an identity on a shared ``FunctionModel`` and the populated
        ``jit_addr`` survives. One source-level call site therefore has two
        different exception semantics depending on where the value came from.
        """
        if type(other) is FunctionType and other.signature == self.signature:
            return Conversion.safe
        return None


@lower_cast(DeriveFunctionType, FunctionType)
def lower_cast_derive_to_function_type(context, builder, fromty, toty, val):
    """Identity cast. Both types use ``FunctionModel``, so the struct is shared."""
    return val


class DeriveWAP(CompileResultWAP):
    """``CompileResultWAP`` that also captures the numba-callconv entry point.

    ``CompileResultWAP`` records only the cfunc wrapper address, which is the
    entry point that cannot carry an exception out.
    """

    def __init__(self, cres):
        super().__init__(cres)
        self.jit_address = cres.library.get_pointer_to_function(
            cres.fndesc.llvm_func_name)
        if self.jit_address <= 0:
            raise ValueError(
                f"no callconv entry point for {cres.fndesc.llvm_func_name}")


@typeof_impl.register(DeriveWAP)
def typeof_derive_wap(val, c):
    return DeriveFunctionType(val.signature())


register_model(DeriveFunctionType)(FunctionModel)


def _get_derive_jit_address(func, sig):
    """Resolve the callconv entry address during unboxing.

    Called from jitted unboxing code with the GIL held, mirroring how numba's own
    first-class unboxing reaches back into Python. Raising here surfaces at the
    unbox boundary rather than corrupting the struct, so a value that is not a
    :class:`DeriveWAP` fails loudly instead of producing a null ``jit_addr`` that
    would silently reinstate the swallow.
    """
    if isinstance(func, DeriveWAP):
        return func.jit_address
    raise TypeError(
        f"DeriveFunctionType value must be a DeriveWAP, got {type(func)}")


def _lower_get_derive_jit_address(context, builder, func, sig):
    """Emit the call to :func:`_get_derive_jit_address`.

    Follows numba's ``_lower_get_address`` with ``failure_mode='return_null'``:
    a null result returns NULL from the unboxing function, which propagates the
    Python exception rather than swallowing it.
    """
    pyapi = context.get_python_api(builder)
    modname = context.insert_const_string(builder.module, __name__)
    mod = pyapi.import_module(modname)
    fn = pyapi.object_getattr_string(mod, "_get_derive_jit_address")
    pyapi.decref(mod)
    sig_obj = pyapi.unserialize(pyapi.serialize_object(sig))
    addr = pyapi.call_function_objargs(fn, (func, sig_obj))
    # `fn` and `sig_obj` are new references and `addr` does not borrow from either,
    # so release them before the branch below: the null path returns early, and
    # unboxing runs on every call for a derive passed as an argument.
    pyapi.decref(fn)
    pyapi.decref(sig_obj)
    with builder.if_then(cgutils.is_null(builder, addr), likely=False):
        builder.ret(pyapi.get_null_object())
    return addr


@unbox(DeriveFunctionType)
def unbox_derive_function_type(typ, obj, c):
    typ = typ.get_precise()
    sfunc = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    addr = lower_get_wrapper_address(
        c.context, c.builder, obj, typ.signature, failure_mode="return_null")
    sfunc.c_addr = c.pyapi.long_as_voidptr(addr)
    c.pyapi.decref(addr)

    llty = c.context.get_value_type(types.voidptr)
    sfunc.py_addr = c.builder.ptrtoint(obj, llty)

    addr = _lower_get_derive_jit_address(c.context, c.builder, obj, typ.signature)
    sfunc.jit_addr = c.pyapi.long_as_voidptr(addr)
    c.pyapi.decref(addr)

    return NativeValue(sfunc._getvalue())


@box(DeriveFunctionType)
def box_derive_function_type(typ, val, c):
    return box_function_type(typ, val, c)


@lower_constant(DeriveFunctionType)
def lower_constant_derive_function_type(context, builder, typ, pyval):
    """Lower a derive reached as a compile-time constant from jitted code.

    ``c_addr`` and ``py_addr`` follow numba's own lowering for a
    ``WrapperAddressProtocol`` value. ``jit_addr`` is declared symbolically and
    the compile result's library is linked in, which is the pattern numba uses
    for its ``Dispatcher -> FunctionType`` cast, and which is what makes the
    entry point resolve as a symbol rather than as a bare pointer.

    Resolving it as a symbol is also what makes a constant-lowered derive
    cacheable at all: the dead ``c_addr``/``py_addr`` globals are eliminated
    before numba scans the final module, so a caller that only calls the derive
    reports no dynamic globals and caches. A baked address would leave them live
    and numba would refuse to cache the caller rather than store an address that
    is randomized per process.

    That caching is not free: the caller's cached binary binds the derive's
    code, so editing the derive's body in another module serves a stale binary
    until the cache is cleared.

    There is deliberately no fallback to a baked address. A value of this type
    always takes the propagating call, so a `jit_addr` that failed to resolve
    would be called unconditionally, and failing the compilation is the only
    honest outcome.
    """
    typ = typ.get_precise()
    assert typ.check_signature(pyval.signature())
    sfunc = cgutils.create_struct_proxy(typ)(context, builder)
    sfunc.c_addr = context.add_dynamic_addr(
        builder, pyval.__wrapper_address__(), info=str(typ))
    sfunc.py_addr = context.add_dynamic_addr(
        builder, id(pyval), info=type(pyval).__name__)
    fn = context.declare_function(builder.module, pyval.cres.fndesc)
    sfunc.jit_addr = builder.bitcast(fn, context.get_value_type(types.voidptr))
    context.active_code_library.add_linking_library(pyval.cres.library)
    return sfunc._getvalue()


def rewrap_derive(derive):
    """Upgrade a foreign ``CompileResultWAP`` so its exceptions propagate.

    A derive compiled by numbox's own :func:`~numbox.utils.highlevel.cres` is
    already a :class:`DeriveWAP`. One built directly against numba is not, and
    would keep the swallowing convention. The compile result it carries is all
    that is needed to upgrade it.

    Anything else, including ``None``, is returned unchanged so callers can apply
    this unconditionally.

    On a numba without the ``jit_addr`` slot there is nothing to upgrade into: the
    struct has no field to hold the entry point, so producing a
    :class:`DeriveWAP` there would only yield a value that cannot be unboxed.

    The upgraded wrapper is memoized onto the object it upgrades, and that is
    required rather than an optimization. ``py_addr`` holds the derive's address
    without taking a reference, so a wrapper minted fresh per call would be freed
    as soon as the caller returned, leaving every `Work` built from it pointing at
    released memory. Hanging it off the original ties its lifetime to the object
    the caller already holds, which is the lifetime the address assumed all along.
    """
    if not jit_addr_supported():
        return derive
    if not isinstance(derive, CompileResultWAP) or isinstance(derive, DeriveWAP):
        return derive
    upgraded = getattr(derive, _UPGRADED_ATTR, None)
    if upgraded is None:
        upgraded = DeriveWAP(derive.cres)
        setattr(derive, _UPGRADED_ATTR, upgraded)
    return upgraded
