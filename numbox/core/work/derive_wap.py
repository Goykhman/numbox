"""First-class derive values that carry a numba-callconv entry point.

Why this exists
---------------

A ``Work.derive`` is invoked through a numba first-class ``FunctionType`` call.
numba's own lowering for such a call branches on the function model's
``jit_addr`` slot: non-null selects the numba calling convention plus
``return_status_propagate``, so an exception unwinds into the caller; null
selects the C wrapper, which numba documents as not supporting exceptions and
which discards the exception, zero-fills the return value and reports the
failure only as an unraisable on stderr.

:func:`numbox.utils.highlevel.cres` produces a ``CompileResultWAP``, and numba's
``_get_jit_address`` yields an address only for a ``Dispatcher``, returning 0 for
everything else. A cres-backed derive therefore always took the swallowing path:
``calculate()`` returned normally, ``data`` was left zero-filled, and ``derived``
was set anyway so the wrong value was cached permanently. For ``unicode_type``
data the zeroed string struct has a NULL data pointer, so reading it back from
Python segfaulted the interpreter.

:class:`DeriveWAP` captures the compile result's numba-callconv entry point and
:class:`DeriveFunctionType` populates ``jit_addr`` from it on both unboxing and
constant lowering, which lets :func:`numbox.core.work.work._call_derive` emit the
propagating call.

Two details worth knowing
-------------------------

``_call_derive`` never consulted ``jit_addr`` at all: it read function-struct
slot 0 unconditionally. So the swallow on the ``Work`` path came from numbox's
own intrinsic rather than from numba's lowering, and an njit dispatcher passed as
an explicitly ``FunctionType``-typed argument swallowed there even though numba
had populated its ``jit_addr``. That is why ``_call_derive`` also branches at
runtime for plain ``FunctionType`` fields rather than selecting purely on the
numbox-owned type.

The decorators below are numba's public extension API, save for
``lower_constant``, which ``numba.extending`` does not re-export, but what they
register against is not: ``FunctionModel``, ``CompileResultWAP``, ``Conversion``,
``box_function_type`` and ``lower_get_wrapper_address`` all sit outside
``numba.extending``, ``JIT_ADDR_SLOT`` hardcodes ``FunctionModel``'s field order,
and the constant lowering drives ``context.declare_function`` and
``context.active_code_library`` directly. What does hold is that no numba
internals are patched: nothing here replaces numba behaviour, it only registers
against it.
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

from numbox.core.configurations import function_struct_size


__all__ = ["DeriveFunctionType", "DeriveWAP", "jit_addr_supported", "rewrap_derive"]


#: Index of ``jit_addr`` in the ``FunctionModel`` struct ``(c_addr, py_addr, jit_addr)``.
JIT_ADDR_SLOT = 2

#: Where :func:`rewrap_derive` parks the upgraded wrapper on the object it upgrades,
#: so that the address stored in ``py_addr`` stays backed for as long as the caller
#: holds the original.
_UPGRADED_ATTR = "_numbox_derive_wap"


def jit_addr_supported() -> bool:
    """Whether the running numba exposes the ``jit_addr`` slot.

    The slot was added to ``FunctionModel`` in numba 0.61. On 0.60 the mechanism
    does not exist, so numbox keeps its previous behaviour there rather than
    shipping a half-installed variant.
    """
    return function_struct_size >= 3


class DeriveFunctionType(FunctionType):
    """First-class function type whose values always carry a populated ``jit_addr``.

    Kept distinct from ``FunctionType`` so that ``_call_derive`` can select the
    propagating calling convention at compile time, with no runtime branch, for
    every derive numbox itself compiled.
    """

    def can_convert_to(self, typingctx, other):
        """Permit passing a :class:`DeriveWAP` where a plain ``FunctionType`` of
        the same signature is declared.

        The value degrades to the C convention there, which is the behaviour
        those call sites had before, so an explicit-signature ``njit`` that names
        ``FunctionType`` keeps working unchanged.
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
