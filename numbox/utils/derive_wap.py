"""First-class function values whose exceptions propagate to the jitted code that calls them.

numba can call a first-class function value through two entry points. A call through the numba entry point
propagates an exception to the caller. A call through the C entry point, the kind of wrapper that ``@cfunc`` builds,
discards the exception: numba writes it to stderr as unraisable, and the call returns zero
(`numba issue 8246 <https://github.com/numba/numba/issues/8246>`_). numba records the numba entry point only for a
dispatcher. :class:`DeriveWAP` is a function value that records the numba entry point of a compiled function. Its
numba type, :class:`DeriveFunctionType`, sets that entry point when Python passes the value to jitted code and when
jitted code reads the value as a constant. :func:`~numbox.utils.highlevel.cres` returns such values, and
:func:`rewrap_derive` converts numba's own ``CompileResultWAP`` into one. A value passed from Python to a parameter
declared with the plain ``FunctionType`` loses its numba entry point on the way in, as
:meth:`DeriveFunctionType.can_convert_to` explains.

A value propagates its exception only through a call that takes the numba entry point when the value has one. numba
makes such a call when it calls a function value itself
(``numba.core.lowering.Lower.__call_first_class_function_pointer``). A call that always takes the C entry point
discards the exception, however the value was built. Code that calls a function value inside an ``@intrinsic`` must
therefore read the value's ``jit_addr`` and, if it is set, emit ``call_conv.call_function`` followed by
``return_status_propagate``. ``numbox.core.work.work._call_derive`` is such code. It takes the numba entry point
without a test for a ``DeriveFunctionType``, and it tests ``jit_addr`` at run time for a plain ``FunctionType``, which
numba sets for a dispatcher passed as an argument. The work page describes, in its section on exception handling, when
the exception of a derive propagates.
"""
# The decorators below belong to numba's public extension API, except lower_constant, which numba.extending does not
# re-export. What they register against is not public: FunctionModel, CompileResultWAP, Conversion, box_function_type
# and lower_get_wrapper_address are outside numba.extending, and the constant lowering calls context.declare_function
# and context.active_code_library directly. The module patches nothing in numba; it only registers implementations
# for its own types.
from numba.core import cgutils, types
from numba.core.imputils import lower_cast, lower_constant
from numba.core.typeconv import Conversion
from numba.core.types.function_type import CompileResultWAP, FunctionType
from numba.core.typing.typeof import typeof_impl
from numba.experimental.function_type import (
    FunctionModel, box_function_type, lower_get_wrapper_address,
)
from numba.extending import NativeValue, box, register_model, unbox


__all__ = ["DeriveFunctionType", "DeriveWAP", "rewrap_derive"]


#: The attribute under which :func:`rewrap_derive` stores the upgraded wrapper on the object that it upgrades, so that
#: the address in ``py_addr`` stays valid for as long as the caller holds the original.
_UPGRADED_ATTR = "_numbox_derive_wap"


# DeriveFunctionType must never compare equal to the FunctionType of the same signature. The two hash alike, because
# numba hashes a type on its key, and numba's Type.__eq__ keeps them apart by comparing their classes. numba interns
# every type in a cache keyed by weak references, and a weak reference compares equal when its referent does, so two
# equal types would collapse onto whichever was created first. If the plain type came first, DeriveFunctionType(sig)
# would return it, and every cres value would discard its exception as a plain value does. If this type came first,
# FunctionType(sig) would return it, and every plain function value would fail to unbox.
# test_the_derive_type_stays_distinct_from_the_plain_function_type checks that the two stay apart.
#
# Because the two types differ, numba below 0.68.0 refuses a tuple that holds a DeriveWAP and another kind of
# function value with the same number of arguments. Its unified_function_type in numba/core/utils.py asserts that
# the function types are equal, and the AssertionError has no message. numba 0.68.0 declines to unify the two types
# instead, and the tuple keeps a type for each element. Under python -O the assertion is removed, and numba below
# 0.68.0 gives the whole tuple the type of its first element. A DeriveWAP after a value of another kind is then
# called through its C entry point and discards its exception. A value of another kind after a DeriveWAP fails to
# unbox with the TypeError of _get_derive_jit_address, which names the value's class.
class DeriveFunctionType(FunctionType):
    """The numba type of a :class:`DeriveWAP`: a first-class function type whose values always hold a numba entry point.

    numbox's ``_call_derive`` recognizes this type when it compiles a call, and it calls a value of the type through
    the numba entry point, with no test at run time.
    """

    def can_convert_to(self, typingctx, other):
        """Return a safe conversion to the plain ``FunctionType`` of the same signature, and None for any other type.

        The conversion lets a :class:`DeriveWAP` reach a parameter declared with the plain ``FunctionType``. Whether its
        exception then propagates depends on where the value comes from. Inside jitted code, the cast in
        :func:`lower_cast_derive_to_function_type` leaves the value unchanged, so it keeps its numba entry point. From
        Python, numba unboxes the value as the plain ``FunctionType``, which sets no numba entry point, so a call of
        the value discards its exception. One call site can therefore propagate an exception or discard it, depending
        on where the value comes from.
        """
        if type(other) is FunctionType and other.signature == self.signature:
            return Conversion.safe
        return None


@lower_cast(DeriveFunctionType, FunctionType)
def lower_cast_derive_to_function_type(context, builder, fromty, toty, val):
    """Return the value unchanged: both types use numba's ``FunctionModel``, so the numba entry point stays set."""
    return val


class DeriveWAP(CompileResultWAP):
    """A first-class function value for one compile result that records the numba entry point as well as the C one.

    numba's ``CompileResultWAP`` records only the C entry point, through which an exception cannot propagate. The
    constructor raises ``ValueError`` if the compile result's library gives no address for the numba entry point.
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
    """Return the numba type of a :class:`DeriveWAP`, the ``DeriveFunctionType`` of its signature."""
    return DeriveFunctionType(val.signature())


# A value of the type has numba's FunctionModel, three addresses: c_addr, the C entry point; py_addr, the Python
# object; and jit_addr, the numba entry point. When numba calls a function value, it tests jit_addr at run time. If
# jit_addr is set, the call follows numba's calling convention and then return_status_propagate. If it is null, the
# call goes through the C entry point, which a comment on FunctionModel in numba/experimental/function_type.py
# describes as not supporting exceptions.
register_model(DeriveFunctionType)(FunctionModel)


def _get_derive_jit_address(func, sig):
    """Return the numba entry point of ``func`` while a :class:`DeriveFunctionType` value is unboxed.

    The unboxing code calls this function with the GIL held, as numba's own unboxing of a function value calls back
    into Python. ``TypeError`` is raised if ``func`` is not a :class:`DeriveWAP`. The error stops the unboxing, so
    such a value fails with a message instead of passing with no numba entry point and discarding its exception.
    """
    if isinstance(func, DeriveWAP):
        return func.jit_address
    raise TypeError(
        f"DeriveFunctionType value must be a DeriveWAP, got {type(func)}")


def _lower_get_derive_jit_address(context, builder, func, sig):
    """Emit a call of :func:`_get_derive_jit_address` and return its result, a new reference.

    The code follows numba's ``_lower_get_address`` with ``failure_mode='return_null'``: if the call returns NULL, the
    unboxing function returns NULL, and the Python exception propagates to the caller.
    """
    pyapi = context.get_python_api(builder)
    modname = context.insert_const_string(builder.module, __name__)
    mod = pyapi.import_module(modname)
    fn = pyapi.object_getattr_string(mod, "_get_derive_jit_address")
    pyapi.decref(mod)
    sig_obj = pyapi.unserialize(pyapi.serialize_object(sig))
    addr = pyapi.call_function_objargs(fn, (func, sig_obj))
    # fn and sig_obj are new references, and addr does not borrow from either. They are released before the branch
    # below, because the NULL path returns early, and the unboxing runs on every call that passes a derive as an
    # argument; a reference not released there would leak on each such call.
    pyapi.decref(fn)
    pyapi.decref(sig_obj)
    with builder.if_then(cgutils.is_null(builder, addr), likely=False):
        builder.ret(pyapi.get_null_object())
    return addr


@unbox(DeriveFunctionType)
def unbox_derive_function_type(typ, obj, c):
    """Unbox a :class:`DeriveWAP` that Python passes to jitted code into its three addresses, the numba one included.

    The code follows numba's ``unbox_function_type`` and takes the numba entry point from
    :func:`_get_derive_jit_address`. A value that is not a ``DeriveWAP`` makes that function raise ``TypeError``, and
    the call from Python fails with it.
    """
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
    """Box a value for Python as numba's ``box_function_type`` does, returning the ``DeriveWAP`` at ``py_addr``."""
    return box_function_type(typ, val, c)


@lower_constant(DeriveFunctionType)
def lower_constant_derive_function_type(context, builder, typ, pyval):
    """Lower a :class:`DeriveWAP` that jitted code reads as a compile-time constant, such as a module-level global.

    The lowered value holds the numba entry point as a symbol, so an exception raised in the body propagates, and numba
    can cache a caller that only calls the value. The cache entry of such a caller holds its own copy of the body's
    compiled code. The cached caller may keep running the old body after an edit to the body in another module, and
    nothing warns; deleting the caller's cache files is the only remedy. A caller that calls a proxied function through
    its dispatcher is not affected, because the cache guard of ``@proxy`` discards its stale cache entry when it loads.
    The function has no fallback to a fixed address. A value of this type is always called through its numba entry
    point, so an entry point that does not resolve fails the compilation.
    """
    typ = typ.get_precise()
    assert typ.check_signature(pyval.signature())
    sfunc = cgutils.create_struct_proxy(typ)(context, builder)
    # c_addr and py_addr follow numba's own lowering of a WrapperAddressProtocol value.
    sfunc.c_addr = context.add_dynamic_addr(
        builder, pyval.__wrapper_address__(), info=str(typ))
    sfunc.py_addr = context.add_dynamic_addr(
        builder, id(pyval), info=type(pyval).__name__)
    # jit_addr is declared as a symbol, and the compile result's library is linked in, as numba does when it casts a
    # dispatcher to a FunctionType. The symbol also makes the caller cacheable. A caller that only calls the value
    # leaves the globals of c_addr and py_addr unused, LLVM removes them before numba scans the final module, and
    # numba finds no dynamic global. A fixed address for jit_addr would be such a global, used by the call, and numba
    # would refuse to cache the caller rather than store an address that differs in every process.
    fn = context.declare_function(builder.module, pyval.cres.fndesc)
    sfunc.jit_addr = builder.bitcast(fn, context.get_value_type(types.voidptr))
    # The caller's cached binary includes the body's code from this linked library, so it can go stale. LLVM inlines
    # the body here, not numba. The numba-IR inline='always' that @proxy puts on its generated wrapper applies only
    # where a jitted caller calls the dispatcher, and a constant reference never reaches it. A body small enough for
    # LLVM to inline leaves no separate definition, and the caller keeps running the old body. A larger body is linked
    # in as a weak linkonce_odr definition. Its mangled name holds numba's per-process abi tag v<uid>, which numba's
    # cache key does not cover. The caller runs that embedded copy when the body's name in the loading process differs
    # from it, and the body's current definition when the names match. The name comes from the process that last
    # compiled the body and is stored in the body's own cache entry. The first process that compiles the body after an
    # edit therefore decides the outcome for every later process.
    context.active_code_library.add_linking_library(pyval.cres.library)
    return sfunc._getvalue()


def rewrap_derive(derive):
    """Upgrade ``derive`` to a :class:`DeriveWAP` if it is a ``CompileResultWAP``, so that its exceptions propagate.

    A value that :func:`~numbox.utils.highlevel.cres` compiled is already a ``DeriveWAP`` and is returned as it is. So
    is any other value, ``None`` included, so a caller can apply the function to every derive. A ``CompileResultWAP``
    built with numba directly would keep only the C entry point, and the compile result that it carries is all that the
    upgrade needs.

    The upgraded wrapper is stored on the original object, and a second call returns the same wrapper. The storage is
    required, not an optimization. ``py_addr`` holds the wrapper's address without a reference, so a new wrapper for
    each call would be freed when the caller returned, and every ``Work`` built from it would point to freed memory.
    Stored on the original, the wrapper lives as long as the object that the caller already holds.
    """
    if not isinstance(derive, CompileResultWAP) or isinstance(derive, DeriveWAP):
        return derive
    upgraded = getattr(derive, _UPGRADED_ATTR, None)
    if upgraded is None:
        upgraded = DeriveWAP(derive.cres)
        setattr(derive, _UPGRADED_ATTR, upgraded)
    return upgraded
