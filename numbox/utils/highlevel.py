import hashlib
import re
from inspect import getmodule, getsource
from io import StringIO
from numba import njit
from numba.core.itanium_mangler import mangle_type_or_value
from numba.core.types import Type
from numba.core.types.functions import Dispatcher
from numba.core.types.function_type import CompileResultWAP
from numba.core.typing.templates import Signature
from numba.experimental.function_type import FunctionType
from numba.experimental.structref import define_boxing, new, StructRefProxy
from numba.extending import overload, overload_method
from textwrap import dedent, indent
from typing import Callable, Iterable, Optional

from numbox.core.configurations import jit_options as jit_options_
from numbox.utils.fingerprint import (
    _Unfingerprintable, _canon_value, _fingerprint_function,
    _fingerprint_function_best_effort, _loaded_global_names,
)
from numbox.utils.preprocessing import _materialize_anchor, _structref_anchor_path
from numbox.utils.standard import make_params_strings


def _file_anchor():
    raise NotImplementedError


def _method_identity(method, user_ns) -> tuple[str, bool]:
    """Return ``(identity_text, cacheable)`` for a structref method.

    A ``make_structref`` method's source is inlined into a generated overload
    exec'd against the merged namespace, so the values passed through the ``ns``
    argument reach the compiled body but are invisible to ``getsource``. This
    folds the method's deep fingerprint together with the values of the ``ns``
    entries it references, so two methods with identical source but different
    ``ns`` values get distinct hashes (which flow into the content-addressed
    anchor file name, forcing numba to recompile rather than serve a stale
    binary). Un-fingerprintable method, or an un-canonicalizable ``ns`` value,
    makes it uncacheable -- the struct is then compiled without an on-disk
    cache, never reused, never wrong.

    Which ``ns`` entries the method references is read off the instruction
    stream, not ``co_names``: the values arrive as globals of the exec'd
    namespace, so only a global read can reach one, while ``co_names`` would also
    offer up every attribute name. A method doing ``self.x`` against an ``ns``
    that happens to carry an ``x`` would otherwise fold a value it never reads --
    and drop the struct's cache entirely if that value were opaque.
    """
    try:
        fingerprint = _fingerprint_function(method, set())
        cacheable = True
    except (_Unfingerprintable, RecursionError):
        fingerprint = _fingerprint_function_best_effort(method)
        cacheable = False
    parts = [fingerprint]
    if user_ns:
        for name in sorted(n for n in _loaded_global_names(method.__code__) if n in user_ns):
            try:
                parts.append(f"ns:{name}={_canon_value(user_ns[name], set())}")
            except (_Unfingerprintable, RecursionError):
                parts.append(f"ns:{name}=<opaque:{type(user_ns[name]).__name__}>")
                cacheable = False
    return ";".join(parts), cacheable


def cres(sig, **kwargs):
    """ Returns Python proxy to `FunctionType` rather than `CPUDispatcher` returned by `njit` """
    if not isinstance(sig, Signature):
        raise ValueError(f"Expected a single signature, found {sig} of type {type(sig)}")

    def _(func):
        func_jit = njit(sig, **kwargs)(func)
        sigs = func_jit.nopython_signatures
        assert len(sigs) == 1, f"Ambiguous signature, {sigs}"
        func_cres = func_jit.get_compile_result(sigs[0])
        cres_wap = CompileResultWAP(func_cres)
        return cres_wap
    return _


def cres_if_available(lib, sig, **kwargs):
    """Like ``cres(sig, **kwargs)``, but stubs out the wrapper if the C
    symbol matching ``func.__name__`` is absent from ``lib``.

    Use for binding sets that target multiple library versions where some
    symbols may only exist in newer releases. Callers get a stub that
    raises ``NotImplementedError`` instead of a confusing LLVM link error
    at call time.
    """
    def _(func):
        if hasattr(lib, func.__name__):
            return cres(sig, **kwargs)(func)

        def stub(*args, **_kwargs):
            raise NotImplementedError(f"{func.__name__} is not available")
        stub.__name__ = func.__name__
        return stub
    return _


def determine_field_index(struct_ty, field_name):
    for i_, field_pair in enumerate(struct_ty._fields):
        if field_pair[0] == field_name:
            return i_
    raise ValueError(f"{field_name} not in {struct_ty}")


def _type_identity(ty: Type) -> tuple[str, bool]:
    """Return ``(hash, cacheable)`` for a numba type, for folding into a
    content-addressed name."""
    cacheable = True
    ty_str = str(ty)
    if isinstance(ty, Dispatcher):
        # numba's own on-disk index misses a Dispatcher-carrying signature every run
        # and appends a fresh data entry, so the unit's .nbc grows per process forever.
        cacheable = False
        try:
            body = "dispatcher(" + _fingerprint_function(ty.dispatcher.py_func, set()) + ")"
        except (_Unfingerprintable, RecursionError):
            body = "dispatcher-besteffort(" + _fingerprint_function_best_effort(ty.dispatcher.py_func) + ")"
    elif "Record(" in ty_str:
        # A Record mangles by a creation-order counter (_code), so it hashes differently
        # in a cold process than a warm one. Its str is structural, so hash that instead.
        body = "structural(" + ty_str + ")"
    else:
        body = mangle_type_or_value(ty)
    if cacheable and "CPUDispatcher(" in ty_str:
        # A Dispatcher nested in a container
        cacheable = False
    return hashlib.sha256(body.encode("utf-8")).hexdigest(), cacheable


def hash_type(ty: Type) -> str:
    """Process-stable content hash of a numba type; see :func:`_type_identity`."""
    return _type_identity(ty)[0]


def _signature_identity(sig: Signature) -> tuple[str, bool]:
    """Return ``(canonical_text, cacheable)`` for a numba ``Signature``, folding every
    argument and return type through :func:`_type_identity`."""
    ret_hash, ret_ok = _type_identity(sig.return_type)
    arg_ids = [_type_identity(a) for a in sig.args]
    text = ret_hash + "(" + ",".join(h for h, _ok in arg_ids) + ")"
    cacheable = ret_ok and all(ok for _h, ok in arg_ids)
    return text, cacheable


def make_structref_code_txt(
    struct_name: str,
    struct_fields: Iterable[str] | dict[str, Type],
    struct_type_class: type | Type,
    struct_methods: Optional[dict[str, Callable]] = None,
    user_ns: Optional[dict] = None,
):
    cacheable = True
    if isinstance(struct_fields, dict):
        struct_fields, fields_types = list(struct_fields.keys()), list(struct_fields.values())
    else:
        assert isinstance(struct_fields, (list, tuple)), struct_fields
        fields_types = None
    struct_fields_str = ", ".join([field for field in struct_fields])
    make_name = f"make_{struct_name.lower()}"
    new_returns = f"{make_name}({struct_fields_str})"
    repr_str = f"f'{struct_name}(" + ", ".join([f"{field}={{self.{field}}}" for field in struct_fields]) + ")'"
    code_txt = StringIO()
    code_txt.write(f"""
class {struct_name}(StructRefProxy):
    def __new__(cls, {struct_fields_str}):
        return {new_returns}

    def __repr__(self):
        return {repr_str}
""")
    for field in struct_fields:
        code_txt.write(f"""
    @property
    @njit(**jit_options)
    def {field}(self):
        return self.{field}
""")
    methods_code_txt = StringIO()
    if struct_methods is not None:
        assert isinstance(struct_methods, dict), f"""
    Expected dictionary of methods names to callable, got {struct_methods}"""
        for method_name, method in struct_methods.items():
            params_str, names_params_str = make_params_strings(method)
            names_params_lst = names_params_str.split(", ")
            self_name = names_params_lst[0]
            names_params_str_wo_self = ", ".join(names_params_lst[1:])
            method_source = dedent(getsource(method))
            method_identity, method_cacheable = _method_identity(method, user_ns)
            method_hash = hashlib.sha256(method_identity.encode("utf-8")).hexdigest()
            cacheable = cacheable and method_cacheable
            code_txt.write(f"""
    def {method_name}({params_str}):
        return {self_name}.{method_name}_{method_hash}({names_params_str_wo_self})

    @njit(**jit_options)
    def {method_name}_{method_hash}({params_str}):
        return {self_name}.{method_name}({names_params_str_wo_self})
""")
            method_source = re.sub(r"\bdef\s+([a-zA-Z_]\w*)\b", "def _", method_source)
            methods_code_txt.write(f"""
@overload_method({struct_type_class.__name__}, "{method_name}", jit_options=jit_options)
def ol_{method_name}({params_str}):
{indent(method_source, "    ")}
    return _
""")
    code_txt.write(f"""
define_boxing({struct_type_class.__name__}, {struct_name})
""")
    struct_type_name = f"{struct_name}Type"
    struct_fields_ty_str = ", ".join([f"{field}_ty" for field in struct_fields])
    struct_type_code_block = ""
    if fields_types is None:
        struct_type_code_block = f"""    fields_types = [{struct_fields_ty_str}]
    fields_and_their_types = list(zip(fields, fields_types))
    {struct_name}Type = {struct_type_class.__name__}(fields_and_their_types)
"""
    else:
        code_txt.write(f"""
fields_and_their_types = list(zip(fields, fields_types))
{struct_name}Type = {struct_type_class.__name__}(fields_and_their_types)
""")
    ctor_code_block = "\n".join([f"        struct_.{field} = {field}" for field in struct_fields])
    code_txt.write(f"""
@overload({struct_name}, strict=False, jit_options=jit_options)
def ol_{struct_name.lower()}({struct_fields_ty_str}):
{struct_type_code_block}
    def ctor({struct_fields_str}):
        struct_ = new({struct_type_name})
{ctor_code_block}
        return struct_
    return ctor
""")
    if fields_types is not None:
        code_txt.write(f"""
{make_name}_sig = {struct_name}Type(*fields_types)
""")
    else:
        code_txt.write(f"""
{make_name}_sig = None
""")
    code_txt.write(f"""
@njit({make_name}_sig, **jit_options)
def {make_name}({struct_fields_str}):
    return {struct_name}({struct_fields_str})
""")
    code_txt = code_txt.getvalue() + methods_code_txt.getvalue()
    return code_txt, fields_types, cacheable


def make_structref(
    struct_name: str,
    struct_fields: Iterable[str] | dict[str, Type],
    struct_type_class: type | Type,
    *,
    struct_methods: Optional[dict[str, Callable]] = None,
    jit_options: Optional[dict] = None,
    ns: Optional[dict] = None
):
    """
    Makes structure type with `struct_name` and `struct_fields` from the StructRef type class.

    A unique `struct_type_class` for each structref needs to be provided.
    If caching of code that will be using the created struct type is desired,
    these type class(es) need/s to be defined in a python module that is *not* executed.
    (Same requirement is also to observed even when the full definition of StructRef
    is entirely hard-coded rather than created dynamically.)

    In particular, that's why `struct_type_class` cannot be incorporated into
    the dynamic compile / exec routine here.

    Dictionary of methods to be bound to the created structref can be provided as well.
    Struct methods will get inlined into the caller if numba deems it to be optimal
    (even if `jit_options` says otherwise), therefore changing the methods code
    without poking the jitted caller can result in a stale cache - when the latter is
    cached. This is not an exclusive limitation of a dynamic structref creation via
    this function and is equally true when the structref definition is coded explicitly.

    Anchor file
    -----------
    The generated ``code_txt`` is written to a content-addressed file
    under numba's cache directory and that file -- not ``highlevel.py``
    -- is used as the ``compile()`` anchor. See the "Cache-anchor
    mechanism" section in ``docs/numbox.utils.rst`` for the rationale.
    """
    code_txt, fields_types, cacheable = make_structref_code_txt(
        struct_name, struct_fields, struct_type_class, struct_methods, user_ns=ns
    )
    if jit_options is None:
        jit_options = jit_options_
    if not cacheable:
        # A method (or an ns value it captures) has no canonical fingerprint, so
        # its content-addressed identity cannot be trusted to change when the
        # behaviour does; compile the struct without an on-disk cache.
        jit_options = {**jit_options, "cache": False}
    ns = ns or {}
    ns = {
        **ns,
        **getmodule(_file_anchor).__dict__,
        **{
            "fields": struct_fields,
            "fields_types": fields_types,
            "define_boxing": define_boxing,
            "jit_options": jit_options,
            "new": new,
            "njit": njit,
            "overload": overload,
            "overload_method": overload_method,
            "StructRefProxy": StructRefProxy,
            struct_type_class.__name__: struct_type_class
        }
    }
    anchor = _structref_anchor_path(struct_name, code_txt)
    _materialize_anchor(anchor, code_txt)
    code = compile(code_txt, str(anchor), mode="exec")
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns[struct_name]


def prune_type(ty):
    if isinstance(ty, Dispatcher):
        sigs = ty.get_call_signatures()[0]
        assert len(sigs) == 1, f"Ambiguous signature, {sigs}"
        ty = FunctionType(sigs[0])
    return ty
