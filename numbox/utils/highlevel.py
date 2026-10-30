"""Functions that define numba StructRef types in one call and compile functions into first-class function values.

:func:`make_structref` generates the code of a StructRef type with its constructor, field getters and methods, writes
the code to a cache anchor and compiles it, so that numba can cache the struct. :func:`cres` compiles a function for
one signature into a :class:`~numbox.utils.derive_wap.DeriveWAP`, and :func:`cres_if_available` does the same if a C
library has the function's symbol. The module also computes the process-stable identities of numba types and
signatures that numbox puts into the names of generated code.
"""
import hashlib
import re
from inspect import getmodule, getsource
from io import StringIO
from numba import njit
from numba.core.itanium_mangler import mangle_type_or_value
from numba.core.types import Type
from numba.core.types.functions import Dispatcher
from numba.core.typing.templates import Signature
from numba.experimental.function_type import FunctionType
from numba.experimental.structref import define_boxing, new, StructRefProxy
from numba.extending import overload, overload_method
from textwrap import dedent, indent
from typing import Callable, Iterable, Optional

from numbox.core.cache_key import _keyed
from numbox.core.configurations import jit_options as jit_options_
from numbox.utils.derive_wap import DeriveWAP
from numbox.utils.fingerprint import (
    _Unfingerprintable, _canon_value, _fingerprint_function,
    _fingerprint_function_best_effort, _loaded_global_names,
)
from numbox.utils.preprocessing import _anchored_or_uncached, _structref_anchor_path, bounded_stem
from numbox.utils.standard import make_params_strings


def _file_anchor():
    raise NotImplementedError


def _method_identity(method, user_ns) -> tuple[str, bool]:
    """Return ``(identity, cacheable)`` for a struct method: a text that identifies it, and whether it may be cached.

    The text joins the method's fingerprint with the values of the ``ns`` entries that the method reads as globals.
    ``make_structref`` copies the method's source into generated code that runs in a namespace holding ``ns``. Those
    values therefore reach the compiled method, but the method's own fingerprint does not include them. Two methods
    with the same source and different ``ns`` values have different texts, so their helpers have different names and
    their code is written to different anchors. numba then compiles the second method instead of loading the binary of
    the first. ``cacheable`` is False if the method has no
    fingerprint or reads an ``ns`` value that has none. The struct then compiles without an on-disk cache, so it is
    never loaded stale.
    """
    try:
        fingerprint = _fingerprint_function(method, set())
        cacheable = True
    except (_Unfingerprintable, RecursionError):
        fingerprint = _fingerprint_function_best_effort(method)
        cacheable = False
    parts = [fingerprint]
    # The ns values reach the method as globals of the generated code, so only a global read can reach one. The names
    # come from the opcodes, not from co_names, which also holds every attribute name. With co_names, a method that
    # reads self.x would fold an ns entry named x that it never reads, and an entry without a fingerprint would turn
    # the struct's cache off.
    if user_ns:
        for name in sorted(n for n in _loaded_global_names(method.__code__) if n in user_ns):
            try:
                parts.append(f"ns:{name}={_canon_value(user_ns[name], set())}")
            except (_Unfingerprintable, RecursionError):
                parts.append(f"ns:{name}=<opaque:{type(user_ns[name]).__name__}>")
                cacheable = False
    return ";".join(parts), cacheable


def cres(sig, **kwargs):
    """Compile the decorated function for the signature ``sig`` and return it as a first-class function value.

    ``sig`` must be one numba ``Signature``, such as ``float64(float64)``; anything else raises ``ValueError``. The
    keyword arguments are passed to ``njit``, and the function compiles when it is decorated. The result is a
    :class:`~numbox.utils.derive_wap.DeriveWAP`, not a dispatcher, so a cached jitted function that takes it as an
    argument loads from the cache in the next process. An exception raised in its body propagates to a jitted caller,
    as :mod:`numbox.utils.derive_wap` explains.
    """
    if not isinstance(sig, Signature):
        raise ValueError(f"Expected a single signature, found {sig} of type {type(sig)}")

    def _(func):
        with _keyed(func):
            func_jit = njit(sig, **kwargs)(func)
        sigs = func_jit.nopython_signatures
        assert len(sigs) == 1, f"Ambiguous signature, {sigs}"
        func_cres = func_jit.get_compile_result(sigs[0])
        wrapper = DeriveWAP(func_cres)
        # The wrapper stores the Python function, because the compile result identifies the body only by an address and
        # by a mangled name with a per-process counter. A function that captures the wrapper is fingerprinted through
        # this attribute (_canon_value in numbox.utils.fingerprint).
        wrapper._numbox_py_func = func
        # The wrapper also stores the dispatcher, whose cache numbox.core.cache_key.explain_cache_key reads.
        wrapper._numbox_dispatcher = func_jit
        return wrapper
    return _


def cres_if_available(lib, sig, **kwargs):
    """Return what ``cres(sig, **kwargs)`` returns if ``lib`` has a symbol named after the function; else return a stub.

    ``lib`` is the ``ctypes.CDLL`` handle of a C library, as ``numbox.core.bindings.utils.load_lib`` returns it. Use the
    decorator for a binding of a C function that older versions of the library lack. The stub is a plain Python function
    with the decorated function's name. A call of the stub from Python raises ``NotImplementedError`` instead of failing
    with an LLVM link error, and jitted code that calls the stub fails to compile.
    """
    def _(func):
        if hasattr(lib, func.__name__):
            return cres(sig, **kwargs)(func)

        def stub(*args, **_kwargs):
            """Raise ``NotImplementedError``, because the library has no symbol for this function."""
            raise NotImplementedError(f"{func.__name__} is not available")
        stub.__name__ = func.__name__
        return stub
    return _


def determine_field_index(struct_ty, field_name):
    """Return the position of the field ``field_name`` in the structref type ``struct_ty``, or raise ``ValueError``."""
    for i_, field_pair in enumerate(struct_ty._fields):
        if field_pair[0] == field_name:
            return i_
    raise ValueError(f"{field_name} not in {struct_ty}")


def _type_identity(ty: Type) -> tuple[str, bool]:
    """Return ``(digest, cacheable)`` for the numba type ``ty``: a SHA-256 that is the same in every process, and
    whether code typed with ``ty`` may be cached.

    A dispatcher type, or a container type that holds one, is not cacheable. A Record type is hashed by its ``str``, a
    dispatcher type by the fingerprint of its Python function, and any other type by numba's mangled name for it.
    """
    cacheable = True
    ty_str = str(ty)
    if isinstance(ty, Dispatcher):
        # A dispatcher type holds the address of its Python function, which differs from one process to the next. A
        # signature that holds one misses numba's on-disk index in every process, and numba appends a new data entry
        # each time, so the cache files grow without bound.
        cacheable = False
        try:
            body = "dispatcher(" + _fingerprint_function(ty.dispatcher.py_func, set()) + ")"
        except (_Unfingerprintable, RecursionError):
            body = "dispatcher-besteffort(" + _fingerprint_function_best_effort(ty.dispatcher.py_func) + ")"
    elif "Record(" in ty_str:
        # numba mangles a Record type with its _code, a counter of the order in which the process creates types, so
        # the mangled name differs between a fresh process and one that has created other types first. The str of a
        # Record describes its fields, so the str is hashed instead.
        body = "structural(" + ty_str + ")"
    else:
        body = mangle_type_or_value(ty)
    if cacheable and "CPUDispatcher(" in ty_str:
        # A dispatcher inside a container type, such as a tuple, makes the type uncacheable as well.
        cacheable = False
    return hashlib.sha256(body.encode("utf-8")).hexdigest(), cacheable


def hash_type(ty: Type) -> str:
    """Return a SHA-256 hex digest of the numba type ``ty`` that is the same in every process.

    A Record type is hashed by its ``str``, a dispatcher type by the fingerprint of its Python function, and any other
    type by numba's mangled name for it.
    """
    return _type_identity(ty)[0]


def _signature_identity(sig: Signature) -> tuple[str, bool]:
    """Return ``(text, cacheable)`` for the numba signature ``sig``: the digests of its return and argument types, and
    whether all of them are cacheable, as :func:`_type_identity` returns them."""
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
    """Return ``(code, fields_types, cacheable)``: the source that :func:`make_structref` compiles for a struct.

    The arguments are those of :func:`make_structref`, with ``user_ns`` for its ``ns``. ``fields_types`` is the list of
    the field types, or None if ``struct_fields`` holds only names. ``cacheable`` is False if a method, or an ``ns``
    value that a method reads, has no fingerprint.
    """
    cacheable = True
    if isinstance(struct_fields, dict):
        struct_fields, fields_types = list(struct_fields.keys()), list(struct_fields.values())
    else:
        assert isinstance(struct_fields, (list, tuple)), struct_fields
        fields_types = None
    struct_fields_str = ", ".join([field for field in struct_fields])
    # numba puts the qualified names of the generated functions into the names of its cache files. Those qualified
    # names include the name of the class whose body defines the jitted getters and method helpers. The names are
    # bounded, so that a struct name of any length keeps the file names within the file system's limit. After the
    # class body runs, the generated code sets the class's __name__ and __qualname__ to the struct's full name and
    # binds the struct's name to the class.
    class_name = bounded_stem(struct_name)
    make_name = f"make_{class_name.lower()}"
    new_returns = f"{make_name}({struct_fields_str})"
    repr_str = f"f'{struct_name}(" + ", ".join([f"{field}={{self.{field}}}" for field in struct_fields]) + ")'"
    code_txt = StringIO()
    code_txt.write(f"""
class {class_name}(StructRefProxy):
    def __new__(cls, {struct_fields_str}):
        return {new_returns}

    def __repr__(self):
        return {repr_str}
""")
    # The getter of a field with a long name is defined under the field's bounded name, then assigned to the field's
    # own name and deleted. If an earlier field had the bounded name as its own name, the getter's definition would
    # replace that field's property. The del statement would then remove that name from the class. numbox adds a leading
    # underscore until the bounded name differs from every field name and method name.
    taken = set(struct_fields) | set(struct_methods or ())
    for field in struct_fields:
        getter_name = bounded_stem(field)
        while getter_name != field and getter_name in taken:
            getter_name = "_" + getter_name
        code_txt.write(f"""
    @property
    @njit(**jit_options)
    def {getter_name}(self):
        return self.{field}
""")
        if getter_name != field:
            code_txt.write(f"""
    {field} = {getter_name}
    del {getter_name}
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
            thunk_name = f"{bounded_stem(method_name)}_{method_hash}"
            # Python code calls the method of the class, which calls a jitted helper named after the method and the
            # hash of its identity; the helper calls the method that the overload below defines for jitted code.
            # Through the hash in the helper's name, the method's identity, including its ns values, becomes part of the
            # generated code and therefore of the anchor's digest.
            code_txt.write(f"""
    def {method_name}({params_str}):
        return {self_name}.{thunk_name}({names_params_str_wo_self})

    @njit(**jit_options)
    def {thunk_name}({params_str}):
        return {self_name}.{method_name}({names_params_str_wo_self})
""")
            # The method's source becomes the body of the overload under the name _, which the overload returns. The
            # substitution renames every def in the source, a nested one included, so a method that defines and calls
            # a nested function fails to compile.
            method_source = re.sub(r"\bdef\s+([a-zA-Z_]\w*)\b", "def _", method_source)
            methods_code_txt.write(f"""
@overload_method({struct_type_class.__name__}, "{method_name}", jit_options=jit_options)
def ol_{bounded_stem(method_name)}({params_str}):
{indent(method_source, "    ")}
    return _
""")
    if class_name != struct_name:
        code_txt.write(f"""
{class_name}.__name__ = {struct_name!r}
{class_name}.__qualname__ = {struct_name!r}
{struct_name} = {class_name}
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
def ol_{class_name.lower()}({struct_fields_ty_str}):
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
    """Define a StructRef type named ``struct_name`` with the fields ``struct_fields``, and return its class.

    ``struct_fields`` is a dict of field names to numba types, or a sequence of field names whose types numba infers
    from the arguments of each call to the class. ``struct_type_class`` is an empty subclass of numba's ``StructRef``
    registered with numba's ``@register``, and each struct needs one of its own. If code that uses the struct is to be
    cached, define the type class in a module that the program imports, not in the script that runs. numba pickles the
    struct's type into its cache, and a class from the running script does not match the cached copy in the next run.
    A StructRef written by hand has the same requirement. For the same reason, ``make_structref`` cannot define the
    type class in the code that it generates.

    ``struct_methods`` maps method names to functions. Each function becomes a method that Python code and jitted code
    can call, and it receives the struct as its first argument. A method must not define a nested function, because
    numbox renames every ``def`` in the method's source. numba may inline a method into a jitted caller regardless of
    what ``jit_options`` specifies about inlining, and a cached caller holds the method's compiled code in either case.
    After an edit to a method, a cached caller in another file can therefore continue to run the old code, as it would
    with a StructRef written by hand.

    ``ns`` maps names to the globals that the methods read, such as a module or a jitted helper, because the methods run
    in a namespace of their own. numbox overrides an entry of ``ns`` whose name is also a global of this module or one
    of the names that the generated code uses, such as ``fields`` or ``jit_options``. ``jit_options`` defaults to
    numbox's default jit options.

    numbox writes the generated code to a cache anchor under numba's cache directory, not to ``highlevel.py``, and
    compiles it under the anchor's file name. numba quotes code from the anchor in its messages, so numbox writes the
    anchor even with caching disabled, and in that case it ignores a failed write. With caching enabled, numbox writes
    the anchor and then asks numba whether it can cache a function of the anchor; numba can answer only when the file
    exists on disk. If the write fails or numba cannot cache in that location, the struct compiles without a cache
    after one warning, and an anchor that was written remains on disk. If a method, or an ``ns`` value that a method
    reads, has no fingerprint, the struct compiles without a cache and without a warning. The utils page explains the
    anchors in its section on the cache-anchor mechanism.
    """
    code_txt, fields_types, cacheable = make_structref_code_txt(
        struct_name, struct_fields, struct_type_class, struct_methods, user_ns=ns
    )
    if jit_options is None:
        jit_options = jit_options_
    if not cacheable:
        # A method, or an ns value that it reads, has no fingerprint, so the anchor's name might not change when the
        # method's behavior does. The struct therefore compiles without an on-disk cache.
        jit_options = {**jit_options, "cache": False}
    anchor = _structref_anchor_path(struct_name, code_txt)
    jit_options = _anchored_or_uncached(anchor, code_txt, jit_options)
    ns = ns or {}
    # The generated code runs in a namespace made of ns, then this module's globals, then the names that the code
    # uses; a later entry replaces an earlier one of the same name. The functions that the code defines take this
    # module's __name__ as their __module__, so numbox's cache key applies to them.
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
    code = compile(code_txt, str(anchor), mode="exec")
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns[struct_name]


def prune_type(ty):
    """Return ``ty``, or for a dispatcher type the ``FunctionType`` of the dispatcher's one compiled signature.

    A dispatcher type whose dispatcher has no compiled signature, or more than one, fails an ``assert``.
    """
    if isinstance(ty, Dispatcher):
        sigs = ty.get_call_signatures()[0]
        assert len(sigs) == 1, f"Ambiguous signature, {sigs}"
        ty = FunctionType(sigs[0])
    return ty
