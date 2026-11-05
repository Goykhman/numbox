"""Node specs and the builder that compiles them into a jitted graph of :class:`~numbox.core.work.work.Work` nodes.
:class:`End` and :class:`Derived` declare the nodes, and each spec is registered under its name in a registry, by
default ``_specs_registry``. :func:`make_graph` compiles a function, the graph maker, that creates the requested access
nodes and every node they depend on, and returns the access nodes as ``Work`` nodes. The module also computes the names
under which numba caches the graph maker and the derives, so that numba compiles them again after a change to a node.
"""
from collections import namedtuple
from hashlib import sha256
from inspect import getfile, getmodule
from io import StringIO
from itertools import chain
from numba import njit, typeof
from numba.core.types import Type
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple as PyTuple, Union

from numbox.core.cache_key import _settings_canon
from numbox.core.configurations import jit_options as jit_options_
from numbox.core.work.lowlevel_work_utils import ll_make_work
from numbox.utils.fingerprint import (
    _Unfingerprintable, _effective_flags, _fingerprint_function,
    _fingerprint_function_best_effort, _flags_canon, _loaded_global_names,
)
from numbox.utils.highlevel import cres, _signature_identity, _type_identity
from numbox.utils.preprocessing import _anchor_or_error, _anchor_root, _cached_at_or_uncached, _orphan_anchor_sweep


def _file_anchor():
    raise NotImplementedError


_specs_registry = dict()


class _End(NamedTuple):
    name: str
    init_value: Any
    registry: dict = None
    ty: Optional[type | Type] = None


def _new(cls, super_proxy, *args, **kwargs):
    name = kwargs.get("name")
    assert name, "`name` key-word argument has not been provided"
    registry = kwargs.get("registry", _specs_registry)
    if name in registry:
        raise ValueError(f"Node '{name}' has already been defined on this graph. Pick a different name.")
    spec_ = super_proxy.__new__(cls, *args, **kwargs)
    registry[name] = spec_
    return spec_


class End(_End):
    """Spec of an end node: a node whose value is given rather than computed.

    ``name`` and ``registry`` must be passed by keyword, because the registration reads them from the keywords. The
    spec is registered under ``name`` in ``registry``, by default the module's ``_specs_registry``, and a name that the
    registry already holds raises ``ValueError``. ``init_value`` is the node's initial value. ``ty``, a numba type,
    sets the node's type in place of the type of ``init_value``, and numba converts the initial value to it.
    """
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        return _new(cls, super(), *args, **kwargs)


class _Derived(NamedTuple):
    name: str
    init_value: Any
    derive: Callable
    sources: Sequence[Union['Derived', End]] = ()
    registry: dict = None
    ty: Optional[type | Type] = None


class Derived(_Derived):
    """Spec of a derived node: a node whose value ``derive`` computes from the values of ``sources``.

    ``derive`` is a plain Python function with one parameter per source, and :func:`make_graph` compiles it for the
    node's type over the types of the sources. ``sources`` is a tuple of specs. ``init_value`` is the value that the
    node holds until it is calculated. ``name``, ``registry`` and ``ty`` follow the rules of :class:`End`.
    """
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        return _new(cls, super(), *args, **kwargs)


SpecTy = Derived | End


def _input_line(input_: End, ns: dict, initializers: dict):
    name_ = input_.name
    init_ = input_.init_value
    init_name = f"{name_}_init"
    ns[init_name] = init_
    initializers[init_name] = init_
    ty_ = input_.ty
    if ty_ is not None:
        type_name = f"{name_}_ty"
        ns[type_name] = ty_
        return f"""{name_} = ll_make_work("{name_}", {init_name}, (), None, {type_name})"""
    return f"""{name_} = ll_make_work("{name_}", {init_name}, (), None)"""


def get_ty(spec_):
    """Return the numba type of a spec's node: its ``ty`` if given, otherwise numba's type for its ``init_value``."""
    return spec_.ty or typeof(spec_.init_value)


_derive_funcs = {}


def _derive_fingerprint(derive) -> tuple[str, bool, bool]:
    """Return ``(fingerprint, derive_cacheable, kernel_safe)`` for a derive.

    The fingerprint covers the derive's code, its defaults and closure, and the values of the globals it reads, and it
    recurses into the functions among those globals. The hash that names the graph maker includes the fingerprint.
    numbox computes the fingerprint from the code object, not from the source text, so a derive defined by ``exec`` or
    at an interactive prompt can be fingerprinted as well.

    A derive that loads a name from its module's globals compiles without a cache: ``derive_cacheable`` is False, and
    every process compiles the derive again. The fingerprint does not cover everything such a name can lead to, such
    as the implementation of an ``@overload``. It does hold the values of those globals, so the graph maker stays
    cacheable (``kernel_safe`` True) and gets a new name when one of them changes.

    A derive that ``_fingerprint_function`` cannot fingerprint gets the placeholder fingerprint of
    ``_fingerprint_function_best_effort`` instead. The placeholder contains no address and is therefore the same in
    every process. It does not change, however, when a value without a canonical form changes. The derive and the
    graph maker then both compile without a cache (``kernel_safe`` False), as ``compile_kernel`` does with a formula it
    cannot fingerprint.
    """
    try:
        fingerprint = _fingerprint_function(derive, set())
    except (_Unfingerprintable, RecursionError):
        return _fingerprint_function_best_effort(derive), False, False
    # The check reads the instruction stream (_loaded_global_names), not co_names, because co_names also holds
    # attribute names: the record field c1 in a_[t].c1 is a LOAD_ATTR, and a module global named c1 would otherwise
    # count as module state. A cfg.SCALE read still counts, through the LOAD_GLOBAL of its base cfg.
    # _fingerprint_function reads globals the same way, so such a name does not force the placeholder fingerprint.
    reads_module_state = any(
        name in derive.__globals__ for name in _loaded_global_names(derive.__code__)
    )
    return fingerprint, not reads_module_state, True


_DERIVE_ANCHOR_SUBDIR = "numbox-derive"
# Remove the `.tmp-*` files that a writer killed by SIGKILL left among the derive anchors, as each module that writes
# anchors does for its own directory.
_orphan_anchor_sweep(_DERIVE_ANCHOR_SUBDIR)


def _derive_anchor_cres(derive_sig, sig_canon, derive, derive_fp, jit_options):
    """Compile ``derive`` for ``derive_sig`` under a cache entry keyed on what it reads, or return ``None``.

    numbox's key for a cached ``cres`` function includes the jit flags and numba's settings (see
    :mod:`numbox.core.cache_key`). Like numba's key, it does not include the values of the function's globals or the
    code of the functions it calls. The derive is therefore compiled through a generated wrapper, named after a digest
    of the derive's fingerprint, its signature, the flags and the settings.

    The result is ``None`` when a flag has no canonical form, when no anchor file can be written, or when numba has no
    cache location for the anchor. The caller must then compile the derive without a cache, because only the wrapper's
    name keys a cached derive on what it reads.
    """
    flags = _effective_flags(jit_options)
    flags_canon, ok_flags = _flags_canon(flags)
    if not ok_flags:
        # A key cannot include a flag that has no canonical form. A key without the flag cannot distinguish the
        # binaries that two values of the flag produce.
        return None
    # The digest names the wrapper and its anchor file, and numba names the wrapper's cache files after both. A change
    # to the fingerprint, the signature, the flags or the settings therefore gives a new cache entry.
    # compile_kernel._compile names its kernels the same way.
    digest = sha256(
        f"derive-anchor-v2\n{derive_fp}\n# sig: {sig_canon}\n# flags: {flags_canon}"
        f"\n# settings: {_settings_canon()}".encode("utf-8")
    ).hexdigest()[:16]
    name = f"_derive_{digest}"
    params = ", ".join(f"a{i}" for i in range(len(derive_sig.args)))
    src = (
        f"@_cres(_derive_sig, **_derive_jit_options)\n"
        f"def {name}({params}):\n"
        f"    return _inner({params})\n"
    )
    anchor = _anchor_root(_DERIVE_ANCHOR_SUBDIR) / f"{name}.py"
    if _anchor_or_error(anchor, src) is not None:
        # No anchor could be written, or numba has no cache location for it. An anchor already on disk has none when
        # its directory is no longer writable and numba has no other location.
        return None
    # The wrapper calls the derive through a dispatcher compiled without a cache. numba links the derive's code into
    # the wrapper's cached code, and the derive's own cache entry would be keyed on neither its flags nor what it
    # reads.
    ns = {
        "_cres": cres,
        "_derive_sig": derive_sig,
        "_derive_jit_options": jit_options,
        "_inner": njit(**{**flags, "cache": False})(derive),
        # numba imports __name__ to rebuild a cached function's environment in another process, so it must name an
        # importable module; compile_kernel sets it the same way.
        "__name__": __name__,
    }
    code = compile(src, str(anchor), "exec")
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns[name]


def _derived_cres(ty, sources: Sequence[End], derive, jit_options=None, derive_fp=None):
    jit_options = jit_options if jit_options is not None else {}
    sources_ty = []
    for source in sources:
        source_ty = get_ty(source)
        sources_ty.append(source_ty)
    derive_sig = ty(*sources_ty)
    sig_canon, sig_cacheable = _signature_identity(derive_sig)
    if jit_options.get("cache") and derive_fp is not None and sig_cacheable:
        anchored = _derive_anchor_cres(derive_sig, sig_canon, derive, derive_fp, jit_options)
        if anchored is not None:
            _derive_funcs[id(anchored)] = derive
            return anchored
        # Only the anchor keys a cached derive on what it reads. A plain cache=True compile would store the derive under
        # a cache entry named after its own source file and qualified name, keyed on neither its globals' values nor
        # its callees' code. The derive therefore compiles without a cache, and every process compiles it again.
        jit_options = {**jit_options, "cache": False}
    elif jit_options.get("cache") and not sig_cacheable:
        # The node's type or a source's type contains a dispatcher, directly or inside a container type, so
        # _signature_identity reports the signature as uncacheable. In every new process, numba finds no entry for
        # such a signature in its cache index and adds one. A dispatcher whose body cannot be fingerprinted also has
        # only a placeholder identity, which does not change when the body does. The derive therefore compiles without
        # a cache.
        jit_options = {**jit_options, "cache": False}
    try:
        derive_cres = cres(derive_sig, **jit_options)(derive)
    except RuntimeError as e:
        # numba raises RuntimeError for cache=True when it has no cache locator for the function's source file, as for
        # a derive defined by exec or read from <stdin>. The derive then compiles without a cache, and a derive
        # compiled without a cache cannot be stale. A derive defined in a module file or a notebook has a locator.
        # make_graph always passes a fingerprint, so its cached derives are compiled through the anchor. This
        # compilation has caching enabled only for a caller that passes no fingerprint.
        if jit_options.get("cache") and "locator" in str(e):
            derive_cres = cres(derive_sig, **{**jit_options, "cache": False})(derive)
        else:
            raise
    _derive_funcs[id(derive_cres)] = derive
    return derive_cres


def _derived_line(
    derived_: Derived, ns: dict, initializers: dict, derive_hashes: list, kernel_safe_flags: list,
    _make_args: list, jit_options=None
):
    name_ = derived_.name
    init_ = derived_.init_value
    sources_ = ", ".join([s.name for s in derived_.sources])
    sources_ = sources_ + ", " if sources_ and "," not in sources_ else sources_
    ty_ = get_ty(derived_)
    derive_func = derived_.derive
    derive_fp, derive_cacheable, kernel_safe = _derive_fingerprint(derive_func)
    derive_hashes.append(derive_fp)
    kernel_safe_flags.append(kernel_safe)
    derive_jit = jit_options if derive_cacheable else {**(jit_options or {}), "cache": False}
    derive_ = _derived_cres(ty_, derived_.sources, derive_func, derive_jit, derive_fp)
    derive_name = f"{name_}_derive"
    init_name = f"{name_}_init"
    _make_args.append(derive_name)
    ns[derive_name] = derive_
    ns[init_name] = init_
    initializers[init_name] = init_
    return f"""{name_} = ll_make_work("{name_}", {init_name}, ({sources_}), {derive_name})"""


def code_block_hash(code_txt: str):
    """Return the SHA-256 hex digest of ``code_txt``.

    ``_kernel_fingerprint`` hashes the graph maker's code text together with the derives' fingerprints, the nodes'
    types, the jit flags and numba's settings, and the maker is named after the digest. A change to any of them changes
    the maker's name, so numba does not load the code that it cached under the earlier name.
    """
    return sha256(code_txt.encode("utf-8")).hexdigest()


def _kernel_fingerprint(code_block: str, derive_hashes: list, type_sigs: list, jit_options=None) -> tuple[str, bool]:
    """Return ``(hash, flags_cacheable)`` for the generated graph maker, whose name is ``_make_<hash>``.

    The hash covers the code text, the derives' fingerprints, each node's type (``type_sigs``), the jit flags in effect
    and the ``NUMBA_*`` settings that change the emitted code (see :mod:`numbox.core.cache_key`). A change to any of
    them gives the maker a new name. ``flags_cacheable`` is False when a flag has no canonical form, such as a
    ``pipeline_class`` or a ``locals`` dictionary of numba types. The caller must then compile the maker without a
    cache, as ``_derive_anchor_cres`` and ``compile_kernel`` do.
    """
    # A node's type appears in the code text only as the global name {name}_ty, if at all, so type_sigs adds it to
    # the hash. The text shows the flags only as @njit(**jit_options). If the hash did not include the flags, two
    # graphs that differ only in error_model or fastmath would share a name. numbox's key for the maker's cache also
    # includes the flags. In addition, numba stores the two graphs in separate cache files, because their names differ.
    # Without a canonical form, _flags_canon falls back to repr. The repr of an object with the default repr differs
    # between processes, and a repr may not distinguish two different values.
    flags_canon, ok_flags = _flags_canon(_effective_flags(jit_options))
    return code_block_hash(
        f"code_block = {code_block} derive_hashes = {derive_hashes} type_sigs = {type_sigs}"
        f" flags = {flags_canon} settings = {_settings_canon()}"
    ), ok_flags


def _infer_end_and_derived_nodes(spec: SpecTy, all_inputs_: Dict[str, Type], all_derived_: Dict[str, Type], registry):
    if spec.name in all_inputs_ or spec.name in all_derived_:
        return
    if isinstance(spec, End):
        all_inputs_[spec.name] = get_ty(spec)
        return
    for source in spec.sources:
        _infer_end_and_derived_nodes(source, all_inputs_, all_derived_, registry)
    all_derived_[spec.name] = get_ty(spec)


def infer_end_and_derived_nodes(access_nodes: PyTuple[SpecTy, ...], registry):
    """Return the specs of the end nodes and of the derived nodes that ``access_nodes`` need, as two lists.

    The walk starts at each access node and follows the sources depth first, so a derived node comes after the
    derived nodes it reads; the access nodes are included. Each spec in the lists is the one that ``registry`` holds
    under the node's name, and a name that ``registry`` lacks raises ``KeyError``.
    """
    all_inputs_ = dict()
    all_derived_ = dict()
    for access_node in access_nodes:
        _infer_end_and_derived_nodes(access_node, all_inputs_, all_derived_, registry)
    all_inputs_lst = [registry[name] for name in all_inputs_.keys()]
    all_derived_lst = [registry[name] for name in all_derived_.keys()]
    return all_inputs_lst, all_derived_lst


def make_graph(
    *access_nodes: SpecTy,
    registry: Optional[dict] = None,
    jit_options: Optional[dict] = None
):
    """Create the graph that the given access node specs need, and return the access nodes as ``Work`` nodes.

    The specs are looked up by name in ``registry``, by default ``_specs_registry``. The result is a named tuple with
    one field per access node, named after the node, and each call creates new nodes.

    ``jit_options`` take precedence over numbox's jit options for each derive and for the graph maker, the function that
    creates the nodes. With caching enabled, which is the default, numba caches the graph maker as a function defined
    in this module's file. numba compiles the maker again after a change to a node's name, sources, type or derive, to
    a jit option other than ``cache``, or to a numba setting that changes the generated code. A change to an
    ``init_value`` alone reuses the cached maker, because the initial values are the maker's arguments.

    The maker compiles without a cache when numbox cannot fingerprint a derive, when a node's type contains a
    dispatcher, or when a jit option has no canonical form. If numba has no cache location for this module's file,
    ``make_graph`` issues a ``RuntimeWarning`` and compiles the maker without a cache. A derive that reads its module's
    globals always compiles without a cache.
    """
    if registry is None:
        registry = _specs_registry
    all_inputs_, all_derived_ = infer_end_and_derived_nodes(access_nodes, registry)
    if jit_options is None:
        jit_options = {}
    jit_options = {**jit_options_, **jit_options}
    # numba caches the graph maker as a function defined in this file. A caller's cache option overrides the cache
    # setting that numbox's own jit options carry for this location, so _cached_at_or_uncached asks numba again, and
    # it turns caching off with a warning where numba has no cache location. The derives are cached under their own
    # anchors and use the options unchanged.
    kernel_options = _cached_at_or_uncached(getfile(_file_anchor), jit_options)
    ns = {
        **getmodule(_file_anchor).__dict__,
        **{"jit_options": kernel_options, "ll_make_work": ll_make_work, "njit": njit}
    }
    _make_args = []
    code_txt = StringIO()
    initializers = {}
    derive_hashes = []
    kernel_safe_flags = []
    for input_ in all_inputs_:
        line_ = _input_line(input_, ns, initializers)
        code_txt.write(f"\n\t{line_}")
    for derived_ in all_derived_:
        line_ = _derived_line(
            derived_, ns, initializers, derive_hashes, kernel_safe_flags, _make_args, jit_options
        )
        code_txt.write(f"\n\t{line_}")
    type_ids = [(n.name, _type_identity(get_ty(n))) for n in chain(all_inputs_, all_derived_)]
    type_sigs = [(name, h) for name, (h, _c) in type_ids]
    hash_, flags_cacheable = _kernel_fingerprint(code_txt.getvalue(), derive_hashes, type_sigs, jit_options)
    kernel_cacheable = flags_cacheable and all(kernel_safe_flags) and all(c for _name, (_h, c) in type_ids)
    if not kernel_cacheable:
        # The graph maker compiles without a cache when a derive has only a placeholder fingerprint, when a node's type
        # contains a dispatcher, or when a jit flag has no canonical form. A placeholder does not change when the value
        # it replaces changes. A flag's repr may differ between processes. In every new process, numba finds no entry
        # in its cache index for a type that contains a dispatcher. The maker's name stays as computed above.
        ns["jit_options"] = {**kernel_options, "cache": False}
    access_nodes_names = [n.name for n in access_nodes]
    tup_ = ", ".join(access_nodes_names) + ","
    code_txt.write(f"""\n\taccess_tuple = ({tup_})""")
    code_txt.write("\n\treturn access_tuple")
    code_txt = code_txt.getvalue()
    # The derives and the initial values are the maker's parameters, not constants in its code. Only the numba types
    # of the initial values, in type_sigs, are part of the hash that names the maker. A new init_value of the same type
    # therefore reuses the cached maker. This holds for a StructRef as well, because its repr is not part of the hash.
    make_params = ", ".join(chain(_make_args, initializers.keys()))
    make_name = f"_make_{hash_}"
    code_txt = f"""
@njit(**jit_options)
def {make_name}({make_params}):""" + code_txt + f"""
access_tuple_ = {make_name}({make_params})
"""
    code = compile(code_txt, getfile(_file_anchor), mode="exec")
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    access_tuple_ = ns["access_tuple_"]
    Access = namedtuple("Access", access_nodes_names)
    return Access(*access_tuple_)
