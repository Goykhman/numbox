from collections import namedtuple
from hashlib import sha256
from inspect import getfile, getmodule
from io import StringIO
from itertools import chain
from numba import njit, typeof
from numba.core.types import Type
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple as PyTuple, Union

from numbox.core.configurations import jit_options as jit_options_
from numbox.core.work.lowlevel_work_utils import ll_make_work
from numbox.utils.fingerprint import (
    _Unfingerprintable, _codegen_env_canon, _effective_flags, _fingerprint_function,
    _fingerprint_function_best_effort, _flags_canon, _loaded_global_names,
)
from numbox.utils.highlevel import cres, _signature_identity, _type_identity
from numbox.utils.preprocessing import _anchor_root, _materialize_anchor, _orphan_anchor_sweep


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
    return spec_.ty or typeof(spec_.init_value)


_derive_funcs = {}


def _derive_fingerprint(derive) -> tuple[str, bool, bool]:
    """Return ``(fingerprint, derive_cacheable, kernel_safe)`` for a derive.

    Routes the derive through the deep walker so its referenced-global *values*
    enter the kernel fingerprint, replacing a bare ``sha256(getsource(...))``
    that was blind to them (and raised ``OSError`` on exec/REPL-defined derives).

    A derive that reads module globals is fingerprinted but **not** cacheable:
    it is compiled as a standalone dispatcher (not inlined into the
    content-addressed kernel), and numba's own cache key covers only
    ``co_code`` + closure, so a changed global would silently serve a stale
    binary. Such a derive is compiled uncached -- recompiled per process,
    never wrong -- mirroring ``compile_kernel``'s degrade path. Its fingerprint
    is still process-stable and captures the global's value at build time, so
    the enclosing kernel stays cacheable (``kernel_safe`` True): a changed global
    re-keys the kernel.

    An **un-fingerprintable** derive is different: ``_fingerprint_function``
    raises, and the best-effort placeholder used instead is process-stable (never
    an address, so the kernel name no longer explodes per run) but cannot be
    trusted to change when the un-canonicalizable state does. Both the derive and
    the enclosing kernel must therefore compile uncached (``kernel_safe`` False):
    recompiled per process, never wrong. This is the orphan-per-run class
    -- ``@id``-based names and unconditional kernel caching re-opened it.

    "Reads module globals" is measured off the instruction stream
    (``_loaded_global_names``), not ``co_names``: a record-field access such as
    ``a_[t].c1`` is a ``LOAD_ATTR`` whose name would otherwise collide with a
    module global of the same name, spuriously marking the derive uncacheable
    (and, via ``_fingerprint_function``, forcing the best-effort fallback below).
    A genuine ``cfg.SCALE`` read still counts -- its base ``cfg`` is a real
    ``LOAD_GLOBAL`` -- so module-state derives stay uncacheable.
    """
    try:
        fingerprint = _fingerprint_function(derive, set())
    except (_Unfingerprintable, RecursionError):
        return _fingerprint_function_best_effort(derive), False, False
    reads_module_state = any(
        name in derive.__globals__ for name in _loaded_global_names(derive.__code__)
    )
    return fingerprint, not reads_module_state, True


_DERIVE_ANCHOR_SUBDIR = "numbox-derive"
# Clear `.tmp-*` anchors left behind by a SIGKILL'd writer, as every other
# anchor-writing module does for its own subdir.
_orphan_anchor_sweep(_DERIVE_ANCHOR_SUBDIR)


def _derive_anchor_cres(derive_sig, sig_canon, derive, derive_fp, jit_options):
    """Compile ``derive`` under a cache identity that depends on the jit flags.

    Folding the flags into the kernel name re-keys the kernel but not the derive:
    the derive is compiled as its own ``cres``, and numba's cache file for it is
    named after the derive's own source file and qualname, so a graph built under
    ``error_model="numpy"`` and one built under the default model share it. The
    second process then links the first's binary -- a division by zero returns
    ``inf`` where it must raise.

    So the derive is compiled through a generated wrapper whose *name* carries a
    digest of the derive fingerprint plus the effective flags, written to its own
    on-disk anchor. A flag change moves the name, which moves the anchor, which
    moves numba's cache file. Mirrors ``compile_kernel._compile``.

    The wrapper delegates to the derive njit-wrapped with ``cache=False``: it is
    linked into the wrapper's cached artifact, and it must not carry a cache of
    its own, which would be flag-blind exactly like the one this works around.

    Returns ``None`` if the flags cannot be canonicalized or no anchor can be
    written; the caller must then compile the derive **uncached**, because the
    anchor is the only thing making a cached derive flag-safe.
    """
    flags = _effective_flags(jit_options)
    flags_canon, ok_flags = _flags_canon(flags)
    if not ok_flags:
        # A flag with no canonical form cannot be keyed on; a key blind to a flag
        # cannot protect the binary that flag produced.
        return None
    digest = sha256(
        f"derive-anchor-v2\n{derive_fp}\n# sig: {sig_canon}\n# flags: {flags_canon}"
        f"\n# codegen_env: {_codegen_env_canon()}".encode("utf-8")
    ).hexdigest()[:16]
    name = f"_derive_{digest}"
    params = ", ".join(f"a{i}" for i in range(len(derive_sig.args)))
    src = (
        f"@_cres(_derive_sig, **_derive_jit_options)\n"
        f"def {name}({params}):\n"
        f"    return _inner({params})\n"
    )
    anchor = _anchor_root(_DERIVE_ANCHOR_SUBDIR) / f"{name}.py"
    try:
        anchor.parent.mkdir(parents=True, exist_ok=True)
        _materialize_anchor(anchor, src)
    except OSError:
        return None
    ns = {
        "_cres": cres,
        "_derive_sig": derive_sig,
        "_derive_jit_options": jit_options,
        "_inner": njit(**{**flags, "cache": False})(derive),
        # __name__ must be an importable module so numba can rebuild the cached
        # overload's environment in another process; mirrors compile_kernel.
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
        # The anchor is what makes a cached derive flag-safe. Falling back to a
        # plain cache=True compile here would restore exactly the flag-blind
        # cache entry -- named after the derive's own source file and qualname --
        # that the anchor exists to avoid, so the degrade must drop the cache
        # too: recompiled per process, never wrong.
        jit_options = {**jit_options, "cache": False}
    elif jit_options.get("cache") and not sig_cacheable:
        # A source type -- e.g. a Dispatcher wrapping an un-fingerprintable body
        # -- has only a best-effort identity: the anchor name would be process-
        # stable but blind to a change in that type, so a later process could link
        # a stale derive binary. Drop the cache instead: recompiled per process,
        # never wrong.
        jit_options = {**jit_options, "cache": False}
    try:
        derive_cres = cres(derive_sig, **jit_options)(derive)
    except RuntimeError as e:
        # numba refuses cache=True for a function whose source file it cannot
        # locate (an exec/<string>- or <stdin>-defined derive). Fall back to
        # uncached rather than crashing -- a body numba cannot cache cannot go
        # stale anyway. Derives defined in real modules or notebooks have a
        # locator and are unaffected.
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
    """ Re-compile and re-save cache when source code has changed. """
    return sha256(code_txt.encode("utf-8")).hexdigest()


def _kernel_fingerprint(code_block: str, derive_hashes: list, type_sigs: list, jit_options=None) -> tuple[str, bool]:
    """Return ``(hash, flags_cacheable)`` for the generated kernel.

    The hash is baked into the name ``_make_<hash>``.

    Folds each node's declared type (``type_sigs``) in alongside the body text and
    derive source hashes, since a declared ty reaches the generated code only as the
    bare global name ``{name}_ty`` and so is not otherwise captured by the body.

    Also folds the resolved jit flags and the process codegen env knobs. The body
    text renders them only as the literal ``@njit(**jit_options)``, so without this
    two graphs differing solely in ``error_model`` or ``fastmath`` produced the same
    ``_make_<hash>`` name, shared numba's cache key, and the second process linked
    the first's binary -- a kernel cached under ``error_model="numpy"`` returned
    ``inf`` for a division by zero that the default model must raise on.

    ``flags_cacheable`` is False when a flag had no canonical form (a
    ``pipeline_class``, a numba-typed ``locals``). The fold then degrades to
    ``repr``, which is neither process-stable for a default-repr object nor
    guaranteed to distinguish two behaviourally different values, so the caller
    must compile the kernel uncached -- the same contract ``_derive_anchor_cres``
    and ``compile_kernel`` already honour for this flag.
    """
    flags_canon, ok_flags = _flags_canon(_effective_flags(jit_options))
    return code_block_hash(
        f"code_block = {code_block} derive_hashes = {derive_hashes} type_sigs = {type_sigs}"
        f" flags = {flags_canon} codegen_env = {_codegen_env_canon()}"
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
    if registry is None:
        registry = _specs_registry
    all_inputs_, all_derived_ = infer_end_and_derived_nodes(access_nodes, registry)
    if jit_options is None:
        jit_options = {}
    jit_options = {**jit_options_, **jit_options}
    ns = {
        **getmodule(_file_anchor).__dict__,
        **{"jit_options": jit_options, "ll_make_work": ll_make_work, "njit": njit}
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
        # A derive with only a best-effort fingerprint, a node typed as a
        # Dispatcher wrapping an un-fingerprintable body, or a jit flag with no
        # canonical form has a content-blind identity folded into the kernel name;
        # compile the kernel without an on-disk cache so a stale binary cannot be
        # linked -- recompiled per process, never wrong. The
        # name (fed by type_sigs) is unchanged, so nothing else re-keys.
        ns["jit_options"] = {**jit_options, "cache": False}
    access_nodes_names = [n.name for n in access_nodes]
    tup_ = ", ".join(access_nodes_names) + ","
    code_txt.write(f"""\n\taccess_tuple = ({tup_})""")
    code_txt.write("\n\treturn access_tuple")
    code_txt = code_txt.getvalue()
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
