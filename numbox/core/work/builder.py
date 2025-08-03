from collections import namedtuple
from hashlib import sha256
from inspect import getfile, getmodule, getsource
from io import StringIO
from itertools import chain
from numba import njit, typeof
from numba.core.types import Type
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple as PyTuple, Union

from numbox.core.configurations import default_jit_options
from numbox.core.work.lowlevel_work_utils import ll_make_work
from numbox.utils.highlevel import cres


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


def _derived_cres(ty, sources: Sequence[End], derive, jit_options=None):
    jit_options = jit_options if jit_options is not None else {}
    sources_ty = []
    for source in sources:
        source_ty = get_ty(source)
        sources_ty.append(source_ty)
    derive_sig = ty(*sources_ty)
    derive_cres = cres(derive_sig, **jit_options)(derive)
    _derive_funcs[id(derive_cres)] = derive
    return derive_cres


def _derived_line(
    derived_: Derived, ns: dict, initializers: dict, derive_hashes: list, _make_args: list, jit_options=None
):
    name_ = derived_.name
    init_ = derived_.init_value
    sources_ = ", ".join([s.name for s in derived_.sources])
    sources_ = sources_ + ", " if sources_ and "," not in sources_ else sources_
    ty_ = get_ty(derived_)
    derive_func = derived_.derive
    derive_hashes.append(sha256(getsource(derive_func).encode("utf-8")).hexdigest())
    derive_ = _derived_cres(ty_, derived_.sources, derive_func, jit_options)
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
    jit_options = {**default_jit_options, **jit_options}
    ns = {
        **getmodule(_file_anchor).__dict__,
        **{"jit_options": jit_options, "ll_make_work": ll_make_work, "njit": njit}
    }
    _make_args = []
    code_txt = StringIO()
    initializers = {}
    derive_hashes = []
    for input_ in all_inputs_:
        line_ = _input_line(input_, ns, initializers)
        code_txt.write(f"\n\t{line_}")
    for derived_ in all_derived_:
        line_ = _derived_line(derived_, ns, initializers, derive_hashes, _make_args, jit_options)
        code_txt.write(f"\n\t{line_}")
    hash_str = f"code_block = {code_txt.getvalue()} initializers = {list(initializers.values())} derive_hashes = {derive_hashes}"  # noqa: E501
    hash_ = code_block_hash(hash_str)
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
    exec(code, ns)
    access_tuple_ = ns["access_tuple_"]
    Access = namedtuple("Access", access_nodes_names)
    return Access(*access_tuple_)
