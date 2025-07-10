from hashlib import md5
from inspect import getfile, getmodule, getsource
from io import StringIO
from itertools import chain
from numba import njit, typeof
from numba.core.types import Type
from typing import Any, Callable, NamedTuple, Optional, Sequence, Union

from numbox.core.configurations import default_jit_options
from numbox.core.work.lowlevel_work_utils import ll_make_work
from numbox.utils.highlevel import cres


def _file_anchor():
    raise NotImplementedError


_nodes_names = set()


class _End(NamedTuple):
    name: str
    init_value: Any
    ty: Optional[type | Type] = None


def _new(cls, super_proxy, *args, **kwargs):
    name = kwargs.get("name")
    assert name, "`name` key-word argument has not been provided"
    if name in _nodes_names:
        raise ValueError(f"Node '{name}' has already been defined on this graph. Pick a different name.")
    _nodes_names.add(name)
    return super_proxy.__new__(cls, *args, **kwargs)


class End(_End):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        return _new(cls, super(), *args, **kwargs)


class _Derived(NamedTuple):
    name: str
    init_value: Any
    derive: Callable
    sources: Sequence[Union['Derived', End]]
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
    sources_ = sources_ + ", " if "," not in sources_ else sources_
    ty_ = get_ty(derived_)
    derive_func = derived_.derive
    derive_hashes.append(md5(getsource(derive_func).encode("utf-8")).hexdigest())
    derive_ = _derived_cres(ty_, derived_.sources, derive_func, jit_options)
    derive_name = f"{name_}_derive"
    init_name = f"{name_}_init"
    _make_args.append(derive_name)
    ns[derive_name] = derive_
    ns[init_name] = init_
    initializers[init_name] = init_
    return f"""{name_} = ll_make_work("{name_}", {init_name}, ({sources_}), {derive_name})"""


def _verify_access_nodes(
    all_inputs_: Sequence[End],
    all_derived_: Sequence[Derived],
    access_nodes: Sequence[SpecTy]
):
    for access_node in access_nodes:
        assert access_node in all_inputs_ or access_node in all_derived_, f"{access_node} cannot be reached"


def code_block_hash(code_txt: str):
    """ Re-compile and re-save cache when source code has changed. """
    return md5(code_txt.encode("utf-8")).hexdigest()


def make_graph(
    all_inputs_: Sequence[End],
    all_derived_: Sequence[Derived],
    access_nodes: SpecTy | Sequence[SpecTy],
    jit_options: Optional[dict] = None
):
    if isinstance(access_nodes, SpecTy):
        access_nodes = (access_nodes,)
    _verify_access_nodes(all_inputs_, all_derived_, access_nodes)
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
    derive_hashes=[]
    for input_ in all_inputs_:
        line_ = _input_line(input_, ns, initializers)
        code_txt.write(f"\n\t{line_}")
    for derived_ in all_derived_:
        line_ = _derived_line(derived_, ns, initializers, derive_hashes, _make_args, jit_options)
        code_txt.write(f"\n\t{line_}")
    hash_str = f"code_block = {code_txt.getvalue()} initializers = {list(initializers.values())} derive_hashes = {derive_hashes}"
    hash_ = code_block_hash(hash_str)
    code_txt.write(f"""\n\taccess_tuple = ({", ".join([n.name for n in access_nodes])})""")
    code_txt.write(f"\n\treturn access_tuple")
    code_txt = code_txt.getvalue()
    make_params = ", ".join(chain(_make_args, initializers.keys()))
    make_name = f"_make_{hash_}"
    code_txt = f"""
@njit(**jit_options)
def {make_name}({make_params}):""" + code_txt + f"""
return_node_ = {make_name}({make_params})
"""
    code = compile(code_txt, getfile(_file_anchor), mode="exec")
    exec(code, ns)
    return_node_ = ns["return_node_"]
    return return_node_
