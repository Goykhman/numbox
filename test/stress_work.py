import numpy
import random

from inspect import getfile, getmodule
from numba import njit
from numba.core.types import float64, unicode_type
from numba.typed import Dict
from numpy.random import randint, seed

from numbox.core.any.any_type import AnyType
from numbox.core.configurations import default_jit_options
from numbox.core.work.loader_utils import load_array_row_into_dict
from numbox.core.work.lowlevel_work_utils import ll_make_work
from numbox.utils.highlevel import cres
from numbox.utils.timer import timer


random.seed(1)
seed(1)


NUM_OF_ENTITIES_DEFAULT = 1000
NUM_OF_PURE_INPUTS_DEFAULT = 1000


all_nodes = {}


def create_pure_inputs(num_of_pure_inputs: int = NUM_OF_PURE_INPUTS_DEFAULT):
    lines = []
    for i in range(num_of_pure_inputs):
        w = f'w_{i} = ll_make_work("w_{i}", 0.0, (), None)'
        lines.append(w)
    return num_of_pure_inputs, lines


@cres(float64(float64), **default_jit_options)
def calc_1(x):
    return (3.14 + x) / 1.41


@cres(float64(float64, float64), **default_jit_options)
def calc_2(x, y):
    return x * (y - 1.41) + 3.14


@cres(float64(float64, float64, float64), **default_jit_options)
def calc_3(x, y, z):
    return x + y * z if z >= 1 else z * 3 if z >= 0 else x + 2.17 * y + 3.14 * z


def make_derived_node(num_sources, i_, j_):
    derive = "calc_1_" if num_sources == 1 else "calc_2_" if num_sources == 2 else "calc_3_"
    sources = ", ".join([f"w_{k_}" for k_ in range(j_, j_ + num_sources)])
    sources = sources + ", " if "," not in sources else sources
    sources = f"({sources})"
    declaration = f'w_{i_} = ll_make_work("w_{i_}", 0.0, {sources}, {derive})'
    return declaration


def create_tree_level(lower_level_start: int, lower_level_end: int):
    lines = []
    lower_level_index_ = lower_level_start
    node_index_ = lower_level_end
    lower_level_start_new = node_index_
    while lower_level_index_ < lower_level_end - 3:
        num_sources = randint(1, 4)
        lines.append(make_derived_node(num_sources, node_index_, lower_level_index_))
        node_index_ += 1
        lower_level_index_ += num_sources
    if lower_level_index_ < lower_level_end:
        lines.append(make_derived_node(lower_level_end - lower_level_index_, node_index_, lower_level_index_))
        node_index_ += 1
    lower_level_end_new = node_index_
    return lines, lower_level_start_new, lower_level_end_new


def create_derived_nodes(num_of_pure_inputs: int = NUM_OF_PURE_INPUTS_DEFAULT):
    s = 0
    e, all_lines = create_pure_inputs(num_of_pure_inputs)
    while e - s > 1:
        lines, s, e = create_tree_level(s, e)
        all_lines.extend(lines)
    return all_lines


def make_create_nodes_func(num_of_pure_inputs):
    lines_ = create_derived_nodes(num_of_pure_inputs)
    func_block = "\n".join([f"    {l_}" for l_ in lines_])
    code_txt = f"""
def create_nodes(calc_1_, calc_2_, calc_3_):
{func_block}
    return w_{len(lines_) - 1}
"""
    ns = getmodule(create_pure_inputs).__dict__
    ns["ll_make_work"] = ll_make_work
    code = compile(code_txt, getfile(create_pure_inputs), mode="exec")
    exec(code, ns)
    return ns["create_nodes"]


@timer
def do_create_nodes(create_nodes_):
    create_nodes = njit(**default_jit_options)(create_nodes_)
    w = create_nodes(calc_1, calc_2, calc_3)
    return w


@timer
def do_calculate(w_):
    w_.calculate()


def single_run(num_of_inputs):
    create_nodes = make_create_nodes_func(num_of_inputs)
    w = do_create_nodes(create_nodes)
    do_calculate(w)
    return w


def prepare_input_data(num_of_sources=NUM_OF_PURE_INPUTS_DEFAULT, num_of_entities=NUM_OF_ENTITIES_DEFAULT):
    data_ty = numpy.dtype([(f"w_{i}", numpy.float64) for i in range(num_of_sources)], align=True)
    data_ = numpy.empty((num_of_entities,), dtype=data_ty)
    for i in range(num_of_sources):
        value_ = 10 * numpy.random.rand(num_of_entities)
        data_[f"w_{i}"] = value_
    return data_


@njit(**default_jit_options)
def run_entity(total, loader_dict):
    total.load(loader_dict)
    total.calculate()
    return total.data


@timer
@njit(**default_jit_options)
def run(total, data, loader_dict, num_of_entities=NUM_OF_ENTITIES_DEFAULT):
    total_data = numpy.empty((num_of_entities,), dtype=numpy.float64)
    for i in range(num_of_entities):
        load_array_row_into_dict(data, i, loader_dict)
        total_data[i] = run_entity(total, loader_dict)
    return total_data


def multiple_run(num_of_inputs, num_of_entities):
    create_nodes = make_create_nodes_func(num_of_inputs)
    node = do_create_nodes(create_nodes)
    data = prepare_input_data(num_of_inputs, num_of_entities)
    loader_dict = Dict.empty(unicode_type, AnyType)
    total_data = run(node, data, loader_dict, num_of_entities)
    return total_data
