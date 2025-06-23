import random

from inspect import getfile, getmodule
from numba import float64, njit
from numpy import ndarray
from numpy.random import randint, seed
from time import perf_counter

from numbox.core.configurations import default_jit_options
from numbox.core.work.print_tree import make_image
from numbox.core.work.lowlevel_work_utils import ll_make_work
from numbox.core.work.work import make_work, Work
from numbox.core.work.work_utils import make_init_data, make_work_helper
from numbox.utils.highlevel import cres


random.seed(1)
seed(1)


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
    # print(f"code_txt =\n{code_txt}")
    code = compile(code_txt, getfile(create_pure_inputs), mode="exec")
    exec(code, ns)
    return ns["create_nodes"]


if __name__ == "__main__":
    create_nodes = make_create_nodes_func(1000)
    t0 = perf_counter()
    create_nodes = njit(**default_jit_options)(create_nodes)
    w = create_nodes(calc_1, calc_2, calc_3)
    t1 = perf_counter()
    print(f"Created nodes in {t1-t0}")
    # w_image = make_image(w)
    # print(w_image)

    t2 = perf_counter()
    w.calculate()
    t3 = perf_counter()
    print(f"Calculated in {t3 - t2}")

    t2 = perf_counter()
    w.calculate()
    t3 = perf_counter()
    print(f"Calculated in {t3 - t2}")

    t2 = perf_counter()
    w.calculate()
    t3 = perf_counter()
    print(f"Calculated in {t3 - t2}")
    print(w.data)
