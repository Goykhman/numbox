"""Build a large random tree of ``Work`` nodes in one jitted function, and time its creation and its calculation.

``make_create_nodes_func`` generates the source of a function ``create_nodes``. That function creates the end nodes
with ``ll_make_work`` and then, level by level, derived nodes that each read one to three consecutive nodes of the
level below, until one root node remains. The derive of a node is ``calc_1``, ``calc_2`` or ``calc_3``, according to
its number of sources. ``single_run`` creates the tree and calculates the root once. ``multiple_run`` loads the values
of each entity from a structured array into the end nodes and calculates the root for each entity, in jitted code. The
``timer`` decorator logs the duration of each timed step at INFO. ``random`` and NumPy's generator are seeded with 1 at
import, so a process that makes the same calls builds the same trees. ``stress_work_runner`` runs both functions.
"""
import numpy
import random

from inspect import getfile, getmodule
from numba import njit
from numba.core.types import float64, unicode_type
from numba.typed import Dict
from numpy.random import randint, seed

from numbox.core.any.any_type import AnyType
from numbox.core.configurations import jit_options
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
    """Return ``num_of_pure_inputs`` and, for each end node ``w_<i>``, a line of source that creates it."""
    lines = []
    for i in range(num_of_pure_inputs):
        w = f'w_{i} = ll_make_work("w_{i}", 0.0, (), None)'
        lines.append(w)
    return num_of_pure_inputs, lines


@cres(float64(float64), **jit_options)
def calc_1(x):
    return (3.14 + x) / 1.41


@cres(float64(float64, float64), **jit_options)
def calc_2(x, y):
    return x * (y - 1.41) + 3.14


@cres(float64(float64, float64, float64), **jit_options)
def calc_3(x, y, z):
    return x + y * z if z >= 1 else z * 3 if z >= 0 else x + 2.17 * y + 3.14 * z


def make_derived_node(num_sources, i_, j_):
    """Return the line of source that creates node ``w_<i_>`` from the ``num_sources`` nodes that start at ``w_<j_>``.

    The derive is ``calc_1_``, ``calc_2_`` or ``calc_3_``, the parameter of ``create_nodes`` for that number of sources.
    """
    derive = "calc_1_" if num_sources == 1 else "calc_2_" if num_sources == 2 else "calc_3_"
    sources = ", ".join([f"w_{k_}" for k_ in range(j_, j_ + num_sources)])
    sources = sources + ", " if "," not in sources else sources
    sources = f"({sources})"
    declaration = f'w_{i_} = ll_make_work("w_{i_}", 0.0, {sources}, {derive})'
    return declaration


def create_tree_level(lower_level_start: int, lower_level_end: int):
    """Return the lines that create the level above the nodes ``lower_level_start`` to ``lower_level_end - 1``.

    The result also holds the index of the first node of the new level and the index after its last node. Each new
    node reads one to three consecutive nodes of the level below, a number drawn at random, and the new nodes together
    read every node of that level once.
    """
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
    """Return the lines that create the end nodes and every level of derived nodes, up to the root."""
    s = 0
    e, all_lines = create_pure_inputs(num_of_pure_inputs)
    while e - s > 1:
        lines, s, e = create_tree_level(s, e)
        all_lines.extend(lines)
    return all_lines


def make_create_nodes_func(num_of_pure_inputs):
    """Return the generated Python function ``create_nodes(calc_1_, calc_2_, calc_3_)``.

    The function creates a tree of ``num_of_pure_inputs`` end nodes and returns its root.
    """
    lines_ = create_derived_nodes(num_of_pure_inputs)
    func_block = "\n".join([f"    {l_}" for l_ in lines_])
    code_txt = f"""
def create_nodes(calc_1_, calc_2_, calc_3_):
{func_block}
    return w_{len(lines_) - 1}
"""
    # create_nodes is defined in this module's namespace, and its code carries this module's file name, so numba finds
    # a cache location for it.
    ns = getmodule(create_pure_inputs).__dict__
    ns["ll_make_work"] = ll_make_work
    code = compile(code_txt, getfile(create_pure_inputs), mode="exec")
    exec(code, ns)
    return ns["create_nodes"]


@timer
def do_create_nodes(create_nodes_):
    """Compile ``create_nodes_`` with numbox's jit options, call it with the three derives, and return the root."""
    create_nodes = njit(**jit_options)(create_nodes_)
    w = create_nodes(calc_1, calc_2, calc_3)
    return w


@timer
def do_calculate(w_):
    """Calculate the node ``w_`` and the nodes that it depends on."""
    w_.calculate()


def single_run(num_of_inputs):
    """Create a tree of ``num_of_inputs`` end nodes, calculate its root once, and return the root."""
    create_nodes = make_create_nodes_func(num_of_inputs)
    w = do_create_nodes(create_nodes)
    do_calculate(w)
    return w


def prepare_input_data(num_of_sources=NUM_OF_PURE_INPUTS_DEFAULT, num_of_entities=NUM_OF_ENTITIES_DEFAULT):
    """Return a structured array with a row for each entity and a float64 field ``w_<i>`` for each end node.

    The values are random, at least 0 and below 10.
    """
    data_ty = numpy.dtype([(f"w_{i}", numpy.float64) for i in range(num_of_sources)], align=True)
    data_ = numpy.empty((num_of_entities,), dtype=data_ty)
    for i in range(num_of_sources):
        value_ = 10 * numpy.random.rand(num_of_entities)
        data_[f"w_{i}"] = value_
    return data_


@njit(**jit_options)
def run_entity(total, loader_dict):
    """Load the values of ``loader_dict`` into the graph of ``total``, calculate ``total``, and return its value."""
    total.load(loader_dict)
    total.calculate()
    return total.data


@timer
@njit(**jit_options)
def run(total, data, loader_dict, num_of_entities=NUM_OF_ENTITIES_DEFAULT):
    """Calculate ``total`` for each row of ``data``, loaded into its end nodes, and return the values as an array."""
    total_data = numpy.empty((num_of_entities,), dtype=numpy.float64)
    for i in range(num_of_entities):
        load_array_row_into_dict(data, i, loader_dict)
        total_data[i] = run_entity(total, loader_dict)
    return total_data


def multiple_run(num_of_inputs, num_of_entities):
    """Create a tree of ``num_of_inputs`` end nodes and calculate its root for ``num_of_entities`` random entities.

    The result is the array of the root's values, one for each entity.
    """
    create_nodes = make_create_nodes_func(num_of_inputs)
    node = do_create_nodes(create_nodes)
    data = prepare_input_data(num_of_inputs, num_of_entities)
    loader_dict = Dict.empty(unicode_type, AnyType)
    total_data = run(node, data, loader_dict, num_of_entities)
    return total_data
