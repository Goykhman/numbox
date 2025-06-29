import numpy
from numbox.core.work.node import make_node
from numbox.core.work.print_tree import make_image
from numbox.core.work.work import make_work
from test.auxiliary_utils import collect_and_run_tests
from test.random_graph import random_graph
from test.core.random_image_ref import random_image_ref


def test_make_node_graph():
    n1 = make_node("first")
    n2 = make_node("second")
    n3 = make_node("third", inputs=(n1, n2))
    n4 = make_node("fourth")
    n5 = make_node("fifth", inputs=(n3, n4))
    tree_image = make_image(n5)
    tree_image_ref = """
fifth--third---first
       |       |
       |       second
       |
       fourth"""
    assert tree_image == tree_image_ref


def test_node_graph_random():
    numpy.random.seed(137)
    tree, all_nodes = random_graph(50)
    tree_image = make_image(tree)
    assert tree_image == random_image_ref
    assert all_nodes["vayg"].all_inputs_names() == ["tankwfu", "nwuh", "yjthkuaqw"]


def test_make_work_graph():
    w1 = make_work("first", 0.0)
    w2 = make_work("second", 0.0, sources=(w1,))
    w3 = make_work("third", 0.0, sources=(w2,))
    w4 = make_work("fourth", 0.0)
    w5 = make_work("fifth", 0.0, sources=(w3, w4))
    tree_image = make_image(w5)
    tree_image_ref = """
fifth--third---second--first
       |
       fourth"""
    assert tree_image == tree_image_ref


if __name__ == "__main__":
    collect_and_run_tests(__name__)
