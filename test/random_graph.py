import numpy
import string
from numbox.core.node import make_node


letters = [c for c in string.ascii_lowercase]


def generate_string(all_nodes):
    while True:
        n = numpy.random.randint(2, 11)
        name = "".join(numpy.random.choice(letters, n))
        if name not in all_nodes:
            return name


def random_node(all_nodes, max_size):
    assert max_size > 0, "cannot generate any more nodes"
    name = generate_string(all_nodes)
    if max_size == 1:
        return 0, make_node(name)
    num_inputs = numpy.random.randint(0, numpy.sqrt(max_size))
    max_size -= num_inputs
    node_inputs = []
    for i in range(num_inputs):
        max_size, input_ = random_node(all_nodes, max_size)
        node_inputs.append(input_)
    all_nodes[name] = make_node(name, inputs=tuple(node_inputs))
    return max_size, all_nodes[name]


def random_graph(max_size):
    all_nodes = {}
    _, tree_ = random_node(all_nodes, max_size)
    return tree_, all_nodes
