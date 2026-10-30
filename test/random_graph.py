"""A generator of random trees of ``Node`` objects, for the drawing test in ``test_print_tree``."""
import numpy
import string
from numbox.core.work.node import make_node


letters = [c for c in string.ascii_lowercase]


def generate_string(all_nodes):
    """Return a random name of 2 to 10 lowercase letters that is not a key of ``all_nodes``."""
    while True:
        n = numpy.random.randint(2, 11)
        name = "".join(numpy.random.choice(letters, n))
        if name not in all_nodes:
            return name


def random_node(all_nodes, max_size):
    """Create a node with random inputs, and return the remaining budget and the node.

    ``max_size`` is a budget that limits the number of nodes. The function draws a number of inputs below
    ``sqrt(max_size)``, subtracts it from the budget, and creates each input in the same way from the budget that
    remains. Each node is added to ``all_nodes`` under its name, except a node created with a budget of 1, which has no
    inputs and leaves a budget of 0. A budget of 0 fails an assertion.
    """
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
    """Return a random tree of ``Node`` objects and a dictionary of its nodes by name.

    ``max_size`` is the budget of nodes that ``random_node`` describes. The tree depends on NumPy's global random state,
    which ``test_print_tree`` seeds with 137 to get a fixed tree.
    """
    all_nodes = {}
    _, tree_ = random_node(all_nodes, max_size)
    return tree_, all_nodes
