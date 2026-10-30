"""Represent a variable of a graph and its dependencies as linked Python objects, one `Node` for each variable.

`make_node` follows the inputs of each variable through a graph's registry and returns the node of the requested
variable. A node holds the nodes of its inputs. A variable that several other variables read has one node, and the
nodes of those variables share it.
"""
from numbox.core.variable.variable import Namespace, Variable


class Node:
    """A variable with the nodes of its inputs, in the order of its formula's parameters."""
    def __init__(self, variable: Variable, inputs: list['Node']):
        self.variable = variable
        self.inputs = inputs

    def get_input(self, i):
        """Return the node of the input at position `i`."""
        return self.inputs[i]

    def get_inputs_names(self):
        """Return the qualified names of the inputs, in order."""
        return [inp.variable.qual_name() for inp in self.inputs]

    def __str__(self):
        return self.variable.qual_name()


def make_node(name: str, source: str, registry_: dict[str, Namespace]):
    """Return the node of the variable `name` of the namespace `source`, linked to the nodes of its dependencies.

    `registry_` maps each namespace name to its namespace, as `Graph.registry` does. The function creates one node for
    each variable, and the nodes of all the variables that read it share that node. The function recurses through the
    inputs without a check for cycles, so a cycle raises ``RecursionError``. An unknown namespace raises ``KeyError``.
    """
    made = {}

    def _make(name_: str, source_: str):
        key = (name_, source_)
        node_ = made.get(key)
        if node_:
            return node_
        variable_ = registry_[source_][name_]
        inputs_ = [_make(inp_name, inp_source) for inp_name, inp_source in variable_.inputs.items()]
        node_ = Node(variable_, inputs_)
        made[key] = node_
        return node_

    return _make(name, source)
