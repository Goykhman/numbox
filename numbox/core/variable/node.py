from numbox.core.variable.variable import Namespace, Variable


class Node:
    def __init__(self, variable: Variable, inputs: list['Node']):
        self.variable = variable
        self.inputs = inputs

    def get_input(self, i):
        return self.inputs[i]

    def get_inputs_names(self):
        return [inp.variable.qual_name() for inp in self.inputs]

    def __str__(self):
        return self.variable.qual_name()


def make_node(name: str, source: str, registry_: dict[str, Namespace]):
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
