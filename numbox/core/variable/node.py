from numbox.core.variable.variable import Namespace


class Node:
    def __init__(self, name: str, source: str, registry: dict[str, Namespace]):
        self.variable = registry[source][name]
        self.inputs = [Node(inp_name, inp_source, registry) for inp_name, inp_source in self.variable.inputs.items()]

    def get_input(self, i):
        return self.inputs[i]

    def get_inputs_names(self):
        return [inp.variable.qual_name() for inp in self.inputs]

    def __str__(self):
        return self.variable.qual_name()
