"""Tests of numbox.core.variable.variable, the framework that builds a graph of variables and evaluates it in Python.

The tests build graphs from variable specifications and compile them into a CompiledGraph. They check the namespaces,
the external variables, the values that execute and recompute write, the text of Graph.explain, the result of
Graph.dependents_of and the text drawing that make_image returns for a node's tree. In one test, the formulas store a
quantity in an external Storage object at each step and read it at the next step. Another test checks that recompute
calculates a variable from its formula when the same call changes one of the variable's inputs. The last test checks
that Variable refuses a dot in a variable's name and accepts one in a namespace's name. test_compile_kernel.py tests
graphs compiled into one kernel.
"""
from inspect import getsource
from textwrap import dedent
from typing import Any

import pytest

from numbox.core.variable.variable import CompiledGraph, Graph, Values, Variables, Variable
from numbox.core.variable.node import make_node
from numbox.core.work.print_tree import make_image

from test.auxiliary_utils import collect_and_run_tests


def test_basic():
    """CompiledGraph evaluates the first example of the numbox.core.variable documentation as the documentation shows.

    The section "Building and evaluating a graph" holds the example and its values after execute and after recompute.
    The test also checks the order of the compiled nodes, the text of explain, the dependents of basket.y and the tree
    of variables2.u.
    """

    def derive_x(y_):
        return 2 * y_

    def derive_a(x_):
        return x_ - 74

    def derive_u(a_):
        return 2 * a_

    all_vars_ = {
        "x": {"name": "x", "inputs": {"y": "basket"}, "formula": derive_x},
        "a": {"name": "a", "inputs": {"x": "variables1"}, "formula": derive_a},
        "u": {"name": "u", "inputs": {"a": "variables1"}, "formula": derive_u}
    }

    all_vars = {
        name: {
            **data,
            "metadata": dedent(getsource(data["formula"])) if data["formula"] else None
        } for name, data in all_vars_.items()
    }

    graph = Graph(
        variables_lists={
            "variables1": [all_vars["x"], all_vars["a"]],
            "variables2": [all_vars["u"]],
        },
        external_source_names=["basket"]
    )

    required = ["variables2.u"]
    compiled = graph.compile(required)
    assert isinstance(compiled, CompiledGraph)

    registry = graph.registry
    variables1 = registry["variables1"]
    variables2 = registry["variables2"]

    assert isinstance(variables1, Variables)

    assert list(variables1.keys()) == ["x", "a"]
    assert list(variables2.keys()) == ["u"]

    assert isinstance(variables1["x"], Variable)

    required_external_variables = compiled.required_external_variables

    assert list(required_external_variables.keys()) == ["basket"]
    basket = required_external_variables["basket"]
    assert list(basket.keys()) == ["y"]
    assert basket["y"].name == "y"

    basket_ = registry["basket"]
    assert basket_["y"] is basket["y"]

    values = Values()
    compiled.execute(
        external_values={"basket": {"y": 137}},
        values=values,
    )

    x_var = variables1["x"]
    a_var = variables1["a"]
    u_var = variables2["u"]

    assert values.get(x_var).value == 274
    assert values.get(a_var).value == 200
    assert values.get(u_var).value == 400

    dag = compiled.ordered_nodes
    assert [v.variable.name for v in dag] == ["y", "x", "a", "u"]

    compiled.recompute({"basket": {"y": 1}}, values)
    assert values.get(basket["y"]).value == 1
    assert values.get(x_var).value == 2
    assert values.get(a_var).value == -72
    assert values.get(u_var).value == -144

    assert graph.explain("variables2.u") == """
'variables2.u' depends on ('variables1.a',) via

def derive_u(a_):
    return 2 * a_

'variables1.a' depends on ('variables1.x',) via

def derive_a(x_):
    return x_ - 74

'variables1.x' depends on ('basket.y',) via

def derive_x(y_):
    return 2 * y_

'y' comes from external source 'basket'
"""

    assert graph.dependents_of("basket.y") == {"variables1.x", "variables2.u", "basket.y", "variables1.a"}

    variables2_u_node = make_node("u", "variables2", graph.registry)
    assert make_image(variables2_u_node) == """
variables2.u--variables1.a--variables1.x--basket.y"""


class Storage:
    """A store of the values of a multistep calculation, by step and by name, outside the graph.

    The graph holds the store itself as a value. A formula reads the value of the previous step from the store and
    stores the value of the current step, so the value of each step is available at the next step. ``get`` raises
    ``ValueError`` for a step or a name without a value. ``set`` raises ``RuntimeError`` for a name that the step
    already holds. ``clean`` deletes the values of the given steps.
    """
    def __init__(self, start_step: int = 1):
        self._data: dict[int, dict[str, Any]] = {start_step - 1: {}}

    def get(self, step_: int, name_: str):
        data_at_step = self._data.get(step_)
        if data_at_step is None:
            raise ValueError(f"No mapping data available at time step {step_}.")
        data = data_at_step.get(name_)
        if data is None:
            raise ValueError(f"No data available for {name_} at {step_}")
        return data

    def set(self, step_: int, name_: str, val_: Any):
        data_at_step = self._data.setdefault(step_, {})
        if name_ in data_at_step:
            raise RuntimeError(f"{name_} at {step_} has already been stored.")
        data_at_step[name_] = val_

    def clean(self, *steps_to_clean):
        for step_ in steps_to_clean:
            self._data.pop(step_, None)


def initialiaze_storage(
    s_: int, init_quantity_: float, storage_: Storage
) -> Storage:
    """Store ``init_quantity_`` as the quantity of step 0 if ``s_`` is 1, and return ``storage_`` itself."""
    if s_ == 1:
        storage_.set(0, "quantity", init_quantity_)
    return storage_


def calculate_quantity(s_: int, f_: float, storage_: Storage):
    """Store and return the quantity of step ``s_``: the quantity of the previous step times ``f_``."""
    quantity_ = storage_.get(s_ - 1, "quantity")
    quantity_ *= f_
    storage_.set(s_, "quantity", quantity_)
    return quantity_


def calculate_output(quantity_):
    return quantity_ + 10.0


specs_ = {
    "vars_": [
        {
            "name": "storage",
            "inputs": {"s": "configs", "init_quantity": "configs", "storage": "buffer"},
            "formula": initialiaze_storage
        },
        {
            "name": "quantity",
            "inputs": {"s": "configs", "f": "configs", "storage": "vars_"},
            "formula": calculate_quantity
        },
        {
            "name": "output",
            "inputs": {"quantity": "vars_"},
            "formula": calculate_output
        }
    ]
}

start_step = 1
steps = 6

configs = {
    "s": start_step,
    "f": 0.5,
    "init_quantity": 512
}


storage = Storage(start_step)
buffer = {"storage": storage}


def test_storage():
    """The formulas of a graph store a quantity in an external Storage object at each step and read it at the next step.

    execute runs step 1. Each later step is a recompute with a new configs.s, in which vars_.quantity multiplies the
    quantity of the previous step by configs.f. After each step, the test deletes the stored steps older than the
    previous one, and at the end it checks the remaining steps and the tree of vars_.output.
    """
    g = Graph(specs_, ["buffer", "configs"])
    compiled = g.compile(["vars_.output"])
    values = Values()
    results = []
    for s_ in range(start_step, steps):
        if s_ == 1:
            compiled.execute({
                "buffer": buffer,
                "configs": configs
            }, values)
        else:
            compiled.recompute({
                "configs": {
                    "s": s_
                }
            }, values)
        results.append({
            "step": s_,
            "output": values.get(g.registry["vars_"]["output"]).value,
        })
        storage.clean(*list(range(s_ - 1)))
    assert results == [
        {"step": 1, "output": 512 / 2 + 10},
        {"step": 2, "output": 512 / 4 + 10},
        {"step": 3, "output": 512 / 8 + 10},
        {"step": 4, "output": 512 / 16 + 10},
        {"step": 5, "output": 512 / 32 + 10}
    ]
    assert storage._data == {4: {"quantity": 512 / 2 ** 4}, 5: {"quantity": 512 / 2 ** 5}}
    vars_output_node = make_node("output", "vars_", g.registry)
    assert make_image(vars_output_node) == """
vars_.output--vars_.quantity--configs.s
                              |
                              configs.f
                              |
                              vars_.storage--configs.s
                                             |
                                             configs.init_quantity
                                             |
                                             buffer.storage"""


def test_recompute_graph_priority_over_co_changed_override():
    """recompute discards a given value of a calculated variable when the same call changes one of its inputs.

    CompiledGraph.recompute then calculates the variable from its formula, so the formulas determine the final values.
    """
    def derive_b(a_):
        return 10 * a_

    def derive_c(b_):
        return b_ + 1

    def _spec(name, inputs, formula):
        return {"name": name, "inputs": inputs, "formula": formula,
                "metadata": dedent(getsource(formula))}

    graph = Graph(
        variables_lists={"vars_": [
            _spec("b", {"a": "ext"}, derive_b),
            _spec("c", {"b": "vars_"}, derive_c),
        ]},
        external_source_names=["ext"],
    )
    compiled = graph.compile(["vars_.c"])
    values = Values()
    b = graph.registry["vars_"]["b"]
    c = graph.registry["vars_"]["c"]

    compiled.execute({"ext": {"a": 1}}, values)
    assert values.get(b).value == 10
    assert values.get(c).value == 11

    # The call sets b to 999 and changes a to 2. b depends on a, so recompute discards 999 and calculates b = 20 and
    # c = 21.
    compiled.recompute({"ext": {"a": 2}, "vars_": {"b": 999}}, values)
    assert values.get(b).value == 20
    assert values.get(c).value == 21


def test_variable_name_rejects_qual_sep_but_source_allows_it():
    """A variable's name must not contain a dot, the qualified-name separator, while a namespace's name may.

    numbox splits a qualified name at its last dot. The split is therefore correct when a namespace's name contains a
    dot, and wrong when a variable's name contains one.
    """
    with pytest.raises(ValueError, match="reserved"):
        Variable(name="a.b", source="s")
    v = Variable(name="b", source="a.b")
    assert v.qual_name() == "a.b.b"


if __name__ == "__main__":
    collect_and_run_tests(__name__)
