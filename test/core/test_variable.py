from numbox.core.variable.variable import CompiledGraph, Graph, Values, Variables, Variable
from test.auxiliary_utils import collect_and_run_tests


def test_basic():
    x = {"name": "x", "inputs": {"y": "basket"}, "formula": lambda y_: 2 * y_, "cacheable": True}
    a = {"name": "a", "inputs": {"x": "variables1"}, "formula": lambda x_: x_ - 74, "cacheable": True}
    u = {"name": "u", "inputs": {"a": "variables1"}, "formula": lambda a_: a_ * 2, "cacheable": True}

    graph = Graph(
        variables_lists={
            "variables1": [x, a],
            "variables2": [u],
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

    assert list(variables1.variables.keys()) == ["x", "a"]
    assert list(variables2.variables.keys()) == ["u"]

    assert isinstance(variables1.variables["x"], Variable)

    required_external_variables = compiled.required_external_variables

    assert list(required_external_variables.keys()) == ["basket"]
    basket = required_external_variables["basket"]
    assert list(basket.keys()) == ["y"]
    assert basket["y"].name == "y"

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


if __name__ == "__main__":
    collect_and_run_tests(__name__)
