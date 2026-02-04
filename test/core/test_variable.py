from textwrap import dedent

from inspect import getsource
from numbox.core.variable.variable import CompiledGraph, Graph, Values, Variables, Variable
from test.auxiliary_utils import collect_and_run_tests


def test_basic():

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


if __name__ == "__main__":
    collect_and_run_tests(__name__)
