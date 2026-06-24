from inspect import getsource
from textwrap import dedent
from typing import Any

import pytest

from numbox.core.variable.variable import CompiledGraph, Graph, Value, Values, Variables, Variable
from numbox.core.variable.node import make_node
from numbox.core.work.print_tree import make_image

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
    """ External storage for graph calculation outputs, serving
     as an intra-step link for multistep calculations.
     This storage will own all the calculation outputs, which
     can in turn be referenced by values on the graph.
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
    """ Anything that needs to be initialized.
     Returns reference to the original instance of
     `Storage`, modified in-place. """
    if s_ == 1:
        storage_.set(0, "quantity", init_quantity_)
    return storage_


def calculate_quantity(s_: int, f_: float, storage_: Storage):
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


def _uaf_graph():
    # N = f(X, M); X = g(a); M = h(b); only N requested.
    def derive_X(a_):
        return a_ + 100

    def derive_M(b_):
        return b_ + 200

    def derive_N(X_, M_):
        return X_ + M_

    all_vars = {
        "X": {"name": "X", "inputs": {"a": "ext"}, "formula": derive_X},
        "M": {"name": "M", "inputs": {"b": "ext"}, "formula": derive_M},
        "N": {"name": "N", "inputs": {"X": "vars", "M": "vars"}, "formula": derive_N},
    }
    graph = Graph(
        variables_lists={"vars": [all_vars["X"], all_vars["M"], all_vars["N"]]},
        external_source_names=["ext"],
    )
    compiled = graph.compile(["vars.N"])
    return graph, compiled


def test_values_pop_is_strict_on_missing_key():
    values = Values()
    absent = Variable(name="absent", source="vars")
    with pytest.raises(KeyError):
        values.pop(absent)


def test_execute_clean_storage_frees_non_requested_intermediates():
    graph, compiled = _uaf_graph()
    values = Values()
    n_var = graph.registry["vars"]["N"]
    x_var = graph.registry["vars"]["X"]
    m_var = graph.registry["vars"]["M"]
    compiled.execute(external_values={"ext": {"a": 1, "b": 2}}, values=values, clean_storage=True)
    assert values.get(n_var).value == 303          # requested output computed
    assert n_var in values._values                 # requested output retained
    assert x_var not in values._values             # non-requested intermediates freed at last use
    assert m_var not in values._values


def test_execute_clean_storage_marks_storage_terminal():
    graph, compiled = _uaf_graph()
    values = Values()
    compiled.execute(external_values={"ext": {"a": 1, "b": 2}}, values=values, clean_storage=True)


def test_execute_clean_storage_is_rerunnable():
    graph, compiled = _uaf_graph()
    values = Values()
    n_var = graph.registry["vars"]["N"]
    compiled.execute(external_values={"ext": {"a": 1, "b": 2}}, values=values, clean_storage=True)
    assert values.get(n_var).value == 303
    # re-running execute (the "one more go" path) re-derives everything from the externals
    compiled.execute(external_values={"ext": {"a": 10, "b": 2}}, values=values, clean_storage=True)
    assert values.get(n_var).value == 312          # X=110, M=202


def test_execute_without_clean_storage_retains_all_values():
    graph, compiled = _uaf_graph()
    values = Values()
    n_var = graph.registry["vars"]["N"]
    x_var = graph.registry["vars"]["X"]
    m_var = graph.registry["vars"]["M"]
    compiled.execute(external_values={"ext": {"a": 1, "b": 2}}, values=values)
    assert {x_var, m_var, n_var} <= set(values._values)   # default execute keeps every value


class _Storage:
    """A minimal Storage conforming to get/pop/__iter__"""
    def __init__(self):
        self._values = {}

    def get(self, variable):
        if variable not in self._values:
            self._values[variable] = Value(variable=variable)
        return self._values[variable]

    def pop(self, variable):
        self._values.pop(variable)

    def __iter__(self):
        return iter(self._values.keys())


def test_execute_without_clean_storage_works_on_minimal_storage():
    graph, compiled = _uaf_graph()
    n_var = graph.registry["vars"]["N"]
    storage = _Storage()
    compiled.execute(external_values={"ext": {"a": 1, "b": 2}}, values=storage)
    assert storage.get(n_var).value == 303


class _RecordingValues(Values):
    """Values that tracks the peak number of simultaneously-resident entries."""
    def __init__(self):
        super().__init__()
        self.peak = 0

    def get(self, variable):
        value = super().get(variable)
        self.peak = max(self.peak, len(self._values))
        return value


def _chain_graph(length=4):
    # ext.a -> v0 -> v1 -> ... -> v{length-1}; only the last is requested
    def make_f(k):
        def f(x):
            return x + k
        return f
    specs = []
    prev_name, prev_src = "a", "ext"
    for i in range(length):
        specs.append({"name": f"v{i}", "inputs": {prev_name: prev_src}, "formula": make_f(i + 1)})
        prev_name, prev_src = f"v{i}", "vars"
    graph = Graph(variables_lists={"vars": specs}, external_source_names=["ext"])
    compiled = graph.compile([f"vars.v{length - 1}"])
    return graph, compiled


def _diamond_graph():
    # ext.c feeds both L and R; both feed requested D (c has two consumers)
    def derive_L(c_):
        return c_ + 10

    def derive_R(c_):
        return c_ + 20

    def derive_D(L_, R_):
        return L_ + R_
    specs = [
        {"name": "L", "inputs": {"c": "ext"}, "formula": derive_L},
        {"name": "R", "inputs": {"c": "ext"}, "formula": derive_R},
        {"name": "D", "inputs": {"L": "vars", "R": "vars"}, "formula": derive_D},
    ]
    graph = Graph(variables_lists={"vars": specs}, external_source_names=["ext"])
    compiled = graph.compile(["vars.D"])
    return graph, compiled


def test_execute_clean_storage_caps_peak_residency():
    graph, compiled = _chain_graph(length=4)
    out_var = graph.registry["vars"]["v3"]

    clean = _RecordingValues()
    compiled.execute(external_values={"ext": {"a": 10}}, values=clean, clean_storage=True)
    assert clean.get(out_var).value == 20          # v0=11, v1=13, v2=16, v3=20

    full = _RecordingValues()
    compiled.execute(external_values={"ext": {"a": 10}}, values=full)
    # the headline property: mid-pass eviction caps peak well below retaining everything
    assert clean.peak <= 2
    assert clean.peak < full.peak


def test_execute_clean_storage_diamond_frees_at_last_consumer():
    graph, compiled = _diamond_graph()
    d_var = graph.registry["vars"]["D"]
    values = Values()
    # c has two consumers; were last_used to pick the earlier one, the later consumer would
    # read a freed c and raise "Uninitialized input" -- a clean run succeeding proves eviction
    # waits for the LAST consumer.
    compiled.execute(external_values={"ext": {"c": 5}}, values=values, clean_storage=True)
    assert values.get(d_var).value == 40           # L=15, R=25


def test_execute_clean_storage_all_requested_frees_only_externals():
    def derive_X(a_):
        return a_ + 100

    def derive_M(b_):
        return b_ + 200

    def derive_N(X_, M_):
        return X_ + M_
    specs = [
        {"name": "X", "inputs": {"a": "ext"}, "formula": derive_X},
        {"name": "M", "inputs": {"b": "ext"}, "formula": derive_M},
        {"name": "N", "inputs": {"X": "vars", "M": "vars"}, "formula": derive_N},
    ]
    graph = Graph(variables_lists={"vars": specs}, external_source_names=["ext"])
    compiled = graph.compile(["vars.X", "vars.M", "vars.N"])
    values = Values()
    compiled.execute(external_values={"ext": {"a": 1, "b": 2}}, values=values, clean_storage=True)
    x_var = graph.registry["vars"]["X"]
    m_var = graph.registry["vars"]["M"]
    n_var = graph.registry["vars"]["N"]
    a_ext = compiled.required_external_variables["ext"]["a"]
    b_ext = compiled.required_external_variables["ext"]["b"]
    assert {x_var, m_var, n_var} <= set(values._values)   # all requested -> no intermediate freed


if __name__ == "__main__":
    collect_and_run_tests(__name__)
