"""Tests of ``Node``, a node of a graph that holds only its name and its inputs.

The tests read a small graph with ``get_input``, ``get_inputs_names`` and ``all_inputs_names``, and check the
``NumbaError`` that ``get_input`` raises for an index out of range. ``test_print_tree`` draws graphs of ``Node``
objects.
"""
import pytest
from numba.core.errors import NumbaError
from numbox.core.work.node import make_node
from test.auxiliary_utils import ansi_escape, collect_and_run_tests


def test():
    """``get_input``, ``get_inputs_names`` and ``all_inputs_names`` read the inputs of a small graph of nodes.

    ``get_input`` returns an input as a ``Node``, and ``get_inputs_names`` lists the direct inputs. ``all_inputs_names``
    lists every node that the node depends on, depth first. If ``get_input`` raises ``NumbaError`` for an index past the
    last input, the test checks its message.
    """
    n1 = make_node("n1")
    n2 = make_node("n2", (n1,))
    n3 = make_node("n3")
    n4 = make_node("n4", (n2, n3))
    assert n4.get_input(0).name == "n2"
    assert n4.get_input(1).name == "n3"
    assert n4.get_input(0).get_input(0).name == "n1"
    try:
        _ = n1.get_input(0)
    except NumbaError as e:
        assert ansi_escape.sub("", str(e)) == "Requested input 0 while the node has 0 inputs"

    assert n1.get_inputs_names() == []
    assert n4.get_inputs_names() == ["n2", "n3"]
    assert n4.all_inputs_names() == ["n2", "n1", "n3"]


def test_get_input_rejects_negative_index():
    """``get_input(-1)`` raises ``NumbaError`` instead of returning the last input, as Python's indexing would."""
    n1 = make_node("n1")
    n2 = make_node("n2", (n1,))
    with pytest.raises(NumbaError):
        n2.get_input(-1)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
