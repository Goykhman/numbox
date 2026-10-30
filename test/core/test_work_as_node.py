"""Tests of ``Work.as_node``, which returns a node's ``Node`` representation.

The first call creates the representation and stores it in the node, and later calls return the stored ``Node``.
"""
from numbox.utils.meminfo import structref_meminfo
from numbox.core.work.node import Node
from numbox.core.work.work_utils import make_work_helper
from test.auxiliary_utils import collect_and_run_tests


def test_as_node():
    """Every call of ``as_node`` on a node returns the ``Node`` that the first call created and stored.

    The test compares the addresses of the MemInfo and of the payload of the returned objects.
    """
    w1 = make_work_helper("w1", 0.0)
    w1_as_node_1 = w1.as_node()
    w1_meminfo_p_1, w1_data_p_1 = structref_meminfo(w1_as_node_1)
    assert isinstance(w1_as_node_1, Node)

    # The second call returns the Node that the first call stored.
    w1_as_node_2 = w1.as_node()
    w1_meminfo_p_2, w1_data_p_2 = structref_meminfo(w1_as_node_2)
    assert isinstance(w1_as_node_2, Node)
    assert w1_meminfo_p_1 == w1_meminfo_p_2
    assert w1_data_p_1 == w1_data_p_2

    w2 = make_work_helper("w2", 0.0)
    w3 = make_work_helper("w3", 0.0, (w2,), lambda w2_: 0.0)
    # make_inputs_vector creates the Node of w2, the only source of w3, and stores it in w2. w1 is not in the graph of
    # w3, so its stored Node is unchanged.
    _ = w3.make_inputs_vector()

    # The third call also returns the Node that the first call stored.
    w1_as_node_3 = w1.as_node()
    w1_meminfo_p_3, w1_data_p_3 = structref_meminfo(w1_as_node_3)
    assert isinstance(w1_as_node_3, Node)
    assert w1_meminfo_p_1 == w1_meminfo_p_3
    assert w1_data_p_1 == w1_data_p_3


if __name__ == "__main__":
    collect_and_run_tests(__name__)
