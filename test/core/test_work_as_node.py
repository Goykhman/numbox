from numbox.utils.meminfo import structref_meminfo
from numbox.core.work.node import Node
from numbox.core.work.work_utils import make_work_helper
from test.auxiliary_utils import collect_and_run_tests


def test_as_node():
    w1 = make_work_helper("w1", 0.0)
    w1_as_node_1 = w1.as_node()
    w1_meminfo_p_1, w1_data_p_1 = structref_meminfo(w1_as_node_1)
    assert isinstance(w1_as_node_1, Node)

    w1_as_node_2 = w1.as_node()  # returns cached node
    w1_meminfo_p_2, w1_data_p_2 = structref_meminfo(w1_as_node_2)
    assert isinstance(w1_as_node_2, Node)
    assert w1_meminfo_p_1 == w1_meminfo_p_2
    assert w1_data_p_1 == w1_data_p_2

    w2 = make_work_helper("w2", 0.0)
    w3 = make_work_helper("w3", 0.0, (w2,), lambda w2_: 0.0)
    _ = w3.make_inputs_vector()  # stores w2 as node, re-uses w1's previously created `node`

    w1_as_node_3 = w1.as_node()  # returns cached node
    w1_meminfo_p_3, w1_data_p_3 = structref_meminfo(w1_as_node_3)
    assert isinstance(w1_as_node_3, Node)
    assert w1_meminfo_p_1 == w1_meminfo_p_3
    assert w1_data_p_1 == w1_data_p_3


if __name__ == "__main__":
    collect_and_run_tests(__name__)
