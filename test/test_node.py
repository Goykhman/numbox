from numba.core.errors import NumbaError
from numbox.core.node import make_node


def test():
    n1 = make_node("n1")
    n2 = make_node("n2", (n1,))
    n3 = make_node("n3")
    n4 = make_node("n4", (n2, n3))
    assert n4.get_source(0).name == "n2"
    assert n4.get_source(1).name == "n3"
    assert n4.get_source(0).get_source(0).name == "n1"
    try:
        _ = n1.get_source(0)
    except NumbaError as e:
        assert str(e) == "Requested source 0 while the node has 0 sources"

    assert n1.get_sources_names() == []
    assert n4.get_sources_names() == ["n2", "n3"]
