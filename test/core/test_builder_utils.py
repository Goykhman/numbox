from numbox.core.work.builder import End, Derived, make_graph
from numbox.core.work.builder_utils import infer_sources_dependencies
from numbox.core.work.print_tree import make_image
from test.auxiliary_utils import collect_and_run_tests


def test_1():
    n10 = End(name="n10", init_value=0.0)
    f10 = Derived(name="f10", init_value=0.0, sources=(n10,), derive=lambda x: x)
    g1 = make_graph(f10)
    sources_dependencies = infer_sources_dependencies(g1)
    assert sources_dependencies["n10"] == {"f10"}


def test_2():
    n1 = End(name="n1", init_value=0.0)
    n2 = End(name="n2", init_value=0.0)
    f1 = Derived(name="f1", init_value=0.0, sources=(n1, n2), derive=lambda x, y: x + y)
    f2 = Derived(name="f2", init_value=0.0, sources=(f1,), derive=lambda x: 2 * x)
    f3 = Derived(name="f3", init_value=0.0, sources=(f2,), derive=lambda x: 2 * x)
    f4 = Derived(name="f4", init_value=0.0, sources=(f3,), derive=lambda x: 2 * x)
    g1 = make_graph(f4, f3)
    sources_dependencies = infer_sources_dependencies(g1)
    assert sources_dependencies["n1"] == {"f1", "f2", "f3", "f4"}
    assert sources_dependencies["n2"] == {"f1", "f2", "f3", "f4"}
    assert sources_dependencies["f1"] == {"f2", "f3", "f4"}
    assert sources_dependencies["f2"] == {"f3", "f4"}
    assert sources_dependencies["f3"] == {"f4"}
    assert sources_dependencies["f4"] == set()


def test_3():
    m5 = End(name="m5", init_value=0.0)
    m3 = Derived(name="m3", init_value=0.0, sources=(m5,), derive=lambda x: x)
    m2 = Derived(name="m2", init_value=0.0, sources=(m3,), derive=lambda x: x)
    m4 = End(name="m4", init_value=0.0)
    m1 = Derived(name="m1", init_value=0.0, sources=(m2, m4), derive=lambda x: 2 * x)
    sources_dependencies = infer_sources_dependencies((m1,))
    assert sources_dependencies == {
        "m1": set(),
        "m2": {"m1"},
        "m3": {"m1", "m2"},
        "m4": {"m1"},
        "m5": {"m1", "m2", "m3"}
    }


def test_4():
    reg = {}
    a = End(name="a", init_value=0.0, registry=reg)
    b = End(name="b", init_value=0.0, registry=reg)
    d = Derived(name="d", init_value=0.0, sources=(b,), derive=lambda x: x, registry=reg)
    e = Derived(name="e", init_value=0.0, sources=(a, b), derive=lambda x, y: x + y, registry=reg)
    c = Derived(name="c", init_value=0.0, sources=(a, b), derive=lambda x, y: x + y, registry=reg)
    f = Derived(name="f", init_value=0.0, sources=(c, d, e), derive=lambda x, y, z: x + y + z, registry=reg)
    g = make_graph(f, registry=reg)
    f_ = g.f
    assert make_image(f_) == """
f--c--a
   |  |
   |  b
   |
   d--b
   |
   e--a
      |
      b"""
    sources_dependencies = infer_sources_dependencies(g)
    assert sources_dependencies == {
        "f": set(),
        "c": {"f"},
        "a": {"e", "f", "c"},
        "b": {"f", "e", "d", "c"},
        "d": {"f"},
        "e": {"f"}
    }


if __name__ == "__main__":
    collect_and_run_tests(__name__)
