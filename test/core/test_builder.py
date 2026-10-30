"""Tests of the builder: the specs ``End`` and ``Derived``, ``make_graph``, and the cache of the graph maker.

The first tests build graphs from specs, among them the graph of the builder overview on the work page. They check the
values, the drawings of ``make_image``, the text of ``explain``, ``combine``, ``load``, local registries and the refusal
of a repeated name. The later tests check the name under which numba caches the graph maker. The name changes with a
node's declared type and with the jit flags. It stays the same for a new ``init_value`` of the same type, and it is the
same in every process for a graph with a record type or with a derive that reads an attribute named like a module
global. A derive defined by ``exec`` compiles through an anchor file of its own. A derive whose anchor cannot be built
compiles without a cache, and the maker compiles without a cache when a jit flag has no canonical form. The tests that
count cache files run a script in fresh processes against one cache directory under ``tmp_path``.
``test_cache_crossprocess`` covers more cases of the maker's cache across processes, ``test_builder_utils`` covers
``infer_sources_dependencies``, and ``test_derive_wap`` covers the exceptions of derives.
"""
import os
import subprocess
import sys
import textwrap

import numpy
import pytest

from numba import from_dtype
from numba.core.types import Array, float32, float64, int16, int32, int64, unicode_type
from numba.typed.typeddict import Dict
from numpy import isclose

from numbox.core.any.any_type import AnyType, make_any
from numbox.core.work.work import Work
from numbox.core.work.builder import Derived, End, make_graph
from numbox.core.work.combine_utils import make_sheaf_dict
from numbox.core.work.explain import explain
from numbox.core.work.print_tree import make_image
from test.auxiliary_utils import collect_and_run_tests


w1_ = End(name="w1", init_value=137, ty=int16)
w2_ = End(name="w2", init_value=3.14)
w5_ = End(name="w5", init_value=10)
w6_ = End(name="w6", init_value=7.5)


def derive_w3(w1_, w2_):
    if w1_ < 0:
        return 0.0
    elif w1_ < 1:
        return 2 * w2_
    return 3 * w2_


def derive_w4(w1_):
    return 2 * w1_


def derive_w7(w3_, w5_):
    return w3_ + (w5_ ** 2)


def derive_w8(w6_, w2_):
    if w6_ > 5:
        return w6_ * w2_
    else:
        return w6_ + w2_


def derive_w9(w3_, w4_, w7_):
    return (w4_ - w3_) / (abs(w7_) + 1e-5)


def derive_w10(w3_, w4_, w7_, w8_, w9_):
    return (w3_ + w4_ + w7_) * 0.1 + (w8_ - w9_)


w3_ = Derived(name="w3", init_value=0.0, derive=derive_w3, sources=(w1_, w2_))
w4_ = Derived(name="w4", init_value=0.0, derive=derive_w4, sources=(w1_,))
w7_ = Derived(name="w7", init_value=0.0, derive=derive_w7, sources=(w3_, w5_))
w8_ = Derived(name="w8", init_value=0.0, derive=derive_w8, sources=(w6_, w2_))
w9_ = Derived(name="w9", init_value=0.0, derive=derive_w9, sources=(w3_, w4_, w7_))
w10_ = Derived(name="w10", init_value=0.0, derive=derive_w10, sources=(w3_, w4_, w7_, w8_, w9_))


def test_1():
    """``make_graph`` returns the access node ``w3`` as a ``Work`` node, and ``calculate`` computes its value.

    The value is 0 until ``calculate`` runs, and ``make_image`` draws ``w3`` with its sources ``w1`` and ``w2``.
    """
    access = make_graph(w3_)
    w3 = access.w3
    assert isinstance(w3, Work)

    assert w3.data == 0
    w3.calculate()
    assert isclose(w3.data, 9.42)
    assert make_image(w3) == """
w3--w1
    |
    w2"""


def test_2():
    """One ``make_graph`` call returns the access nodes ``w3`` and ``w4``, which share the source ``w1``.

    Each node calculates its own value, and ``make_image`` draws ``w4`` with its one source.
    """
    access = make_graph(w3_, w4_)
    w3 = access.w3
    w4 = access.w4
    assert w3.data == 0
    w3.calculate()
    assert isclose(w3.data, 9.42)
    assert w4.data == 0
    w4.calculate()
    assert isclose(w4.data, 274)
    assert make_image(w4) == """
w4--w1"""


def test_3():
    """The graph of the builder overview on the work page gives the values and the ``explain`` text that the page shows.

    The test also checks the drawings of ``w9`` and ``w10``, lists the inputs of ``w7``, and reads the fourth input of
    ``w10`` with ``get_input``. Through ``w10``, it collects values with ``combine`` and loads a new value of ``w1``
    with ``load``. The next ``calculate`` computes ``w4`` from the new value.
    """
    access_ = make_graph(w7_, w9_, w10_)
    w7 = access_.w7
    w9 = access_.w9
    w10 = access_.w10
    assert make_image(access_.w9) == """
w9--w3--w1
    |   |
    |   w2
    |
    w4--w1
    |
    w7--w3--w1
        |   |
        |   w2
        |
        w5"""
    assert make_image(access_.w10) == """
w10--w3--w1
     |   |
     |   w2
     |
     w4--w1
     |
     w7--w3--w1
     |   |   |
     |   |   w2
     |   |
     |   w5
     |
     w8--w6
     |   |
     |   w2
     |
     w9--w3--w1
         |   |
         |   w2
         |
         w4--w1
         |
         w7--w3--w1
             |   |
             |   w2
             |
             w5"""
    assert w10.data == 0
    w10.calculate()
    assert isclose(w7.data, 109.42)
    assert isclose(w9.data, 2.418022)
    assert isclose(w10.data, 60.416)
    assert w7.all_inputs_names() == ["w3", "w1", "w2", "w5"]
    w8_r = w10.get_input(3)
    assert w8_r.name == "w8"

    requested = ("w1", "w4", "w7", "w8")
    sheaf = make_sheaf_dict(requested)
    w10.combine(sheaf)
    assert isclose(sheaf["w4"].get_as(float64), 274)
    assert isclose(sheaf["w7"].get_as(float64), 109.42)
    assert isclose(sheaf["w8"].get_as(float64), 23.55)

    load_data = Dict.empty(key_type=unicode_type, value_type=AnyType)
    # make_any(12) holds an int64, while w1 is an int16 node. load reads the value as an int16 without a conversion,
    # so w1 receives the two low-order bytes of the int64, which hold 12 on a little-endian machine.
    load_data["w1"] = make_any(12)
    assert sheaf["w1"].get_as(int16) == 137
    w10.load(load_data)
    w10.combine(sheaf)
    assert sheaf["w1"].get_as(int16) == 12
    w10.calculate()
    w10.combine(sheaf)
    assert isclose(sheaf["w4"].get_as(float64), 24)

    derivation_of_w9 = explain(w9)
    assert derivation_of_w9 == """All required end nodes: ['w1', 'w2', 'w5']

w1: end node

w2: end node

w3: derive_w3(w1, w2)

    def derive_w3(w1_, w2_):
        if w1_ < 0:
            return 0.0
        elif w1_ < 1:
            return 2 * w2_
        return 3 * w2_

w4: derive_w4(w1)

    def derive_w4(w1_):
        return 2 * w1_

w5: end node

w7: derive_w7(w3, w5)

    def derive_w7(w3_, w5_):
        return w3_ + (w5_ ** 2)

w9: derive_w9(w3, w4, w7)

    def derive_w9(w3_, w4_, w7_):
        return (w4_ - w3_) / (abs(w7_) + 1e-5)
"""


T = 10
tau = End(name="tau", init_value=numpy.arange(T), ty=Array(int64, 1, "C", readonly=False, aligned=True))
e1 = End(name="e1", init_value=1.4142135623730951)
e2 = End(name="e2", init_value=1.7320508075688772)
a_ty = numpy.dtype([("c1", numpy.float64), ("c2", numpy.float64)])
a = End(name="a", init_value=numpy.empty((T,), a_ty), ty=Array(from_dtype(a_ty), 1, "C", readonly=False, aligned=True))


def derive_c1(tau_, a_, e1_):
    for t in tau_:
        a_[t]["c1"] = e1_ ** 2 + 2 * (t + 1)
    return 0


def derive_c2(tau_, a_, e2_, c1_):
    for t in tau_:
        a_[t]["c2"] = 3 * e2_ ** 2 - 4 * (t + 1) + a_[t].c1
    return 0


c1 = Derived(name="c1", init_value=1, derive=derive_c1, sources=(tau, a, e1))
c2 = Derived(name="c2", init_value=1, derive=derive_c2, sources=(tau, a, e2, c1))


def test_4():
    """Derives can fill an array that an end node holds, in place.

    ``c1`` and ``c2`` write the two fields of the record array ``a`` and return 0 as their own value. ``c2`` reads the
    field that ``c1`` wrote, because ``c1`` is one of its sources and ``calculate`` computes it first.
    """
    access_ = make_graph(c1, c2, a)
    c1_ = access_.c1
    c2_ = access_.c2
    a_ = access_.a
    assert c1_.data == 1
    c1_.calculate()
    assert c1_.data == 0
    assert isclose(a_.data[5]["c1"], 14)
    assert c2_.data == 1
    c2_.calculate()
    assert c2_.data == 0
    assert isclose(a_.data[6]["c2"], -19 + a_.data[6]["c1"])
    assert make_image(c2_) == """
c2--tau
    |
    a
    |
    e2
    |
    c1---tau
         |
         a
         |
         e1"""


def test_5():
    """An end node's ``init_value`` can be a ``StructRef``, and the node's data then has the struct's fields."""
    from test.common_structrefs import S1
    e1 = End(name="test_5_e1", init_value=S1(141, 137, 3.14))
    access_ = make_graph(e1)
    e1_ = getattr(access_, "test_5_e1")
    assert e1_.data.x1 == 141
    assert e1_.data.x2 == 137
    assert isclose(e1_.data.x3, 3.14)


x_ = End(name="x", init_value=numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32))
y_ = End(name="y", init_value=numpy.array([0.5, 0.25, 0.75], dtype=numpy.float32))
threshold_ = End(name="threshold", init_value=1.5, ty=float32)
alpha_ = End(name="alpha", init_value=0.1, ty=float32)
beta_ = End(name="beta", init_value=0.01, ty=float32)


def derive_mask(x_, threshold_):
    return x_ > threshold_


mask_ = Derived(
    name="mask",
    init_value=numpy.full_like(x_.init_value, False, dtype=bool),
    derive=derive_mask,
    sources=(x_, threshold_),
)


def derive_scaled_y(y_, mask_, alpha_):
    return numpy.where(mask_, y_ * alpha_, y_)


scaled_y_ = Derived(
    name="scaled_y",
    init_value=numpy.zeros_like(y_.init_value, dtype=numpy.float32),
    derive=derive_scaled_y,
    sources=(y_, mask_, alpha_),
)


def derive_weighted_sum(x_, scaled_y_, beta_):
    return numpy.sum(x_ * scaled_y_ + beta_)


weighted_sum_ = Derived(
    name="weighted_sum",
    init_value=float32(0.0),
    derive=derive_weighted_sum,
    sources=(x_, scaled_y_, beta_),
)


def derive_running_avg(x_, y_):
    avg = numpy.zeros_like(x_, dtype=numpy.float32)
    for i in range(len(x_)):
        avg[i] = numpy.mean(x_[:i+1] + y_[:i+1])
    return avg


running_avg_ = Derived(
    name="running_avg",
    init_value=numpy.zeros_like(x_.init_value, dtype=numpy.float32),
    derive=derive_running_avg,
    sources=(x_, y_),
)


def derive_interaction(running_avg_, scaled_y_):
    return numpy.tanh(running_avg_ * scaled_y_)


interaction_ = Derived(
    name="interaction",
    init_value=numpy.zeros_like(x_.init_value, dtype=numpy.float32),
    derive=derive_interaction,
    sources=(running_avg_, scaled_y_),
)


def derive_output(interaction_, weighted_sum_):
    return numpy.mean(interaction_) + weighted_sum_


output_ = Derived(
    name="output",
    init_value=float32(0.0),
    derive=derive_output,
    sources=(interaction_, weighted_sum_),
)


def test_6():
    """A graph of arrays and float32 scalars, whose derives call NumPy functions, calculates the expected value.

    ``all_end_nodes`` lists the end nodes in the order in which a depth-first walk from ``output`` reaches them, and
    ``explain`` lists each node after its sources. The test also checks the drawing of the graph.
    """
    output = make_graph(output_).output
    assert output.data == 0
    assert output.all_end_nodes() == ["x", "y", "threshold", "alpha", "beta"]
    derivation_of_output = explain(output)
    assert derivation_of_output == """All required end nodes: ['x', 'y', 'threshold', 'alpha', 'beta']

x: end node

y: end node

running_avg: derive_running_avg(x, y)

    def derive_running_avg(x_, y_):
        avg = numpy.zeros_like(x_, dtype=numpy.float32)
        for i in range(len(x_)):
            avg[i] = numpy.mean(x_[:i+1] + y_[:i+1])
        return avg

threshold: end node

mask: derive_mask(x, threshold)

    def derive_mask(x_, threshold_):
        return x_ > threshold_

alpha: end node

scaled_y: derive_scaled_y(y, mask, alpha)

    def derive_scaled_y(y_, mask_, alpha_):
        return numpy.where(mask_, y_ * alpha_, y_)

interaction: derive_interaction(running_avg, scaled_y)

    def derive_interaction(running_avg_, scaled_y_):
        return numpy.tanh(running_avg_ * scaled_y_)

beta: end node

weighted_sum: derive_weighted_sum(x, scaled_y, beta)

    def derive_weighted_sum(x_, scaled_y_, beta_):
        return numpy.sum(x_ * scaled_y_ + beta_)

output: derive_output(interaction, weighted_sum)

    def derive_output(interaction_, weighted_sum_):
        return numpy.mean(interaction_) + weighted_sum_
"""
    output.calculate()
    assert isclose(output.data, 1.09410762)
    assert make_image(output) == """
output--interaction---running_avg--x
        |             |            |
        |             |            y
        |             |
        |             scaled_y-----y
        |                          |
        |                          mask---x
        |                          |      |
        |                          |      threshold
        |                          |
        |                          alpha
        |
        weighted_sum--x
                      |
                      scaled_y-----y
                      |            |
                      |            mask---x
                      |            |      |
                      |            |      threshold
                      |            |
                      |            alpha
                      |
                      beta"""


def test_7():
    """A second ``End`` spec with a name that the default registry already holds raises ``ValueError``."""
    node1 = End(name="node1", init_value=3.14)  # noqa: F841
    with pytest.raises(ValueError) as e:
        node1_double = End(name="node1", init_value=3.14)  # noqa: F841
        assert str(e) == "Node 'node1' has already been defined on this graph. Pick a different name."


def test_8():
    """Two local registries can hold specs of the same names, and ``make_graph`` builds each graph from its registry.

    Each registry holds its specs in the order of their creation, and the default registry holds none of them.
    """
    reg_1 = {}
    end_1 = End(name="end_1", init_value=0.0, registry=reg_1)
    assert reg_1["end_1"] == end_1
    reg_2 = {}
    end_1_another = End(name="end_1", init_value=0.0, registry=reg_2)
    assert reg_2["end_1"] == end_1_another
    der_1 = Derived(name="der_1", init_value=0.0, sources=(end_1,), registry=reg_1, derive=lambda x: x + 2.17)
    accessors_1 = make_graph(der_1, registry=reg_1)
    der_1_ = accessors_1.der_1
    der_1_.calculate()
    assert isclose(der_1_.data, 2.17)
    der_1_another = Derived(
        name="der_1", init_value=0.0, sources=(end_1_another,), registry=reg_2, derive=lambda x: x + 3.14
    )
    accessors_2 = make_graph(der_1_another, registry=reg_2)
    der_1_another_ = accessors_2.der_1
    der_1_another_.calculate()
    assert isclose(der_1_another_.data, 3.14)
    assert list(reg_1.keys()) == ["end_1", "der_1"]
    assert list(reg_2.keys()) == ["end_1", "der_1"]
    assert reg_1["der_1"] == der_1
    assert reg_2["der_1"] == der_1_another
    from numbox.core.work.builder import _specs_registry
    assert not (_specs_registry.get("end_1") or _specs_registry.get("der_1"))


def test_9():
    """A derived node can have no sources, and its derive then takes no arguments."""

    def calc1():
        return 3.14

    d1_ = Derived(name="d1", init_value=0.0, derive=calc1)
    access = make_graph(d1_)
    d1 = access.d1
    assert d1.data == 0.0
    d1.calculate()
    assert isclose(d1.data, 3.14)


_GRAPH_DRIVER = textwrap.dedent('''
    import os
    import numpy as np
    from numbox.core.work.builder import End, make_graph
    v = np.full((4,), float(os.environ["PROBE_VAL"]), np.float64)
    make_graph(End(name="probe_a", init_value=v))
    print("OK")
''')


def _run_graph_driver(tmp_path, cache, probe_val):
    script = tmp_path / "graph_drv.py"
    script.write_text(_GRAPH_DRIVER)
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache), PROBE_VAL=probe_val)
    out = subprocess.run([sys.executable, str(script)], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr


def test_make_graph_cache_key_content_independent(tmp_path):
    """A new value of the same type for an ``init_value`` reuses the cached graph maker.

    The maker receives the initial values as arguments, so only their numba types are part of the maker's name. The
    first process compiles and caches the maker. The second builds the graph with another value of the same type and
    writes no new maker cache file.
    """
    cache = tmp_path / "nbcache"
    cache.mkdir()
    _run_graph_driver(tmp_path, cache, "1.0")
    n_cold = sum(1 for _ in cache.rglob("builder._make*.nbc"))
    assert n_cold > 0
    _run_graph_driver(tmp_path, cache, "2.0")
    n_warm = sum(1 for _ in cache.rglob("builder._make*.nbc"))
    assert n_warm == n_cold, (
        "make_graph wrote a new _make .nbc for a same-typed init with different "
        "contents (cache key depends on init values, not just types)")


_ATTR_COLLIDE_DRIVER = textwrap.dedent('''
    import numpy as np
    from numbox.core.work.builder import End, Derived, make_graph

    # A module global whose name collides with the array attribute the derive reads (``a_.size``),
    # and which has no canonical fingerprint. Folding globals off co_names would fold -- and choke on --
    # this, forcing the address-bearing derive fallback and a fresh kernel every process.
    size = np.array([object()], dtype=object)


    def derive_s(a_):
        return a_.size * 1.0


    a = End(name="a", init_value=np.zeros(4, dtype=np.float64))
    s = Derived(name="s", init_value=0.0, sources=(a,), derive=derive_s)
    acc = make_graph(s)
    acc.s.calculate()
    print("RESULT", acc.s.data)
''')


def test_attribute_named_global_does_not_grow_the_cache(tmp_path):
    """The graph maker is cached once when a module global has the name of an attribute that the derive reads.

    The derive reads ``a_.size``, and the module global ``size`` holds an object array, which has no fingerprint.
    numbox finds the globals that a derive reads in its instructions rather than in ``co_names``, which also holds
    attribute names, so numbox does not fingerprint the global. The node holds a plain float64 array rather than a
    record array, so the result does not depend on how numba names a record type, which the next test covers. Three
    fresh processes run against one cache directory. The first caches the maker, and the other two add no maker.
    """
    cache = tmp_path / "nbcache"
    cache.mkdir()
    script = tmp_path / "attr_collide_drv.py"
    script.write_text(_ATTR_COLLIDE_DRIVER)
    counts = []
    for _ in range(3):
        out = subprocess.run([sys.executable, str(script)],
                             env=dict(os.environ, NUMBA_CACHE_DIR=str(cache)),
                             capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, out.stderr
        counts.append(sum(1 for _ in cache.rglob("builder._make*.nbc")))
    assert counts[0] > 0, "the graph kernel was never cached"
    assert counts[1] == counts[0] and counts[2] == counts[0], (
        f"the kernel was reminted across processes (cache grows): {counts}")


_RECORD_GRAPH_DRIVER = textwrap.dedent('''
    import numpy as np
    from numba import from_dtype
    from numba.core.types import Array
    from numbox.core.work.builder import End, Derived, make_graph

    dt = np.dtype([("c1", np.float64), ("c2", np.float64)])
    T = 4
    tau = End(name="tau", init_value=np.arange(T))
    a = End(name="a", init_value=np.zeros(T, dtype=dt), ty=Array(from_dtype(dt), 1, "C"))


    def derive_c1(tau_, a_):
        for t in tau_:
            a_[t]["c1"] = t * 1.0
        return 0


    c1 = Derived(name="c1", init_value=0, sources=(tau, a), derive=derive_c1)
    acc = make_graph(c1, a)
    acc.c1.calculate()
    print("RESULT", acc.a.data[2]["c1"])
''')


def test_record_typed_graph_kernel_is_process_stable(tmp_path):
    """A graph with a node of a record array type gives the graph maker the same name in every process.

    numba names a ``Record`` type with a counter of the order in which the process creates types. The counter differs
    between a process that compiles the graph, and so creates many types first, and a process that loads the graph from
    the cache. numbox therefore hashes the record's ``str``, which describes its fields. Three fresh processes run
    against one cache directory. The first caches the maker, and the other two add no maker.
    """
    cache = tmp_path / "nbcache"
    cache.mkdir()
    script = tmp_path / "record_graph_drv.py"
    script.write_text(_RECORD_GRAPH_DRIVER)
    counts = []
    for _ in range(3):
        out = subprocess.run([sys.executable, str(script)],
                             env=dict(os.environ, NUMBA_CACHE_DIR=str(cache)),
                             capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, out.stderr
        counts.append(sum(1 for _ in cache.rglob("builder._make*.nbc")))
    assert counts[0] > 0, "the graph kernel was never cached"
    assert counts[1] == counts[0] and counts[2] == counts[0], (
        f"a record-typed graph kernel was reminted across processes (cache grows): {counts}")


_TYPED_GRAPH_DRIVER = textwrap.dedent('''
    import sys
    from numbox.core.work.builder import End, Derived, make_graph

    def inc(x):
        return x + 1

    t = sys.argv[1]
    iv_e, iv_d = (2.0, 0.0) if t == "f" else (2, 0)
    e = End(name="e", init_value=iv_e)
    d = Derived(name="d", init_value=iv_d, sources=(e,), derive=inc)
    acc = make_graph(d)
    acc.d.calculate()
    print("RESULT", t, acc.d.data, type(acc.d.data).__name__)
''')


def _run_typed_driver(tmp_path, cache, t):
    script = tmp_path / ("typed_drv_%s.py" % t)
    script.write_text(_TYPED_GRAPH_DRIVER)
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache))
    out = subprocess.run([sys.executable, str(script), t], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    return [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")][0].split()


def test_make_graph_distinct_types_dispatch(tmp_path):
    """Two processes that share a cache directory build the same graph, first with float64 and then with int64 nodes.

    Each process gets a result of its own type. The node types are part of the hash that names the graph maker, so the
    int64 graph has a maker of its own and does not load the maker cached for the float64 graph.
    """
    cache = tmp_path / "nbcache_types"
    cache.mkdir()
    rf = _run_typed_driver(tmp_path, cache, "f")
    ri = _run_typed_driver(tmp_path, cache, "i")
    assert rf == ["RESULT", "f", "3.0", "float"]
    assert ri == ["RESULT", "i", "3", "int"]


def test_make_graph_kernel_name_depends_on_declared_type(monkeypatch):
    """Two graphs that differ only in the declared ``ty`` of an ``End`` get different graph maker names.

    The maker's code names the declared type only through a global, so ``_kernel_fingerprint`` hashes each node's
    type. If the two makers had one name, numba could load the maker cached for one graph when the other is built, and
    that maker would create nodes of the wrong data type.
    """
    import numbox.core.work.builder as builder_mod
    captured = []
    orig = builder_mod._kernel_fingerprint

    def spy(*args, **kwargs):
        h, ok = orig(*args, **kwargs)
        captured.append(h)
        return h, ok
    monkeypatch.setattr(builder_mod, "_kernel_fingerprint", spy)

    reg32, reg64 = {}, {}
    make_graph(End(name="x", init_value=7, ty=int32, registry=reg32), registry=reg32)
    make_graph(End(name="x", init_value=7, ty=int64, registry=reg64), registry=reg64)

    assert len(captured) == 2
    assert captured[0] != captured[1]


def test_make_graph_kernel_name_depends_on_jit_flags(monkeypatch):
    """A change to a jit flag gives the graph maker a new name, and a change to ``cache`` alone does not.

    numba caches the maker together with its CPython wrapper, and ``nogil`` changes the wrapper even when the maker's
    body stays the same. With ``nogil=True``, the wrapper calls ``PyEval_SaveThread`` and ``PyEval_RestoreThread``
    around the call. The anchor of a derive includes the flags, but the maker's hash includes only each derive's
    fingerprint, so ``_kernel_fingerprint`` adds the flags itself. numbox's cache key for the maker includes the flags
    as well.

    The test varies ``nogil`` because ``error_model``, ``fastmath`` and ``boundscheck`` change only arithmetic and
    array indexing. The maker's body, a sequence of ``ll_make_work`` calls and a tuple, has neither. ``cache`` does not
    change the compiled code, so it must not change the name.
    """
    import numbox.core.work.builder as builder_mod
    captured = []
    orig = builder_mod._kernel_fingerprint

    def spy(*args, **kwargs):
        h, ok = orig(*args, **kwargs)
        captured.append(h)
        return h, ok
    monkeypatch.setattr(builder_mod, "_kernel_fingerprint", spy)

    # Each iteration builds the same graph, and only the jit flags vary. The node keeps one name, because the name is
    # part of the maker's code, and another name would change the hash by itself.
    for opts in ({"cache": False, "nogil": True},
                 {"cache": False, "nogil": False},
                 {"cache": True, "nogil": False}):
        reg = {}
        make_graph(End(name="flag_x", init_value=7, ty=int64, registry=reg),
                   registry=reg, jit_options=opts)

    assert len(captured) == 3
    assert captured[0] != captured[1], "nogil absent from the kernel fingerprint"
    assert captured[1] == captured[2], (
        "`cache` re-keyed the kernel; it does not change the emitted binary and "
        "folding it would split the cache for nothing"
    )


def test_make_graph_exec_defined_derive_is_cached_through_its_anchor():
    """Derives defined by ``exec`` compute their values, also when two of them differ only in a constant.

    numba has no cache locator for ``<string>``, the file name of a function defined by ``exec``. ``make_graph``
    compiles such a derive through a wrapper in an anchor file that numbox writes, so numba can cache it. The anchor's
    name includes a digest of the derive's fingerprint. The second derive, which differs from the first only in a
    constant, therefore gets an anchor of its own and computes its own value.
    """
    reg = {}
    exec_ns = {}
    exec("def d(x):\n    return x + 1.0\n", exec_ns)  # nosec B102 - test fixture
    x_ = End(name="exec_x", init_value=2.0, registry=reg)
    y_ = Derived(name="exec_y", init_value=0.0, derive=exec_ns["d"], sources=(x_,), registry=reg)
    access = make_graph(y_, registry=reg)
    access.exec_y.calculate()
    assert isclose(access.exec_y.data, 3.0)

    # This derive differs from the first only in a constant. Its fingerprint differs, so its anchor has another name,
    # and numba does not load the code compiled for the first derive.
    reg2 = {}
    exec_ns2 = {}
    exec("def d(x):\n    return x + 100.0\n", exec_ns2)  # nosec B102 - test fixture
    x2_ = End(name="exec_x2", init_value=2.0, registry=reg2)
    y2_ = Derived(name="exec_y2", init_value=0.0, derive=exec_ns2["d"], sources=(x2_,), registry=reg2)
    access2 = make_graph(y2_, registry=reg2)
    access2.exec_y2.calculate()
    assert isclose(access2.exec_y2.data, 102.0)


def test_derive_anchor_degrade_drops_the_cache(monkeypatch):
    """If the anchor of a derive cannot be built, the derive compiles without a cache.

    Only the anchor keys a cached derive on what the derive reads. A plain ``cres(..., cache=True)`` compile names its
    cache file after the derive's source file and qualified name, and numbox's key for it includes the flags but not
    the values of the derive's globals or the code of its callees. Without the anchor, numba could therefore load a
    stale derive from the cache after one of those changes. ``_derive_anchor_cres`` returns ``None`` when a flag has no
    canonical form, when no anchor file can be written, or when numba has no cache location for the anchor. The test
    replaces it with a function that always returns ``None``.
    """
    from numbox.core.work import builder as builder_mod

    recorded = {}
    real_cres = builder_mod.cres

    def recording_cres(sig, **kwargs):
        recorded.update(kwargs)
        return real_cres(sig, **kwargs)

    monkeypatch.setattr(builder_mod, "cres", recording_cres)
    monkeypatch.setattr(builder_mod, "_derive_anchor_cres", lambda *a, **k: None)

    reg = {}
    x = End(name="degrade_x", init_value=1.0, ty=float64, registry=reg)

    def derive_degrade(x):
        return x * 2.0

    y = Derived(name="degrade_y", init_value=0.0, derive=derive_degrade,
                sources=(x,), ty=float64, registry=reg)

    builder_mod._derived_cres(
        float64, y.sources, derive_degrade,
        {"cache": True}, "fingerprint-placeholder",
    )
    assert recorded.get("cache") is False, (
        f"anchor degrade compiled with cache={recorded.get('cache')!r}; a plain "
        "cache=True compile is the entry, blind to what the derive reads, that the anchor exists to avoid"
    )


def test_make_graph_uncanonical_jit_flags_drop_the_kernel_cache(monkeypatch):
    """A jit flag without a canonical form makes the graph maker compile without a cache, and canonical flags keep it.

    ``_flags_canon`` falls back to ``repr(sorted(flags.items(), key=repr))`` for a flag that it cannot fingerprint,
    such as a ``pipeline_class`` or a ``locals`` dictionary of numba types. The ``repr`` of an object with the default
    ``repr`` differs between processes, and two different values can have the same ``repr``. A maker name built on it
    could therefore change from one process to the next, or be the same for two graphs that compile differently. The
    derive anchor and ``compile_kernel`` compile without a cache in this case, and the graph maker does the same.
    """
    from numba.core.compiler import Compiler
    from numbox.core.work import builder as builder_mod

    recorded = {}
    real_njit = builder_mod.njit

    def recording_njit(**kwargs):
        recorded.update(kwargs)
        return real_njit(**kwargs)

    monkeypatch.setattr(builder_mod, "njit", recording_njit)

    # Compiler is numba's default pipeline, so the graph compiles as usual. Passing it explicitly gives the jit options
    # a flag that has no canonical form.
    reg = {}
    make_graph(
        End(name="uncanon_x", init_value=7, ty=int64, registry=reg),
        registry=reg, jit_options={"cache": True, "pipeline_class": Compiler},
    )
    assert recorded.get("cache") is False, (
        f"kernel compiled with cache={recorded.get('cache')!r} while its name folds a "
        "repr-only flag value; the key cannot see what the flag changed"
    )

    recorded.clear()
    reg_ok = {}
    make_graph(
        End(name="canon_x", init_value=7, ty=int64, registry=reg_ok),
        registry=reg_ok, jit_options={"cache": True, "nogil": True},
    )
    assert recorded.get("cache") is True, (
        "a fully canonicalizable flag set must keep the kernel cached; dropping it "
        "here would disable graph caching outright"
    )


if __name__ == "__main__":
    collect_and_run_tests(__name__)
