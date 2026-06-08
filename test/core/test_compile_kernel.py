import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from numba import njit
from numba.core.dispatcher import Dispatcher
from numba.core.errors import TypingError
from numba.core.types import float64
from numbox.core.variable.compile_kernel import (
    _sanitize, _assign_identifiers, _wrap_formula, _generate_body, _compile,
    compile_kernel, CompiledKernel,
)
from numbox.core.variable.variable import Variable, Graph, Values
from numbox.utils.highlevel import cres


def test_sanitize_basic():
    assert _sanitize("variables.a") == "variables_a"
    assert _sanitize("first-name") == "first_name"
    assert _sanitize("3m") == "v_3m"
    assert _sanitize("a..b") == "a_b"
    assert _sanitize("") == "v_"


def test_assign_identifiers_unique_and_valid():
    v1 = Variable(name="c", source="a_b")     # qual a_b.c -> base a_b_c
    v2 = Variable(name="b_c", source="a")     # qual a.b_c -> base a_b_c (collision)
    idents = _assign_identifiers([v1, v2])
    assert idents[v1] != idents[v2]
    assert all(s.isidentifier() for s in idents.values())


def test_assign_identifiers_formula_prefix_collision():
    node = Variable(name="x", source="variables")        # base variables_x
    clash = Variable(name="variables_x", source="f")     # base f_variables_x == f_<node temp>
    idents = _assign_identifiers([node, clash])
    temps = set(idents.values())
    fgs = {"f_" + t for t in temps}
    assert temps.isdisjoint(fgs)                          # no temp equals any formula global


def test_assign_identifiers_deterministic():
    v1 = Variable(name="c", source="a_b")
    v2 = Variable(name="b_c", source="a")
    assert _assign_identifiers([v1, v2]) == _assign_identifiers([v1, v2])


def test_assign_identifiers_invalid_char_and_leading_digit():
    v1 = Variable(name="first-name", source="ext")   # invalid char
    v2 = Variable(name="3m", source="ext")           # leading digit
    idents = _assign_identifiers([v1, v2])
    assert all(s.isidentifier() for s in idents.values())
    assert idents[v1] != idents[v2]


def _diamond_graph():
    return Graph(
        variables_lists={"variables": [
            {"name": "x", "inputs": {"y": "basket"}, "formula": njit(lambda y: 2 * y)},
            {"name": "a", "inputs": {"x": "variables"}, "formula": njit(lambda x: x - 74)},
            {"name": "b", "inputs": {"x": "variables"}, "formula": njit(lambda x: x + 0.5)},
            {"name": "u", "inputs": {"a": "variables", "b": "variables"},
             "formula": njit(lambda a, b: a + b)},
        ]},
        external_source_names=["basket"],
    )


def test_wrap_formula_passthrough_and_wrap():
    d = njit(lambda x: x)
    assert _wrap_formula(d) is d

    def plain(x):
        return x + 1
    assert isinstance(_wrap_formula(plain), Dispatcher)


def test_generate_body_shape():
    g = _diamond_graph()
    compiled = g.compile(["variables.u", "variables.a"])
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    source, bindings, params, outputs = _generate_body(compiled, ["variables.u", "variables.a"], idents)
    y_var = next(v for v in idents if v.qual_name() == "basket.y")
    assert params == [("basket", "y", idents[y_var])]
    assert outputs == ["variables.u", "variables.a"]
    assert source.startswith("def _kernel(")
    assert source.rstrip().endswith(",)")
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    expected = {"f_" + idents[n.variable] for n in compiled.ordered_nodes if n.variable not in external}
    assert set(bindings) == expected


def test_generate_body_errors():
    g = _diamond_graph()
    compiled = g.compile(["variables.u"])
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    with pytest.raises(ValueError):
        _generate_body(compiled, [], idents)
    with pytest.raises(ValueError):
        _generate_body(compiled, ["variables.nope"], idents)

    gph = Graph(
        variables_lists={"variables": [
            {"name": "x", "inputs": {"y": "basket"}, "formula": njit(lambda y: 2 * y)},
            {"name": "broken", "inputs": {"x": "variables"}, "formula": None},
        ]},
        external_source_names=["basket"],
    )
    c2 = gph.compile(["variables.broken"])
    id2 = _assign_identifiers([n.variable for n in c2.ordered_nodes])
    with pytest.raises(ValueError):
        _generate_body(c2, ["variables.broken"], id2)


def test_generate_body_external_as_only_output():
    g = _diamond_graph()
    compiled = g.compile(["basket.y"])
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    source, bindings, params, outputs = _generate_body(compiled, ["basket.y"], idents)
    assert bindings == {}
    assert outputs == ["basket.y"]
    assert "pass" not in source
    y_ident = params[0][2]
    assert source == f"def _kernel({y_ident}):\n    return ({y_ident},)\n"


def test_safe_getsource_named_function_and_cres():
    from numbox.core.variable.compile_kernel import _safe_getsource

    @njit
    def named(x):
        return x + 41
    src = _safe_getsource(named)
    assert "return x + 41" in src

    wap = cres(float64(float64))(lambda x: x * 2.0)
    s = _safe_getsource(wap)             # must not raise
    assert isinstance(s, str) and s      # non-empty (repr fallback is acceptable)


def test_compile_runs():
    src = "def _kernel(y):\n    x = f_x(y)\n    return (x,)\n"
    bindings = {"f_x": njit(lambda y: 2 * y)}
    kernel = _compile(src, bindings, None, True)
    assert kernel(10) == (20,)


def test_compile_anchor_is_content_addressed(tmp_path, monkeypatch):
    import numbox.utils.preprocessing as pp
    monkeypatch.setattr(pp, "_anchor_root", lambda subdir: tmp_path)
    src = "def _kernel(y):\n    x = f_x(y)\n    return (x,)\n"
    _compile(src, {"f_x": njit(lambda y: 2 * y)}, None, True)
    before = set(tmp_path.glob("_kernel_*.py"))
    assert before, "first _compile must create an anchor"
    _compile(src, {"f_x": njit(lambda y: 3 * y)}, None, True)
    after = set(tmp_path.glob("_kernel_*.py"))
    assert after - before, "different formula must produce a new anchor"


def test_compile_cache_survives_fresh_process(tmp_path):
    script = textwrap.dedent('''
        from numba import njit
        from numbox.core.variable.compile_kernel import _compile
        src = "def _kernel(y):\\n    x = f_x(y)\\n    return (x,)\\n"
        k = _compile(src, {"f_x": njit(lambda y: 2 * y)}, None, True)
        print("RESULT", k(10)[0])
    ''')
    f = tmp_path / "ck_warm.py"
    f.write_text(script)
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "nbcache")}
    for _ in range(2):
        p = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, env=env)
        assert p.returncode == 0, p.stdout + p.stderr
        assert "RESULT 20" in p.stdout, p.stdout + p.stderr


def _pure(graph, required, external_values):
    compiled = graph.compile(required)
    values = Values()
    compiled.execute(external_values, values)
    by_qual = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    return {q: values.get(by_qual[q]).value for q in required}


def test_compile_kernel_matches_pure_python_diamond():
    g = _diamond_graph()
    req = ["variables.u", "variables.a"]
    ck = compile_kernel(g, req)
    assert isinstance(ck, CompiledKernel)
    ext = {"basket": {"y": 100}}
    assert ck.execute(ext) == _pure(g, req, ext)
    assert ck.params == ["basket.y"]
    assert ck.outputs == req
    assert tuple(ck.kernel(100)) == tuple(_pure(g, req, ext)[q] for q in req)


def test_compile_kernel_single_output_and_str_required():
    g = _diamond_graph()
    ck = compile_kernel(g, "variables.u")
    assert ck.outputs == ["variables.u"]
    assert ck.execute({"basket": {"y": 100}}) == {"variables.u": 326.5}


def test_compile_kernel_auto_specialization():
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u"])
    assert ck.execute({"basket": {"y": 100}})["variables.u"] == 326.5
    assert ck.execute({"basket": {"y": 100.0}})["variables.u"] == 326.5


def test_compile_kernel_missing_external_raises():
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u"])
    with pytest.raises(KeyError) as exc:
        ck.execute({"basket": {}})
    assert "basket.y" in str(exc.value)


def test_compile_kernel_missing_external_source_raises():
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u"])
    with pytest.raises(KeyError) as exc:
        ck.execute({})                      # entire 'basket' source absent
    assert "basket.y" in str(exc.value)


def test_identifier_collision_graph_runs():
    g = Graph(
        variables_lists={
            "a_b": [{"name": "c", "inputs": {"y": "ext"}, "formula": njit(lambda y: y + 1)}],
            "a": [{"name": "b_c", "inputs": {"y": "ext"}, "formula": njit(lambda y: y + 2)}],
        },
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["a_b.c", "a.b_c"])
    assert ck.execute({"ext": {"y": 10}}) == {"a_b.c": 11, "a.b_c": 12}


def test_invalid_char_external_name_runs():
    g = Graph(
        variables_lists={"variables": [
            {"name": "out", "inputs": {"first-name": "ext"}, "formula": njit(lambda v: v * 2)},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["variables.out"])
    assert ck.execute({"ext": {"first-name": 5}}) == {"variables.out": 10}


def test_constant_and_array_formulas():
    g = Graph(
        variables_lists={"variables": [
            {"name": "k", "inputs": {}, "formula": njit(lambda: 7.0)},
            {"name": "u", "inputs": {"k": "variables", "y": "ext"}, "formula": njit(lambda k, y: k + y)},
            {"name": "arr", "inputs": {"y": "ext"}, "formula": njit(lambda y: np.arange(y))},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["variables.u", "variables.arr"])
    out = ck.execute({"ext": {"y": 3}})
    assert out["variables.u"] == 10.0
    assert list(out["variables.arr"]) == [0, 1, 2]


def test_autowrap_plain_python_formula():
    def plain(y):
        return y * 3
    g = Graph(
        variables_lists={"variables": [
            {"name": "o", "inputs": {"y": "ext"}, "formula": plain},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["variables.o"])
    assert ck.execute({"ext": {"y": 4}}) == {"variables.o": 12}


def test_cres_formula():
    sub = cres(float64(float64, float64))(lambda a, b: a - b)
    g = Graph(
        variables_lists={"variables": [
            {"name": "u", "inputs": {"p": "ext", "q": "ext"}, "formula": sub},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["variables.u"])
    assert ck.execute({"ext": {"p": 1.5, "q": 2.0}}) == {"variables.u": -0.5}


def test_non_jittable_formula_fails_at_first_call_not_compile():
    def bad(y):
        return open("/tmp/_ck_nope.txt", "w")
    g = Graph(
        variables_lists={"variables": [
            {"name": "b", "inputs": {"y": "ext"}, "formula": bad},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["variables.b"])     # must NOT raise here (lazy)
    with pytest.raises(TypingError):
        ck.execute({"ext": {"y": 1}})           # error surfaces at first call


def test_cache_no_skeleton_collision(tmp_path):
    # Two kernels with identical skeleton (one leaf -> one formula -> return) but
    # different formulas (y*10 vs y*1000). With cache=True they must NOT load each
    # other's cached binary. Run twice in fresh interpreters so the 2nd run reads
    # the on-disk cache.
    script = textwrap.dedent('''
        from numba import njit
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel

        def build(mult):
            g = Graph(
                variables_lists={"v": [
                    {"name": "o", "inputs": {"y": "e"}, "formula": njit(lambda y: y * mult)},
                ]},
                external_source_names=["e"],
            )
            return compile_kernel(g, ["v.o"], cache=True)

        a = build(10)
        b = build(1000)
        ra = a.execute({"e": {"y": 1}})["v.o"]
        rb = b.execute({"e": {"y": 1}})["v.o"]
        assert ra == 10, ra
        assert rb == 1000, rb
        print("OK", ra, rb)
    ''')
    f = tmp_path / "ck_cache_probe.py"
    f.write_text(script)
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "nbcache")}
    for _ in range(2):
        p = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, env=env)
        assert p.returncode == 0, p.stdout + p.stderr
        assert "OK 10 1000" in p.stdout, p.stdout + p.stderr


def test_assign_identifiers_avoids_python_keyword():
    # qual_name ".for" sanitizes to the keyword "for"; must get a suffix
    v = Variable(name="for", source="")
    ident = _assign_identifiers([v])[v]
    import keyword as _kw
    assert not _kw.iskeyword(ident)
    assert ident.startswith("for_")


def test_compile_kernel_keyword_node_name():
    # namespace "_" + variable "for" -> qual_name "_.for" -> sanitizes to "for"
    g = Graph(
        variables_lists={"_": [
            {"name": "for", "inputs": {"y": "e"}, "formula": njit(lambda y: y + 1)},
        ]},
        external_source_names=["e"],
    )
    ck = compile_kernel(g, ["_.for"])
    assert ck.execute({"e": {"y": 5}}) == {"_.for": 6}


def test_compile_kernel_newline_in_name_is_safe():
    # an external input name containing a newline must not break the generated
    # source via the trailing comment
    g = Graph(
        variables_lists={"v": [
            {"name": "o", "inputs": {"a\nx": "e"}, "formula": njit(lambda a: a + 1)},
        ]},
        external_source_names=["e"],
    )
    ck = compile_kernel(g, ["v.o"])
    # the raw newline from the name must not appear unescaped in the generated
    # source (that would split the trailing comment into an executable line)
    assert "e.a\nx" not in ck.source
    assert ck.execute({"e": {"a\nx": 10}}) == {"v.o": 11}


def test_compile_kernel_duplicate_required_deduped():
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u", "variables.u"])
    assert ck.outputs == ["variables.u"]
    assert ck.execute({"basket": {"y": 100}}) == {"variables.u": 326.5}
    assert ck.kernel(100) == (326.5,)
