import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from numba import njit, cfunc, vectorize
from numba.core.dispatcher import Dispatcher
from numba.core.types import float64
from numbox.core.variable.compile_kernel import (
    _sanitize, _assign_identifiers, _wrap_formula, _generate_body, _compile,
    compile_kernel, CompiledKernel,
)
from numbox.core.variable.variable import Variable, Graph, Values
from numbox.utils.highlevel import cres
from test.auxiliary_utils import assert_njit_cache_survives_subprocess_roundtrip


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


def test_fingerprint_named_function_and_cres():
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    @njit
    def named(x):
        return x + 41
    text, ok = _formula_fingerprint(named)
    assert ok
    assert named.py_func.__qualname__ in text

    wap = cres(float64(float64))(lambda x: x * 2.0)
    text2, ok2 = _formula_fingerprint(wap)   # must not raise
    assert not ok2                           # cres WAP is un-fingerprintable
    assert isinstance(text2, str) and text2  # non-empty per-object fallback


def test_compile_runs():
    src = "def _kernel(y):\n    x = f_x(y)\n    return (x,)\n"
    bindings = {"f_x": njit(lambda y: 2 * y)}
    kernel = _compile(src, bindings, None, True)
    assert kernel(10) == (20,)


def test_compile_anchor_is_content_addressed(tmp_path, monkeypatch):
    import numbox.core.variable.compile_kernel as ck_mod
    monkeypatch.setattr(ck_mod, "_anchor_root", lambda subdir: tmp_path)
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


def test_non_jittable_formula_demotes_at_first_call():
    def bad(y):
        return json.dumps({"y": y}) and y + 100.0
    g = Graph(
        variables_lists={"variables": [
            {"name": "b", "inputs": {"y": "ext"}, "formula": bad},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["variables.b"])     # still must NOT raise here (lazy)
    assert ck.execute({"ext": {"y": 1.0}}) == {"variables.b": bad(1.0)}  # first call succeeds via demotion
    assert ck.partition.mode == "segmented"
    assert "variables.b" in ck.partition.python_nodes
    reasons = {q: r for s in ck.partition.segments for q, r in s.reasons.items()}
    assert reasons["variables.b"]


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


def test_fingerprint_fallback_is_per_object():
    # A callable whose source can't be fingerprinted and whose __repr__ is
    # non-unique must still hash distinctly per object (via the id() suffix),
    # so two such formulas never collide in the content-addressed cache.
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    class _Konst:
        def __repr__(self):
            return "<konst>"

        def __call__(self, x):
            return x

    a, b = _Konst(), _Konst()
    text_a, ok_a = _formula_fingerprint(a)
    text_b, ok_b = _formula_fingerprint(b)
    assert not ok_a and not ok_b
    assert text_a != text_b
    assert " @" in text_a and " @" in text_b


def test_kernel_dispatcher_collectable_after_release():
    import gc
    import weakref

    def f(x):
        return x * 2.0

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    ck = compile_kernel(g, "calc.y", cache=False)
    assert ck.execute({"ext": {"x": 3.0}}) == {"calc.y": 6.0}
    assert ck.kernel(4) == (8.0,)
    ref = weakref.ref(ck.kernel)
    del ck
    for _ in range(3):
        gc.collect()
    assert ref() is None


_CACHE_PROBE = """
    import ast
    import pathlib
    import sys
    from numbox.core.variable.variable import Graph
    from numbox.core.variable.compile_kernel import compile_kernel

    def f(x):
        return x + 1.0

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    kwargs = ast.literal_eval(sys.argv[1])
    ck = compile_kernel(g, "calc.y", **kwargs)
    print(ck.execute({"ext": {"x": 1.0}})["calc.y"])
"""


def _run_cache_probe(tmp_path, kwargs_src, extra_env=None):
    f = tmp_path / "probe.py"
    f.write_text(textwrap.dedent(_CACHE_PROBE))
    cache_dir = tmp_path / "nbcache"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir), **(extra_env or {})}
    p = subprocess.run(
        [sys.executable, str(f), kwargs_src],
        capture_output=True, text=True, env=env,
    )
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "2.0"
    files = [q for q in cache_dir.rglob("*") if q.is_file()] if cache_dir.exists() else []
    return files, p.stderr


def test_cache_false_writes_nothing(tmp_path):
    files, _ = _run_cache_probe(tmp_path, "{'cache': False}")
    assert files == []


def test_cache_precedence_env_knob(tmp_path):
    files, _ = _run_cache_probe(
        tmp_path, "{}", extra_env={"NUMBOX_JIT_OPTIONS": '{"cache": false}'})
    assert files == []


def test_cache_precedence_jit_options(tmp_path):
    files, _ = _run_cache_probe(tmp_path, "{'jit_options': {'cache': False}}")
    assert files == []


def test_cache_precedence_param_wins(tmp_path):
    files, _ = _run_cache_probe(
        tmp_path, "{'cache': True, 'jit_options': {'cache': False}}")
    assert files != []


@pytest.mark.skipif(sys.platform == "win32", reason="chmod-based read-only dir is POSIX-only")
def test_readonly_cache_dir_degrades_gracefully(tmp_path):
    cache_dir = tmp_path / "nbcache"
    cache_dir.mkdir()
    cache_dir.chmod(0o500)
    try:
        files, stderr = _run_cache_probe(tmp_path, "{}")
        assert files == []
        assert "cache directory unusable" in stderr
    finally:
        cache_dir.chmod(0o700)


def test_fingerprint_same_line_lambdas_distinct():
    from numbox.core.variable.compile_kernel import _formula_fingerprint
    f10, f1000 = (lambda y: y * 10.0), (lambda y: y * 1000.0)
    fp_a, ok_a = _formula_fingerprint(f10)
    fp_b, ok_b = _formula_fingerprint(f1000)
    assert ok_a and ok_b
    assert fp_a != fp_b


def test_fingerprint_covers_globals_helpers_defaults():
    from numbox.core.variable.compile_kernel import _formula_fingerprint
    src = textwrap.dedent("""
        SCALE = {scale}
        def helper(v):
            return v {op} 1.0
        def f(x, m={default}):
            return helper(x) * SCALE * m
    """)
    variants = {}
    for key, (scale, op, default) in {
        "base": ("2.0", "+", "1.0"),
        "global": ("3.0", "+", "1.0"),
        "helper": ("2.0", "-", "1.0"),
        "default": ("2.0", "+", "5.0"),
    }.items():
        ns = {}
        exec(compile(src.format(scale=scale, op=op, default=default), f"<fp-{key}>", "exec"), ns)
        fp, ok = _formula_fingerprint(ns["f"])
        assert ok
        variants[key] = fp
    assert len(set(variants.values())) == 4


def test_fingerprint_large_array_cells_distinct():
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    def factory(a):
        return lambda x: x + a[500]

    a1, a2 = np.zeros(2000), np.zeros(2000)
    a2[500] = 1.0
    assert repr(a1) == repr(a2)
    fp1, _ = _formula_fingerprint(factory(a1))
    fp2, _ = _formula_fingerprint(factory(a2))
    assert fp1 != fp2


def test_fingerprint_set_cells_order_stable():
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    def factory(s):
        return lambda x: x if x in s else -x

    s1 = {"alpha", "beta", "gamma"}
    s2 = set(reversed(sorted(s1)))
    assert _formula_fingerprint(factory(s1))[0] == _formula_fingerprint(factory(s2))[0]


def test_fingerprint_recursive_helpers_terminate():
    from numbox.core.variable.compile_kernel import _formula_fingerprint
    ns = {}
    exec(textwrap.dedent("""
        def even(n):
            return n == 0 or odd(n - 1)
        def odd(n):
            return n != 0 and even(n - 1)
        def f(x):
            return x if even(int(x)) else -x
    """), ns)
    fp, ok = _formula_fingerprint(ns["f"])
    assert ok and "recursive(" in fp


def test_fingerprint_fallback_paths():
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    class Boom:
        def __repr__(self):
            raise RuntimeError("no repr")

    def factory(cfg):
        return lambda x: x if cfg is None else x

    boom_lambda = factory(Boom())
    assert boom_lambda.__closure__ is not None
    fp, ok = _formula_fingerprint(boom_lambda)
    assert not ok and " @" in fp

    @cres(float64(float64))
    def wap(x):
        return x * 3.0

    fp2, ok2 = _formula_fingerprint(wap)
    assert not ok2


def test_fingerprint_wrapped_decorator_uses_executing_callable():
    import functools
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    def shared(x):
        return x + 1.0

    def make_wrapper(rate):
        @functools.wraps(shared)
        def wrapper(x):
            return shared(x) * rate
        return wrapper

    low, high = make_wrapper(1.05), make_wrapper(1.20)
    assert low.__wrapped__ is shared and high.__wrapped__ is shared
    fp_low, ok_low = _formula_fingerprint(low)
    fp_high, ok_high = _formula_fingerprint(high)
    assert ok_low and ok_high
    assert fp_low != fp_high


def test_fingerprint_zero_d_vs_one_d_array_distinct():
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    def factory(a):
        return lambda x: x + a.sum()

    fp0, ok0 = _formula_fingerprint(factory(np.array(3.5)))
    fp1, ok1 = _formula_fingerprint(factory(np.array([3.5])))
    assert ok0 and ok1
    assert fp0 != fp1


def test_fingerprint_deep_nesting_downgrades_not_crashes():
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    deep = [0.0]
    for _ in range(5000):
        deep = [deep]

    def factory(d):
        return lambda x: x if d is None else x

    fp, ok = _formula_fingerprint(factory(deep))
    assert not ok and " @" in fp


_DIGEST_PROBE = """
    import sys
    sys.path.insert(0, {moddir!r})
    import formulas_mod
    from numbox.core.variable.variable import Graph
    from numbox.core.variable.compile_kernel import compile_kernel

    g = Graph({{"calc": [{{"name": "y", "inputs": {{"x": "ext"}}, "formula": formulas_mod.f}}]}}, ["ext"])
    ck = compile_kernel(g, "calc.y")
    print(ck.execute({{"ext": {{"x": 10.0}}}})["calc.y"])
"""


def test_digest_global_change_invalidates_cache(tmp_path):
    moddir = tmp_path / "mods"
    moddir.mkdir()
    mod = moddir / "formulas_mod.py"
    runner = tmp_path / "run.py"
    runner.write_text(textwrap.dedent(_DIGEST_PROBE.format(moddir=str(moddir))))
    # Disable .pyc caching: the two writes can land in the same mtime tick and
    # are the same byte length, so a cached bytecode would mask the second
    # source -- we want the subprocess to recompile from the new SCALE each run.
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "nbcache"),
           "PYTHONDONTWRITEBYTECODE": "1"}

    mod.write_text("SCALE = 2.0\ndef f(x):\n    return x * SCALE\n")
    p1 = subprocess.run([sys.executable, str(runner)], capture_output=True, text=True, env=env)
    assert p1.returncode == 0, p1.stderr
    assert p1.stdout.strip() == "20.0"

    mod.write_text("SCALE = 3.0\ndef f(x):\n    return x * SCALE\n")
    p2 = subprocess.run([sys.executable, str(runner)], capture_output=True, text=True, env=env)
    assert p2.returncode == 0, p2.stderr
    assert p2.stdout.strip() == "30.0"


def test_digest_large_array_closure_no_collision():
    def factory(a):
        return lambda x: x + a[500]

    results = []
    for fill in (0.0, 1.0):
        a = np.zeros(2000)
        a[500] = fill
        g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": factory(a)}]}, ["ext"])
        results.append(compile_kernel(g, "calc.y").execute({"ext": {"x": 1.0}})["calc.y"])
    assert results == [1.0, 2.0]


def test_digest_same_line_lambdas_no_collision():
    f10, f1000 = (lambda y: y * 10.0), (lambda y: y * 1000.0)
    results = []
    for f in (f10, f1000):
        g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
        results.append(compile_kernel(g, "calc.y").execute({"ext": {"x": 1.0}})["calc.y"])
    assert results == [10.0, 1000.0]


def test_digest_includes_jit_flags():
    def f(x):
        return 1.0 / x

    def build():
        return Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])

    # numba's default error_model is "python" (1.0/0.0 raises); "numpy" returns
    # inf. Same formula, two distinct jit-flag sets -> two distinct kernels with
    # different behavior: the flags must be part of the cache digest.
    ck_default = compile_kernel(build(), "calc.y")
    with pytest.raises(ZeroDivisionError):
        ck_default.execute({"ext": {"x": 0.0}})
    ck_numpy = compile_kernel(build(), "calc.y", jit_options={"error_model": "numpy"})
    assert ck_numpy.execute({"ext": {"x": 0.0}})["calc.y"] == np.inf


def test_digest_cres_kernel_uncached_and_quiet(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent("""
        import warnings
        from numba.core.types import float64
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel
        from numbox.utils.highlevel import cres

        @cres(float64(float64))
        def f(x):
            return x - 1.0

        g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ck = compile_kernel(g, "calc.y")
            print(ck.execute({"ext": {"x": 3.0}})["calc.y"])
    """))
    cache_dir = tmp_path / "nbcache"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir)}
    p = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "2.0"
    files = [q for q in cache_dir.rglob("*") if q.is_file()] if cache_dir.exists() else []
    assert files == []


def test_digest_mixed_cres_graph_uncached(tmp_path):
    # A plain-python formula is fingerprintable, but a cres/CompileResultWAP node
    # in the same kernel is not -- the uncacheable verdict must propagate to the
    # whole kernel: zero cache files, correct results for both nodes. (An object
    # cell in a lambda body cannot be njit-compiled at all, so this mixed-graph
    # form is the runnable analogue of the un-fingerprintable-formula case.)
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent("""
        from numba.core.types import float64
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel
        from numbox.utils.highlevel import cres

        @cres(float64(float64))
        def sub_one(x):
            return x - 1.0

        def add_two(x):
            return x + 2.0

        g = Graph({"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": sub_one},
            {"name": "b", "inputs": {"x": "ext"}, "formula": add_two},
        ]}, ["ext"])
        out = compile_kernel(g, ["calc.a", "calc.b"]).execute({"ext": {"x": 10.0}})
        print(out["calc.a"], out["calc.b"])
    """))
    cache_dir = tmp_path / "nbcache"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir)}
    p = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "9.0 12.0"
    files = [q for q in cache_dir.rglob("*") if q.is_file()] if cache_dir.exists() else []
    assert files == []


def test_dufunc_and_cfunc_formulas_accepted(tmp_path):
    # A CFunc node poisons the whole kernel's cacheability (its .address is an
    # ASLR-randomized pointer numba cannot disk-cache), so no anchor and no numba
    # cache files may exist -- and compiling must stay warning-free.
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent("""
        import warnings
        from numba import vectorize, cfunc
        from numba.core.types import float64
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel

        d = vectorize(lambda a: a + 0.5)
        c = cfunc(float64(float64))(lambda a: a * 2.0)
        g = Graph({"calc": [
            {"name": "u", "inputs": {"x": "ext"}, "formula": d},
            {"name": "v", "inputs": {"u": "calc"}, "formula": c},
        ]}, ["ext"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ck = compile_kernel(g, "calc.v")
            print(ck.execute({"ext": {"x": 1.0}})["calc.v"])
    """))
    cache_dir = tmp_path / "nbcache"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir)}
    p = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "3.0"
    files = [q for q in cache_dir.rglob("*") if q.is_file()] if cache_dir.exists() else []
    assert files == []


def test_dufunc_only_kernel_caches_cleanly(tmp_path):
    # A single DUFunc-formula node must disk-cache cleanly: run twice sharing one
    # NUMBA_CACHE_DIR under simplefilter("error") (any NumbaWarning fails it). After
    # run 1 the cache holds at least one .nbi; run 2 is a warm hit that rewrites
    # nothing -- the (path, mtime) set of all .nbc/.nbi files is identical.
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent("""
        import warnings
        from numba import vectorize
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel

        d = vectorize(lambda a: a + 0.5)
        g = Graph({"calc": [
            {"name": "u", "inputs": {"x": "ext"}, "formula": d},
        ]}, ["ext"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ck = compile_kernel(g, "calc.u")
            print(ck.execute({"ext": {"x": 1.0}})["calc.u"])
    """))
    cache_dir = tmp_path / "nbcache"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir)}

    def cache_snapshot():
        return {
            (str(q), q.stat().st_mtime_ns)
            for q in cache_dir.rglob("*")
            if q.is_file() and q.suffix in (".nbc", ".nbi")
        }

    p1 = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, env=env)
    assert p1.returncode == 0, p1.stderr
    assert p1.stdout.strip() == "1.5"
    snap1 = cache_snapshot()
    assert any(q.suffix == ".nbi" for q in cache_dir.rglob("*") if q.is_file())

    p2 = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, env=env)
    assert p2.returncode == 0, p2.stderr
    assert p2.stdout.strip() == "1.5"
    assert cache_snapshot() == snap1


def test_dufunc_cfunc_fingerprints_cacheable_and_distinct():
    from numba import vectorize, cfunc
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    def inner(a):
        return a + 0.5

    d = vectorize(inner)
    c = cfunc(float64(float64))(inner)
    fp_plain, ok_plain = _formula_fingerprint(inner)
    fp_d, ok_d = _formula_fingerprint(d)
    fp_c, ok_c = _formula_fingerprint(c)
    assert ok_plain and ok_d
    assert not ok_c
    assert len({fp_plain, fp_d, fp_c}) == 3


def test_not_callable_formula_rejected_eagerly():
    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": "lambda y: y"}]}, ["ext"])
    with pytest.raises(TypeError, match=r"calc\.y.*not callable"):
        compile_kernel(g, "calc.y")


def test_arity_mismatch_rejected_eagerly():
    def one_arg(x):
        return x

    g = Graph({"calc": [{"name": "y", "inputs": {"a": "ext", "b": "ext"}, "formula": one_arg}]}, ["ext"])
    with pytest.raises(ValueError, match=r"calc\.y.*2 declared input"):
        compile_kernel(g, "calc.y")


def test_kwonly_formula_rejected_eagerly():
    def kw_only(*, y):
        return y

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": kw_only}]}, ["ext"])
    with pytest.raises(ValueError, match=r"calc\.y"):
        compile_kernel(g, "calc.y")


def test_varargs_formula_still_accepted():
    def star(*vals):
        return vals[0] + vals[1]

    g = Graph({"calc": [{"name": "y", "inputs": {"a": "ext", "b": "ext"}, "formula": star}]}, ["ext"])
    out = compile_kernel(g, "calc.y").execute({"ext": {"a": 1.0, "b": 2.0}})
    assert out == {"calc.y": 3.0}


def test_required_validation_messages():
    def f(x):
        return x

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    with pytest.raises(TypeError, match=r"required entries.*42"):
        compile_kernel(g, ["calc.y", 42])
    with pytest.raises(TypeError, match=r"required entries"):
        compile_kernel(g, [{"name": "calc.y"}])
    with pytest.raises(ValueError, match=r"'caly'.*not qualified"):
        compile_kernel(g, "caly")
    with pytest.raises(ValueError, match=r"cannot be resolved.*nope"):
        compile_kernel(g, "calc.nope")
    g2 = Graph({"calc": [{"name": "y", "inputs": {"x": "bad_source"}, "formula": f}]}, ["ext"])
    with pytest.raises(ValueError, match=r"or one of its dependencies.*bad_source"):
        compile_kernel(g2, "calc.y")


def test_external_typo_warns_but_compiles():
    def f(x):
        return x

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    with pytest.warns(UserWarning, match="did not exist before compilation"):
        ck = compile_kernel(g, ["ext.tpyo"])
    assert ck.execute({"ext": {"tpyo": 5.0}}) == {"ext.tpyo": 5.0}


def test_external_variable_with_formula_rejected():
    def f(x):
        return x

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    g.external["ext"].update("x", Variable(name="x", source="ext", formula=lambda: 1.0))
    with pytest.raises(ValueError, match=r"ext\.x.*external but carries a formula"):
        compile_kernel(g, "calc.y")


def test_deep_chain_recursion_error_is_contextual():
    depth = sys.getrecursionlimit() + 100

    def step(x):
        return x

    specs = [{"name": "n0", "inputs": {"x": "ext"}, "formula": step}]
    specs += [
        {"name": f"n{i}", "inputs": {f"n{i - 1}": "calc"}, "formula": step}
        for i in range(1, depth)
    ]
    g = Graph({"calc": specs}, ["ext"])
    with pytest.raises(RecursionError, match="setrecursionlimit"):
        compile_kernel(g, f"calc.n{depth - 1}")


def test_compile_kernel_cache_save_side(tmp_path):
    assert_njit_cache_survives_subprocess_roundtrip(
        tmp_path,
        """
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel

        def f(x):
            return x * 2.0

        def h(y):
            return y + 1.0

        g = Graph({"calc": [
            {"name": "y", "inputs": {"x": "ext"}, "formula": f},
            {"name": "z", "inputs": {"y": "calc"}, "formula": h},
        ]}, ["ext"])
        ck = compile_kernel(g, "calc.z")
        print(ck.execute({"ext": {"x": 3.0}})["calc.z"])
        """,
        ["7.0"],
    )


def test_external_only_output_end_to_end():
    g = Graph({"calc": []}, ["ext"])
    with pytest.warns(UserWarning, match="did not exist before compilation"):
        ck = compile_kernel(g, "ext.x")
    assert ck.params == ["ext.x"]
    assert ck.execute({"ext": {"x": 5.5}}) == {"ext.x": 5.5}


def test_mixed_outputs_end_to_end():
    def f(x):
        return x * 10.0

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    with pytest.warns(UserWarning, match="did not exist before compilation"):
        ck = compile_kernel(g, ["calc.y", "ext.x"])
    assert ck.outputs == ["calc.y", "ext.x"]
    assert ck.execute({"ext": {"x": 2.0}}) == {"calc.y": 20.0, "ext.x": 2.0}


def test_fingerprint_object_array_is_unfingerprintable():
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    def factory(a):
        return lambda x: x if a is None else x

    arr = np.array([object(), object()], dtype=object)
    fp, ok = _formula_fingerprint(factory(arr))
    assert not ok and " @" in fp


def test_object_array_closure_kernel_uncached(tmp_path):
    # An object-dtype array in a formula's closure is un-fingerprintable, and numba
    # cannot njit-compile a formula that closes over a pyobject array, so the node is
    # demoted to a Python segment and the call succeeds in plain Python. With the only
    # node demoted there are zero jit segments, so nothing fingerprintable is ever
    # compiled -- the load-bearing claim remains the empty cache dir, proving the
    # un-fingerprintable/unjittable content never wrote an anchor.
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent("""
        import numpy as np
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel

        objs = np.array([object(), object()], dtype=object)

        def factory(a):
            keep = a
            return lambda x: x + (0.0 if keep is None else 1.0)

        g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": factory(objs)}]}, ["ext"])
        ck = compile_kernel(g, "calc.y")
        out = ck.execute({"ext": {"x": 1.0}})["calc.y"]
        assert out == 2.0, out
        print("DONE")
    """))
    cache_dir = tmp_path / "nbcache"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir)}
    p = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "DONE"
    files = [q for q in cache_dir.rglob("*") if q.is_file()] if cache_dir.exists() else []
    assert files == []


def test_fingerprint_numpy_scalar_cells_cacheable_and_distinct():
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    def factory(k):
        return lambda x: x + k

    fp5, ok5 = _formula_fingerprint(factory(np.int64(5)))
    fp6, ok6 = _formula_fingerprint(factory(np.int64(6)))
    fpf, okf = _formula_fingerprint(factory(np.float32(2.5)))
    assert ok5 and ok6 and okf
    assert fp5 != fp6
    assert len({fp5, fp6, fpf}) == 3


def test_fingerprint_dispatcher_bad_targetoption_downgrades_not_crashes():
    from numba import njit
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    @njit
    def d(x):
        return x + 1.0

    d.targetoptions["_review_probe"] = object()  # un-canonicalizable
    fp, ok = _formula_fingerprint(d)
    assert not ok and " @" in fp


def test_compile_self_referential_jit_option_no_recursionerror(tmp_path):
    loop = []
    loop.append(loop)

    def f(x):
        return x + 1.0

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    # an un-canonicalizable jit flag must not let RecursionError escape the digest path;
    # numba will reject the unknown option with its OWN error instead.
    with pytest.raises(Exception) as ei:
        compile_kernel(g, "calc.y", jit_options={"_review_loop": loop}).execute({"ext": {"x": 1.0}})
    assert not isinstance(ei.value, RecursionError)


def test_fingerprint_raising_repr_callable_degrades_not_crashes():
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    class BoomRepr:
        def __call__(self, x):
            return x + 1.0

        def __repr__(self):
            raise RuntimeError("repr boom")

    fp, ok = _formula_fingerprint(BoomRepr())   # must not raise
    assert not ok
    assert isinstance(fp, str) and "repr-failed" in fp


def _nodes_from(graph, required):
    compiled = graph.compile(required)
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    return [n for n in compiled.ordered_nodes if n.variable not in external], external


def _parallel_chains_graph():
    # py-chain p1->p2 and jit-chain j1->j2 from one external, joined by jit sink.
    return Graph(
        variables_lists={"variables": [
            {"name": "p1", "inputs": {"x": "ext"}, "formula": lambda x: x + 1.0},
            {"name": "p2", "inputs": {"p1": "variables"}, "formula": lambda p1: p1 * 2.0},
            {"name": "j1", "inputs": {"x": "ext"}, "formula": lambda x: x - 1.0},
            {"name": "j2", "inputs": {"j1": "variables"}, "formula": lambda j1: j1 * 3.0},
            {"name": "out", "inputs": {"p2": "variables", "j2": "variables"},
             "formula": lambda p2, j2: p2 + j2},
        ]},
        external_source_names=["ext"],
    )


def test_linearize_minimizes_runs_on_parallel_chains():
    from numbox.core.variable._kernel_partition import build_runs, linearize
    g = _parallel_chains_graph()
    nodes, external = _nodes_from(g, ["variables.out"])
    demoted = {n.variable for n in nodes if n.variable.name in ("p1", "p2")}
    order = linearize(nodes, demoted)
    runs = build_runs(order, demoted)
    assert [kind for kind, _ in runs] == ["python", "jit"]  # 2 runs, not 3
    # dependencies respected
    pos = {n.variable.qual_name(): i for i, n in enumerate(order)}
    for n in order:
        for inp in n.inputs:
            if inp.qual_name() in pos:
                assert pos[inp.qual_name()] < pos[n.variable.qual_name()]


def test_linearize_deterministic():
    from numbox.core.variable._kernel_partition import linearize
    g = _parallel_chains_graph()
    nodes, _ = _nodes_from(g, ["variables.out"])
    demoted = {n.variable for n in nodes if n.variable.name == "j1"}
    first = [n.variable.qual_name() for n in linearize(nodes, demoted)]
    second = [n.variable.qual_name() for n in linearize(nodes, demoted)]
    assert first == second


def test_liveness_and_runs():
    from numbox.core.variable._kernel_partition import build_runs, linearize, segment_liveness
    g = _parallel_chains_graph()
    nodes, external = _nodes_from(g, ["variables.out"])
    by_name = {n.variable.name: n.variable for n in nodes}
    demoted = {by_name["p1"], by_name["p2"]}
    order = linearize(nodes, demoted)
    runs = build_runs(order, demoted)
    required_vars = [by_name["out"]]
    jit_run = next(r for kind, r in runs if kind == "jit")
    live_in, live_out = segment_liveness(jit_run, set(external), required_vars, order)
    in_names = [v.qual_name() for v in live_in]
    out_names = [v.qual_name() for v in live_out]
    assert in_names == sorted(in_names) and out_names == sorted(out_names)
    assert "ext.x" in in_names and "variables.p2" in in_names
    assert out_names == ["variables.out"]  # only the required sink escapes


def test_plan_run_threads_values():
    from numbox.core.variable._kernel_partition import _JitStep, _Plan, _PyStep
    ax = Variable(name="x", source="ext")
    av = Variable(name="v", source="calc")
    aw = Variable(name="w", source="calc")
    plan = _Plan(
        steps=(
            _JitStep(dispatcher=lambda x: (x + 1.0,), in_vars=(ax,), out_vars=(av,)),
            _PyStep(var=aw, py_callable=lambda v: v * 10.0, in_vars=(av,)),
        ),
        external_vars=(ax,),
        output_vars=(aw, av),
    )
    assert plan.run((2.0,)) == (30.0, 3.0)


def test_partition_report_str_and_python_nodes():
    from numbox.core.variable._kernel_partition import PartitionReport, Segment
    rep = PartitionReport(mode="segmented", segments=(
        Segment(kind="jit", nodes=("calc.a",), inputs=("ext.x",), outputs=("calc.a",),
                source="def _kernel(x):\n    return (x,)\n", reasons={}),
        Segment(kind="python", nodes=("calc.b",), inputs=("calc.a",), outputs=("calc.b",),
                source=None, reasons={"calc.b": "TypingError: nope"}),
    ))
    assert rep.python_nodes == {"calc.b"}
    text = str(rep)
    assert "segmented" in text and "calc.b" in text and "TypingError: nope" in text


def test_fused_source_golden():
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u", "variables.a"], cache=False)
    expected = (
        "def _kernel(basket_y):\n"
        "    variables_x = f_variables_x(basket_y)  # 'variables.x' = f('basket.y')\n"
        "    variables_a = f_variables_a(variables_x)  # 'variables.a' = f('variables.x')\n"
        "    variables_b = f_variables_b(variables_x)  # 'variables.b' = f('variables.x')\n"
        "    variables_u = f_variables_u(variables_a, variables_b)"
        "  # 'variables.u' = f('variables.a', 'variables.b')\n"
        "    return (variables_u, variables_a,)\n"
    )
    assert ck.source == expected


def test_generate_segment_body():
    from numbox.core.variable.compile_kernel import _generate_segment_body
    g = _diamond_graph()
    compiled = g.compile(["variables.u", "variables.a"])
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    nodes = [n for n in compiled.ordered_nodes if n.variable not in external]
    run = nodes[:2]                       # variables.x, variables.a
    by_name = {n.variable.name: n.variable for n in nodes}
    live_in = (next(iter(external)),)     # basket.y
    live_out = (by_name["a"], by_name["x"])
    source, bindings, params, outputs = _generate_segment_body(run, live_in, live_out, idents)
    assert source == (
        "def _kernel(basket_y):\n"
        "    variables_x = f_variables_x(basket_y)  # 'variables.x' = f('basket.y')\n"
        "    variables_a = f_variables_a(variables_x)  # 'variables.a' = f('variables.x')\n"
        "    return (variables_a, variables_x,)\n"
    )
    assert [p[2] for p in params] == ["basket_y"]
    assert outputs == ["variables.a", "variables.x"]
    assert set(bindings) == {"f_variables_x", "f_variables_a"}


def test_generate_segment_body_empty_run():
    from numbox.core.variable.compile_kernel import _generate_segment_body
    g = _diamond_graph()
    compiled = g.compile(["variables.u"])
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    y = next(iter(external))
    source, bindings, params, outputs = _generate_segment_body([], (y,), (y,), idents)
    assert source == f"def _kernel({idents[y]}):\n    return ({idents[y]},)\n"
    assert bindings == {}


class _Opaque:
    """A value numba.typeof cannot type; arithmetic works in Python."""
    def __init__(self, v):
        self.v = v

    def __mul__(self, other):
        return _Opaque(self.v * other)

    def __add__(self, other):
        return _Opaque(self.v + other)

    def __sub__(self, other):
        return _Opaque(self.v - other)

    def __truediv__(self, other):
        return _Opaque(self.v / other)


def _bindings_by_var(graph, required):
    compiled = graph.compile(required)
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    _, bindings, _, _ = _generate_body(compiled, required, idents)
    by_var = {
        n.variable: bindings["f_" + idents[n.variable]]
        for n in compiled.ordered_nodes if n.variable not in external
    }
    return compiled, external, by_var


def test_discover_demotes_unjittable_and_keeps_values():
    from numbox.core.variable._kernel_partition import discover

    def uses_json(v):
        json.dumps({"k": 1})
        return v * 3.0

    g = Graph(
        variables_lists={"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": lambda x: x + 1.0},
            {"name": "b", "inputs": {"a": "calc"}, "formula": uses_json},
            {"name": "c", "inputs": {"b": "calc"}, "formula": lambda b: b - 0.5},
        ]},
        external_source_names=["ext"],
    )
    compiled, external, by_var = _bindings_by_var(g, ["calc.c"])
    ext_x = next(iter(external))
    values = {ext_x: 2.0}
    demoted = discover(compiled.ordered_nodes, external, values, by_var)
    reasons = {v.qual_name(): r for v, r in demoted.items()}
    assert set(reasons) == {"calc.b"}
    assert reasons["calc.b"].startswith("TypingError:")
    by_name = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    assert values[by_name["calc.c"]] == (2.0 + 1.0) * 3.0 - 0.5


def test_discover_demotes_object_chain():
    from numbox.core.variable._kernel_partition import discover
    g = Graph(
        variables_lists={"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": lambda x: _Opaque(x)},
            {"name": "b", "inputs": {"a": "calc"}, "formula": lambda a: a.v * 2.0},
            {"name": "c", "inputs": {"b": "calc"}, "formula": lambda b: b + 1.0},
        ]},
        external_source_names=["ext"],
    )
    compiled, external, by_var = _bindings_by_var(g, ["calc.c"])
    values = {next(iter(external)): 4.0}
    demoted = discover(compiled.ordered_nodes, external, values, by_var)
    reasons = {v.qual_name(): r for v, r in demoted.items()}
    assert set(reasons) == {"calc.a", "calc.b"}
    assert "is not numba-typeable" in reasons["calc.b"]
    by_name = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    assert values[by_name["calc.c"]] == 4.0 * 2.0 + 1.0


def test_discover_runtime_error_propagates():
    from numbox.core.variable._kernel_partition import discover
    g = Graph(
        variables_lists={"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": lambda x: x / 0},
        ]},
        external_source_names=["ext"],
    )
    compiled, external, by_var = _bindings_by_var(g, ["calc.a"])
    values = {next(iter(external)): 1}
    with pytest.raises(ZeroDivisionError):
        discover(compiled.ordered_nodes, external, values, by_var)


@pytest.mark.parametrize("make_formula", [
    lambda: cres(float64(float64))(lambda x: x * 5.0),
    lambda: cfunc(float64(float64))(lambda x: x * 5.0),
    lambda: vectorize(["float64(float64)"])(lambda x: x * 5.0),
], ids=["cres", "cfunc", "dufunc"])
def test_discover_exotic_via_shim(make_formula):
    from numbox.core.variable._kernel_partition import discover
    fn = make_formula()
    g = Graph(
        variables_lists={"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": fn},
        ]},
        external_source_names=["ext"],
    )
    compiled, external, by_var = _bindings_by_var(g, ["calc.a"])
    values = {next(iter(external)): 3.0}
    demoted = discover(compiled.ordered_nodes, external, values, by_var)
    assert demoted == {}
    by_name = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    assert values[by_name["calc.a"]] == 15.0


def test_discover_reason_informative_for_typing_failure():
    from numbox.core.variable._kernel_partition import discover

    def probe_fails(v):
        return v.hex() and 0.0 or 3.0

    g = Graph(
        variables_lists={"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": probe_fails},
        ]},
        external_source_names=["ext"],
    )
    compiled, external, by_var = _bindings_by_var(g, ["calc.a"])
    values = {next(iter(external)): 2.0}
    demoted = discover(compiled.ordered_nodes, external, values, by_var)
    reasons = {v.qual_name(): r for v, r in demoted.items()}
    assert set(reasons) == {"calc.a"}
    assert not reasons["calc.a"].endswith("(step: nopython frontend)")
    by_name = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    assert values[by_name["calc.a"]] == 3.0


def test_discover_exotic_untypeable_input_raises():
    from numbox.core.variable._kernel_partition import discover
    fn = cres(float64(float64))(lambda v: v * 2.0)
    g = Graph(
        variables_lists={"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": lambda x: _Opaque(x)},
            {"name": "b", "inputs": {"a": "calc"}, "formula": fn},
        ]},
        external_source_names=["ext"],
    )
    compiled, external, by_var = _bindings_by_var(g, ["calc.b"])
    values = {next(iter(external)): 1.0}
    with pytest.raises(TypeError, match="no Python fallback"):
        discover(compiled.ordered_nodes, external, values, by_var)


def test_fused_mode_resolution():
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u", "variables.a"], cache=False)
    assert ck.partition is None
    early = ck.kernel                      # resolver grabbed before first call
    assert not isinstance(early, Dispatcher)
    assert early(100) == (326.5, 126)
    assert ck.partition is not None
    assert ck.partition.mode == "fused"
    (seg,) = ck.partition.segments
    assert seg.kind == "jit"
    assert seg.nodes == ("variables.x", "variables.a", "variables.b", "variables.u")
    assert seg.inputs == ("basket.y",)
    assert seg.outputs == ("variables.u", "variables.a")
    assert seg.source == ck.source and seg.reasons == {}
    assert isinstance(ck.kernel, Dispatcher)
    assert ck.kernel(100) == (326.5, 126)
    assert early(100) == (326.5, 126)      # early ref still valid post-resolution


def _chain_graph_with_python_middle():
    def n3(v):
        json.dumps({"k": 1})
        return v * 3.0

    return Graph(
        variables_lists={"calc": [
            {"name": "n1", "inputs": {"x": "ext"}, "formula": lambda x: x + 1.0},
            {"name": "n2", "inputs": {"n1": "calc"}, "formula": lambda n1: n1 * 2.0},
            {"name": "n3", "inputs": {"n2": "calc"}, "formula": n3},
            {"name": "n4", "inputs": {"n3": "calc"}, "formula": lambda n3: n3 - 4.0},
            {"name": "n5", "inputs": {"n4": "calc"}, "formula": lambda n4: n4 / 2.0},
        ]},
        external_source_names=["ext"],
    )


def _compiled_graph_result(graph, required, external_values):
    compiled = graph.compile(required)
    values = Values()
    compiled.execute(external_values, values)
    by_qual = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    return {q: values.get(by_qual[q]).value for q in required}


def test_goykhman_example_two_segments():
    g = _chain_graph_with_python_middle()
    ck = compile_kernel(g, "calc.n5", cache=False)
    expected = _compiled_graph_result(
        _chain_graph_with_python_middle(), ["calc.n5"], {"ext": {"x": 7.0}}
    )
    assert ck.execute({"ext": {"x": 7.0}}) == expected          # call 1: warm-up
    assert ck.execute({"ext": {"x": 7.0}}) == expected          # call 2: plan
    rep = ck.partition
    assert rep.mode == "segmented"
    kinds = [(s.kind, s.nodes) for s in rep.segments]
    assert kinds == [
        ("jit", ("calc.n1", "calc.n2")),
        ("python", ("calc.n3",)),
        ("jit", ("calc.n4", "calc.n5")),
    ]
    assert rep.python_nodes == {"calc.n3"}
    assert rep.segments[1].reasons["calc.n3"]
    assert rep.segments[0].source is not None and rep.segments[1].source is None


def test_segmented_object_chain_groups_python_run():
    g = Graph(
        variables_lists={"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": lambda x: x * 2.0},
            {"name": "b", "inputs": {"a": "calc"}, "formula": lambda a: _Opaque(a)},
            {"name": "c", "inputs": {"b": "calc"}, "formula": lambda b: b.v + 1.0},
            {"name": "d", "inputs": {"c": "calc"}, "formula": lambda c: c * 10.0},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, "calc.d", cache=False)
    assert ck.kernel(3.0) == ((3.0 * 2.0 + 1.0) * 10.0,)
    assert ck.kernel(3.0) == ((3.0 * 2.0 + 1.0) * 10.0,)
    kinds = [(s.kind, s.nodes) for s in ck.partition.segments]
    assert ("python", ("calc.b", "calc.c")) in kinds


def test_segmented_all_python():
    def f(x):
        json.dumps({"k": 1})
        return x + 2.0

    g = Graph(
        variables_lists={"calc": [{"name": "a", "inputs": {"x": "ext"}, "formula": f}]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, "calc.a", cache=False)
    assert ck.kernel(1.0) == (3.0,)
    assert ck.kernel(1.0) == (3.0,)
    assert all(s.kind == "python" for s in ck.partition.segments)


def test_plan_replacement_on_new_signature():
    g = _chain_graph_with_python_middle()
    ck = compile_kernel(g, "calc.n5", cache=False)
    assert ck.kernel(7.0) == ((((7.0 + 1.0) * 2.0 * 3.0) - 4.0) / 2.0,)
    assert len([s for s in ck.partition.segments if s.kind == "jit"]) == 2
    out, = ck.kernel(_Opaque(7.0))          # breaks segment 1 -> re-discovery
    assert isinstance(out, _Opaque) and out.v == (((7.0 + 1.0) * 2.0 * 3.0) - 4.0) / 2.0
    assert [s.kind for s in ck.partition.segments] == ["python"]
    assert ck.kernel(7.0) == ((((7.0 + 1.0) * 2.0 * 3.0) - 4.0) / 2.0,)


_SEGMENT_CACHE_SCRIPT = textwrap.dedent("""
    import json, sys
    from numbox.core.variable.compile_kernel import compile_kernel
    from numbox.core.variable.variable import Graph

    def n3(v):
        json.dumps({"k": 1})
        return v * 3.0

    g = Graph(
        variables_lists={"calc": [
            {"name": "n1", "inputs": {"x": "ext"}, "formula": lambda x: x + 1.0},
            {"name": "n2", "inputs": {"n1": "calc"}, "formula": lambda n1: n1 * 2.0},
            {"name": "n3", "inputs": {"n2": "calc"}, "formula": n3},
            {"name": "n4", "inputs": {"n3": "calc"}, "formula": lambda n3: n3 - 4.0},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, "calc.n4", cache=True)
    out = ck.kernel(7.0)
    assert out == (((7.0 + 1.0) * 2.0 * 3.0) - 4.0,), out
    assert ck.partition.mode == "segmented", ck.partition.mode
    sys.exit(0)
""")


def test_segment_cache_survives_subprocess_roundtrip(tmp_path):
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "nbcache")}
    saved = None
    for attempt in ("save", "load"):
        proc = subprocess.run(
            [sys.executable, "-c", _SEGMENT_CACHE_SCRIPT],
            capture_output=True, text=True, env=env,
        )
        assert proc.returncode == 0, f"{attempt}: {proc.stderr}"
        nbc = list((tmp_path / "nbcache").rglob("*.nbc"))
        if attempt == "save":
            assert nbc, "segment compile produced no cache entries"
            saved = len(nbc)
        else:
            assert len(nbc) == saved, "second process recompiled instead of loading"


def test_shared_digest_identical_segments():
    def n3(v):
        json.dumps({"k": 1})
        return v + 0.0

    g = Graph(
        variables_lists={"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": lambda x: x * 2.0},
            {"name": "p", "inputs": {"a": "calc"}, "formula": n3},
            {"name": "b", "inputs": {"p": "calc"}, "formula": lambda x: x * 2.0},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, "calc.b", cache=False)
    assert ck.kernel(3.0) == (3.0 * 2.0 * 2.0,)
    assert ck.kernel(3.0) == (3.0 * 2.0 * 2.0,)
