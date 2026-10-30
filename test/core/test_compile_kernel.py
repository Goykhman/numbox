"""Tests of compile_kernel, which compiles a graph of variables into one numba function, the kernel.

The tests cover the identifiers and the source that compile_kernel generates, the fingerprints of the formulas, and the
digest that names a kernel. They cover the cache on disk: the precedence of the settings that enable it, a directory
that numbox cannot write, and kernels and segments that a second process loads. They cover the formulas that leave a
kernel uncached, the checks of the request and of the formulas, and the search for the formulas that numba cannot
compile, with the segments around them. The recompute tests compare the kernel's results with those of CompiledGraph.
Several tests run a probe script in a child process with a NUMBA_CACHE_DIR under the test's temporary directory, and
check the files that it writes there. Some run a second child, which loads what the first child cached.
test_compile_kernel_cache.py tests that numba never loads stale code for a kernel from its cache, and
test_params_jitability.py tests the kernels built from declared types.
"""
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
    _generate_body, _compile, compile_kernel, CompiledKernel,
)
from numbox.core.variable.utils import (
    _sanitize, _assign_identifiers, _wrap_formula,
)
from numbox.core.variable.variable import CompiledNode, Variable, Graph, Values
from numbox.core.variable._kernel_partition import compute_boundary, cone_liveness
from numbox.utils.highlevel import cres
from test.auxiliary_utils import assert_njit_cache_survives_subprocess_roundtrip


def test_sanitize_basic():
    """_sanitize turns a qualified name into an identifier that the kernel's source can use.

    Each run of characters that an identifier cannot contain becomes one underscore, and _sanitize adds the prefix
    ``v_`` to a name that is empty or starts with a digit.
    """
    assert _sanitize("variables.a") == "variables_a"
    assert _sanitize("first-name") == "first_name"
    assert _sanitize("3m") == "v_3m"
    assert _sanitize("a..b") == "a_b"
    assert _sanitize("") == "v_"


def test_assign_identifiers_unique_and_valid():
    """a_b.c and a.b_c both sanitize to a_b_c, and _assign_identifiers assigns them distinct valid identifiers."""
    v1 = Variable(name="c", source="a_b")
    v2 = Variable(name="b_c", source="a")
    idents = _assign_identifiers([v1, v2])
    assert idents[v1] != idents[v2]
    assert all(s.isidentifier() for s in idents.values())


def test_assign_identifiers_formula_prefix_collision():
    """No identifier equals the global through which the kernel calls another variable's formula.

    f.variables_x sanitizes to f_variables_x, the name of the global through which the kernel calls the formula of
    variables.x, so _assign_identifiers adds a suffix to it.
    """
    node = Variable(name="x", source="variables")
    clash = Variable(name="variables_x", source="f")
    idents = _assign_identifiers([node, clash])
    temps = set(idents.values())
    fgs = {"f_" + t for t in temps}
    assert temps.isdisjoint(fgs)


def test_assign_identifiers_deterministic():
    """_assign_identifiers returns the same identifiers on every call for the same variables in the same order.

    The identifiers are part of the kernel's source and so of its digest, which must be the same in every process.
    """
    v1 = Variable(name="c", source="a_b")
    v2 = Variable(name="b_c", source="a")
    assert _assign_identifiers([v1, v2]) == _assign_identifiers([v1, v2])


def test_assign_identifiers_invalid_char_and_leading_digit():
    """_assign_identifiers assigns distinct valid identifiers to two names that are not Python identifiers.

    first-name contains a hyphen, and 3m starts with a digit.
    """
    v1 = Variable(name="first-name", source="ext")
    v2 = Variable(name="3m", source="ext")
    idents = _assign_identifiers([v1, v2])
    assert all(s.isidentifier() for s in idents.values())
    assert idents[v1] != idents[v2]


def _diamond_graph():
    """Return the diamond graph, whose formulas are @njit dispatchers.

    x is 2 * basket.y, a is x - 74, b is x + 0.5, and u is a + b.
    """
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
    """_wrap_formula returns a dispatcher unchanged and decorates a plain Python function with @njit."""
    d = njit(lambda x: x)
    assert _wrap_formula(d) is d

    def plain(x):
        return x + 1
    assert isinstance(_wrap_formula(plain), Dispatcher)


def test_generate_body_shape():
    """_generate_body returns the parts of the diamond graph's kernel.

    The parameters are the external variables, the outputs follow the request, and the bindings hold one formula for
    each calculated variable.
    """
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
    """_generate_body raises ValueError for an empty request, an unknown name, or a variable without a formula."""
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
    """A request that names only an external variable produces a kernel without formulas, which returns its argument."""
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
    """The fingerprint of an @njit formula names its function and allows caching.

    A cres formula has neither py_func nor __wrapped__, so its text names the object by its repr and id, and the kernel
    compiles without a cache.
    """
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    @njit
    def named(x):
        return x + 41
    text, ok = _formula_fingerprint(named)
    assert ok
    assert named.py_func.__qualname__ in text

    wap = cres(float64(float64))(lambda x: x * 2.0)
    text2, ok2 = _formula_fingerprint(wap)
    assert not ok2
    assert isinstance(text2, str) and text2


def test_compile_runs():
    """_compile turns a kernel's source and bindings into a dispatcher that computes the kernel's result."""
    src = "def _kernel(y):\n    x = f_x(y)\n    return (x,)\n"
    bindings = {"f_x": njit(lambda y: 2 * y)}
    kernel = _compile(src, bindings, None, True)
    assert kernel(10) == (20,)


def test_compile_anchor_is_content_addressed(tmp_path, monkeypatch):
    """_compile writes an anchor file for a cached kernel, and a new anchor when a formula changes.

    The anchor's name holds the digest, so each version of the formulas has an anchor of its own.
    """
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
    """A cached kernel returns the correct result in a second process, which loads it from the first process's cache."""
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
    """Return the values of ``required``, by qualified name, that CompiledGraph computes from ``external_values``."""
    compiled = graph.compile(required)
    values = Values()
    compiled.execute(external_values, values)
    by_qual = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    return {q: values.get(by_qual[q]).value for q in required}


def test_compile_kernel_matches_pure_python_diamond():
    """A kernel of the diamond graph computes what CompiledGraph computes, through execute and through the kernel.

    params lists the external variables, and outputs keeps the order of the request.
    """
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
    """compile_kernel takes one qualified name as a string and returns a result with one output."""
    g = _diamond_graph()
    ck = compile_kernel(g, "variables.u")
    assert ck.outputs == ["variables.u"]
    assert ck.execute({"basket": {"y": 100}}) == {"variables.u": 326.5}


def test_compile_kernel_auto_specialization():
    """One kernel computes the same result for an int and a float value of basket.y.

    numba compiles a version of the kernel for each argument type.
    """
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u"])
    assert ck.execute({"basket": {"y": 100}})["variables.u"] == 326.5
    assert ck.execute({"basket": {"y": 100.0}})["variables.u"] == 326.5


def test_compile_kernel_missing_external_raises():
    """execute raises KeyError, naming basket.y, when the dictionary lacks the value of basket.y."""
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u"])
    with pytest.raises(KeyError) as exc:
        ck.execute({"basket": {}})
    assert "basket.y" in str(exc.value)


def test_compile_kernel_missing_external_source_raises():
    """execute raises KeyError, naming basket.y, when the dictionary lacks the whole source basket."""
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u"])
    with pytest.raises(KeyError) as exc:
        ck.execute({})
    assert "basket.y" in str(exc.value)


def test_identifier_collision_graph_runs():
    """A kernel computes two variables whose qualified names, a_b.c and a.b_c, sanitize to the same string."""
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
    """A kernel runs with first-name, an external name that is not a Python identifier."""
    g = Graph(
        variables_lists={"variables": [
            {"name": "out", "inputs": {"first-name": "ext"}, "formula": njit(lambda v: v * 2)},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["variables.out"])
    assert ck.execute({"ext": {"first-name": 5}}) == {"variables.out": 10}


def test_constant_and_array_formulas():
    """A kernel computes a formula without inputs, the constant 7.0, and a formula that returns an array."""
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
    """compile_kernel compiles a plain Python formula, which numbox decorates with @njit."""
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
    """A kernel calls a cres formula, a first-class function that numba compiled for one signature."""
    sub = cres(float64(float64, float64))(lambda a, b: a - b)
    g = Graph(
        variables_lists={"variables": [
            {"name": "u", "inputs": {"p": "ext", "q": "ext"}, "formula": sub},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["variables.u"])
    assert ck.execute({"ext": {"p": 1.5, "q": 2.0}}) == {"variables.u": -0.5}


def test_a_raising_cres_formula_propagates():
    """An exception in a cres formula propagates to the caller of execute, and the next call computes its result.

    cres is a general utility, so compile_kernel relies on it to propagate exceptions, as Work does. If the exception
    did not propagate, the kernel would return a zero-filled result and no error.
    """
    def blow_up(a, b):
        if a > b:
            raise ValueError("formula boom")
        return a - b

    formula = cres(float64(float64, float64))(blow_up)
    g = Graph(
        variables_lists={"variables": [
            {"name": "u", "inputs": {"p": "ext", "q": "ext"}, "formula": formula},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["variables.u"])

    with pytest.raises(ValueError, match="formula boom"):
        ck.execute({"ext": {"p": 3.0, "q": 2.0}})

    assert ck.execute({"ext": {"p": 1.5, "q": 2.0}}) == {"variables.u": -0.5}


def test_non_jittable_formula_demotes_at_first_call():
    """compile_kernel accepts a formula that numba cannot compile, because numba compiles nothing before the first call.

    The first call runs the formula in Python, and the partition records numba's error as the reason.
    """
    def bad(y):
        return json.dumps({"y": y}) and y + 100.0
    g = Graph(
        variables_lists={"variables": [
            {"name": "b", "inputs": {"y": "ext"}, "formula": bad},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["variables.b"])
    assert ck.execute({"ext": {"y": 1.0}}) == {"variables.b": bad(1.0)}
    assert ck.partition.mode == "segmented"
    assert "variables.b" in ck.partition.python_nodes
    reasons = {q: r for s in ck.partition.segments for q, r in s.reasons.items()}
    assert reasons["variables.b"]


def test_cache_no_skeleton_collision(tmp_path):
    """Two cached kernels with one source and formulas that differ only in a closure value do not share code.

    The same lambda closes over 10 in one kernel and over 1000 in the other. The probe runs twice, so the second run
    reads both kernels from the cache that the first run wrote.
    """
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
    """_assign_identifiers adds a suffix to an identifier that would be a Python keyword.

    The qualified name .for sanitizes to the keyword for.
    """
    v = Variable(name="for", source="")
    ident = _assign_identifiers([v])[v]
    import keyword as _kw
    assert not _kw.iskeyword(ident)
    assert ident.startswith("for_")


def test_compile_kernel_keyword_node_name():
    """A kernel runs for the variable _.for, whose qualified name sanitizes to the keyword for."""
    g = Graph(
        variables_lists={"_": [
            {"name": "for", "inputs": {"y": "e"}, "formula": njit(lambda y: y + 1)},
        ]},
        external_source_names=["e"],
    )
    ck = compile_kernel(g, ["_.for"])
    assert ck.execute({"e": {"y": 5}}) == {"_.for": 6}


def test_compile_kernel_newline_in_name_is_safe():
    """An external name with a newline leaves the kernel's source valid.

    The comment after each line of the source quotes the names with repr, so the newline appears as an escape.
    """
    g = Graph(
        variables_lists={"v": [
            {"name": "o", "inputs": {"a\nx": "e"}, "formula": njit(lambda a: a + 1)},
        ]},
        external_source_names=["e"],
    )
    ck = compile_kernel(g, ["v.o"])
    # A raw newline would end the comment, and the rest of the name would become a line of code.
    assert "e.a\nx" not in ck.source
    assert ck.execute({"e": {"a\nx": 10}}) == {"v.o": 11}


def test_compile_kernel_duplicate_required_raises():
    """compile_kernel refuses a request that names a variable twice, because the kernel's results are positional."""
    g = _diamond_graph()
    with pytest.raises(ValueError, match="duplicate entries"):
        compile_kernel(g, ["variables.u", "variables.u"])


def test_fingerprint_fallback_is_per_object():
    """Two callables without a fingerprint and with the same repr have different texts, because each text holds an id.

    A text built from the repr alone would assign one digest to two different formulas.
    """
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
    """The garbage collector frees a kernel's dispatcher once its CompiledKernel is deleted.

    numbox keeps no other reference to the kernels that it compiles, so a program that compiles many kernels does not
    keep every one alive.
    """
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


# _CACHE_PROBE compiles a kernel of one formula with the keyword arguments in argv[1] and prints its result for 1.0.
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
    """Run _CACHE_PROBE in a new process with ``kwargs_src`` as the keyword arguments of compile_kernel.

    The process's NUMBA_CACHE_DIR is ``tmp_path / "nbcache"``, and ``extra_env`` adds environment variables. The
    function asserts that the probe prints 2.0, and returns the files under the cache directory and the probe's stderr.
    """
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
    """``cache=False`` leaves NUMBA_CACHE_DIR without a file: no anchor and no numba cache file."""
    files, _ = _run_cache_probe(tmp_path, "{'cache': False}")
    assert files == []


def test_cache_precedence_env_knob(tmp_path):
    """``NUMBOX_JIT_OPTIONS='{"cache": false}'`` disables the kernel's cache.

    The setting applies because neither the cache argument nor jit_options sets ``cache``.
    """
    files, _ = _run_cache_probe(
        tmp_path, "{}", extra_env={"NUMBOX_JIT_OPTIONS": '{"cache": false}'})
    assert files == []


def test_cache_precedence_jit_options(tmp_path):
    """The ``cache`` option of jit_options disables the kernel's cache when the cache argument is None."""
    files, _ = _run_cache_probe(tmp_path, "{'jit_options': {'cache': False}}")
    assert files == []


def test_cache_precedence_param_wins(tmp_path):
    """The cache argument takes precedence over the cache option of jit_options, so ``cache=True`` caches the kernel."""
    files, _ = _run_cache_probe(
        tmp_path, "{'cache': True, 'jit_options': {'cache': False}}")
    assert files != []


@pytest.mark.skipif(sys.platform == "win32", reason="chmod-based read-only dir is POSIX-only")
def test_readonly_cache_dir_degrades_gracefully(tmp_path):
    """A NUMBA_CACHE_DIR that numbox cannot write makes the kernel compile without a cache, after a RuntimeWarning."""
    cache_dir = tmp_path / "nbcache"
    cache_dir.mkdir()
    cache_dir.chmod(0o500)
    try:
        files, stderr = _run_cache_probe(tmp_path, "{}")
        assert files == []
        assert "compiles without a cache" in stderr
    finally:
        cache_dir.chmod(0o700)


def test_fingerprint_same_line_lambdas_distinct():
    """Two lambdas on one line, y * 10.0 and y * 1000.0, have different fingerprints.

    numba's own key for a function covers its bytecode but not its constants, so it cannot distinguish the two.
    """
    from numbox.core.variable.compile_kernel import _formula_fingerprint
    f10, f1000 = (lambda y: y * 10.0), (lambda y: y * 1000.0)
    fp_a, ok_a = _formula_fingerprint(f10)
    fp_b, ok_b = _formula_fingerprint(f1000)
    assert ok_a and ok_b
    assert fp_a != fp_b


def test_fingerprint_covers_globals_helpers_defaults():
    """The fingerprint changes with the value of a global, the code of a helper and the value of a default argument.

    A formula and three variants, each of which changes one of the three, have four fingerprints.
    """
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
    """The fingerprint of a closure over a 2000-element array holds the array's bytes.

    Two arrays that differ in one element have two fingerprints, although NumPy abbreviates both to the same repr.
    """
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
    """Two sets of the same strings, built in different orders, have one fingerprint, which sorts the elements.

    The iteration order of a set of strings changes from process to process with string hash randomization.
    """
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    def factory(s):
        return lambda x: x if x in s else -x

    s1 = {"alpha", "beta", "gamma"}
    s2 = set(reversed(sorted(s1)))
    assert _formula_fingerprint(factory(s1))[0] == _formula_fingerprint(factory(s2))[0]


def test_fingerprint_recursive_helpers_terminate():
    """The fingerprint of a formula that calls two mutually recursive helpers terminates.

    The walk folds a function that it has already entered by its qualified name, as ``recursive(<name>)``.
    """
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
    """A formula over a closure value without a fingerprint, and a cres formula, each leave the kernel uncached.

    For both, _formula_fingerprint returns a text and the flag False. The text for the closure names the formula by its
    id.
    """
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
    """The fingerprint of a functools.wraps wrapper is that of the wrapper, not of the function in ``__wrapped__``.

    Two wrappers of one function with different rates compute different values, so their digests must differ.
    """
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
    """A 0-d and a 1-d array with the same bytes have two fingerprints, because the fingerprint holds the shape.

    numba compiles a 0-d and a 1-d array into different code.
    """
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    def factory(a):
        return lambda x: x + a.sum()

    fp0, ok0 = _formula_fingerprint(factory(np.array(3.5)))
    fp1, ok1 = _formula_fingerprint(factory(np.array([3.5])))
    assert ok0 and ok1
    assert fp0 != fp1


def test_fingerprint_deep_nesting_downgrades_not_crashes():
    """A closure value nested in 5000 levels of lists leaves the kernel uncached instead of raising RecursionError.

    _formula_fingerprint catches the RecursionError and returns the text that names the formula by its id.
    """
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    deep = [0.0]
    for _ in range(5000):
        deep = [deep]

    def factory(d):
        return lambda x: x if d is None else x

    fp, ok = _formula_fingerprint(factory(deep))
    assert not ok and " @" in fp


# _DIGEST_PROBE imports formulas_mod from the directory given and prints the result of a kernel of its formula f.
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
    """A kernel compiles again when a global that its formula reads changes between two processes.

    The second process must print 30.0 for SCALE = 3.0, not the 20.0 of the kernel that the first process cached.
    """
    moddir = tmp_path / "mods"
    moddir.mkdir()
    mod = moddir / "formulas_mod.py"
    runner = tmp_path / "run.py"
    runner.write_text(textwrap.dedent(_DIGEST_PROBE.format(moddir=str(moddir))))
    # PYTHONDONTWRITEBYTECODE prevents the first process from writing a .pyc of formulas_mod. Python accepts a .pyc
    # whose recorded source size and modification time, in whole seconds, match the source. The two versions of the
    # module have the same size and can be written in the same second, so the second process would load SCALE = 2.0.
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
    """Kernels whose formulas close over 2000-element arrays that differ in one element compute their own results."""
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
    """Kernels of two lambdas on one line, y * 10.0 and y * 1000.0, compute their own results."""
    f10, f1000 = (lambda y: y * 10.0), (lambda y: y * 1000.0)
    results = []
    for f in (f10, f1000):
        g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
        results.append(compile_kernel(g, "calc.y").execute({"ext": {"x": 1.0}})["calc.y"])
    assert results == [10.0, 1000.0]


def test_digest_includes_jit_flags():
    """Two kernels of one formula, under two error models, each behave according to their own error model.

    Under numba's default error model, "python", 1.0 / 0.0 raises ZeroDivisionError, and under "numpy" it returns inf.
    The digest holds the jit options, so the two kernels compile under two names.
    """
    def f(x):
        return 1.0 / x

    def build():
        return Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])

    ck_default = compile_kernel(build(), "calc.y")
    with pytest.raises(ZeroDivisionError):
        ck_default.execute({"ext": {"x": 0.0}})
    ck_numpy = compile_kernel(build(), "calc.y", jit_options={"error_model": "numpy"})
    assert ck_numpy.execute({"ext": {"x": 0.0}})["calc.y"] == np.inf


def test_digest_cres_kernel_uncached_and_quiet(tmp_path):
    """A kernel with a cres formula compiles without a cache and without a warning.

    The probe turns warnings into errors, and its NUMBA_CACHE_DIR stays empty.
    """
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
    """A cres formula leaves the whole kernel uncached although the kernel's other formula has a fingerprint.

    The kernel computes both values, and the probe's NUMBA_CACHE_DIR stays empty. The test uses a cres formula because
    another formula without a fingerprint, such as a lambda that closes over an object, would not compile.
    """
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
    """A kernel with a @vectorize and a @cfunc formula computes its result without a warning and stays uncached.

    The kernel calls the @cfunc at its address, which differs from process to process, so numba cannot cache the
    kernel. The probe's NUMBA_CACHE_DIR therefore holds no anchor and no cache file.
    """
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
    """A kernel with one @vectorize formula is cached without a warning, and a second process loads it unchanged.

    After the first run, the cache holds at least one index file. The second run shares the cache and rewrites no .nbc
    or .nbi file. Both runs turn warnings into errors.
    """
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
    """One function has three different fingerprints as a plain formula, as a @vectorize function and as a @cfunc.

    Only the @cfunc leaves the kernel uncached.
    """
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
    """compile_kernel refuses a formula that is not callable, before any call, with a TypeError naming the variable."""
    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": "lambda y: y"}]}, ["ext"])
    with pytest.raises(TypeError, match=r"calc\.y.*not callable"):
        compile_kernel(g, "calc.y")


def test_arity_mismatch_rejected_eagerly():
    """compile_kernel refuses a formula that cannot take its two inputs as positional arguments."""
    def one_arg(x):
        return x

    g = Graph({"calc": [{"name": "y", "inputs": {"a": "ext", "b": "ext"}, "formula": one_arg}]}, ["ext"])
    with pytest.raises(ValueError, match=r"calc\.y.*2 declared input"):
        compile_kernel(g, "calc.y")


def test_kwonly_formula_rejected_eagerly():
    """compile_kernel refuses a formula with a keyword-only parameter, because the kernel passes inputs by position."""
    def kw_only(*, y):
        return y

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": kw_only}]}, ["ext"])
    with pytest.raises(ValueError, match=r"calc\.y"):
        compile_kernel(g, "calc.y")


def test_varargs_formula_still_accepted():
    """compile_kernel accepts a formula that takes its inputs as ``*vals``, and numba compiles it."""
    def star(*vals):
        return vals[0] + vals[1]

    g = Graph({"calc": [{"name": "y", "inputs": {"a": "ext", "b": "ext"}, "formula": star}]}, ["ext"])
    out = compile_kernel(g, "calc.y").execute({"ext": {"a": 1.0, "b": 2.0}})
    assert out == {"calc.y": 3.0}


def test_required_validation_messages():
    """compile_kernel refuses a bad request with an error that names the problem.

    It raises TypeError for an entry that is not a string, and ValueError for a name that is not qualified, a name that
    the graph cannot resolve, and an input in an unknown namespace.
    """
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


def test_external_typo_compiles_to_phantom_input():
    """A misspelled external name becomes one more parameter of the kernel instead of an error.

    An External namespace creates any name that is looked up in it. execute then raises KeyError, "Missing external
    value", when the dictionary holds only the correct name.
    """
    def f(x):
        return x

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    ck = compile_kernel(g, ["ext.tpyo"])
    assert ck.params == ["ext.tpyo"]
    assert ck.execute({"ext": {"tpyo": 5.0}}) == {"ext.tpyo": 5.0}
    with pytest.raises(KeyError, match="Missing external value"):
        ck.execute({"ext": {"x": 5.0}})


def test_deep_chain_recursion_error_is_contextual():
    """compile_kernel raises RecursionError for a chain deeper than the recursion limit.

    The error's message suggests raising the limit.
    """
    depth = sys.getrecursionlimit() + 100

    def step(x):
        return x

    specs = [{"name": "n0", "inputs": {"x": "ext"}, "formula": step}]
    specs += [
        {"name": f"n{i}", "inputs": {f"n{i - 1}": "calc"}, "formula": step}
        for i in range(1, depth)
    ]
    g = Graph({"calc": specs}, ["ext"])
    with pytest.raises(RecursionError, match="recursion limit"):
        compile_kernel(g, f"calc.n{depth - 1}")


def test_compile_kernel_cache_save_side(tmp_path):
    """A kernel of two formulas is cached in a first process and loaded by a second, which rewrites no cache file."""
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
    """A request that names only one external variable produces a kernel that returns the external value."""
    g = Graph({"calc": []}, ["ext"])
    ck = compile_kernel(g, "ext.x")
    assert ck.params == ["ext.x"]
    assert ck.execute({"ext": {"x": 5.5}}) == {"ext.x": 5.5}


def test_mixed_outputs_end_to_end():
    """A kernel returns a calculated and an external variable in the order of the request."""
    def f(x):
        return x * 10.0

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    ck = compile_kernel(g, ["calc.y", "ext.x"])
    assert ck.outputs == ["calc.y", "ext.x"]
    assert ck.execute({"ext": {"x": 2.0}}) == {"calc.y": 20.0, "ext.x": 2.0}


def test_fingerprint_object_array_is_unfingerprintable():
    """A closure over an object array has no fingerprint, because the array's bytes are the addresses of its objects."""
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    def factory(a):
        return lambda x: x if a is None else x

    arr = np.array([object(), object()], dtype=object)
    fp, ok = _formula_fingerprint(factory(arr))
    assert not ok and " @" in fp


def test_object_array_closure_kernel_uncached(tmp_path):
    """A formula that closes over an object array runs in Python, and the kernel writes nothing to the cache.

    The array has no fingerprint, and numba cannot compile a formula that reads it, so the search runs the formula in
    Python. The partition then has no compiled segment, and the probe's NUMBA_CACHE_DIR stays empty.
    """
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
    """NumPy scalars in a closure have fingerprints that allow caching.

    np.int64(5), np.int64(6) and np.float32(2.5) have three fingerprints, because each holds the scalar's type and
    bytes.
    """
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
    """A dispatcher whose target options hold a value without a fingerprint leaves the kernel uncached."""
    from numba import njit
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    @njit
    def d(x):
        return x + 1.0

    d.targetoptions["_review_probe"] = object()  # object() has no fingerprint.
    fp, ok = _formula_fingerprint(d)
    assert not ok and " @" in fp


def test_compile_self_referential_jit_option_no_recursionerror(tmp_path):
    """A jit option whose value contains itself does not make the digest raise RecursionError.

    numbox finds no canonical form for the option, so it warns and compiles without a cache. numba then raises its
    own KeyError for the unknown option at the first call.
    """
    loop = []
    loop.append(loop)

    def f(x):
        return x + 1.0

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    with pytest.raises(Exception) as ei:
        compile_kernel(g, "calc.y", jit_options={"_review_loop": loop}).execute({"ext": {"x": 1.0}})
    assert not isinstance(ei.value, RecursionError)


def test_fingerprint_raising_repr_callable_degrades_not_crashes():
    """_formula_fingerprint returns a text for a callable formula whose __repr__ raises, instead of an error.

    The text names the formula as repr-failed.
    """
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    class BoomRepr:
        def __call__(self, x):
            return x + 1.0

        def __repr__(self):
            raise RuntimeError("repr boom")

    fp, ok = _formula_fingerprint(BoomRepr())
    assert not ok
    assert isinstance(fp, str) and "repr-failed" in fp


def _nodes_from(graph, required):
    """Return the calculated nodes of ``required`` in order, and the set of its external variables."""
    compiled = graph.compile(required)
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    return [n for n in compiled.ordered_nodes if n.variable not in external], external


def _parallel_chains_graph():
    """Return two chains from ext.x, p1 to p2 and j1 to j2, and the variable out, which adds p2 and j2.

    Every formula can compile; the tests choose the formulas that run in Python.
    """
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
    """linearize orders the parallel chains as one Python run and one compiled run, with each node after its inputs.

    An order that started with the compiled chain would need three runs, because out must follow p2.
    """
    from numbox.core.variable._kernel_partition import build_runs, linearize
    g = _parallel_chains_graph()
    nodes, external = _nodes_from(g, ["variables.out"])
    demoted = {n.variable for n in nodes if n.variable.name in ("p1", "p2")}
    order = linearize(nodes, demoted)
    runs = build_runs(order, demoted)
    assert [kind for kind, _ in runs] == ["python", "jit"]
    pos = {n.variable.qual_name(): i for i, n in enumerate(order)}
    for n in order:
        for inp in n.inputs:
            if inp.qual_name() in pos:
                assert pos[inp.qual_name()] < pos[n.variable.qual_name()]


def test_linearize_deterministic():
    """linearize returns the same order on two calls, because it breaks ties by qualified name.

    The order determines the segments and their sources, and therefore the digests in the segments' names.
    """
    from numbox.core.variable._kernel_partition import linearize
    g = _parallel_chains_graph()
    nodes, _ = _nodes_from(g, ["variables.out"])
    demoted = {n.variable for n in nodes if n.variable.name == "j1"}
    first = [n.variable.qual_name() for n in linearize(nodes, demoted)]
    second = [n.variable.qual_name() for n in linearize(nodes, demoted)]
    assert first == second


def test_liveness_and_runs():
    """The compiled run of the parallel chains reads ext.x and variables.p2, sorted, and returns only variables.out.

    variables.out is requested, and no later step reads j1 or j2.
    """
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
    assert out_names == ["variables.out"]


def test_plan_run_threads_values():
    """_Plan.run passes each step's outputs to the later steps and returns the outputs in the requested order."""
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
    """A partition report lists its Python formulas, and its text names the mode, the variables and each reason."""
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
    """compile_kernel generates exactly this source for the diamond graph.

    The source is part of the digest, so a change to the generated text changes every kernel's name, and numba
    compiles every cached kernel again.
    """
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
    """_generate_segment_body returns the source, parameters, outputs and bindings of a run of two formulas.

    The source returns the outputs in the order given, which the callers sort by qualified name.
    """
    from numbox.core.variable.compile_kernel import _generate_segment_body
    g = _diamond_graph()
    compiled = g.compile(["variables.u", "variables.a"])
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    nodes = [n for n in compiled.ordered_nodes if n.variable not in external]
    # The run holds variables.x and variables.a, and its one input is basket.y.
    run = nodes[:2]
    by_name = {n.variable.name: n.variable for n in nodes}
    live_in = (next(iter(external)),)
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
    """For a run without formulas, _generate_segment_body returns a source that returns its input, and no bindings."""
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
    """A value that numba cannot type, with arithmetic that works in Python."""
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
    """Return the compiled graph of ``required``, its external variables, and the formula of each calculated variable.

    The formulas are the dispatchers that compile_kernel makes from them.
    """
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
    """The search finds the formula that numba cannot compile, runs it in Python, and records numba's TypingError.

    The search still computes every value.
    """
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
    """The search runs calc.a and calc.b in Python, and computes calc.c with numba.

    numba cannot compile calc.a, which creates an _Opaque, and cannot type the input of calc.b, so the reason recorded
    for calc.b is that the input is not numba-typeable.
    """
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
    """An exception that a compiled formula raises during the search propagates; the formula does not run in Python."""
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
    """The search calls a cres, @cfunc or @vectorize formula through a one-line @njit function and computes its value.

    Such a formula has no Python version, so it never runs in Python.
    """
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
    """The reason for a formula that numba cannot compile is the first informative line of numba's message.

    The search skips the line about the nopython pipeline, with which every typing error begins.
    """
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
    """A cres formula whose input numba cannot type makes the search raise TypeError: it has no Python fallback."""
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
    """After the first call, the kernel is fused: partition is a fused report, and kernel is numba's dispatcher.

    A reference to kernel taken before the first call still computes the result afterwards.
    """
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u", "variables.a"], cache=False)
    assert ck.partition is None
    early = ck.kernel
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
    assert early(100) == (326.5, 126)


def _chain_graph_with_python_middle():
    """Return the chain calc.n1 to calc.n5 from ext.x, with a Python formula in the middle.

    calc.n3 calls json.dumps, which numba cannot compile.
    """
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
    """Return the values of ``required``, by qualified name, that CompiledGraph computes from ``external_values``."""
    compiled = graph.compile(required)
    values = Values()
    compiled.execute(external_values, values)
    by_qual = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    return {q: values.get(by_qual[q]).value for q in required}


def test_python_middle_node_two_segments():
    """A chain with a Python formula in the middle runs as two compiled segments around it.

    The first call searches, and the second runs the plan. Both compute what CompiledGraph computes.
    """
    g = _chain_graph_with_python_middle()
    ck = compile_kernel(g, "calc.n5", cache=False)
    expected = _compiled_graph_result(
        _chain_graph_with_python_middle(), ["calc.n5"], {"ext": {"x": 7.0}}
    )
    assert ck.execute({"ext": {"x": 7.0}}) == expected
    assert ck.execute({"ext": {"x": 7.0}}) == expected
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
    """Two consecutive formulas that run in Python form one Python run.

    numba cannot compile calc.b, whose result is an _Opaque, and cannot type that result as the input of calc.c.
    """
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


def test_segmented_python_run_inputs_exclude_values_made_inside_the_run():
    """The report of a Python run lists as inputs only the values that the run does not compute itself.

    calc.c reads calc.b, a value of the run, and ext.x, a value from outside it.
    """
    g = Graph(
        variables_lists={"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": lambda x: x * 2.0},
            {"name": "b", "inputs": {"a": "calc"}, "formula": lambda a: _Opaque(a)},
            {"name": "c", "inputs": {"b": "calc", "x": "ext"}, "formula": lambda b, x: b.v + x},
            {"name": "d", "inputs": {"c": "calc"}, "formula": lambda c: c * 10.0},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, "calc.d", cache=False)
    assert ck.kernel(3.0) == ((3.0 * 2.0 + 3.0) * 10.0,)
    assert ck.kernel(3.0) == ((3.0 * 2.0 + 3.0) * 10.0,)
    py = [s for s in ck.partition.segments if s.kind == "python"]
    assert [(s.nodes, s.inputs, s.outputs) for s in py] == [
        (("calc.b", "calc.c"), ("calc.a", "ext.x"), ("calc.b", "calc.c")),
    ]


def test_segmented_jit_run_inputs_exclude_values_made_inside_the_run():
    """Each compiled segment's report lists as inputs only the values that the segment does not compute itself."""
    ck = compile_kernel(_chain_graph_with_python_middle(), "calc.n5", cache=False)
    assert ck.kernel(7.0) == (((7.0 + 1.0) * 2.0 * 3.0 - 4.0) / 2.0,)
    assert ck.kernel(7.0) == (((7.0 + 1.0) * 2.0 * 3.0 - 4.0) / 2.0,)
    jit = [s for s in ck.partition.segments if s.kind == "jit"]
    assert [(s.nodes, s.inputs, s.outputs) for s in jit] == [
        (("calc.n1", "calc.n2"), ("ext.x",), ("calc.n2",)),
        (("calc.n4", "calc.n5"), ("calc.n3",), ("calc.n5",)),
    ]


def test_segmented_all_python():
    """A graph whose only formula runs in Python has a partition of Python segments alone."""
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
    """A segmented kernel searches again when a call's argument makes a compiled segment fail.

    For an _Opaque argument, the new search runs every formula in Python, and the resulting plan of Python formulas
    also computes the result for a float.
    """
    g = _chain_graph_with_python_middle()
    ck = compile_kernel(g, "calc.n5", cache=False)
    assert ck.kernel(7.0) == ((((7.0 + 1.0) * 2.0 * 3.0) - 4.0) / 2.0,)
    assert len([s for s in ck.partition.segments if s.kind == "jit"]) == 2
    # numba cannot type the _Opaque, so the first segment fails and the kernel searches again.
    out, = ck.kernel(_Opaque(7.0))
    assert isinstance(out, _Opaque) and out.v == (((7.0 + 1.0) * 2.0 * 3.0) - 4.0) / 2.0
    assert [s.kind for s in ck.partition.segments] == ["python"]
    assert ck.kernel(7.0) == ((((7.0 + 1.0) * 2.0 * 3.0) - 4.0) / 2.0,)


# _SEGMENT_CACHE_SCRIPT compiles a cached kernel whose formula calc.n3 runs in Python, and prints its result for 7.0.
_SEGMENT_CACHE_SCRIPT = """
    import json
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
    print(out[0])
"""


def test_segment_cache_survives_subprocess_roundtrip(tmp_path):
    """A segmented kernel's segments are cached: a second process loads them and rewrites no cache file.

    The second process repeats the search, which compiles each formula without a cache.
    """
    expected = (((7.0 + 1.0) * 2.0 * 3.0) - 4.0)
    assert_njit_cache_survives_subprocess_roundtrip(
        tmp_path,
        _SEGMENT_CACHE_SCRIPT,
        [repr(expected)],
    )


def test_identical_shape_segments_compile_and_run():
    """A segmented kernel runs when its two compiled segments have the same structure and formulas of the same code.

    The two formulas, ``lambda x: x * 2.0``, have one fingerprint, so only the segments' sources make their digests
    differ.
    """
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


def test_linearize_run_count_is_optimal_bruteforce():
    """linearize finds the fewest runs on 300 random graphs of 2 to 6 nodes.

    The test compares its number of runs with the minimum over every order that respects the dependencies.
    """
    import itertools
    import random

    from numbox.core.variable._kernel_partition import build_runs, linearize

    rng = random.Random(11)
    for _ in range(300):
        n = rng.randint(2, 6)
        edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < 0.3]
        colors = [rng.randint(0, 1) for _ in range(n)]
        vars_ = [Variable(name=f"v{i}", source="s") for i in range(n)]
        nodes = [
            CompiledNode(variable=vars_[i], inputs=[vars_[u] for u, v in edges if v == i])
            for i in range(n)
        ]
        demoted = {vars_[i] for i in range(n) if colors[i] == 1}
        greedy_runs = len(build_runs(linearize(nodes, demoted), demoted))
        best = None
        for perm in itertools.permutations(range(n)):
            pos = {v: i for i, v in enumerate(perm)}
            if any(pos[u] > pos[v] for u, v in edges):
                continue
            runs = 1 + sum(1 for a, b in zip(perm, perm[1:]) if colors[a] != colors[b])
            best = runs if best is None else min(best, runs)
        assert greedy_runs == best, (edges, colors, greedy_runs, best)


def test_discovery_path_honors_jit_options():
    """The search compiles each formula under the kernel's jit options, as the segments do.

    An external value that numba cannot type makes the first call search. Under ``error_model="numpy"``, 1.0 / 0.0
    returns inf in the search and in the segment that the second call runs; under numba's default it would raise.
    """
    class Opaque:
        pass

    specs = [
        {"name": "p", "inputs": {"opaque": "ext"}, "formula": lambda o: 0.0},
        {"name": "r", "inputs": {"x": "ext", "y": "ext"}, "formula": lambda a, b: a / b},
    ]
    g = Graph({"calc": specs}, ["ext"])
    ck = compile_kernel(g, ["calc.r", "calc.p"], jit_options={"error_model": "numpy"})

    out = ck.execute({"ext": {"x": 1.0, "y": 0.0, "opaque": Opaque()}})
    assert out["calc.r"] == float("inf")
    assert ck.partition.mode == "segmented"
    # The second call runs the plan of segments.
    out2 = ck.execute({"ext": {"x": 1.0, "y": 0.0, "opaque": Opaque()}})
    assert out2["calc.r"] == float("inf")


def _compiled_parts(g, req):
    """Return the compiled graph of ``req`` and its variables by qualified name, then two sets of variables.

    The two sets hold the external variables and the requested variables.
    """
    cg = g.compile(req)
    by_qual = {n.variable.qual_name(): n.variable for n in cg.ordered_nodes}
    sources = {v for vs in cg.required_external_variables.values() for v in vs.values()}
    required = {by_qual[q] for q in req}
    return cg, by_qual, sources, required


def _boundary_reference(cg, sources, required):
    """Return the boundary by its definition, as a reference for compute_boundary.

    The boundary holds the requested variables and every variable with a reader that a change can reach without
    passing through the variable.
    """
    by_var = {n.variable: n for n in cg.ordered_nodes}
    succ = {}
    for n in cg.ordered_nodes:
        for inp in n.inputs:
            if inp in by_var:
                succ.setdefault(inp, set()).add(n.variable)
    seeded = {n.variable for n in cg.ordered_nodes
              if n.variable in sources or any(i not in by_var for i in n.inputs)}

    def reachable_avoiding(target, avoid):
        seen, stack = set(), [s for s in seeded if s != avoid]
        while stack:
            cur = stack.pop()
            if cur == target:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(nxt for nxt in succ.get(cur, ()) if nxt != avoid)
        return False

    boundary = set(required)
    for n in cg.ordered_nodes:
        x = n.variable
        if any(reachable_avoiding(c, x) for c in succ.get(x, ())):
            boundary.add(x)
    return boundary


def _chain_graph():
    """Return the chain m = basket.y + 1.0, n = m * 2.0 and p = n - 3.0, whose formulas are @njit dispatchers."""
    return Graph(
        variables_lists={"variables": [
            {"name": "m", "inputs": {"y": "basket"}, "formula": njit(lambda y: y + 1.0)},
            {"name": "n", "inputs": {"m": "variables"}, "formula": njit(lambda m: m * 2.0)},
            {"name": "p", "inputs": {"n": "variables"}, "formula": njit(lambda n: n - 3.0)},
        ]},
        external_source_names=["basket"],
    )


def _fanout_graph():
    """Return two independent outputs, o1 = basket.a + 1.0 and o2 = basket.b + 2.0."""
    return Graph(
        variables_lists={"variables": [
            {"name": "o1", "inputs": {"a": "basket"}, "formula": njit(lambda a: a + 1.0)},
            {"name": "o2", "inputs": {"b": "basket"}, "formula": njit(lambda b: b + 2.0)},
        ]},
        external_source_names=["basket"],
    )


def _two_source_reconvergent_graph():
    """Return a graph with two sources: p reads basket.a, q reads basket.b, j multiplies p and q, and r is j + 5.0."""
    return Graph(
        variables_lists={"variables": [
            {"name": "p", "inputs": {"a": "basket"}, "formula": njit(lambda a: a + 1.0)},
            {"name": "q", "inputs": {"b": "basket"}, "formula": njit(lambda b: b - 1.0)},
            {"name": "j", "inputs": {"p": "variables", "q": "variables"},
             "formula": njit(lambda p, q: p * q)},
            {"name": "r", "inputs": {"j": "variables"}, "formula": njit(lambda j: j + 5.0)},
        ]},
        external_source_names=["basket"],
    )


def test_boundary_chain_is_required_only():
    """The boundary of a chain holds only the requested variable, because each variable dominates its one reader."""
    g = _chain_graph()
    cg, _, sources, required = _compiled_parts(g, ["variables.p"])
    b = {v.qual_name() for v in compute_boundary(cg.ordered_nodes, sources, required)}
    assert b == {"variables.p"}


def test_boundary_diamond_persists_join_inputs():
    """The boundary of the diamond holds u and the join's inputs, a and b.

    A change can reach u through b without passing through a, so a recompute plan may read a from the store. The same
    holds for b.
    """
    g = _diamond_graph()
    cg, _, sources, required = _compiled_parts(g, ["variables.u"])
    b = {v.qual_name() for v in compute_boundary(cg.ordered_nodes, sources, required)}
    assert b == {"variables.a", "variables.b", "variables.u"}


@pytest.mark.parametrize("factory,req", [
    (_chain_graph, ["variables.p"]),
    (_diamond_graph, ["variables.u"]),
    (_fanout_graph, ["variables.o1", "variables.o2"]),
    (_two_source_reconvergent_graph, ["variables.r"]),
])
def test_boundary_matches_reference(factory, req):
    """compute_boundary agrees with the reference definition on a chain, the diamond, two outputs and two sources."""
    g = factory()
    cg, _, sources, required = _compiled_parts(g, req)
    got = compute_boundary(cg.ordered_nodes, sources, required)
    assert got == _boundary_reference(cg, sources, required)


def test_cone_liveness_single_run_uses_required_and_boundary():
    """A compiled run of a recompute plan returns the variables that are requested or in the boundary.

    The run of a, b and u, the variables that a change of x affects, returns a and u but not b, which no later step
    reads. It reads x, which it does not compute.
    """
    g = _diamond_graph()
    cg, by_qual, _, _ = _compiled_parts(g, ["variables.u"])
    nodes = {n.variable: n for n in cg.ordered_nodes}
    a, b, u, x = (by_qual["variables.a"], by_qual["variables.b"],
                  by_qual["variables.u"], by_qual["variables.x"])
    run = [nodes[a], nodes[b], nodes[u]]
    boundary = {a, u}
    live_in, live_out = cone_liveness(run, run, {u}, boundary)
    assert set(live_out) == {a, u}
    assert x in set(live_in)
    assert list(live_out) == sorted(live_out, key=lambda v: v.qual_name())


def test_cone_liveness_cross_segment_handoff():
    """A run of a recompute plan returns a variable that a later step reads, even if the variable is not requested.

    m is neither requested nor in the boundary, but the next run computes n from it.
    """
    g = _chain_graph()
    cg, by_qual, _, _ = _compiled_parts(g, ["variables.p"])
    nodes = {n.variable: n for n in cg.ordered_nodes}
    m, n_ = by_qual["variables.m"], by_qual["variables.n"]
    cone_order = [nodes[m], nodes[n_]]
    run1 = [nodes[m]]
    live_in, live_out = cone_liveness(run1, cone_order, set(), set())
    assert m in set(live_out)


def test_recompute_precondition_before_full_call():
    """recompute before the first call raises RuntimeError, because the store has no starting values."""
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u"])
    with pytest.raises(RuntimeError) as exc:
        ck.recompute({"basket": {"y": 100}})
    assert "recompute" in str(exc.value).lower()


def test_recompute_noop_matches_kernel_fused():
    """recompute without changes after the first call of a fused kernel fills the store and returns the same values."""
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u", "variables.a"])
    full = tuple(ck.kernel(100))
    assert tuple(ck.recompute({})) == full


def _interp_recompute_sequence(g, req, ext, changes):
    """Return the requested values after CompiledGraph.execute and after each CompiledGraph.recompute of ``changes``."""
    cg = g.compile(req)
    by_qual = {n.variable.qual_name(): n.variable for n in cg.ordered_nodes}
    values = Values()
    cg.execute(ext, values)
    out = [tuple(values.get(by_qual[q]).value for q in req)]
    for ch in changes:
        cg.recompute(ch, values)
        out.append(tuple(values.get(by_qual[q]).value for q in req))
    return out


def _kernel_recompute_sequence(g, req, ext, changes):
    """Return the requested values after execute and after each recompute of ``changes``, from a compiled kernel."""
    ck = compile_kernel(g, req)
    out = [tuple(ck.execute(ext)[q] for q in req)]
    for ch in changes:
        out.append(tuple(ck.recompute(ch)))
    return out


@pytest.mark.parametrize("factory,req,ext,changes", [
    (_chain_graph, ["variables.p"], {"basket": {"y": 10.0}},
     [{"basket": {"y": 11.0}}, {"basket": {"y": 12.0}}, {"basket": {"y": 13.0}}]),
    (_diamond_graph, ["variables.u", "variables.a"], {"basket": {"y": 100}},
     [{"basket": {"y": 101}}, {"basket": {"y": 102}}, {"basket": {"y": 103}}]),
    # In the graph with two sources, a change of a alone and a change of b alone recompute different variables, and j
    # then reads its other input from the store. The sequence therefore fails if a plan does not write a join input
    # back into the store. A graph with one source recomputes every variable at each change.
    (_two_source_reconvergent_graph, ["variables.r"], {"basket": {"a": 2.0, "b": 3.0}},
     [{"basket": {"a": 5.0}}, {"basket": {"b": 7.0}}, {"basket": {"a": 9.0, "b": 1.0}}]),
])
def test_recompute_equiv_external(factory, req, ext, changes):
    """After each external change, a kernel's recompute returns what CompiledGraph.recompute returns."""
    g = factory()
    assert _kernel_recompute_sequence(g, req, ext, changes) == \
        _interp_recompute_sequence(g, req, ext, changes)


def test_recompute_equiv_mixed_cone():
    """recompute on a segmented kernel returns what CompiledGraph.recompute returns.

    The plan runs calc.n3 in Python between compiled segments, and the changes are new values of ext.x.
    """
    g = _chain_graph_with_python_middle()
    req = ["calc.n5"]
    ext = {"ext": {"x": 7.0}}
    changes = [{"ext": {"x": 8.0}}, {"ext": {"x": 9.0}}]
    assert _kernel_recompute_sequence(g, req, ext, changes) == \
        _interp_recompute_sequence(g, req, ext, changes)


def test_recompute_runtime_error_propagates():
    """An exception that a formula raises in a recompute plan propagates without a retry.

    recompute retries only after a NumbaError, and 1.0 / 0.0 raises ZeroDivisionError.
    """
    g = Graph(
        variables_lists={"variables": [
            {"name": "z", "inputs": {"y": "basket"}, "formula": njit(lambda y: 1.0 / y)},
        ]},
        external_source_names=["basket"],
    )
    ck = compile_kernel(g, ["variables.z"])
    ck.execute({"basket": {"y": 1.0}})
    with pytest.raises(ZeroDivisionError):
        ck.recompute({"basket": {"y": 0.0}})


def test_recompute_cache_reuses_plan():
    """A second recompute of the same changed name reuses the compiled plan."""
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u"])
    ck.execute({"basket": {"y": 100}})
    ck.recompute({"basket": {"y": 101}})
    key = next(iter(ck._cone_cache))
    plan_first = ck._cone_cache[key]
    ck.recompute({"basket": {"y": 102}})
    assert ck._cone_cache[key] is plan_first


def test_recompute_cache_cap_evicts():
    """A kernel whose cap is set to 1 keeps one recompute plan, and builds a discarded plan again when it is needed."""
    g = _fanout_graph()
    ck = compile_kernel(g, ["variables.o1", "variables.o2"])
    ck._cone_cap = 1
    ck.execute({"basket": {"a": 1.0, "b": 2.0}})
    # The plan for o2 replaces the plan for o1.
    ck.recompute({"basket": {"a": 3.0}})
    ck.recompute({"basket": {"b": 4.0}})
    assert len(ck._cone_cache) == 1
    assert tuple(ck.recompute({"basket": {"a": 5.0}})) == (5.0 + 1.0, 4.0 + 2.0)


def test_recompute_typechange_flushes_and_recovers():
    """recompute returns the correct value after basket.y changes from float64 to int64.

    recompute does not retry: the kernel reuses the plan compiled for float64, and numba compiles the plan's segment
    again for int64.
    """
    g = _chain_graph()
    ck = compile_kernel(g, ["variables.p"])
    ck.execute({"basket": {"y": 10.0}})
    ck.recompute({"basket": {"y": 11.0}})
    assert len(ck._cone_cache) >= 1
    out = ck.recompute({"basket": {"y": np.int64(12)}})
    assert out == (((12 + 1.0) * 2.0) - 3.0,)


def test_recompute_interior_override_matches_interpreted():
    """Overrides of the calculated variable variables.x produce the values that CompiledGraph produces.

    The changes override variables.x, change basket.y, and override variables.x again.
    """
    g = _diamond_graph()
    req, ext = ["variables.u"], {"basket": {"y": 100}}
    changes = [{"variables": {"x": 5.0}}, {"basket": {"y": 101}}, {"variables": {"x": 9.0}}]
    assert _kernel_recompute_sequence(g, req, ext, changes) == \
        _interp_recompute_sequence(g, req, ext, changes)


def test_recompute_interior_override_expands_sources_and_flushes():
    """The first override of a calculated variable adds it to the change sources and discards every cached plan.

    The plans depend on the boundary, which the change sources determine.
    """
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u"])
    ck.execute({"basket": {"y": 100}})
    ck.recompute({"basket": {"y": 101}})
    assert ck._cone_cache
    ext_key = next(iter(ck._cone_cache))
    ck.recompute({"variables": {"x": 5.0}})
    assert any(v.qual_name() == "variables.x" for v in ck._sources)
    assert ext_key not in ck._cone_cache


def test_recompute_external_and_interior_yield_distinct_keys():
    """A change of basket.y and an override of variables.x cache two plans under two keys.

    The first plan recomputes x, a, b and u from basket.y, and the second recomputes a, b and u from variables.x, so
    the keys differ in both of their parts. Both plans compute the values that CompiledGraph computes.
    """
    g = _diamond_graph()
    req, ext = ["variables.u"], {"basket": {"y": 100}}
    # The override is the first change, because the first override of a calculated variable discards every cached plan.
    changes = [{"variables": {"x": 5.0}}, {"basket": {"y": 101}}]
    ck = compile_kernel(g, req)
    out = [tuple(ck.execute(ext)[q] for q in req)]
    for ch in changes:
        out.append(tuple(ck.recompute(ch)))
    assert len(ck._cone_cache) == 2
    keys = list(ck._cone_cache)
    assert len({k[0] for k in keys}) == 2
    assert len({k[1] for k in keys}) == 2
    assert out == _interp_recompute_sequence(g, req, ext, changes)


def test_recompute_graph_priority_over_co_changed_interior_override():
    """CompiledKernel.recompute discards a given value of a calculated variable when the same call changes its input.

    It then calculates the variable from its formula, as CompiledGraph.recompute does.
    """
    g = Graph(
        variables_lists={"vars_": [
            {"name": "b", "inputs": {"a": "ext"}, "formula": njit(lambda a: 10 * a)},
            {"name": "c", "inputs": {"b": "vars_"}, "formula": njit(lambda b: b + 1)},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["vars_.b", "vars_.c"])
    b = g.registry["vars_"]["b"]
    c = g.registry["vars_"]["c"]
    assert ck.execute({"ext": {"a": 1}}) == {"vars_.b": 10, "vars_.c": 11}
    # The call sets b to 999 and changes ext.a to 2. b depends on ext.a, so recompute calculates b = 20.
    ck.recompute({"ext": {"a": 2}, "vars_": {"b": 999}})
    assert ck._store[b] == 20
    assert ck._store[c] == 21
