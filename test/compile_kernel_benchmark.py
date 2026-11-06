"""Benchmark compile_kernel against CompiledGraph on a chain of distinct formulas.

The benchmark measures how much faster one fused @njit kernel is than ``CompiledGraph.execute`` over the same nodes,
whose formulas are jitted as well. The difference is the gain of compiling the iteration over the ordered nodes, apart
from the gain of compiling each formula. The benchmark also measures how large the fused kernel's numba cache grows,
and how long the kernel takes to compile for a large graph, when each formula is a ``CPUDispatcher``. The compile
report compiles the formulas with ``proxy`` as well, so that the kernel calls each formula as an external function
instead of holding its code, and compares the two.

The graph is a chain of ``N`` arithmetic nodes over 1-d float64 arrays, ``v_i = f_i(v_{i-1}, b)``, already in
topological order, so that the benchmark exercises the iteration over the nodes. Each node has a formula of its own.
Under the default profile, mixed, a third of the formulas compute a transcendental function (sin, cos, tanh, exp or
sqrt), and the others compute an affine form. The graph is therefore heterogeneous, and numba compiles each node's
formula separately.

Run it from the repository root, with numbox installed::

    python -m test.compile_kernel_benchmark                       # perf, N=200, mixed
    python -m test.compile_kernel_benchmark --nodes 1000 --size 10000
    python -m test.compile_kernel_benchmark --profile cheap       # dispatch-bound regime
    python -m test.compile_kernel_benchmark --compile-report      # + cache/compile
    python -m test.compile_kernel_benchmark --python-nodes 3      # segmented, 3 Python nodes
    python -m test.compile_kernel_benchmark --recompute           # incremental recompute
    python test/compile_kernel_benchmark.py --help

``--profile`` selects the cost of the formulas, so that the hot-path results below can be reproduced with this file.
``cheap`` selects the regime in which dispatch dominates and the speedup of fusion is largest. ``mixed``, the default,
produces the results reported here, and ``expensive`` selects the regime in which computation dominates.

``--nodes`` sets the length of the chain. Graph.compile orders the chain with a recursive walk, so main raises Python's
recursion limit to 20000 before it builds the graph. The results below use chains of 1000 nodes. A cold compile of the
fused kernel takes a long time, so each mode prints a notice before it compiles.

----------------------------------------------------------------------------
The sample results below were measured on an AMD Ryzen 5 7640HS under Linux x86-64, with CPython 3.12, numba 0.65.1
and the default profile, mixed; the numbers differ on other machines. The hot path's times are in microseconds per
call, the best of 30::

    N=1000, S=1000    fused 3311   cg_execute 6192 (1.9x)   iter-only 5689 (1.7x)   numpy 4829 (1.5x)
    N=1000, S=10000   fused 32737  cg_execute 62589 (1.9x)  iter-only 38155 (1.2x)  numpy 31135 (0.95x)

Here CompiledGraph.execute takes 1.9 times as long as the fused kernel, while with --profile cheap it takes more than
10 times as long. A third of the mixed profile's nodes compute transcendental functions, so the work inside the
formulas outweighs the Python dispatch between them, which fusion removes. The iteration-only row shows the same
pattern: it takes 1.7 times as long as the fused kernel at S=1000 and 1.2 times at S=10000, where the fused kernel is
about as fast as NumPy (0.95). Compiling the iteration therefore yields a large gain on a graph of cheap formulas, in
which dispatch dominates, and a small gain on a graph of expensive ones. --profile cheap and --profile expensive show
the two extremes.

The compile report (--compile-report) at N=1000 measured a cold run and then a new process with a warm cache. ``total``
is the comparable cold cost, because the njit formulas compile inside kernel_compile, at the kernel's first call, and
the proxy formulas compile inside build, when proxy decorates them. The child processes of the report's proxy rows
raise ValueError, as run_compile_report explains, so this file cannot reproduce the proxy numbers below::

    bindings        total cold   warm restart   on-disk cache
    CPUDispatcher   ~155 s       ~0.8 s         ~23 MiB   (1 kernel file)
    proxy           ~168 s       ~114 s (*)     ~1.2 MiB  (kernel only)
    proxy (cached)  ~172 s       ~3.8 s         ~57 MiB   (~4000 files)

The kernel of CPUDispatcher formulas holds the code of all N formulas: one cache of about 23 MiB, a cold compile of
about 155 s and a warm start of about 0.8 s. With proxy, the kernel calls each formula as an external function, and the
kernel's cache is about 20 times smaller, about 1.2 MiB. The formulas themselves are then uncached, so every process
compiles all N of them again (*): the warm start takes about 114 s, longer than with njit. With ``cache=True`` for the
proxy formulas, the warm start takes about 3.8 s. Each of the N distinct formulas then has cache files of its own, about
4000 files and 57 MiB in total, more than the single cache of the njit kernel. proxy therefore makes the kernel's cache
smaller but the whole cache larger, once the formulas are distinct and cached. proxy also needs a signature and a named
function with a source file for each formula.
----------------------------------------------------------------------------
"""
import argparse
import getpass
import importlib.util
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import numba
import numpy
from numba import njit, float64

from numbox.core.variable.variable import Graph, Values
from numbox.core.variable.compile_kernel import compile_kernel
from numbox.core.proxy.proxy import proxy

# SIG is the signature for which proxy compiles each formula: two 1-d float64 arrays in, one out.
SIG = float64[:](float64[:], float64[:])


# --- formula generation ------------------------------------------------------
# Each node's formula takes a and b and returns a 1-d float64 array: a is the previous node's value, and b is a fixed
# external value. The chain can be a thousand nodes deep, so each formula must keep the values bounded: it applies a
# bounded transcendental function to a, or multiplies a by a factor whose absolute value is below 1. b is the same at
# every node, so a formula may apply any bounded function to it. Under the mixed profile, a third of the nodes compute
# an expensive transcendental function, which also bounds their value.
#
# load_formulas writes the N functions into a module file and imports it. numba names a function's cache files after
# its source file, its qualified name and its first line, so each formula has cache files of its own. proxy needs the
# module and the source file of each function that it wraps. compile_kernel reads neither: its digest folds the code
# objects of the formulas.

def _body(i, profile):
    """Return the expression of node ``i``'s formula in ``a`` and ``b``, which keeps the chain's values bounded.

    ``profile`` selects the forms: "cheap" uses affine forms alone, so dispatch dominates; "expensive" uses
    transcendental forms alone, so computation dominates; "mixed" uses transcendental forms for every third node. Each
    form bounds ``a`` with a transcendental function or scales it by a factor below 1. ``b`` is the same external value
    at every node, so a form may apply any bounded function to it.
    """
    # ca lies between 0.30 and 0.90, so ca * a contracts. cb lies between 0.10 and 0.40, and cc between -0.5 and 0.5.
    ca = round(0.30 + (i % 13) * 0.05, 3)
    cb = round(0.10 + (i % 7) * 0.05, 3)
    cc = round(-0.5 + (i % 11) * 0.1, 3)
    cheap_a = [f"{ca}*a", f"-{ca}*a + {cc}", f"{ca}*a - {cb}*b"]
    exp_a = [
        f"numpy.sin({ca}*a)",
        f"numpy.tanh(a - {cb}*b)",
        f"numpy.cos(a)*{ca}",
        f"numpy.exp(-numpy.abs({ca}*a))",
        f"numpy.sqrt(numpy.abs(a))*{ca}",
    ]
    cheap_b = [f"{cb}*b", f"{cb}*b*b", f"-{cb}*b"]
    exp_b = [f"{cb}*numpy.sin(b)", f"{cb}*numpy.abs(b)", f"{cb}*numpy.cos(b)"]
    if profile == "cheap":
        a_forms, b_forms = cheap_a, cheap_b
    elif profile == "expensive":
        a_forms, b_forms = exp_a, exp_b
    elif i % 3 == 0:  # The mixed profile makes every third node expensive.
        a_forms, b_forms = exp_a, exp_b
    else:
        a_forms, b_forms = cheap_a, cheap_b
    return f"{a_forms[i % len(a_forms)]} + {b_forms[i % len(b_forms)]}"


def python_node_indices(n_nodes, k):
    """Return the indices of the ``k`` nodes that run in Python, spaced evenly through the chain.

    The indices are ``i * (n_nodes // (k + 1))`` for ``i`` from 1 to ``k``, and there are none if ``k`` is 0 or less.
    load_formulas puts a call of ``json.dumps``, which numba cannot compile, first in the formulas of these nodes. The
    kernel's search therefore runs them in Python, and numba compiles the nodes between them as segments.
    """
    if k <= 0:
        return frozenset()
    step = n_nodes // (k + 1)
    return frozenset(i * step for i in range(1, k + 1))


# _FORMULAS_CACHE holds the functions of each generated module, so that a process imports each module once.
_FORMULAS_CACHE = {}


def _atomic_write(path, text):
    """Write ``text`` to ``path`` through a temporary file and ``os.replace``.

    A process that reads ``path`` therefore never sees a partly written file.
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _formulas_dir():
    """Return ``ck_bench_<user>`` under the temporary directory, the directory of the generated formula modules.

    The function creates the directory if it is missing. On a system with user ids, it raises RuntimeError if another
    user owns the directory, because the benchmark imports the modules that it holds. Otherwise it makes the directory
    private to the user.
    """
    user = re.sub(r"[^0-9A-Za-z._-]", "_", getpass.getuser())
    d = pathlib.Path(tempfile.gettempdir()) / f"ck_bench_{user}"
    d.mkdir(mode=0o700, exist_ok=True)
    if hasattr(os, "getuid"):
        if d.stat().st_uid != os.getuid():
            raise RuntimeError(f"{d} exists but is not owned by the current user")
        d.chmod(0o700)  # mkdir does not change the mode of an existing directory, so chmod sets it.
    return d


def load_formulas(n_nodes, profile, python_nodes=frozenset()):
    """Return the ``n_nodes`` distinct formulas ``f0`` to ``f{n_nodes - 1}`` of ``profile``, as a list.

    The function writes the formulas into a module file and imports it, so that numba and inspect find each function's
    source. The module's name, and with it the file's name, holds the profile, the number of nodes and the indices of
    the Python nodes. The child processes of the compile report therefore regenerate the same module, and numba's cache
    of its functions stays valid across processes. A second call with the same arguments returns the same functions.

    ``python_nodes``, a set of node indices, makes those formulas run in Python. Their body keeps the same arithmetic
    after a first statement, a call of ``json.dumps``, which numba cannot compile. With ``python_nodes`` empty, the
    module has no ``json`` import and its name has no suffix, so the module without Python nodes and a module with
    them never share a cache.
    """
    cache_key = (profile, n_nodes, python_nodes)
    cached = _FORMULAS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    lines = (["import json", "import numpy", "", ""] if python_nodes
             else ["import numpy", "", ""])
    for i in range(n_nodes):
        lines.append(f"def f{i}(a, b):")
        if i in python_nodes:
            lines.append('    json.dumps({"k": 1})')  # numba cannot compile json.dumps, so the node runs in Python.
        lines += [f"    return {_body(i, profile)}", "", ""]
    text = "\n".join(lines)
    suffix = "_py" + "-".join(map(str, sorted(python_nodes))) if python_nodes else ""
    mod_name = f"_ck_bench_formulas_{profile}_{n_nodes}{suffix}"
    path = _formulas_dir() / f"{mod_name}.py"
    # The module is written only when its text changes. A rewrite would change the file's modification time, which
    # numba's cache index records, so the warm proxy_cached run would compile every formula again.
    if not path.exists() or path.read_text() != text:
        _atomic_write(path, text)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    funcs = [getattr(module, f"f{i}") for i in range(n_nodes)]
    _FORMULAS_CACHE[cache_key] = funcs
    return funcs


def make_formula(fn, kind):
    """Return ``fn`` wrapped as ``kind`` names: "njit", "proxy", or "proxy_cached" for proxy with caching enabled.

    A kernel holds the code of an @njit formula and calls a proxy formula as an external function. Any other ``kind``
    raises ValueError.
    """
    if kind == "njit":
        return njit(fn)
    if kind == "proxy":
        return proxy(SIG)(fn)
    if kind == "proxy_cached":
        return proxy(SIG, jit_options={"cache": True})(fn)
    raise ValueError(kind)


def build_graph(n_nodes, kind, formulas, python_nodes=frozenset()):
    """Return a chain of ``n_nodes`` nodes and the qualified name of its last node.

    vars.v0 reads ext.a and ext.b, and each later node reads the previous node and ext.b. make_formula wraps each
    formula as ``kind`` names, except that a node in ``python_nodes`` keeps its plain Python function. CompiledGraph
    calls a formula as it is given, and an @njit wrapper of a formula that numba cannot compile would fail.
    compile_kernel's search finds such a formula and runs it in Python.
    """
    def formula_for(i):
        return formulas[i] if i in python_nodes else make_formula(formulas[i], kind)
    specs = [{
        "name": "v0",
        "inputs": {"a": "ext", "b": "ext"},
        "formula": formula_for(0),
    }]
    for i in range(1, n_nodes):
        specs.append({
            "name": f"v{i}",
            "inputs": {f"v{i - 1}": "vars", "b": "ext"},
            "formula": formula_for(i),
        })
    graph = Graph({"vars": specs}, external_source_names=["ext"])
    return graph, f"vars.v{n_nodes - 1}"


def make_externals(size, seed=0):
    """Return two arrays of ``size`` values from a standard normal distribution, drawn with the seed ``seed``."""
    rng = numpy.random.default_rng(seed)
    return rng.standard_normal(size), rng.standard_normal(size)


def reference_numpy(formulas, a, b):
    """Evaluate the chain with the plain Python formulas on NumPy arrays, and return the last value.

    The benchmark checks the kernels against the result and times the evaluation as the vectorized baseline.
    """
    v = formulas[0](a, b)
    for f in formulas[1:]:
        v = f(v, b)
    return v


def best_median(fn, repeats):
    """Return the best and the median time of one call of ``fn``, in nanoseconds, over ``repeats`` calls.

    A first call, which is not timed, precedes them.
    """
    fn()
    samples = [_timed(fn) for _ in range(repeats)]
    samples.sort()
    return samples[0], samples[len(samples) // 2]


def _timed(fn):
    """Return the time of one call of ``fn``, in nanoseconds."""
    t0 = time.perf_counter_ns()
    fn()
    return time.perf_counter_ns() - t0


def run_perf(n_nodes, size, repeats, profile):
    """Print the hot-path times of the fused kernel, CompiledGraph and NumPy on a chain of ``n_nodes`` formulas.

    The rows are the fused kernel, CompiledGraph.execute, CompiledGraph's loop over the nodes alone (iter-only) and the
    NumPy evaluation. Each row shows the best and the median time of ``repeats`` calls on arrays of ``size`` values.
    The function first checks that the fused kernel and CompiledGraph compute the same values as the NumPy evaluation.
    """
    formulas = load_formulas(n_nodes, profile)
    a, b = make_externals(size)
    ref = reference_numpy(formulas, a, b)
    assert numpy.all(numpy.isfinite(ref)), "chain diverged -- a formula is amplifying"

    graph, required = build_graph(n_nodes, "njit", formulas)

    print(f"compiling fused kernel for N={n_nodes} nodes (cold cache is slow)...",
          flush=True)
    t0 = time.perf_counter_ns()
    ck = compile_kernel(graph, required)
    ck.kernel(a, b)
    print(f"  fused kernel ready in {(time.perf_counter_ns() - t0) / 1e9:.1f}s",
          flush=True)

    compiled = graph.compile([required])
    last_var = {n.variable.qual_name(): n.variable
                for n in compiled.ordered_nodes}[required]

    # The fused kernel and CompiledGraph must compute the same values as the NumPy evaluation.
    fused_out = ck.kernel(a, b)[0]
    vs0 = Values()
    compiled.execute({"ext": {"a": a, "b": b}}, vs0)
    assert numpy.max(numpy.abs(fused_out - ref)) < 1e-9, "fused kernel mismatch"
    assert numpy.max(numpy.abs(vs0.get(last_var).value - ref)) < 1e-9, "CompiledGraph mismatch"

    # The iteration-only row writes the external values once, through a private method, because no public method
    # writes them without also running the formulas. Its timing then covers the loop over the nodes and the calls of
    # the formulas. The untimed first call of best_median creates the Value objects of vs_iter, so that no timed call
    # inserts into its dict.
    vs_iter = Values()
    compiled._assign_external_values({"ext": {"a": a, "b": b}}, vs_iter)

    rows = [
        ("fused kernel", best_median(lambda: ck.kernel(a, b), repeats)),
        ("cg_execute", best_median(
            lambda: compiled.execute({"ext": {"a": a, "b": b}}, Values()), repeats)),
        ("cg_calculate (iter only)", best_median(
            lambda: compiled._calculate(compiled.ordered_nodes, vs_iter), repeats)),
        ("numpy", best_median(lambda: reference_numpy(formulas, a, b), repeats)),
    ]
    fused_best = rows[0][1][0]

    print(f"\nhot path, profile={profile}, N={n_nodes} nodes, array size S={size}, "
          f"best of {repeats} (microseconds/call):")
    print(f"  {'path':<26}{'best':>10}{'median':>10}{'vs fused':>10}")
    for name, (best, med) in rows:
        print(f"  {name:<26}{best / 1e3:10.2f}{med / 1e3:10.2f}{best / fused_best:9.2f}x")


# --- segmented orchestration report (the --python-nodes mode) ----------------

def run_python_nodes_mode(n_nodes, size, repeats, profile, k):
    """Print the cost of a segmented kernel whose chain holds ``k`` formulas that numba cannot compile.

    compile_kernel cannot fuse such a graph. The kernel's first call searches for the Python formulas and builds a plan
    of compiled segments with Python steps between them. The function reports the time of that first call and the
    hot-path time of the plan, against CompiledGraph and NumPy on the same chain.
    """
    py_idx = python_node_indices(n_nodes, k)
    formulas = load_formulas(n_nodes, profile, py_idx)
    a, b = make_externals(size)
    ref = reference_numpy(formulas, a, b)
    assert numpy.all(numpy.isfinite(ref)), "chain diverged -- a formula is amplifying"

    graph, required = build_graph(n_nodes, "njit", formulas, py_idx)

    print(f"compiling segmented kernel for N={n_nodes} nodes, K={k} python nodes "
          f"at {sorted(py_idx)} (cold cache is slow)...", flush=True)
    t0 = time.perf_counter()
    ck = compile_kernel(graph, required)
    t_compile = time.perf_counter() - t0

    # The first call searches: it compiles each formula separately, finds the Python formulas and builds the plan of
    # segments. Every later call runs that plan.
    t0 = time.perf_counter()
    first = ck.kernel(a, b)
    t_first = time.perf_counter() - t0

    rep = ck.partition
    n_seg = len([s for s in rep.segments if s.kind == "jit"])
    n_py = len(rep.python_nodes)

    # The segmented kernel and CompiledGraph must compute the same values as the NumPy evaluation.
    compiled = graph.compile([required])
    last_var = {n.variable.qual_name(): n.variable
                for n in compiled.ordered_nodes}[required]
    vs0 = Values()
    compiled.execute({"ext": {"a": a, "b": b}}, vs0)
    assert numpy.max(numpy.abs(first[0] - ref)) < 1e-9, "segmented kernel mismatch"
    assert numpy.max(numpy.abs(vs0.get(last_var).value - ref)) < 1e-9, "CompiledGraph mismatch"

    rows = [
        ("segmented kernel", best_median(lambda: ck.kernel(a, b), repeats)),
        ("cg_execute", best_median(
            lambda: compiled.execute({"ext": {"a": a, "b": b}}, Values()), repeats)),
        ("numpy", best_median(lambda: reference_numpy(formulas, a, b), repeats)),
    ]
    seg_best = rows[0][1][0]

    print(f"\nsegmented orchestration, profile={profile}, N={n_nodes} nodes, "
          f"K={k} python nodes, array size S={size}:")
    print(f"  mode                   {rep.mode}")
    print(f"  jit segments           {n_seg}")
    print(f"  python nodes           {n_py}")
    print(f"  compile_kernel()       {t_compile:.3f}s  (defers all compilation to the first call)")
    print(f"  first call (discovery) {t_first:.3f}s  (one-time per process; per-node probe "
          f"compiles do not disk-cache away, only the segment compiles do)")
    print(f"\nhot path, best of {repeats} (microseconds/call):")
    print(f"  {'path':<26}{'best':>10}{'median':>10}{'vs segmented':>14}")
    for name, (best, med) in rows:
        print(f"  {name:<26}{best / 1e3:10.2f}{med / 1e3:10.2f}{best / seg_best:13.2f}x")


# --- compile / cache report (CPUDispatcher against proxy) --------------------

def _cache_size(cache_dir):
    """Return the total size in bytes and the number of numba's .nbi and .nbc files under ``cache_dir``."""
    files = [p for p in pathlib.Path(cache_dir).rglob("*") if p.suffix in (".nbi", ".nbc")]
    return sum(p.stat().st_size for p in files), len(files)


def _compile_worker(n_nodes, kind, profile):
    """Build, compile and call the kernel once in this process, and print one line of the compile report.

    make_formula wraps the formulas as ``kind`` names. The line reports the time to build the graph, the time of
    compile_kernel and the first call, their total, the size and number of the cache files under NUMBA_CACHE_DIR, and
    whether numba warned that it cannot cache a function.
    """
    cache_dir = os.environ["NUMBA_CACHE_DIR"]
    formulas = load_formulas(n_nodes, profile)
    a, b = make_externals(64)
    ref = reference_numpy(formulas, a, b)

    t0 = time.perf_counter_ns()
    graph, required = build_graph(n_nodes, kind, formulas)
    build_ms = (time.perf_counter_ns() - t0) / 1e6

    cannot_cache = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t1 = time.perf_counter_ns()
        ck = compile_kernel(graph, required)
        out = ck.kernel(a, b)[0]
        kernel_ms = (time.perf_counter_ns() - t1) / 1e6
        cannot_cache = any(
            issubclass(w.category, numba.NumbaWarning)
            and "cannot cache" in str(w.message).lower()
            for w in caught)

    assert numpy.max(numpy.abs(out - ref)) < 1e-9, "correctness mismatch"
    size, nfiles = _cache_size(cache_dir)
    print(f"  {kind:<13} build={build_ms:9.1f}ms  kernel_compile={kernel_ms:10.1f}ms  "
          f"total={build_ms + kernel_ms:10.1f}ms  cache={size / 1024:9.1f}KiB "
          f"({nfiles} files)  cacheable={'NO' if cannot_cache else 'yes'}")


def run_compile_report(n_nodes, profile):
    """Print the cold and the warm compile of the kernel for each kind of formula: njit, proxy and proxy_cached.

    The function creates a NUMBA_CACHE_DIR for each kind and deletes it afterwards. A first child process compiles
    into the empty cache (COLD), and a second child process runs with the cache that the first child wrote (WARM). Each
    child prints one line.

    The child processes of the proxy and proxy_cached rows raise ValueError. proxy anchors its wrapper at the first line
    of the decorated function, which must be line 8 or later of its file, and load_formulas writes ``f0`` at line 4.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    print(f"\ncompile / cache report, profile={profile}, N={n_nodes} nodes (COLD "
          f"writes cache, WARM is a fresh process reusing it; cold compile can "
          f"take minutes per binding kind -- see docstring):", flush=True)
    print("  ('total' is the apples-to-apples cost: njit formulas compile inside "
          "kernel_compile, proxy formulas inside build)", flush=True)
    for kind in ("njit", "proxy", "proxy_cached"):
        cache_dir = tempfile.mkdtemp(prefix=f"ck_bench_{kind}_")
        env = {**os.environ, "NUMBA_CACHE_DIR": cache_dir}
        try:
            for label in ("COLD", "WARM"):
                print(f"{label} ", end="", flush=True)
                # The child runs in the repository root, so that -m test.compile_kernel_benchmark finds the module
                # wherever the parent started.
                subprocess.run(
                    [sys.executable, "-m", "test.compile_kernel_benchmark",
                     "--_worker", kind, "--nodes", str(n_nodes),
                     "--profile", profile],
                    cwd=repo_root, env=env, check=True)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)


# --- recompute report (the --recompute mode) ---------------------------------


def _affine(c, d):
    """Return an @njit formula that computes ``c * a + d`` for its argument ``a``.

    Each call returns a new function. A chain of such formulas stays bounded if ``|c|`` is below 1.
    """
    def f(a):
        return c * a + d
    return njit(f)


def build_recompute_chain(n_nodes, suffix=50):
    """Return a chain of ``n_nodes`` affine nodes over float64 arrays, with ext.a as its one input.

    The result is ``(graph, required, late_name)``. ``late_name`` is the name, without its namespace vars, of the node
    ``suffix`` steps from the end. An override of that node affects only the nodes after it, so recompute runs a short
    suffix of the chain.
    """
    specs = [{"name": "v0", "inputs": {"a": "ext"}, "formula": _affine(0.5, 0.1)}]
    for i in range(1, n_nodes):
        c = round(0.30 + (i % 13) * 0.05, 3)
        d = round(-0.20 + (i % 7) * 0.05, 3)
        specs.append({"name": f"v{i}", "inputs": {f"v{i - 1}": "vars"},
                      "formula": _affine(c, d)})
    graph = Graph({"vars": specs}, external_source_names=["ext"])
    late = max(1, n_nodes - suffix)
    return graph, f"vars.v{n_nodes - 1}", f"v{late}"


def build_recompute_diamond():
    """Return ``(graph, required, ext)`` for the diamond: x reads basket.y, a and b read x, and u reads a and b.

    ``ext`` holds None for basket.y, a value that the caller replaces with an array.
    """
    graph = Graph(
        variables_lists={"variables": [
            {"name": "x", "inputs": {"y": "basket"}, "formula": _affine(2.0, 0.0)},
            {"name": "a", "inputs": {"x": "variables"}, "formula": _affine(0.7, -1.0)},
            {"name": "b", "inputs": {"x": "variables"}, "formula": _affine(0.3, 0.5)},
            {"name": "u", "inputs": {"a": "variables", "b": "variables"},
             "formula": njit(lambda a, b: a + b)},
        ]},
        external_source_names=["basket"],
    )
    return graph, ["variables.u"], {"basket": {"y": None}}


def build_recompute_fanout(width, depth):
    """Return ``width`` independent columns, each a chain of ``depth`` affine nodes from an input to an output.

    Column ``k`` reads ``ext.i{k}``, computes ``vars.c{k}_0`` to ``vars.c{k}_{depth - 1}``, and ends at the output
    ``vars.o{k}``. The result is ``(graph, required, ext, col_inputs)``. ``ext`` holds None for each input, and
    ``col_inputs`` lists the ``(source, name)`` pair of each column's input.
    """
    specs = []
    col_inputs = []
    for k in range(width):
        specs.append({"name": f"c{k}_0", "inputs": {f"i{k}": "ext"},
                      "formula": _affine(0.5, 0.1)})
        for j in range(1, depth):
            c = round(0.30 + ((k + j) % 13) * 0.05, 3)
            specs.append({"name": f"c{k}_{j}", "inputs": {f"c{k}_{j - 1}": "vars"},
                          "formula": _affine(c, 0.05)})
        specs.append({"name": f"o{k}", "inputs": {f"c{k}_{depth - 1}": "vars"},
                      "formula": _affine(1.0, 0.0)})
        col_inputs.append(("ext", f"i{k}"))
    graph = Graph({"vars": specs}, external_source_names=["ext"])
    required = [f"vars.o{k}" for k in range(width)]
    ext = {"ext": {f"i{k}": None for k in range(width)}}
    return graph, required, ext, col_inputs


def _interp_recompute_driver(graph, required, seed_ext, next_change):
    """Return a function without arguments that runs one CompiledGraph.recompute.

    The store is a Values instance that CompiledGraph.execute filled from ``seed_ext``. ``next_change`` is a function
    without arguments that returns the next ``changed`` dict, such as the cyclic function of run_recompute_mode, so
    that CompiledGraph receives the same sequence of changes as the kernel.
    """
    req = required if isinstance(required, list) else [required]
    cg = graph.compile(req)
    values = Values()
    cg.execute(seed_ext, values)

    def step():
        cg.recompute(next_change(), values)
    return step


def run_recompute_mode(size, repeats, chain_n, fan_width, fan_depth):
    """Print the steady-state time of recompute in three regimes, against a full call and CompiledGraph.recompute.

    A full call fills the kernel's store once. Each regime then runs recompute calls that change a small part of the
    input. It compares the time of one recompute with a call of the kernel, which computes the whole graph again, and
    with CompiledGraph.recompute:

    - deep chain: a chain of ``chain_n`` nodes with one input. A change of the input affects the whole chain, so
      recompute cannot be faster than a full call there. The regime therefore overrides a node late in the chain, whose
      dependents are about the last 50 nodes. recompute runs only those, while a full call computes every node, so the
      gain grows with the length of the chain.
    - diamond: a and b read x, and u reads a and b. A change of x affects the whole small graph, so the regime measures
      the cost of the values that the plan writes back into the store at a join, not a gain of fusion.
    - fan-out: ``fan_width`` independent columns of ``fan_depth`` nodes. Each recompute changes the input of the next
      column, in turn, so the kernel uses its cache of plans for every column. recompute runs one column, while a full
      call computes all of them. If there are more columns than the cache's cap of 64 plans, every recompute builds its
      plan again.
    """
    a, _ = make_externals(size)

    def cyclic(values):
        """Return a function that returns the elements of ``values`` in turn, starting again after the last."""
        i = [0]

        def nxt():
            v = values[i[0] % len(values)]
            i[0] += 1
            return v
        return nxt

    rows = []

    # --- deep chain: an override of a late node, whose dependents form a short suffix ---
    suffix = 50
    g, req, late = build_recompute_chain(chain_n, suffix)
    ext = {"ext": {"a": a}}
    print(f"compiling fused kernel for the deep-chain regime (N={chain_n} nodes, "
          f"cold cache is slow)...", flush=True)
    t0 = time.perf_counter()
    ck = compile_kernel(g, req)
    ck.execute(ext)
    print(f"  ready in {time.perf_counter() - t0:.1f}s", flush=True)
    overrides = [a * (0.9 + 0.01 * t) for t in range(8)]
    nxt = cyclic([{"vars": {late: o}} for o in overrides])
    # The first recompute fills the store and compiles the plan, and the second reuses the plan.
    ck.recompute(nxt())
    ck.recompute(nxt())
    steady = best_median(lambda: ck.recompute(nxt()), repeats)[0]
    # The baseline is a call of the kernel, which computes the whole chain from the external value: the work that
    # recompute avoids by running only the suffix. A kernel that took the overridden node as an external value would
    # compute a smaller graph, so it would not be a fair baseline.
    full = best_median(lambda: ck.kernel(a), repeats)[0]
    interp = best_median(
        _interp_recompute_driver(g, req, ext, cyclic(
            [{"vars": {late: o}} for o in overrides])), repeats)[0]
    cones = len(ck._cone_cache)
    rows.append((f"deep chain (interior override, {suffix}-node suffix cone)",
                 steady, full, interp, cones, f"N={chain_n}"))

    # --- diamond: change the shared root -----------------------------------------
    g, req, ext = build_recompute_diamond()
    ext = {"basket": {"y": a}}
    ck = compile_kernel(g, req)
    ck.execute(ext)
    ys = [a * (1.0 + 0.01 * t) for t in range(8)]
    nxt = cyclic([{"basket": {"y": y}} for y in ys])
    ck.recompute(nxt())
    ck.recompute(nxt())
    steady = best_median(lambda: ck.recompute(nxt()), repeats)[0]
    full = best_median(lambda: ck.kernel(a), repeats)[0]
    interp = best_median(
        _interp_recompute_driver(g, req, ext, cyclic(
            [{"basket": {"y": y}} for y in ys])), repeats)[0]
    cones = len(ck._cone_cache)
    rows.append(("diamond (root change, whole cone)",
                 steady, full, interp, cones, "join u"))

    # --- fan-out: cycle a different column each recompute -------------------------
    g, req, ext, col_inputs = build_recompute_fanout(fan_width, fan_depth)
    ext = {"ext": {name: a for _src, name in col_inputs}}
    ck = compile_kernel(g, req)
    full_args = tuple(a for _ in col_inputs)
    ck.execute(ext)
    col_changes = [{src: {name: a * (1.0 + 0.01 * k)}}
                   for k, (src, name) in enumerate(col_inputs)]
    nxt = cyclic(col_changes)
    # One recompute per column compiles and caches every column's plan before the timed calls. If fan_width is at most
    # the cap of 64 plans, every timed recompute reuses a plan. If fan_width exceeds the cap, the kernel discards the
    # plan used least recently. Every timed recompute then builds its plan again, which makes the steady-state time
    # larger.
    for _ in range(fan_width):
        ck.recompute(nxt())
    steady = best_median(lambda: ck.recompute(nxt()), repeats)[0]
    full = best_median(lambda: ck.kernel(*full_args), repeats)[0]
    interp = best_median(
        _interp_recompute_driver(g, req, ext, cyclic(col_changes)),
        repeats)[0]
    cones = len(ck._cone_cache)
    rows.append((f"fan-out {fan_width} cols x {fan_depth} (1-col cone, cycled)",
                 steady, full, interp, cones, f"cap={ck._cone_cap}"))

    print(f"\nrecompute report, array size S={size}, best of {repeats} "
          f"(microseconds/call):")
    print(f"  {'regime':<46}{'recompute':>11}{'full re-run':>13}"
          f"{'interpreted':>13}{'cones':>7}  {'note'}")
    for name, steady, full, interp, cones, note in rows:
        speedup = full / steady if steady else float("nan")
        print(f"  {name:<46}{steady / 1e3:11.2f}{full / 1e3:13.2f}"
              f"{interp / 1e3:13.2f}{cones:7d}  {note} ({speedup:.2f}x vs full)")


def main():
    """Parse the command line and run the modes that it selects, as the module's docstring describes."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nodes", type=int, default=None,
                   help="chain length (default 200 for the perf modes, 600 for "
                        "--recompute so the deep-chain suffix win is visible)")
    p.add_argument("--size", type=int, default=None,
                   help="array length (default 1000 for the perf modes, 5000 for "
                        "--recompute so per-node compute dominates orchestration)")
    p.add_argument("--repeats", type=int, default=30, help="timed reps (default 30)")
    p.add_argument("--profile", choices=("cheap", "mixed", "expensive"), default="mixed",
                   help="per-node cost mix (default mixed: ~1/3 expensive). 'cheap' "
                        "is dispatch-bound (big fusion win); 'expensive' compute-bound")
    p.add_argument("--compile-report", action="store_true",
                   help="also run the CPUDispatcher-vs-proxy compile/cache comparison")
    p.add_argument("--python-nodes", type=int, default=0,
                   help="inject K evenly-spaced non-jittable nodes and run the "
                        "segmented-orchestration report instead of the fused perf "
                        "run (default 0: unchanged fused perf run)")
    p.add_argument("--recompute", action="store_true",
                   help="run the incremental-recompute report (steady-state recompute "
                        "vs full kernel re-run vs interpreted CompiledGraph.recompute) "
                        "across deep-chain, diamond, and fan-out regimes instead of the "
                        "fused perf run")
    p.add_argument("--fan-width", type=int, default=50,
                   help="fan-out regime: number of independent columns (default 50, "
                        "under the 64 cone-plan cap -> clean cache reuse; set above 64 "
                        "to exercise LRU thrash)")
    p.add_argument("--fan-depth", type=int, default=12,
                   help="fan-out regime: per-column chain depth (default 12)")
    p.add_argument("--_worker", help=argparse.SUPPRESS)  # The compile report runs a child with it, for one measurement.
    args = p.parse_args()

    # Graph.compile orders the variables with a recursive depth-first walk, so a chain of N nodes needs a recursion
    # limit above N, and Python's default is 1000. main sets the limit, so that an import of the module leaves it
    # unchanged.
    sys.setrecursionlimit(20000)

    if args._worker:
        _compile_worker(args.nodes if args.nodes is not None else 200,
                        args._worker, args.profile)
        return

    if args.recompute:
        chain_n = args.nodes if args.nodes is not None else 600
        size = args.size if args.size is not None else 5000
        run_recompute_mode(size, args.repeats, chain_n,
                           args.fan_width, args.fan_depth)
        return

    nodes = args.nodes if args.nodes is not None else 200
    size = args.size if args.size is not None else 1000
    if args.python_nodes > 0:
        run_python_nodes_mode(nodes, size, args.repeats, args.profile,
                              args.python_nodes)
    else:
        run_perf(nodes, size, args.repeats, args.profile)
    if args.compile_report:
        run_compile_report(nodes, args.profile)


if __name__ == "__main__":
    main()
