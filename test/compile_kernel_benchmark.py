"""Benchmark: a fused ``compile_kernel`` kernel vs ``CompiledGraph`` execution.

Answers the two questions raised in review of the ``compile_kernel`` PR:

1. How much faster is the single fused ``@njit`` kernel than iterating a
   ``CompiledGraph`` whose formulas are themselves jitted -- i.e. how much comes
   from jitting the *iteration over the topologically ordered nodes*, separate
   from jitting the individual formulas?
2. With the per-node bindings being ``CPUDispatcher`` objects, how big does the
   fused kernel's numba cache get and how long does it take to compile for a
   large graph -- and does lowering the bindings through ``proxy`` (an external
   call per node instead of an inlined body) help?

The graph under test is a chain of ``N`` distinct arithmetic nodes over 1-d
float64 arrays (``v_i = f_i(v_{i-1}, b)``), "already ordered", which stresses
exactly the node iteration. Every node has its own generated formula -- a mix of
cheap affine ops and (~1/3 of nodes) more expensive transcendentals (sin / cos /
tanh / exp / sqrt) -- so the graph is genuinely heterogeneous and each node is a
distinct numba compile, not one of a few ops repeated.

Run it (from the repo root, with numbox installed)::

    python -m test.compile_kernel_benchmark                       # perf, N=200, mixed
    python -m test.compile_kernel_benchmark --nodes 1000 --size 10000
    python -m test.compile_kernel_benchmark --profile cheap       # dispatch-bound regime
    python -m test.compile_kernel_benchmark --compile-report      # + cache/compile
    python -m test.compile_kernel_benchmark --recompute           # incremental recompute
    python test/compile_kernel_benchmark.py --help

``--profile {cheap,mixed,expensive}`` selects the per-node cost mix, so every
claim below is reproducible from this one file: ``cheap`` reproduces the
dispatch-bound regime (large fusion win), ``mixed`` (default) the heterogeneous
numbers reported here, ``expensive`` the compute-bound extreme.

``--nodes`` drives a linear (depth-N) chain, so it is bounded by Python's
recursion limit inside ``Graph.compile`` (a recursive topological sort); tested
to 1000. The fused kernel compile is intentionally slow on a cold cache -- both
the perf path and the compile report print a heads-up before compiling.

----------------------------------------------------------------------------
Sample results (AMD Ryzen 5 7640HS, Linux x86-64, CPython 3.12, numba 0.65.1;
default --profile mixed; your numbers will vary). Hot path, microseconds/call,
best of 30:

    N=1000, S=1000    fused 3311   cg_execute 6192 (1.9x)   iter-only 5689 (1.7x)   numpy 4829 (1.5x)
    N=1000, S=10000   fused 32737  cg_execute 62589 (1.9x)  iter-only 38155 (1.2x)  numpy 31135 (0.95x)

The fused kernel is ~1.9x faster than CompiledGraph's public execute here -- far
less than the >10x it reaches with --profile cheap, because ~1/3 of the nodes do
transcendental work, so per-node compute now dominates the per-node Python
dispatch that fusion removes. Isolating the iteration (iter-only) shows the same
thing: a 1.7x win at S=1000 collapses to 1.2x at S=10000, where fusion roughly
matches numpy (0.95x). So the jitted-iteration win is large for cheap,
dispatch-bound graphs and small for compute-bound ones (run --profile cheap vs
--profile expensive to see both ends).

Compile / cache (--compile-report) at N=1000, cold then warm restart. ``total``
is the apples-to-apples cold cost (njit formulas compile lazily inside
kernel_compile, proxy formulas eagerly inside build):

    bindings        total cold   warm restart   on-disk cache
    CPUDispatcher   ~155 s       ~0.8 s         ~23 MiB   (1 kernel file)
    proxy           ~168 s       ~114 s (*)     ~1.2 MiB  (kernel only)
    proxy (cached)  ~172 s       ~3.8 s         ~57 MiB   (~4000 files)

The CPUDispatcher kernel inlines all N distinct bodies -> one ~23 MiB cache,
~155 s cold compile, fast ~0.8 s warm restart. proxy lowers each node to an
external call, shrinking the *kernel* cache ~20x to ~1.2 MiB -- but (*) with
uncached formulas every process recompiles all N (warm restart ~114 s, worse than
njit). Making the proxy formulas cache=True fixes warm (~3.8 s), but with N
genuinely-distinct formulas each caches separately, so the *total* footprint
balloons to ~57 MiB across ~4000 files -- larger than the monolithic njit cache.
So "proxy = smaller cache" holds for the kernel, but reverses for the whole
on-disk footprint once the formulas are distinct and made cacheable. proxy also
needs an explicit signature + a named, source-resolvable function per formula.
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

SIG = float64[:](float64[:], float64[:])


# --- formula generation ------------------------------------------------------
# Each node's formula takes (a, b) -> 1-d float64 array, where `a` is the
# previous node's value (fed in 1000x deep) and `b` is a fixed external. For the
# chain to stay finite, the `a`-dependence must be NON-AMPLIFYING: a bounded
# transcendental of `a`, or an affine `ca*a` with |ca| < 1 (contractive). `b` is
# constant down the chain, so any O(1) function of `b` is safe. Roughly a third
# of the nodes use an expensive transcendental (which doubles as the stabilizer).
#
# To make every node a genuinely DISTINCT numba compile (not 4 ops cycled), the N
# functions are emitted into a real importable module, one per source line:
# numba keys its on-disk cache on (file, lineno), so distinct lines => distinct
# compiles even where two bodies are structurally similar. A real module file
# also keeps inspect.getsource/getmodule working, which proxy lowering and
# compile_kernel's content-addressed cache both require.

def _body(i, profile):
    """Deterministic, value-bounded, distinct formula body for node ``i``.

    ``profile`` sets the per-node cost mix: 'cheap' (affine only -> dispatch-
    bound), 'expensive' (transcendental only -> compute-bound), or 'mixed'
    (~1/3 expensive). Every form is non-amplifying in ``a`` so the chain stays
    finite; ``b`` is the fixed external, so any O(1) function of it is safe.
    """
    ca = round(0.30 + (i % 13) * 0.05, 3)   # 0.30..0.90  (|ca| < 1 -> contractive)
    cb = round(0.10 + (i % 7) * 0.05, 3)    # 0.10..0.40
    cc = round(-0.5 + (i % 11) * 0.1, 3)    # small offset
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
    elif i % 3 == 0:                         # mixed: ~1/3 expensive
        a_forms, b_forms = exp_a, exp_b
    else:
        a_forms, b_forms = cheap_a, cheap_b
    return f"{a_forms[i % len(a_forms)]} + {b_forms[i % len(b_forms)]}"


def python_node_indices(n_nodes, k):
    """Indices of the ``k`` evenly-spaced interior nodes to make non-jittable.

    ``i * (n_nodes // (k + 1))`` for ``i = 1..k``; empty when ``k <= 0``. Each
    such node's formula gains a ``json.dumps`` first statement that numba cannot
    type, so ``compile_kernel`` demotes it to a Python step and the surrounding
    jit nodes fuse into separate segments around it.
    """
    if k <= 0:
        return frozenset()
    step = n_nodes // (k + 1)
    return frozenset(i * step for i in range(1, k + 1))


_FORMULAS_CACHE = {}


def _atomic_write(path, text):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _formulas_dir():
    user = re.sub(r"[^0-9A-Za-z._-]", "_", getpass.getuser())
    d = pathlib.Path(tempfile.gettempdir()) / f"ck_bench_{user}"
    d.mkdir(mode=0o700, exist_ok=True)
    if hasattr(os, "getuid"):
        if d.stat().st_uid != os.getuid():
            raise RuntimeError(f"{d} exists but is not owned by the current user")
        d.chmod(0o700)  # a dir created mode-0755 by an older version stays self-only
    return d


def load_formulas(n_nodes, profile, python_nodes=frozenset()):
    """Return ``[f0 .. f{n-1}]``: N genuinely-distinct arithmetic formulas.

    Generated into a real importable module (one function per source line, so
    each is a distinct numba compile and inspect can recover its source). The
    module path is content-addressed by ``(profile, n_nodes)`` so subprocess
    workers in the compile report regenerate an identical module -> stable
    cross-process cache.

    ``python_nodes`` (a set of node indices) makes those nodes non-jittable:
    their body keeps the same arithmetic but is prefixed with a ``json.dumps``
    call numba cannot compile, so ``compile_kernel`` runs them as Python steps.
    Empty (the default) reproduces the original all-jittable module byte for
    byte, including its name, so the two paths never share a stale cache.
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
            lines.append('    json.dumps({"k": 1})')   # untypeable -> Python step
        lines += [f"    return {_body(i, profile)}", "", ""]
    text = "\n".join(lines)
    suffix = "_py" + "-".join(map(str, sorted(python_nodes))) if python_nodes else ""
    mod_name = f"_ck_bench_formulas_{profile}_{n_nodes}{suffix}"
    path = _formulas_dir() / f"{mod_name}.py"
    # Only (re)write when content actually changes: rewriting bumps the file's
    # mtime, which invalidates numba's source-stamp and defeats the cross-process
    # formula cache (the warm proxy_cached restart would recompile every body).
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
    if kind == "njit":
        return njit(fn)                                      # CPUDispatcher binding
    if kind == "proxy":
        return proxy(SIG)(fn)                                # external-call binding
    if kind == "proxy_cached":
        return proxy(SIG, jit_options={"cache": True})(fn)   # + cacheable formula
    raise ValueError(kind)


def build_graph(n_nodes, kind, formulas, python_nodes=frozenset()):
    """Chain of ``n_nodes`` distinct nodes; returns ``(graph, required_qual)``.

    Nodes in ``python_nodes`` keep their plain-Python formula (no ``njit``
    wrap), so a non-jittable body runs in Python under both CompiledGraph and
    compile_kernel's discovery. Empty (the default) wraps every node per
    ``kind``, unchanged.
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
    rng = numpy.random.default_rng(seed)
    return rng.standard_normal(size), rng.standard_normal(size)


def reference_numpy(formulas, a, b):
    """Pure-numpy evaluation of the same chain (correctness + vectorized baseline)."""
    v = formulas[0](a, b)
    for f in formulas[1:]:
        v = f(v, b)
    return v


def best_median(fn, repeats):
    """Warm once, then return (best, median) per-call nanoseconds over ``repeats``."""
    fn()
    samples = [_timed(fn) for _ in range(repeats)]
    samples.sort()
    return samples[0], samples[len(samples) // 2]


def _timed(fn):
    t0 = time.perf_counter_ns()
    fn()
    return time.perf_counter_ns() - t0


def run_perf(n_nodes, size, repeats, profile):
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

    # Correctness: fused kernel and CompiledGraph must match pure numpy.
    fused_out = ck.kernel(a, b)[0]
    vs0 = Values()
    compiled.execute({"ext": {"a": a, "b": b}}, vs0)
    assert numpy.max(numpy.abs(fused_out - ref)) < 1e-9, "fused kernel mismatch"
    assert numpy.max(numpy.abs(vs0.get(last_var).value - ref)) < 1e-9, "CompiledGraph mismatch"

    # Iteration-only: pre-load externals once (private call -- no public entry
    # point assigns externals without also iterating), so the timing isolates the
    # node loop. best_median's warmup primes vs_iter's Value objects, so all timed
    # reps measure pure iteration + formula dispatch without dict-insertion churn.
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
# Injects K evenly-spaced non-jittable nodes into the N-node chain. compile_kernel
# can no longer fuse the whole graph: its first call discovers the non-jittable
# nodes and builds a plan of fused @njit segments with Python steps between them.
# This reports the discovery (first-call) cost and the steady-state per-node cost
# of the segmented plan against CompiledGraph on the same graph.

def run_python_nodes_mode(n_nodes, size, repeats, profile, k):
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

    # First call IS discovery: warm-up + probe that resolves the partition and
    # builds the segmented plan. Every later call runs that plan.
    t0 = time.perf_counter()
    first = ck.kernel(a, b)
    t_first = time.perf_counter() - t0

    rep = ck.partition
    n_seg = len([s for s in rep.segments if s.kind == "jit"])
    n_py = len(rep.python_nodes)

    # Correctness: segmented kernel and CompiledGraph must match pure numpy.
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


# --- compile / cache report (his CPUDispatcher-vs-proxy question) ------------
# Each measurement runs in a fresh subprocess with an isolated NUMBA_CACHE_DIR so
# the cache is clean and measurable; the same process re-runs warm to show the
# cross-process ("post-cache") reload cost.

def _cache_size(cache_dir):
    files = [p for p in pathlib.Path(cache_dir).rglob("*") if p.suffix in (".nbi", ".nbc")]
    return sum(p.stat().st_size for p in files), len(files)


def _compile_worker(n_nodes, kind, profile):
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
                # cwd=repo_root so `-m test.compile_kernel_benchmark` resolves
                # regardless of where the parent was launched from.
                subprocess.run(
                    [sys.executable, "-m", "test.compile_kernel_benchmark",
                     "--_worker", kind, "--nodes", str(n_nodes),
                     "--profile", profile],
                    cwd=repo_root, env=env, check=True)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)


# --- recompute report (the --recompute mode) ---------------------------------
# Seeds a store once with a full call, then drives many `recompute` calls that
# change a small input subset, and contrasts the steady-state cost against a full
# `kernel(...)` re-run (the same change applied as a from-scratch evaluation each
# iteration) and the interpreted `CompiledGraph.recompute`. Three regimes probe
# the three places recompute can win, break even, or lose:
#
#   deep chain      -- a long single-input chain. A change to the FIRST node has a
#                      cone == the whole chain, so recompute cannot beat a full
#                      re-run there. To show the real win we change a LATE INTERIOR
#                      node (an interior override): its downstream cone is a short
#                      SUFFIX of the chain (the last ~50 nodes), so recompute re-runs
#                      only that suffix while the full re-run still evaluates all N
#                      nodes. The win therefore grows with N for a fixed suffix.
#   diamond         -- x -> {a, b} -> u. Changing the shared root x re-affects the
#                      whole small graph; this regime measures the boundary-
#                      persistence machinery cost on a join, not a fusion win.
#   fan-out columns -- K independent input->chain->output columns. Each recompute
#                      changes a DIFFERENT single column input, cycling through all
#                      K so the bounded cone-plan cache is exercised (and, when
#                      K exceeds the cap, thrashes); recompute touches one column's
#                      cone while a full re-run evaluates all K columns.


def _affine(c, d):
    """Distinct contractive affine njit formula ``a -> c*a + d`` (|c| < 1)."""
    def f(a):
        return c * a + d
    return njit(f)


def build_recompute_chain(n_nodes, suffix=50):
    """Single-input chain of ``n_nodes`` contractive affine nodes over float64 arrays.

    Returns ``(graph, required, late_name)``. ``late_name`` is the qual_name of a node
    ``suffix`` steps from the end: overriding it makes the affected cone a short suffix
    of the chain, so recompute re-runs only that suffix.
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
    """The join graph x -> {a, b} -> u; returns ``(graph, required, ext)``."""
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
    """``width`` independent columns, each an ``ext.c{k}`` -> depth-long affine chain
    -> output ``vars.o{k}``. Returns ``(graph, required, ext, col_inputs)`` where
    ``col_inputs`` is the list of per-column external (src, name) change keys."""
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
    """Return a 0-arg callable that does ONE interpreted recompute step (warm store).

    ``next_change`` is a 0-arg getter returning the next ``changed`` dict (e.g. the
    cyclic getter), so the interpreted path cycles the same change sequence as the
    kernel path."""
    req = required if isinstance(required, list) else [required]
    cg = graph.compile(req)
    values = Values()
    cg.execute(seed_ext, values)

    def step():
        cg.recompute(next_change(), values)
    return step


def run_recompute_mode(size, repeats, chain_n, fan_width, fan_depth):
    a, _ = make_externals(size)

    def cyclic(values):
        i = [0]

        def nxt():
            v = values[i[0] % len(values)]
            i[0] += 1
            return v
        return nxt

    rows = []

    # --- deep chain: late interior override -> short suffix cone ------------------
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
    ck.recompute(nxt())                                  # warm: cone compile
    ck.recompute(nxt())
    steady = best_median(lambda: ck.recompute(nxt()), repeats)[0]
    # Honest full-re-run baseline: the bare kernel recomputing the whole graph from
    # the seed external -- exactly the work recompute's suffix cone avoids. (A
    # from-scratch kernel that took the overridden interior as an external would be a
    # different, smaller graph, so it is not the apples-to-apples baseline.)
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
    # Warm a full cycle so every column's cone plan is compiled and cached before
    # timing. When fan_width <= cap this is a clean steady state (all hits); when
    # fan_width > cap the LRU evicts and the timed loop pays re-fusion -- the
    # cache-thrash degradation case, visible as a slow steady-state number.
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
    p.add_argument("--_worker", help=argparse.SUPPRESS)  # internal: one compile measurement
    args = p.parse_args()

    # Graph.compile's topological sort is a recursive DFS; a depth-N chain needs
    # headroom over the default ~1000 limit. Set here (not at import) so merely
    # importing this module never perturbs the process-wide recursion limit.
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
