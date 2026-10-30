"""Cross-process tests of numbox's cached code for graphs and structs, and of the hashes that name that code.

The tests cover the graph maker and the derives of ``make_graph``, the methods of ``make_structref``, ``hash_type`` and
the fallback fingerprint of ``digest``. Most tests run a probe in two or three child processes that share one
``NUMBA_CACHE_DIR``. The input that changes between the runs is one that numba's cache key does not cover: a module
global, an ``ns`` value, a jit option, or a global that a node's dispatcher reads. The tests assert that a later process
computes the new result instead of loading a stale entry, and that a graph compiled without a cache leaves no cache
files behind. The digest test compares two hash seeds instead. numbox's own part of the cache key is tested in
``test_cache_key.py``.
"""
import os
import subprocess
import sys
import textwrap


def _run_probe(probe, env):
    r = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert r.returncode == 0, f"probe failed:\n{r.stderr}"
    return r.stdout.strip()


def test_make_graph_derive_reading_global_is_not_stale_across_processes(tmp_path):
    """A ``make_graph`` derive that reads a module global returns the new value after the global changes.

    numba's cache key for the derive covers its bytecode and closure, not the global. A cached derive would therefore
    return the first process's value, 2.0 instead of 3.0, without a warning. numbox compiles a derive that reads its
    module's globals without a cache.
    """
    probe = tmp_path / "graph_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        from numbox.core.work.builder import End, Derived, make_graph

        G = float(os.environ["G_VAL"])
        reg = {}
        x = End(name="x", init_value=1.0, registry=reg)

        def derive_y(x):
            return x * G

        y = Derived(name="y", init_value=0.0, derive=derive_y, sources=(x,), registry=reg)
        access = make_graph(y, registry=reg)
        access.y.calculate()
        print(access.y.data)
    '''), encoding="utf-8")

    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    env["G_VAL"] = "2.0"
    assert _run_probe(probe, env) == "2.0"  # The first process fills the empty cache.

    env["G_VAL"] = "3.0"
    got = _run_probe(probe, env)  # The second process shares the cache and reads a different global.
    assert got == "3.0", f"derive served a stale cached global: got {got}, expected 3.0"


def test_make_structref_method_ns_value_is_not_stale_across_processes(tmp_path):
    """A ``make_structref`` method that reads a value passed through ``ns`` returns the new value after it changes.

    The method's source is the same in both processes, and only the ``ns`` value differs. The hash in the name of the
    method's jitted helper covers the ``ns`` values that the method reads, so the generated code and the name of its
    cache anchor change with the value, and numba compiles the method again. A hash of the source alone would match,
    and numba would load the code compiled with the first value: 6.0 instead of 10.0.
    """
    probe = tmp_path / "struct_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        from numba.experimental.structref import register
        from numba.core.types import StructRef
        from numbox.utils.highlevel import make_structref

        @register
        class GStructTypeClass(StructRef):
            pass

        G = float(os.environ["G_VAL"])

        def scale(self):
            return self.x * G

        make_g = make_structref(
            "GStruct", ("x",), GStructTypeClass,
            struct_methods={"scale": scale}, ns={"G": G},
        )
        s = make_g(2.0)
        print(s.scale())
    '''), encoding="utf-8")

    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    env["G_VAL"] = "3.0"
    assert _run_probe(probe, env) == "6.0"  # The first process fills the empty cache.

    env["G_VAL"] = "5.0"
    got = _run_probe(probe, env)  # The second process shares the cache and passes a different ns value.
    assert got == "10.0", f"method served a stale ns value: got {got}, expected 10.0"


def test_digest_fallback_is_pythonhashseed_stable(tmp_path):
    """``digest`` gives the same hash under two hash seeds for a function that holds a constant set of strings.

    The function reads the ``@intrinsic`` ``_call_lib_func``, so the strict fingerprint fails, and ``digest`` uses the
    best-effort fingerprint, which sorts the elements of a set. A hash of the bare code object would include the
    frozenset in its iteration order, which depends on ``PYTHONHASHSEED``. The digest would then differ in every
    process, every process would miss the cache, and the cache directory would grow without bound.
    """
    probe = tmp_path / "digest_probe.py"
    probe.write_text(textwrap.dedent('''
        from numbox.core.bindings.call import _call_lib_func
        from numbox.utils.digest import digest

        def cb(x):
            s = {"alpha", "beta", "gamma", "delta", "epsilon"}
            # references the @intrinsic _call_lib_func -> strict walker aborts ->
            # the set-bearing body goes through the best-effort fallback.
            return _call_lib_func("cos", (x,)) if "alpha" in s else 0.0

        print(digest("StateType", (cb,)))
    '''), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    env["PYTHONHASHSEED"] = "0"
    d0 = _run_probe(probe, env)
    env["PYTHONHASHSEED"] = "123456789"
    d1 = _run_probe(probe, env)
    assert d0 == d1, f"digest fallback leaked set iteration order: {d0} != {d1}"


def test_make_graph_kernel_is_not_stale_across_jit_flags(tmp_path):
    """A graph built under the default ``error_model`` does not load code compiled under ``error_model="numpy"``.

    Under the numpy model, ``1.0 / 0.0`` evaluates to ``inf``. Under the default python model, the division raises
    ``ZeroDivisionError``; the probe catches it, and the node keeps its initial value, 0.0. The source of the graph
    maker shows the jit options only as ``@njit(**jit_options)``, and numba names a derive's cache files after the
    derive's own source and qualified name. Names that did not fold the options would therefore be the same under both
    models, and the second process would print the first one's ``inf``. The graph maker's fingerprint folds the jit
    options in effect, and the derive compiles through an anchor whose name folds them as well.
    """
    probe = tmp_path / "flags_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        from numba.core.types import float64
        from numbox.core.work.builder import End, Derived, make_graph

        opts = {"cache": True}
        if os.environ["ERROR_MODEL"] == "numpy":
            opts["error_model"] = "numpy"

        reg = {}
        num = End(name="num", init_value=1.0, ty=float64, registry=reg)
        den = End(name="den", init_value=0.0, ty=float64, registry=reg)

        def derive_ratio(num, den):
            return num / den

        ratio = Derived(name="ratio", init_value=0.0, derive=derive_ratio,
                        sources=(num, den), ty=float64, registry=reg)
        access = make_graph(ratio, registry=reg, jit_options=opts)
        try:
            access.ratio.calculate()
        except ZeroDivisionError:
            pass
        print(access.ratio.data)
    '''), encoding="utf-8")

    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    env["ERROR_MODEL"] = "numpy"
    assert _run_probe(probe, env) == "inf"  # The first process fills the empty cache.

    env["ERROR_MODEL"] = "python"
    got = _run_probe(probe, env)  # The second process shares the cache and uses the default model.
    assert got == "0.0", (
        f"kernel/derive served a binary compiled under another error_model: "
        f"got {got}, expected 0.0"
    )

    # A third run of the same configuration must add no index file, so the graph maker and the derive keep the names
    # under which the second run cached them.
    cache_root = tmp_path / "nbcache"
    before = sorted(p.name for p in cache_root.rglob("*.nbi"))
    assert _run_probe(probe, env) == "0.0"
    after = sorted(p.name for p in cache_root.rglob("*.nbi"))
    assert before == after, f"identical rebuild recompiled instead of reusing: {set(after) - set(before)}"


def test_dispatcher_typed_node_hash_is_process_stable(tmp_path):
    """``hash_type`` of a dispatcher type, and the fingerprint of a graph that holds one, are the same in every process.

    numba's text for a dispatcher type, such as ``type(CPUDispatcher(<function f at 0x7f...>))``, holds an address that
    differs between processes. A graph maker named after that text would receive a new name in every process, so each
    lookup would miss the cache and leave new cache files behind. numbox fingerprints a dispatcher by the content of
    its Python function instead. The test also checks that a change to a global that the dispatcher reads changes the
    hash, so the stability does not come from ignoring the content.
    """
    probe = tmp_path / "disp_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        import numbox.core.work.builder as builder_mod
        from numba import njit
        from numba.core.types import float64
        from numbox.core.work.builder import End, Derived, make_graph
        from numbox.utils.highlevel import hash_type

        captured = []
        orig = builder_mod._kernel_fingerprint

        def spy(*a, **k):
            h, ok = orig(*a, **k)
            captured.append(h)
            return h, ok

        builder_mod._kernel_fingerprint = spy

        BUMP = float(os.environ["BUMP"])

        @njit(float64(float64))
        def helper(x):
            return x + BUMP

        reg = {}
        x = End(name="x", init_value=1.0, ty=float64, registry=reg)
        h = End(name="h", init_value=helper, registry=reg)

        def derive_y(x, h):
            return h(x)

        y = Derived(name="y", init_value=0.0, derive=derive_y, sources=(x, h),
                    ty=float64, registry=reg)
        make_graph(y, registry=reg, jit_options={"cache": True})
        print(hash_type(helper._numba_type_), captured[0])
    '''), encoding="utf-8")

    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    env["BUMP"] = "1.0"
    first = _run_probe(probe, env)
    second = _run_probe(probe, env)
    assert first == second, (
        f"Dispatcher-typed hash is process-unstable: {first} != {second}; the "
        "kernel name would change every run and never cache-hit"
    )

    env["BUMP"] = "2.0"
    changed = _run_probe(probe, env)
    assert changed != first, (
        "a changed dispatcher body did not re-key; process-stability must not be "
        "bought by ignoring the content"
    )


def test_unfingerprintable_derive_makes_kernel_uncached_no_growth(tmp_path):
    """A derive that reads a global without a canonical form makes the graph maker compile without a cache.

    The global here is a numba type used as a cast. The strict fingerprint cannot handle it, so numbox gives the derive
    the best-effort fingerprint, which is the same in every process. That fingerprint does not change when the value
    without a canonical form changes, so numbox compiles the derive and the graph maker without a cache, and three runs
    leave no ``builder._make*.nbc`` file. A name built from the derive's ``id()`` would instead change in every process
    and add a cache entry each time.
    """
    probe = tmp_path / "unfp_derive_probe.py"
    probe.write_text(textwrap.dedent('''
        from numba import float64 as nb_float64
        from numbox.core.work.builder import End, Derived, make_graph

        reg = {}
        x = End(name="x", init_value=2.0, ty=nb_float64, registry=reg)

        def derive_y(x):
            return nb_float64(x) * 2.0   # LOAD_GLOBAL of a numba Type -> unfingerprintable

        y = Derived(name="y", init_value=0.0, derive=derive_y, sources=(x,),
                    ty=nb_float64, registry=reg)
        access = make_graph(y, registry=reg, jit_options={"cache": True})
        access.y.calculate()
        print(access.y.data)
    '''), encoding="utf-8")

    env = dict(os.environ)
    cache = tmp_path / "nbcache"
    env["NUMBA_CACHE_DIR"] = str(cache)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    counts = []
    for _ in range(3):
        assert _run_probe(probe, env) == "4.0"  # The value is correct, and the maker compiles without a cache.
        counts.append(sum(1 for _ in cache.rglob("builder._make*.nbc")))
    assert counts[1] == counts[0] and counts[2] == counts[0], (
        f"an un-fingerprintable derive reminted the kernel per process: "
        f"builder._make*.nbc counts {counts}")
    assert counts[0] == 0, (
        f"an un-fingerprintable-derive kernel must be uncached (best-effort hash is "
        f"content-blind), but {counts[0]} _make .nbc were written")


def test_intrinsic_referencing_node_makes_kernel_uncached_no_growth(tmp_path):
    """A node holding a dispatcher whose body has no strict fingerprint makes the graph maker compile without a cache.

    The dispatcher's body calls a numba type as a cast, which the strict fingerprint cannot handle. ``_type_identity``
    fingerprints the function best-effort, with a result that is the same in every process, and it reports every
    dispatcher type as uncacheable. ``make_graph`` therefore compiles the maker without a cache, and three runs leave
    no ``builder._make*.nbc`` file. numba's mangled name for a dispatcher type holds an address, so a maker named after
    it would add a cache entry in every process.
    """
    probe = tmp_path / "intrin_node_probe.py"
    probe.write_text(textwrap.dedent('''
        from numba import njit, float64 as nb_float64
        from numbox.core.work.builder import End, Derived, make_graph

        @njit(nb_float64(nb_float64))
        def helper(x):
            return nb_float64(x) * 2.0    # numba Type cast in the body -> unfingerprintable

        reg = {}
        x = End(name="x", init_value=3.0, ty=nb_float64, registry=reg)
        h = End(name="h", init_value=helper, registry=reg)

        def derive_y(x, h):
            return h(x)

        y = Derived(name="y", init_value=0.0, derive=derive_y, sources=(x, h),
                    ty=nb_float64, registry=reg)
        access = make_graph(y, registry=reg, jit_options={"cache": True})
        access.y.calculate()
        print(access.y.data)
    '''), encoding="utf-8")

    env = dict(os.environ)
    cache = tmp_path / "nbcache"
    env["NUMBA_CACHE_DIR"] = str(cache)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    counts = []
    for _ in range(3):
        assert _run_probe(probe, env) == "6.0"
        counts.append(sum(1 for _ in cache.rglob("builder._make*.nbc")))
    assert counts[1] == counts[0] and counts[2] == counts[0], (
        f"a Dispatcher-typed node with an un-fingerprintable body reminted the "
        f"kernel per process: builder._make*.nbc counts {counts}")
    assert counts[0] == 0, (
        f"the kernel must be uncached when a node type has only a best-effort "
        f"identity, but {counts[0]} _make .nbc were written")


def test_dispatcher_typed_component_is_uncached_no_growth(tmp_path):
    """A graph whose source node holds an ``njit`` dispatcher leaves no cache files for the graph maker or the derive.

    numba cannot reuse a cached signature that holds a dispatcher type in another process: even under a name that is
    the same in every process, the index misses in every run, and numba appends a new ``.nbc`` file. numbox therefore
    compiles the graph maker and the derive without a cache. The helper's body has a strict fingerprint, so this is
    the ordinary case, not a degenerate one.

    The test covers only the code that numbox names. numba's own cache of the ``Work`` methods, such as
    ``work.Work.calculate``, ``work.Work.data`` and ``work._calculate_``, still grows by one entry per process for such
    a node, because those ``@njit(cache=True)`` methods are specialized on a ``Work`` type that holds the dispatcher
    type. That growth is a limitation of numba, outside this test, and it never loads a stale binary.
    """
    probe = tmp_path / "disp_component_probe.py"
    probe.write_text(textwrap.dedent('''
        from numba import njit, float64 as nb_float64
        from numbox.core.work.builder import End, Derived, make_graph

        @njit(nb_float64(nb_float64))
        def helper(x):
            return x + 1.0             # fully fingerprintable body

        reg = {}
        x = End(name="x", init_value=1.0, ty=nb_float64, registry=reg)
        h = End(name="h", init_value=helper, registry=reg)

        def derive_y(x, h):
            return h(x)

        y = Derived(name="y", init_value=0.0, derive=derive_y, sources=(x, h),
                    ty=nb_float64, registry=reg)
        access = make_graph(y, registry=reg, jit_options={"cache": True})
        access.y.calculate()
        print(access.y.data)
    '''), encoding="utf-8")

    env = dict(os.environ)
    cache = tmp_path / "nbcache"
    env["NUMBA_CACHE_DIR"] = str(cache)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    kernel_counts, derive_counts = [], []
    for _ in range(3):
        assert _run_probe(probe, env) == "2.0"
        kernel_counts.append(sum(1 for _ in cache.rglob("builder._make*.nbc")))
        derive_counts.append(sum(1 for _ in cache.rglob("_derive_*.nbc")))
    assert kernel_counts == [0, 0, 0], (
        f"a Dispatcher-typed node left kernel cache pairs behind (numba cannot "
        f"cross-process-cache it): builder._make*.nbc counts {kernel_counts}")
    assert derive_counts == [0, 0, 0], (
        f"a Dispatcher-typed source left derive cache pairs behind: "
        f"_derive_*.nbc counts {derive_counts}")


def test_nested_dispatcher_node_kernel_and_derive_uncached_no_growth(tmp_path):
    """A node whose type is a tuple that holds an ``njit`` dispatcher leaves no cache files for the maker or the derive.

    The tuple type is not a ``Dispatcher``, so ``_type_identity`` also looks for ``CPUDispatcher(`` in the type's text
    and reports the type as uncacheable. Without that check, numbox would hash the type by numba's mangled name, which
    holds an address. The maker and the derive would then add cache entries in every process. numba's own cache of the
    ``Work`` methods still grows for such a node, as the previous test describes.
    """
    probe = tmp_path / "nested_disp_probe.py"
    probe.write_text(textwrap.dedent('''
        from numba import njit, float64 as nb_float64
        from numbox.core.work.builder import End, Derived, make_graph

        @njit(nb_float64(nb_float64))
        def helper(x):
            return x + 1.0

        reg = {}
        # a heterogeneous tuple node: Tuple(float64, Dispatcher) -- nested dispatcher
        p = End(name="p", init_value=(2.0, helper), registry=reg)

        def derive_y(p):
            return p[0] + 1.0

        y = Derived(name="y", init_value=0.0, derive=derive_y, sources=(p,),
                    ty=nb_float64, registry=reg)
        access = make_graph(y, registry=reg, jit_options={"cache": True})
        access.y.calculate()
        print(access.y.data)
    '''), encoding="utf-8")

    env = dict(os.environ)
    cache = tmp_path / "nbcache"
    env["NUMBA_CACHE_DIR"] = str(cache)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    kernel_counts, derive_counts = [], []
    for _ in range(3):
        assert _run_probe(probe, env) == "3.0"
        kernel_counts.append(sum(1 for _ in cache.rglob("builder._make*.nbc")))
        derive_counts.append(sum(1 for _ in cache.rglob("_derive_*.nbc")))
    assert kernel_counts == [0, 0, 0], (
        f"a nested-Dispatcher node left kernel cache pairs behind: {kernel_counts}")
    assert derive_counts == [0, 0, 0], (
        f"a nested-Dispatcher node left derive cache pairs behind: {derive_counts}")


def test_module_state_derive_keeps_kernel_cacheable(tmp_path):
    """A derive that reads a module global compiles without a cache, but the graph maker stays cacheable.

    The maker's name folds the global's value, so an unchanged second run loads the maker from the cache instead of
    adding an entry. The derive has a strict fingerprint, and ``_derive_fingerprint`` returns ``kernel_safe`` True for
    it, so the uncached path of a derive without a fingerprint does not apply.
    """
    probe = tmp_path / "modstate_probe.py"
    probe.write_text(textwrap.dedent('''
        from numba import float64 as nb_float64
        from numbox.core.work.builder import End, Derived, make_graph

        G = 2.0    # plain module global, canonicalizable -> derive fingerprints

        reg = {}
        x = End(name="x", init_value=3.0, ty=nb_float64, registry=reg)

        def derive_y(x):
            return x * G

        y = Derived(name="y", init_value=0.0, derive=derive_y, sources=(x,),
                    ty=nb_float64, registry=reg)
        access = make_graph(y, registry=reg, jit_options={"cache": True})
        access.y.calculate()
        print(access.y.data)
    '''), encoding="utf-8")

    env = dict(os.environ)
    cache = tmp_path / "nbcache"
    env["NUMBA_CACHE_DIR"] = str(cache)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    counts = []
    for _ in range(3):
        assert _run_probe(probe, env) == "6.0"
        counts.append(sum(1 for _ in cache.rglob("builder._make*.nbc")))
    assert counts[0] > 0, "a module-state-reading graph's kernel must still cache"
    assert counts[1] == counts[0] and counts[2] == counts[0], (
        f"a module-state kernel reminted across processes with an unchanged global: {counts}")
