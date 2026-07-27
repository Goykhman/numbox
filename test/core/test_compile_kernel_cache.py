"""compile_kernel cache-correctness regression tests.

Folding jit flags into the outer ``_kernel_<digest>`` name cannot, by itself,
invalidate an independently numba-cached unit the kernel links against, nor
cover values the digest fingerprint misses. Each test pins one such hazard --
fingerprint-level where the mechanism is deterministic, and a two-process
shared-``NUMBA_CACHE_DIR`` probe where the failure is cross-process.
"""
import os
import subprocess
import sys
import textwrap
import types

import pytest
from numba import njit, typeof, vectorize
from numba.core.caching import NullCache

from numbox.core.variable.compile_kernel import (
    _compile, _formula_fingerprint, _is_self_cached, _references_self_cached,
)
from numbox.utils.fingerprint import _fingerprint_function


# A module-level cache=True dispatcher has a real locator (this file), so numba
# enables its FunctionCache -- the self-cached formula the fix must detect.
_cached_formula = njit(cache=True)(lambda y: y + 1.0)
_plain_formula = njit(lambda y: y + 1.0)                     # cache=False -> safe
_KERNEL_SRC = "def _kernel(a0):\n    return (f_x(a0),)\n"


# A plain formula that CALLS a module-level cache=True dispatcher (rather than
# being one) -- the transitive case CK1's top-level check would miss.
def _formula_calls_cached(x):
    return _cached_formula(x)


# The cached dispatcher reached only through a default argument, or nested in a
# container global -- channels the digest folds, so the guard must too.
def _formula_default_arg(x, h=_cached_formula):
    return h(x)


_HELPERS = (_cached_formula,)


def _formula_container_global(x):
    return _HELPERS[0](x)


# The cached dispatcher reached as a module attribute (helpers.helper(x)).
_helpers_mod = types.ModuleType("test_ck_helpers_mod")
_helpers_mod.helper = _cached_formula


def _formula_module_attr(x):
    return _helpers_mod.helper(x)


def _run_probe(probe, env):
    r = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert r.returncode == 0, f"probe failed:\n{r.stderr}"
    return r.stdout.strip()


def _shared_cache_env(tmp_path):
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    return env


def _body(x):
    # fastmath's nnan/reassociation makes (x + 1e308) - 1e308 collapse to x
    # instead of 0.0 -- a visible, one-line flag-sensitive result.
    return (x + 1e308) - 1e308


# DUFunc (@vectorize) formula targetoptions must enter the digest
# ---------------------------------------------------------------------------

def test_dufunc_targetoptions_distinguish_fingerprint():
    """A @vectorize formula's targetoptions must enter the fingerprint, else two
    kernels differing only in a DUFunc flag share one digest and the outer cached
    kernel is served stale on a cache hit."""
    f_plain = vectorize(["float64(float64)"], nopython=True)(_body)
    f_fast = vectorize(["float64(float64)"], nopython=True, fastmath=True)(_body)
    fp_plain, cacheable_plain = _formula_fingerprint(f_plain)
    fp_fast, _ = _formula_fingerprint(f_fast)
    assert cacheable_plain is True
    assert fp_plain != fp_fast, "DUFunc fastmath flag absent from fingerprint (digest collision)"


def test_dufunc_fastmath_flip_not_stale_across_processes(tmp_path):
    """Two processes sharing one cache dir, a @vectorize formula whose fastmath
    flips between runs: the second process must recompile, not serve the first's
    stale binary. Without the targetoptions fold both runs share one digest."""
    probe = tmp_path / "dufunc_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        import numpy as np
        from numba import vectorize
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel

        fm = bool(int(os.environ["FASTMATH"]))
        f = vectorize(["float64(float64)"], nopython=True, fastmath=fm)(
            lambda x: (x + 1e308) - 1e308)
        g = Graph(
            variables_lists={"variables": [
                {"name": "out", "inputs": {"y": "basket"}, "formula": f}]},
            external_source_names=["basket"],
        )
        ck = compile_kernel(g, "variables.out")
        print(float(np.asarray(ck.kernel(1.0)[0]).ravel()[0]))
    '''), encoding="utf-8")

    env = _shared_cache_env(tmp_path)
    env["FASTMATH"] = "0"
    assert _run_probe(probe, env) == "0.0"             # cold: caches under its digest
    env["FASTMATH"] = "1"
    got = _run_probe(probe, env)                        # shared cache, flag flipped
    assert got == "1.0", f"DUFunc fastmath flip served a stale cached kernel: got {got}, expected 1.0"


# A value read through module indirection (cfg.SCALE) must enter the fingerprint
# ---------------------------------------------------------------------------

def _func_reading_module_attr(scale_value):
    """Build a fresh function ``f(x): return x * cfg.SCALE`` whose globals hold a
    module ``cfg`` with ``SCALE = scale_value`` -- the one-level indirection that
    was invisible to the fingerprint."""
    cfg = types.ModuleType("cfg_probe_mod")
    cfg.SCALE = scale_value
    code = compile("def f(x):\n    return x * cfg.SCALE\n", "<probe>", "exec").co_consts[0]
    return types.FunctionType(code, {"cfg": cfg})


def test_module_attribute_value_change_rekeys_fingerprint():
    """cfg.SCALE is baked into the compiled binary by numba but is one module hop
    from a global, so its value must enter the fingerprint or a change is stale."""
    assert _fingerprint_function(_func_reading_module_attr(2.0), set()) \
        != _fingerprint_function(_func_reading_module_attr(3.0), set())


def test_module_function_attr_stays_a_stable_reference():
    """A module attribute that is a function/ufunc (np.sqrt) is resolved by
    identity, not frozen as data -- it must not be folded (which would recurse
    into numpy) nor make the formula uncacheable."""
    import numpy as np

    def mk():
        code = compile("def f(x):\n    return np.sqrt(x) + 1.0\n", "<p>", "exec").co_consts[0]
        return types.FunctionType(code, {"np": np})

    fp = _fingerprint_function(mk(), set())
    assert fp == _fingerprint_function(mk(), set())     # stable
    assert "sqrt=function" not in fp                    # ufunc not folded as a value


def test_module_constant_change_not_stale_across_processes(tmp_path):
    """Two processes sharing one cache dir, a formula reading a module constant
    whose value changes between runs: the second must recompile, not serve the
    first's baked-in value."""
    probe = tmp_path / "modattr_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        import sys
        import types
        import numpy as np
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel

        cfg = types.ModuleType("ck_cfg_probe")
        cfg.SCALE = float(os.environ["SCALE"])
        sys.modules["ck_cfg_probe"] = cfg
        import ck_cfg_probe  # noqa: E402,F401  -- now a module global of this script

        def scale(y):
            return y * ck_cfg_probe.SCALE

        g = Graph(
            variables_lists={"variables": [
                {"name": "out", "inputs": {"y": "basket"}, "formula": scale}]},
            external_source_names=["basket"],
        )
        ck = compile_kernel(g, "variables.out")
        print(float(np.asarray(ck.kernel(10.0)[0]).ravel()[0]))
    '''), encoding="utf-8")

    env = _shared_cache_env(tmp_path)
    env["SCALE"] = "2.0"
    assert _run_probe(probe, env) == "20.0"            # cold: caches with SCALE=2
    env["SCALE"] = "3.0"
    got = _run_probe(probe, env)                        # shared cache, constant changed
    assert got == "30.0", f"module constant change served a stale kernel: got {got}, expected 30.0"


# A self-cached formula makes the unit uncacheable (no poisoned artifact)
# ---------------------------------------------------------------------------

def test_is_self_cached_detects_only_cache_true_dispatchers():
    assert _is_self_cached(_cached_formula) is True
    assert _is_self_cached(_plain_formula) is False


def test_self_cached_dispatcher_formula_makes_unit_uncacheable():
    """A user @njit(cache=True) formula keeps its own flag/global-blind numba
    cache; folding flags into the outer name cannot invalidate it, and a fresh
    outer would link + serialize the stale inner binary (permanent poisoning).
    The unit must compile uncached, with a warning."""
    assert not isinstance(_cached_formula._cache, NullCache)   # precondition
    with pytest.warns(UserWarning, match="a @njit.cache=True. dispatcher"):
        disp = _compile(_KERNEL_SRC, {"f_x": _cached_formula}, None, None)
    assert isinstance(disp._cache, NullCache), "outer kernel must not be disk-cached"


def test_plain_formula_unit_stays_cacheable():
    """A cache=False formula has no independent on-disk cache and re-keys with the
    flag-folded outer, so the unit must still cache (no false-positive uncaching)."""
    disp = _compile(_KERNEL_SRC, {"f_x": _plain_formula}, None, None)
    assert not isinstance(disp._cache, NullCache), "plain-formula kernel should cache"


def test_references_self_cached_detects_a_cached_callee():
    assert _references_self_cached(_formula_calls_cached, set()) is True
    assert _references_self_cached(_plain_formula, set()) is False


def test_formula_calling_cache_true_dispatcher_makes_unit_uncacheable():
    """A plain formula that CALLS a @njit(cache=True) dispatcher links the callee's
    flag/global-blind binary; the unit must be uncacheable too, else a stale inner
    is serialized into a fresh-digest artifact (poisoning one call deeper)."""
    with pytest.warns(UserWarning, match="reference, a @njit"):
        disp = _compile(_KERNEL_SRC, {"f_x": njit(_formula_calls_cached)}, None, None)
    assert isinstance(disp._cache, NullCache), "unit linking a cache=True callee must not cache"


def test_formula_calling_cache_true_dispatcher_not_poisoned(tmp_path):
    """End-to-end: a plain formula calling a cache=True dispatcher whose flags flip
    between processes must leave NO numbox-owned outer cache artifact (no poisoning
    surface). numba still serves the user dispatcher's own stale binary in-process
    -- an accepted numba limitation, not a numbox defect."""
    probe = tmp_path / "calls_cached_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        import numpy as np
        from numba import njit
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel

        helper = njit(cache=True, error_model=os.environ["EM"])(lambda y: 1.0 / y)

        def f(x):
            return helper(x)

        g = Graph(
            variables_lists={"variables": [
                {"name": "out", "inputs": {"y": "basket"}, "formula": f}]},
            external_source_names=["basket"],
        )
        ck = compile_kernel(g, "variables.out")
        try:
            print(float(np.asarray(ck.kernel(0.0)[0]).ravel()[0]))
        except ZeroDivisionError:
            print("ZeroDivisionError")
    '''), encoding="utf-8")

    env = _shared_cache_env(tmp_path)
    env["EM"] = "numpy"
    _run_probe(probe, env)
    env["EM"] = "python"
    _run_probe(probe, env)
    outer = list((tmp_path / "nbcache").rglob("_kernel_*"))
    assert not outer, f"numbox cached an outer kernel despite a cache=True callee (poisoning): {outer}"


def test_references_self_cached_through_default_argument():
    assert _references_self_cached(_formula_default_arg, set()) is True


def test_references_self_cached_through_container_global():
    assert _references_self_cached(_formula_container_global, set()) is True


def test_references_self_cached_through_module_attribute():
    assert _references_self_cached(_formula_module_attr, set()) is True


def test_default_arg_cached_dispatcher_makes_unit_uncacheable():
    with pytest.warns(UserWarning, match="reference, a @njit"):
        disp = _compile(_KERNEL_SRC, {"f_x": njit(_formula_default_arg)}, None, None)
    assert isinstance(disp._cache, NullCache)


def test_dispatcher_typed_declared_sig_makes_unit_uncacheable():
    """A declared signature carrying a Dispatcher type (a function-typed external)
    makes the unit uncacheable. ``declared_sigs`` folded raw ``repr`` of each type;
    a Dispatcher's ``type(CPUDispatcher(<function f at 0x...>))`` carried an ASLR
    address, orphaning a ``_kernel_`` cache pair per run, and numba cannot
    cross-process-cache a Dispatcher signature anyway. The
    declared types now route through ``_type_identity``, which reports a Dispatcher
    un-cacheable, so ``_compile`` drops the on-disk cache.
    """
    disp = _compile(
        _KERNEL_SRC, {"f_x": _plain_formula}, None, None,
        declared_sigs=((typeof(_plain_formula),),),
    )
    assert isinstance(disp._cache, NullCache), (
        "a Dispatcher-typed declared signature must make the unit uncacheable")


# Chained (cfg.sub.SCALE) and closure-captured module attributes must fold too
# ---------------------------------------------------------------------------

def _func_reading_chained_module_attr(scale):
    cfg = types.ModuleType("cfg_chain_probe")
    sub = types.ModuleType("cfg_chain_probe.sub")
    cfg.sub = sub
    sub.SCALE = scale
    code = compile("def f(x):\n    return x * cfg.sub.SCALE\n", "<p>", "exec").co_consts[0]
    return types.FunctionType(code, {"cfg": cfg})


def test_chained_module_attribute_rekeys_fingerprint():
    assert _fingerprint_function(_func_reading_chained_module_attr(2.0), set()) \
        != _fingerprint_function(_func_reading_chained_module_attr(3.0), set())


def _closure_module_formula(scale):
    mod = types.ModuleType("cmod_closure_probe")
    mod.SCALE = scale

    def make(m):
        def f(x):
            return x * m.SCALE
        return f
    return make(mod)


def test_closure_captured_module_attribute_rekeys_fingerprint():
    assert _fingerprint_function(_closure_module_formula(2.0), set()) \
        != _fingerprint_function(_closure_module_formula(3.0), set())


def _module_attr_dispatcher_formula(other_val):
    """f(x) = cfg.compute(x), where cfg.compute is a Dispatcher capturing the
    module global OTHER -- numba links it and freezes OTHER, so the digest must
    fold the module-attr dispatcher's fingerprint (as it does a direct global)."""
    disp_code = compile("def compute(z):\n    return z + OTHER\n", "<d>", "exec").co_consts[0]
    compute = njit(cache=False)(types.FunctionType(disp_code, {"OTHER": other_val}))
    cfg = types.ModuleType("cfg_disp_fp_probe")
    cfg.compute = compute
    f_code = compile("def f(x):\n    return cfg.compute(x)\n", "<f>", "exec").co_consts[0]
    return types.FunctionType(f_code, {"cfg": cfg})


def test_module_attr_dispatcher_captured_value_rekeys_fingerprint():
    assert _fingerprint_function(_module_attr_dispatcher_formula(10.0), set()) \
        != _fingerprint_function(_module_attr_dispatcher_formula(999.0), set())


def _module_attr_dufunc_formula(other_val):
    """f(x) = cfg.uf(x), where cfg.uf is a @vectorize DUFunc capturing OTHER --
    numba links its loop and freezes OTHER, so the digest must fold it too."""
    wrapped = types.FunctionType(
        compile("def uf(z):\n    return z + OTHER\n", "<d>", "exec").co_consts[0], {"OTHER": other_val})
    uf = vectorize(["float64(float64)"], nopython=True)(wrapped)
    cfg = types.ModuleType("cfg_uf_fp_probe")
    cfg.uf = uf
    f_code = compile("def f(x):\n    return cfg.uf(x)\n", "<f>", "exec").co_consts[0]
    return types.FunctionType(f_code, {"cfg": cfg})


def test_module_attr_dufunc_captured_value_rekeys_fingerprint():
    assert _fingerprint_function(_module_attr_dufunc_formula(10.0), set()) \
        != _fingerprint_function(_module_attr_dufunc_formula(999.0), set())


# A value a module exposes ONLY via a PEP 562 __getattr__ is deliberately NOT
# folded: resolving referenced names via getattr would fire an unrelated module's
# __getattr__ (numpy's) on co-name collisions, importing submodules and emitting
# deprecation warnings on every compile. Documented limitation -- no test asserts
# the lazy-attribute case is covered.


# Result-affecting numba codegen env knobs beyond BOUNDSCHECK enter the digest
# ---------------------------------------------------------------------------

def test_vectorize_env_knob_enters_digest(monkeypatch):
    import numbox.utils.fingerprint as fp
    src = "def _kernel(a0):\n    return (a0,)\n"
    monkeypatch.setattr(fp.numba_config, "LOOP_VECTORIZE", 0, raising=False)
    name0 = _compile(src, {}, None, None).py_func.__name__
    monkeypatch.setattr(fp.numba_config, "LOOP_VECTORIZE", 1, raising=False)
    name1 = _compile(src, {}, None, None).py_func.__name__
    assert name0 != name1, "LOOP_VECTORIZE absent from the cache digest"


# NUMBA_BOUNDSCHECK is an env codegen knob outside the jit flags and numba's key
# ---------------------------------------------------------------------------

def test_boundscheck_env_enters_digest(monkeypatch):
    """NUMBA_BOUNDSCHECK overrides even an explicit boundscheck flag at lowering
    but is in neither the jit flags nor numba's cache key; it must enter the
    digest so a bounds-check flip does not cache-hit the unchecked binary."""
    import numbox.utils.fingerprint as fp
    src = "def _kernel(a0):\n    return (a0,)\n"
    monkeypatch.setattr(fp.numba_config, "BOUNDSCHECK", None)
    name_off = _compile(src, {}, None, None).py_func.__name__
    monkeypatch.setattr(fp.numba_config, "BOUNDSCHECK", 1)
    name_on = _compile(src, {}, None, None).py_func.__name__
    assert name_off != name_on, "BOUNDSCHECK absent from the cache digest"


# Un-canonicalizable jit flags must warn, not silently disable caching
# ---------------------------------------------------------------------------

def test_uncanonicalizable_flags_warn_and_disable_cache():
    from numba.core.compiler import Compiler
    src = "def _kernel(a0):\n    return (a0,)\n"
    with pytest.warns(UserWarning, match="no canonical"):
        disp = _compile(src, {}, {"pipeline_class": Compiler}, None)
    assert isinstance(disp._cache, NullCache)


# The declared-return probe memo must key on flags, not skip a flag-dependent probe
# ---------------------------------------------------------------------------

def _declared_locals_graph():
    """A declared float64 node whose formula returns a local `m`. Under
    ``locals={'m': float32}`` the natural return type becomes float32, which
    violates the declared float64 -- caught by the build-time probe."""
    from numba.core.types import float64
    from numbox.core.variable.variable import Graph, Params

    def f(x):
        m = x
        return m
    g = Graph(variables_lists={"variables": [
        {"name": "a", "inputs": {"x": "e"}, "formula": f, "params": Params(type=float64)},
    ]}, external_source_names=["e"])
    g.external["e"].declare("x", Params(type=float64))
    return g


def test_validated_returns_memo_keys_on_flags():
    """A prior default-flags build must not let a later build under a
    return-type-affecting flag skip the declared-return probe via a flag-blind
    memo key -- else a float32-under-declared-float64 violation is accepted."""
    from numba.core.types import float32
    from numbox.core.variable.compile_kernel import compile_kernel, _validated_returns

    saved = set(_validated_returns)
    _validated_returns.clear()
    try:
        compile_kernel(_declared_locals_graph(), "variables.a")   # default flags: OK, memoized
        with pytest.raises(ValueError, match="float32"):
            compile_kernel(_declared_locals_graph(), "variables.a",
                           jit_options={"locals": {"m": float32}})
    finally:
        _validated_returns.clear()
        _validated_returns.update(saved)
