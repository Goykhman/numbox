"""Tests that numba never loads stale code for a kernel from its cache.

numba finds a cached kernel by the kernel's name, which is a digest of its formulas and of what they read. The digest
must therefore hold every value that changes the compiled code: the target options of a @vectorize formula, the module
attributes that a formula reads, the declared types, and the ``NUMBA_*`` settings that change the emitted code. A
formula that is or calls a dispatcher compiled with ``cache=True`` must leave the kernel uncached, because the kernel's
name cannot make numba refresh that dispatcher's own cache. Most tests compare fingerprints, kernel names or the cache
of a compiled dispatcher in one process. Three run a probe script in two processes that share one ``NUMBA_CACHE_DIR``;
a stale kernel would show as the first process's result or as a cached kernel file. The last test checks that the
record of passed return-type checks is keyed on the jit options. test_compile_kernel.py tests the other inputs of the
digest and the cache in general.
"""
import os
import subprocess
import sys
import textwrap
import types

import pytest
from numba import njit, typeof, vectorize
from numba.core import config as numba_config
from numba.core.caching import NullCache

from numbox.core.variable.compile_kernel import (
    _compile, _formula_fingerprint, _is_self_cached, _references_self_cached,
)
from numbox.utils.fingerprint import _fingerprint_function


# A dispatcher compiled with cache=True in this file has a source file that numba can locate, so numba gives it a
# FunctionCache. _is_self_cached must detect it as a formula with a cache of its own.
_cached_formula = njit(cache=True)(lambda y: y + 1.0)
_plain_formula = njit(lambda y: y + 1.0)  # Without cache=True, the dispatcher has no cache of its own.
# _KERNEL_SRC is the source of a kernel with one formula, f_x, for the tests that call _compile directly.
_KERNEL_SRC = "def _kernel(a0):\n    return (f_x(a0),)\n"


# _formula_calls_cached is a plain formula that calls the cache=True dispatcher instead of being one. _is_self_cached,
# which inspects the formula alone, does not detect it; _references_self_cached follows the call.
def _formula_calls_cached(x):
    return _cached_formula(x)


# The next two formulas reach the cache=True dispatcher only through a default argument or through a tuple held in a
# global. The digest folds both channels, so _references_self_cached must follow them as well.
def _formula_default_arg(x, h=_cached_formula):
    return h(x)


_HELPERS = (_cached_formula,)


def _formula_container_global(x):
    return _HELPERS[0](x)


# _formula_module_attr reaches the cache=True dispatcher as a module attribute, as in helpers.helper(x).
_helpers_mod = types.ModuleType("test_ck_helpers_mod")
_helpers_mod.helper = _cached_formula


def _formula_module_attr(x):
    return _helpers_mod.helper(x)


def _run_probe(probe, env):
    """Run ``probe`` in a new process with the environment ``env``, and return its stdout with ``str.strip`` applied.

    The test fails with the probe's stderr if the probe exits with a status other than 0.
    """
    r = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert r.returncode == 0, f"probe failed:\n{r.stderr}"
    return r.stdout.strip()


def _shared_cache_env(tmp_path):
    """Return a copy of the environment in which every probe uses one cache directory and imports this numbox.

    ``NUMBA_CACHE_DIR`` is ``tmp_path / "nbcache"``, and ``PYTHONPATH`` holds this process's ``sys.path``.
    """
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    return env


def _body(x):
    # Under fastmath, LLVM reduces (x + 1e308) - 1e308 to x. Without fastmath, x + 1e308 rounds to 1e308 for x = 1.0,
    # and the result is 0.0. The result therefore shows whether numba compiled the body under fastmath.
    return (x + 1e308) - 1e308


# The digest holds the target options of a @vectorize formula.
# ---------------------------------------------------------------------------

def test_dufunc_targetoptions_distinguish_fingerprint():
    """The fingerprint of a @vectorize formula holds its target options, so fastmath changes it.

    Without the options, two kernels that differ only in the formula's options would share one digest, and numba would
    load the kernel cached for the other options.
    """
    f_plain = vectorize(["float64(float64)"], nopython=True)(_body)
    f_fast = vectorize(["float64(float64)"], nopython=True, fastmath=True)(_body)
    fp_plain, cacheable_plain = _formula_fingerprint(f_plain)
    fp_fast, _ = _formula_fingerprint(f_fast)
    assert cacheable_plain is True
    assert fp_plain != fp_fast, "DUFunc fastmath flag absent from fingerprint (digest collision)"


def test_dufunc_fastmath_flip_not_stale_across_processes(tmp_path):
    """A kernel with a @vectorize formula compiles again when the formula's fastmath option changes between processes.

    The second process shares the first one's NUMBA_CACHE_DIR and must print 1.0; a kernel that it loaded from the
    first process's cache would print 0.0.
    """
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
    # The first process compiles the kernel without fastmath and caches it.
    assert _run_probe(probe, env) == "0.0"
    env["FASTMATH"] = "1"
    # The second process uses the same cache directory and compiles the formula with fastmath.
    got = _run_probe(probe, env)
    assert got == "1.0", f"DUFunc fastmath flip served a stale cached kernel: got {got}, expected 1.0"


# The fingerprint holds the value of a module attribute that a formula reads, such as cfg.SCALE.
# ---------------------------------------------------------------------------

def _func_reading_module_attr(scale_value):
    """Return a new function ``f(x): return x * cfg.SCALE`` whose global ``cfg`` is a module.

    The module's ``SCALE`` is ``scale_value``. The function reads it as a module attribute, not as a global.
    """
    cfg = types.ModuleType("cfg_probe_mod")
    cfg.SCALE = scale_value
    code = compile("def f(x):\n    return x * cfg.SCALE\n", "<probe>", "exec").co_consts[0]
    return types.FunctionType(code, {"cfg": cfg})


def test_module_attribute_value_change_rekeys_fingerprint():
    """The fingerprint of a function changes with the value of cfg.SCALE, a module attribute that it reads.

    numba compiles the attribute's value into the code as a constant, so a fingerprint without it would let numba load
    a kernel compiled with the old value.
    """
    assert _fingerprint_function(_func_reading_module_attr(2.0), set()) \
        != _fingerprint_function(_func_reading_module_attr(3.0), set())


def test_module_function_attr_stays_a_stable_reference():
    """The fingerprint leaves out np.sqrt, a ufunc that a function reads as a module attribute.

    numba resolves a ufunc by identity, so the fingerprint needs no text for it, and the function stays cacheable. The
    fingerprint is the same for two copies of the function, and the walk does not descend into numpy.
    """
    import numpy as np

    def mk():
        code = compile("def f(x):\n    return np.sqrt(x) + 1.0\n", "<p>", "exec").co_consts[0]
        return types.FunctionType(code, {"np": np})

    fp = _fingerprint_function(mk(), set())
    assert fp == _fingerprint_function(mk(), set())
    assert "sqrt=function" not in fp


def test_module_constant_change_not_stale_across_processes(tmp_path):
    """A kernel compiles again when a module attribute that its formula reads changes between processes.

    The second process shares the first one's NUMBA_CACHE_DIR and reads ck_cfg_probe.SCALE as 3.0 instead of 2.0. It
    must print 30.0; a kernel loaded from the first process's cache would print 20.0.
    """
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
    # The first process compiles the kernel with SCALE = 2.0 and caches it.
    assert _run_probe(probe, env) == "20.0"
    env["SCALE"] = "3.0"
    # The second process uses the same cache directory with SCALE = 3.0.
    got = _run_probe(probe, env)
    assert got == "30.0", f"module constant change served a stale kernel: got {got}, expected 30.0"


# A formula with a cache of its own leaves the kernel uncached.
# ---------------------------------------------------------------------------

def test_is_self_cached_detects_only_cache_true_dispatchers():
    """_is_self_cached is True for a dispatcher compiled with ``cache=True`` and False for one compiled without it."""
    assert _is_self_cached(_cached_formula) is True
    assert _is_self_cached(_plain_formula) is False


def test_self_cached_dispatcher_formula_makes_unit_uncacheable():
    """A formula compiled with ``@njit(cache=True)`` makes _compile warn and compile the kernel without a cache.

    numba's key for the formula's own cache leaves out its jit options and the globals that it reads, so that cache can
    hold stale code. The kernel's digest cannot refresh it, and a kernel cached with that code would contain it as well.
    """
    assert not isinstance(_cached_formula._cache, NullCache)  # The formula has a cache of its own.
    with pytest.warns(UserWarning, match="a @njit.cache=True. dispatcher"):
        disp = _compile(_KERNEL_SRC, {"f_x": _cached_formula}, None, None)
    assert isinstance(disp._cache, NullCache), "outer kernel must not be disk-cached"


def test_plain_formula_unit_stays_cacheable():
    """A formula compiled without ``cache=True`` leaves the kernel cached.

    Such a formula has no cache of its own, and the kernel's digest holds its code and options. Compiling the kernel
    without a cache would cost a compilation in every process and protect against nothing.
    """
    disp = _compile(_KERNEL_SRC, {"f_x": _plain_formula}, None, None)
    assert not isinstance(disp._cache, NullCache), "plain-formula kernel should cache"


def test_references_self_cached_detects_a_cached_callee():
    """_references_self_cached finds a ``cache=True`` dispatcher that a formula calls.

    It finds none in a dispatcher compiled without ``cache=True``.
    """
    assert _references_self_cached(_formula_calls_cached, set()) is True
    assert _references_self_cached(_plain_formula, set()) is False


def test_formula_calling_cache_true_dispatcher_makes_unit_uncacheable():
    """A formula that calls a ``cache=True`` dispatcher makes _compile warn and compile the kernel without a cache.

    The kernel links the callee's code from the callee's own cache, which can be stale. A cached kernel would keep that
    code under a new digest.
    """
    with pytest.warns(UserWarning, match="reference, a @njit"):
        disp = _compile(_KERNEL_SRC, {"f_x": njit(_formula_calls_cached)}, None, None)
    assert isinstance(disp._cache, NullCache), "unit linking a cache=True callee must not cache"


def test_formula_calling_cache_true_dispatcher_not_poisoned(tmp_path):
    """numbox writes no cached kernel when a formula calls a ``cache=True`` dispatcher whose error model changes.

    The two processes share one NUMBA_CACHE_DIR. The second process still runs the dispatcher's code from the first
    process's error model, because numba's key for the dispatcher's own cache leaves the error model out. The test
    does not check that result.
    """
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
    """_references_self_cached follows a default argument to the ``cache=True`` dispatcher."""
    assert _references_self_cached(_formula_default_arg, set()) is True


def test_references_self_cached_through_container_global():
    """_references_self_cached follows a tuple held in a global to the ``cache=True`` dispatcher."""
    assert _references_self_cached(_formula_container_global, set()) is True


def test_references_self_cached_through_module_attribute():
    """_references_self_cached follows a module attribute to the ``cache=True`` dispatcher."""
    assert _references_self_cached(_formula_module_attr, set()) is True


def test_default_arg_cached_dispatcher_makes_unit_uncacheable():
    """A ``cache=True`` dispatcher in a default argument also makes _compile warn and compile without a cache."""
    with pytest.warns(UserWarning, match="reference, a @njit"):
        disp = _compile(_KERNEL_SRC, {"f_x": njit(_formula_default_arg)}, None, None)
    assert isinstance(disp._cache, NullCache)


def test_dispatcher_typed_declared_sig_makes_unit_uncacheable():
    """A declared signature that holds a dispatcher type, the type of an @njit function, leaves the kernel uncached.

    numba's cache index misses such a signature in every process and adds an entry each time, so _type_identity marks
    the type uncacheable and _compile compiles the kernel without a cache.
    """
    disp = _compile(
        _KERNEL_SRC, {"f_x": _plain_formula}, None, None,
        declared_sigs=((typeof(_plain_formula),),),
    )
    assert isinstance(disp._cache, NullCache), (
        "a Dispatcher-typed declared signature must make the unit uncacheable")


# The fingerprint holds a module attribute reached through a submodule, or through a module in a closure.
# ---------------------------------------------------------------------------

def _func_reading_chained_module_attr(scale):
    """Return a new function ``f(x): return x * cfg.sub.SCALE`` whose ``cfg.sub.SCALE`` is ``scale``."""
    cfg = types.ModuleType("cfg_chain_probe")
    sub = types.ModuleType("cfg_chain_probe.sub")
    cfg.sub = sub
    sub.SCALE = scale
    code = compile("def f(x):\n    return x * cfg.sub.SCALE\n", "<p>", "exec").co_consts[0]
    return types.FunctionType(code, {"cfg": cfg})


def test_chained_module_attribute_rekeys_fingerprint():
    """The fingerprint changes with the value of cfg.sub.SCALE, which the function reaches through a submodule."""
    assert _fingerprint_function(_func_reading_chained_module_attr(2.0), set()) \
        != _fingerprint_function(_func_reading_chained_module_attr(3.0), set())


def _closure_module_formula(scale):
    """Return a function that reads ``m.SCALE`` from a module ``m`` in its closure; ``m.SCALE`` is ``scale``."""
    mod = types.ModuleType("cmod_closure_probe")
    mod.SCALE = scale

    def make(m):
        def f(x):
            return x * m.SCALE
        return f
    return make(mod)


def test_closure_captured_module_attribute_rekeys_fingerprint():
    """The fingerprint changes with the value of m.SCALE, which the function reads from a module in its closure."""
    assert _fingerprint_function(_closure_module_formula(2.0), set()) \
        != _fingerprint_function(_closure_module_formula(3.0), set())


def _module_attr_dispatcher_formula(other_val):
    """Return a function that calls ``cfg.compute``, a dispatcher whose global ``OTHER`` is ``other_val``.

    numba links the dispatcher's code into the kernel and compiles in the value of OTHER, so the digest must fold a
    dispatcher reached as a module attribute, as it folds a dispatcher read as a global.
    """
    disp_code = compile("def compute(z):\n    return z + OTHER\n", "<d>", "exec").co_consts[0]
    compute = njit(cache=False)(types.FunctionType(disp_code, {"OTHER": other_val}))
    cfg = types.ModuleType("cfg_disp_fp_probe")
    cfg.compute = compute
    f_code = compile("def f(x):\n    return cfg.compute(x)\n", "<f>", "exec").co_consts[0]
    return types.FunctionType(f_code, {"cfg": cfg})


def test_module_attr_dispatcher_captured_value_rekeys_fingerprint():
    """The fingerprint changes with a global of the dispatcher that the function calls as cfg.compute."""
    assert _fingerprint_function(_module_attr_dispatcher_formula(10.0), set()) \
        != _fingerprint_function(_module_attr_dispatcher_formula(999.0), set())


def _module_attr_dufunc_formula(other_val):
    """Return a function that calls ``cfg.uf``, a @vectorize function whose scalar body reads the global ``OTHER``.

    ``OTHER`` is ``other_val``. numba links the function's compiled loop and compiles in the value of OTHER, so the
    digest must fold it as well.
    """
    wrapped = types.FunctionType(
        compile("def uf(z):\n    return z + OTHER\n", "<d>", "exec").co_consts[0], {"OTHER": other_val})
    uf = vectorize(["float64(float64)"], nopython=True)(wrapped)
    cfg = types.ModuleType("cfg_uf_fp_probe")
    cfg.uf = uf
    f_code = compile("def f(x):\n    return cfg.uf(x)\n", "<f>", "exec").co_consts[0]
    return types.FunctionType(f_code, {"cfg": cfg})


def test_module_attr_dufunc_captured_value_rekeys_fingerprint():
    """The fingerprint changes with a global of the @vectorize function that the function calls as cfg.uf."""
    assert _fingerprint_function(_module_attr_dufunc_formula(10.0), set()) \
        != _fingerprint_function(_module_attr_dufunc_formula(999.0), set())


# The digest does not hold a value that a module provides only through a module-level __getattr__ (PEP 562). If the
# fingerprint looked a referenced name up with getattr, it would run the __getattr__ of an unrelated module, such as
# numpy's, for every name of co_names that the module lacks. That imports submodules and emits deprecation warnings at
# every compilation. No test covers such a value, because the digest leaves it out on purpose; the variable
# documentation lists the limit.


# The digest holds the NUMBA_* settings that change the emitted code.
# ---------------------------------------------------------------------------

def test_vectorize_env_knob_enters_digest(monkeypatch):
    """``NUMBA_LOOP_VECTORIZE``, which numba reads into ``numba.config.LOOP_VECTORIZE``, changes the kernel's name."""
    src = "def _kernel(a0):\n    return (a0,)\n"
    monkeypatch.setattr(numba_config, "LOOP_VECTORIZE", 0, raising=False)
    name0 = _compile(src, {}, None, None).py_func.__name__
    monkeypatch.setattr(numba_config, "LOOP_VECTORIZE", 1, raising=False)
    name1 = _compile(src, {}, None, None).py_func.__name__
    assert name0 != name1, "LOOP_VECTORIZE absent from the cache digest"


# NUMBA_BOUNDSCHECK and NUMBA_DISABLE_INTEL_SVML change the code, and the jit options and numba's key leave them out.
# ---------------------------------------------------------------------------

def test_boundscheck_env_enters_digest(monkeypatch):
    """``NUMBA_BOUNDSCHECK`` changes the kernel's name.

    The setting overrides even an explicit boundscheck option, and neither the jit options nor numba's key holds it.
    If the digest left it out, numba would load a kernel compiled under the other setting.
    """
    src = "def _kernel(a0):\n    return (a0,)\n"
    monkeypatch.setattr(numba_config, "BOUNDSCHECK", None)
    name_off = _compile(src, {}, None, None).py_func.__name__
    monkeypatch.setattr(numba_config, "BOUNDSCHECK", 1)
    name_on = _compile(src, {}, None, None).py_func.__name__
    assert name_off != name_on, "BOUNDSCHECK absent from the cache digest"


def test_svml_env_enters_digest(monkeypatch):
    """``NUMBA_DISABLE_INTEL_SVML`` changes the kernel's name.

    If Intel's SVML library is installed, the setting decides whether numba calls SVML for vectorized math. numba's key
    does not hold the setting.
    """
    src = "def _kernel(a0):\n    return (a0,)\n"
    monkeypatch.setattr(numba_config, "DISABLE_INTEL_SVML", 0)
    name_svml = _compile(src, {}, None, None).py_func.__name__
    monkeypatch.setattr(numba_config, "DISABLE_INTEL_SVML", 1)
    name_plain = _compile(src, {}, None, None).py_func.__name__
    assert name_svml != name_plain, "DISABLE_INTEL_SVML absent from the cache digest"


# A jit option without a canonical form makes _compile warn and compile without a cache.
# ---------------------------------------------------------------------------

def test_uncanonicalizable_flags_warn_and_disable_cache():
    """A ``pipeline_class`` option has no canonical form, so _compile warns and compiles the kernel without a cache.

    A digest without the option could not distinguish the kernels that two pipelines compile.
    """
    from numba.core.compiler import Compiler
    src = "def _kernel(a0):\n    return (a0,)\n"
    with pytest.warns(UserWarning, match="no canonical"):
        disp = _compile(src, {}, {"pipeline_class": Compiler}, None)
    assert isinstance(disp._cache, NullCache)


# The record of passed return-type checks is keyed on the jit options.
# ---------------------------------------------------------------------------

def _declared_locals_graph():
    """Return a graph with one variable declared float64, whose formula returns its local ``m``.

    Under the jit option ``locals={'m': float32}``, numba infers float32 as the return type, and the return-type check
    refuses the declaration.
    """
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
    """A return-type check that passed under the default options runs again under an option that changes the type.

    The record of passed checks is keyed on the jit options, so the second compile_kernel checks again and refuses
    float32 for the float64 declaration.
    """
    from numba.core.types import float32
    from numbox.core.variable.compile_kernel import compile_kernel, _validated_returns

    saved = set(_validated_returns)
    _validated_returns.clear()
    try:
        # The check passes under the default options, and compile_kernel records it.
        compile_kernel(_declared_locals_graph(), "variables.a")
        with pytest.raises(ValueError, match="float32"):
            compile_kernel(_declared_locals_graph(), "variables.a",
                           jit_options={"locals": {"m": float32}})
    finally:
        _validated_returns.clear()
        _validated_returns.update(saved)
