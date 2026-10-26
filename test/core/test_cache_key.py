"""numbox's part of numba's cache key: the settings and flags it holds with numbox's version, the opt-in decorators,
what numbox compiles, and the fallback to uncached.

The cross-process tests run a probe in child processes against one ``NUMBA_CACHE_DIR``. Each child reports a cache
hit or miss per function and the values it computed. The report is a JSON file the child writes, because several
settings flood a child's stdout with numba's own dumps.
"""
import ast
import collections
import inspect
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

from concurrent.futures import ThreadPoolExecutor

import numba
import numbox
import numpy as np
import pytest

from numba import njit
from numba.core import compiler as numba_compiler, config as numba_config
from numba.core.compiler import Compiler
from numba.core.cpu_options import FastMathOptions, InlineOptions, ParallelOptions
from numba.core.registry import cpu_target
from numba.experimental import jitclass

from numbox.core import cache_key
from numbox.core.cache_key import (
    KEYED_SETTINGS, UNKEYED_SETTINGS, UncachedWarning, explain_cache_key, keyed_cfunc, keyed_njit,
)
from numbox.core.proxy import proxy as proxy_module
from numbox.core.proxy.proxy import proxy
from numbox.core.work import builder
from numbox.utils import highlevel


def _child_env(cache_dir, **extra):
    """The test's environment for a child, with numba's defaults for every keyed setting and the package's options."""
    env = {k: v for k, v in os.environ.items() if k not in KEYED_SETTINGS and k != "NUMBOX_JIT_OPTIONS"}
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra)
    return env


def _run(script, env, out, *args):
    r = subprocess.run(
        [sys.executable, str(script), str(out), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    assert r.returncode == 0, f"{script.name} {args} failed:\n{r.stderr[-4000:]}"
    with open(out, encoding="utf-8") as f:
        return json.load(f)


# ---- The classification ----

_ENVIRON_READ = re.compile(
    r"""os\.environ(?:\.get)?\s*[\[(]\s*['"](NUMBA_[A-Z0-9_]+)['"]|['"](NUMBA_[A-Z0-9_]+)['"]\s+in\s+os\.environ"""
)


def test_every_setting_numba_reads_is_classified():
    """A numba release that reads a ``NUMBA_*`` setting nobody has classified fails here, on the matrix cell that
    runs it: the setting may change the code numba emits, and then it belongs in the key. numba reads most settings
    in ``config.py``, and a few through ``os.environ`` elsewhere; both are scanned."""
    read = set(re.findall(r'"(NUMBA_[A-Z0-9_]+)"', inspect.getsource(numba_config)))
    root = pathlib.Path(numba.__file__).parent
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("tests/") or "/tests/" in rel:
            continue
        for match in _ENVIRON_READ.finditer(path.read_text(encoding="utf-8", errors="replace")):
            read.add(match.group(1) or match.group(2))
    assert not set(KEYED_SETTINGS) & set(UNKEYED_SETTINGS)
    unclassified = read - set(KEYED_SETTINGS) - set(UNKEYED_SETTINGS)
    assert not unclassified, f"numba {numba.__version__} reads settings nobody has classified: {sorted(unclassified)}"


def test_every_keyed_setting_names_the_config_attribute_numba_reads_it_into():
    source = inspect.getsource(numba_config)
    for env, attribute in KEYED_SETTINGS.items():
        if f'"{env}"' not in source:
            continue
        assert re.search(rf'^\s*{attribute}\s*=\s*_readenv\(\s*"{env}"', source, re.M), (env, attribute)
        assert hasattr(numba_config, attribute), (env, attribute)


# ---- The key, in code ----

@keyed_njit(cache=True)
def _keyed_for_settings(x):
    return x + 1


def _key_of(func):
    """The key a lookup of ``func`` for an ``int64`` asks numba's cache for now, with numbox's part where numbox keys
    it."""
    return func._cache._index_key((numba.int64,), cpu_target.target_context.codegen())


@pytest.mark.parametrize("env", sorted(KEYED_SETTINGS))
def test_a_setting_changed_in_code_changes_the_key(env, monkeypatch):
    """numbox reads each setting when numba looks the function up, so a setting changed in code changes the key used
    by the next lookup."""
    key_before = _key_of(_keyed_for_settings)
    monkeypatch.setattr(numba_config, KEYED_SETTINGS[env], "numbox-test-value", raising=False)
    assert explain_cache_key(_keyed_for_settings)["settings"][env] == "'numbox-test-value'"
    key_after = _key_of(_keyed_for_settings)
    assert key_after != key_before
    assert repr((env, "'numbox-test-value'")) in key_after[-1][2]


def test_the_key_is_numbas_with_numboxs_version_flags_and_settings_beside_it(monkeypatch):
    """numbox appends one part to numba's key, and only to a cache it keys; a plain ``njit`` keeps numba's key as it
    is. A new numbox version is a new key."""
    plain = njit(cache=True)(_keyed_for_settings.py_func)
    numbas = _key_of(plain)
    key = _key_of(_keyed_for_settings)
    assert key[:-1] == numbas
    explained = explain_cache_key(_keyed_for_settings)
    assert key[-1] == (numbox.__version__, explained["flags"], cache_key._settings_canon())
    monkeypatch.setattr(numbox, "__version__", "0.0.0+numbox-test")
    assert _key_of(_keyed_for_settings)[-1][0] == "0.0.0+numbox-test"
    assert _key_of(plain) == numbas


def test_numbas_opt_level_keeps_max_apart_from_three():
    """numba reads ``NUMBA_OPT=max`` into an ``int`` of value 3 that remembers it was ``max``, and builds a different
    code generator for it; the key keeps the two apart."""
    assert cache_key._setting_value(numba_config._OptLevel("max")) == "'max'"
    assert cache_key._setting_value(numba_config._OptLevel(3)) == "3"
    assert cache_key._setting_value(3) == "3"
    assert cache_key._setting_value(True) == cache_key._setting_value(1)


_SHARED_OPTIONS = '''
import json
import sys

import numba
import numpy as np

from numbox.core.cache_key import keyed_njit

shared = keyed_njit(cache=True, parallel=True, locals={"acc": numba.float64})


def total(n):
    acc = 0.0
    for i in numba.prange(n):
        acc += i
    return acc


def doubled(n):
    acc = 0.0
    for i in numba.prange(n):
        acc += 2 * i
    return acc


total_, doubled_ = shared(total), shared(doubled)
results = [total_(3), total_(np.int32(3)), doubled_(3)]
with open(sys.argv[1], "w", encoding="utf-8") as out:
    json.dump([results, [sum(f.stats.cache_hits.values()) for f in (total_, doubled_)]], out)
'''


def test_options_numba_changes_in_a_compile_key_every_compile_alike_in_the_next_process(tmp_path):
    """One decorator gives two functions one ``locals`` dict, and one of them compiles for two signatures. numba's
    first compile adds names to ``locals``, where a cache hit adds none: the next process must still find every entry
    the first saved."""
    script = tmp_path / "shared_options.py"
    script.write_text(_SHARED_OPTIONS, encoding="utf-8")
    cache = tmp_path / "cache"
    assert _run(script, _child_env(cache), tmp_path / "one.json") == [[3.0, 3.0, 6.0], [0, 0]]
    assert _run(script, _child_env(cache), tmp_path / "two.json") == [[3.0, 3.0, 6.0], [2, 1]]


# ---- Settings, across processes ----

_SWEEP_MODULE = '''
import numpy as np
from numba import prange

from numbox.core.cache_key import keyed_njit


@keyed_njit(cache=True)
def index(a, i):
    return a[i]


@keyed_njit(cache=True, parallel=True)
def parallel_tuple(a, b, c):
    t = (a, b, c)
    out = np.empty_like(a)
    for i in prange(a.shape[0]):
        out[i] = t[0][i] + t[1][i] + t[2][i]
    return out
'''

_SWEEP_RUN = '''
import json
import sys

import numpy as np
from numba.core import config

import sweep_module
from numbox.core.cache_key import KEYED_SETTINGS, _setting_value

out = {"settings": {env: _setting_value(getattr(config, attr, None)) for env, attr in KEYED_SETTINGS.items()}}
a = np.arange(3.0)
out["index"] = float(sweep_module.index(a, 1))
out["index_hits"] = sum(sweep_module.index.stats.cache_hits.values())
out["index_misses"] = sum(sweep_module.index.stats.cache_misses.values())
if out["settings"]["NUMBA_BOUNDSCHECK"] == "1":
    try:
        sweep_module.index(a, 5)
        out["out_of_bounds"] = "no error"
    except IndexError:
        out["out_of_bounds"] = "IndexError"
if "parallel" in sys.argv[2:]:
    try:
        out["parallel_tuple"] = float(sweep_module.parallel_tuple(a, a, a).sum())
        out["parallel_tuple_hits"] = sum(sweep_module.parallel_tuple.stats.cache_hits.values())
    except Exception as exc:
        out["parallel_tuple"] = type(exc).__name__
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(out, f)
'''

#: The value each keyed setting is flipped to in the sweep, against numba's default. ``NUMBA_ENABLE_AVX`` is flipped
#: against what the first child reports, which depends on the host.
_FLIPS = {
    "NUMBA_BOUNDSCHECK": "1",
    "NUMBA_OPT": "0",
    "NUMBA_LOOP_VECTORIZE": "0",
    "NUMBA_SLP_VECTORIZE": "1",
    "NUMBA_DEBUGINFO": "1",
    "NUMBA_ENABLE_PROFILING": "1",
    "NUMBA_EXTEND_VARIABLE_LIFETIMES": "1",
    "NUMBA_LLVM_REFPRUNE_PASS": "0",
    "NUMBA_LLVM_REFPRUNE_FLAGS": "per_bb",
    "NUMBA_DEBUG_NRT": "1",
    "NUMBA_DEBUG_NRT_STACK_LIMIT": "5",
    "NUMBA_FULL_TRACEBACKS": "1",
    "NUMBA_DEVELOPER_MODE": "1",
    "NUMBA_DEBUG_JIT": "1",
    "NUMBA_DEBUG_ARRAY_OPT": "1",
    "NUMBA_DEBUG_ARRAY_OPT_RUNTIME": "1",
    "NUMBA_PARFOR_MAX_TUPLE_SIZE": "2",
    "NUMBA_GDB_BINARY": "gdb-of-the-cache-key-test",
    "NUMBA_DUMP_ASSEMBLY": "1",
    "NUMBA_DEBUG": "1",
    "NUMBA_DIFF_IR": "1",
    "NUMBA_DISABLE_INTEL_SVML": "1",
}


def test_the_sweep_flips_every_keyed_setting_but_enable_avx():
    assert set(_FLIPS) | {"NUMBA_ENABLE_AVX"} == set(KEYED_SETTINGS)


def test_every_keyed_setting_misses_when_flipped_and_hits_when_flipped_back(tmp_path):
    """Each keyed setting, in child processes. A first child compiles and caches under numba's defaults. A child with
    the setting flipped must miss, and must show the changed behavior where it shows (a bounds check, a parfor too
    wide to compile). A child flipped back on the same cache must find the first entry. Each flip runs on a copy of
    the first cache, so the flips run side by side."""
    (tmp_path / "sweep_module.py").write_text(_SWEEP_MODULE, encoding="utf-8")
    script = tmp_path / "sweep_run.py"
    script.write_text(_SWEEP_RUN, encoding="utf-8")
    base = tmp_path / "cache"
    cold = _run(script, _child_env(base), tmp_path / "cold.json", "parallel")
    assert (cold["index"], cold["index_hits"], cold["index_misses"]) == (1.0, 0, 1)
    assert cold["parallel_tuple"] == 9.0
    flips = dict(_FLIPS, NUMBA_ENABLE_AVX="0" if cold["settings"]["NUMBA_ENABLE_AVX"] == "1" else "1")
    source = inspect.getsource(numba_config)
    flips = {env: value for env, value in flips.items() if f'"{env}"' in source}  # those this numba reads

    def flip(env):
        cache = tmp_path / f"cache-{env}"
        shutil.copytree(base, cache)
        args = ("parallel",) if env == "NUMBA_PARFOR_MAX_TUPLE_SIZE" else ()
        flipped = _run(script, _child_env(cache, **{env: flips[env]}), tmp_path / f"{env}-flipped.json", *args)
        back = _run(script, _child_env(cache), tmp_path / f"{env}-back.json", "parallel")
        return env, flipped, back

    with ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 2)) as pool:
        results = {env: (flipped, back) for env, flipped, back in pool.map(flip, sorted(flips))}
    for env, (flipped, back) in results.items():
        assert flipped["settings"][env] != cold["settings"][env], f"{env}={flips[env]} did not take"
        assert (flipped["index"], flipped["index_misses"]) == (1.0, 1), f"{env} flipped was served the default's binary"
        assert (back["index_hits"], back["parallel_tuple_hits"]) == (1, 1), f"{env} flipped back missed"
    assert results["NUMBA_BOUNDSCHECK"][0]["out_of_bounds"] == "IndexError"
    assert results["NUMBA_PARFOR_MAX_TUPLE_SIZE"][0]["parallel_tuple"] == "UnsupportedParforsError"


# ---- The opt-in ----

def _plain(x):
    return x + 1


def test_keyed_njit_keys_lazy_and_eager_compiles_and_njit_does_not():
    lazy = keyed_njit(cache=True)(_plain)
    eager = keyed_njit("float64(float64)", cache=True)(_plain)
    called = keyed_njit(_plain, cache=True)
    plain = njit(cache=True)(_plain)
    for keyed in (lazy, eager, called):
        assert keyed._cache._numbox_keyed[0] == "njit"
        assert set(explain_cache_key(keyed)) == {"numbox", "flags", "settings"}
    with pytest.raises(ValueError, match="has no cache numbox keys"):
        explain_cache_key(plain)
    with pytest.raises(ValueError, match="has no cache numbox keys"):
        explain_cache_key(keyed_njit(_plain))
    assert eager(1.0) == 2.0


def test_keyed_cfunc_is_keyed_apart_from_an_njit_of_the_same_function():
    """numba keys a ``@cfunc`` and an ``@njit`` of one function and signature alike, though their compiled results
    differ; numbox's flags hold the decorator."""
    c = keyed_cfunc("float64(float64)", cache=True)(_plain)
    j = keyed_njit("float64(float64)", cache=True)(_plain)
    assert c._cache._numbox_keyed[0] == "cfunc"
    assert explain_cache_key(c)["flags"].startswith("kind=cfunc;")
    assert explain_cache_key(j)["flags"].startswith("kind=njit;")
    assert c.ctypes(1.0) == 2.0


def test_the_flags_are_in_the_key():
    fast = keyed_njit(cache=True, fastmath=True)(_plain)
    plain = keyed_njit(cache=True)(_plain)
    typed = keyed_njit(cache=True, locals={"y": numba.float32})(_plain)
    assert explain_cache_key(fast)["flags"] != explain_cache_key(plain)["flags"]
    assert ";fastmath=fastmath[fast];" in explain_cache_key(fast)["flags"]
    assert explain_cache_key(typed)["flags"] != explain_cache_key(plain)["flags"]
    assert "float32" in explain_cache_key(typed)["flags"].partition(";locals=")[2]


def test_a_set_of_fastmath_flags_and_a_dict_of_locals_are_written_in_one_order():
    """``fastmath`` takes a set of flag names, whose order changes with each process's hash seed, and two dicts of the
    same locals given in another order are the same locals."""
    flags = {"nnan", "ninf", "nsz", "arcp", "contract", "afn", "reassoc"}
    assert cache_key._flag_value_text(FastMathOptions(flags), "x") == "fastmath[" + ",".join(sorted(flags)) + "]"
    one = {"b": numba.float32, "a": numba.int64}
    forward = keyed_njit(cache=True, locals=one)(_plain)
    backward = keyed_njit(cache=True, locals=dict(reversed(list(one.items()))))(_plain)
    assert explain_cache_key(forward)["flags"] == explain_cache_key(backward)["flags"]


def test_options_numba_takes_in_another_form_are_keyed_as_numba_takes_them():
    """numba turns a numpy bool into a bool, a dict of fast-math flags into the set of those enabled, and an options
    object into its values. The key holds what numba compiles under, so each keys as its plain form does."""
    def flags(**options):
        return explain_cache_key(keyed_njit(cache=True, **options)(_plain))["flags"]

    assert flags(boundscheck=np.True_, nogil=np.True_) == flags(boundscheck=True, nogil=True)
    assert flags(fastmath={"nnan": np.True_, "ninf": False}) == flags(fastmath={"nnan"})
    assert flags(fastmath=FastMathOptions({"nsz", "arcp"})) == flags(fastmath={"arcp", "nsz"})
    assert flags(parallel=ParallelOptions({"fusion": np.False_})) == flags(parallel=ParallelOptions({"fusion": False}))
    assert flags(parallel=ParallelOptions({"fusion": False})) != flags(parallel=True)


def _cost_model(expr, caller, callee):
    return True


def _plus_two(x):
    return x + 2


def test_a_function_as_the_inline_cost_model_compiles_uncached_with_a_warning():
    """The key holds ``inline='always'`` and ``'never'``. A function given as the cost model is code the key does not
    hold, so the function compiles without a cache."""
    assert cache_key._flag_value_text(InlineOptions("always"), "x") == "inline['always']"
    modeled = keyed_njit(cache=True, inline=_cost_model)(_plus_two)
    modeled._cache.flush()
    with pytest.warns(UncachedWarning, match="the jit flag inline is a cost model"):
        assert modeled(1) == 3
    assert modeled._cache._cache_file._load_index() == {}


def _pipeline_for(error_model):
    class Pipeline(Compiler):
        def __init__(self, typingctx, targetctx, library, args, return_type, flags, locals):
            flags = flags.copy()
            flags.error_model = error_model
            super().__init__(typingctx, targetctx, library, args, return_type, flags, locals)
    return Pipeline


def _divides_by(a, b):
    return a / b


def test_a_pipeline_class_of_ones_own_compiles_uncached_with_a_warning():
    divides = keyed_njit(cache=True, pipeline_class=_pipeline_for("numpy"))(_divides_by)
    divides._cache.flush()
    with pytest.warns(UncachedWarning, match=r"the pipeline class is _pipeline_for\.<locals>\.Pipeline, not numba's"):
        assert divides(1.0, 0.0) == np.inf
    assert divides._cache._cache_file._load_index() == {}


def _halves(a):
    return a / 2.0


class _PipelineReadAfresh:
    """A pipeline callable without a ``__qualname__`` whose ``repr`` returns a different string each time."""

    def __init__(self):
        self.reads = 0

    def __call__(self, *args):
        return Compiler(*args)

    def __repr__(self):
        self.reads += 1
        return f"a pipeline read {self.reads} times"


def test_a_pipeline_without_a_qualname_compiles_uncached_with_one_warning():
    """The refusal message names the pipeline's type, not its ``repr``, so numbox issues one ``UncachedWarning`` per
    process for a function compiled under a pipeline whose ``repr`` returns a different string each time."""
    halves = keyed_njit(cache=True, pipeline_class=_PipelineReadAfresh())(_halves)
    halves._cache.flush()
    with pytest.warns(UncachedWarning, match="the pipeline class is an instance of _PipelineReadAfresh") as caught:
        assert halves(1.0) == 0.5
        assert halves(np.float32(1.0)) == 0.5
    assert len([w for w in caught if issubclass(w.category, UncachedWarning)]) == 1
    assert halves._cache._cache_file._load_index() == {}


def test_a_type_in_locals_that_names_an_address_is_refused_and_one_that_only_reads_alike_is_not():
    for addressed in (numba.typeof(njit(_plain)), numba.deferred_type(), _Tally.class_type.instance_type):
        with pytest.raises(cache_key._Unkeyable, match="names an object by its address"):
            cache_key._type_text(addressed, "a type in locals")
    record = numba.from_dtype(np.dtype([("reg_0x04", np.int32)]))
    for stable in (record, numba.types.literal("0x04")):
        assert cache_key._type_text(stable, "a type in locals").startswith("numbatype(")


def _recompiled(x):
    return x + 1


def test_a_recompile_after_a_setting_changed_in_code_saves_the_new_value(monkeypatch):
    """Each lookup reads the part afresh, so a recompile under a setting changed in code saves under that setting,
    not under the part the first compile read."""
    recompiled = keyed_njit(cache=True)(_recompiled)
    recompiled._cache.flush()
    assert recompiled(1) == 2
    monkeypatch.setattr(numba_config, "BOUNDSCHECK", 1)
    recompiled.recompile()
    now = [dict(ast.literal_eval(key[-1][2])) for key in recompiled._cache._cache_file._load_index()]
    assert any(settings["NUMBA_BOUNDSCHECK"] == "1" for settings in now)


@jitclass([("n", numba.int64)])
class _Tally:
    def __init__(self, n):
        self.n = n


def _counts(n):
    t = _Tally(n)
    return t.n


def test_a_flag_with_no_stable_form_compiles_uncached_with_a_warning():
    """numba writes a jitclass's type after an address no other process shares; held in ``locals``, it makes the
    function compile without a cache, once warned."""
    counts = keyed_njit(cache=True, locals={"t": _Tally.class_type.instance_type})(_counts)
    with pytest.warns(UncachedWarning, match="a type in locals is the numba type"):
        assert counts(3) == 3
    assert counts._cache._cache_file._load_index() == {}
    with pytest.raises(ValueError, match="compiles without an on-disk cache"):
        explain_cache_key(counts)


def test_explain_cache_key_lists_the_parts():
    parts = explain_cache_key(_keyed_for_settings)
    assert set(parts) == {"numbox", "flags", "settings"}
    assert parts["numbox"] == numbox.__version__
    assert parts["flags"].startswith("kind=njit;flags=[")
    assert set(parts["settings"]) == set(KEYED_SETTINGS)
    assert all(isinstance(value, str) for value in parts["settings"].values())


def test_the_keyed_decorators_take_numbas_arguments_unchanged():
    assert keyed_njit(signature_or_function=_plain, cache=True)._cache._numbox_keyed[0] == "njit"
    c = keyed_cfunc("float64(float64)", {}, True)(_plain)
    assert c._cache._numbox_keyed[0] == "cfunc"
    assert c.ctypes(1.0) == 2.0


@keyed_njit(cache=True)
def _divides(a, b):
    return a / b


@keyed_njit(cache=True, error_model="numpy")
def _divides_under_numpy(a, b):
    return _divides(a, b)


def test_a_callee_compiled_inside_a_caller_is_keyed_with_the_flags_it_takes_from_it():
    """numba gives a callee the caller's ``error_model`` where the callee sets none: the callee's key says so, apart
    from the key of a direct call."""
    _divides_under_numpy._cache.flush()
    _divides._cache.flush()
    assert _divides_under_numpy(1.0, 0.0) == np.inf
    taken = _saved_flags(_divides)
    assert taken and all(";error_model='numpy';" in flags for flags in taken)
    assert ";error_model='python';" in explain_cache_key(_divides)["flags"]


def _saved_flags(func):
    return [key[-1][1] for key in func._cache._cache_file._load_index()]


def _sums_in_parallel(n):
    acc = 0.0
    for i in numba.prange(n):
        acc += i
    return acc


def _sums_with_a_typed_local(n):
    acc = 0.0
    for i in range(n):
        acc += i
    return acc


def test_a_compile_saves_the_flags_a_later_lookup_reads():
    """numba's SSA pass adds names to ``locals`` during a compile, and the key leaves them out, so the entry a compile
    saves is the one a later lookup asks for."""
    typed = keyed_njit(cache=True, locals={"acc": numba.float64})(_sums_with_a_typed_local)
    typed._cache.flush()
    before = explain_cache_key(typed)["flags"]
    assert typed(4) == 6.0
    assert set(typed._compiler.locals) != {"acc"}
    assert _saved_flags(typed) == [before] == [explain_cache_key(typed)["flags"]]


def test_a_parallel_dict_compiles_uncached_with_a_warning():
    """numba takes each choice out of a ``parallel`` dict as it parses it, so a later compile given the same dict takes
    numba's defaults. The key does not hold such a dict, and the function compiles without a cache."""
    parallel = keyed_njit(cache=True, parallel={"fusion": False})(_sums_in_parallel)
    parallel._cache.flush()
    with pytest.warns(UncachedWarning, match="the jit flag parallel is a dict"):
        assert parallel(4) == 6.0
    assert parallel._cache._cache_file._load_index() == {}
    with pytest.raises(ValueError, match="the jit flag parallel is a dict"):
        explain_cache_key(parallel)


def _sums_in_parallel_given_a_dict(n):
    acc = 0.0
    for i in numba.prange(n):
        acc += 3 * i
    return acc


def _sums_in_parallel_given_a_dict_subclass(n):
    acc = 0.0
    for i in numba.prange(n):
        acc += 2 * i
    return acc


def test_a_refused_parallel_dict_reaches_numba_whole(monkeypatch):
    """The refusal comes before numbox parses the options, so numba's first compile takes the choices the dict gives;
    a dict subclass is refused as a dict is."""
    seen = {}
    compile_extra = numba_compiler.compile_extra

    def recording(*args, **kwargs):
        seen.setdefault(args[2].__name__, []).append(bool(kwargs["flags"].auto_parallel.fusion))
        return compile_extra(*args, **kwargs)

    monkeypatch.setattr(numba_compiler, "compile_extra", recording)
    for given, body in (({"fusion": False}, _sums_in_parallel_given_a_dict),
                        (collections.OrderedDict(fusion=False), _sums_in_parallel_given_a_dict_subclass)):
        refused = keyed_njit(cache=True, parallel=given)(body)
        refused._cache.flush()
        with pytest.warns(UncachedWarning, match="the jit flag parallel is a dict"):
            refused(4)
        assert seen[body.__name__] == [False]


def test_a_cfuncs_key_holds_the_flags_numba_compiles_it_under():
    """numba's compiler changes a C callback's flags after the parse: no CPython wrapper, no object mode. The key holds
    them as numba compiles them, so ``nopython=True``, which a C callback always is, keys as its absence does."""
    flags = explain_cache_key(keyed_cfunc("float64(float64)", cache=True)(_plain))["flags"]
    assert ";no_cpython_wrapper=True;" in flags and ";enable_pyobject=False;" in flags
    assert explain_cache_key(keyed_cfunc("float64(float64)", cache=True, nopython=True)(_plain))["flags"] == flags


def _sums_into_a_renamed_variable(n):
    x = 0
    for i in range(n):
        x = x + i
    return x


def test_a_name_with_a_dot_is_keyed_unless_numbas_ssa_pass_added_it():
    """numba's SSA pass names a variable's later definitions after it, ``x.1`` and on. Given ``x`` in ``locals``, it
    writes those names into ``locals`` with ``x``'s type, and the key leaves them out. Given ``x.1`` alone, numba types
    that definition by it, and the key holds it."""
    dotted = keyed_njit(cache=True, locals={"x.1": numba.float64})(_sums_into_a_renamed_variable)
    plain = keyed_njit(cache=True)(_sums_into_a_renamed_variable)
    dotted._cache.flush()
    assert isinstance(dotted(4), float) and isinstance(plain(4), int)
    assert "x.1" in explain_cache_key(dotted)["flags"].partition(";locals=")[2]
    given = keyed_njit(cache=True, locals={"x": numba.float64})(_sums_into_a_renamed_variable)
    given._cache.flush()
    before = explain_cache_key(given)["flags"]
    assert isinstance(given(4), float)
    assert any(name.startswith("x.") for name in given._compiler.locals)
    assert explain_cache_key(given)["flags"] == before


def test_explain_cache_key_reads_what_cres_returns():
    wrapper = highlevel.cres(numba.float64(numba.float64), cache=True)(_halved_by_cres)
    assert explain_cache_key(wrapper)["flags"].startswith("kind=njit;")


# ---- What numbox compiles ----

def _keyed_at_decoration(monkeypatch, module):
    """Record, for every function ``module`` hands to its ``njit``, whether numbox keys it at that moment."""
    seen = []
    real = module.njit

    def spy(*args, **kwargs):
        decorator = real(*args, **kwargs)

        def wrap(func):
            seen.append(cache_key._keys(func))
            return decorator(func)
        return wrap

    monkeypatch.setattr(module, "njit", spy)
    return seen


def test_numboxs_own_functions_are_keyed_and_a_module_named_alike_is_not():
    """A function of a numbox module is keyed however it is decorated, ``vector_push`` with numba's plain ``njit``
    among them; a module whose name only starts with the letters of numbox is not numbox."""
    from numbox.core.vector.vector import vector_push
    assert vector_push._cache._numbox_keyed[0] == "njit"

    def named():
        pass

    named.__module__ = "numbox.elsewhere"
    assert cache_key._keys(named)
    named.__module__ = "numboxer"
    assert not cache_key._keys(named)


def _halved(x):
    return x / 2.0


def _halved_by_cres(x):
    return x / 2.0


def test_a_function_passed_through_cres_is_keyed(monkeypatch):
    seen = _keyed_at_decoration(monkeypatch, highlevel)
    highlevel.cres(numba.float64(numba.float64), cache=True)(_halved)
    assert seen == [True]
    assert not cache_key._keys(_halved)


def test_a_function_passed_through_proxy_is_keyed_and_so_is_its_wrapper(monkeypatch):
    seen = _keyed_at_decoration(monkeypatch, proxy_module)
    wrapper = proxy(numba.float64(numba.float64), jit_options={"cache": True})(_halved)
    assert seen == [True]
    assert wrapper._cache._numbox_keyed[0] == "njit"
    assert wrapper(3.0) == 1.5
    assert explain_cache_key(wrapper.as_func)["flags"].startswith("kind=njit;")


def test_the_builders_kernel_names_fold_the_keyed_settings(monkeypatch):
    before, _ = builder._kernel_fingerprint("body", [], [], {"cache": True})
    monkeypatch.setattr(numba_config, "BOUNDSCHECK", 1)
    after, _ = builder._kernel_fingerprint("body", [], [], {"cache": True})
    assert after != before


def test_the_builders_derive_anchors_fold_the_keyed_settings(monkeypatch):
    names = []

    def no_anchor(anchor, src):
        names.append(anchor.name)
        return "no anchor in this test"

    monkeypatch.setattr(builder, "_anchor_or_error", no_anchor)
    args = (numba.float64(numba.float64), "float64(float64)", _halved, "a derive's fingerprint", {"cache": True})
    assert builder._derive_anchor_cres(*args) is None
    monkeypatch.setattr(numba_config, "BOUNDSCHECK", 1)
    assert builder._derive_anchor_cres(*args) is None
    assert len(names) == 2 and names[0] != names[1]


_CALLS_BINDING = '''
import json
import sys

from numbox.core.bindings.libm import cos
from numbox.core.cache_key import keyed_njit


@keyed_njit(cache=True)
def calls_cos(x):
    return cos(x) + 1.0


with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump([calls_cos(0.0), sum(calls_cos.stats.cache_hits.values())], fh)
'''


def test_a_keyed_caller_of_a_binding_hits_warm(tmp_path):
    script = tmp_path / "calls_binding.py"
    script.write_text(_CALLS_BINDING, encoding="utf-8")
    cache = tmp_path / "cache"
    assert _run(script, _child_env(cache), tmp_path / "one.json") == [2.0, 0]
    assert _run(script, _child_env(cache), tmp_path / "two.json") == [2.0, 1]


_BINDING = '''
import json
import sys

from numbox.core.bindings.libm import cos

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump([cos(0.0), sum(cos.stats.cache_hits.values()), cos._cache._numbox_keyed[0]], f)
'''


def test_a_binding_misses_under_a_changed_numbox_jit_options(tmp_path):
    """The package's options reach every binding; numba's key does not hold them, numbox's does."""
    script = tmp_path / "binding.py"
    script.write_text(_BINDING, encoding="utf-8")
    cache = tmp_path / "cache"
    assert _run(script, _child_env(cache), tmp_path / "one.json") == [1.0, 0, "njit"]
    assert _run(script, _child_env(cache), tmp_path / "two.json") == [1.0, 1, "njit"]
    fast = _child_env(cache, NUMBOX_JIT_OPTIONS='{"cache": true, "fastmath": true}')
    assert _run(script, fast, tmp_path / "three.json") == [1.0, 0, "njit"]
    assert _run(script, _child_env(cache), tmp_path / "four.json") == [1.0, 1, "njit"]


# ---- Installing the key ----

_NUMBOX_CORE_LOADS = (
    "import sys, numbox.core; print(*sorted(m for m in sys.modules if m.partition('.')[0] == 'numbox'))"
)


def test_every_module_numbox_core_loads_imports_first_in_a_fresh_interpreter(tmp_path):
    """numbox.core installs the cache key as it loads: a module it loads must not need numbox.core while it loads
    itself, or imported first it meets its own half-loaded self."""
    env = _child_env(tmp_path / "cache")
    listing = subprocess.run([sys.executable, "-c", _NUMBOX_CORE_LOADS], capture_output=True, text=True, env=env)
    assert listing.returncode == 0, listing.stderr[-2000:]
    modules = listing.stdout.split()
    assert "numbox.core.cache_key" in modules
    failed = {}
    for name in modules:
        r = subprocess.run([sys.executable, "-c", f"import {name}"], capture_output=True, text=True, env=env)
        if r.returncode:
            failed[name] = r.stderr.strip().splitlines()[-1]
    assert not failed, failed
