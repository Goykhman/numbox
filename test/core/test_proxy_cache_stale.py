"""Cross-process tests of how numbox repairs a cached caller of a proxied body after the body changes.

``@proxy`` publishes each body under an alias that folds the body's fingerprint, so an edit to the body renames the
alias. numba's cache key for a ``cache=True`` caller does not cover its callees, so after the edit the caller's entry
still loads and refers to an alias that the new process does not register. Without numbox's cache check, the process
then crashes, as ``test_stale_proxy_alias_is_discarded_and_recompiled`` describes. The tests check that numbox discards
such an entry with a warning and that numba compiles the caller again, for ``@njit`` and ``@guvectorize`` callers, for a
binding whose C symbol disappears, and in strict mode. Each scenario also includes a binding that is never edited,
with a caller in the same file and cache directory. That caller must load from the cache without a warning, so a fix
that disabled caching or discarded the whole directory fails. Other tests cover a caller that reads ``.as_func`` as a
module-level constant. Its code refers to the companion alias, under which numbox publishes the body's numba entry
point, and numbox repairs it in the same way. The in-process behavior of aliases, collisions and traps is tested in
``test_proxy.py``.
"""
import os
import subprocess
import sys
import textwrap

import pytest

from numbox.core.proxy.proxy import _undefined_symbols

_EDITED = "proxied_binding.py"
_HEALED = {"call_scale": "compiled", "call_scale2": "compiled", "call_offset": "served", "plain": "served"}
_OLD_BODY = {"call_scale": "11.0", "call_scale2": "12.0", "call_offset": "108.0", "plain": "12.0"}
_NEW_BODY = {"call_scale": "16.0", "call_scale2": "17.0", "call_offset": "108.0", "plain": "12.0"}


def _write_binding(path, name, expr):
    """Write a module with one proxied binding named ``name`` that returns ``expr``.

    A rewrite of an existing file moves its modification time 10 seconds forward instead of relying on the clock. The
    new body has the same length as the old one. numba's source stamp is ``(st_mtime, st_size)``, and CPython
    validates a ``.pyc`` file by the same pair. On a file system with coarse timestamps, a rewrite within the same
    second would otherwise leave the stamp unchanged, and numba would not detect the edit. If the stamp still matches
    after the rewrite, the function raises ``RuntimeError``, because the scenario would then test nothing. ``proxy``
    requires the decorator line to be on line 8 or later, and the generated module puts it on line 11.
    """
    before = path.stat() if path.exists() else None
    path.write_text(textwrap.dedent(f'''
        """A proxied binding. The alias published for this body folds the body's fingerprint, so any edit below
        renames it and leaves every cached object that referenced the old name pointing at nothing.
        """
        from numba import types
        from numbox.core.proxy.proxy import proxy

        SIG = types.float64(types.float64)


        @proxy(SIG, jit_options={{"cache": True}})
        def {name}(x):
            return {expr}
    '''), encoding="utf-8")
    if before is None:
        return
    os.utime(path, (before.st_mtime + 10,) * 2)
    after = path.stat()
    if (after.st_mtime, after.st_size) == (before.st_mtime, before.st_size):
        raise RuntimeError(
            f"{path.name} kept its (st_mtime, st_size) across the rewrite, so the edit is invisible to numba and "
            "the scenario would quietly prove nothing"
        )


def _write_scenario(tmp_path):
    """Write the bindings, their cached callers and the probe script into ``tmp_path``, and return the probe's path.

    The two bindings are in separate files, so an edit to one changes neither the alias nor the source stamp of the
    other. The four cached callers share one file, so the controls share the module, the cache directory and the probe
    run with the stale callers. ``call_scale`` and ``call_scale2`` both call the edited binding, so the test can check
    that numbox discards every stale entry, not only the first one that numba loads. ``plain`` refers to no alias, so
    the cache check returns early for it, and it must load from the cache as before.
    """
    _write_binding(tmp_path / _EDITED, "scale", "x * 2.0")
    _write_binding(tmp_path / "proxied_binding_stable.py", "offset", "x + 100.0")
    (tmp_path / "cached_callers.py").write_text(textwrap.dedent('''
        """Cached callers of both bindings, in a file the edit never touches, so their numba cache keys still match
        afterwards and the warm entries are served."""
        from numba import njit

        from proxied_binding import scale
        from proxied_binding_stable import offset


        @njit(cache=True)
        def call_scale(x):
            return scale(x) + 1.0


        @njit(cache=True)
        def call_scale2(x):
            return scale(x) + 2.0


        @njit(cache=True)
        def call_offset(x):
            return offset(x) + 3.0


        @njit(cache=True)
        def plain(x):
            return x + 7.0
    '''), encoding="utf-8")

    probe = tmp_path / "stale_alias_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        import warnings

        from numba.core.types import float64

        import proxied_binding as edited
        import proxied_binding_stable as stable

        print("ALIAS_edited", edited.scale._numbox_proxy_alias, flush=True)
        print("ALIAS_stable", stable.offset._numbox_proxy_alias, flush=True)
        print("BINDINGS_IMPORTED", flush=True)

        # The discard warning is caught here rather than read off stderr: an inherited -W setting would otherwise
        # decide whether it is printed, swallowed, or fatal, and the default filter shows a given warning once.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from cached_callers import call_offset, call_scale, call_scale2, plain
            fns = {"call_scale": call_scale, "call_scale2": call_scale2, "call_offset": call_offset, "plain": plain}
            print("CALLERS_IMPORTED", flush=True)
            if os.environ["PRECOMPILE"] == "1":
                for name, fn in fns.items():
                    fn.compile((float64,))
                    print("COMPILED", name, flush=True)
            for name, fn in fns.items():
                print(f"RESULT_{name}", fn(5.0), flush=True)

        for name, fn in fns.items():
            hits = sum(fn.stats.cache_hits.values())
            misses = sum(fn.stats.cache_misses.values())
            state = "served" if hits and not misses else "compiled" if misses and not hits else f"{hits}/{misses}"
            print(f"STATS_{name}", state, flush=True)
        print("WARNINGS", " | ".join(f"{w.category.__name__}: {w.message}".replace("\\n", " ") for w in caught))
        print("DONE ok", flush=True)
    '''), encoding="utf-8")
    return probe


def _edit_body(tmp_path):
    """Change the edited binding's body from ``x * 2.0`` to ``x * 3.0``, a change that renames the alias."""
    _write_binding(tmp_path / _EDITED, "scale", "x * 3.0")


def _probe_env(tmp_path, cache_name, precompile=False):
    """Return the environment for one probe run, with its own ``NUMBA_CACHE_DIR`` under ``tmp_path``.

    numbox writes its own anchor files under ``NUMBA_CACHE_DIR`` as well, so each test keeps the whole scenario out of
    the developer's cache. ``PYTHONPATH`` holds ``tmp_path`` and this process's ``sys.path``, so the probe imports
    numbox from where this test found it. ``tmp_path`` is listed explicitly, because ``PYTHONSAFEPATH`` removes the
    script's directory from the path. Bytecode writing is disabled, because CPython also checks a ``.pyc`` file by
    ``(st_mtime, st_size)`` and could miss a same-length edit.

    The function removes inherited variables that would change what the scenario tests. ``PYTHONWARNINGS`` and
    ``PYTHONDEVMODE`` change the warning filters, and ``PYTHONWARNINGS=error`` would turn an unrelated warning into an
    error that ends the probe. ``NUMBA_DISABLE_JIT`` would disable compilation. Under ``NUMBA_OPT=0``, numba does not
    cache a caller of a proxied function and warns that the caller uses dynamic globals. ``NUMBOX_PROXY_CACHE_STRICT``
    would turn every repair below into an error, so a strict run sets it explicitly.
    """
    env = dict(os.environ)
    for hostile in ("PYTHONWARNINGS", "PYTHONDEVMODE", "NUMBA_DISABLE_JIT", "NUMBA_OPT", "NUMBOX_PROXY_CACHE_STRICT"):
        env.pop(hostile, None)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / cache_name)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path)] + sys.path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PRECOMPILE"] = "1" if precompile else "0"
    return env


def _run_probe(probe, env):
    """Run a probe to completion and return the ``CompletedProcess``.

    Some scenarios expect the child to crash, so the caller decides which return code is acceptable. With a timeout
    of 300 seconds, ``subprocess.run`` kills a child that crashed but does not exit, for example because a crash
    reporter keeps the process alive.
    """
    return subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=300,
    )


def _report(r):
    """Return a run's return code, stdout and stderr as one text, so that a failure shows how the child ended."""
    return f"rc={r.returncode}\n--- stdout ---\n{r.stdout}--- stderr ---\n{r.stderr}"


def _fields(r):
    """Return the probe's ``KEY value`` lines as a dict, after asserting that the probe printed ``DONE ok``.

    ``DONE`` is missing if the child crashed before the end.
    """
    fields = dict(line.split(" ", 1) for line in r.stdout.splitlines() if " " in line)
    assert fields.get("DONE") == "ok", f"the probe did not run to completion\n{_report(r)}"
    return fields


def _warm(probe, env):
    """Run the probe to fill the cache, and return its fields.

    A failure here means that the scenario is broken, not that the behavior under test failed, so the function raises
    ``RuntimeError`` instead of failing an assertion.
    """
    r = _run_probe(probe, env)
    if r.returncode != 0 or "DONE ok" not in r.stdout:
        raise RuntimeError(f"warming the cache failed\n{_report(r)}")
    return _fields(r)


def _stats(fields, names=_HEALED):
    return {name: fields[f"STATS_{name}"] for name in names}


def _results(fields, names=_HEALED):
    return {name: fields[f"RESULT_{name}"] for name in names}


def test_unedited_rerun_is_served_from_cache(tmp_path):
    """An unchanged second run loads every caller from the cache and returns the same results.

    The other tests depend on this: a caller that numba compiled again would never load a stale entry. The test runs
    the probe twice against one cache directory.
    """
    probe = _write_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")

    cold = _warm(probe, env)
    assert _stats(cold) == dict.fromkeys(_HEALED, "compiled"), _stats(cold)
    assert _results(cold) == _OLD_BODY, _results(cold)

    warm = _warm(probe, env)
    assert _stats(warm) == dict.fromkeys(_HEALED, "served"), f"an unchanged re-run recompiled: {_stats(warm)}"
    assert _results(warm) == _results(cold), _results(warm)


def test_editing_a_proxied_body_renames_only_that_binding_alias(tmp_path):
    """The edit renames the alias of the edited binding and leaves the alias of the other binding unchanged.

    The edit therefore makes only the callers of the edited binding stale, and the other binding remains a valid
    control. The test runs the edited scenario against a new cache directory, so no stale entry is involved, and it
    checks the new results.
    """
    probe = _write_scenario(tmp_path)
    before = _warm(probe, _probe_env(tmp_path, "nbcache"))

    _edit_body(tmp_path)
    after = _warm(probe, _probe_env(tmp_path, "nbcache_after_edit"))

    assert after["ALIAS_edited"] != before["ALIAS_edited"], "the edited body kept its alias"
    assert after["ALIAS_stable"] == before["ALIAS_stable"], "the untouched body was re-keyed"
    assert _results(after) == _NEW_BODY, _results(after)


@pytest.mark.parametrize("precompile", [False, True], ids=["call", "compile"])
def test_stale_proxy_alias_is_discarded_and_recompiled(tmp_path, precompile):
    """After an edit to a proxied body, numbox discards the stale cached callers, and numba compiles them again.

    Without the cache check, the process crashes. In the ``call`` case, the first call of a stale caller crashes with a
    segmentation fault: stdout stops after the callers are imported, and stderr is empty. LLVM's runtime linker resolves
    an object's external symbols in one batch, so the one missing alias zeroes every external relocation in the loaded
    object. In the ``compile`` case, the stale load itself succeeds, because no code in the object has run yet. The
    next cached load of any object then aborts the process with ``LLVM ERROR: Symbol not found`` and the old alias,
    because the execution engine retains the error of the first load.

    With the check, both callers of the edited body compile again, the caller of the other binding loads from the
    cache, and a ``StaleProxyCacheWarning`` names the old alias. A third run loads every caller from the cache without
    a warning, so the repair costs one compilation, not one in every later process.
    """
    probe = _write_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache", precompile=precompile)
    stale_alias = _warm(probe, env)["ALIAS_edited"]

    _edit_body(tmp_path)
    r = _run_probe(probe, env)
    if r.returncode == 0 and _fields(r)["STATS_call_offset"] != "served":
        raise RuntimeError(f"the edit re-keyed the callers too, so no stale entry was ever loaded\n{_report(r)}")
    assert r.returncode == 0, f"a cache entry referencing an unregistered proxy alias reached the engine\n{_report(r)}"
    fields = _fields(r)
    assert _results(fields) == _NEW_BODY, _results(fields)
    assert _stats(fields) == _HEALED, f"the wrong entries were discarded: {_stats(fields)}"
    assert "StaleProxyCacheWarning" in fields["WARNINGS"], f"the discard was not announced: {fields['WARNINGS']}"
    assert stale_alias in fields["WARNINGS"], f"no warning named {stale_alias}: {fields['WARNINGS']}"

    healed = _fields(_run_probe(probe, env))
    assert _stats(healed) == dict.fromkeys(_HEALED, "served"), f"the recompile was not written back: {_stats(healed)}"
    assert "StaleProxyCacheWarning" not in healed["WARNINGS"], f"the heal repeats forever: {healed['WARNINGS']}"


_CONST_FNS = ("const_scale", "disp_scale", "const_offset", "disp_offset")
_CONST_OLD_BODY = {"const_scale": "11.0", "disp_scale": "12.0", "const_offset": "108.0", "disp_offset": "109.0",
                   "direct": "10.0"}
_CONST_NEW_BODY = {"const_scale": "16.0", "disp_scale": "17.0", "const_offset": "108.0", "disp_offset": "109.0",
                   "direct": "15.0"}


def _write_const_caller_scenario(tmp_path):
    """Write the two bindings, four cached callers and the probe script, and return the probe's path.

    A ``const_*`` caller reads a binding's ``.as_func`` as a module-level constant. numbox lowers the constant as a
    declaration of the companion alias, the name under which the body's numba entry point is published, so the
    caller's object imports that name. A ``disp_*`` caller calls the dispatcher, and its code refers to the alias of
    the body's C entry point. The cache check looks for both names. Both routes call the edited binding and the
    unchanged one. The callers of the unchanged binding are the control: a repair of the constant route that stopped
    caching it would make ``const_offset`` compile again. The signatures are explicit, so every caller compiles at
    import, inside the probe's warning capture.
    """
    _write_binding(tmp_path / _EDITED, "scale", "x * 2.0")
    _write_binding(tmp_path / "proxied_binding_stable.py", "offset", "x + 100.0")
    (tmp_path / "const_callers.py").write_text(textwrap.dedent('''
        """Cached callers of both bindings by both routes, in a file the edit never touches."""
        from numba import njit
        from numba.core.types import float64

        from proxied_binding import scale
        from proxied_binding_stable import offset

        SCALE_AS_FUNC = scale.as_func
        OFFSET_AS_FUNC = offset.as_func


        @njit(float64(float64), cache=True)
        def const_scale(x):
            return SCALE_AS_FUNC(x) + 1.0


        @njit(float64(float64), cache=True)
        def disp_scale(x):
            return scale(x) + 2.0


        @njit(float64(float64), cache=True)
        def const_offset(x):
            return OFFSET_AS_FUNC(x) + 3.0


        @njit(float64(float64), cache=True)
        def disp_offset(x):
            return offset(x) + 4.0
    '''), encoding="utf-8")

    probe = tmp_path / "const_caller_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        import pathlib
        import warnings

        import proxied_binding as edited

        print("ALIAS_edited", edited.scale._numbox_proxy_alias, flush=True)

        # Whether numba refused to cache a caller is reported as a warning at its compile, so the import has to
        # happen under the capture, as does every call: the refusal is re-emitted on each run.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import const_callers
            fns = {name: getattr(const_callers, name)
                   for name in ("const_scale", "disp_scale", "const_offset", "disp_offset")}
            for name, fn in fns.items():
                print(f"RESULT_{name}", fn(5.0), flush=True)
            print("RESULT_direct", edited.scale(5.0), flush=True)

        # Counting the files a caller actually left behind is the only way to tell "cached" from "refused, and
        # recompiled quietly enough to look the same": both report a miss on every run.
        cache_dir = pathlib.Path(os.environ["NUMBA_CACHE_DIR"])
        for name, fn in fns.items():
            hits = sum(fn.stats.cache_hits.values())
            misses = sum(fn.stats.cache_misses.values())
            state = "served" if hits and not misses else "compiled" if misses and not hits else f"{hits}/{misses}"
            print(f"STATS_{name}", state, flush=True)
            for kind in ("nbi", "nbc"):
                written = sum(1 for _ in cache_dir.rglob(f"const_callers.{name}-*.{kind}"))
                print(f"{kind.upper()}_{name}", written, flush=True)
        print("WARNINGS", " | ".join(f"{w.category.__name__}: {w.message}".replace("\\n", " ") for w in caught))
        print("DONE ok", flush=True)
    '''), encoding="utf-8")
    return probe


def test_a_const_reference_caller_becomes_cacheable(tmp_path):
    """numba caches a ``cache=True`` caller that reads a proxied binding's ``.as_func`` as a compile-time constant.

    numbox lowers the constant by declaring the body's numba entry point as a symbol, the companion alias. The globals
    that hold the value's other addresses are unused, and LLVM removes them before numba checks the module for dynamic
    globals. Otherwise numba would refuse to cache the caller on every run, with ``Cannot cache
    compiled function "const_scale" as it uses dynamic globals``, and every process would compile it again. The test
    asserts that no such warning appears and that the cache holds the caller's index and data files. ``disp_offset``,
    a dispatcher caller of the unchanged binding, must also load from the cache in the second run.
    """
    probe = _write_const_caller_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")

    cold = _warm(probe, env)
    assert _results(cold, _CONST_OLD_BODY) == _CONST_OLD_BODY, _results(cold, _CONST_OLD_BODY)
    assert cold["STATS_const_scale"] == "compiled", f"a cold run served something: {_stats(cold, _CONST_FNS)}"

    warm = _warm(probe, env)
    assert warm["STATS_disp_offset"] == "served", f"the control caller recompiled: {_stats(warm, _CONST_FNS)}"
    refused = 'Cannot cache compiled function "const_scale"'

    assert warm["STATS_const_scale"] == "served", f"the const caller did not cache: {_stats(warm, _CONST_FNS)}"
    assert refused not in cold["WARNINGS"], f"numba declined the const caller: {cold['WARNINGS']}"
    assert refused not in warm["WARNINGS"], f"numba declined the const caller: {warm['WARNINGS']}"
    assert int(cold["NBI_const_scale"]) >= 1 and int(cold["NBC_const_scale"]) >= 1, (
        f"nothing was written for the const caller: {cold['NBI_const_scale']}/{cold['NBC_const_scale']}"
    )
    assert warm["STATS_const_offset"] == "served", (
        f"the second const caller did not cache, so only one binding's const route is pinned: "
        f"{_stats(warm, _CONST_FNS)}"
    )


def test_a_const_reference_caller_is_repaired_after_an_edit(tmp_path):
    """After an edit, numbox discards a cached caller that reads ``.as_func`` as a constant, and it runs the new body.

    The edit renames the body's alias and the companion alias. The caller's cached object imports the old companion,
    which the new process does not register. The cache check therefore discards the entry with a
    ``StaleProxyCacheWarning`` that names the caller and the old companion, and numba compiles the caller again. The
    dispatcher caller is repaired in the same run. If the constant were lowered with the body's code linked into the
    caller, the caller's object would import no alias, and ``const_scale`` would return 11.0, the old body's result,
    without a warning.

    ``const_offset`` and ``disp_offset`` are the control: a repair that discarded the whole directory would make them
    compile again. A third run loads every caller from the cache without a warning, so the repair costs one
    compilation, not one in every later process.
    """
    probe = _write_const_caller_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")
    cold = _warm(probe, env)
    assert _results(cold, _CONST_OLD_BODY) == _CONST_OLD_BODY
    stale_companion = cold["ALIAS_edited"] + "_jit"

    _edit_body(tmp_path)
    fields = _fields(_run_probe(probe, env))

    assert _results(fields, _CONST_NEW_BODY) == _CONST_NEW_BODY, (
        f"a caller did not reach the new body: {_results(fields, _CONST_NEW_BODY)}"
    )
    assert fields["STATS_const_scale"] == "compiled" and fields["STATS_disp_scale"] == "compiled", (
        f"a caller of the edited binding was not compiled again: {_stats(fields, _CONST_FNS)}"
    )
    assert fields["STATS_const_offset"] == "served" and fields["STATS_disp_offset"] == "served", (
        f"the untouched binding's callers were invalidated too: {_stats(fields, _CONST_FNS)}"
    )
    assert "const_callers.const_scale-" in fields["WARNINGS"], (
        f"the discard of the const caller went unannounced: {fields['WARNINGS']}"
    )
    assert stale_companion in fields["WARNINGS"], f"no warning named {stale_companion}: {fields['WARNINGS']}"
    assert "const_callers.disp_scale-" in fields["WARNINGS"], (
        f"the discard of the dispatcher caller went unannounced: {fields['WARNINGS']}"
    )

    healed = _fields(_run_probe(probe, env))
    assert _results(healed, _CONST_NEW_BODY) == _CONST_NEW_BODY, _results(healed, _CONST_NEW_BODY)
    assert _stats(healed, _CONST_FNS) == dict.fromkeys(_CONST_FNS, "served"), (
        f"the recompile was not written back: {_stats(healed, _CONST_FNS)}"
    )
    assert "StaleProxyCacheWarning" not in healed["WARNINGS"], f"the repair repeats forever: {healed['WARNINGS']}"


def _write_cres_const_scenario(tmp_path):
    """Write two ``cres`` functions, two cached callers that hold them as constants, and the probe; return its path.

    The functions and their callers are in two files, so an edit to a function leaves the callers' file unchanged.
    Neither file imports ``numbox.core.proxy``. The probe reports whether the cache check is installed, because the
    import of ``numbox.utils.highlevel`` alone must install it. Without the check, a cached caller that imports an
    alias this process has not registered would crash the process when numba loads it.
    """
    (tmp_path / "cres_ops.py").write_text(textwrap.dedent('''
        """Two functions compiled with cres. The edit changes the first and leaves the second as the control."""
        from numba.core.types import float64

        from numbox.utils.highlevel import cres


        @cres(float64(float64), cache=True)
        def scale(x):
            return x * 2.0


        @cres(float64(float64), cache=True)
        def offset(x):
            return x + 100.0
    '''), encoding="utf-8")
    (tmp_path / "cres_const_callers.py").write_text(textwrap.dedent('''
        """Cached callers that read the two functions as module-level constants, in a file the edit never touches."""
        from numba import njit
        from numba.core.types import float64

        from cres_ops import offset, scale


        @njit(float64(float64), cache=True)
        def const_scale(x):
            return scale(x) + 1.0


        @njit(float64(float64), cache=True)
        def const_offset(x):
            return offset(x) + 3.0
    '''), encoding="utf-8")

    probe = tmp_path / "cres_const_probe.py"
    probe.write_text(textwrap.dedent('''
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import cres_const_callers
            fns = {name: getattr(cres_const_callers, name) for name in ("const_scale", "const_offset")}
            for name, fn in fns.items():
                print(f"RESULT_{name}", fn(5.0), flush=True)

        for name, fn in fns.items():
            hits = sum(fn.stats.cache_hits.values())
            misses = sum(fn.stats.cache_misses.values())
            state = "served" if hits and not misses else "compiled" if misses and not hits else f"{hits}/{misses}"
            print(f"STATS_{name}", state, flush=True)

        import cres_ops
        from numba.core import caching

        print("ALIAS_scale", cres_ops.scale._numbox_jit_alias, flush=True)
        print("GUARD", getattr(caching, "_numbox_proxy_alias_guard", False), flush=True)
        print("WARNINGS", " | ".join(f"{w.category.__name__}: {w.message}".replace("\\n", " ") for w in caught))
        print("DONE ok", flush=True)
    '''), encoding="utf-8")
    return probe


_CRES_CONST_FNS = ("const_scale", "const_offset")


def test_a_caller_of_a_cres_constant_is_repaired_after_an_edit(tmp_path):
    """After an edit to a ``cres`` function, numbox discards a cached caller that holds it as a constant.

    ``cres`` publishes the function's numba entry point under an alias, and the caller's cached object imports that
    name. The edit renames the alias, so the cache check discards the caller's entry with a ``StaleProxyCacheWarning``
    that names the caller and the old alias, and numba compiles the caller again. If the function's code were linked
    into the caller instead, ``const_scale`` would keep returning 11.0, the old function's result, without a warning.

    ``const_offset`` is the control: its function is in the edited file but is unchanged, so its alias is the same and
    the caller must still load from the cache. A third run loads both callers from the cache without a warning.
    """
    probe = _write_cres_const_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")

    cold = _warm(probe, env)
    assert _results(cold, _CRES_CONST_FNS) == {"const_scale": "11.0", "const_offset": "108.0"}
    assert cold["GUARD"] == "True", "the cache check is not installed in a process that only uses cres"
    warm = _warm(probe, env)
    assert _stats(warm, _CRES_CONST_FNS) == dict.fromkeys(_CRES_CONST_FNS, "served"), (
        f"a caller of a cres constant did not cache: {_stats(warm, _CRES_CONST_FNS)}\n{warm['WARNINGS']}"
    )

    _rewrite(tmp_path / "cres_ops.py", "x * 2.0", "x * 3.0")
    fields = _fields(_run_probe(probe, env))

    assert _results(fields, _CRES_CONST_FNS) == {"const_scale": "16.0", "const_offset": "108.0"}, (
        f"the caller did not reach the new function: {_results(fields, _CRES_CONST_FNS)}"
    )
    assert _stats(fields, _CRES_CONST_FNS) == {"const_scale": "compiled", "const_offset": "served"}, (
        f"the edit did not invalidate exactly the caller of the edited function: {_stats(fields, _CRES_CONST_FNS)}"
    )
    assert "cres_const_callers.const_scale-" in fields["WARNINGS"], (
        f"the discard of the caller went unannounced: {fields['WARNINGS']}"
    )
    assert cold["ALIAS_scale"] in fields["WARNINGS"], f"no warning named {cold['ALIAS_scale']}: {fields['WARNINGS']}"
    assert fields["ALIAS_scale"] != cold["ALIAS_scale"], "the edit did not rename the alias"

    healed = _fields(_run_probe(probe, env))
    assert _results(healed, _CRES_CONST_FNS) == {"const_scale": "16.0", "const_offset": "108.0"}
    assert _stats(healed, _CRES_CONST_FNS) == dict.fromkeys(_CRES_CONST_FNS, "served"), (
        f"the recompile was not written back: {_stats(healed, _CRES_CONST_FNS)}"
    )
    assert "StaleProxyCacheWarning" not in healed["WARNINGS"], f"the repair repeats forever: {healed['WARNINGS']}"


def _write_shared_alias_scenario(tmp_path):
    """Write a factory that makes two ``cres`` functions with one alias, a cached caller of one, and the probe.

    The two functions call ``twin_helpers.op``, a module attribute that holds ``numpy.sin`` for one and ``numpy.cos``
    for the other. numba compiles the ufunc into each function, but the alias leaves a ufunc out, so both functions
    compute one alias. ``TWINS`` selects which of them the process makes: ``alone`` makes only the sine, and
    ``sin_first`` and ``cos_first`` make both in that order. The caller holds the sine as a constant.
    """
    (tmp_path / "twin_helpers.py").write_text(textwrap.dedent('''
        """A module attribute that holds a ufunc. The functions in twin_ops read it, and the factory there rebinds it."""
        import numpy

        op = numpy.sin
    '''), encoding="utf-8")
    (tmp_path / "twin_ops.py").write_text(textwrap.dedent('''
        """Two functions that one factory makes. Their code is equal, and they differ in the ufunc compiled in."""
        import os

        import numpy
        from numba.core.types import float64

        import twin_helpers
        from numbox.utils.highlevel import cres


        def make(op):
            twin_helpers.op = op

            @cres(float64(float64))
            def twin(x):
                return twin_helpers.op(x)
            return twin


        order = os.environ["TWINS"]
        COS = make(numpy.cos) if order == "cos_first" else None
        SIN = make(numpy.sin)
        if order == "sin_first":
            COS = make(numpy.cos)
    '''), encoding="utf-8")
    (tmp_path / "twin_callers.py").write_text(textwrap.dedent('''
        """A cached caller that reads the sine as a module-level constant."""
        from numba import njit
        from numba.core.types import float64

        from twin_ops import SIN


        @njit(float64(float64), cache=True)
        def call_sin(x):
            return SIN(x)
    '''), encoding="utf-8")

    probe = tmp_path / "twin_probe.py"
    probe.write_text(textwrap.dedent('''
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import twin_callers
            print("RESULT_call_sin", twin_callers.call_sin(0.0), flush=True)

        fn = twin_callers.call_sin
        hits = sum(fn.stats.cache_hits.values())
        misses = sum(fn.stats.cache_misses.values())
        state = "served" if hits and not misses else "compiled" if misses and not hits else f"{hits}/{misses}"
        print("STATS_call_sin", state, flush=True)

        import twin_ops

        print("ALIAS_sin", twin_ops.SIN._numbox_jit_alias, flush=True)
        print("ALIAS_cos", getattr(twin_ops.COS, "_numbox_jit_alias", "none"), flush=True)
        print("WARNINGS", " | ".join(f"{w.category.__name__}: {w.message}".replace("\\n", " ") for w in caught))
        print("DONE ok", flush=True)
    '''), encoding="utf-8")
    return probe


@pytest.mark.parametrize("order", ["sin_first", "cos_first"])
def test_a_caller_of_a_constant_is_not_served_another_function_with_the_same_alias(tmp_path, order):
    """A cached caller that holds a function as a constant never runs another function that computes the same alias.

    The caller is cached in a process that makes only the sine, and a second run there serves it from the cache. The
    next process makes the cosine as well, which computes the same alias. The caller must still return ``sin(0.0)``,
    and numbox must discard its cache entry instead of loading it. The entry refers to the alias's name, and that name
    belongs to whichever function the process made first. The name resolves to the cosine in the order ``cos_first``,
    so a caller loaded from the cache would return 1.0. In the order ``sin_first``, the name would resolve to the
    cosine only if the second registration replaced the first.

    A further run in the same process shape must discard the entry again. The entry that the previous run wrote refers
    to a name that two functions share, or to a name used only in the process that wrote it.
    """
    probe = _write_shared_alias_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")

    env["TWINS"] = "alone"
    cold = _warm(probe, env)
    assert cold["RESULT_call_sin"] == "0.0"
    warm = _warm(probe, env)
    assert warm["STATS_call_sin"] == "served", f"the caller of a constant did not cache: {warm['WARNINGS']}"

    env["TWINS"] = order
    for run in ("first", "second"):
        fields = _fields(_run_probe(probe, env))
        assert fields["ALIAS_sin"].split("_jit")[0] == fields["ALIAS_cos"].split("_jit")[0], (
            "the two functions did not compute one alias, so the test would pass without exercising the rule"
        )
        assert fields["RESULT_call_sin"] == "0.0", (
            f"on the {run} run with both functions, the caller of the sine ran the cosine\n{fields['WARNINGS']}"
        )
        assert fields["STATS_call_sin"] == "compiled", (
            f"on the {run} run, the caller was loaded from the cache although two functions share its alias"
        )
        assert "twin_callers.call_sin-" in fields["WARNINGS"], f"the discard went unannounced: {fields['WARNINGS']}"


def _rewrite(path, old, new):
    """Replace ``old`` with ``new`` in ``path``, and move the file's modification time 10 seconds forward.

    ``_write_binding`` moves the time in the same way, so that numba detects the edit.
    """
    before = path.stat()
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    os.utime(path, (before.st_mtime + 10,) * 2)


def _write_two_proxy_scenario(tmp_path):
    """Write two bindings in separate files, one cached caller of both, and the probe script; return the probe's path.

    The caller's object imports both aliases. Each binding's ``@proxy`` line is on line 9, and ``proxy`` requires line
    8 or later. An edit to both bodies renames both aliases, so the caller's object refers to two names that the
    process does not register. A warning that named only the first stale alias would then name too few.
    """
    (tmp_path / "bind_a.py").write_text(textwrap.dedent('''
        """binding a."""
        from numba import types
        from numbox.core.proxy.proxy import proxy

        SIG = types.float64(types.float64)


        @proxy(SIG, jit_options={"cache": True})
        def scale(x):
            return x * 2.0
    '''), encoding="utf-8")
    (tmp_path / "bind_b.py").write_text(textwrap.dedent('''
        """binding b."""
        from numba import types
        from numbox.core.proxy.proxy import proxy

        SIG = types.float64(types.float64)


        @proxy(SIG, jit_options={"cache": True})
        def shift(x):
            return x + 10.0
    '''), encoding="utf-8")
    (tmp_path / "two_caller.py").write_text(textwrap.dedent('''
        """One cached caller of both proxies, in a file the edits never touch."""
        from numba import njit

        from bind_a import scale
        from bind_b import shift


        @njit(cache=True)
        def call_both(x):
            return scale(x) + shift(x) + 1.0
    '''), encoding="utf-8")
    probe = tmp_path / "two_probe.py"
    probe.write_text(textwrap.dedent('''
        import warnings

        import bind_a
        import bind_b

        print("ALIAS_a", bind_a.scale._numbox_proxy_alias, flush=True)
        print("ALIAS_b", bind_b.shift._numbox_proxy_alias, flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from two_caller import call_both
            print("RESULT", call_both(5.0), flush=True)
        stale = [w for w in caught if w.category.__name__ == "StaleProxyCacheWarning"]
        print("N_STALE_WARNINGS", len(stale), flush=True)
        for w in stale:
            print("WARN", str(w.message).replace(chr(10), " "), flush=True)
        print("DONE ok", flush=True)
    '''), encoding="utf-8")
    return probe


def test_a_multi_alias_object_names_every_stale_alias_no_undercount(tmp_path):
    """A cached object that imports two stale aliases causes one ``StaleProxyCacheWarning``, which names both.

    ``_stale_proxy_aliases`` returns every stale alias that the object imports, and ``_guarded_rebuild`` joins them in
    one message. An implementation that stopped at the first stale symbol would name only one. The test reads the old
    aliases from the first run's output and asserts nothing about compilation counts.
    """
    probe = _write_two_proxy_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")

    warm = _fields(_run_probe(probe, env))
    old_a, old_b = warm["ALIAS_a"], warm["ALIAS_b"]

    for name, old, new in (("bind_a.py", "x * 2.0", "x * 3.0"), ("bind_b.py", "x + 10.0", "x + 20.0")):
        _rewrite(tmp_path / name, old, new)

    fields = _fields(_run_probe(probe, env))
    assert fields["ALIAS_a"] != old_a and fields["ALIAS_b"] != old_b, "an edited body kept its alias"
    assert fields["N_STALE_WARNINGS"] == "1", (
        f"the single stale object should draw exactly one warning, got {fields['N_STALE_WARNINGS']}"
    )
    warned = fields["WARN"]
    for retired in (old_a, old_b):
        assert retired in warned, f"the warning undercounted -- it did not name {retired}: {warned}"


def test_the_discard_can_be_escalated_to_an_error(tmp_path):
    """A warnings filter can turn ``StaleProxyCacheWarning`` into an error, and the stale load then fails.

    The warning is a class of the public module ``numbox.core.proxy.proxy``, so a filter can select it without catching
    every ``RuntimeWarning``. The error is raised before numbox discards the entry, so the stale entry stays on disk for
    inspection.
    """
    probe = _write_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")
    _warm(probe, env)

    strict = tmp_path / "strict_probe.py"
    strict.write_text(textwrap.dedent('''
        import warnings

        from numbox.core.proxy.proxy import StaleProxyCacheWarning

        warnings.filterwarnings("error", category=StaleProxyCacheWarning)
        from cached_callers import call_scale
        print("RESULT", call_scale(5.0), flush=True)
    '''), encoding="utf-8")

    _edit_body(tmp_path)
    r = _run_probe(strict, env)
    assert r.returncode != 0, f"the stale entry was healed despite the filter\n{_report(r)}"
    assert "StaleProxyCacheWarning" in r.stderr, f"escalated to something unnamed\n{_report(r)}"


def _write_optional_binding(tmp_path):
    """Write a fake library whose ``scale`` attribute exists only if ``SYMBOL_PRESENT`` is ``1``, and a binding over it.

    The two tests below share the binding. One calls it through its dispatcher, whose code refers to the alias. The
    other reads ``.as_func`` as a compile-time constant, whose code refers to the companion alias. One helper writes
    the binding for both, so the two tests compare the same binding.
    """
    (tmp_path / "fakelib.py").write_text(textwrap.dedent('''
        """Stands in for a C library whose symbol is present in one process and gone in the next."""
        import os


        class _Lib:
            pass


        lib = _Lib()
        if os.environ["SYMBOL_PRESENT"] == "1":
            lib.scale = object()
    '''), encoding="utf-8")
    (tmp_path / "optional_binding.py").write_text(textwrap.dedent('''
        """A binding that exists only when its C symbol does."""
        from numba import types
        from numbox.core.proxy.proxy import proxy_if_available

        from fakelib import lib

        SIG = types.float64(types.float64)


        @proxy_if_available(lib, SIG, jit_options={"cache": True})
        def scale(x):
            return x * 2.0
    '''), encoding="utf-8")


def test_a_binding_that_disappeared_is_discarded_rather_than_called(tmp_path):
    """A cached caller of a ``proxy_if_available`` binding whose symbol has disappeared is discarded, not called.

    In a process without the symbol, numbox registers a trap under the alias that the cached caller refers to, so the
    alias resolves. A call would run the trap, which raises inside a ``@cfunc``. numba discards the exception there and
    returns zero, and the caller would compute with the zero and exit with status 0. The cache check discards the
    entry instead, and the recompile fails with the same ``TypingError`` as with an empty cache.
    """
    _write_optional_binding(tmp_path)
    (tmp_path / "optional_caller.py").write_text(textwrap.dedent('''
        from numba import njit

        from optional_binding import scale


        @njit(cache=True)
        def call_scale(x):
            return scale(x) + 1.0
    '''), encoding="utf-8")
    probe = tmp_path / "optional_probe.py"
    probe.write_text(textwrap.dedent('''
        import warnings

        import optional_binding  # noqa: F401

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from optional_caller import call_scale
            try:
                print("RESULT", call_scale(5.0), flush=True)
            except Exception as exc:
                print("RAISED", type(exc).__name__, flush=True)
        print("WARNINGS", " | ".join(f"{w.category.__name__}" for w in caught))
        print("DONE ok", flush=True)
    '''), encoding="utf-8")

    env = _probe_env(tmp_path, "nbcache")
    env["SYMBOL_PRESENT"] = "1"
    assert _warm(probe, env)["RESULT"] == "11.0"

    env["SYMBOL_PRESENT"] = "0"
    fields = _fields(_run_probe(probe, env))
    assert "RESULT" not in fields, f"the vanished binding was called and returned {fields.get('RESULT')}"
    assert fields["RAISED"] == "TypingError", f"failed in an unexpected way: {fields['RAISED']}"
    assert "StaleProxyCacheWarning" in fields["WARNINGS"], f"the discard was not announced: {fields['WARNINGS']}"


def test_a_const_reference_caller_of_a_binding_that_disappeared_is_discarded(tmp_path):
    """A cached caller that reads a vanished binding's ``.as_func`` as a constant is discarded, not loaded.

    The caller's cached object imports the companion alias of the binding's numba entry point. In a process without
    the binding, nothing registers that name, so the cache check discards the entry with a ``StaleProxyCacheWarning``.
    The recompile then fails with the ``TypingError`` that the same source gives with an empty cache, because the
    module-level variable holds ``None``. In strict mode, the stale entry raises ``StaleProxyCacheError`` instead.

    The caller guards the constant with a conditional expression, so its module imports where the binding is absent.
    If the cached entry loaded there, the caller would return the vanished binding's result, 11.0, and a body that
    called a real C symbol would crash the process with a segmentation fault and an empty stderr. This body calls no
    C symbol, so a regression shows as a result that the test can assert on, not as a crash.
    """
    _write_optional_binding(tmp_path)
    (tmp_path / "const_optional_caller.py").write_text(textwrap.dedent('''
        """A cached caller reading the binding's ``.as_func`` through a guard demoted to a conditional expression,
        so the module imports and the caller is defined even where the binding is not."""
        from numba import njit

        from optional_binding import scale

        SCALE_AS_FUNC = scale.as_func if hasattr(scale, "as_func") else None


        @njit(cache=True)
        def const_scale(x):
            return SCALE_AS_FUNC(x) + 1.0
    '''), encoding="utf-8")
    probe = tmp_path / "const_optional_probe.py"
    probe.write_text(textwrap.dedent('''
        import warnings

        import optional_binding  # noqa: F401

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from const_optional_caller import const_scale
            try:
                print("RESULT", const_scale(5.0), flush=True)
            except Exception as exc:
                print("RAISED", type(exc).__name__, flush=True)
        hits = sum(const_scale.stats.cache_hits.values())
        misses = sum(const_scale.stats.cache_misses.values())
        print("STATS", "served" if hits and not misses else "compiled" if misses and not hits else f"{hits}/{misses}")
        print("WARNINGS", " | ".join(f"{w.category.__name__}" for w in caught))
        print("DONE ok", flush=True)
    '''), encoding="utf-8")

    env = _probe_env(tmp_path, "nbcache")
    env["SYMBOL_PRESENT"] = "1"
    assert _warm(probe, env)["RESULT"] == "11.0"

    env["SYMBOL_PRESENT"] = "0"
    warm = _fields(_run_probe(probe, env))

    cold_env = _probe_env(tmp_path, "nbcache_absent_cold")
    cold_env["SYMBOL_PRESENT"] = "0"
    cold = _fields(_run_probe(probe, cold_env))
    assert cold.get("RAISED") == "TypingError", f"a cold cache did not reach the clean typing error: {cold}"

    assert "RESULT" not in warm, f"the vanished binding was called and returned {warm.get('RESULT')}"
    assert warm.get("RAISED") == "TypingError", f"the warm run failed in another way than the cold one: {warm}"
    assert "StaleProxyCacheWarning" in warm["WARNINGS"], f"the discard was not announced: {warm}"

    # The failed recompile wrote nothing, so the stale entry is still on disk for the strict run.
    strict_env = dict(env)
    strict_env["NUMBOX_PROXY_CACHE_STRICT"] = "1"
    strict = _fields(_run_probe(probe, strict_env))
    assert strict.get("RAISED") == "StaleProxyCacheError", f"strict mode did not stop at the stale entry: {strict}"


def test_the_mach_o_reader_strips_the_leading_underscore(tmp_path):
    """The Mach-O reader returns an imported alias without the leading underscore that Mach-O adds to every C symbol.

    The scenarios above reach the Mach-O reader only on macOS, so this test calls it on a Mach-O object that llvmlite's
    LLVM emits on any platform. A reader that kept the underscore would compare ``_numbox_pxy_...`` with the prefix,
    match nothing, and pass every stale object. That result cannot be distinguished from the result for an object
    with no alias to check. The test skips if llvmlite cannot emit Mach-O here.
    """
    emit = tmp_path / "emit_macho.py"
    emit.write_text(textwrap.dedent('''
        import platform
        import sys

        from llvmlite import binding as ll
        from llvmlite import ir

        arch = {"aarch64": "arm64", "AMD64": "x86_64"}.get(platform.machine(), platform.machine())
        triple = f"{arch}-apple-darwin"
        try:
            ll.initialize_all_targets()
            ll.initialize_all_asmprinters()
            target = ll.Target.from_triple(triple)
        except Exception as exc:
            print("SKIP", triple, exc)
            sys.exit(0)

        mod = ir.Module(name="fixture")
        mod.triple = triple
        fnty = ir.FunctionType(ir.DoubleType(), [ir.DoubleType()])
        callee = ir.Function(mod, fnty, name="numbox_pxy_probe_0123456789abcdef")
        fn = ir.Function(mod, fnty, name="caller")
        builder = ir.IRBuilder(fn.append_basic_block("entry"))
        builder.ret(builder.call(callee, [fn.args[0]]))

        tm = target.create_target_machine(reloc="static", codemodel="large")
        obj = tm.emit_object(ll.parse_assembly(str(mod)))
        with open(sys.argv[1], "wb") as f:
            f.write(obj)
        print("OK", len(obj))
    '''), encoding="utf-8")

    obj_path = tmp_path / "fixture.o"
    r = subprocess.run([sys.executable, str(emit), str(obj_path)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    if r.stdout.startswith("SKIP"):
        pytest.skip(f"llvmlite cannot emit Mach-O here: {r.stdout.strip()}")

    obj = obj_path.read_bytes()
    assert obj[:4] == b"\xcf\xfa\xed\xfe", f"not a 64-bit little-endian Mach-O: {obj[:4]!r}"
    found = _undefined_symbols(obj)
    assert "numbox_pxy_probe_0123456789abcdef" in found, f"the aliased import was not found: {sorted(found)}"
    assert "_numbox_pxy_probe_0123456789abcdef" not in found, "the leading underscore was left on"


def test_the_reader_recognises_the_objects_numba_actually_caches(tmp_path):
    """The symbol reader parses the objects that numba caches on this platform and finds the aliases they import.

    The other tests reach the reader only through a stale entry, so they observe the cache check only through the
    absence of a crash. A reader that recognized no format would return no symbols, find no stale alias and pass every
    object, as if the check were not installed. The only evidence would be a crash on the platform whose format
    stopped matching, with no message that names the cause.

    The format is not obvious from the platform. On Windows the ELF reader matches, because numba's JIT emits ELF
    objects there as well, so a ``get_object_format`` result of COFF does not describe the bytes of the object. The
    test therefore records what the reader receives during a warm run. It needs no stale entry and no edit, so it
    still tests the reader if the crash scenario no longer occurs.
    """
    probe = _write_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")
    _warm(probe, env)

    reader = tmp_path / "reader_probe.py"
    reader.write_text(textwrap.dedent('''
        """Re-run the warm callers with the format reader recording every object it is asked about."""
        import numbox.core.proxy.proxy as proxy_mod

        seen = []
        read_symbols = proxy_mod._undefined_symbols


        def recording(object_code):
            found = read_symbols(object_code)
            seen.append((object_code[:4].hex(), any(s.startswith(proxy_mod._ALIAS_PREFIX) for s in found)))
            return found


        proxy_mod._undefined_symbols = recording

        from cached_callers import call_scale
        print("RESULT", call_scale(5.0), flush=True)
        print("INSPECTED", len(seen), flush=True)
        print("WITH_ALIAS", sum(1 for _magic, found_alias in seen if found_alias), flush=True)
        print("MAGIC", ",".join(sorted({magic for magic, _found_alias in seen})) or "-", flush=True)
        print("DONE ok", flush=True)
    '''), encoding="utf-8")

    fields = _fields(_run_probe(reader, env))
    assert fields["RESULT"] == _OLD_BODY["call_scale"], f"the warm caller did not run: {fields['RESULT']}"
    assert int(fields["INSPECTED"]) > 0, (
        "no alias-bearing object reached the reader, so nothing here was tested: the callers were recompiled "
        "rather than served, or the guard is not installed"
    )
    assert int(fields["WITH_ALIAS"]) > 0, (
        f"the reader parsed no alias out of any object numba cached here, so it passes every stale entry "
        f"untouched: leading magic bytes {fields['MAGIC']}"
    )


def test_strict_mode_raises_on_a_stale_alias_before_healing(tmp_path):
    """In strict mode, loading an entry with a stale alias raises ``StaleProxyCacheError``, and the entry stays on disk.

    By default, numbox discards a stale entry and warns, and numba compiles the caller again. A user who investigates
    a cache problem needs the process to stop at the first stale entry, name it, and leave it on disk. The strict run
    must therefore exit with ``StaleProxyCacheError`` on stderr. A second run without strict mode then finds the same
    entry stale and repairs it with a warning. If strict mode had compiled the entry again, that run would load the
    new body without a warning.
    """
    probe = _write_scenario(tmp_path)
    _warm(probe, _probe_env(tmp_path, "nbcache"))
    _edit_body(tmp_path)

    strict_env = _probe_env(tmp_path, "nbcache")
    strict_env["NUMBOX_PROXY_CACHE_STRICT"] = "1"
    r = _run_probe(probe, strict_env)
    assert r.returncode != 0, f"strict mode healed the stale entry instead of aborting\n{_report(r)}"
    assert "StaleProxyCacheError" in r.stderr, f"the abort was not the named strict error\n{_report(r)}"

    healed = _run_probe(probe, _probe_env(tmp_path, "nbcache"))
    assert healed.returncode == 0, f"the entry strict left behind did not heal with the knob off\n{_report(healed)}"
    fields = _fields(healed)
    assert _results(fields) == _NEW_BODY, _results(fields)
    assert "StaleProxyCacheWarning" in fields["WARNINGS"], (
        f"strict recompiled the entry after all -- nothing was left stale to heal: {fields['WARNINGS']}"
    )


def test_strict_mode_is_quiet_on_a_healthy_warm_cache(tmp_path):
    """Strict mode raises nothing on an unedited warm cache.

    Every caller imports an alias that resolves, and every object parses, so no entry is stale or unreadable. The
    strict run loads every caller from the cache, as the default does, and stderr holds no error.
    """
    probe = _write_scenario(tmp_path)
    _warm(probe, _probe_env(tmp_path, "nbcache"))

    strict_env = _probe_env(tmp_path, "nbcache")
    strict_env["NUMBOX_PROXY_CACHE_STRICT"] = "1"
    r = _run_probe(probe, strict_env)
    assert r.returncode == 0, f"strict mode falsely rejected a healthy warm cache\n{_report(r)}"
    fields = _fields(r)
    assert _stats(fields) == dict.fromkeys(_HEALED, "served"), f"a healthy strict run recompiled: {_stats(fields)}"
    assert _results(fields) == _OLD_BODY, _results(fields)
    assert "Error" not in r.stderr, f"strict emitted an error on a healthy cache\n{_report(r)}"


def test_strict_mode_raises_when_a_payload_cannot_be_read(tmp_path, monkeypatch):
    """In strict mode, a payload that the check cannot unpack raises ``UnvalidatedProxyCacheError``.

    By default, numbox loads such a payload unchecked, as ``test_an_unrecognised_cache_payload_is_handed_on_untouched``
    shows. This test passes the same payloads with ``NUMBOX_PROXY_CACHE_STRICT`` set.
    """
    from numbox.core.proxy.proxy import UnvalidatedProxyCacheError, _guarded_rebuild

    class _Impl:
        filename_base = "unreadable"

    def original(self, target_context, payload):
        return "loaded"

    rebuild = _guarded_rebuild(original, lambda payload: payload[0])
    monkeypatch.setenv("NUMBOX_PROXY_CACHE_STRICT", "1")
    for payload in ({"keyword": "style"}, object(), ()):
        with pytest.raises(UnvalidatedProxyCacheError):
            rebuild(_Impl(), None, payload)


def test_strict_mode_still_loads_a_well_formed_object_with_no_stale_alias(tmp_path, monkeypatch):
    """An object that parses, holds the alias prefix and imports no alias loads with and without strict mode.

    Such an object cannot be distinguished from a valid function with no proxied callee that holds an alias-like
    string, as in ``test_strict_mode_does_not_abort_on_a_healthy_object_embedding_the_prefix`` below. Strict mode
    therefore raises only for a payload whose read raises.
    ``test_the_reader_recognises_the_objects_numba_actually_caches`` checks instead that the reader parses the objects
    that numba emits.
    """
    from numbox.core.proxy.proxy import _guarded_rebuild

    class _Impl:
        filename_base = "prefix-without-import"

    def original(self, target_context, payload):
        return "loaded"

    # The blob holds the alias prefix, so the check does not return early. The blob is neither ELF nor Mach-O, so the
    # reader treats it as COFF, which it does not read, and finds no imported alias.
    blob = b"numbox_pxy_scale_deadbeefdeadbeef" + b"\x00" * 64
    payload = ("libname", "object", (blob,))
    rebuild = _guarded_rebuild(original, lambda p: p)

    assert rebuild(_Impl(), None, payload) == "loaded", "the default must load such an object untouched"
    monkeypatch.setenv("NUMBOX_PROXY_CACHE_STRICT", "1")
    assert rebuild(_Impl(), None, payload) == "loaded", "strict must not abort on a cleanly-read prefix-bearing object"


def test_strict_mode_does_not_abort_on_a_healthy_object_embedding_the_prefix(tmp_path):
    """A cached function with no proxied callee, whose object holds the alias prefix in a string, loads in strict mode.

    The cache check is installed only in a process that imports numbox, so the runner imports it. A user who debugs
    the proxy cache enables strict mode, and such a user is the most likely to have jitted code that handles
    ``numbox_pxy_`` strings. The function returns a character of the literal, so the literal stays in the object
    instead of being optimized away.
    """
    (tmp_path / "prefix_literal_mod.py").write_text(textwrap.dedent('''
        from numba import njit


        @njit(cache=True)
        def has_literal(i):
            label = "numbox_pxy_scale_deadbeefdeadbeef"
            return label[i]
    '''), encoding="utf-8")
    runner = tmp_path / "prefix_literal_run.py"
    runner.write_text(textwrap.dedent('''
        import numbox.core.proxy.proxy  # noqa: F401 -- importing numbox is what installs the cache guard
        from prefix_literal_mod import has_literal
        print("RESULT", has_literal(0), flush=True)
        print("DONE ok", flush=True)
    '''), encoding="utf-8")

    env = _probe_env(tmp_path, "nbcache")
    assert _warm(runner, env)["RESULT"] == "n"

    strict_env = dict(env)
    strict_env["NUMBOX_PROXY_CACHE_STRICT"] = "1"
    r = _run_probe(runner, strict_env)
    assert r.returncode == 0, f"strict mode false-aborted on a healthy non-proxy cache\n{_report(r)}"
    assert _fields(r)["RESULT"] == "n", _report(r)


@pytest.mark.parametrize("value,expected", [
    (None, False), ("", False), ("0", False), ("false", False), ("FALSE", False), ("no", False),
    ("off", False), ("OFF", False), (" 0 ", False),
    ("1", True), ("true", True), ("yes", True), ("on", True), ("2", True), ("anything", True),
])
def test_strict_cache_mode_reads_the_env_truthily(value, expected, monkeypatch):
    """``_strict_cache_mode`` is off for an unset or empty value and for ``0``, ``false``, ``no`` and ``off``.

    Any other value turns it on, and letter case and surrounding white space are ignored. The off values are
    documented, and a user relies on them to disable strict mode for one run with ``=0``. If a change dropped one of
    them, ``export NUMBOX_PROXY_CACHE_STRICT=0`` could enable strict mode, and stale entries would raise errors instead
    of being repaired. The rest of the suite sets the variable only to ``1``, so this test checks each off value.
    """
    from numbox.core.configurations import _strict_cache_mode

    if value is None:
        monkeypatch.delenv("NUMBOX_PROXY_CACHE_STRICT", raising=False)
    else:
        monkeypatch.setenv("NUMBOX_PROXY_CACHE_STRICT", value)
    assert _strict_cache_mode() is expected


def test_an_unrecognised_cache_payload_is_handed_on_untouched(tmp_path):
    """A payload whose shape the check cannot unpack loads unchecked.

    The shape of the payload is what a numba release is most likely to change, so a shape that the check cannot read
    must skip the check and not prevent the load. Every numba version in the supported range passes a tuple.
    """
    from numbox.core.proxy.proxy import _guarded_rebuild

    class _Impl:
        filename_base = "unreadable"

    def original(self, target_context, payload):
        return "loaded"

    rebuild = _guarded_rebuild(original, lambda payload: payload[0])
    for payload in ({"keyword": "style"}, object(), ()):
        assert rebuild(_Impl(), None, payload) == "loaded", f"{type(payload).__name__} payload broke the load"


def test_a_guvectorize_caller_heals_too(tmp_path):
    """numbox also repairs a cached ``@guvectorize`` caller of an edited proxied body.

    numba caches a gufunc's wrapper library through ``CodeLibraryCacheImpl``, whose payload differs from the payload of
    ``CompileResultCacheImpl``: an ``@njit`` entry holds the serialized library inside a compile result, and the
    gufunc's payload is the library itself. Only this test exercises the second implementation. The gufunc compiles at
    import, not at its first call, so the warning capture wraps the import.
    """
    _write_binding(tmp_path / _EDITED, "scale", "x * 2.0")
    (tmp_path / "gufunc_caller.py").write_text(textwrap.dedent('''
        """A cached gufunc calling the proxied body, in a file the edit never touches."""
        from numba import guvectorize
        from numba.core.types import float64

        from proxied_binding import scale


        @guvectorize([(float64[:], float64[:])], "(n)->(n)", cache=True, nopython=True)
        def gu_scale(x, out):
            for i in range(x.shape[0]):
                out[i] = scale(x[i]) + 1.0
    '''), encoding="utf-8")
    probe = tmp_path / "gufunc_probe.py"
    probe.write_text(textwrap.dedent('''
        import warnings

        import numpy as np

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from gufunc_caller import gu_scale
            result = gu_scale(np.array([5.0]))[0]
        print("RESULT_gu_scale", result, flush=True)
        print("WARNINGS", " | ".join(f"{w.category.__name__}: {w.message}".replace("\\n", " ") for w in caught))
        print("DONE ok", flush=True)
    '''), encoding="utf-8")

    env = _probe_env(tmp_path, "nbcache")
    assert _warm(probe, env)["RESULT_gu_scale"] == "11.0"

    _edit_body(tmp_path)
    fields = _fields(_run_probe(probe, env))
    assert fields["RESULT_gu_scale"] == "16.0", f"the gufunc served a stale body: {fields['RESULT_gu_scale']}"
    assert "StaleProxyCacheWarning" in fields["WARNINGS"], f"the discard was not announced: {fields['WARNINGS']}"
