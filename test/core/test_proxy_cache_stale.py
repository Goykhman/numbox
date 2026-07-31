"""Cross-process regression tests for a stale ``@proxy`` cfunc alias in a warm numba cache.

``@proxy`` publishes each proxied body's cfunc wrapper under a process-stable alias whose name folds a fingerprint of
that body, so editing the body renames the alias. A ``cache=True`` caller living in another file keys only on its own
source -- numba's key says nothing about its callees -- so after such an edit the caller still cache-hits and loads an
object referencing a symbol this process never registered. RuntimeDyld resolves an object's externals in one batch, so
that single missing name zeroes *every* external relocation in the object, and the process dies inside the CPython
argument-unpacking wrapper before any user code runs: a bare segfault with an empty stderr. The engine keeps that
failure, so the next cached object loaded afterwards -- a sound one will do -- aborts the process instead, naming the
alias that no longer exists.
Both shapes are landmines for anyone who edits a proxied binding without first clearing the cache.

The target behaviour is that the loader notices the unregistered alias, discards that one entry, warns, and recompiles.
Each scenario therefore also carries a second binding that is never edited, with its own caller in the same file and
the same cache directory; that caller must still be served from cache and draw no warning, so a fix that switches
caching off or invalidates the whole directory cannot pass. The premise -- warm entries really are served, and the edit
re-keys one binding and only one -- is pinned separately, by tests that never go near the crash.
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
    """Write a one-function proxied binding module.

    A rewrite bumps mtime forward by hand rather than trusting the clock: the new body is the same length as the old
    one, and both numba's source stamp and CPython's pyc validation compare ``(st_mtime, st_size)``, so where mtime
    granularity is coarse a same-second rewrite is indistinguishable from the original and the edit would never be
    seen. The padding above the decorator is deliberate too -- the cache anchor requires the decorator line to be the
    function's ``co_firstlineno``, and it may not sit above the generated wrapper's own decorator line.
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
    """Lay the scenario out on disk and hand back the probe script.

    The two bindings get separate files so the control is untouched at file level as well: nothing about it, alias or
    source stamp, can move when the other body is edited. The three callers share one file to make the control as
    tight as possible: same module, same cache directory, same run. Two callers of the edited body rather than one is
    what pins that *every* stale entry is discarded, not just the first the loader reaches, and ``plain`` -- cached,
    but referencing no alias at all -- is the control for the other direction: it takes the validator's early exit,
    and must survive untouched.
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
    """Rewrite the edited binding's body, which is the whole trigger: same file, same length, different alias."""
    _write_binding(tmp_path / _EDITED, "scale", "x * 3.0")


def _probe_env(tmp_path, cache_name, precompile=False):
    """Environment for one probe run.

    ``NUMBA_CACHE_DIR`` also relocates numbox's own anchor files, so a per-test directory keeps the whole scenario out
    of the developer's shared cache. The parent's ``sys.path`` is forwarded because the probe imports numbox from
    wherever this test found it, and ``tmp_path`` is named explicitly rather than relying on the script directory,
    which ``PYTHONSAFEPATH`` suppresses. Bytecode writing is off because a pyc validated on ``(st_mtime, st_size)``
    is the other way a same-length edit can go unnoticed. The inherited warning and numba knobs are dropped because
    each of them can empty the scenario out: they decide whether an unrelated warning ends the probe, whether
    anything is compiled at all, and -- at ``NUMBA_OPT=0``, where the proxy call is not folded to a plain extern
    reference -- whether numba is willing to cache the callers in the first place. ``NUMBOX_PROXY_CACHE_STRICT``
    is dropped for the same reason: an exported strict knob would turn every heal scenario below into a hard
    error, so a strict run sets it back on explicitly rather than inheriting it.
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
    """Run a probe to completion and hand back the whole result: these scenarios expect the child to die, so the
    caller, not this helper, decides what an acceptable return code is. The timeout is the only defence against a
    crashed child that never exits -- a fault can park a process behind a crash reporter."""
    return subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=300,
    )


def _report(r):
    """Full transcript, so a failure shows which crash shape happened -- or that neither did."""
    return f"rc={r.returncode}\n--- stdout ---\n{r.stdout}--- stderr ---\n{r.stderr}"


def _fields(r):
    """The probe's ``KEY value`` lines. ``DONE`` is absent whenever the child died mid-run."""
    fields = dict(line.split(" ", 1) for line in r.stdout.splitlines() if " " in line)
    assert fields.get("DONE") == "ok", f"the probe did not run to completion\n{_report(r)}"
    return fields


def _warm(probe, env):
    """Warm the cache and return the probe's fields.

    A failure here is a broken scenario rather than the behaviour under test, so it is raised as an error: an
    assertion would be reported as a behavioural failure, and a dead harness must not be mistaken for one.
    """
    r = _run_probe(probe, env)
    if r.returncode != 0 or "DONE ok" not in r.stdout:
        raise RuntimeError(f"warming the cache failed\n{_report(r)}")
    return _fields(r)


def _stats(fields):
    return {name: fields[f"STATS_{name}"] for name in _HEALED}


def _results(fields):
    return {name: fields[f"RESULT_{name}"] for name in _HEALED}


def test_unedited_rerun_is_served_from_cache(tmp_path):
    """Nothing below proves anything unless a warm entry really is served: a caller that recompiled anyway could
    never be handed a stale object. One cold run, then an identical one against the same cache directory."""
    probe = _write_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")

    cold = _warm(probe, env)
    assert _stats(cold) == dict.fromkeys(_HEALED, "compiled"), _stats(cold)
    assert _results(cold) == _OLD_BODY, _results(cold)

    warm = _warm(probe, env)
    assert _stats(warm) == dict.fromkeys(_HEALED, "served"), f"an unchanged re-run recompiled: {_stats(warm)}"
    assert _results(warm) == _results(cold), _results(warm)


def test_editing_a_proxied_body_renames_only_that_binding_alias(tmp_path):
    """The edit re-keys exactly one binding, which is what makes one caller stale and leaves the control valid.

    The edited body is run against a *fresh* cache directory so this stays clear of the crash: it answers what the new
    body is worth, and which aliases moved, without a stale entry in the way.
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
    """Reaching a warm caller after the proxied body was edited must recompile it, not crash.

    Unfixed, calling it segfaults on the first call: the one missing alias zeroed every external relocation in the
    loaded object, so control never reaches the compiled body -- stdout stops dead after the callers are imported and
    stderr is empty, which is as undiagnosable as a failure gets. Compiling the callers explicitly instead moves the
    failure: the stale load itself is silent, because nothing has jumped into the object yet, and the next cached load
    of any object aborts the process outright naming the retired alias, because the execution engine keeps the first
    failure.

    Fixed, both callers of the edited body are discarded and recompiled against it while the caller of the untouched
    binding is still served, and the discard is announced by a warning that names the alias nobody registered. The
    last run pins the heal as a one-time recompile: an implementation that refused to serve those entries without
    replacing them would leave every later process paying full compilation again.
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


def _rewrite(path, old, new):
    """Replace a body expression and bump the source stamp, as ``_write_binding`` does, so the edit is seen."""
    before = path.stat()
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    os.utime(path, (before.st_mtime + 10,) * 2)


def _write_two_proxy_scenario(tmp_path):
    """One cached caller of two *different* proxied bodies in separate files, so its object imports two aliases.

    Each binding sits below the cache anchor's minimum line. Editing both bodies renames both aliases, so the
    single caller object references two names this process never registered -- the case that would undercount
    if the warning named only the first stale alias it found rather than all of them.
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
    """One cached object that imports two stale aliases must name *both* in its discard warning.

    The correctness review saw a diagnostic that could undercount: an object referencing several retired
    aliases getting a warning that named only some. The guard returns the whole set and the caller joins it,
    so a single warning for the one stale object here has to carry both retired names -- an implementation
    that stopped at the first stale symbol would name one and pass this by. Nothing numeric about compilation
    is asserted; the retired aliases are captured from the warm run's own output.
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
    """Healing silently is the default, not the only option: a caller who would rather know can escalate.

    The warning is a named class reachable from the public module path, so a filter can single it out without
    catching every ``RuntimeWarning``. Escalation aborts before the heal, which leaves the stale entry on disk for
    inspection rather than replacing it.
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


def test_a_binding_that_disappeared_is_discarded_rather_than_called(tmp_path):
    """A ``proxy_if_available`` binding that is gone leaves its alias resolvable, and that is the dangerous case.

    The absent path registers a trap under the alias a warm caller baked in, so the symbol resolves and the object
    looks loadable. Calling it is the worst available outcome: the trap raises inside a ``@cfunc``, numba swallows
    that at the C boundary and returns zero, and the caller computes on the zero and exits successfully. Discarding
    the entry instead reaches the same clean typing error a cold cache gives, which is what this pins.
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


def test_the_mach_o_reader_strips_the_leading_underscore(tmp_path):
    """The scenarios above cannot reach the Mach-O reader anywhere but macOS, so it is driven directly against
    a real Mach-O object emitted by the same LLVM numba compiles with.

    What this pins is the underscore Mach-O puts in front of every C symbol. A reader that forgets to strip it
    compares ``_numbox_pxy_...`` against the prefix, matches nothing, and passes every stale object -- which is
    indistinguishable from having nothing to do, on the one platform where nobody can watch it fail.
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
    """Whatever container numba caches on this platform, the reader must be able to parse it.

    Every other test here reaches the reader only through a stale entry, so each one proves the guard works by
    watching a crash *not* happen. That leaves the guard's most basic premise unpinned: that the objects numba
    hands it are a format it reads at all. A reader that recognises nothing is silent and total -- it returns no
    symbols, finds no stale alias, and passes every object exactly as if the guard were absent -- and the only
    evidence is a segfault on whichever platform stopped matching, with nothing naming the cause.

    Which branch matches is not obvious from the platform: the Windows jobs are served by the ELF reader, because
    LLVM rewrites a COFF target to ELF for the JIT, so ``get_object_format`` naming COFF says nothing about the
    bytes. This asserts the outcome rather than that reasoning -- a warm re-run, with the reader recording what it
    was handed -- so the day it stops holding is a named failure here rather than a crash somewhere else. It needs
    no stale entry and no edit, so it stands up even if the crash scenario itself ever stops reproducing.
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
    """``NUMBOX_PROXY_CACHE_STRICT`` turns the silent self-heal into a hard error that leaves the entry alone.

    The default is to discard the stale entry, warn, and recompile in place. A caller debugging why a cache
    misbehaves wants the opposite: stop at the first stale entry, name it, and leave it on disk to inspect.
    The strict run must therefore die with ``StaleProxyCacheError`` named on stderr rather than exit clean.
    That it did *not* recompile is then proved by a second run with the knob off, which still finds the same
    entry stale and heals it -- had strict recompiled, this run would serve the new body silently with no
    warning at all.
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
    """Strict mode must not cry wolf. On an unedited warm cache every proxy caller imports a resolvable alias,
    so the prefix-present-but-no-import check can never fire and no error is raised: the run serves the cached
    bodies exactly as the default would, and neither strict error name appears on stderr."""
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
    """The abstention path -- a payload whose shape the reader cannot unpack -- fails open by default and
    raises under the knob. This is the counterpart of ``test_an_unrecognised_cache_payload_is_handed_on_untouched``:
    the same payloads that are handed on untouched with the knob off must abort with the named strict error on."""
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
    """A cleanly-read object that carries the prefix but imports no matching alias is loaded, strict or not.

    Such an object cannot be told apart from a healthy, entirely non-proxy function that merely embeds an
    alias-shaped string constant (see ``test_..._embedding_the_prefix`` below), so strict mode does not
    escalate on it -- only a payload that *raises* when read is a validation gap. Reader blindness on a real
    numba object is caught in CI instead, by ``test_the_reader_recognises_the_objects_numba_actually_caches``.
    """
    from numbox.core.proxy.proxy import _guarded_rebuild

    class _Impl:
        filename_base = "prefix-without-import"

    def original(self, target_context, payload):
        return "loaded"

    # Carries the alias prefix so the fast path does not short-circuit, but as data in a blob that is neither
    # ELF nor Mach-O; the reader parses it cleanly and finds no imported alias.
    blob = b"numbox_pxy_scale_deadbeefdeadbeef" + b"\x00" * 64
    payload = ("libname", "object", (blob,))
    rebuild = _guarded_rebuild(original, lambda p: p)

    assert rebuild(_Impl(), None, payload) == "loaded", "the default must load such an object untouched"
    monkeypatch.setenv("NUMBOX_PROXY_CACHE_STRICT", "1")
    assert rebuild(_Impl(), None, payload) == "loaded", "strict must not abort on a cleanly-read prefix-bearing object"


def test_strict_mode_does_not_abort_on_a_healthy_object_embedding_the_prefix(tmp_path):
    """The above, end to end: a real cached function with no proxy involvement, carrying the alias prefix as a
    runtime string constant, must load under strict mode -- not false-abort.

    The guard only runs where numbox is imported, so the runner imports it; strict mode is exactly what a
    person debugging the proxy cache turns on, and such a person is the most likely to have jitted code that
    handles ``numbox_pxy_`` strings. The literal is returned so it materializes into the object rather than
    being folded away.
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
    """The knob's off values are a documented contract a user relies on to disable it per run (``=0``). Pin each
    one, so dropping an element of the falsey set -- which would turn ``export NUMBOX_PROXY_CACHE_STRICT=0`` into
    hard cache-load errors -- fails here rather than passing a suite that only ever sets the knob to ``1``."""
    from numbox.core.configurations import _strict_cache_mode

    if value is None:
        monkeypatch.delenv("NUMBOX_PROXY_CACHE_STRICT", raising=False)
    else:
        monkeypatch.setenv("NUMBOX_PROXY_CACHE_STRICT", value)
    assert _strict_cache_mode() is expected


def test_an_unrecognised_cache_payload_is_handed_on_untouched(tmp_path):
    """The payload's shape is the assumption most likely to change under this guard, so a shape it cannot read
    must cost only the validation -- never the load. Every numba in the supported range passes a tuple here."""
    from numbox.core.proxy.proxy import _guarded_rebuild

    class _Impl:
        filename_base = "unreadable"

    def original(self, target_context, payload):
        return "loaded"

    rebuild = _guarded_rebuild(original, lambda payload: payload[0])
    for payload in ({"keyword": "style"}, object(), ()):
        assert rebuild(_Impl(), None, payload) == "loaded", f"{type(payload).__name__} payload broke the load"


def test_a_guvectorize_caller_heals_too(tmp_path):
    """A gufunc reaches the cache through a second implementation, whose payload is shaped differently.

    ``@njit`` stores the serialized library inside a compile result; a gufunc's wrapper library is the payload, so
    the two are unwrapped differently and only this exercises the second one. It also compiles eagerly at import
    rather than on first call, which is why the warning capture has to wrap the import.
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
