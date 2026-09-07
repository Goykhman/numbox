from inspect import isclass, isfunction
import logging
import os
import re
import subprocess
import sys
import textwrap
from ctypes import addressof, c_char, c_char_p, c_int64, c_void_p
from io import BytesIO

from numba import njit
from numba.core import types
from numba.extending import intrinsic


# https://stackoverflow.com/a/14693789
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def collect_and_run_tests(module_name):
    module = sys.modules[module_name]
    for name, item in module.__dict__.items():
        if isfunction(item) and name.startswith("test_"):
            logger.info(f" Running {name}")
            item()
        elif isclass(item) and name.startswith("Test"):
            for attr, val in item.__dict__.items():
                if isinstance(val, staticmethod) and attr.startswith("test_"):
                    logger.info(f" Running {val.__qualname__}")
                    val()


def open_libm():
    """Open the platform's math library and return a fresh handle, or ``None``.

    The handle answers to ``ceil``, ``cos`` and ``floor``, every symbol a caller needs
    to be present, or the result is ``None`` and there is nothing on this machine to run
    those tests against. Returning ``None`` rather than raising is the point: a platform
    with no usable C runtime should skip, the way one where ``find_library`` comes up
    empty already does, not turn into a failure that reads like a defect in the code
    under test. Gating on a subset gives that away, because a library holding only the
    subset is admitted and the callers reaching past it fail instead of skipping.

    ``find_library("m")`` resolves it on Linux (a soname) and on macOS (a path into
    the dyld shared cache). On Windows the C runtime is not loadable by name --
    ``find_msvcrt()`` returns ``None`` for every VS2015+ build -- so try ``msvcrt``,
    the legacy CRT shim :func:`numbox.core.bindings.utils.load_lib` already falls back
    to, and then ``ucrtbase``, the CRT the interpreter itself links. All three are C89,
    so the shim carries them on any image that has the shim at all; ``ucrtbase`` is
    there for an image that does not. Falling through rather than giving up on the
    first candidate keeps the coverage a skip would give away.

    Each call opens its own handle rather than sharing a cached one, so a caller
    comparing two handles is comparing two.
    """
    import ctypes
    from ctypes.util import find_library

    from numbox.core.bindings.utils import load_lib_path, platform_

    def _with_the_symbols(lib):
        return lib if all(hasattr(lib, name) for name in ("ceil", "cos", "floor")) else None

    if platform_ == "Windows":
        for name in ("msvcrt", "ucrtbase"):
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            if _with_the_symbols(lib) is not None:
                return lib
        return None
    path = find_library("m")
    return _with_the_symbols(load_lib_path(path)) if path is not None else None


@intrinsic
def _deref_int64_intp(typingctx, p_int_ty):
    sig = types.int64(types.intp)

    def codegen(context, builder, signature, args):
        p_ty_ll = context.get_value_type(p_int_ty).as_pointer()
        ptr = builder.inttoptr(args[0], p_ty_ll)
        return builder.load(ptr)
    return sig, codegen


@njit
def deref_int64_intp(p_int):
    return _deref_int64_intp(p_int)


def test_deref_int64_intp():
    v1 = c_int64(137)
    v1_p = addressof(v1)
    assert deref_int64_intp(v1_p) == 137


def str_from_p_as_int(p_as_int):
    b = BytesIO()
    while True:
        c = c_char.from_address(p_as_int).value
        if c == b"\x00":
            break
        b.write(c)
        p_as_int += 1
    return b.getvalue().decode("utf-8")


def test_str_from_p_as_int():
    s1_ = "a random string"
    s1_b = s1_.encode("utf-8")
    s1 = c_char_p(s1_b)
    s1_p = c_void_p.from_buffer(s1).value
    assert str_from_p_as_int(s1_p) == s1_


def assert_njit_cache_survives_subprocess_roundtrip(
        tmp_path, probe_source, expected_stdout_lines,
):
    """Verify ``@njit(cache=True)`` callers in *probe_source* survive a
    process restart (the load-bearing property of ``@proxy``-wrapped
    bindings and of the fmtio variadic-intrinsic dispatch).

    Mechanics:

    - Writes *probe_source* to a fresh ``.py`` file under ``tmp_path``.
      (``@njit(cache=True)`` refuses to cache functions whose
      ``co_filename`` is ``<string>``, so the probe has to live in a real
      file rather than be passed via ``python -c``.)
    - Runs the probe twice in subprocess, sharing
      ``NUMBA_CACHE_DIR = tmp_path/numba-cache``. The cold run populates
      the cache; the warm run must reuse it.
    - Asserts each run's stdout splits to *expected_stdout_lines*.
    - Asserts the cache state is byte-identical between the two runs:
      * the set of ``*.nbc`` + ``*.nbi`` file paths is unchanged
        (catches added / removed files — e.g. a new specialization or
        a sweep that deleted something)
      * each file's mtime is unchanged (catches in-place rewrites,
        which would happen on a cache miss that recompiles to the
        same signature)

    Globbing both ``.nbc`` and ``.nbi`` is strictly tighter than
    ``.nbc`` alone: empirical inspection on the current numba shows
    neither is touched on cache hit, but checking both pins against a
    future numba release that might write ``.nbi`` metadata on hit.

    *probe_source* is passed through ``textwrap.dedent`` so callers can
    use triple-quoted strings with leading indentation. The probe must
    import its own dependencies and ``print()`` whatever it wants
    asserted; no other communication channel.

    *expected_stdout_lines* is compared via ``splitlines()`` so platform
    line-endings (``\\n`` vs ``\\r\\n``) don't matter.
    """
    import pathlib  # local — avoid a top-level import in the test utils
    probe = pathlib.Path(tmp_path) / "probe.py"
    # Write the probe explicitly as UTF-8. Without encoding= Path.write_text
    # uses locale.getpreferredencoding() which is cp1252 on Windows runners
    # and would raise UnicodeEncodeError on any non-Latin-1 char in the
    # probe source (arrows, em-dashes, accented chars in test data, ...).
    probe.write_text(textwrap.dedent(probe_source), encoding="utf-8")
    cache_dir = pathlib.Path(tmp_path) / "numba-cache"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir)}

    def _run(label):
        # Force UTF-8 decoding on both ends: PYTHONIOENCODING so the
        # subprocess uses UTF-8 for stdout/stderr (overrides Windows
        # cp1252 default), and encoding="utf-8" so capture_output decodes
        # the captured bytes as UTF-8 (otherwise text=True would use the
        # parent's locale, which is cp1252 on Windows runners).
        r = subprocess.run(
            [sys.executable, str(probe)],
            env={**env, "PYTHONIOENCODING": "utf-8"},
            capture_output=True, text=True, encoding="utf-8",
        )
        assert r.returncode == 0, (
            f"{label} run failed (rc={r.returncode}):\n"
            f"--- stdout ---\n{r.stdout}\n"
            f"--- stderr ---\n{r.stderr}"
        )
        got = r.stdout.splitlines()
        assert got == list(expected_stdout_lines), (
            f"{label} run stdout mismatch:\n"
            f"  expected: {expected_stdout_lines!r}\n"
            f"  got:      {got!r}\n"
            f"--- stderr ---\n{r.stderr}"
        )

    _run("cold")

    cache_after_cold = sorted(cache_dir.rglob("*.nb[ci]"))
    assert any(p.suffix == ".nbc" for p in cache_after_cold), (
        f"cold run did not write any .nbc cache file under {cache_dir}"
    )
    mtimes_cold = {p: p.stat().st_mtime_ns for p in cache_after_cold}

    _run("warm")

    cache_after_warm = sorted(cache_dir.rglob("*.nb[ci]"))
    assert cache_after_warm == cache_after_cold, (
        f"warm run added or removed cache files; "
        f"cold={len(cache_after_cold)} warm={len(cache_after_warm)}; "
        f"diff added={set(cache_after_warm) - set(cache_after_cold)} "
        f"removed={set(cache_after_cold) - set(cache_after_warm)}"
    )
    for p in cache_after_warm:
        assert p.stat().st_mtime_ns == mtimes_cold[p], (
            f"warm run rewrote cache file {p} — cache was not hit"
        )
