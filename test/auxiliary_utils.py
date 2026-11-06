"""Helpers that the test modules share.

``collect_and_run_tests`` runs the tests of a test module that is run as a script, without pytest. ``open_libm`` opens
the C math library for the tests that need a ``ctypes`` handle. ``deref_int64_intp``, with its intrinsic
``_deref_int64_intp``, reads an ``int64`` at an address, and ``str_from_p_as_int`` reads a NUL-terminated string.
``assert_njit_cache_survives_subprocess_roundtrip`` checks that ``cache=True`` functions load from numba's cache in a
second process. pytest does not collect this file in a plain run, because its name does not match pytest's default
``python_files`` patterns; its two tests run when the file is named on the command line.
"""
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


# The pattern matches an ANSI escape sequence; it comes from https://stackoverflow.com/a/14693789.
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def collect_and_run_tests(module_name):
    """Run the tests of the module named ``module_name``, for a test module that is run as a script.

    The function calls the module-level functions named ``test_*`` and the static methods named ``test_*`` of the
    classes named ``Test*``, in definition order, and it logs each name. It calls each test without arguments, so a
    test that takes a pytest fixture raises ``TypeError``. The first test that raises stops the run.
    """
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
    """Return a new handle of the platform's C math library, or None if no library here exports ceil, cos and floor.

    The callers bind ``ceil``, ``cos`` and ``floor``, and they skip their tests when the function returns None, so a
    platform without a usable C runtime skips those tests instead of failing them. The function checks all three
    symbols. A check of fewer symbols would accept a library that lacks one of them, and a caller that binds the
    missing symbol would then fail instead of skipping. Each call opens a new handle, so attributes such as ``floor``
    taken from two handles are two distinct objects.
    """
    import ctypes
    from ctypes.util import find_library

    from numbox.core.bindings.utils import load_lib_path, platform_

    def _with_the_symbols(lib):
        return lib if all(hasattr(lib, name) for name in ("ceil", "cos", "floor")) else None

    if platform_ == "Windows":
        # find_msvcrt() returns None on every CPython built with Visual Studio 2015 or later, so the function tries
        # two names in turn. msvcrt is the legacy C runtime that numbox.core.bindings.utils.load_lib also falls back
        # to, and ucrtbase is the Universal C Runtime that CPython itself links. ceil, cos and floor are C89
        # functions, so msvcrt.dll exports them wherever it exists, and ucrtbase covers a Windows image without
        # msvcrt.dll. The loop moves to the second name when the first cannot be loaded or lacks a symbol, so that the
        # tests skip only when neither runtime has the symbols.
        for name in ("msvcrt", "ucrtbase"):
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            if _with_the_symbols(lib) is not None:
                return lib
        return None
    # find_library("m") returns a soname, such as libm.so.6, on Linux and a path into the dyld shared cache on macOS;
    # load_lib_path opens either.
    path = find_library("m")
    return _with_the_symbols(load_lib_path(path)) if path is not None else None


@intrinsic
def _deref_int64_intp(typingctx, p_int_ty):
    """Return the ``int64`` at the address that the ``intp`` argument holds, in jitted code."""
    sig = types.int64(types.intp)

    def codegen(context, builder, signature, args):
        p_ty_ll = context.get_value_type(p_int_ty).as_pointer()
        ptr = builder.inttoptr(args[0], p_ty_ll)
        return builder.load(ptr)
    return sig, codegen


@njit
def deref_int64_intp(p_int):
    """Return the ``int64`` at the address ``p_int``, from Python or from jitted code."""
    return _deref_int64_intp(p_int)


def test_deref_int64_intp():
    v1 = c_int64(137)
    v1_p = addressof(v1)
    assert deref_int64_intp(v1_p) == 137


def str_from_p_as_int(p_as_int):
    """Return the NUL-terminated string at the address ``p_as_int``, decoded as UTF-8.

    The function reads the bytes one at a time with ``ctypes``, without numba. The caller keeps the memory alive
    during the call.
    """
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
    """Assert that the ``cache=True`` functions of ``probe_source`` load from numba's cache in a second process.

    The function writes ``probe_source``, dedented, to ``probe.py`` under ``tmp_path`` and runs it twice in a
    subprocess, with ``NUMBA_CACHE_DIR`` set to ``tmp_path/numba-cache``. The first run compiles the functions and
    writes the cache, and the second run must load it. The function asserts that each run exits with status 0 and
    prints ``expected_stdout_lines``, and that the first run writes at least one ``.nbc`` file. It also asserts that
    the second run leaves the set of ``.nbc`` and ``.nbi`` files and their modification times unchanged, so a second
    run that compiled again fails the check.

    The probe imports what it uses and prints each value that the caller asserts, because the function reads only
    the probe's standard output. The comparison splits the output into lines, so ``\\n`` and ``\\r\\n`` line endings
    compare equal. A cached caller of numbox's proxied bindings or of its formatted I/O functions keeps working when a
    later process loads it, and the tests check that property with this helper.
    """
    import pathlib
    # The probe is a file because numba refuses to cache a function whose source file is "<string>", the file name
    # that Python gives to code run with python -c. numba then raises
    # "RuntimeError: cannot cache function ...: no locator available".
    probe = pathlib.Path(tmp_path) / "probe.py"
    # Without the encoding argument, Path.write_text uses the locale's preferred encoding, which is cp1252 on Windows
    # runners and raises UnicodeEncodeError for a character in the probe that cp1252 lacks, such as the arrow U+2192.
    probe.write_text(textwrap.dedent(probe_source), encoding="utf-8")
    cache_dir = pathlib.Path(tmp_path) / "numba-cache"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir)}

    def _run(label):
        # PYTHONIOENCODING makes the subprocess write its stdout and stderr in UTF-8 instead of Windows' cp1252, and
        # encoding="utf-8" makes subprocess.run decode the captured bytes as UTF-8 instead of in the locale of the
        # parent process, which is cp1252 on Windows runners.
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

    # The check covers numba's index files (.nbi) as well as its data files (.nbc). numba writes neither on a cache
    # hit, so a numba release that rewrote the index on a hit would also fail the check.
    cache_after_cold = sorted(cache_dir.rglob("*.nb[ci]"))
    assert any(p.suffix == ".nbc" for p in cache_after_cold), (
        f"cold run did not write any .nbc cache file under {cache_dir}"
    )
    mtimes_cold = {p: p.stat().st_mtime_ns for p in cache_after_cold}

    _run("warm")

    # The set of paths detects a file that the second run added or removed, such as one for a new specialization.
    # The modification times detect a file that it rewrote in place, as numba does when a cache miss compiles the
    # same signature again.
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
