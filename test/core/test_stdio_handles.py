"""Tests of stdout, stderr and stdin in numbox.core.bindings.stdio, called from jitted code.

The tests check that each function returns a nonzero ``FILE *``, that the three values differ, and that text written
with ``fputs`` through ``stdout()`` or ``stderr()`` reaches the matching file descriptor, which pytest's ``capfd``
fixture captures. The two ``capfd`` tests skip on Windows. ``test_fmtio.py`` tests the formatted output functions,
which also write through these streams.
"""
import pytest
from numba import njit

from numbox.core.bindings.stdio import stdout, stderr, stdin
from numbox.core.bindings.libc import fputs, fflush
from numbox.core.bindings.utils import platform_
from numbox.utils.lowlevel import get_unicode_data_p


@njit(cache=True)
def _write_to_stderr(text_p):
    fputs(text_p, stderr())
    fflush(stderr())


@njit(cache=True)
def _write_to_stdout(text_p):
    fputs(text_p, stdout())
    fflush(stdout())


def test_stdout_handle_nonzero():
    @njit(cache=True)
    def get():
        return stdout()
    assert get() != 0


def test_stderr_handle_nonzero():
    @njit(cache=True)
    def get():
        return stderr()
    assert get() != 0


def test_stdin_handle_nonzero():
    @njit(cache=True)
    def get():
        return stdin()
    assert get() != 0


def test_stdio_handles_are_distinct():
    """Check that ``stdout()``, ``stderr()`` and ``stdin()`` return three different nonzero ``FILE *`` values.

    If the stdio module mapped two streams to one symbol, such as ``__stdoutp`` on macOS, each value would still be
    nonzero, but two of them would be equal. The test detects that without ``capfd``, so it also runs on Windows. A
    table that swapped two names would give three distinct values; the two ``fputs`` tests below detect that case.
    """
    @njit(cache=True)
    def get_all():
        return stdout(), stderr(), stdin()
    out, err, inp = get_all()
    assert out != 0 and err != 0 and inp != 0
    assert out != err, f"stdout == stderr ({out:#x})"
    assert out != inp, f"stdout == stdin ({out:#x})"
    assert err != inp, f"stderr == stdin ({err:#x})"


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level fputs+fflush to stderr() on Windows",
)
def test_stderr_fputs_roundtrip(capfd):
    """Check that text written with ``fputs`` through ``stderr()`` reaches standard error."""
    p = get_unicode_data_p("ok-err\n")
    _write_to_stderr(p)
    out, err = capfd.readouterr()
    assert err == "ok-err\n"


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level fputs+fflush to stdout() on Windows",
)
def test_stdout_fputs_roundtrip(capfd):
    """Check that text written with ``fputs`` through ``stdout()`` reaches standard output and not standard error.

    If the stdio module swapped the symbols of ``stdout`` and ``stderr``, the nonzero and distinct checks above would
    still pass, but this test would find the text in standard error.
    """
    p = get_unicode_data_p("ok-out\n")
    _write_to_stdout(p)
    out, err = capfd.readouterr()
    assert out == "ok-out\n"
    assert err == "", f"unexpected bytes captured on stderr: {err!r}"
