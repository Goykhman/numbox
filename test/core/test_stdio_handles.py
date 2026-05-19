import pytest
from numba import njit

from numbox.core.bindings import stdout, stderr, stdin, fputs, fflush
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
    """stdout / stderr / stdin must resolve to three different FILE* values.

    Smoke-tests-by-address can miss a regression that transposes platform
    symbol names (e.g. macOS ``_DATA_SYMBOL_BY_NAME`` swapping
    ``__stdoutp`` and ``__stderrp``): each individual ``addr != 0`` check
    passes, but two handles point at the same FILE. This test catches
    that class of mistake without needing capfd.
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
    p = get_unicode_data_p("ok-err\n")
    _write_to_stderr(p)
    out, err = capfd.readouterr()
    assert err == "ok-err\n"


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level fputs+fflush to stdout() on Windows",
)
def test_stdout_fputs_roundtrip(capfd):
    """Verifies stdout() resolves to the actual standard-output FILE* by
    writing through it and checking ``capfd``'s ``out`` capture (not
    ``err``). If the Linux/Darwin data-symbol table ever transposed
    ``stdout`` and ``stderr``, the smoke tests above would still pass
    but this test would receive the bytes on ``err`` instead.
    """
    p = get_unicode_data_p("ok-out\n")
    _write_to_stdout(p)
    out, err = capfd.readouterr()
    assert out == "ok-out\n"
    assert err == "", f"unexpected bytes captured on stderr: {err!r}"
