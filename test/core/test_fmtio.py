"""Tests of printf, fprintf, snprintf and sscanf in numbox.core.bindings.fmtio, under Python and in jitted code.

The tests read C's output through pytest's ``capfd`` fixture or from the buffer that ``snprintf`` fills. They check
how numbox passes each value to C's variadic call: a ``float32`` as a double, and a narrow integer or a ``bool`` as
64 bits. Other tests cover ``%s`` with a ``str`` or an address, the truncation rule of ``snprintf`` on POSIX and on
Windows, and UTF-8 text in the format. Further tests cover the checks that refuse a call when numba compiles it,
``sscanf``, the same source run in both modes, and cached callers in a second process. The tests that read C's
output through ``capfd`` skip on Windows. ``test_stdio_handles.py`` tests the standard streams themselves.
"""
import numpy as np
import pytest
from numba import njit
from numba.core.errors import TypingError

from numbox.core.bindings.libc import fflush
from numbox.core.bindings.fmtio import fprintf, printf, snprintf, sscanf
from numbox.core.bindings.stdio import stderr, stdout
from numbox.core.bindings.utils import platform_
from numbox.utils.lowlevel import array_data_p, get_unicode_data_p
from test.auxiliary_utils import assert_njit_cache_survives_subprocess_roundtrip


# C buffers its stdout when the stream goes to a file or a pipe, as it does under pytest's capfd, so each helper
# flushes the stream after printf, and capfd captures the text before the test reads it. C writes stderr without a
# buffer on Linux, but on Windows it buffers a stderr that goes to a file or a pipe, so the helpers flush stderr as
# well.


@njit(cache=True)
def _printf_int(n):
    rc = printf("got %d\n", n)
    fflush(stdout())
    return rc


@njit(cache=True)
def _printf_no_args():
    rc = printf("just literal\n")
    fflush(stdout())
    return rc


@njit(cache=True)
def _printf_float64(x):
    rc = printf("%.3f\n", x)
    fflush(stdout())
    return rc


@njit(cache=True)
def _printf_float32(x):
    # %f reads a double, so _promote_for_varargs extends the float32 to a float64.
    rc = printf("%.3f\n", np.float32(x))
    fflush(stdout())
    return rc


@njit(cache=True)
def _printf_int8():
    rc = printf("[%d %d]\n", np.int8(-7), np.int8(42))
    fflush(stdout())
    return rc


@njit(cache=True)
def _printf_bool():
    # _promote_for_varargs zero-extends each bool to 64 bits; test_printf_bool_is_promoted_to_int32 explains why.
    rc = printf("[%d %d]\n", True, False)
    fflush(stdout())
    return rc


@njit(cache=True)
def _printf_string_via_pointer(s_p):
    """Print the string at the address ``s_p``, an ``intp`` that the caller passes instead of a ``str``, with ``%s``."""
    rc = printf("hi %s!\n", s_p)
    fflush(stdout())
    return rc


@njit(cache=True)
def _printf_string_native(s):
    """Print the ``str`` ``s`` with ``%s``; the overload passes C the address of its characters."""
    rc = printf("hi %s!\n", s)
    fflush(stdout())
    return rc


@njit(cache=True)
def _fprintf_stderr(n):
    rc = fprintf(stderr(), "err %d\n", n)
    fflush(stderr())
    return rc


@njit(cache=True)
def _fprintf_stdout(n):
    rc = fprintf(stdout(), "out %d\n", n)
    fflush(stdout())
    return rc


@njit(cache=True)
def _snprintf_into(buf, lo, hi):
    return snprintf(array_data_p(buf), buf.size, "[%d:%d]", lo, hi)


@njit(cache=True)
def _snprintf_no_args(buf):
    return snprintf(array_data_p(buf), buf.size, "literal")


# NON_ASCII_FMT holds U+00E9, the letter e with an acute accent, which UTF-8 encodes as the two bytes b"\xc3\xa9", so
# the text for n=42 is the 9 bytes b"caf\xc3\xa9=42\n". numbox encodes the format as UTF-8 when numba compiles the
# call. An ASCII encoding would raise UnicodeEncodeError there, so the helpers below compile only because the encoding
# is UTF-8.
NON_ASCII_FMT = "café=%d\n"
NON_ASCII_EXPECTED = "café=42\n".encode("utf-8")  # b'caf\xc3\xa9=42\n'


@njit(cache=True)
def _printf_utf8(n):
    rc = printf(NON_ASCII_FMT, n)
    fflush(stdout())
    return rc


@njit(cache=True)
def _fprintf_utf8(n):
    rc = fprintf(stdout(), NON_ASCII_FMT, n)
    fflush(stdout())
    return rc


@njit(cache=True)
def _snprintf_utf8(buf, n):
    return snprintf(array_data_p(buf), buf.size, NON_ASCII_FMT, n)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_int_roundtrip(capfd):
    rc = _printf_int(42)
    out, err = capfd.readouterr()
    assert out == "got 42\n", repr(out)
    assert rc == len("got 42\n")
    assert err == ""


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_no_args(capfd):
    rc = _printf_no_args()
    out, _ = capfd.readouterr()
    assert out == "just literal\n"
    assert rc == len("just literal\n")


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_float64(capfd):
    _printf_float64(3.14159)
    out, _ = capfd.readouterr()
    assert out == "3.142\n", repr(out)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_float32_is_promoted_to_double(capfd):
    """Check that a ``float32`` given to ``%f`` reaches C as a double.

    C's default argument promotion converts a ``float`` to a ``double`` in a variadic call, and ``%f`` reads 8 bytes.
    If numbox passed the 4-byte value without its ``fpext`` instruction, C would print an unrelated number.
    """
    _printf_float32(3.14159)
    out, _ = capfd.readouterr()
    # np.float32(3.14159) is 3.1415901..., which %.3f rounds to "3.142". The check also accepts "3.141"; either text
    # shows that C received the number, because a misread value would print as an unrelated number.
    assert out.strip() in ("3.141", "3.142"), repr(out)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_int8_is_promoted_to_int32(capfd):
    """Check that ``int8`` values reach C intact for ``%d``.

    ``_promote_for_varargs`` sign-extends a signed integer narrower than 64 bits to 64 bits, which covers C's
    promotion to ``int``. If it passed the 8-bit value unextended, the upper bits that C reads for ``%d`` would be
    undefined, and C could print a number other than -7.
    """
    _printf_int8()
    out, _ = capfd.readouterr()
    assert out == "[-7 42]\n", repr(out)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_bool_is_promoted_to_int32(capfd):
    """Check that ``True`` and ``False`` print as 1 and 0 with ``%d``.

    numba represents a ``bool`` value as a 1-bit LLVM integer, and ``_promote_for_varargs`` zero-extends it to 64
    bits. numba's ``Boolean`` type is a subclass of neither ``Integer`` nor ``Number``, so the function handles it in
    a separate branch. Without the extension, the upper bits that C reads for ``%d`` would be undefined.
    """
    _printf_bool()
    out, _ = capfd.readouterr()
    assert out == "[1 0]\n", repr(out)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_string_via_pointer(capfd):
    """Check that ``%s`` prints the NUL-terminated string at an address that the caller passes as an ``intp``."""
    s_p = get_unicode_data_p("world")
    _printf_string_via_pointer(s_p)
    out, _ = capfd.readouterr()
    assert out == "hi world!\n", repr(out)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_string_native(capfd):
    """Check that ``%s`` prints a ``str`` that the caller passes without converting it to an address.

    The overload passes C the address of the string's characters through ``get_unicode_data_p``, so the caller does
    not call that function. The same call also runs under Python, where ``%`` formats the ``str``.
    """
    _printf_string_native("world")
    out, _ = capfd.readouterr()
    assert out == "hi world!\n", repr(out)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_fprintf_to_stderr(capfd):
    rc = _fprintf_stderr(13)
    out, err = capfd.readouterr()
    assert err == "err 13\n", repr(err)
    assert out == "", f"unexpected stdout: {out!r}"
    assert rc == len("err 13\n")


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_fprintf_to_stdout(capfd):
    """Check that ``fprintf(stdout(), ...)`` writes to standard output and not to standard error.

    Text on standard error would mean that the module swapped the handles of the two streams.
    """
    _fprintf_stdout(99)
    out, err = capfd.readouterr()
    assert out == "out 99\n", repr(out)
    assert err == ""


def test_snprintf_basic():
    """Check that ``snprintf`` writes a text that fits and returns its length without the NUL, on every platform."""
    buf = np.zeros(64, dtype=np.uint8)
    rc = _snprintf_into(buf, 7, 11)
    assert rc == len("[7:11]"), rc
    nul = buf.tolist().index(0)
    assert bytes(buf[:nul]) == b"[7:11]"


def test_snprintf_no_args():
    buf = np.zeros(32, dtype=np.uint8)
    rc = _snprintf_no_args(buf)
    assert rc == len("literal")
    nul = buf.tolist().index(0)
    assert bytes(buf[:nul]) == b"literal"


def test_snprintf_truncation_detection():
    """Check that ``snprintf`` reports a text longer than the buffer, by the test ``rc < 0 or rc >= size``.

    On Linux and macOS, jitted code calls C99's ``snprintf``, which returns the length of the whole text and ends the
    buffer with a NUL. On Windows, fmtio calls the ``_snprintf`` of ``msvcrt.dll`` instead, which returns -1 and does
    not end the buffer with a NUL, so the caller cannot determine the length of the whole text. The ``snprintf``
    docstring in ``numbox/core/bindings/fmtio.py`` states both rules.
    """
    buf = np.full(5, 0xFF, dtype=np.uint8)  # The 0xFF bytes show whether snprintf wrote the buffer.
    rc = _snprintf_into(buf, 12345, 67890)
    full_msg = b"[12345:67890]"  # 13 bytes

    # The condition rc < 0 or rc >= size detects the truncation on every platform.
    truncated = (rc < 0) or (rc >= buf.size)
    assert truncated, (
        f"expected truncation indicator: rc={rc}, buf.size={buf.size}"
    )

    if platform_ == "Windows":
        # _snprintf returns -1 for a text that does not fit.
        assert rc == -1, f"Windows _snprintf returns -1 on truncation; got {rc}"
        # _snprintf writes no NUL in that case, so the test does not check buf[-1].
    else:
        # C99's snprintf returns the length of the whole text and always ends the buffer with a NUL.
        assert rc == len(full_msg), rc
        assert buf[-1] == 0, (
            f"snprintf must NUL-terminate on POSIX; got {buf.tolist()!r}"
        )
        truncated_prefix = bytes(buf[:buf.size - 1])
        assert truncated_prefix == full_msg[:buf.size - 1], (
            truncated_prefix, full_msg
        )


def test_printf_non_literal_format_raises():
    """Check that a format passed as an argument fails to compile with a ``TypingError`` that names ``printf``.

    numba must read the format as a constant, such as a string literal, a module-level string or a closure variable.
    A format passed as an argument is a ``unicode_type`` value, which numba cannot read as a constant when it compiles
    the call.
    """
    @njit
    def caller(fmt):
        return printf(fmt, 1)

    with pytest.raises(TypingError, match=r"printf.*literal"):
        caller("dynamic %d")


def test_fprintf_non_literal_format_raises():
    @njit
    def caller(fmt):
        return fprintf(stderr(), fmt, 1)

    with pytest.raises(TypingError, match=r"fprintf.*literal"):
        caller("dynamic %d")


def test_snprintf_non_literal_format_raises():
    @njit
    def caller(buf, fmt):
        return snprintf(array_data_p(buf), buf.size, fmt, 1)

    buf = np.zeros(32, dtype=np.uint8)
    with pytest.raises(TypingError, match=r"snprintf.*literal"):
        caller(buf, "dynamic %d")


def test_fprintf_rejects_non_intp_fp():
    """Check that ``fprintf`` refuses a ``FILE *`` that is not an ``intp`` when numba compiles the call.

    The overload refuses every type but ``intp`` for ``fp``, here a ``float64``, with a ``TypingError`` that names
    ``fp``.
    """
    @njit
    def caller():
        # The first argument is a float64, not the intp that fprintf takes.
        return fprintf(np.float64(0.0), "x\n")

    with pytest.raises(TypingError, match=r"fprintf.*fp.*intp"):
        caller()


def test_snprintf_rejects_non_intp_buf():
    """Check that ``snprintf`` refuses a buffer address that is not an ``intp``, here a ``float64``."""
    @njit
    def caller():
        return snprintf(np.float64(0.0), 32, "x")

    with pytest.raises(TypingError, match=r"snprintf.*buf.*intp"):
        caller()


def test_snprintf_rejects_non_intp_size():
    """Check that ``snprintf`` refuses a size that is not an ``intp``, here an ``np.int32``.

    The call declares C's ``size_t`` parameter as ``intp``, which has the width of a pointer, so the size must have
    that type.
    """
    @njit
    def caller(buf_p):
        return snprintf(buf_p, np.int32(32), "x")

    with pytest.raises(TypingError, match=r"snprintf.*size.*intp"):
        # buf_p is a valid intp, so the size is the argument that fails the check.
        caller(np.intp(0))


# ============================================================================
# sscanf
# ============================================================================

@njit(cache=True)
def _sscanf_int(text_p, out_arr):
    return sscanf(text_p, "%d", array_data_p(out_arr))


@njit(cache=True)
def _sscanf_int_long(text_p, out_arr):
    return sscanf(text_p, "%lld", array_data_p(out_arr))


@njit(cache=True)
def _sscanf_double(text_p, out_arr):
    return sscanf(text_p, "%lf", array_data_p(out_arr))


@njit(cache=True)
def _sscanf_pair(text_p, n_out, x_out):
    return sscanf(text_p, "%d %lf", array_data_p(n_out), array_data_p(x_out))


@njit(cache=True)
def _sscanf_two_ints(text_p, a_out, b_out):
    return sscanf(text_p, "%d %d", array_data_p(a_out), array_data_p(b_out))


def test_sscanf_int_roundtrip():
    """Check that ``sscanf`` parses "42" into an ``int32`` slot and returns 1, the number of values that it stored."""
    out = np.zeros(1, dtype=np.int32)
    rc = _sscanf_int(get_unicode_data_p("42"), out)
    assert rc == 1, rc
    assert out[0] == 42, out[0]


def test_sscanf_int_negative():
    out = np.zeros(1, dtype=np.int32)
    rc = _sscanf_int(get_unicode_data_p("-17"), out)
    assert rc == 1
    assert out[0] == -17


def test_sscanf_int64_with_lld():
    """Check that ``%lld`` stores an ``int64``.

    ``%lld`` writes 8 bytes, so the caller provides an ``int64`` slot; with an ``int32`` slot, C would overwrite the 4
    bytes after it.
    """
    out = np.zeros(1, dtype=np.int64)
    rc = _sscanf_int_long(get_unicode_data_p("12345678901234"), out)
    assert rc == 1
    assert out[0] == 12345678901234


def test_sscanf_double():
    out = np.zeros(1, dtype=np.float64)
    rc = _sscanf_double(get_unicode_data_p("3.141592653589793"), out)
    assert rc == 1
    assert abs(out[0] - 3.141592653589793) < 1e-15, out[0]


def test_sscanf_multi_field():
    """Check that ``"%d %lf"`` stores an ``int32`` and a ``float64``, and that ``sscanf`` returns 2."""
    n_out = np.zeros(1, dtype=np.int32)
    x_out = np.zeros(1, dtype=np.float64)
    rc = _sscanf_pair(get_unicode_data_p("42 3.14"), n_out, x_out)
    assert rc == 2, rc
    assert n_out[0] == 42
    assert abs(x_out[0] - 3.14) < 1e-12


def test_sscanf_returns_partial_count_on_failed_conversion():
    """Check that ``sscanf`` returns the number of values stored before the first conversion that fails.

    For ``"%d %d"`` and the text "42 not_a_number", ``sscanf`` stores 42, leaves the second slot unchanged and
    returns 1.
    """
    a = np.zeros(1, dtype=np.int32)
    b = np.full(1, -999, dtype=np.int32)  # sscanf leaves this slot unchanged when its conversion fails.
    rc = _sscanf_two_ints(get_unicode_data_p("42 not_a_number"), a, b)
    assert rc == 1, rc
    assert a[0] == 42
    assert b[0] == -999, "b should not have been assigned"


def test_sscanf_returns_eof_on_empty_input():
    """Check that ``sscanf`` returns ``EOF``, -1, for empty text.

    C11 7.21.6.7 states that ``sscanf`` returns ``EOF`` if an input failure occurs before the first conversion has
    completed, and the end of the text is such a failure.
    """
    out = np.zeros(1, dtype=np.int32)
    rc = _sscanf_int(get_unicode_data_p(""), out)
    # EOF is -1 in the C libraries that numbox supports.
    assert rc == -1, rc


def test_sscanf_non_literal_format_raises():
    @njit
    def caller(text_p, out_arr, fmt):
        return sscanf(text_p, fmt, array_data_p(out_arr))

    out = np.zeros(1, dtype=np.int32)
    with pytest.raises(TypingError, match=r"sscanf.*literal"):
        caller(get_unicode_data_p("42"), out, "%d")


def test_sscanf_rejects_non_intp_output_arg():
    """Check that ``sscanf`` refuses a storage argument that is not an ``intp``, here a ``float64``.

    The check cannot refuse an ``int64`` value that is not an address, because ``intp`` is ``int64``. C would then
    write through that value as if it were an address.
    """
    @njit
    def caller():
        # The storage argument is a float64, not an address.
        return sscanf(get_unicode_data_p("42"), "%d", np.float64(0.0))

    with pytest.raises(TypingError, match=r"sscanf.*intp"):
        caller()


def test_sscanf_rejects_non_intp_int_output_arg():
    """Check that ``sscanf`` also refuses an ``np.int32`` storage argument.

    Each argument after the format is the address of the storage for one conversion, and numbox passes every address
    as an ``intp``.
    """
    @njit
    def caller():
        return sscanf(get_unicode_data_p("42"), "%d", np.int32(7))

    with pytest.raises(TypingError, match=r"sscanf.*intp"):
        caller()


def test_sscanf_rejects_non_intp_buf():
    """Check that ``sscanf`` refuses a ``str`` as ``buf``; the caller passes the address of the text instead."""
    @njit
    def caller(out_arr):
        # The text is a str, not its address from get_unicode_data_p.
        return sscanf("42", "%d", array_data_p(out_arr))

    out = np.zeros(1, dtype=np.int32)
    with pytest.raises(TypingError, match=r"sscanf.*buf.*intp"):
        caller(out)


# ============================================================================
# The same source under Python and in jitted code: each test runs one function in both modes and compares the results.
# ============================================================================


def _dual_kernel_printf(n, label):
    """Print ``n`` and ``label`` with ``printf``, flush C's stdout, and return ``2 * n``.

    The tests run the function under Python and through ``njit``. The values follow the format as separate
    arguments, and ``label`` is a ``str``: Python's ``%`` formats it under Python, and in jitted code the overload
    passes C its address.
    """
    printf("step %d: %s\n", n, label)
    fflush(stdout())
    return n * 2


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_dual_mode_printf_runs_same_source(capfd):
    """Check that ``printf`` prints the same text and returns the same value under Python and in jitted code.

    Under Python, ``printf`` formats with ``%`` and writes to ``sys.stdout``. In jitted code, numba compiles the call
    into ``_printf_intrinsic``, which calls C's ``printf``.
    """
    py_rc = _dual_kernel_printf(7, "before")
    py_out, _ = capfd.readouterr()

    njit_rc = njit(_dual_kernel_printf)(7, "before")
    njit_out, _ = capfd.readouterr()

    assert py_out == "step 7: before\n", repr(py_out)
    assert njit_out == "step 7: before\n", repr(njit_out)
    assert py_rc == njit_rc == 14


def _dual_kernel_fprintf_to_stderr(code, msg):
    fprintf(stderr(), "WARN [%d]: %s\n", code, msg)
    fflush(stderr())


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_dual_mode_fprintf_stderr_runs_same_source(capfd):
    """Check that ``fprintf(stderr(), ...)`` writes to standard error in both modes.

    Under Python, ``fprintf`` writes to ``sys.stderr``, which it finds by the address of C's ``stderr``. In jitted
    code, C's ``fprintf`` writes to the stream. Text on standard output would mean that the module swapped the handles
    of the two streams.
    """
    _dual_kernel_fprintf_to_stderr(7, "disk")
    py_out, py_err = capfd.readouterr()

    njit(_dual_kernel_fprintf_to_stderr)(7, "disk")
    njit_out, njit_err = capfd.readouterr()

    assert py_err == "WARN [7]: disk\n", repr(py_err)
    assert njit_err == "WARN [7]: disk\n", repr(njit_err)
    assert py_out == "" and njit_out == ""


def _dual_kernel_snprintf(buf, lo, hi):
    return snprintf(array_data_p(buf), buf.size, "[%d:%d]", lo, hi)


def test_dual_mode_snprintf_runs_same_source():
    """Check that ``snprintf`` writes the same bytes and returns the same length in both modes, for a text that fits.

    Under Python, ``ctypes.memmove`` writes the text into the caller's buffer; in jitted code, C's ``snprintf`` does.
    """
    buf_py = np.zeros(32, dtype=np.uint8)
    py_n = _dual_kernel_snprintf(buf_py, 7, 11)
    py_nul = buf_py.tolist().index(0)

    buf_njit = np.zeros(32, dtype=np.uint8)
    njit_n = njit(_dual_kernel_snprintf)(buf_njit, 7, 11)
    njit_nul = buf_njit.tolist().index(0)

    assert py_n == njit_n == len(b"[7:11]")
    assert bytes(buf_py[:py_nul]) == b"[7:11]"
    assert bytes(buf_njit[:njit_nul]) == b"[7:11]"


def _dual_kernel_printf_lld_with_int32(x):
    """Print ``x`` with ``%lld`` and flush C's stdout; the test passes an ``int32`` as ``x``."""
    printf("%lld\n", x)
    fflush(stdout())


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_dual_mode_int32_promotes_to_int64_for_lld(capfd):
    """Check that ``printf("%lld", x)`` prints an ``int32`` correctly in both modes.

    In jitted code, ``_promote_for_varargs`` sign-extends the ``int32`` to the 64 bits that ``%lld`` reads; without the
    extension, the upper 4 bytes that C reads would be undefined. Under Python, ``printf`` removes the ``ll`` before
    Python's ``%`` operator formats the value, because ``%`` raises ``ValueError`` for ``%lld``.
    """
    _dual_kernel_printf_lld_with_int32(np.int32(7))
    py_out, _ = capfd.readouterr()
    njit(_dual_kernel_printf_lld_with_int32)(np.int32(7))
    njit_out, _ = capfd.readouterr()
    assert py_out == "7\n", repr(py_out)
    assert njit_out == "7\n", repr(njit_out)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_dual_mode_string_auto_conversion(capfd):
    """Check that a ``str`` argument prints the same in both modes.

    The caller passes the ``str`` without ``get_unicode_data_p``. The overload finds the ``str`` by its numba type and
    passes C the address of its characters.
    """
    def kernel(label):
        printf("hello %s\n", label)
        fflush(stdout())

    kernel("world")
    py_out, _ = capfd.readouterr()
    njit(kernel)("world")
    njit_out, _ = capfd.readouterr()
    assert py_out == "hello world\n", repr(py_out)
    assert njit_out == "hello world\n", repr(njit_out)


def test_fprintf_pure_python_rejects_arbitrary_fp():
    """Check that ``fprintf`` under Python refuses a ``FILE *`` other than the three standard streams.

    Under Python, ``fprintf`` writes only to ``sys.stdout``, ``sys.stderr`` and ``sys.stdin``, which it finds by the
    address of the matching C stream. For any other address, such as one from ``fopen``, it raises ``RuntimeError``,
    and the message names ``@njit`` and Python's ``open()``.
    """
    fake_fp = 0xdeadbeef
    with pytest.raises(RuntimeError, match=r"fprintf.*pure-Python.*stdout"):
        fprintf(fake_fp, "x\n")


def test_sscanf_raises_in_pure_python():
    """Check that ``sscanf`` raises ``NotImplementedError`` under Python, because it runs only in jitted code."""
    out = np.zeros(1, dtype=np.int32)
    with pytest.raises(NotImplementedError, match=r"@njit-only"):
        sscanf(int(get_unicode_data_p("42")), "%d", int(array_data_p(out)))


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_accepts_utf8_format_literal(capfd):
    """Check that a format with non-ASCII text compiles and that C prints that text in UTF-8.

    numbox encodes the format as UTF-8 when numba compiles the call, and C copies the bytes outside a conversion
    unchanged.
    """
    _printf_utf8(42)
    out, _ = capfd.readouterr()
    # capfd decodes the captured bytes as UTF-8, so out holds the original characters. The test compares both the
    # str and its UTF-8 bytes.
    assert out == "café=42\n", repr(out)
    assert out.encode("utf-8") == NON_ASCII_EXPECTED


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_fprintf_accepts_utf8_format_literal(capfd):
    _fprintf_utf8(42)
    out, _ = capfd.readouterr()
    assert out == "café=42\n", repr(out)
    assert out.encode("utf-8") == NON_ASCII_EXPECTED


def test_snprintf_accepts_utf8_format_literal():
    """Check the UTF-8 bytes of a non-ASCII format in the buffer that ``snprintf`` fills, without ``capfd``."""
    buf = np.zeros(32, dtype=np.uint8)
    rc = _snprintf_utf8(buf, 42)
    # The result differs by platform only for a text that does not fit. For this text, every platform returns its
    # length in bytes without the NUL.
    assert rc == len(NON_ASCII_EXPECTED), (rc, len(NON_ASCII_EXPECTED))
    nul = buf.tolist().index(0)
    assert bytes(buf[:nul]) == NON_ASCII_EXPECTED, bytes(buf[:nul])


def test_fmtio_caller_survives_subprocess_round_trip(tmp_path):
    """Check that a cached jitted caller of ``snprintf`` works in a second process without a recompile.

    The caller's compiled code names C's ``snprintf`` and holds the format as a UTF-8 constant, with no address from
    the compiling process. The second process therefore loads the cached code, and the modification times of the
    cache files do not change. ``assert_njit_cache_survives_subprocess_roundtrip`` in ``test/auxiliary_utils.py``
    states what the helper asserts.
    """
    assert_njit_cache_survives_subprocess_roundtrip(
        tmp_path,
        probe_source="""
            import numpy as np
            from numba import njit
            from numbox.core.bindings.fmtio import snprintf
            from numbox.utils.lowlevel import array_data_p

            @njit(cache=True)
            def go():
                buf = np.zeros(32, dtype=np.uint8)
                n = snprintf(array_data_p(buf), buf.size, "[%d]", 42)
                return n

            v = go()
            assert v == 4, v
            print(v)
        """,
        expected_stdout_lines=["4"],
    )


def test_all_binding_families_survive_subprocess_round_trip(tmp_path):
    """Check that one cached jitted function that calls a binding of each family works in a second process.

    The families are errno, the stdio handles, ``strerror_safe``, libc strings (``strcmp`` and ``strchr``), libc memory
    (``memset``, ``memcpy`` and ``memcmp``), the environment (``getenv``) and formatted I/O (``snprintf`` and
    ``printf``). The probe prints "OK <family>" or "FAIL <family>" with ``printf`` for each check, and both processes
    must print the expected lines, and the second run must leave the cache files unchanged.
    ``test_proxy_caller_survives_subprocess_round_trip`` and ``test_fmtio_caller_survives_subprocess_round_trip`` each
    run a single binding in the same way.
    """
    assert_njit_cache_survives_subprocess_roundtrip(
        tmp_path,
        probe_source=r"""
            import numpy as np
            from numba import njit
            from numbox.core.bindings.errno import errno_get, errno_set
            from numbox.core.bindings.stdio import stdout, stderr
            from numbox.core.bindings.strerror import strerror_safe
            from numbox.core.bindings.libc import strcmp, strchr, memset, memcpy, memcmp, getenv, fflush
            from numbox.core.bindings.fmtio import snprintf, printf
            from numbox.utils.lowlevel import (
                array_data_p, get_unicode_data_p,
            )

            @njit(cache=True)
            def report(label_ptr, ok):
                if ok:
                    printf("OK %s\n", label_ptr)
                else:
                    printf("FAIL %s\n", label_ptr)

            @njit(cache=True)
            def exercise():
                # errno round-trip
                errno_set(7)
                report(get_unicode_data_p("errno"), errno_get() == 7)

                # stdio handles -- non-zero and distinct
                op = stdout()
                ep = stderr()
                report(get_unicode_data_p("stdio_handles"),
                       op != 0 and ep != 0 and op != ep)

                # strerror_safe: rc == 0 on success
                sbuf = np.zeros(64, dtype=np.uint8)
                sr = strerror_safe(2, array_data_p(sbuf), sbuf.size)
                report(get_unicode_data_p("strerror_safe"), sr == 0)

                # strcmp on equal strings returns 0
                a = get_unicode_data_p("hello")
                report(get_unicode_data_p("strcmp"), strcmp(a, a) == 0)

                # strchr finds 'l' (ASCII 108) in "hello"
                report(get_unicode_data_p("strchr"),
                       strchr(a, np.int32(108)) != 0)

                # memset writes 0x7f (no high-bit), memcpy duplicates,
                # memcmp returns 0. Avoids the macOS 3.14 quirk where
                # `src[0] == 0xab` evaluated False even after memset --
                # the high-bit-set comparison was the brittle part.
                src = np.zeros(8, dtype=np.uint8)
                memset(array_data_p(src), 0x7f, 8)
                dst = np.zeros(8, dtype=np.uint8)
                memcpy(array_data_p(dst), array_data_p(src), 8)
                eq = memcmp(array_data_p(src), array_data_p(dst), 8)
                report(get_unicode_data_p("mem"), eq == 0)

                # getenv of a deliberately-unset variable returns NULL
                miss = getenv(get_unicode_data_p("NUMBOX_NEVER_SET_xyzzy_4f1c"))
                report(get_unicode_data_p("getenv"), miss == 0)

                # snprintf success: "[42]" is 4 bytes (excluding NUL).
                # On Windows MSVCRT _snprintf returns the same non-
                # negative count for fits-in-buffer calls, so the
                # comparison is portable.
                nbuf = np.zeros(16, dtype=np.uint8)
                snp = snprintf(array_data_p(nbuf), nbuf.size, "[%d]", 42)
                report(get_unicode_data_p("snprintf"), snp == 4)

                # printf success: returns byte count written.
                pr = printf("trailer\n")  # 8 bytes incl newline
                fflush(stdout())
                report(get_unicode_data_p("printf"), pr == 8)

            exercise()
        """,
        expected_stdout_lines=[
            "OK errno",
            "OK stdio_handles",
            "OK strerror_safe",
            "OK strcmp",
            "OK strchr",
            "OK mem",
            "OK getenv",
            "OK snprintf",
            "trailer",
            "OK printf",
        ],
    )


# ---------------------------------------------------------------------------
# The checks on argument types, on %n and on the ' flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", ["printf", "fprintf", "snprintf"])
def test_njit_writer_rejects_array_arg(fn_name):
    """Check that ``printf``, ``fprintf`` and ``snprintf`` refuse an array argument when numba compiles the call.

    ``_validate_writer_arg_type`` accepts only a float, an integer, a ``bool`` or a ``str``, and it accepts a pointer
    as an ``intp`` integer. Its error message contains "unsupported type". The overload runs the check, and the
    intrinsic runs it again.
    """
    src = {
        "printf": "from numbox.core.bindings.fmtio import printf\n"
                  "import numpy as np\n"
                  "@njit\ndef k(a): printf('%d', a)\n",
        "fprintf": "from numbox.core.bindings.fmtio import fprintf\nfrom numbox.core.bindings.stdio import stdout\n"
                   "import numpy as np\n"
                   "@njit\ndef k(a): fprintf(stdout(), '%d', a)\n",
        "snprintf": "from numbox.core.bindings.fmtio import snprintf\n"
                    "from numbox.utils.lowlevel import array_data_p\n"
                    "import numpy as np\n"
                    "@njit\ndef k(buf, a): snprintf(array_data_p(buf), buf.size, '%d', a)\n",
    }[fn_name]
    ns = {"njit": njit}
    exec(src, ns)
    arr = np.zeros(3, dtype=np.int32)
    with pytest.raises(TypingError, match="unsupported type"):
        if fn_name == "snprintf":
            ns["k"](np.zeros(16, dtype=np.uint8), arr)
        else:
            ns["k"](arr)


@pytest.mark.parametrize("fn_name", ["printf", "fprintf", "snprintf"])
def test_njit_writer_rejects_complex_arg(fn_name):
    src = {
        "printf": "from numbox.core.bindings.fmtio import printf\n"
                  "@njit\ndef k(c): printf('%f', c)\n",
        "fprintf": "from numbox.core.bindings.fmtio import fprintf\nfrom numbox.core.bindings.stdio import stdout\n"
                   "@njit\ndef k(c): fprintf(stdout(), '%f', c)\n",
        "snprintf": "from numbox.core.bindings.fmtio import snprintf\n"
                    "from numbox.utils.lowlevel import array_data_p\n"
                    "import numpy as np\n"
                    "@njit\ndef k(buf, c): snprintf(array_data_p(buf), buf.size, '%f', c)\n",
    }[fn_name]
    ns = {"njit": njit}
    exec(src, ns)
    val = np.complex128(1 + 2j)
    with pytest.raises(TypingError, match="unsupported type"):
        if fn_name == "snprintf":
            ns["k"](np.zeros(16, dtype=np.uint8), val)
        else:
            ns["k"](val)


@pytest.mark.parametrize("fn_name", ["printf", "fprintf", "snprintf"])
def test_njit_writer_rejects_tuple_arg(fn_name):
    src = {
        "printf": "from numbox.core.bindings.fmtio import printf\n"
                  "@njit\ndef k(t): printf('%d %d', t)\n",
        "fprintf": "from numbox.core.bindings.fmtio import fprintf\nfrom numbox.core.bindings.stdio import stdout\n"
                   "@njit\ndef k(t): fprintf(stdout(), '%d %d', t)\n",
        "snprintf": "from numbox.core.bindings.fmtio import snprintf\n"
                    "from numbox.utils.lowlevel import array_data_p\n"
                    "import numpy as np\n"
                    "@njit\ndef k(buf, t): snprintf(array_data_p(buf), buf.size, '%d %d', t)\n",
    }[fn_name]
    ns = {"njit": njit}
    exec(src, ns)
    with pytest.raises(TypingError, match="unsupported type"):
        if fn_name == "snprintf":
            ns["k"](np.zeros(16, dtype=np.uint8), (7, 8))
        else:
            ns["k"]((7, 8))


@pytest.mark.parametrize(
    "fmt", ["%n", "%ln", "%lln", "%hn", "%hhn", "%5n", "%5.3n", "before %n after",
            "%qn", "%I32n", "%I64n",
            "%1$n", "%2$n", "%1$ln", "%99$n",
            "%1$*2$n", "%1$.*2$n", "%1$*2$.*3$n"]
)
def test_njit_printf_rejects_percent_n(fmt):
    """Check that ``printf`` refuses ``%n`` with a width, a precision, a length modifier or a position.

    ``%n`` makes C write the number of characters printed so far through a pointer argument, which is a memory-safety
    hazard. Python's ``%`` operator raises ``ValueError`` for ``%n``, so jitted code refuses it as well.
    """
    src = f"""
from numbox.core.bindings.fmtio import printf
from numbox.utils.lowlevel import array_data_p
@njit
def k(out):
    printf({fmt!r}, array_data_p(out))
"""
    ns = {"njit": njit}
    exec(src, ns)
    out = np.zeros(1, dtype=np.intp)
    with pytest.raises(TypingError, match="%n.*not allowed"):
        ns["k"](out)


@pytest.mark.parametrize("fn", ["printf", "fprintf", "snprintf"])
def test_njit_writer_rejects_percent_n(fn):
    if fn == "printf":
        src = "from numbox.core.bindings.fmtio import printf\n" \
              "from numbox.utils.lowlevel import array_data_p\n" \
              "@njit\ndef k(out): printf('count=%n', array_data_p(out))\n"
    elif fn == "fprintf":
        src = "from numbox.core.bindings.fmtio import fprintf\nfrom numbox.core.bindings.stdio import stdout\n" \
              "from numbox.utils.lowlevel import array_data_p\n" \
              "@njit\ndef k(out): fprintf(stdout(), 'count=%n', array_data_p(out))\n"
    else:
        src = "from numbox.core.bindings.fmtio import snprintf\n" \
              "from numbox.utils.lowlevel import array_data_p\n" \
              "import numpy as np\n" \
              "@njit\ndef k(buf, out): snprintf(array_data_p(buf), buf.size, 'count=%n', array_data_p(out))\n"
    ns = {"njit": njit}
    exec(src, ns)
    out = np.zeros(1, dtype=np.intp)
    with pytest.raises(TypingError, match="%n.*not allowed"):
        if fn == "snprintf":
            ns["k"](np.zeros(16, dtype=np.uint8), out)
        else:
            ns["k"](out)


def test_njit_printf_grouping_flag_prevents_varargs_misread():
    """Check that ``"%'d %d"`` with one argument fails to compile with a ``TypingError`` about grouping.

    ``_FMT_SPEC_RE`` does not recognize the ``'`` flag, so without the grouping check the format would count as one
    conversion and compile with one argument. The second ``%d`` would then read an argument slot that the call never
    filled.
    """
    @njit
    def k(x):
        printf("%'d %d\n", x)
    with pytest.raises(TypingError, match="grouping"):
        k(1000000)


@pytest.mark.parametrize(
    "fmt", ["%'d", "%-'d", "%0'8d", "%'i", "%'u", "%'x", "%'f",
            "total %'d items", "%'n", "%d %'d"]
)
def test_njit_printf_rejects_grouping_flag(fmt):
    """Check that ``printf`` refuses the ``'`` flag, which groups the digits of a number by thousands.

    POSIX defines the flag, but C99 and Python's ``%`` operator lack it, so jitted code refuses it, as it refuses
    ``%n``. ``_PERCENT_N_RE`` does not match ``%'n``, because ``'`` is not among its flags, so the grouping check is
    the one that refuses that format.
    """
    src = f"""
from numbox.core.bindings.fmtio import printf
@njit
def k(x):
    printf({fmt!r}, x)
"""
    ns = {"njit": njit}
    exec(src, ns)
    with pytest.raises(TypingError, match="grouping"):
        ns["k"](1000000)


@pytest.mark.parametrize("fn", ["printf", "fprintf", "snprintf"])
def test_njit_writer_rejects_grouping_flag(fn):
    if fn == "printf":
        src = "from numbox.core.bindings.fmtio import printf\n" \
              "@njit\ndef k(x): printf(\"%'d\", x)\n"
    elif fn == "fprintf":
        src = "from numbox.core.bindings.fmtio import fprintf\nfrom numbox.core.bindings.stdio import stdout\n" \
              "@njit\ndef k(x): fprintf(stdout(), \"%'d\", x)\n"
    else:
        src = "from numbox.core.bindings.fmtio import snprintf\n" \
              "from numbox.utils.lowlevel import array_data_p\n" \
              "@njit\ndef k(buf, x): snprintf(array_data_p(buf), buf.size, \"%'d\", x)\n"
    ns = {"njit": njit}
    exec(src, ns)
    with pytest.raises(TypingError, match="grouping"):
        if fn == "snprintf":
            ns["k"](np.zeros(32, dtype=np.uint8), 1000000)
        else:
            ns["k"](1000000)


def test_pure_python_printf_rejects_grouping_flag():
    """Check that ``printf`` under Python raises a ``ValueError`` about grouping for the ``'`` flag.

    The message is the one that jitted code gives with its ``TypingError``, not Python's generic "unsupported format
    character".
    """
    with pytest.raises(ValueError, match="grouping"):
        printf("%'d\n", 1000000)


def test_njit_sscanf_accepts_percent_n():
    """Check that ``sscanf`` accepts ``%n``, which stores the number of characters read so far.

    The output functions refuse ``%n``, which writes through a pointer. In ``sscanf``, every conversion writes through
    a pointer, and ``%n`` reads no input. For ``"%d%n"`` and the text "12345 rest", ``sscanf`` stores 12345 and 5. It
    returns 1, because ``%n`` does not count as a stored value.
    """

    @njit(cache=False)
    def parse(buf_p, n_out, pos_out):
        return sscanf(buf_p, "%d%n", array_data_p(n_out), array_data_p(pos_out))

    buf = get_unicode_data_p("12345 rest")
    # %d and %n each write a 4-byte int. Each slot is an 8-byte intp that starts at zero, so on a little-endian
    # machine it reads back as the number that sscanf stored.
    n_out = np.zeros(1, dtype=np.intp)
    pos_out = np.zeros(1, dtype=np.intp)
    rc = parse(buf, n_out, pos_out)
    assert rc == 1, f"sscanf returned {rc}, expected 1 successful conversion"
    assert n_out[0] == 12345
    assert pos_out[0] == 5


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_njit_writer_accepts_literal_percent_percent_n(capfd):
    """Check that ``%%n``, a literal ``%`` followed by ``n``, compiles and prints ``%n``."""

    @njit(cache=False)
    def k():
        rc = printf("100%%n done\n")
        fflush(stdout())
        return rc

    rc = k()
    assert rc == len("100%n done\n")
    out, _ = capfd.readouterr()
    assert out == "100%n done\n"


def test_python_printf_rejects_percent_n():
    with pytest.raises(ValueError, match="%n.*not allowed"):
        printf("count=%n", 42)


@pytest.mark.parametrize("fmt", ["%1$n", "%2$n", "%1$ln", "%99$n",
                                 "%1$*2$n", "%1$.*2$n", "%1$*2$.*3$n"])
def test_python_printf_rejects_positional_percent_n(fmt):
    """Check that ``printf`` under Python refuses ``%n`` with a POSIX position, such as ``%1$n`` or ``%1$*2$n``.

    glibc supports positional conversions, and ``%1$n`` writes through a pointer as ``%n`` does. A pattern that
    matched only a plain ``%n`` would not refuse such a format, so ``_PERCENT_N_RE`` also matches a position in the
    conversion, in its width and in its precision.
    """
    with pytest.raises(ValueError, match="%n.*not allowed"):
        printf(fmt, 42)


def test_python_fprintf_rejects_percent_n():
    with pytest.raises(ValueError, match="%n.*not allowed"):
        fprintf(stdout(), "count=%n", 42)


def test_python_snprintf_rejects_percent_n():
    buf = np.zeros(16, dtype=np.uint8)
    with pytest.raises(ValueError, match="%n.*not allowed"):
        snprintf(array_data_p(buf), buf.size, "count=%n", 42)


def test_python_snprintf_truncation_nul_terminates():
    """Check that ``snprintf`` under Python ends a truncated text with a NUL, as C99 requires.

    The test passes an 8-byte buffer and the 11-byte text "hello world". The buffer then holds the 7 bytes "hello w"
    and a NUL at index 7, and ``snprintf`` returns 11.
    """
    buf = np.zeros(8, dtype=np.uint8)
    rc = snprintf(array_data_p(buf), buf.size, "hello world")
    assert rc == 11
    assert bytes(buf[:7]) == b"hello w"
    assert buf[7] == 0, f"expected NUL at index 7, got {buf[7]}"


def test_python_printf_accepts_literal_percent_percent_n(capfd):
    rc = printf("100%%n done\n")
    assert rc == len("100%n done\n")
    out, _ = capfd.readouterr()
    assert out == "100%n done\n"


@pytest.mark.parametrize("modifier", ["z", "j", "t", "q", "I32", "I64"])
def test_python_printf_strips_extended_length_modifiers(modifier, capfd):
    """Check that ``printf`` under Python removes the length modifiers ``z``, ``j``, ``t``, ``q``, ``I32`` and ``I64``.

    Python's ``%`` operator raises ``ValueError`` for each of them. ``printf`` removes them, as it removes ``hh``,
    ``h``, ``l``, ``ll`` and ``L``, so a format that compiles in jitted code also runs under Python.
    """
    rc = printf(f"%{modifier}d\n", 42)
    out, _ = capfd.readouterr()
    assert out == "42\n", repr(out)
    assert rc == 3


def test_printf_too_few_args_for_specifiers_raises():
    """Check that ``"%d %d"`` with one argument fails to compile with a ``TypingError``.

    The second ``%d`` would read an argument slot that the call never filled, so numbox requires as many arguments as
    conversions when numba compiles the call.
    """
    @njit
    def caller(x):
        return printf("%d %d\n", x)

    with pytest.raises(TypingError, match=r"printf.*specifier"):
        caller(1)


def test_printf_percent_s_with_non_pointer_int_raises():
    """Check that ``%s`` refuses an ``int32`` argument when numba compiles the call.

    C would read the integer as a ``char *``. numbox cannot refuse an ``int64`` there, because an address is an
    ``intp`` and ``intp`` is ``int64``, but an integer narrower than a pointer cannot be an address.
    """
    @njit
    def caller(x):
        return printf("%s\n", x)

    with pytest.raises(TypingError, match=r"printf.*specifier.*str"):
        caller(np.int32(5))


def test_printf_percent_d_with_float_arg_raises():
    """Check that ``%d`` refuses a float argument when numba compiles the call.

    C would read an integer where the call passed a double, and it would print a wrong value.
    """
    @njit
    def caller(x):
        return printf("%d\n", x)

    with pytest.raises(TypingError, match=r"printf.*specifier.*int"):
        caller(3.5)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_matched_spec_and_args_compiles_both_modes(capfd):
    """Check that a format whose conversions match its arguments compiles and prints the same text in both modes."""
    @njit
    def jit_caller(n, ratio):
        rc = printf("n=%d r=%.2f\n", n, ratio)
        fflush(stdout())
        return rc

    jit_caller(7, 3.5)
    out_jit, _ = capfd.readouterr()
    printf("n=%d r=%.2f\n", 7, 3.5)
    out_py, _ = capfd.readouterr()
    assert out_jit == "n=7 r=3.50\n", repr(out_jit)
    assert out_py == out_jit


def test_validate_format_accepts_uintp_for_string_and_pointer():
    """Check that ``_validate_format_vs_args`` accepts a ``uintp`` for ``%s`` and ``%p``, as it accepts an ``intp``.

    A ``uintp`` is the unsigned integer of pointer width, so it can hold an address.
    """
    from numba.core.types import Tuple, uintp
    from numbox.core.bindings.fmtio import _validate_format_vs_args

    _validate_format_vs_args("printf", "%s %p", Tuple([uintp, uintp]))
