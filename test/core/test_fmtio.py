"""Tests for the variadic printf/fprintf/snprintf/sscanf intrinsics in
numbox.core.bindings._fmtio.

Coverage:
- Basic round-trips via capfd / buffer-write
- C ABI default-argument promotion (float32 -> double, int8/16 -> int32)
- Pointer-as-string via %s
- Truncation detection on snprintf
- TypingError on non-literal format strings
- TypingError on non-tuple args
- Empty args tuple
- UTF-8 format string acceptance
- @njit(cache=True) survives a subprocess round-trip — these intrinsics
  emit a direct libc extern call, JIT linker resolves per process
- sscanf: parse-into-numpy roundtrip, multi-field, EOF, partial-match,
  intp-output enforcement
"""
import numpy as np
import pytest
from numba import njit
from numba.core.errors import TypingError

from numbox.core.bindings import (
    fflush,
    fprintf,
    printf,
    snprintf,
    sscanf,
    stderr,
    stdout,
)
from numbox.core.bindings.utils import platform_
from numbox.utils.lowlevel import array_data_p, get_unicode_data_p
from test.auxiliary_utils import assert_njit_cache_survives_subprocess_roundtrip


# stdout is block-buffered when not a terminal (e.g. under pytest's capfd
# redirection), so each test helper flushes after printf so pytest's
# capture sees the output before the test returns. stderr is unbuffered
# but we flush it too for consistency / future-proofing.


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
    # float32 must be promoted to double for %f
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
    # numba's Boolean type is i8 at the LLVM level (1 or 0); without
    # explicit promotion to int32+ in _promote_for_varargs, printf reading
    # %d would consume the i8 plus garbage bytes in the high bits of the
    # register slot.
    rc = printf("[%d %d]\n", True, False)
    fflush(stdout())
    return rc


@njit(cache=True)
def _printf_string_via_pointer(s_p):
    """User explicitly passes get_unicode_data_p — still supported for
    backward compatibility and for cases where the user has a precomputed
    intp pointer."""
    rc = printf("hi %s!\n", s_p)
    fflush(stdout())
    return rc


@njit(cache=True)
def _printf_string_native(s):
    """User passes a Python str directly — the @overload auto-converts
    via get_unicode_data_p so libc sees a NUL-terminated C string."""
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


# Non-ASCII format-string helpers. "café=" + "%d" + "\n": the 'é' is U+00E9,
# which encodes as the two-byte UTF-8 sequence b"\xc3\xa9". The expected
# rendered bytes for n=42 are b"caf\xc3\xa9=42\n" (9 bytes). With ASCII
# encoding these helpers would have raised UnicodeEncodeError at compile
# time, so the very fact that they compile + execute is the load-bearing
# proof that the format-string encoding is UTF-8.
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
    """C ABI default-arg promotion: float32 passed to %f must be widened to
    double before being placed in the variadic call. If the binding didn't
    do the fpext, libc printf would read 64 bits of which the high 32 are
    garbage — output would be wildly wrong."""
    _printf_float32(3.14159)
    out, _ = capfd.readouterr()
    # float32(3.14159) rounds to ~3.1415927; %.3f truncates to "3.142".
    # Allow either "3.141" or "3.142" depending on rounding; both prove the
    # value made it through cleanly (i.e. the float32 was promoted, not
    # read as garbage bits).
    assert out.strip() in ("3.141", "3.142"), repr(out)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_int8_is_promoted_to_int32(capfd):
    """C ABI default-arg promotion: int8/int16 must be widened to int32.
    Without it, libc would read 32 bits where only 8 were written and the
    high bits would be garbage."""
    _printf_int8()
    out, _ = capfd.readouterr()
    assert out == "[-7 42]\n", repr(out)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_bool_is_promoted_to_int32(capfd):
    """C ABI default-arg promotion: numba's Boolean type is i8 at the LLVM
    level. Without explicit zext to int32, printf reading %d would consume
    the bool's 1 byte + 3 garbage bytes from the variadic register slot.
    The binding's _promote_for_varargs handles Boolean explicitly because
    numba's Boolean is NOT a subclass of Integer (they're siblings under
    Number) — the int<32 widening branch wouldn't catch it on its own."""
    _printf_bool()
    out, _ = capfd.readouterr()
    assert out == "[1 0]\n", repr(out)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_string_via_pointer(capfd):
    """%s with a precomputed NUL-terminated string pointer (intp). Caller-
    side get_unicode_data_p is still supported for backward compat and
    for cases where the user has a precomputed pointer."""
    s_p = get_unicode_data_p("world")
    _printf_string_via_pointer(s_p)
    out, _ = capfd.readouterr()
    assert out == "hi world!\n", repr(out)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_string_native(capfd):
    """%s with a numba unicode_type (Python str) — the @overload auto-
    converts via get_unicode_data_p. User doesn't need to call it
    themselves. This is the dual-mode-friendly idiom."""
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
    """fprintf(stdout(), ...) must land on stdout, not stderr — guards
    against transposed FILE* handles."""
    _fprintf_stdout(99)
    out, err = capfd.readouterr()
    assert out == "out 99\n", repr(out)
    assert err == ""


def test_snprintf_basic():
    """A fits-in-buffer call writes the expected bytes and returns the
    written-count (excluding NUL). This part of the contract is identical
    across all platforms."""
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
    """snprintf truncation behavior diverges by platform — see the
    snprintf docstring in numbox/core/bindings/_fmtio.py.

    - **Linux/macOS** (POSIX/C99 ``snprintf``): ``rc`` is the would-have-
      written count (excluding NUL); ``rc >= size`` signals truncation;
      the buffer is always NUL-terminated when ``size > 0``.
    - **Windows** (MSVCRT ``_snprintf``, what numba's ``cgutils.snprintf``
      resolves to): ``rc < 0`` signals truncation; the buffer is NOT
      guaranteed to be NUL-terminated. The would-have-written count is
      not recoverable.

    The portable check ``(rc < 0) or (rc >= size)`` works on both."""
    buf = np.full(5, 0xFF, dtype=np.uint8)  # 0xFF pre-fill catches no-write
    rc = _snprintf_into(buf, 12345, 67890)
    full_msg = b"[12345:67890]"  # 13 bytes

    # Portable truncation signal — must hold on every platform.
    truncated = (rc < 0) or (rc >= buf.size)
    assert truncated, (
        f"expected truncation indicator: rc={rc}, buf.size={buf.size}"
    )

    if platform_ == "Windows":
        # MSVCRT _snprintf returns -1 on truncation; no NUL-term guarantee.
        assert rc == -1, f"Windows _snprintf returns -1 on truncation; got {rc}"
        # buf[-1] is not guaranteed to be 0; don't assert on it.
    else:
        # POSIX/C99 snprintf returns the would-have-written count and
        # always NUL-terminates within the buffer.
        assert rc == len(full_msg), rc
        assert buf[-1] == 0, (
            f"snprintf must NUL-terminate on POSIX; got {buf.tolist()!r}"
        )
        truncated_prefix = bytes(buf[:buf.size - 1])
        assert truncated_prefix == full_msg[:buf.size - 1], (
            truncated_prefix, full_msg
        )


def test_printf_non_literal_format_raises():
    """The format string MUST be a Python str literal at the call site —
    numba surfaces it as a Literal[str] type. A runtime-built unicode is
    rejected at typing time with a clean TypingError naming the binding.
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
    """fprintf's FILE* arg must be intp (pointer-as-int). Passing a wider
    or narrower type produces a clean TypingError instead of an opaque
    inttoptr/IR-lowering failure that would either truncate the address
    or produce malformed IR."""
    @njit
    def caller():
        # np.float64 in the FILE* slot — definitely not an intp pointer
        return fprintf(np.float64(0.0), "x\n")

    with pytest.raises(TypingError, match=r"fprintf.*fp.*intp"):
        caller()


def test_snprintf_rejects_non_intp_buf():
    """snprintf's destination buffer pointer must be intp."""
    @njit
    def caller():
        return snprintf(np.float64(0.0), 32, "x")

    with pytest.raises(TypingError, match=r"snprintf.*buf.*intp"):
        caller()


def test_snprintf_rejects_non_intp_size():
    """snprintf's size must be intp (size_t-as-int). Passing a narrower
    integer would declare the libc snprintf with the wrong size_t width
    in the LLVM IR signature, corrupting the variadic ABI on 64-bit
    platforms."""
    @njit
    def caller(buf_p):
        return snprintf(buf_p, np.int32(32), "x")

    with pytest.raises(TypingError, match=r"snprintf.*size.*intp"):
        # need a valid intp for buf so the failure is on size, not buf
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
    """Parse a single int32 from a NUL-terminated input buffer. rc must be
    1 (one item assigned); the int32 numpy slot must hold the parsed value.
    """
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
    """%lld writes 8 bytes — the caller MUST provide an int64-sized slot.
    int32 here would silently corrupt adjacent memory."""
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
    """Multi-field parse — %d into int32, %lf into float64. rc == 2."""
    n_out = np.zeros(1, dtype=np.int32)
    x_out = np.zeros(1, dtype=np.float64)
    rc = _sscanf_pair(get_unicode_data_p("42 3.14"), n_out, x_out)
    assert rc == 2, rc
    assert n_out[0] == 42
    assert abs(x_out[0] - 3.14) < 1e-12


def test_sscanf_returns_partial_count_on_failed_conversion():
    """sscanf returns the count assigned BEFORE the first conversion that
    fails. Format "%d %d" against input "42 not_a_number" assigns 42 to the
    first slot, fails to parse the second, returns 1."""
    a = np.zeros(1, dtype=np.int32)
    b = np.full(1, -999, dtype=np.int32)  # sentinel: untouched on failure
    rc = _sscanf_two_ints(get_unicode_data_p("42 not_a_number"), a, b)
    assert rc == 1, rc
    assert a[0] == 42
    assert b[0] == -999, "b should not have been assigned"


def test_sscanf_returns_eof_on_empty_input():
    """Per C11 7.21.6.2: sscanf returns EOF (-1) if an input failure occurs
    before any conversion. Empty input + a conversion spec triggers EOF."""
    out = np.zeros(1, dtype=np.int32)
    rc = _sscanf_int(get_unicode_data_p(""), out)
    # POSIX/glibc/musl: -1 (EOF). Some libcs may use a different negative
    # value, but EOF is canonically -1 on every platform numbox supports.
    assert rc == -1, rc


def test_sscanf_non_literal_format_raises():
    @njit
    def caller(text_p, out_arr, fmt):
        return sscanf(text_p, fmt, array_data_p(out_arr))

    out = np.zeros(1, dtype=np.int32)
    with pytest.raises(TypingError, match=r"sscanf.*literal"):
        caller(get_unicode_data_p("42"), out, "%d")


def test_sscanf_rejects_non_intp_output_arg():
    """sscanf's variadic outputs MUST be intp (pointer-as-int). The binding
    validates this at typing time so the user can't accidentally pass an
    integer value where a pointer is expected (which would have sscanf
    write through whatever bits happen to be there → segfault or memory
    corruption)."""
    @njit
    def caller():
        # np.float64 value (not a pointer) in the output slot
        return sscanf(get_unicode_data_p("42"), "%d", np.float64(0.0))

    with pytest.raises(TypingError, match=r"sscanf.*intp"):
        caller()


def test_sscanf_rejects_non_intp_int_output_arg():
    """Even a smaller int (np.int32, not pointer-width) is rejected — the
    variadic slot is pointer-sized on the ABI side, and intp is the
    consistent pointer-as-int type used elsewhere in numbox.
    """
    @njit
    def caller():
        return sscanf(get_unicode_data_p("42"), "%d", np.int32(7))

    with pytest.raises(TypingError, match=r"sscanf.*intp"):
        caller()


def test_sscanf_rejects_non_intp_buf():
    """The input buffer must be intp too (a pointer to the NUL-terminated
    input bytes)."""
    @njit
    def caller(out_arr):
        # Passing a unicode_type directly instead of get_unicode_data_p
        return sscanf("42", "%d", array_data_p(out_arr))

    out = np.zeros(1, dtype=np.int32)
    with pytest.raises(TypingError, match=r"sscanf.*buf.*intp"):
        caller(out)


# ============================================================================
# Dual-mode coverage — the key contract: same source runs in both modes,
# producing equivalent output. These tests exercise that promise directly.
# ============================================================================


def _dual_kernel_printf(n, label):
    """A non-decorated kernel that we run both directly (pure Python)
    and after applying @njit. The body uses *args syntax (no tuple) and
    a string literal arg (gets auto-converted via get_unicode_data_p in
    @njit; pure Python uses % natively)."""
    printf("step %d: %s\n", n, label)
    fflush(stdout())
    return n * 2


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_dual_mode_printf_runs_same_source(capfd):
    """The printf binding meets numba's dual-mode contract: the same kernel
    source runs in plain Python and under @njit with identical observable
    output. Pure-Python uses sys.stdout.write + Python's % operator;
    @njit routes via _printf_intrinsic → libc printf. The user's *args
    syntax stays unchanged."""
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
    """fprintf(stderr(), ...) dual-mode: routes via sys.stderr in pure
    Python (address-cache lookup), via libc fprintf in @njit. Both must
    land on stderr (not stdout) — guards against transposed handles."""
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
    """snprintf dual-mode: pure-Python uses ctypes.memmove into the
    caller-allocated buffer; @njit uses libc snprintf. Both must produce
    identical bytes for a fits-in-buffer call."""
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
    """Exercises the int32→int64 widening promotion: %lld reads 8 bytes,
    x is int32 (4 bytes). Without widening, @njit would read 4 bytes of
    garbage in the high half of the variadic register slot. With widening,
    same source produces the same number in both modes."""
    printf("%lld\n", x)
    fflush(stdout())


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_dual_mode_int32_promotes_to_int64_for_lld(capfd):
    """The int64-widening promotion in _promote_for_varargs ensures
    `printf("%lld", int32_val)` works in @njit. Pure-Python's % operator
    ignores length modifiers and uses the value's natural width, so it
    works there too — dual-mode equivalence."""
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
    """The @overload's type-based str detection auto-wraps unicode_type
    args with get_unicode_data_p so libc %s sees a NUL-terminated C
    string. User code passes a raw str — no get_unicode_data_p
    ceremony — and it works in both modes."""
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
    """fprintf in pure-Python mode only supports stdio handle FILE*
    addresses (looked up in the address cache). Arbitrary FILE* (e.g.
    fopen-returned) can't be dereferenced from Python without ctypes,
    so the binding raises a clear error pointing the user at either
    @njit or Python's open()."""
    fake_fp = 0xdeadbeef
    with pytest.raises(RuntimeError, match=r"fprintf.*pure-Python.*stdout"):
        fprintf(fake_fp, "x\n")


def test_sscanf_raises_in_pure_python():
    """sscanf is @njit-only — direct call from Python raises a clear error."""
    out = np.zeros(1, dtype=np.int32)
    with pytest.raises(NotImplementedError, match=r"@njit-only"):
        sscanf(int(get_unicode_data_p("42")), "%d", int(array_data_p(out)))


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_accepts_utf8_format_literal(capfd):
    """The format string is encoded as UTF-8 at codegen time, so non-ASCII
    literals compile cleanly and render as UTF-8 bytes through libc printf.
    With the prior ASCII encoding this would have raised UnicodeEncodeError
    at numbox-compile time (when numba lowers the @njit caller)."""
    _printf_utf8(42)
    out, _ = capfd.readouterr()
    # capfd's readouterr() decodes stdout bytes as utf-8 by default, so we
    # get back the original codepoints. Compare against the str form AND
    # the underlying UTF-8 byte sequence to pin both layers.
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
    """snprintf gives us byte-level access to the written buffer, so we
    can assert the UTF-8 byte sequence directly without any decoding
    indirection (capfd is not involved)."""
    buf = np.zeros(32, dtype=np.uint8)
    rc = _snprintf_utf8(buf, 42)
    # rc semantics differ by platform (see snprintf docstring), but for a
    # message that fits comfortably in the buffer, both platforms return
    # the byte-count written (excluding NUL).
    assert rc == len(NON_ASCII_EXPECTED), (rc, len(NON_ASCII_EXPECTED))
    nul = buf.tolist().index(0)
    assert bytes(buf[:nul]) == NON_ASCII_EXPECTED, bytes(buf[:nul])


def test_fmtio_caller_survives_subprocess_round_trip(tmp_path):
    """``@njit(cache=True)`` callers of the variadic formatted-I/O bindings
    survive a process restart. Cached caller IR references the libc extern
    symbol and a deterministic UTF-8 format-string global constant —
    never a runtime address — so the warm subprocess loads the cached
    code unchanged (mtimes preserved). See
    ``assert_njit_cache_survives_subprocess_roundtrip`` in
    ``test/auxiliary_utils.py`` for the assertion contract.
    """
    assert_njit_cache_survives_subprocess_roundtrip(
        tmp_path,
        probe_source="""
            import numpy as np
            from numba import njit
            from numbox.core.bindings import snprintf
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
    """Fan-out cache-survival probe — exercises one binding from each of
    the major families in a single @njit(cache=True) caller and asserts
    each binding's expected return value, then verifies the entire
    .nbc + .nbi cache is unchanged on the warm subprocess.

    The implicit-mechanism tests (``test_fmtio_caller_survives_subprocess_round_trip``
    and ``test_proxy_caller_survives_subprocess_round_trip``) already pin
    that the @proxy + variadic-intrinsic wrappers cache cleanly at import
    time. This test adds explicit per-family CALL coverage so a binding
    bug that only surfaces when the cached IR is actually invoked in the
    warm process (rather than just loaded) would also be caught.

    Each binding check inside the probe prints one line via
    ``printf("OK <family>\\n")`` or ``printf("FAIL <family>\\n")``.
    Deterministic output, no hand-counted byte widths, no fragile
    boolean expressions involving uint8 / sign-extension. The prior
    variant of this test returned multiple int values from the @njit
    function and Python-printed them, which broke on macOS 3.14 (the
    ``1 if (eq == 0 and src[0] == 0xab) else 0`` expression evaluated
    to 0 there) and on Windows (where ``probe.write_text`` defaulted
    to cp1252 and the source's em-dash / arrow chars raised
    ``UnicodeEncodeError``). This version sidesteps both: ASCII-only
    probe source, OK/FAIL strings printed via printf, and the helper
    explicitly writes UTF-8 + sets ``PYTHONIOENCODING=utf-8``.

    Families exercised: errno, stdio handles, strerror_safe, libc
    strings (strcmp, strchr), libc memory (memset, memcpy, memcmp),
    environ (getenv), variadic I/O (snprintf, printf).
    """
    assert_njit_cache_survives_subprocess_roundtrip(
        tmp_path,
        probe_source=r"""
            import numpy as np
            from numba import njit
            from numbox.core.bindings import (
                errno_get, errno_set,
                stdout, stderr,
                strerror_safe,
                strcmp, strchr,
                memset, memcpy, memcmp,
                getenv,
                snprintf, printf, fflush,
            )
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
# Variadic arg-type and %n-rejection tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", ["printf", "fprintf", "snprintf"])
def test_njit_writer_rejects_array_arg(fn_name):
    """numpy arrays passed directly as variadic args would silently flow
    into libc as LLVM aggregates without ``_validate_writer_arg_type``;
    libc would read garbage from the next stack slot. Both the typing-
    function layer (intrinsic) and the @overload should reject."""
    src = {
        "printf": "from numbox.core.bindings import printf\n"
                  "import numpy as np\n"
                  "@njit\ndef k(a): printf('%d', a)\n",
        "fprintf": "from numbox.core.bindings import fprintf, stdout\n"
                   "import numpy as np\n"
                   "@njit\ndef k(a): fprintf(stdout(), '%d', a)\n",
        "snprintf": "from numbox.core.bindings import snprintf\n"
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
        "printf": "from numbox.core.bindings import printf\n"
                  "@njit\ndef k(c): printf('%f', c)\n",
        "fprintf": "from numbox.core.bindings import fprintf, stdout\n"
                   "@njit\ndef k(c): fprintf(stdout(), '%f', c)\n",
        "snprintf": "from numbox.core.bindings import snprintf\n"
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
        "printf": "from numbox.core.bindings import printf\n"
                  "@njit\ndef k(t): printf('%d %d', t)\n",
        "fprintf": "from numbox.core.bindings import fprintf, stdout\n"
                   "@njit\ndef k(t): fprintf(stdout(), '%d %d', t)\n",
        "snprintf": "from numbox.core.bindings import snprintf\n"
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
            "%qn", "%I32n", "%I64n"]
)
def test_njit_printf_rejects_percent_n(fmt):
    """``%n`` writes the byte-count-written-so-far through a caller pointer
    arg — memory-safety hazard, also diverges from pure-Python's `%`
    operator which raises ValueError on ``%n``. Reject at typing time."""
    src = f"""
from numbox.core.bindings import printf
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
        src = "from numbox.core.bindings import printf\n" \
              "from numbox.utils.lowlevel import array_data_p\n" \
              "@njit\ndef k(out): printf('count=%n', array_data_p(out))\n"
    elif fn == "fprintf":
        src = "from numbox.core.bindings import fprintf, stdout\n" \
              "from numbox.utils.lowlevel import array_data_p\n" \
              "@njit\ndef k(out): fprintf(stdout(), 'count=%n', array_data_p(out))\n"
    else:
        src = "from numbox.core.bindings import snprintf\n" \
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


def test_njit_sscanf_accepts_percent_n():
    """``%n`` is legitimately useful in sscanf — it returns the read
    position (how many characters were consumed so far), not a write
    through a caller pointer. Allowed."""

    @njit(cache=False)
    def parse(buf_p, n_out, pos_out):
        return sscanf(buf_p, "%d%n", array_data_p(n_out), array_data_p(pos_out))

    buf = get_unicode_data_p("12345 rest")
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
    """``%%n`` is the literal characters `%n`, not a directive — must NOT
    be rejected."""

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


def test_python_fprintf_rejects_percent_n():
    with pytest.raises(ValueError, match="%n.*not allowed"):
        fprintf(stdout(), "count=%n", 42)


def test_python_snprintf_rejects_percent_n():
    buf = np.zeros(16, dtype=np.uint8)
    with pytest.raises(ValueError, match="%n.*not allowed"):
        snprintf(array_data_p(buf), buf.size, "count=%n", 42)


def test_python_snprintf_truncation_nul_terminates():
    """Pure-Python snprintf must NUL-terminate the buffer even when output
    is truncated (C99 / POSIX semantics). Regression test: with a buffer
    of size 8 and an 11-char output, buf should hold 7 content bytes +
    NUL terminator at index 7, not 8 content bytes with no terminator."""
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
