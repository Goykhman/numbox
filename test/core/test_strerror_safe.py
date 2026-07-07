import errno
import threading

import numpy as np
import pytest
from numba import njit

from numbox.core.bindings.strerror import strerror_safe
from numbox.core.bindings.utils import platform_
from numbox.utils.lowlevel import array_data_p, get_str_from_p_as_int


# Invalid errno value far outside any libc's mapped range (POSIX errno is
# typically <200; glibc reserves through ~133 for application use).
INVALID_ERRNO = 99999


@njit(cache=True)
def _describe(errnum, buf, buflen):
    buf_p = array_data_p(buf)
    rc = strerror_safe(errnum, buf_p, buflen)
    return rc, buf_p


def _strerror(errnum, buflen=128):
    """Helper: return (rc, msg) for a single errno."""
    buf = np.zeros(buflen, dtype=np.uint8)
    rc, buf_p = _describe(errnum, buf, buf.size)
    msg = get_str_from_p_as_int(buf_p)
    return rc, msg


def test_strerror_safe_enoent_roundtrip():
    """Known errno produces a non-empty, non-fallback message.

    Locale-tolerant: we don't pin LANG/LC_MESSAGES, so the exact English
    "No such file or directory" string is not asserted. Instead we verify
    that ENOENT and EACCES produce *different* messages — a property that
    holds in every locale (no locale collapses distinct errno values into
    the same translation) — and that neither matches the libc's
    "Unknown error N" fallback shape that would indicate the errno wasn't
    recognized.
    """
    rc_enoent, msg_enoent = _strerror(errno.ENOENT)
    rc_eacces, msg_eacces = _strerror(errno.EACCES)
    assert rc_enoent == 0, f"ENOENT roundtrip rc={rc_enoent}"
    assert rc_eacces == 0, f"EACCES roundtrip rc={rc_eacces}"
    assert msg_enoent, "ENOENT produced empty message"
    assert msg_eacces, "EACCES produced empty message"
    # Distinct errnos must produce distinct messages on any sane locale.
    assert msg_enoent != msg_eacces, (
        f"ENOENT and EACCES produced identical messages "
        f"({msg_enoent!r}) — likely both fell through to a fallback"
    )
    # Neither message should look like the unrecognized-errno fallback.
    # glibc renders "Unknown error N"; musl uses "No error information";
    # Windows uses "Unknown error". Reject any "unknown error" prefix in
    # any case.
    for errnum, msg in [(errno.ENOENT, msg_enoent), (errno.EACCES, msg_eacces)]:
        assert not msg.lower().startswith("unknown error"), (
            f"errno {errnum} produced fallback message {msg!r}"
        )


def test_strerror_safe_invalid_errno_uses_fallback():
    """An unrecognized errno produces a fallback message (and either rc=0 or
    a positive errno indicator), with the buffer still NUL-terminated.

    Per the POSIX strerror_r spec and Windows strerror_s docs, exact rc
    behavior on unknown errno varies across libcs:

      - glibc: writes "Unknown error N", returns 0
      - musl:  writes "No error information", returns EINVAL
      - macOS: writes "Unknown error: N", returns 0
      - Windows: returns EINVAL, writes a fallback

    The test asserts the *portable* contract: the buffer is non-empty,
    NUL-terminated, and either (a) the rc indicates an error, or (b) the
    message is a recognized "unknown/unrecognized" fallback shape. This
    pins down the EINVAL-or-fallback path that the strerror_safe docstring
    promises without overfitting to one libc.
    """
    rc, msg = _strerror(INVALID_ERRNO)
    assert msg, "invalid errno produced empty message"
    looks_like_fallback = (
        "unknown" in msg.lower()
        or "unrecognized" in msg.lower()
        or "no error information" in msg.lower()
    )
    assert rc != 0 or looks_like_fallback, (
        f"invalid errno {INVALID_ERRNO} produced rc=0 and recognized "
        f"message {msg!r} — strerror_safe failed to flag it as unknown"
    )


def test_strerror_safe_short_buffer():
    # Pre-fill with 0xFF so the trailing-NUL check actually probes the
    # implementation: a zero-initialized buffer would pass the assertion
    # even if strerror_safe wrote nothing.
    buf = np.full(2, 0xFF, dtype=np.uint8)
    rc, _ = _describe(errno.ENOENT, buf, buf.size)
    # POSIX strerror_r returns ERANGE on short buffer; Windows strerror_s may
    # truncate-and-succeed instead. The portable contract is NUL-termination
    # within the caller's buffer — verify that directly rather than reading
    # the buffer via a NUL-scanning helper (which could read past the
    # allocation if the implementation didn't NUL-terminate).
    assert buf[-1] == 0, f"buffer not NUL-terminated (rc={rc})"


def test_strerror_safe_two_threads_no_contamination():
    results = {}
    barrier = threading.Barrier(2)

    def worker(tid, errnum):
        buf = np.zeros(128, dtype=np.uint8)
        barrier.wait()
        rc, buf_p = _describe(errnum, buf, buf.size)
        msg = get_str_from_p_as_int(buf_p)
        results[tid] = (rc, msg)

    t0 = threading.Thread(target=worker, args=(0, errno.ENOENT))
    t1 = threading.Thread(target=worker, args=(1, errno.EACCES))
    t0.start()
    t1.start()
    t0.join()
    t1.join()
    assert results[0][0] == 0
    assert results[1][0] == 0
    assert results[0][1] != results[1][1]


def test_strerror_safe_rejects_int64_errnum():
    """The private _strerror_safe intrinsic requires int32 errnum at typing
    time. The user-facing strerror_safe wrapper casts to int32 first, so
    this guard only fires on direct misuse.
    """
    from numba.core.errors import TypingError
    from numbox.core.bindings.strerror import _strerror_safe

    @njit
    def caller(errnum, buf, buflen):
        # errnum arrives as int64 (Python default); no cast — should fail
        return _strerror_safe(errnum, buf, buflen)

    buf = np.zeros(64, dtype=np.uint8)
    with pytest.raises(TypingError, match=r"_strerror_safe.*int32"):
        caller(np.int64(2), array_data_p(buf), buf.size)


@pytest.mark.skipif(platform_ != "Linux", reason="glibc-only IR-inspection probe")
def test_strerror_safe_ir_uses_strerror_r_when_xpg_absent(monkeypatch):
    import llvmlite.binding as ll
    from numbox.core.bindings import strerror as strerror_mod

    original = ll.address_of_symbol

    def fake(name):
        if name == "__xpg_strerror_r":
            return None
        return original(name)

    monkeypatch.setattr(ll, "address_of_symbol", fake)
    ir_text = strerror_mod._render_ir_for_probe()
    assert "strerror_r" in ir_text
    assert "__xpg_strerror_r" not in ir_text
