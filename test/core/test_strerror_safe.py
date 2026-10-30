"""Tests of strerror_safe in numbox.core.bindings.strerror, called from jitted code.

The tests cover a known error number, an unknown one and a buffer too short for the message. Further tests call the
function from two threads at the same time, pass an int64 error number to the private intrinsic, and check the C
symbol that the module declares on Linux. The assertions allow for the ways in which C libraries differ. The first
test compares two messages instead of English text, so it does not depend on the locale.
"""
import errno
import threading

import numpy as np
import pytest
from numba import njit

from numbox.core.bindings.strerror import strerror_safe
from numbox.core.bindings.utils import platform_
from numbox.utils.lowlevel import array_data_p, get_str_from_p_as_int


# 99999 is far above every error number that the supported C libraries define; the largest on Linux is EHWPOISON,
# 133. Each C library therefore treats 99999 as unknown.
INVALID_ERRNO = 99999


@njit(cache=True)
def _describe(errnum, buf, buflen):
    """Call strerror_safe for errnum with the buffer buf, and return its result and the buffer's address."""
    buf_p = array_data_p(buf)
    rc = strerror_safe(errnum, buf_p, buflen)
    return rc, buf_p


def _strerror(errnum, buflen=128):
    """Return strerror_safe's result for errnum and the message that it writes into a new buffer of buflen bytes."""
    buf = np.zeros(buflen, dtype=np.uint8)
    rc, buf_p = _describe(errnum, buf, buf.size)
    msg = get_str_from_p_as_int(buf_p)
    return rc, msg


def test_strerror_safe_enoent_roundtrip():
    """Check that ENOENT and EACCES have distinct messages and that neither message is a fallback.

    The test does not set LANG or LC_MESSAGES, so it does not compare the text with the English "No such file or
    directory". It checks properties that hold in every locale: both calls return 0, both messages are non-empty, the
    two messages differ, and neither starts with "unknown error".
    """
    rc_enoent, msg_enoent = _strerror(errno.ENOENT)
    rc_eacces, msg_eacces = _strerror(errno.EACCES)
    assert rc_enoent == 0, f"ENOENT roundtrip rc={rc_enoent}"
    assert rc_eacces == 0, f"EACCES roundtrip rc={rc_eacces}"
    assert msg_enoent, "ENOENT produced empty message"
    assert msg_eacces, "EACCES produced empty message"
    # Two error numbers have different messages in every locale, so equal messages would mean that both calls wrote
    # the same fallback text.
    assert msg_enoent != msg_eacces, (
        f"ENOENT and EACCES produced identical messages "
        f"({msg_enoent!r}) — likely both fell through to a fallback"
    )
    # glibc writes "Unknown error N" for a number that it does not define, and macOS writes "Unknown error: N". The
    # check refuses a message that starts with "unknown error" in any letter case.
    for errnum, msg in [(errno.ENOENT, msg_enoent), (errno.EACCES, msg_eacces)]:
        assert not msg.lower().startswith("unknown error"), (
            f"errno {errnum} produced fallback message {msg!r}"
        )


def test_strerror_safe_invalid_errno_uses_fallback():
    """Check that strerror_safe reports an unknown error number, through its result or through its message.

    The C libraries differ for an error number that they do not define. glibc returns ``EINVAL`` and writes
    "Unknown error N", and macOS returns ``EINVAL`` and writes "Unknown error: N". musl returns 0 and writes
    "No error information". Microsoft does not document the result of ``strerror_s`` for such a number. The test
    therefore accepts a nonzero result or a message that contains "unknown", "unrecognized" or "no error information",
    and it checks that the message is not empty. The ``strerror_safe`` docstring states the glibc behavior.
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
    """Check that strerror_safe ends the text with a NUL inside a 2-byte buffer, which is too short for the message."""
    # The buffer starts as 0xFF bytes, so only strerror_safe can have written a NUL to its last byte. A buffer of
    # zeros would pass the check if the function wrote nothing.
    buf = np.full(2, 0xFF, dtype=np.uint8)
    rc, _ = _describe(errno.ENOENT, buf, buf.size)
    # POSIX's strerror_r returns ERANGE for a buffer too short for the message. Microsoft documents that strerror_s
    # truncates the message and always ends the buffer with a NUL, but not what it returns then. The test therefore
    # checks only the last byte. It does not read the text with get_str_from_p_as_int, which would read past the
    # buffer if the function had written no NUL.
    assert buf[-1] == 0, f"buffer not NUL-terminated (rc={rc})"


def test_strerror_safe_two_threads_no_contamination():
    """Check that two threads that call strerror_safe at once, for ENOENT and EACCES, receive different messages.

    A barrier releases both threads together. strerror_safe writes into the caller's buffer and keeps no buffer of its
    own, unlike libc's ``strerror``, whose result a later call may overwrite.
    """
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
    """Check that the intrinsic ``_strerror_safe`` refuses an ``int64`` error number when numba compiles the call.

    The C functions take the error number as a 32-bit ``int``. ``strerror_safe`` converts its argument to ``int32``
    before it calls the intrinsic, so only a direct call of the intrinsic reaches this ``TypingError``.
    """
    from numba.core.errors import TypingError
    from numbox.core.bindings.strerror import _strerror_safe

    @njit
    def caller(errnum, buf, buflen):
        # errnum is an int64, and the call passes it to the intrinsic without conversion.
        return _strerror_safe(errnum, buf, buflen)

    buf = np.zeros(64, dtype=np.uint8)
    with pytest.raises(TypingError, match=r"_strerror_safe.*int32"):
        caller(np.int64(2), array_data_p(buf), buf.size)


@pytest.mark.skipif(platform_ != "Linux", reason="glibc-only IR-inspection probe")
def test_strerror_safe_ir_uses_strerror_r_when_xpg_absent(monkeypatch):
    """Check that the module declares ``strerror_r`` on Linux when LLVM finds no ``__xpg_strerror_r``.

    The test makes ``ll.address_of_symbol`` return None for ``__xpg_strerror_r`` and reads the LLVM IR that
    ``_render_ir_for_probe`` returns. It does not call the function, because glibc's plain ``strerror_r`` is the GNU
    form, which returns a ``char *`` and does not match the POSIX-shaped declaration. glibc and musl both export
    ``__xpg_strerror_r``, so the fallback applies only to a C library without that symbol.
    """
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


def _libc_is_glibc():
    import platform
    return platform.libc_ver()[0] == "glibc"


@pytest.mark.skipif(not _libc_is_glibc(), reason="the return codes the docstring states are glibc's")
def test_strerror_safe_returns_glibc_error_codes():
    """With glibc, an unknown error number returns EINVAL and a buffer too short for the message returns ERANGE."""
    rc, msg = _strerror(INVALID_ERRNO)
    assert rc == errno.EINVAL, f"rc={rc}, msg={msg!r}"
    assert msg == f"Unknown error {INVALID_ERRNO}", msg
    buf = np.full(2, 0xFF, dtype=np.uint8)
    rc, _ = _describe(errno.ENOENT, buf, buf.size)
    assert rc == errno.ERANGE, f"rc={rc}"
