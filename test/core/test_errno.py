"""Tests of errno_get and errno_set in numbox.core.bindings.errno, called from jitted code.

The tests cover a set followed by a get in one function, two threads that each set and read their own value, and a
set and a get in each iteration of a ``prange`` loop. The subprocess tests in ``test_proxy.py`` and ``test_fmtio.py``
call the two functions from a cached caller in a second process.
"""
import threading
import numpy as np
from numba import njit, prange

from numbox.core.bindings.errno import errno_get, errno_set


@njit(cache=True)
def _errno_set(v):
    errno_set(v)


@njit(cache=True)
def _errno_get():
    return errno_get()


def test_errno_set_get_roundtrip():
    """Check that errno_get returns the value that errno_set stored, for several values, in one jitted function."""
    @njit(cache=True)
    def rt(v):
        errno_set(v)
        return errno_get()
    for v in (0, 1, 2, 11, 13, 42):
        assert rt(v) == v


def test_errno_per_thread_isolation():
    """Check that two threads that set ``errno`` to different values each read back their own value.

    Each thread calls ``set_busy_get`` 200 times with its own sentinel. ``set_busy_get`` sets ``errno`` and then reads
    it, with the GIL released and with no system call between the two calls. The C library stores a separate
    ``errno`` for each thread, so each read returns the thread's own sentinel. A process-wide ``errno`` would return
    the other sentinel whenever the other thread set it between the two calls. The source places a loop between the
    set and the get, but LLVM removes it, as the comment in ``set_busy_get`` explains. The two calls therefore run
    consecutively, and the test detects a shared ``errno`` only if the other thread sets it in that short interval.

    The test puts no lock, barrier or event between the set and the get, because a wait on one can change ``errno``.
    On Linux, a timed wait on a ``threading.Event`` or a ``threading.Lock`` sets ``errno`` to ``ETIMEDOUT``, so a
    read after the wait would return that value instead of the sentinel.
    """
    SENTINEL_A = 0xABCD
    SENTINEL_B = 0xBEEF
    N_TRIALS = 200
    BUSY = 200_000

    @njit(nogil=True, cache=True)
    def set_busy_get(v, busy):
        errno_set(v)
        s = np.int64(0)
        for i in range(busy):
            s += (i * 7) & 0xff
        # LLVM folds s & 0 to 0, so nothing uses the result of the loop, and LLVM removes the loop as dead code. The
        # compiled function calls errno_set and errno_get consecutively, for any value of busy.
        return errno_get() + (s & 0)

    bad = {"A": 0, "B": 0, "first_bad_A": None, "first_bad_B": None}

    def worker(tid, sentinel):
        for _ in range(N_TRIALS):
            v = set_busy_get(sentinel, BUSY)
            if v != sentinel:
                bad[tid] += 1
                if bad[f"first_bad_{tid}"] is None:
                    bad[f"first_bad_{tid}"] = v

    t_a = threading.Thread(target=worker, args=("A", SENTINEL_A))
    t_b = threading.Thread(target=worker, args=("B", SENTINEL_B))
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    assert bad["A"] == 0, (
        f"thread A observed wrong errno in {bad['A']}/{N_TRIALS} trials "
        f"(first wrong value: {bad['first_bad_A']!r}, expected "
        f"{SENTINEL_A:#x}); errno is not thread-isolated"
    )
    assert bad["B"] == 0, (
        f"thread B observed wrong errno in {bad['B']}/{N_TRIALS} trials "
        f"(first wrong value: {bad['first_bad_B']!r}, expected "
        f"{SENTINEL_B:#x}); errno is not thread-isolated"
    )


def test_errno_set_get_observable_in_parallel_iterations():
    """Check that each iteration of a ``prange`` loop reads back the ``errno`` value that it set.

    The bindings page instructs a caller to read ``errno`` in the iteration that made the C call. This test checks that
    a read in an iteration returns the value that the same iteration set. The loop calls the proxied ``errno_set`` and
    ``errno_get`` by name, so the compiled loop contains no call of ``__errno_location`` that LLVM could move out of an
    iteration. The test does not show that ``errno`` is per thread: an iteration runs its set and its get on one
    thread, and a process-wide ``errno`` would differ only if another thread set it between them.
    ``test_errno_per_thread_isolation`` checks that each thread has its own ``errno``.
    """
    @njit(parallel=True, cache=True)
    def f(n):
        out = np.zeros(n, dtype=np.int32)
        for i in prange(n):
            errno_set(np.int64(i))
            out[i] = errno_get()
        return out
    n = 256
    got = f(n)
    assert (got == np.arange(n, dtype=np.int32)).all()
