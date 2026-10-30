"""Tests for ``numbox.utils.clock``, the monotonic clock in nanoseconds for jitted code.

The tests check that ``monotonic_ns`` returns a positive value and does not decrease between two calls. Two tests
check that the clock measures the work between its calls. The measured time of a loop doubles when the number of
iterations doubles, and the sum of the measured intervals stays within the wall-clock time of the call that contains
them.
"""
import math
import time

from numba import njit

from numbox.utils.clock import monotonic_ns


def test_returns_positive_int64():
    @njit
    def get_time():
        return monotonic_ns()

    t = get_time()
    assert t > 0


def test_monotonically_non_decreasing():
    @njit
    def get_two_times():
        t0 = monotonic_ns()
        t1 = monotonic_ns()
        return t0, t1

    t0, t1 = get_two_times()
    assert t1 >= t0


def test_linear_scaling():
    """The measured time of a loop doubles, within a ratio of 1.5 to 3.0, when the loop runs twice as many iterations.

    Each iteration reads the clock before and after a ``math.sin`` call whose result the loop returns. The clock calls
    an external C function declared without attributes, which LLVM must assume to read and write memory, so LLVM cannot
    remove the calls, merge them or move them out of the loop.
    """
    @njit
    def timed_loop(n):
        total_ns = 0
        sink = 0.0
        for i in range(n):
            t0 = monotonic_ns()
            sink += math.sin(float(i))
            t1 = monotonic_ns()
            total_ns += t1 - t0
        return total_ns, sink

    # This call compiles timed_loop, so the measured calls below do not include the compilation.
    timed_loop(100)

    # The ratio is averaged over five runs, which reduces the noise of CI virtual machines enough for these bounds.
    n_small = 10_000
    n_large = 20_000
    runs = 5
    ratio_sum = 0.0
    for _ in range(runs):
        t_small, _ = timed_loop(n_small)
        t_large, _ = timed_loop(n_large)
        assert t_small > 0, f"expected positive time, got t_small={t_small}"
        assert t_large > 0, f"expected positive time, got t_large={t_large}"
        ratio_sum += t_large / t_small

    ratio = ratio_sum / runs
    assert 1.5 <= ratio <= 3.0, (
        f"Linear scaling ratio {ratio:.3f} out of expected range [1.5, 3.0]; "
        f"averaged over {runs} runs"
    )


def test_wall_clock_consistency():
    """The sum of the clock's intervals in a loop is between 0.1 and 1.0 times the wall-clock time of the call.

    The sum cannot exceed the wall-clock time, which measures the whole call. A ratio near 0 would mean that LLVM had
    removed the clock calls or moved them next to each other.
    """
    @njit
    def timed_loop(n):
        total_ns = 0
        sink = 0.0
        for i in range(n):
            t0 = monotonic_ns()
            sink += math.sin(float(i))
            t1 = monotonic_ns()
            total_ns += t1 - t0
        return total_ns, sink

    # This call compiles timed_loop, so the measured call below does not include the compilation.
    timed_loop(100)

    # The wall clock is time.perf_counter_ns, which uses QueryPerformanceCounter on Windows in every CPython version.
    # Before CPython 3.13, time.monotonic_ns uses GetTickCount64 there, whose resolution of about 15 ms can give zero
    # for a fast loop.
    n = 50_000
    wall_start = time.perf_counter_ns()
    jit_total, _ = timed_loop(n)
    wall_total = time.perf_counter_ns() - wall_start

    assert wall_total > 0, "wall clock elapsed is zero — loop too fast for timer resolution"
    ratio = jit_total / wall_total
    assert 0.1 <= ratio <= 1.0, (
        f"JIT/wall ratio {ratio:.3f} out of expected range [0.1, 1.0]; "
        f"jit_total={jit_total}, wall_total={wall_total}"
    )
