"""A monotonic clock in nanoseconds for jitted code, on Linux, macOS and Windows.

``monotonic_ns()`` returns an ``int64``, the nanoseconds since an unspecified starting point, so only the difference
between two values is meaningful. It is an ``@intrinsic``, so only jitted code can call it. The code it generates
keeps the platform's time structure on the stack, so a call allocates no heap memory and creates no object that
numba's runtime tracks.

On Linux and macOS, the code calls libc's ``clock_gettime(CLOCK_MONOTONIC, &ts)``, where ``ts`` is a
``struct timespec`` of two ``int64`` fields, ``tv_sec`` and ``tv_nsec``. The code declares ``clock_gettime`` with
``get_or_insert_function``, and the platform's dynamic symbol lookup resolves it when the JIT links the code; libc
is always loaded globally.

On Windows, the code calls ``QueryPerformanceCounter`` from ``kernel32.dll``, which this module registers with
LLVM's symbol search through ``load_library_permanently``. The module reads the counter's frequency once, through
ctypes, when it is imported, and the generated code holds the frequency as a constant.
"""
import ctypes
import platform
import time

import llvmlite.binding as ll
from llvmlite import ir
from numba.core.cgutils import get_or_insert_function
from numba.core.types import int64
from numba.extending import intrinsic

_SYSTEM = platform.system()

_i32 = ir.IntType(32)
_i64 = ir.IntType(64)
_i32_0 = ir.Constant(_i32, 0)
_i32_1 = ir.Constant(_i32, 1)
_BILLION = ir.Constant(_i64, 1_000_000_000)


if _SYSTEM == "Windows":
    ll.load_library_permanently("kernel32.dll")

    _freq_buf = ctypes.c_int64(0)
    ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(_freq_buf))
    _QPC_FREQ = int(_freq_buf.value)
    if _QPC_FREQ <= 0:
        raise RuntimeError(
            "QueryPerformanceFrequency returned a non-positive value")
    _QPC_FREQ_CONST = ir.Constant(_i64, _QPC_FREQ)

    @intrinsic
    def monotonic_ns(typingctx):
        """Return the time of the monotonic clock in nanoseconds, from ``QueryPerformanceCounter``."""
        def codegen(context, builder, signature, arguments):
            """Emit the call of ``QueryPerformanceCounter`` and the conversion of its ticks to nanoseconds."""
            counter_ptr = builder.alloca(_i64)
            # The status that the call returns is not checked. The counter is zeroed first, so a call that fails
            # without writing it gives 0.
            builder.store(ir.Constant(_i64, 0), counter_ptr)
            fn_ty = ir.FunctionType(_i32, [_i64.as_pointer()])
            fn = get_or_insert_function(
                builder.module, fn_ty, "QueryPerformanceCounter")
            builder.call(fn, [counter_ptr])
            ticks = builder.load(counter_ptr)
            # ns = (ticks // freq) * 10**9 + (ticks % freq) * 10**9 // freq. The direct ticks * 10**9 // freq would
            # overflow an int64 once ticks passes about 9.2e9, after about 15 minutes of uptime at the usual frequency
            # of 10 MHz. In the split form, the first product overflows only after about 292 years of uptime, and the
            # second stays below freq * 10**9.
            sec = builder.sdiv(ticks, _QPC_FREQ_CONST)
            rem = builder.srem(ticks, _QPC_FREQ_CONST)
            sec_ns = builder.mul(sec, _BILLION)
            rem_ns = builder.sdiv(builder.mul(rem, _BILLION),
                                  _QPC_FREQ_CONST)
            return builder.add(sec_ns, rem_ns)
        return int64(), codegen

else:
    _CLOCK_MONOTONIC = getattr(time, "CLOCK_MONOTONIC", 1)
    _CLK_ID = ir.Constant(_i32, _CLOCK_MONOTONIC)
    _timespec_ty = ir.LiteralStructType([_i64, _i64])

    @intrinsic
    def monotonic_ns(typingctx):
        """Return the time of the monotonic clock in nanoseconds, from ``clock_gettime(CLOCK_MONOTONIC)``."""
        def codegen(context, builder, signature, arguments):
            """Emit the call of ``clock_gettime`` and the conversion of its ``timespec`` to nanoseconds."""
            ts_ptr = builder.alloca(_timespec_ty)
            # The status that the call returns is not checked. The structure is zeroed first, so a call that fails
            # without writing it gives 0.
            zero_ts = ir.Constant(
                _timespec_ty, [ir.Constant(_i64, 0), ir.Constant(_i64, 0)])
            builder.store(zero_ts, ts_ptr)
            fn_ty = ir.FunctionType(_i32, [_i32, _timespec_ty.as_pointer()])
            fn = get_or_insert_function(
                builder.module, fn_ty, "clock_gettime")
            builder.call(fn, [_CLK_ID, ts_ptr])
            sec = builder.load(builder.gep(ts_ptr, [_i32_0, _i32_0]))
            nsec = builder.load(builder.gep(ts_ptr, [_i32_0, _i32_1]))
            return builder.add(builder.mul(sec, _BILLION), nsec)
        return int64(), codegen
