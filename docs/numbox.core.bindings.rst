numbox.core.bindings
====================

Overview
++++++++

Loads dynamic libraries available in the Python environment, such as, `libc`, `libm`, and `libsqlite3`
in global symbol mode (`RTLD_GLOBAL`) via `ctypes`.
This adds global symbols (including native API) exported from those libraries to the LLVM symbol table.
These functions can then be invoked from the numba jitted code [#f1]_, complementing the suite
of numba-supported functionality.

Analogous technique can be expanded as needed for the user custom code.

.. rubric:: References

.. [#f1] See `numbsql <https://github.com/cpcloud/numbsql>`_ for previous work on jit-wrapping FFI imported functions.

Bindings module conventions
+++++++++++++++++++++++++++

Every binding in the ``numbox.core.bindings`` family (``_c``, ``_math``,
``_sqlite_conn`` / ``_stmt`` / ``_bind`` / ``_column`` / ``_exec`` /
``_blob`` / ``_hooks`` / ``_constants`` / ``_sqlite_value`` / ``_sqlite_result`` /
``_sqlite_udf``, ``_stdio``, ``_errno``,
``_strerror``, ``_fmtio``) uses
extern-symbol references via
:func:`~numbox.core.bindings.call._call_lib_func`, so the ABI dispatch is
ASLR-safe. Pointer arguments are typed as ``intp`` -- the caller is
responsible for liveness, alignment, and ownership of the underlying memory.

All wrappers are decorated with :func:`~numbox.core.proxy.proxy.proxy` --
safe to reference from a user ``@njit(cache=True)`` caller. ``proxy``
declares the callee's ``llvm_cfunc_wrapper_name`` as an extern in the
caller's module and lets llvmlite's JIT linker resolve it per process, so
cached caller IR re-resolves the function pointer correctly under ASLR.

References for individual binding semantics:

- POSIX / Linux glibc: `man7.org <https://man7.org/linux/man-pages/man3/>`_
- macOS Darwin: `Apple Open Source Libc
  <https://github.com/apple-oss-distributions/Libc>`_
- Windows UCRT: `Microsoft Learn
  <https://learn.microsoft.com/en-us/cpp/c-runtime-library/c-run-time-library-reference>`_

ABI dispatch
++++++++++++

LLVM's JIT treats ABI lowering as a frontend responsibility — it won't insert the right calling convention
for struct args/returns by itself. ``numbox.core.bindings.call._call_lib_func`` dispatches per platform and
per struct shape, using primitives from ``numbox.core.bindings.abi`` (platform identification via
``_current_platform``, struct-shape classification via ``_classify``, struct-size measurement via
``_struct_bytes``). The two ABI families that matter:

- **Windows x64** — passes aggregates >8 bytes via caller-allocated pointers and returns them via ``sret``;
  sizes 1/2/4/8 go directly in registers.
- **SysV x86-64 / AAPCS64** — pass and return ≤16-byte aggregates directly in GP registers; on SysV x86-64,
  >16-byte by-value args use a ``byval`` + ``optnone`` + ``noinline`` idiom so the LLVM optimizer doesn't
  elide the caller-side stack copy before the callee reads it.

References:

- `llvmlite#300 <https://github.com/numba/llvmlite/issues/300#issuecomment-327235846>`_
- `llvm-project#85417 <https://github.com/llvm/llvm-project/issues/85417>`_
- `Windows x64 calling convention <https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention>`_
- `AAPCS64 <https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst>`_

Stdio handles, errno, and thread-safe strerror
++++++++++++++++++++++++++++++++++++++++++++++

**Stdio handles.** ``stdout()``, ``stderr()``, and ``stdin()`` are exposed as JIT-callable functions
rather than module-level Python constants because the C library's stdio ``FILE *`` values can be either
data symbols or accessor functions:

- **Linux (glibc and musl)** — data symbols (``stdout``, ``stderr``, ``stdin`` global variables, declared in
  `<stdio.h> <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`_)
- **macOS Darwin** — data symbols (``__stdoutp``, ``__stderrp``, ``__stdinp`` — what the libc headers'
  ``stdout`` / ``stderr`` / ``stdin`` macros expand to per
  `Apple's _stdio.h <https://github.com/apple-oss-distributions/Libc/blob/main/include/_stdio.h#L218-L220>`_)
- **Windows** — accessor function (`__acrt_iob_func(0|1|2)
  <https://learn.microsoft.com/en-us/cpp/c-runtime-library/internal-crt-globals-and-functions>`_);
  UCRT-only (Windows 10+ / VS 2015+)

Both shapes are wrapped behind a uniform ``() -> intp`` interface using extern-symbol references in LLVM IR —
never literal addresses — so that ``cache=True`` remains correct under ASLR: the address is resolved at
JIT link time on each run rather than being baked into the cached object.

Example — write to stderr from inside @njit:

.. code-block:: python

    from numba import njit
    from numbox.core.bindings import stderr, fputs, fflush

    @njit(cache=True)
    def log_to_stderr(msg_p):
        fputs(msg_p, stderr())
        fflush(stderr())

**errno.** ``errno_get()`` and ``errno_set(v)`` reach the per-thread errno location on every call via the
platform's accessor function (`__errno_location
<https://man7.org/linux/man-pages/man3/errno.3.html>`_ on Linux glibc and musl,
`__error <https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/intro.2.html>`_
on Darwin, `_errno <https://learn.microsoft.com/en-us/cpp/c-runtime-library/errno-doserrno-sys-errlist-and-sys-nerr>`_
on Windows UCRT). This makes the wrappers correct under ``@njit(parallel=True)``: each ``prange`` worker
sees its own thread's errno. A Python caller observes errno set inside a normal ``@njit`` function (same
OS thread), but not errno set inside a ``prange`` worker (different OS thread).

Example — read and report errno after a syscall-style binding:

.. code-block:: python

    from numba import njit
    from numbox.core.bindings import errno_get, errno_set

    @njit(cache=True)
    def clear_then_call_and_report(do_work):
        errno_set(0)
        result = do_work()
        return result, errno_get()

**Thread-safe strerror.** ``strerror_safe(errnum, buf, buflen)`` writes the error message into a
caller-supplied buffer, returning 0 on success and a positive errno code on failure. The underlying
symbol is selected at lowering time:

- **glibc Linux** — `__xpg_strerror_r
  <https://codebrowser.dev/glibc/glibc/string/xpg-strerror.c.html>`_ (POSIX XSI form, present on glibc
  2.3.4+ which shipped in 2004)
- **musl Linux** — also ``__xpg_strerror_r``, exported as a `weak alias
  <https://git.musl-libc.org/cgit/musl/tree/src/string/strerror_r.c>`_ to musl's own ``strerror_r``
  (which is itself the POSIX form; musl never shipped the GNU char-pointer form)
- **macOS Darwin** — `strerror_r
  <https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/strerror_r.3.html>`_
  (POSIX form)
- **Windows** — `strerror_s
  <https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/strerror-s-strerror-s-wcserror-s-wcserror-s>`_
  with reordered args (buffer, size, errnum)

Other Linux libcs are not supported: on glibc the plain ``strerror_r`` symbol is the
`GNU form <https://man7.org/linux/man-pages/man3/strerror_r.3.html>`_ (returns ``char *``) and would
not match the POSIX-shaped IR this module generates. The Linux selector unconditionally picks
``__xpg_strerror_r``, which resolves correctly on both glibc and musl (the musl path goes through the
weak alias). A ``strerror_r`` fallback remains in the selector as defense-in-depth in case a future
libc drops ``__xpg_strerror_r``, but the fallback is currently unreachable on every supported libc.

The Linux selector logic is verified by an IR-inspection test (Linux-only) that monkeypatches
``ll.address_of_symbol`` to drive the fallback branch. The musl symbol layout (``strerror_r``
and ``__xpg_strerror_r`` both exported, both resolving to the same address via musl's weak
alias) is a runtime invariant the binding depends on; an Alpine-container CI canary that pins
the assumption via ``nm -D`` is a straightforward way to verify it.

Example — render the message for ``ENOENT`` (errno 2 on POSIX) into a buffer:

.. code-block:: python

    import errno
    import numpy as np
    from numba import njit
    from numbox.core.bindings import strerror_safe
    from numbox.utils.lowlevel import array_data_p

    @njit(cache=True)
    def explain(errnum, buf):
        rc = strerror_safe(errnum, array_data_p(buf), buf.size)
        return rc

    buf = np.zeros(128, dtype=np.uint8)
    rc = explain(errno.ENOENT, buf)
    msg = bytes(buf[:buf.tolist().index(0)]).decode()
    # rc == 0; msg is the (locale-dependent) string for ENOENT

Variadic formatted I/O — printf / fprintf / snprintf / sscanf
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``printf``, ``fprintf``, ``snprintf`` (and ``sscanf`` for the parsing
direction) are dual-mode bindings: each is a regular Python function with
an ``@overload`` registration that routes to a private ``@intrinsic``
codegen path when called inside ``@njit``. Same source runs unchanged
in plain Python AND under ``@njit``, matching numba's own convention
for builtins like ``print`` and ``range``.

**Call convention** — C-like ``*args`` after the format string, no tuple
wrapper at the call site::

    printf("x = %d, ratio = %.3f\n", n, ratio)
    fprintf(stderr(), "warning: %s\n", msg)
    snprintf(array_data_p(buf), buf.size, "[%d:%d]", lo, hi)
    sscanf(buf_p, "%d %lf", array_data_p(n_out), array_data_p(x_out))
    printf("no args here\n")

The internal ``@intrinsic`` still uses tuple-as-args because numba's
``@intrinsic`` typing function doesn't accept Python-level ``*args``;
the ``@overload`` impl bundles the user's ``*args`` into a tuple before
calling the intrinsic.

**Dual-mode in action:**

.. code-block:: python

    from numba import njit
    from numbox.core.bindings import printf, fflush, stdout

    def debug_kernel(x, label):
        printf("step %d: %s\n", x, label)
        fflush(stdout())
        return x * 2

    debug_kernel(7, "before")        # pure Python: writes via sys.stdout
    njit(debug_kernel)(7, "before")  # jitted: writes via libc printf
    # NUMBA_DISABLE_JIT=1 mode: also works (decorator becomes no-op)

**Format string must be a literal in @njit.** Required to embed it as an
IR global constant at typing time. A runtime-built ``unicode`` raises a
clean ``TypingError``. In pure-Python mode the format string can be any
str.

**Format string encoding: UTF-8.** Non-ASCII codepoints in the literal
are encoded as UTF-8 byte sequences and embedded into the IR global.
printf treats every non-``%`` byte as opaque pass-through, so the bytes
flow through libc to stdout / ``FILE *`` / the snprintf buffer unmodified.
Modern terminals, files, and Windows 10+ consoles all expect UTF-8.

  .. note::
     ``%-Ns`` width is byte-counted by printf in every libc, so non-ASCII
     output won't right-pad to a codepoint count. That's printf's
     contract. Pad in numba-side string formatting (``f"{s:<10}"``)
     before passing through ``%s`` if codepoint-counted widths matter.

**String args auto-conversion.** When the @overload sees an arg whose
type is ``unicode_type`` or ``Literal[str]``, it auto-wraps the arg with
``get_unicode_data_p`` so libc's ``%s`` receives a NUL-terminated C
string. User code passes a raw Python str — no ``get_unicode_data_p``
ceremony at the call site::

    printf("hello %s\n", "world")  # works in both modes; no get_unicode_data_p

The pre-conversion form (``printf("hello %s\n", get_unicode_data_p(s))``)
is still supported for backward compat or when the user has a
precomputed pointer.

**Integer promotion to 64-bit.** The @njit impl widens every integer
variadic arg to 64-bit before the libc call (``sext`` for signed, ``zext``
for unsigned, ``bool``→``int64`` via ``zext``, ``float32``→``float64``
via ``fpext``). This diverges from strict C ABI (C only promotes
sub-int-width values) but makes ``%lld`` against ``int32`` work in
@njit, matching pure-Python's ``%`` which ignores length modifiers and
uses the value's natural width. The cost is one LLVM widening
instruction per arg — free at runtime.

============   ============================
Numba type     Widened to (in varargs slot)
============   ============================
``float32``    ``float64`` (``fpext``)
``float64``    pass through
``bool``       ``int64`` (``zext``)
``int8`` / ``int16`` / ``int32`` (signed)   ``int64`` (``sext``)
``uint8`` / ``uint16`` / ``uint32`` (unsigned)   ``int64`` (``zext``)
``int64`` / ``uint64`` / ``intp``   pass through
``unicode_type`` / ``Literal[str]``   auto-converted via ``get_unicode_data_p`` → ``intp``
============   ============================

**Unsupported types raise ``TypingError`` at compile time** — numpy
arrays (``Array``), complex numbers (``Complex``), tuples
(``UniTuple``/``Tuple``), records, and anything else outside the table
above. Without that guard, an aggregate LLVM value would flow straight
into libc's variadic call and the printf would read scrambled bytes
from neighboring stack slots — silent miscompilation rather than a
clean error. Pass the field you want explicitly (``arr[0]``, ``c.real``,
``t[0]``).

**Pure-Python format-spec compatibility.** Python's ``str.__mod__`` is
printf-derived but rejects C length modifiers (``%lld``, ``%ld``,
``%lf``, ``%hd``, etc.) with ``ValueError``. The pure-Python impl
strips length modifiers via a regex before formatting (``%lld``→``%d``,
``%.3lf``→``%.3f``, etc.) so the same format string works in both modes.

  .. warning::
     ``%ld`` is the most common cross-platform footgun. On LP64 (Linux,
     macOS) ``long`` is 8 bytes; on Win64 (LLP64) it's 4 bytes. In
     @njit on Win64, ``printf("%ld", int64_val)`` truncates the high
     32 bits. Pure-Python mode hides this because Python's ``%`` ignores
     ``l``. Prefer ``%lld`` + ``int64`` for portable 8-byte width — and
     test on Windows if you have any ``%ld`` in your format strings.

**``snprintf`` truncation rc on Windows** diverges from C99/POSIX.
Linux/macOS @njit and pure-Python all return the would-have-written
count (excluding NUL); Windows @njit targets MSVCRT ``_snprintf`` which
returns ``-1`` on truncation and doesn't guarantee NUL-termination.
The portable check ``(rc < 0) or (rc >= size)`` works on every
platform / mode. (UCRT's C99-compliant ``snprintf`` is a header-only
inline over ``__stdio_common_vsnprintf`` and isn't directly linkable
in the simple C99 calling shape — declaring ``i32 @snprintf(...)`` in
LLVM IR and letting the JIT linker resolve it crashes with an access
violation on Windows.)

**``fprintf`` to non-stdio ``FILE *`` in pure Python.** The Python impl
caches the addresses of ``stdout()`` / ``stderr()`` / ``stdin()`` at
first use and routes those handles to the corresponding ``sys.*``
streams. ``fopen``-returned ``FILE *`` values can't be dereferenced from
Python without ctypes, so they raise a clear ``RuntimeError`` in
pure-Python mode (use ``open()`` + ``f.write()`` for Python-side file
I/O, or wrap the call in ``@njit``).

**``sscanf`` is @njit-only.** Pure-Python parsing is better served by
``int()``, ``float()``, or ``re``. Calling ``sscanf`` from pure Python
raises ``NotImplementedError`` with a pointer at the alternatives.

**Stdout buffering.** ``stdout`` is line-buffered to a terminal and
block-buffered when redirected. The pure-Python impl auto-flushes
``sys.stdout`` after each ``printf`` to mirror the line-buffer-to-tty
behavior; @njit users should call ``fflush(stdout())`` explicitly when
output needs to appear immediately (e.g. before a long-running compute
or under pytest's ``capfd``).

**Caching.** ``@njit(cache=True)`` callers cache cleanly across
processes: each call site emits a direct extern reference to the libc
symbol and a deterministic format-string global constant. The JIT
linker resolves the libc symbol per-process, so the cached IR is
ASLR-safe. No ``@proxy`` indirection needed — the variadic intrinsics
inline the call directly into the caller.

Example — log to stderr with `fprintf(3) <https://man7.org/linux/man-pages/man3/fprintf.3.html>`_,
dual-mode:

.. code-block:: python

    from numba import njit
    from numbox.core.bindings import fprintf, fflush, stderr

    def warn(code, msg):
        fprintf(stderr(), "warning code=%d: %s\n", code, msg)
        fflush(stderr())

    warn(7, "disk getting full")        # plain Python
    njit(warn)(7, "disk getting full")  # jitted — identical output

Example — format into a buffer with `snprintf(3) <https://man7.org/linux/man-pages/man3/snprintf.3.html>`_,
detect truncation, decode:

.. code-block:: python

    import numpy as np
    from numba import njit
    from numbox.core.bindings import snprintf
    from numbox.utils.lowlevel import array_data_p

    def fmt_range(lo, hi, buf):
        return snprintf(array_data_p(buf), buf.size, "[%d:%d]", lo, hi)

    buf = np.zeros(64, dtype=np.uint8)
    n = njit(fmt_range)(7, 11, buf)
    # Portable truncation check (works on Linux/macOS C99 snprintf
    # *and* Windows MSVCRT _snprintf — see snprintf docstring):
    truncated = (n < 0) or (n >= buf.size)
    if not truncated:
        msg = bytes(buf[:n]).decode()  # "[7:11]"

.. warning::
   ``snprintf`` truncation semantics **diverge on Windows**. POSIX / C99
   ``snprintf`` returns the would-have-written count (excluding NUL) and
   always NUL-terminates the buffer when ``size > 0``. The Windows
   binding targets MSVCRT's `_snprintf
   <https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/snprintf-snprintf-snprintf-l-snwprintf-snwprintf-l>`_
   — that returns ``-1`` on truncation and does NOT guarantee
   NUL-termination of the buffer. We attempted to bypass this by
   declaring UCRT's C99 ``snprintf`` directly, but the call crashed with
   an access violation: UCRT exports a header-inline wrapper whose
   in-process shape does not match the naive ``declare i32 @snprintf(...)``
   LLVM declaration. The portable check
   ``(rc < 0) or (rc >= size)`` works on every platform.

**Parsing direction: `sscanf <https://man7.org/linux/man-pages/man3/sscanf.3.html>`_.**
The inverse of the printf family: parse fields from a NUL-terminated input
buffer into caller-supplied output pointers. Shape differs from the writers:

- ``buf`` is an ``intp`` pointing at the input bytes (e.g. from
  ``get_unicode_data_p``).
- Variadic ``*args`` are ``intp`` *output pointers* — each one points at
  writable storage that sscanf fills based on the corresponding format
  specifier. Typically obtained via ``array_data_p`` of a 1-element numpy
  array of the right dtype.
- Returns the count of items successfully assigned (``int32``), or
  ``-1`` (``EOF``) on input failure before the first conversion.

Unlike printf-family, ``sscanf`` is **@njit-only** — pure-Python calls
raise ``NotImplementedError`` pointing the user at Python's ``int()`` /
``float()`` / ``re`` for native parsing. There is **no default-argument
promotion** (pointers don't promote) and **no string auto-conversion**
(args are output pointers, not values). The binding validates only that
every variadic arg has type ``intp``, so you can't accidentally pass an
integer value where a pointer is expected. The pointed-to storage must
still be the right size for the format spec — the binding cannot check
that:

================   =============================================
Format spec        Required output points at
================   =============================================
``%hhd``           ``int8`` (1 byte)
``%hd``            ``int16`` (2 bytes)
``%d``             ``int32`` (4 bytes)
``%ld``            ``int64`` on LP64 (Linux/macOS); ``int32`` on Win64 (LLP64) — ``long`` is 8 bytes on LP64 and 4 bytes on Win64
``%lld``           ``int64`` (8 bytes — portable across LP64 and LLP64)
``%u``             ``uint32``
``%llu``           ``uint64``
``%f``             ``float32`` (4 bytes — NOT double, ``%lf`` is for that)
``%lf``            ``float64`` (8 bytes)
``%s``             ``char`` buffer (caller responsible for size + NUL room)
``%n``             accepted in ``sscanf`` only — writes the byte-count consumed so far to the
                   corresponding output pointer. ``printf`` / ``fprintf`` / ``snprintf`` reject
                   ``%n`` at compile time (``TypingError``) — it's a memory-write hazard widely
                   flagged by static analyzers and disabled by glibc's ``_FORTIFY_SOURCE``;
                   pure-Python's ``%`` operator also rejects it, so rejecting matches dual-mode
                   behavior. Use ``snprintf`` + ``len()`` if you need byte counts in the writer
                   family. ``%%n`` (a literal ``%`` followed by ``n``) is allowed.
================   =============================================

The ``%ld`` row is the most common cross-platform footgun and the
project's own ``CLAUDE.md`` flags it explicitly. Prefer ``%lld`` with
an ``int64`` output slot when you want a portable 8-byte width.

Example — parse a "<int> <double>" pair into typed numpy slots:

.. code-block:: python

    import numpy as np
    from numba import njit
    from numbox.core.bindings import sscanf
    from numbox.utils.lowlevel import array_data_p, get_unicode_data_p

    @njit(cache=True)
    def parse_pair(text_p, n_out, x_out):
        return sscanf(text_p, "%d %lf",
                      array_data_p(n_out), array_data_p(x_out))

    n_out = np.zeros(1, dtype=np.int32)
    x_out = np.zeros(1, dtype=np.float64)
    rc = parse_pair(get_unicode_data_p("42 3.14"), n_out, x_out)
    # rc == 2; n_out[0] == 42; x_out[0] == 3.14

Modules
++++++++

numbox.core.bindings.abi
------------------------

.. automodule:: numbox.core.bindings.abi
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._c
-----------------------

.. automodule:: numbox.core.bindings._c
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._errno
---------------------------

.. automodule:: numbox.core.bindings._errno
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._stdio
---------------------------

.. automodule:: numbox.core.bindings._stdio
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._strerror
------------------------------

.. automodule:: numbox.core.bindings._strerror
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._fmtio
---------------------------

.. automodule:: numbox.core.bindings._fmtio
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._math
--------------------------

.. automodule:: numbox.core.bindings._math
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._sqlite_conn
---------------------------------

.. automodule:: numbox.core.bindings._sqlite_conn
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._sqlite_stmt
---------------------------------

.. automodule:: numbox.core.bindings._sqlite_stmt
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._sqlite_bind
---------------------------------

.. automodule:: numbox.core.bindings._sqlite_bind
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._sqlite_column
-----------------------------------

.. automodule:: numbox.core.bindings._sqlite_column
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._sqlite_exec
---------------------------------

.. automodule:: numbox.core.bindings._sqlite_exec
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._sqlite_blob
---------------------------------

.. automodule:: numbox.core.bindings._sqlite_blob
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._sqlite_hooks
----------------------------------

.. automodule:: numbox.core.bindings._sqlite_hooks
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._sqlite_constants
--------------------------------------

.. automodule:: numbox.core.bindings._sqlite_constants
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._sqlite_value
-----------------------------------

.. automodule:: numbox.core.bindings._sqlite_value
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._sqlite_result
------------------------------------

.. automodule:: numbox.core.bindings._sqlite_result
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._sqlite_udf
---------------------------------

.. automodule:: numbox.core.bindings._sqlite_udf
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._sqlite_udf_helpers
----------------------------------------

.. automodule:: numbox.core.bindings._sqlite_udf_helpers
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._sqlite_vtable
-----------------------------------

.. automodule:: numbox.core.bindings._sqlite_vtable
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.call
-------------------------

.. automodule:: numbox.core.bindings.call
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.signatures
-------------------------------

.. automodule:: numbox.core.bindings.signatures
   :members:
   :show-inheritance:
   :undoc-members:


numbox.core.bindings.utils
--------------------------

.. automodule:: numbox.core.bindings.utils
   :members:
   :show-inheritance:
   :undoc-members:
