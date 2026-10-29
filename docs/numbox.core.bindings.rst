numbox.core.bindings
====================

Overview
++++++++

A binding lets jitted code call a C function. It is a numbox function with the C function's name and arguments, and a
jitted call of it runs the C function. numbox has bindings for the C standard library, the C math library and SQLite.
They include functions that `numba's own support <https://numba.readthedocs.io/en/stable/reference/pysupported.html>`_
lacks: files through ``fopen`` and ``fwrite``, raw memory through ``memcpy``, formatted output to C streams and buffers
through ``printf`` and ``snprintf``, math functions such as ``remainder``, and the SQLite API. A jitted function that
calls bindings can be cached with ``cache=True``, and it keeps working when a later process loads it from the cache. Its
compiled code refers to each binding and each C function by name, never by address, and numbox loads the libraries in
that process as well, so that LLVM finds the names again.

numbox loads the dynamic libraries that it binds, such as ``libc``, ``libm`` and ``libsqlite3``, through ``ctypes``. On
Linux and macOS, ``ctypes.util.find_library`` finds each library by its short name in the system's library search, not
in the Python installation. On Windows, numbox loads the C runtime ``msvcrt.dll`` for libc and libm, and the
``sqlite3.dll`` that comes with Python. On Linux and macOS, numbox opens the libraries in global symbol mode
(``RTLD_GLOBAL``, see `dlopen(3) <https://man7.org/linux/man-pages/man3/dlopen.3.html>`_). Global mode makes the symbols
that the libraries export, their native C API, visible to LLVM, which finds a symbol by name when it links compiled code
into the process. Jitted code can then call these C functions [#f1]_, which complement the functions that numba supports
itself.

The same technique serves for a C library of your own. Load the library with
:func:`~numbox.core.bindings.utils.load_lib` or :func:`~numbox.core.bindings.utils.load_lib_path`, declare each function
with numba's ``ExternalFunction``, and call it from a function decorated with :func:`~numbox.core.proxy.proxy.proxy`.
The ``proxy`` decorator compiles the function once, when its module is imported, and every jitted caller then calls that
one compiled copy by name. Use :func:`~numbox.core.proxy.proxy.proxy_if_available` instead for a symbol that some
versions of the library lack. The proxy page shows such a binding in :ref:`its section on optional C symbols
<proxy_optional_c_symbols>`.

.. rubric:: References

.. [#f1] See `numbsql <https://github.com/cpcloud/numbsql>`_ for earlier work on calling C functions from jitted code
   through a foreign function interface.

Calling a binding
+++++++++++++++++

Import a binding from its module and call it in jitted code:

.. code-block:: python

    from numba import njit

    from numbox.core.bindings.libc import strlen
    from numbox.core.bindings.libm import remainder
    from numbox.utils.lowlevel import get_unicode_data_p


    @njit(cache=True)
    def demo(word):
        return remainder(7.0, 2.0), strlen(get_unicode_data_p(word))


    print(demo("numbox"))   # (-1.0, 6)

C's ``remainder`` rounds the quotient 3.5 to the nearest even integer, 4, and returns ``7.0 - 4 * 2.0``, which is
``-1.0``. ``strlen`` takes a pointer, passed as an ``intp`` integer, and
:func:`~numbox.utils.lowlevel.get_unicode_data_p` returns the address of a string's characters. C reads those characters
as text only for an ASCII string, as the section `Pointers and memory`_ explains. Python code can call the same
bindings, and ``remainder(7.0, 2.0)`` returns ``-1.0`` there too.

Bindings module conventions
+++++++++++++++++++++++++++

numbox defines its bindings in these modules:

- :mod:`numbox.core.bindings.libc`: string, memory, file and environment functions of the C standard library, such
  as ``strlen``, ``memcpy``, ``fopen`` and ``getenv``;
- :mod:`numbox.core.bindings.libm`: the functions of the C math library, on ``float64``;
- :mod:`numbox.core.bindings.stdio`: the standard streams;
- :mod:`numbox.core.bindings.errno`: reading and setting ``errno``;
- :mod:`numbox.core.bindings.strerror`: a thread-safe message for an error number;
- :mod:`numbox.core.bindings.fmtio`: ``printf``, ``fprintf``, ``snprintf`` and ``sscanf``;
- the subpackage ``numbox.core.bindings.sqlite``: the SQLite API, described on
  :doc:`its own page <numbox.core.bindings.sqlite>`.

The remaining modules are internal, so you seldom call them directly:

- :mod:`numbox.core.bindings.call`: ``_call_lib_func``, the internal helper through which the libc, libm and SQLite
  bindings call C, described in `its source
  <https://github.com/Goykhman/numbox/blob/main/numbox/core/bindings/call.py>`_;
- :mod:`numbox.core.bindings.abi`: the rules by which the helper passes values on each platform;
- :mod:`numbox.core.bindings.signatures`: the table of C signatures;
- :mod:`numbox.core.bindings.utils`: :func:`~numbox.core.bindings.utils.load_lib`,
  :func:`~numbox.core.bindings.utils.load_lib_path` and helpers for the binding modules.

The helper knows the calling conventions of x86-64 on Linux, macOS and Windows, and of 64-bit Arm on Linux and macOS. On
another machine, the first import of libc, libm or a SQLite module fails with a ``RuntimeError`` that names the
platform, because the import compiles the bindings through the helper.

A binding in libc, libm, fmtio or the SQLite subpackage has its C function's name and takes the C function's arguments
in their C order. The C function's manual page is therefore its reference: the `Linux man pages
<https://man7.org/linux/man-pages/man3/>`_, `Apple's Libc sources <https://github.com/apple-oss-distributions/Libc>`_
for macOS, and `Microsoft's C runtime reference
<https://learn.microsoft.com/en-us/cpp/c-runtime-library/c-run-time-library-reference>`_ for Windows. libm's ``round``
and ``pow`` are C's functions, not Python's: C's ``round(2.5)`` is ``3.0``, and Python's is ``2``. Import the module, as
in ``from numbox.core.bindings import libm``, and call ``libm.round``, so that the name ``round`` keeps its Python
meaning.

Every binding outside fmtio is a proxied function, made with the ``proxy`` decorator that the Overview describes. numbox
compiles it once, when its module is imported, and a jitted caller calls it by name instead of compiling a copy of its
code into each caller. A caller's own compiled code therefore stays small, and a cached caller works in a later process.
The drawback is that each call remains a function call, because LLVM cannot inline it. Python code can call a proxied
binding as well.

A binding's ``.as_func`` is a first-class function value for the binding's signature, and a jitted function can take it
as an argument, as the proxy page's section on :ref:`passing the function as a value <proxy_passing_as_value>` shows.
Pass ``.as_func``, not the binding itself. The binding is a numba dispatcher, the object that ``@njit`` returns, and its
numba type names the address of the function it wraps, so a cached function that receives it compiles again on every
run.

numba caches the bindings under numbox's jit options, the numba options that the ``NUMBOX_JIT_OPTIONS`` environment
variable sets (:doc:`numbox.core.configurations`). A later import loads the bindings from the cache unless those options
disable caching. numbox also disables caching, with a warning, where numba can write no cache. The package
``numbox.core.bindings`` imports none of its modules, so a module you do not import is never compiled.

Cached callers of a binding keep working after a numbox upgrade from one release to another. The bindings compile again,
because their cache key holds numbox's version, as :ref:`after you upgrade numbox <cache_key_after_upgrade>` explains.
numbox repairs each cached caller of a binding whose code the new release changed, at the caller's next run and with one
warning, as the proxy page says under :ref:`After you upgrade numbox <proxy_after_upgrade>`. The one case that repair
does not cover is a caller that holds a binding's ``.as_func`` in a module-level variable. Delete that caller's cache
files, as :ref:`holding the function as a module-level constant <proxy_module_level_constant>` explains.

A few SQLite bindings name symbols that some builds of SQLite lack, such as ``sqlite3_changes64``, which SQLite added in
version 3.37. Where the symbol is missing, the binding is a stub, and a jitted caller fails to compile, as :ref:`the
proxy page's section on optional C symbols <proxy_optional_c_symbols>` explains. fmtio's functions are not proxied:
numba compiles each call of them into its caller, and the section `Caching callers of the formatted I/O functions`_ says
how that changes a cached caller's behavior after an upgrade.

The bindings in libc, libm and the SQLite subpackage take their signatures from one table, the dictionary ``signatures``
in `numbox/core/bindings/signatures.py
<https://github.com/Goykhman/numbox/blob/main/numbox/core/bindings/signatures.py>`_, keyed by C name. ``strchr``, for
example, is ``intp(intp, int32)``: it takes a pointer and an ``int32`` and returns a pointer. The table writes C's
``int`` as ``int32``, ``long long`` as ``int64``, ``double`` as ``float64``, and ``size_t`` and every pointer as
``intp``. It holds no C ``long``, which has 64 bits on Linux and macOS and 32 bits on Windows; when you declare a C
function that takes a ``long``, use ``int64`` on Linux and macOS and ``int32`` on Windows. A jitted caller may pass
another numeric type, and numba converts it to the table's type without a warning. For example, C receives ``65`` when
the caller passes ``65.7`` or ``2**40 + 65`` for an ``int32``. The stdio, errno and strerror modules declare their
signatures in their own code.

Pointers and memory
+++++++++++++++++++

A pointer is an ``intp`` integer, and numbox does not check it. The caller keeps the memory alive while C uses it, and
makes it large enough, and aligned, for what C reads or writes. :mod:`numbox.utils.lowlevel` converts between Python
objects and pointers:

- :func:`~numbox.utils.lowlevel.array_data_p` returns the address of a NumPy array's data;
- :func:`~numbox.utils.lowlevel.get_unicode_data_p` returns the address of a string's characters;
- :func:`~numbox.utils.lowlevel.get_str_from_p_as_int` copies a NUL-terminated C string into a ``str``, one
  character per byte.

In jitted code, numba frees an array or a string as soon as the code no longer refers to it, and an address taken from
the object does not count as a reference. A temporary, such as the result of ``a + b`` in
``strlen(get_unicode_data_p(a + b))``, is therefore freed before ``strlen`` reads it. numba frees a named array in the
same way when ``array_data_p(x)`` is the last place that names ``x``. C then reads freed memory: a small object may
still hold its old bytes, and a large one ends the process with a segmentation fault. Name the object and refer to it
after the C call, for example by reading its length, or pass it in from the caller, as the examples on this page do. A
string literal or a module-level string is safe: numba holds it as a constant and never frees it, so
``get_unicode_data_p("r")`` inside a call works.

Called from Python, ``get_unicode_data_p`` returns an address inside the ``str`` object itself, so keep the string alive
while C uses it.

C reads a string's characters as text only if the string is ASCII. numba and CPython store each string at one, two or
four bytes per character, a width set by its widest character, and never as UTF-8. For other text, encode the string as
UTF-8 in Python, add a NUL, and pass the address of the bytes as a ``uint8`` array:

.. code-block:: python

    import numpy as np
    from numba import njit

    from numbox.core.bindings.libc import strlen
    from numbox.utils.lowlevel import array_data_p


    @njit(cache=True)
    def byte_count(text):
        return strlen(array_data_p(text))


    text = np.frombuffer("caf\u00e9".encode("utf-8") + b"\0", dtype=np.uint8)
    print(byte_count(text))   # 5: four characters, five UTF-8 bytes

:func:`~numbox.utils.cstrings.c_string` builds the same bytes from Python and keeps them alive for the length of a
``with`` block.

Passing a struct
++++++++++++++++

numbox's bindings pass only numbers and pointers. Store a struct in a one-element NumPy structured array, and pass
``array_data_p`` of the array to a C function that takes a pointer to the struct. Build the dtype with ``align=True``,
so that NumPy places the fields where a C compiler does: ``np.dtype([("x", np.int32), ("y", np.float64)], align=True)``
puts ``y`` at offset 8, as C does, and without ``align=True`` NumPy puts it at offset 4.

A C function that takes a struct by value is harder to call. numba's ``ExternalFunction`` cannot pass a struct by value
at all: a jitted call of a function declared with a tuple argument fails to compile with an ``AssertionError``. Passing
one through numbox's internal helper and its table of signatures is a method specific to numbox. That method is fragile:
a struct written as a numba tuple reaches C intact, on x86-64 Linux only, in these four cases:

- it has a single field;
- it is larger than 16 bytes;
- it is exactly 16 bytes, with one field in each 8-byte half;
- it is exactly 16 bytes, with two or more fields in a half, and neither half is made only of floating-point fields.

``(int32, int32, int64)`` matches the last case and reaches C intact. ``(int32, int32, float64)`` matches no case,
because its upper half holds only a ``float64``. C reads a wrong value for its second ``int32``. A struct smaller than
16 bytes with two or more fields, such as the 8-byte ``div_t`` that C's ``div`` returns, is corrupted as well.

A numba record cannot be passed by value in either direction. Passed as an argument, it reaches C as the address of its
data instead of its fields, and a record result of 16 bytes or less ends the process with a segmentation fault.

The helper also refuses some calls when numba compiles them, with a ``TypingError`` whose message contains the text
below:

- ``is unavailable in the LLVM context``: no loaded library exports the symbol; load its library with ``load_lib``
  first;
- ``Undefined signature for``: the table has no entry for the name;
- ``Record returns >16 bytes are not yet supported``: return a tuple of the same fields instead;
- ``has no fixed bitwidth``: a tuple field, such as a ``voidptr``, has no fixed size; pass the pointer as an ``intp``.

Loading a library
+++++++++++++++++

You need :func:`~numbox.core.bindings.utils.load_lib` only to bind a library of your own, because each numbox binding
module calls it for its own library when imported. ``load_lib`` takes a short name, such as ``"c"``, ``"m"`` or
``"sqlite3"``, and returns the library's handle, an ordinary ``ctypes.CDLL``. The handle stays loaded for the lifetime
of the process. A second call with the same name returns the same handle. On Linux and macOS, ``load_lib`` finds the
file with ``ctypes.util.find_library``. That function uses the system's library search and does not look in the Python
installation, as the `ctypes documentation <https://docs.python.org/3/library/ctypes.html#finding-shared-libraries>`_
describes. It then opens the file in global symbol mode. A plain ``ctypes.CDLL`` opens a library in local mode instead,
where its symbols are not visible to LLVM. A jitted call of one of its functions through ``ExternalFunction`` then ends
the process with a segmentation fault.

When ``find_library`` finds no file, ``load_lib`` raises ``RuntimeError: Could not find shared library for <name>``.
Pass the file's path to :func:`~numbox.core.bindings.utils.load_lib_path` instead, which opens the file in the same mode
and returns a new handle on each call. On Windows, ``"c"`` and ``"m"`` load the C runtime ``msvcrt.dll``. For any other
name, ``load_lib`` prefers the DLL bundled with the Python installation to one found on the ``PATH``, and for
``"sqlite3"`` it takes only the bundled one.

On Linux, every libm binding runs the system's libm. On Windows, 26 of them, such as ``cos``, ``pow``, ``sqrt`` and
``fmod``, run CPython's C runtime instead of the ``msvcrt.dll`` that numbox loads for ``"m"``. numba registers those 26
names for its own math support, and LLVM looks up numba's registration before any library.

When numba's JIT is disabled
++++++++++++++++++++++++++++

Under ``NUMBA_DISABLE_JIT=1``, numba compiles nothing, and importing any module of proxied bindings fails with
``AttributeError: 'function' object has no attribute 'get_compile_result'``. The message means that ``proxy`` needs
numba's compiler, which the setting disables; unset the variable to use these bindings. fmtio still imports, and
``printf`` runs under Python. ``fprintf`` fails with the same ``AttributeError``, because it imports the stdio bindings.
``snprintf`` writes through a ``ctypes`` call that accepts only a Python ``int`` as the address. With numba's JIT
disabled, ``array_data_p`` returns a NumPy integer, so pass ``int(array_data_p(buf))``.

Writing to stdout and stderr
++++++++++++++++++++++++++++

C functions that work on a stream, such as ``fputs``, ``fflush`` and ``fprintf``, take the stream's ``FILE *``.
:func:`~numbox.core.bindings.stdio.stdout`, :func:`~numbox.core.bindings.stdio.stderr` and
:func:`~numbox.core.bindings.stdio.stdin` return the current process's three standard streams as ``intp`` values, in
jitted code and under Python. They are functions, not constants, because a cached caller's compiled code must not
contain an address from the process that compiled it. A stream's address changes from one process to the next, so each
call reads the address in the running process. Here jitted code writes one line to each stream:

.. code-block:: python

    from numba import njit

    from numbox.core.bindings.libc import fflush, fputs
    from numbox.core.bindings.stdio import stderr, stdout
    from numbox.utils.lowlevel import get_unicode_data_p


    @njit(cache=True)
    def report(line, warning):
        fputs(get_unicode_data_p(line), stdout())
        fflush(stdout())
        fputs(get_unicode_data_p(warning), stderr())


    report("42 rows written\n", "2 rows skipped\n")

C keeps a buffer for its stdout, and Python keeps another for ``sys.stdout``. When stdout is a file or a pipe, each
writes its buffer only when the buffer fills, when the program flushes it, or when the process exits. Lines from jitted
code and lines from Python can then appear out of order. If the process crashes, C never writes the text that remains in
its buffer. To keep the order, flush Python's buffer with ``sys.stdout.flush()`` before the jitted call. In jitted code,
call ``fflush(stdout())`` after the write, as ``report`` does; ``fflush(0)`` flushes every C stream. On Linux, C writes
stderr without a buffer (`setbuf(3) <https://man7.org/linux/man-pages/man3/setbuf.3.html>`_), so a warning appears at
once. On Windows, C flushes a stderr that goes to a file or a pipe only when its buffer fills (`Microsoft's stream I/O
<https://learn.microsoft.com/en-us/cpp/c-runtime-library/stream-i-o#stream-io-routines>`_), so call ``fflush(stderr())``
there as well.

Reading errno after a failed call
+++++++++++++++++++++++++++++++++

A C function that fails, such as ``fopen`` on a missing file, returns a failure value and leaves the reason in ``errno``
(`errno(3) <https://man7.org/linux/man-pages/man3/errno.3.html>`_). :func:`~numbox.core.bindings.errno.errno_get`
returns the current thread's ``errno`` as an ``int32``. :func:`~numbox.core.bindings.errno.errno_set` sets it, and
narrows its argument to ``int32``. Read ``errno`` in the jitted function that made the call, immediately after the call,
because a later C call may change it:

.. code-block:: python

    from numba import njit

    from numbox.core.bindings.errno import errno_get
    from numbox.core.bindings.libc import fclose, fopen
    from numbox.utils.lowlevel import get_unicode_data_p


    @njit(cache=True)
    def open_error(path):
        fp = fopen(get_unicode_data_p(path), get_unicode_data_p("r"))
        if fp == 0:
            return errno_get()
        fclose(fp)
        return 0


    print(open_error("/no/such/file"))   # 2, which is errno.ENOENT

Each thread has its own ``errno``, and both functions look up the current thread's on every call. In a ``prange`` loop
(`parallel loops <https://numba.readthedocs.io/en/stable/user/parallel.html#explicit-parallel-loops>`_), an iteration
therefore reads the ``errno`` of the thread that runs it. Read ``errno`` in the iteration that made the C call, because
the caller's ``errno`` after the loop depends on the threading layer. Under OpenMP, which numba selects when TBB is not
installed (`threading layers
<https://numba.readthedocs.io/en/stable/user/threading-layer.html#setting-the-threading-layer-selection-priority>`_),
the calling thread runs some iterations itself. After the loop, its ``errno`` holds the value that those iterations
left. Under the workqueue layer, the calling thread runs no iterations, and its ``errno`` is unchanged.

Turning an error number into a message
--------------------------------------

:func:`~numbox.core.bindings.strerror.strerror_safe` writes the message for an error number into a buffer that the
caller provides, and ends the message with a NUL. It returns 0 on success and an error number on failure. libc's
``strerror`` may return a buffer that a later call overwrites; ``strerror_safe`` keeps no buffer of its own, so threads
can call it at the same time:

.. code-block:: python

    import numpy as np
    from numba import njit

    from numbox.core.bindings.strerror import strerror_safe
    from numbox.utils.lowlevel import array_data_p, get_str_from_p_as_int


    @njit(cache=True)
    def describe(errnum, buf):
        rc = strerror_safe(errnum, array_data_p(buf), buf.size)
        return rc, get_str_from_p_as_int(array_data_p(buf))


    print(describe(2, np.zeros(128, dtype=np.uint8)))   # (0, 'No such file or directory')

With glibc, ``strerror_safe`` returns ``EINVAL`` (22) for an unknown error number and writes ``Unknown error`` followed
by the number. It returns ``ERANGE`` (34) for a buffer too short for the message and writes as much of the message as
fits. On Linux, ``strerror_safe`` calls ``__xpg_strerror_r``, the POSIX form of `strerror_r
<https://man7.org/linux/man-pages/man3/strerror_r.3.html>`__ that glibc exports, and falls back to ``strerror_r`` where
the C library lacks it; on macOS it calls ``strerror_r``, and on Windows `strerror_s
<https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/strerror-s-strerror-s-wcserror-s-wcserror-s>`__.

Formatted output and input
++++++++++++++++++++++++++

:func:`~numbox.core.bindings.fmtio.printf`, :func:`~numbox.core.bindings.fmtio.fprintf`,
:func:`~numbox.core.bindings.fmtio.snprintf` and :func:`~numbox.core.bindings.fmtio.sscanf` take a format string and the
values after it, as C's functions do (`printf(3) <https://man7.org/linux/man-pages/man3/printf.3.html>`__, `sscanf(3)
<https://man7.org/linux/man-pages/man3/sscanf.3.html>`__). They are ordinary Python functions that numba also compiles,
as it compiles ``print``, so one function body can run under Python and in jitted code:

.. code-block:: python

    from numba import njit

    from numbox.core.bindings.fmtio import printf
    from numbox.core.bindings.libc import fflush
    from numbox.core.bindings.stdio import stdout


    def step(n, label):
        printf("step %d: %s\n", n, label)
        fflush(stdout())
        return 2 * n


    step(7, "in Python")
    njit(step)(7, "in jitted code")

Under Python, ``printf`` formats the values with Python's ``%`` operator and writes the text to the current
``sys.stdout``, so a redirection such as ``contextlib.redirect_stdout`` reaches it. ``fprintf`` maps the addresses that
``stdout()``, ``stderr()`` and ``stdin()`` return to the ``sys.stdout``, ``sys.stderr`` and ``sys.stdin`` objects that
it finds at its first call. A redirection installed after that first call does not change where ``fprintf`` writes.
``fprintf`` raises ``RuntimeError`` for any other ``FILE *``, such as one from ``fopen``. ``snprintf`` writes the text
into memory at the address it receives, such as ``array_data_p(buf)``. In jitted code, numba compiles each call into a
call of C's function, which writes to C's streams and accepts any ``FILE *``. ``sscanf`` runs only in jitted code, as
the section `Parsing with sscanf`_ describes.

The format string and its arguments
-----------------------------------

In jitted code, the format must be a string that numba reads as a constant: a string literal, a module-level string or a
closure variable. A format built at run time fails to compile with
``TypingError: printf: format string must be a literal str``; pass the part that varies as an argument instead, such as
a ``%s``. numbox encodes the format as UTF-8, so non-ASCII text in the format prints as written. Each conversion takes
one kind of argument:

======================================================  ========================
Conversions                                             Argument
======================================================  ========================
``%d``, ``%i``, ``%o``, ``%u``, ``%x``, ``%X``, ``%c``  an integer or a ``bool``
``%e``, ``%f``, ``%g``, ``%a`` and their capitals       a float
``%s``                                                  a ``str``, or a pointer
``%p``                                                  a pointer
======================================================  ========================

When numba compiles the call, numbox checks the format against the arguments, as GCC's `-Wformat
<https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html#index-Wformat>`_ does. The compile fails with a ``TypingError``
that names the problem when:

- the format has more or fewer conversions than there are arguments; a ``*`` width or precision counts as one more
  integer argument;
- a conversion gets an argument of another kind than the one it takes;
- an argument has a type that no conversion takes, such as an array, a complex number or a tuple; pass the element or
  the field you mean, such as ``a[0]`` or ``z.real``;
- the format holds ``%n``, which would make C write the number of characters printed so far through a pointer;
- the format holds the ``'`` flag for digit grouping, which POSIX defines and C99 and Python's ``%`` lack.

numba lists each of these errors next to a second message, ``format string must be a literal str, got unicode_type``;
read the message that names the mismatch. A literal percent sign followed by ``n``, ``%%n``, is allowed. Under Python,
``%n`` and the ``'`` flag raise ``ValueError``.

The check cannot distinguish a pointer from an integer, because numba represents a pointer as an ``intp``, which is an
``int64``. A ``%s`` given an ``int64`` value compiles, and C then reads memory at that address, which usually ends the
process with a segmentation fault. Pass ``%s`` a ``str``, or the address of a NUL-terminated buffer. ``%p`` only prints
its argument as an address, so an ``int64`` argument for ``%p`` is harmless.

For ``%s``, numbox passes a ``str`` as the address of its characters, so the ASCII limit from the section
`Pointers and memory`_ applies: ``"caf\xe9"`` prints its last character as the single byte ``0xE9``, not as UTF-8. Pass
other text as a ``uint8`` array of UTF-8 bytes, built as that section shows. A width, as in ``%-10s``, counts bytes, not
characters.

When Python and C print differently
-----------------------------------

If the format uses an ``ll`` conversion for every integer, ``%lld`` for a signed value and ``%llu`` for an unsigned one,
the call prints the same under Python and in jitted code. Other conversions can print differently. Python's ``%``
operator ignores a single ``h``, ``l`` or ``L`` and rejects any other length modifier, as its `documentation
<https://docs.python.org/3/builtins/stdtypes.html#printf-style-string-formatting>`_ says: ``"%lld" % 5`` raises
``ValueError: unsupported format character 'l'``. Under Python, numbox therefore removes every length modifier from the
format before it formats, and with the modifiers gone, Python prints each value at its full size. In jitted code, numbox
passes every integer and ``bool`` as 64 bits and every float as a ``float64``, and C converts each value to the type
that its conversion names, such as ``int`` for ``%d`` or ``signed char`` for ``%hhd``. A value that fits that type
prints the same under Python and in jitted code, and a value that does not fit prints differently:

========  =============  =================  =================
Format    Value          Python             jitted code
========  =============  =================  =================
``%lld``  ``2**40 + 5``  ``1099511627781``  ``1099511627781``
``%d``    ``2**40 + 5``  ``1099511627781``  ``5``
``%hhd``  ``300``        ``300``            ``44``
``%x``    ``-1``         ``-1``             ``ffffffff``
========  =============  =================  =================

The Python path has one known inconsistency: its stripping does not skip the ``%%`` escape, so it also removes a
modifier that follows a literal ``%%``, and ``printf("%%lld\n")`` prints ``%d`` under Python and ``%lld`` in jitted
code. On Windows, C's ``long`` has 32 bits (`Microsoft's data type ranges
<https://learn.microsoft.com/en-us/cpp/cpp/data-type-ranges>`_), so ``%ld`` reads only 32 bits of an ``int64`` there.
Python accepts some calls that jitted code refuses, such as ``%f`` with an integer and ``%d`` with a float. ``%p`` works
only in jitted code, because Python's ``%`` rejects it.

Checking snprintf for truncation
--------------------------------

``snprintf(buf, size, fmt, ...)`` writes at most ``size`` bytes into ``buf``, the terminating NUL included, and returns
the length of the whole text. A return value of ``size`` or more therefore means that ``snprintf`` truncated the text.
In jitted code, ``buf`` and ``size`` must be ``intp`` values. ``a.size``, ``np.intp(16)`` and an ``int64`` argument of
the jitted function, such as a Python ``int`` passed in, are ``intp``, but a literal is not: ``snprintf(p, 16, ...)``
fails to compile with ``TypingError: snprintf: size must be intp (size_t-as-int), got Literal[int](16)``. The message
means that numba typed the literal ``16`` as a literal integer; pass ``a.size`` or ``np.intp(16)`` instead.

On Windows, jitted code calls C's `_snprintf
<https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/snprintf-snprintf-snprintf-l-snwprintf-snwprintf-l>`_
instead. It returns ``-1`` when the text is longer than ``size``, and when the text is exactly ``size`` bytes long it
returns ``size`` and writes no terminating NUL. The test ``n < 0 or n >= size`` covers both cases on every platform:

.. code-block:: python

    import numpy as np
    from numba import njit

    from numbox.core.bindings.fmtio import snprintf
    from numbox.utils.lowlevel import array_data_p


    @njit(cache=True)
    def format_range(buf, lo, hi):
        n = snprintf(array_data_p(buf), buf.size, "[%d:%d]", lo, hi)
        return -1 if n < 0 or n >= buf.size else n


    buf = np.zeros(16, dtype=np.uint8)
    n = format_range(buf, 7, 11)
    print(n, bytes(buf[:n]).decode())   # 6 [7:11]

Under Python, ``snprintf`` follows the C99 rule on every platform.

Parsing with sscanf
-------------------

``sscanf(buf, fmt, ...)`` reads values from the NUL-terminated text at ``buf`` into storage that the caller provides. It
returns the number of values that it stored, or ``-1`` (``EOF``) when the text ends before the first conversion. Unlike
the other three functions, it runs only in jitted code. Under Python it raises ``NotImplementedError``, and ``int()``,
``float()`` or ``re`` parse the text there instead. A jitted function calls ``sscanf`` as follows:

.. code-block:: python

    import numpy as np
    from numba import njit

    from numbox.core.bindings.fmtio import sscanf
    from numbox.utils.lowlevel import array_data_p, get_unicode_data_p


    @njit(cache=True)
    def parse_pair(text):
        count = np.zeros(1, dtype=np.int32)
        ratio = np.zeros(1, dtype=np.float64)
        n = sscanf(get_unicode_data_p(text), "%d %lf", array_data_p(count), array_data_p(ratio))
        return n, count[0], ratio[0]


    print(parse_pair("42 0.25"))   # (2, 42, 0.25)

``buf`` must be an ``intp``, so pass ``get_unicode_data_p(text)``, not the string; the ASCII limit from the section
`Pointers and memory`_ applies to the text. Each argument after the format is the address of the storage for one
conversion, also as an ``intp``. numbox checks only that type, so an ``int64`` value that is not an address compiles. C
then treats the value as an address and writes to the memory that it points to. numbox does not compare the format with
the arguments either, so the caller must make them match the format in number and in size. Each conversion writes
storage of its own size:

==================  ==================================================
Conversion          Storage
==================  ==================================================
``%hhd``            ``int8``
``%hd``             ``int16``
``%d``, ``%u``      ``int32``, ``uint32``
``%lld``, ``%llu``  ``int64``, ``uint64``
``%ld``             ``int64`` on Linux and macOS, ``int32`` on Windows
``%f``              ``float32``
``%lf``             ``float64``
``%s``              a ``uint8`` array
``%n``              ``int32``
==================  ==================================================

Give ``%s`` a width one less than the array's size, such as ``%15s`` for 16 bytes, so that the characters and the NUL
fit. Unlike the output functions, ``sscanf`` accepts ``%n``: it stores the number of characters read so far, and the
return value does not count it.

Caching callers of the formatted I/O functions
----------------------------------------------

A jitted call of ``printf``, ``fprintf``, ``snprintf`` or ``sscanf`` compiles into its caller: the caller's code names
C's function and holds the format as a constant. A caller cached with ``cache=True`` works in a later process, which
finds C's function by name. After you upgrade numbox, however, a cached caller still runs the fmtio code of the numbox
version that compiled it. This happens because numba's cache key covers the caller's own file and not the code of the
functions it calls (:ref:`what the key leaves out <cache_key_leaves_out>`). Delete the caller's cache files after an
upgrade; :ref:`where the cache lands <where_the_cache_lands>` says where they are. Alternatively, decorate the caller
with :func:`~numbox.core.cache_key.numbox_njit`, whose cache key holds numbox's version, so that an upgrade from one
release to another recompiles it. The cache_key page lists :ref:`which functions numbox keys
<cache_key_which_functions>`. It also explains :ref:`what an upgrade changes <cache_key_after_upgrade>`, both between
releases and in a source checkout, where the version stays empty.

Modules
++++++++

numbox.core.bindings.libc
-------------------------

.. automodule:: numbox.core.bindings.libc
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.libm
-------------------------

.. automodule:: numbox.core.bindings.libm
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.stdio
--------------------------

.. automodule:: numbox.core.bindings.stdio
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.errno
--------------------------

.. automodule:: numbox.core.bindings.errno
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.strerror
-----------------------------

.. automodule:: numbox.core.bindings.strerror
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.fmtio
--------------------------

.. automodule:: numbox.core.bindings.fmtio
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.abi
------------------------

.. automodule:: numbox.core.bindings.abi
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
