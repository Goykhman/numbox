"""Bindings for C standard library functions on strings, memory, streams, random numbers and the environment.

Each binding has the C function's name and takes its arguments in C's order, with the types of the function's entry in
:mod:`numbox.core.bindings.signatures`: C's ``int`` as ``int32``, and ``size_t`` and every pointer as ``intp``. A
pointer is an address that the caller keeps alive while C uses it, and numba converts another numeric argument to the
entry's type without a warning. Each binding is a proxied function that jitted code and Python can call, and its
docstring links the C function's Linux manual page. The section "Bindings module conventions" on the bindings page
describes the conventions that all binding modules share.
"""
from numbox.core.proxy.proxy import proxy
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options


__all__ = [
    "rand", "srand",
    "strlen", "puts", "fputs", "fputc", "putchar",
    "fwrite", "fread", "fflush",
    "fopen", "fclose",
    "feof", "ferror", "clearerr",
    "strcmp", "strncmp", "strchr", "strrchr", "strstr", "strncpy", "strerror",
    "memcpy", "memmove", "memset", "memcmp", "memchr",
    "getenv",
]

load_lib("c")


@proxy(signatures.get("rand"), jit_options=jit_options)
def rand():
    """Return the next pseudo-random integer, from 0 to ``RAND_MAX``, of the generator that ``srand`` seeds.

    C reference: `rand(3) <https://man7.org/linux/man-pages/man3/rand.3.html>`_.

    ``RAND_MAX`` is 2147483647 with glibc and on macOS, and 32767 on Windows. C does not require ``rand`` to be safe to
    call from several threads concurrently.
    """
    return _call_lib_func("rand", ())


@proxy(signatures.get("srand"), jit_options=jit_options)
def srand(s):
    """Seed the generator that ``rand`` uses with ``s``.

    C reference: `srand(3) <https://man7.org/linux/man-pages/man3/srand.3.html>`_.

    C takes the seed as an ``unsigned int``: numba converts an integer argument to ``int32`` without a warning, and C
    reads its 32 bits as unsigned.
    """
    return _call_lib_func("srand", (s,))


@proxy(signatures.get("strlen"), jit_options=jit_options)
def strlen(s):
    """Return the number of bytes in the NUL-terminated string at address ``s``, the NUL excluded.

    C reference: `strlen(3) <https://man7.org/linux/man-pages/man3/strlen.3.html>`_.

    ``s`` is an ``intp`` address, such as one from :func:`~numbox.utils.lowlevel.get_unicode_data_p`, and the caller
    keeps the string alive during the call.
    """
    return _call_lib_func("strlen", (s,))


@proxy(signatures.get("puts"), jit_options=jit_options)
def puts(s):
    """Write the NUL-terminated string at address ``s`` and a newline to C's stdout.

    C reference: `puts(3) <https://man7.org/linux/man-pages/man3/puts.3.html>`_.

    The result is non-negative on success and ``EOF`` on error. ``s`` is an ``intp`` address that the caller keeps alive
    during the call.
    """
    return _call_lib_func("puts", (s,))


@proxy(signatures.get("fputs"), jit_options=jit_options)
def fputs(s, fp):
    """Write the NUL-terminated string at address ``s`` to the stream ``fp``, without a newline.

    C reference: `fputs(3) <https://man7.org/linux/man-pages/man3/fputs.3.html>`_.

    The result is non-negative on success and ``EOF`` on error. ``s`` is an ``intp`` address that the caller keeps alive
    during the call, and ``fp`` is a ``FILE *`` as an ``intp``, from ``fopen`` or from
    :mod:`numbox.core.bindings.stdio`.
    """
    return _call_lib_func("fputs", (s, fp))


@proxy(signatures.get("fputc"), jit_options=jit_options)
def fputc(c, fp):
    """Write ``c``, converted to ``unsigned char``, to the stream ``fp``, and return the byte written.

    C reference: `fputc(3) <https://man7.org/linux/man-pages/man3/fputc.3.html>`_.

    The result is ``EOF`` on error. ``fp`` is a ``FILE *`` as an ``intp``.
    """
    return _call_lib_func("fputc", (c, fp))


@proxy(signatures.get("putchar"), jit_options=jit_options)
def putchar(c):
    """Write ``c``, converted to ``unsigned char``, to C's stdout, and return the byte written.

    C reference: `putchar(3) <https://man7.org/linux/man-pages/man3/putchar.3.html>`_.

    The result is ``EOF`` on error.
    """
    return _call_lib_func("putchar", (c,))


@proxy(signatures.get("fwrite"), jit_options=jit_options)
def fwrite(ptr, size, nmemb, fp):
    """Write ``nmemb`` elements of ``size`` bytes each from address ``ptr`` to the stream ``fp``.

    C reference: `fwrite(3) <https://man7.org/linux/man-pages/man3/fwrite.3.html>`_.

    The result is the number of elements written, which is less than ``nmemb`` only on error. ``ptr`` is an ``intp``
    address that the caller keeps alive during the call, and ``fp`` is a ``FILE *`` as an ``intp``.
    """
    return _call_lib_func("fwrite", (ptr, size, nmemb, fp))


@proxy(signatures.get("fread"), jit_options=jit_options)
def fread(ptr, size, nmemb, fp):
    """Read up to ``nmemb`` elements of ``size`` bytes each from the stream ``fp`` into memory at address ``ptr``.

    C reference: `fread(3) <https://man7.org/linux/man-pages/man3/fread.3.html>`_.

    The result is the number of elements read, which is less than ``nmemb`` at the end of the file or on an error;
    ``feof`` and ``ferror`` distinguish the two cases. ``ptr`` is the ``intp`` address of at least ``size * nmemb``
    bytes, which the caller keeps alive during the call.
    """
    return _call_lib_func("fread", (ptr, size, nmemb, fp))


@proxy(signatures.get("fflush"), jit_options=jit_options)
def fflush(fp):
    """Flush the buffered output of the stream ``fp``, or of every output stream if ``fp`` is 0.

    C reference: `fflush(3) <https://man7.org/linux/man-pages/man3/fflush.3.html>`_.

    The result is 0 on success and ``EOF`` on error. Because C buffers its stdout apart from Python's ``sys.stdout``,
    text from jitted code and text from Python can appear out of order when stdout is a file or a pipe, until both are
    flushed.
    """
    return _call_lib_func("fflush", (fp,))


@proxy(signatures.get("fopen"), jit_options=jit_options)
def fopen(path, mode):
    """Open the file whose name is the string at ``path``, in the mode given by the string at ``mode``, as a stream.

    C reference: `fopen(3) <https://man7.org/linux/man-pages/man3/fopen.3.html>`_.

    ``path`` and ``mode``, such as ``"r"``, ``"wb"`` or ``"a+"``, are ``intp`` addresses of NUL-terminated strings. The
    result is a ``FILE *`` as an ``intp``, or 0 on error, with the reason in ``errno``. The caller closes the stream
    with ``fclose``. On Windows, a mode without ``b`` opens the file in text mode, which translates line endings.
    """
    return _call_lib_func("fopen", (path, mode))


@proxy(signatures.get("fclose"), jit_options=jit_options)
def fclose(fp):
    """Flush the buffered output of the stream ``fp``, close the stream, and return 0, or ``EOF`` on error.

    C reference: `fclose(3) <https://man7.org/linux/man-pages/man3/fclose.3.html>`_.

    ``fp`` is invalid after the call, whatever the result, and must not be used again.
    """
    return _call_lib_func("fclose", (fp,))


@proxy(signatures.get("feof"), jit_options=jit_options)
def feof(fp):
    """Return a nonzero value if the end-of-file indicator of the stream ``fp`` is set, and 0 if it is not.

    C reference: `feof(3) <https://man7.org/linux/man-pages/man3/feof.3.html>`_.

    ``clearerr`` clears the indicator, as do C's ``fseek``, ``fsetpos`` and ``rewind`` when they succeed; numbox binds
    none of those three.
    """
    return _call_lib_func("feof", (fp,))


@proxy(signatures.get("ferror"), jit_options=jit_options)
def ferror(fp):
    """Return a nonzero value if the error indicator of the stream ``fp`` is set, and 0 if it is not.

    C reference: `ferror(3) <https://man7.org/linux/man-pages/man3/ferror.3.html>`_.

    ``clearerr`` clears the indicator, as does C's ``rewind``, which numbox does not bind.
    """
    return _call_lib_func("ferror", (fp,))


@proxy(signatures.get("clearerr"), jit_options=jit_options)
def clearerr(fp):
    """Clear the end-of-file and error indicators of the stream ``fp``.

    C reference: `clearerr(3) <https://man7.org/linux/man-pages/man3/clearerr.3.html>`_.
    """
    return _call_lib_func("clearerr", (fp,))


@proxy(signatures.get("strcmp"), jit_options=jit_options)
def strcmp(a, b):
    """Compare the NUL-terminated strings at addresses ``a`` and ``b`` byte by byte, as unsigned bytes.

    C reference: `strcmp(3) <https://man7.org/linux/man-pages/man3/strcmp.3.html>`_.

    The result is negative if ``a`` sorts first, 0 if the strings are equal, and positive if ``b`` sorts first.
    """
    return _call_lib_func("strcmp", (a, b))


@proxy(signatures.get("strncmp"), jit_options=jit_options)
def strncmp(a, b, n):
    """Compare at most the first ``n`` bytes of the NUL-terminated strings at addresses ``a`` and ``b``.

    C reference: `strncmp(3) <https://man7.org/linux/man-pages/man3/strncmp.3.html>`_.

    The comparison stops at the first NUL in either string, and the result follows the sign convention of ``strcmp``. An
    ``n`` of 0 gives 0.
    """
    return _call_lib_func("strncmp", (a, b, n))


@proxy(signatures.get("strchr"), jit_options=jit_options)
def strchr(s, c):
    """Return the address of the first byte equal to ``c`` in the NUL-terminated string at ``s``, or 0 if none.

    C reference: `strchr(3) <https://man7.org/linux/man-pages/man3/strchr.3.html>`_.

    C converts ``c`` to ``char``, and a ``c`` of 0 finds the terminating NUL. numba converts an integer argument to
    ``int32`` without a warning.
    """
    return _call_lib_func("strchr", (s, c))


@proxy(signatures.get("strrchr"), jit_options=jit_options)
def strrchr(s, c):
    """Return the address of the last byte equal to ``c`` in the NUL-terminated string at ``s``, or 0 if none.

    C reference: `strrchr(3) <https://man7.org/linux/man-pages/man3/strrchr.3.html>`_.

    C converts ``c`` to ``char``, and a ``c`` of 0 finds the terminating NUL. numba converts an integer argument to
    ``int32`` without a warning.
    """
    return _call_lib_func("strrchr", (s, c))


@proxy(signatures.get("strstr"), jit_options=jit_options)
def strstr(haystack, needle):
    """Return the address of the first occurrence of the string ``needle`` in the string ``haystack``, or 0 if none.

    C reference: `strstr(3) <https://man7.org/linux/man-pages/man3/strstr.3.html>`_.

    Both arguments are ``intp`` addresses of NUL-terminated strings. An empty ``needle`` gives ``haystack``.
    """
    return _call_lib_func("strstr", (haystack, needle))


@proxy(signatures.get("strncpy"), jit_options=jit_options)
def strncpy(dst, src, n):
    """Copy at most ``n`` bytes of the NUL-terminated string at ``src`` to ``dst``, and return ``dst``.

    C reference: `strncpy(3) <https://man7.org/linux/man-pages/man3/strncpy.3.html>`_.

    If ``src`` is shorter than ``n`` bytes, C fills the rest of the ``n`` bytes with NULs. If it is ``n`` bytes or
    longer, ``dst`` receives ``n`` bytes and no terminating NUL. A caller that needs a NUL-terminated result reserves
    one more byte, and either zeroes the buffer first or writes a NUL at offset ``n`` of ``dst`` after the call.
    """
    return _call_lib_func("strncpy", (dst, src, n))


@proxy(signatures.get("strerror"), jit_options=jit_options)
def strerror(errnum):
    """Return the address of the message string for the error number ``errnum``.

    C reference: `strerror(3) <https://man7.org/linux/man-pages/man3/strerror.3.html>`_.

    C does not require ``strerror`` to be thread-safe: the string may be in a buffer that a later call overwrites.
    :func:`~numbox.core.bindings.strerror.strerror_safe` writes the message into the caller's buffer instead.
    """
    return _call_lib_func("strerror", (errnum,))


@proxy(signatures.get("memcpy"), jit_options=jit_options)
def memcpy(dst, src, n):
    """Copy ``n`` bytes from address ``src`` to address ``dst``, and return ``dst``.

    C reference: `memcpy(3) <https://man7.org/linux/man-pages/man3/memcpy.3.html>`_.

    The two regions must not overlap; ``memmove`` copies between overlapping regions. The caller keeps both regions
    alive during the call.
    """
    return _call_lib_func("memcpy", (dst, src, n))


@proxy(signatures.get("memmove"), jit_options=jit_options)
def memmove(dst, src, n):
    """Copy ``n`` bytes from address ``src`` to address ``dst``, which may overlap, and return ``dst``.

    C reference: `memmove(3) <https://man7.org/linux/man-pages/man3/memmove.3.html>`_.

    The caller keeps both regions alive during the call.
    """
    return _call_lib_func("memmove", (dst, src, n))


@proxy(signatures.get("memset"), jit_options=jit_options)
def memset(dst, c, n):
    """Fill the ``n`` bytes at address ``dst`` with ``c``, converted to ``unsigned char``, and return ``dst``.

    C reference: `memset(3) <https://man7.org/linux/man-pages/man3/memset.3.html>`_.
    """
    return _call_lib_func("memset", (dst, c, n))


@proxy(signatures.get("memcmp"), jit_options=jit_options)
def memcmp(a, b, n):
    """Compare the first ``n`` bytes at addresses ``a`` and ``b``, as unsigned bytes.

    C reference: `memcmp(3) <https://man7.org/linux/man-pages/man3/memcmp.3.html>`_.

    The result follows the sign convention of ``strcmp``, and the comparison does not stop at a NUL.
    """
    return _call_lib_func("memcmp", (a, b, n))


@proxy(signatures.get("memchr"), jit_options=jit_options)
def memchr(s, c, n):
    """Return the address of the first byte equal to ``c`` in the ``n`` bytes at address ``s``, or 0 if none.

    C reference: `memchr(3) <https://man7.org/linux/man-pages/man3/memchr.3.html>`_.

    C converts ``c`` to ``unsigned char``, so it compares ``c & 0xff``.
    """
    return _call_lib_func("memchr", (s, c, n))


@proxy(signatures.get("getenv"), jit_options=jit_options)
def getenv(name):
    """Return the address of the value of the environment variable named by the string at ``name``, or 0 if unset.

    C reference: `getenv(3) <https://man7.org/linux/man-pages/man3/getenv.3.html>`_.

    The environment owns the value: the caller must not modify or free it, and a later ``setenv``, ``putenv`` or
    ``unsetenv`` can invalidate it. To keep the value, copy it into a ``str`` with
    :func:`~numbox.utils.lowlevel.get_str_from_p_as_int` before the environment changes.
    """
    return _call_lib_func("getenv", (name,))
