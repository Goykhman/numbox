"""libc wrappers callable from numba `@njit` code.

See "Bindings module conventions" in ``docs/numbox.core.bindings.rst``
for the ABI-safety, ``@proxy`` caching, and reference-source conventions
shared across all binding modules.
"""
from numbox.core.proxy.proxy import proxy
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib

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


@proxy(signatures.get("rand"), jit_options={"cache": True})
def rand():
    """POSIX `rand() <https://man7.org/linux/man-pages/man3/rand.3.html>`_:
    pseudo-random int in [0, RAND_MAX]. Not thread-safe."""
    return _call_lib_func("rand", ())


@proxy(signatures.get("srand"), jit_options={"cache": True})
def srand(s):
    """POSIX `srand(seed) <https://man7.org/linux/man-pages/man3/rand.3.html>`_:
    seed the rand() generator."""
    return _call_lib_func("srand", (s,))


@proxy(signatures.get("strlen"), jit_options={"cache": True})
def strlen(s):
    """POSIX `strlen(s) <https://man7.org/linux/man-pages/man3/strlen.3.html>`_:
    number of bytes in the C string at `s` up to (but excluding) the trailing NUL.
    """
    return _call_lib_func("strlen", (s,))


@proxy(signatures.get("puts"), jit_options={"cache": True})
def puts(s):
    """POSIX `puts(s) <https://man7.org/linux/man-pages/man3/puts.3.html>`_:
    write the NUL-terminated string at `s` to stdout followed by a newline.
    Returns non-negative on success, EOF on error."""
    return _call_lib_func("puts", (s,))


@proxy(signatures.get("fputs"), jit_options={"cache": True})
def fputs(s, fp):
    """POSIX `fputs(s, fp) <https://man7.org/linux/man-pages/man3/fputs.3.html>`_:
    write the NUL-terminated string at `s` to FILE* `fp` (no trailing newline).
    Returns non-negative on success, EOF on error."""
    return _call_lib_func("fputs", (s, fp))


@proxy(signatures.get("fputc"), jit_options={"cache": True})
def fputc(c, fp):
    """POSIX `fputc(c, fp) <https://man7.org/linux/man-pages/man3/fputc.3.html>`_:
    write byte `c` (as unsigned char) to FILE* `fp`. Returns the byte on success,
    EOF on error."""
    return _call_lib_func("fputc", (c, fp))


@proxy(signatures.get("putchar"), jit_options={"cache": True})
def putchar(c):
    """POSIX `putchar(c) <https://man7.org/linux/man-pages/man3/putchar.3.html>`_:
    write byte `c` (as unsigned char) to stdout."""
    return _call_lib_func("putchar", (c,))


@proxy(signatures.get("fwrite"), jit_options={"cache": True})
def fwrite(ptr, size, nmemb, fp):
    """POSIX `fwrite(ptr, size, nmemb, fp)
    <https://man7.org/linux/man-pages/man3/fwrite.3.html>`_: write `nmemb`
    elements of `size` bytes each from `ptr` to FILE* `fp`. Returns the number
    of elements successfully written (which may be less than `nmemb` on error)."""
    return _call_lib_func("fwrite", (ptr, size, nmemb, fp))


@proxy(signatures.get("fread"), jit_options={"cache": True})
def fread(ptr, size, nmemb, fp):
    """POSIX `fread(ptr, size, nmemb, fp)
    <https://man7.org/linux/man-pages/man3/fread.3.html>`_: read up to `nmemb`
    elements of `size` bytes each from FILE* `fp` into `ptr`. Returns the number
    of elements successfully read; use `feof`/`ferror` to disambiguate short reads."""
    return _call_lib_func("fread", (ptr, size, nmemb, fp))


@proxy(signatures.get("fflush"), jit_options={"cache": True})
def fflush(fp):
    """POSIX `fflush(fp) <https://man7.org/linux/man-pages/man3/fflush.3.html>`_:
    flush the C stdio buffer for FILE* `fp` (pass NULL/0 to flush all streams).
    Returns 0 on success, EOF on error."""
    return _call_lib_func("fflush", (fp,))


@proxy(signatures.get("fopen"), jit_options={"cache": True})
def fopen(path, mode):
    """POSIX `fopen(path, mode) <https://man7.org/linux/man-pages/man3/fopen.3.html>`_:
    open the file at `path` (NUL-terminated) with mode string `mode` (e.g.
    "r", "wb", "a+"). Returns a FILE* (as intp); 0 on error (caller can read
    errno). Owned resource — caller MUST `fclose` to avoid leaks."""
    return _call_lib_func("fopen", (path, mode))


@proxy(signatures.get("fclose"), jit_options={"cache": True})
def fclose(fp):
    """POSIX `fclose(fp) <https://man7.org/linux/man-pages/man3/fclose.3.html>`_:
    close FILE* `fp` and flush any buffered output. Returns 0 on success, EOF
    on error. After this call, `fp` is invalid and must not be reused."""
    return _call_lib_func("fclose", (fp,))


@proxy(signatures.get("feof"), jit_options={"cache": True})
def feof(fp):
    """POSIX `feof(fp) <https://man7.org/linux/man-pages/man3/feof.3.html>`_:
    non-zero iff the end-of-file indicator is set on FILE* `fp`. Cleared by
    `clearerr` or successful repositioning (`fseek`/`rewind`)."""
    return _call_lib_func("feof", (fp,))


@proxy(signatures.get("ferror"), jit_options={"cache": True})
def ferror(fp):
    """POSIX `ferror(fp) <https://man7.org/linux/man-pages/man3/ferror.3.html>`_:
    non-zero iff the error indicator is set on FILE* `fp`. Cleared only by
    `clearerr`."""
    return _call_lib_func("ferror", (fp,))


@proxy(signatures.get("clearerr"), jit_options={"cache": True})
def clearerr(fp):
    """POSIX `clearerr(fp) <https://man7.org/linux/man-pages/man3/clearerr.3.html>`_:
    clear the end-of-file and error indicators on FILE* `fp`."""
    return _call_lib_func("clearerr", (fp,))


@proxy(signatures.get("strcmp"), jit_options={"cache": True})
def strcmp(a, b):
    """POSIX `strcmp(a, b) <https://man7.org/linux/man-pages/man3/strcmp.3.html>`_:
    lexicographic byte comparison of two NUL-terminated strings. Returns
    negative if a<b, zero if equal, positive if a>b."""
    return _call_lib_func("strcmp", (a, b))


@proxy(signatures.get("strncmp"), jit_options={"cache": True})
def strncmp(a, b, n):
    """POSIX `strncmp(a, b, n) <https://man7.org/linux/man-pages/man3/strncmp.3.html>`_:
    `strcmp` bounded to the first `n` bytes (stops early at a NUL on either side).
    `n == 0` always returns 0 by spec."""
    return _call_lib_func("strncmp", (a, b, n))


@proxy(signatures.get("strchr"), jit_options={"cache": True})
def strchr(s, c):
    """POSIX `strchr(s, c) <https://man7.org/linux/man-pages/man3/strchr.3.html>`_:
    pointer to the FIRST occurrence of byte `c` in NUL-terminated string `s`,
    or 0 if not found. `c==0` returns a pointer to the terminating NUL."""
    return _call_lib_func("strchr", (s, c))


@proxy(signatures.get("strrchr"), jit_options={"cache": True})
def strrchr(s, c):
    """POSIX `strrchr(s, c) <https://man7.org/linux/man-pages/man3/strrchr.3.html>`_:
    pointer to the LAST occurrence of byte `c` in NUL-terminated string `s`,
    or 0 if not found."""
    return _call_lib_func("strrchr", (s, c))


@proxy(signatures.get("strstr"), jit_options={"cache": True})
def strstr(haystack, needle):
    """POSIX `strstr(haystack, needle) <https://man7.org/linux/man-pages/man3/strstr.3.html>`_:
    pointer to the first occurrence of NUL-terminated `needle` within
    NUL-terminated `haystack`, or 0 if not found. Empty `needle` returns `haystack`.
    """
    return _call_lib_func("strstr", (haystack, needle))


@proxy(signatures.get("strncpy"), jit_options={"cache": True})
def strncpy(dst, src, n):
    """POSIX `strncpy(dst, src, n) <https://man7.org/linux/man-pages/man3/strncpy.3.html>`_:
    copy at most n bytes from src to dst.

    Does NOT guarantee null termination: if strlen(src) >= n, dst will
    contain n bytes from src with no trailing NUL. Callers that need a
    NUL-terminated result must reserve an extra byte and either pre-zero
    the buffer or explicitly write dst[n] = 0 after the call.
    """
    return _call_lib_func("strncpy", (dst, src, n))


@proxy(signatures.get("strerror"), jit_options={"cache": True})
def strerror(errnum):
    """POSIX `strerror(errnum) <https://man7.org/linux/man-pages/man3/strerror.3.html>`_:
    pointer to the static error-message string for errnum.

    NOT thread-safe — the returned pointer references a per-process static
    buffer that subsequent strerror calls may overwrite. Use ``strerror_safe``
    (in ``numbox.core.bindings._strerror``) for thread-safe operation.
    """
    return _call_lib_func("strerror", (errnum,))


@proxy(signatures.get("memcpy"), jit_options={"cache": True})
def memcpy(dst, src, n):
    """POSIX `memcpy(dst, src, n) <https://man7.org/linux/man-pages/man3/memcpy.3.html>`_:
    copy `n` bytes from `src` to `dst`. Source and destination must NOT overlap
    — use `memmove` for that. Returns `dst`."""
    return _call_lib_func("memcpy", (dst, src, n))


@proxy(signatures.get("memmove"), jit_options={"cache": True})
def memmove(dst, src, n):
    """POSIX `memmove(dst, src, n) <https://man7.org/linux/man-pages/man3/memmove.3.html>`_:
    copy `n` bytes from `src` to `dst`, correctly handling overlapping regions.
    Returns `dst`."""
    return _call_lib_func("memmove", (dst, src, n))


@proxy(signatures.get("memset"), jit_options={"cache": True})
def memset(dst, c, n):
    """POSIX `memset(dst, c, n) <https://man7.org/linux/man-pages/man3/memset.3.html>`_:
    fill the first `n` bytes of `dst` with the byte value `c & 0xff`. Returns `dst`.
    """
    return _call_lib_func("memset", (dst, c, n))


@proxy(signatures.get("memcmp"), jit_options={"cache": True})
def memcmp(a, b, n):
    """POSIX `memcmp(a, b, n) <https://man7.org/linux/man-pages/man3/memcmp.3.html>`_:
    byte-wise compare the first `n` bytes of `a` and `b`. Returns negative, zero,
    or positive — same sign convention as `strcmp`, but on raw bytes (no NUL
    short-circuit)."""
    return _call_lib_func("memcmp", (a, b, n))


@proxy(signatures.get("memchr"), jit_options={"cache": True})
def memchr(s, c, n):
    """POSIX `memchr(s, c, n) <https://man7.org/linux/man-pages/man3/memchr.3.html>`_:
    pointer to the first byte equal to `c & 0xff` in the first `n` bytes at `s`,
    or 0 if not found."""
    return _call_lib_func("memchr", (s, c, n))


@proxy(signatures.get("getenv"), jit_options={"cache": True})
def getenv(name):
    """POSIX `getenv(name) <https://man7.org/linux/man-pages/man3/getenv.3.html>`_:
    pointer to the value string in the process environ table for variable `name`,
    or 0 if unset.

    The returned pointer is owned by the platform environ — do NOT mutate, free,
    or assume it survives a subsequent setenv/putenv. Callers that need a stable
    Python str should copy via `get_str_from_p_as_int` before mutating environ.
    """
    return _call_lib_func("getenv", (name,))
