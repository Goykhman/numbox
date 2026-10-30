"""Cache anchors, the files to which numbox writes the code that it generates at run time so that numba can cache it.

numba caches a function only if it can find the function's source file. Its key for a cache entry covers the
function's bytecode but not the constants or names that the bytecode refers to. numbox therefore writes each piece of
generated code to an anchor whose name ends in a digest of the code, and it compiles the code under that file name.
Two versions of the code that differ only in a literal or a name then have anchors, and cache files, of their own.
The anchors are under ``NUMBA_CACHE_DIR`` or numba's user cache directory. Before cached code compiles, numbox asks
numba whether it can cache a function of the anchor. If numba cannot cache it, the code compiles without a cache after
one warning that names the remedy. The utils page explains the anchors in its section on the cache-anchor mechanism.
"""
import ast
import errno
import hashlib
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

from numbox.core.configurations import (LONGEST_CACHE_FILE_NAME, _OVERLOAD_DIGITS, cache_remedy, check_cache_location,
                                        is_a_cache_error)


def _anchor_root(subdir: str = "numbox-structref") -> Path:
    from numba import config
    from numba.misc.appdirs import AppDirs
    if config.CACHE_DIR:
        return Path(config.CACHE_DIR) / subdir
    return Path(AppDirs(appname="numba", appauthor=False).user_cache_dir) / subdir


# The bound, in bytes, on each name that numbox puts into a file name (bounded_stem). The longest of numba's file
# names for a struct's code is the cache file of a method's jitted helper. With the bound, that name holds the
# anchor's stem of at most 57 bytes, the bounded names of the struct and the method, the method's 64-digit hash,
# numba's line number and its suffixes. That is under 230 bytes for names of any length, and within 255 bytes with
# the 21 bytes that numba's temporary name adds.
_STEM_MAX = 40


def _longest_cache_file_name(stem: str, code_txt: str) -> int:
    """Return the length in bytes of the longest cache file name that numba writes for a function of ``code_txt``.

    ``code_txt`` is the code anchored at a file whose name without its suffix is ``stem``. numba names a function's
    cache files after that stem, the function's qualified name without the angle brackets of ``<locals>``, the line of
    its first decorator and Python's version and ABI flags. A data file's name also holds the overload's number. numba
    writes each file first under a temporary name that is 21 bytes longer: ``.tmp.`` and sixteen hexadecimal digits.
    The length counts that temporary name. It counts the overload's number as ``_OVERLOAD_DIGITS`` digits, three,
    because numba numbers the overloads from one without a bound. The result is 0 if no function of the code is
    decorated.

    The anchor's check writes a file name of this length in the location that numba picks, as the package's check
    does with ``LONGEST_CACHE_FILE_NAME`` for numbox's own modules. A cache directory too close to the path limit then
    fails the check, and the code compiles without a cache after one warning. numba's own test of a location creates a
    temporary file that is unnamed on Linux and has a short name elsewhere. Such a directory accepts that file, so
    without this length the first save would fail while the struct is made. A fixed bound for every struct, such as
    251 bytes, would disable caching on a file system with a lower name limit, such as the 143 bytes of eCryptfs. It
    would do so even for a struct whose names need 58.
    """
    names = []

    def walk(node, outer, jitted):
        """Append the qualified name and line of each function under ``node`` that numba caches to ``names``."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{outer}{child.name}"
                # A function is included if it is decorated, or if it is defined inside a decorated function. numba
                # caches the constructor that an overload returns under the overload's qualified name and <locals>,
                # with the overload's options. A decorator of any kind qualifies, so the work builder's derive, which a
                # numbox decorator jits, is included. A plain function, at the top or in a class body, has no cache
                # file and is not included. A struct's method under its own name is such a function: it only calls the
                # jitted helper. Including a method name of two hundred characters would raise the length above 255
                # and disable the struct's cache needlessly.
                counts = jitted or bool(child.decorator_list)
                if counts:
                    # child.lineno is the line of the def statement. numba uses the line of the first decorator, which
                    # is earlier or the same, so the number of digits counted is never too small.
                    names.append((qualname, child.lineno))
                walk(child, f"{qualname}.<locals>.", counts)
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{outer}{child.name}.", jitted)
            else:
                walk(child, outer, jitted)

    walk(ast.parse(code_txt), "", False)
    version = f"py{sys.version_info[0]}{sys.version_info[1]}{getattr(sys, 'abiflags', '')}"
    return max(
        (len(f"{stem}.{qualname.replace('<', '').replace('>', '')}-{lineno}.{version}.{'9' * _OVERLOAD_DIGITS}.nbc.tmp."
             .encode("utf-8")) + 16
         for qualname, lineno in names), default=0)


# The text that ends each remedy in this module's warnings. Generated code compiles under the package's default jit
# options unless the caller passes options of its own, as make_structref, compile_kernel and the work builder accept,
# and compile_kernel's cache argument overrides both. NUMBOX_JIT_OPTIONS changes only the package's defaults, so the
# text names each of the three ways to disable the cache.
_SILENCE = (
    "compile without a cache to silence this warning: \"cache\" off in the jit options this code was given, or "
    "in its cache argument where it takes one, or NUMBOX_JIT_OPTIONS='{\"cache\": false}' where the options are "
    "the package's"
)


def bounded_stem(name: str) -> str:
    """Return ``name`` shortened to at most 40 bytes of UTF-8, for use in a file name.

    A name of 40 bytes or fewer is returned unchanged; nearly every name is that short. A longer name is truncated to
    the whole characters that fit in 31 bytes, followed by an underscore and the first eight hexadecimal digits of the
    SHA-256 of the whole name. The length is measured in UTF-8, as file systems measure it, so a character of another
    script counts for up to four bytes.

    numbox puts the names of a struct, its fields and its methods, and of the functions it generates code for, into
    the names of the anchors and of numba's cache files. Most file systems allow a name of at most 255 bytes, and
    numba's names repeat the anchor's name and add the function's name. Without this bound, a struct name of about 93
    characters would make them too long.
    """
    encoded = name.encode("utf-8")
    if len(encoded) <= _STEM_MAX:
        return name
    head = encoded[:_STEM_MAX - 9].decode("utf-8", errors="ignore")
    return f"{head}_{hashlib.sha256(encoded).hexdigest()[:8]}"


def _anchor_path(subdir: str, stem: str, code_txt: str) -> Path:
    """Return the path of the anchor for ``code_txt``: ``<stem>_<digest>.py`` in the subdirectory ``subdir``.

    The stem is ``stem`` shortened by :func:`bounded_stem`, and the digest is the first 16 hexadecimal digits of the
    SHA-256 of the code, so each version of the code has a file of its own. The subdirectory is under
    ``NUMBA_CACHE_DIR`` if it is set, and under numba's user cache directory otherwise. The utils page explains why
    in its section on the cache-anchor mechanism.
    """
    # numba's key for a cache entry hashes co_code and the closure values, but not co_consts or co_names (_index_key
    # in numba/core/caching.py). On Python 3.10 to 3.13, an edit that changes only a literal or a name leaves co_code
    # unchanged, because the bytecode refers to both by index. Python 3.14 writes the integers 0 to 255 into the
    # bytecode with LOAD_SMALL_INT, and 3.15 also writes -1, with LOAD_COMMON_CONSTANT; other integers, such as -2
    # and 256, still go through LOAD_CONST. The digest covers the whole code text, so it separates every version.
    digest = hashlib.sha256(code_txt.encode("utf-8")).hexdigest()[:16]
    return _anchor_root(subdir) / f"{bounded_stem(stem)}_{digest}.py"


def _structref_anchor_path(struct_name: str, code_txt: str) -> Path:
    return _anchor_path("numbox-structref", struct_name, code_txt)


def _materialize_anchor(path: Path, code_txt: str) -> None:
    # An anchor that exists is never written again, because its name holds the digest of its code.
    if path.exists():
        return
    # The code is written to a temporary file in the same directory, which is then renamed to the anchor's name, so
    # that another process finds either no anchor or a complete one. The temporary file is removed if the write fails.
    fd, tmp_str = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp-")
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(code_txt)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _anchor_or_error(path: Path, code_txt: str):
    """Write the anchor at ``path`` and check that numba can cache a function of it; return the error, or None.

    Any ``OSError`` from creating the directory or writing the file is returned, whatever its cause, because numba
    itself skips a cache location on any ``OSError``. The causes include ``NUMBA_CACHE_DIR`` pointing into a file, a
    full disk, and a path too long for the file system, through the directory or through a name that numbox made. The
    check is :func:`~numbox.core.configurations.check_cache_location`, the one that the package runs for its own
    modules. numba may choose the user's cache directory for the anchor if the anchor's own directory cannot hold its
    cache. The check's error is returned when it is a cache error, as
    :func:`~numbox.core.configurations.is_a_cache_error` determines: no locator for the file, a location that numba
    cannot use, or the ``UnicodeEncodeError`` that numba raises for a directory path that is not valid UTF-8. Any
    other error propagates.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _materialize_anchor(path, code_txt)
    except OSError as error:
        return error
    # The check writes a file name as long as the longest that numba will write for the code's functions
    # (_longest_cache_file_name), so that a directory too close to the path limit fails this check instead of failing
    # at numba's first save.
    try:
        check_cache_location(path, _longest_cache_file_name(path.stem, code_txt))
    except (RuntimeError, OSError, ValueError) as error:
        if not is_a_cache_error(error):
            raise
        return error
    return None


def _anchored_or_uncached(path: Path, code_txt: str, jit_options: dict) -> dict:
    """Write the anchor of ``code_txt`` at ``path``, and return the jit options to compile the code under.

    numba reads the anchor to cache the code, and it quotes the code from the anchor in its messages, such as a typing
    error. The anchor is therefore written whether or not caching is enabled. With caching disabled, a failed write is
    ignored, and the code still compiles under ``path`` as its file name; ``jit_options`` is returned unchanged. With
    caching enabled, :func:`_anchor_or_error` writes the anchor and checks the location. If it reports an error, the
    function warns once with a ``RuntimeWarning`` that names the remedy, and it returns the options with ``cache`` set
    to False. The code then compiles without a cache instead of failing at the write or at the first decorated
    function.
    """
    if not jit_options.get("cache"):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _materialize_anchor(path, code_txt)
        except OSError:
            pass
        return jit_options
    error = _anchor_or_error(path, code_txt)
    if error is None:
        return jit_options
    if isinstance(error, OSError) and getattr(error, "longest_file_name", 0):
        # The file system that holds the location allows no name as long as the longest name that numba writes for the
        # anchor's functions, which is the length that the check tried. A shorter path does not avoid the error, so the
        # remedy is NUMBA_CACHE_DIR on another file system.
        from numbox.core.configurations import _name_limit
        remedy = (
            f"numba's cache files for the functions anchored here have names of up to {error.longest_file_name} "
            f"bytes, longer than a name can be on that file system ({_name_limit(error.filename)}): NUMBA_CACHE_DIR "
            f"at a directory on a file system that takes a name that long, or {_SILENCE}"
        )
    elif isinstance(error, OSError) and error.errno == errno.ENAMETOOLONG:
        # numbox bounds the names that it makes, so the part of the path that is too long is the directory:
        # NUMBA_CACHE_DIR or the user's cache directory. A shorter NUMBA_CACHE_DIR moves the anchor out of either
        # directory.
        remedy = f"The path is too long for the file system: NUMBA_CACHE_DIR at a shorter path, or {_SILENCE}"
    elif isinstance(error, UnicodeEncodeError):
        # numba hashes the anchor's directory path as UTF-8 to name its cache directory under NUMBA_CACHE_DIR or the
        # user's cache directory, and it raises for a path that is not valid UTF-8. The anchor's directory is writable,
        # because numbox has just written the anchor there, so the remedy is NUMBA_CACHE_DIR at a UTF-8 path rather than
        # a writable directory.
        remedy = (
            f"numba hashes the anchor's directory path as UTF-8 to name its cache directory, and {path.parent} is "
            f"not: NUMBA_CACHE_DIR at a UTF-8 path, or {_SILENCE}"
        )
    else:
        remedy = f"Set NUMBA_CACHE_DIR to a writable directory, or {_SILENCE}"
    warnings.warn(
        f"numba cannot cache {path.name} here ({error}); it compiles without a cache. {remedy}",
        RuntimeWarning, stacklevel=3,
    )
    return {**jit_options, "cache": False}


def _cached_at_or_uncached(py_file: str, jit_options: dict) -> dict:
    """Return the jit options to compile code under that numbox compiles as part of the module file ``py_file``.

    ``jit_options`` is returned unchanged if caching is disabled or if numba can cache a function of ``py_file``.
    Otherwise the function warns once, with the remedy that :func:`~numbox.core.configurations.cache_remedy` returns
    for the file, and returns the options with ``cache`` set to False. ``make_graph`` compiles its kernel as part of
    ``builder.py``, and numba caches the kernel beside that file. Options that a caller passes are passed to numba
    unchanged, without the check that the package's default options undergo at import, so this function runs the same
    check for the file. It checks with ``LONGEST_CACHE_FILE_NAME``, which covers the kernel's names.
    """
    if not jit_options.get("cache"):
        return jit_options
    try:
        check_cache_location(py_file, LONGEST_CACHE_FILE_NAME)
    except (RuntimeError, OSError, KeyError, ValueError) as error:
        if not is_a_cache_error(error):
            raise
        failure = error
    else:
        return jit_options
    warnings.warn(
        f"numba cannot cache the code generated at {os.path.basename(py_file)} here ({failure}); it compiles "
        f"without a cache. {cache_remedy(py_file, failure, _SILENCE)}",
        RuntimeWarning, stacklevel=3,
    )
    return {**jit_options, "cache": False}


_ORPHAN_AGE_SECONDS = 60


def _orphan_anchor_sweep(subdir: str) -> None:
    """Delete the temporary anchor files, ``*.tmp-*``, that a killed writer left in the anchor subdirectory ``subdir``.

    Each module that writes anchors calls this function at import for its own subdirectory. A leftover file only uses
    disk space, so every error here is ignored. Only files older than ``_ORPHAN_AGE_SECONDS`` are deleted, because a
    call of :func:`_materialize_anchor` in another process has a temporary file of the same name pattern between
    ``mkstemp`` and ``replace``. Deleting that file would make its ``replace`` raise ``FileNotFoundError``, and the
    other process's import would fail.
    """
    try:
        root = _anchor_root(subdir)
        if not root.exists():
            return
        cutoff = time.time() - _ORPHAN_AGE_SECONDS
        for orphan in root.glob("*.tmp-*"):
            try:
                if orphan.stat().st_mtime < cutoff:
                    orphan.unlink()
            except OSError:
                pass
    except Exception:
        pass


_orphan_anchor_sweep("numbox-structref")
