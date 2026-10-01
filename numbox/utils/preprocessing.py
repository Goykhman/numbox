"""Source-anchor machinery for dynamically-exec'd code.

Content-addressed anchors keep numba's per-overload cache correct
when two exec'd code blocks differ only in ``co_consts``. See the
"Cache-anchor mechanism" section in ``docs/numbox.utils.rst`` for
the rationale and references.
"""
import errno
import hashlib
import os
import tempfile
import time
import warnings
from pathlib import Path

from numbox.core.configurations import cache_remedy, check_cache_location, is_a_cache_error


def _anchor_root(subdir: str = "numbox-structref") -> Path:
    from numba import config
    from numba.misc.appdirs import AppDirs
    if config.CACHE_DIR:
        return Path(config.CACHE_DIR) / subdir
    return Path(AppDirs(appname="numba", appauthor=False).user_cache_dir) / subdir


_STEM_MAX = 40

# The options generated code compiles under are the package's unless the caller
# gave its own, as make_structref, compile_kernel and the builder take, and
# compile_kernel's cache argument overrides them all; the variable reaches only
# the package's.
_SILENCE = (
    "compile without a cache to silence this warning: \"cache\" off in the jit options this code was given, or "
    "in its cache argument where it takes one, or NUMBOX_JIT_OPTIONS='{\"cache\": false}' where the options are "
    "the package's"
)


def bounded_stem(name: str) -> str:
    """``name`` as the stem of a file name: as it is up to 40 bytes, else whole characters of it within 31 and a digest.

    An anchor's name, and the names numba gives the cache files of the functions it holds, carry the name of
    the struct or the function they were generated for, and a file system allows a name 255 bytes long; numba's
    repeat the stem and the generated function's name, so a struct named with about 93 characters overflowed
    them. Bounded, the longest of numba's file names, a method thunk's, is the anchor's stem of at most 57 bytes,
    the struct's and the method's bounded names, the method's 64-character hash, and numba's line number and
    suffixes: under 230 bytes whatever the names' lengths, and within the 255 with the 21 bytes numba's temporary
    name adds at the write. A name of 40 bytes or fewer, which is nearly every name, keeps the file names it
    had. The measure is the name's UTF-8, which is the file system's: a character of another script takes up to
    four bytes there.
    """
    encoded = name.encode("utf-8")
    if len(encoded) <= _STEM_MAX:
        return name
    head = encoded[:_STEM_MAX - 9].decode("utf-8", errors="ignore")
    return f"{head}_{hashlib.sha256(encoded).hexdigest()[:8]}"


def _anchor_path(subdir: str, stem: str, code_txt: str) -> Path:
    """Content-addressed on-disk source anchor for dynamically-exec'd code, its stem bounded.

    See the "Cache-anchor mechanism" section in
    ``docs/numbox.utils.rst`` for the rationale.
    """
    digest = hashlib.sha256(code_txt.encode("utf-8")).hexdigest()[:16]
    return _anchor_root(subdir) / f"{bounded_stem(stem)}_{digest}.py"


def _structref_anchor_path(struct_name: str, code_txt: str) -> Path:
    return _anchor_path("numbox-structref", struct_name, code_txt)


def _materialize_anchor(path: Path, code_txt: str) -> None:
    if path.exists():
        return
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
    """Write the anchor at ``path`` and ask numba for its cache location; the error where there is none, else None.

    Any ``OSError`` from making the directory or writing the file means there is no cache here, whatever its
    cause: ``NUMBA_CACHE_DIR`` pointing into a file, a full disk, a path too long for the file system, whether
    the directory makes it so or the name numbox made from a struct's or a function's; numba itself passes a
    location over on any ``OSError``. Then numba is asked, the question the package puts for its own modules,
    and no locator, or a location it cannot use, is the cache's; the answer may be the user's cache directory
    where the anchor's own cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _materialize_anchor(path, code_txt)
    except OSError as error:
        return error
    try:
        check_cache_location(path)
    except (RuntimeError, OSError, ValueError) as error:
        if not is_a_cache_error(error):
            raise
        return error
    return None


def _anchored_or_uncached(path: Path, code_txt: str, jit_options: dict) -> dict:
    """``jit_options`` to compile the code anchored at ``path`` under, the anchor written where it can be.

    numba reads the anchor to cache the code it names, and quotes the source from it in its messages, a typing
    error's among them, so it is written whenever it can be: with caching off a write that fails is nothing,
    the path serving as the code's filename as it did. With caching on the anchor is written and numba asked
    for its location, by ``_anchor_or_error``; where that gives an error, the code compiles without a cache
    after one warning naming the remedy, rather than dying at the write or at the first decorated function.
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
    if isinstance(error, OSError) and error.errno == errno.ENAMETOOLONG:
        # The names numbox makes are bounded, so the long part is the
        # directory, NUMBA_CACHE_DIR or the user's cache directory; a short
        # NUMBA_CACHE_DIR moves the anchor out of either.
        remedy = f"The path is too long for the file system: NUMBA_CACHE_DIR at a shorter path, or {_SILENCE}"
    else:
        remedy = f"Set NUMBA_CACHE_DIR to a writable directory, or {_SILENCE}"
    warnings.warn(
        f"numba cannot cache {path.name} here ({error}); it compiles without a cache. {remedy}",
        RuntimeWarning, stacklevel=3,
    )
    return {**jit_options, "cache": False}


def _cached_at_or_uncached(py_file: str, jit_options: dict) -> dict:
    """``jit_options`` to compile code anchored to the module file ``py_file`` under: as given where numba can cache
    a function of that file, else with ``cache`` off after one warning naming the remedy.

    ``make_graph``'s kernel is anchored to the builder's own file and cached beside it, and the options a caller
    gives reach numba as they are, past the package's answer, so the question is put for the file, the way the
    package puts it for its own; the remedy is the package's for the placement.
    """
    if not jit_options.get("cache"):
        return jit_options
    try:
        check_cache_location(py_file)
    except (RuntimeError, OSError, ValueError) as error:
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
    """Best-effort cleanup of orphaned ``.tmp-*`` anchor files from SIGKILL'd
    writers. Called at module import; failures are silent (the orphan is at
    worst harmless disk usage).

    Only sweeps files whose ``mtime`` is older than ``_ORPHAN_AGE_SECONDS``
    so a concurrent ``_materialize_anchor`` call in another process --
    which has the same ``.tmp-*`` shape between ``mkstemp`` and
    ``replace`` -- isn't unlinked mid-flight (the resulting
    ``FileNotFoundError`` on the in-progress writer's ``replace`` would
    abort its caller's import).
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
