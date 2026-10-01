"""Source-anchor machinery for dynamically-exec'd code.

Content-addressed anchors keep numba's per-overload cache correct
when two exec'd code blocks differ only in ``co_consts``. See the
"Cache-anchor mechanism" section in ``docs/numbox.utils.rst`` for
the rationale and references.
"""
import hashlib
import os
import tempfile
import time
from pathlib import Path


def _anchor_root(subdir: str = "numbox-structref") -> Path:
    from numba import config
    from numba.misc.appdirs import AppDirs
    if config.CACHE_DIR:
        return Path(config.CACHE_DIR) / subdir
    return Path(AppDirs(appname="numba", appauthor=False).user_cache_dir) / subdir


_STEM_MAX = 40


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
    root = _anchor_root(subdir)
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(code_txt.encode("utf-8")).hexdigest()[:16]
    return root / f"{bounded_stem(stem)}_{digest}.py"


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
