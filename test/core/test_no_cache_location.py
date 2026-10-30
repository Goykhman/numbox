"""Tests of the check that disables caching where numba cannot cache numbox, and of the remedy that its warning names.

numba sets up a function's cache when it decorates the function, and it raises if it finds no location that it can
write in. :mod:`numbox.core.configurations` asks numba about numbox's own files when it is first imported, so that
numbox then compiles without a cache after one warning. The tests that import numbox do so in a child process with a
copy of the tree and cache directories of its own. They cover read-only installs, ``.zip``, ``.egg`` and ``.whl``
archives, ``.pyc`` files without their source, symbolic links, frozen applications and paths at Linux's length limit.
The tests of :func:`~numbox.core.configurations.cache_remedy` run in this process and replace numba's settings, the
platform and IPython through ``monkeypatch``. They cover each of numba's locators under ``NUMBA_CACHE_DIR`` and
``NUMBA_CACHE_LOCATOR_CLASSES``, notebook cells, names too long for the file system and paths that are not UTF-8. Some
cases depend on numba 0.68, which reads a ``.zip`` member's source. A test that needs a read-only directory is skipped
on Windows and when the tests run as root. The last tests cover the code that numbox generates at run time. numbox
writes that code to an anchor file, and the code compiles without a cache where numba cannot cache the anchor.
``test_jit_options.py`` tests how numbox reads ``NUMBOX_JIT_OPTIONS``.
"""
import ast
import compileall
import errno
import importlib.util
import marshal
import os
import py_compile
import pytest
import re
import shutil
import subprocess
import sys
import zipfile
from importlib.metadata import version
from pathlib import Path


from numbox.core.configurations import LONGEST_CACHE_FILE_NAME, numba_version

REPO = Path(__file__).resolve().parent.parent.parent

# On Windows, chmod does not stop a process from writing in a directory, and root writes in a directory whatever its
# mode. A test that needs a directory that it cannot write in is skipped in both cases.
needs_a_directory_it_cannot_write = pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="needs a directory this user cannot write to")

# The child imports numbox and calls three of its functions. numbox compiles the libm binding cos and
# get_str_from_p_as_int under jit_options when it imports their modules. array_data_p is a plain njit function, which
# numba compiles at its first call and does not cache. The child prints the file that it imported configurations
# from, so that a test can compare that file with the place that PYTHONPATH names.
IMPORT_AND_USE = (
    "import math\n"
    "import numpy as np\n"
    "import numbox.core.configurations as configurations\n"
    "from numbox.core.bindings.libm import cos\n"
    "from numbox.utils.lowlevel import array_data_p, get_str_from_p_as_int\n"
    "assert abs(cos(0.5) - math.cos(0.5)) < 1e-15\n"
    "text = np.frombuffer(b'numbox\\0', dtype=np.uint8).copy()\n"
    "assert get_str_from_p_as_int(array_data_p(text)) == 'numbox'\n"
    "print(configurations.__file__)\n"
)


# ARCHIVE_REMEDY is the remedy for a module whose source is not a file on disk, such as a member of an .egg, while
# NUMBA_CACHE_LOCATOR_CLASSES is unset.
ARCHIVE_REMEDY = (
    "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, install numbox with its "
    "source files on disk, unpacked from any archive, or import it from a .zip holding its source files, which numba "
    "caches in the user's cache directory"
)


def _archive(path, bytecode_naming_the_archive=()):
    """Write numbox's modules into a zip archive at ``path``, and return ``path``, which the caller puts on PYTHONPATH.

    ``bytecode_naming_the_archive`` holds paths relative to the repository, with forward slashes, such as
    ``numbox/core/bindings``. A module named there, or a module directly in a directory named there, is written to the
    archive as a ``.pyc`` alone. Each such ``.pyc`` is compiled to name its own path inside the archive, as the ``-d``
    option of ``compileall`` and the ``dfile`` argument of ``py_compile`` do.
    """
    with zipfile.ZipFile(path, "w") as zipped:
        for source in sorted((REPO / "numbox").rglob("*.py")):
            member = str(source.relative_to(REPO))
            as_bytecode = (source.parent.relative_to(REPO).as_posix() in bytecode_naming_the_archive
                           or source.relative_to(REPO).as_posix() in bytecode_naming_the_archive)
            if as_bytecode:
                compiled = py_compile.compile(str(source), cfile=str(path.parent / (source.name + "c")),
                                              dfile=os.path.join(str(path), member), doraise=True)
                zipped.write(compiled, member + "c")
                os.unlink(compiled)
            else:
                zipped.write(source, member)
    return path


def _run(env, cwd, warnings="always"):
    """Return the result of a child Python that runs ``IMPORT_AND_USE`` under ``env`` and the filter ``warnings``."""
    # Python puts the child's working directory, cwd, first on its path. The callers pass a directory outside the
    # checkout, so that the child imports numbox from the place that PYTHONPATH names.
    return subprocess.run([sys.executable, "-W", warnings, "-c", IMPORT_AND_USE],
                          capture_output=True, text=True, env=env, cwd=str(cwd))


def _index_files(cache_dir):
    """Return the sorted names of the index files, ending in ``.nbi``, that numba wrote under ``cache_dir``."""
    return sorted(path.name for path in Path(cache_dir).rglob("*.nbi"))


def _directory_of_length(root, length):
    """Return a new directory under ``root`` whose path is ``length`` characters long, whatever ``root``'s length.

    The function appends components of 200 characters as long as one more such component would keep the path at least
    two characters shorter than ``length``. The last component has the remaining length, at most 201 characters, so that
    no component is too long for the file system and the directory can be created.
    """
    deep = root
    while len(str(deep)) + 201 < length - 1:
        deep = deep / ("d" * 200)
    deep = deep / ("d" * (length - 1 - len(str(deep))))
    assert len(str(deep)) == length
    deep.mkdir(parents=True)
    return deep


@pytest.mark.parametrize("name", [
    "numbox-0.0.0-py3.12.egg", "numbox-0.0.0-py3-none-any.whl", "my.zip.dir/numbox-0.0.0-py3.12.egg",
])
def test_an_import_from_an_archive_compiles_uncached_with_one_warning_naming_the_remedy(tmp_path, name):
    """An import of numbox from an ``.egg`` or a ``.whl`` succeeds with one warning, and the remedy says that
    ``NUMBA_CACHE_DIR`` has no effect. numba caches no module of such an archive, so without the check the import
    would fail at the first decorated function. Spark ships ``.egg`` archives to its executors with ``--py-files``."""
    # NUMBA_CACHE_DIR names a writable directory, and the test asserts that numba writes nothing there. Under
    # my.zip.dir, numba's .zip locator accepts the file because its path contains ".zip". The locator then raises
    # ValueError, because no part of the path ends in .zip, and is_a_cache_error counts that error as a cache error.
    (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
    archive = _archive(tmp_path / name)
    env = dict(os.environ, PYTHONPATH=str(archive), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "NUMBA_CACHE_DIR has no effect here" in run.stderr and "Set NUMBA_CACHE_DIR" not in run.stderr
    assert _index_files(tmp_path / "cache") == []
    # With caching disabled, the check does not run, and the import issues no warning. Under -W error, a warning would
    # make the child fail.
    quiet = _run(dict(env, NUMBOX_JIT_OPTIONS='{"cache": false}'), tmp_path, warnings="error")
    assert quiet.returncode == 0, quiet.stderr


def test_options_without_a_cache_key_are_asked_for_the_sites_that_cache_under_them(tmp_path):
    """Options without a ``"cache"`` key still run the check, so the SQLite virtual-table module imports from an
    ``.egg`` with one warning. numbox caches its SQLite callbacks unless the options set ``"cache"`` to ``false``, and
    numba would otherwise raise at the first callback that numbox compiles."""
    archive = _archive(tmp_path / "numbox-0.0.0-py3.12.egg")
    env = dict(os.environ, PYTHONPATH=str(archive), NUMBA_CACHE_DIR=str(tmp_path / "cache"),
               NUMBOX_JIT_OPTIONS='{"boundscheck": false}')
    child = "import numbox.core.bindings.sqlite.vtable as m; print(m.__file__)"
    run = subprocess.run([sys.executable, "-W", "always", "-c", child], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path))
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr


windows_path_sep_bug = "numba 0.68.0 reads a .zip member's source by a name built with Windows's separator"


# windows_path_sep_bug begins the remedy for numba issue 10889 (https://github.com/numba/numba/issues/10889), which
# numba 0.68.1 fixes. On Windows, numba 0.68.0 builds the name of a .zip member with backslashes and reads the member's
# source by that name. zipfile then raises KeyError before numba tries any location. A test of a member on that
# platform and release would test the bug instead of its own case, so on_windows_under_numba_0_68_0 skips it.
on_windows_under_numba_0_68_0 = pytest.mark.skipif(
    os.name == "nt" and version("numba") == "0.68.0", reason=windows_path_sep_bug)


def _skip_for_windows_separator_bug(stderr):
    """Skip the test if ``stderr`` holds the remedy for numba issue 10889, a bug of numba 0.68.0 on Windows alone.

    On any other platform, that remedy would mean a fault in ``cache_remedy``, so the helper fails the test there
    instead of skipping it.
    """
    if windows_path_sep_bug in stderr:
        assert os.name == "nt", f"the Windows separator remedy on {os.name}: {stderr}"
        pytest.skip(windows_path_sep_bug)


def test_a_zip_import_is_cached_by_numba(tmp_path):
    """An import from a ``.zip`` gives no warning, and numba caches numbox's functions in the user's cache directory,
    not under ``NUMBA_CACHE_DIR``. The remedy for other archives offers a ``.zip``, and this test shows that numba
    caches one."""
    archive = _archive(tmp_path / "numbox.zip")
    home = tmp_path / "home"
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = _run(env, tmp_path)
    _skip_for_windows_separator_bug(run.stderr)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert "compiles without a cache" not in run.stderr, run.stderr
    assert _index_files(tmp_path / "cache") == []
    if os.name != "nt":
        # On Windows, numba asks the system for the user's cache directory, so HOME and XDG_CACHE_HOME do not change it.
        assert _index_files(home)


@needs_a_directory_it_cannot_write
def test_one_read_only_directory_among_writable_ones_takes_the_fallback(tmp_path):
    """If one directory of numbox is read-only and the user's cache directory cannot be written, the import warns once
    and offers ``NUMBA_CACHE_DIR``, and with the variable set it caches without a warning. numba caches in the
    ``__pycache__`` beside each source, so the check asks numba about each directory of numbox separately."""
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    home = tmp_path / "home"
    home.mkdir()
    read_only = [home, site / "numbox" / "core" / "bindings"]
    for path in read_only:
        path.chmod(0o555)
    try:
        env = dict(os.environ, PYTHONPATH=str(site), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"))
        env.pop("NUMBA_CACHE_DIR", None)
        env.pop("NUMBOX_JIT_OPTIONS", None)
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(site) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        assert "Set NUMBA_CACHE_DIR" in run.stderr
        cured = _run(dict(env, NUMBA_CACHE_DIR=str(tmp_path / "cache")), tmp_path, warnings="error")
        assert cured.returncode == 0, cured.stderr
        assert _index_files(tmp_path / "cache")
    finally:
        for path in read_only:
            path.chmod(0o755)


# Each child imports one module of numbox, which compiles that module's functions, and prints the module's file.
IMPORT_SQLITE_TYPEMAP = "import numbox.core.bindings.sqlite._typemap as m; print(m.__file__)"
IMPORT_LIBM = "import numbox.core.bindings.libm as m; print(m.__file__)"


def test_a_module_that_survives_as_pyc_alone_beside_sourced_ones_takes_the_fallback(tmp_path):
    """A module present as ``.pyc`` alone, beside modules with their source, makes the import warn once with the remedy
    for a source that is not a file on disk. numba looks up the source file that a module's code names, so the check
    asks numba about such a module by itself and not only about its directory."""
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    libm = site / "numbox" / "core" / "bindings" / "libm.py"
    assert compileall.compile_file(str(libm), quiet=1, legacy=True)
    libm.unlink()
    env = dict(os.environ, PYTHONPATH=str(site), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_LIBM], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and str(site) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "source is not a file on disk" in run.stderr and "Set NUMBA_CACHE_DIR" not in run.stderr


def test_a_pyc_alone_compiled_from_a_tree_still_on_disk_caches_by_that_tree(tmp_path):
    """A ``.pyc`` alone, compiled from a tree that is still on disk elsewhere, imports without a warning, and numba
    caches its functions under ``NUMBA_CACHE_DIR``. The module's code records the path of the file that the ``.pyc`` was
    compiled from, and numba looks up that file, not the missing ``.py`` beside the ``.pyc``."""
    built = tmp_path / "build" / "numbox" / "core" / "bindings" / "libm.py"
    built.parent.mkdir(parents=True)
    shutil.copy(REPO / "numbox" / "core" / "bindings" / "libm.py", built)
    assert compileall.compile_file(str(built), quiet=1, legacy=True)
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    libm = site / "numbox" / "core" / "bindings" / "libm.py"
    libm.unlink()
    shutil.copy(built.with_suffix(".pyc"), libm.with_suffix(".pyc"))
    env = dict(os.environ, PYTHONPATH=str(site), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", IMPORT_LIBM], capture_output=True,
                         text=True, env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and str(site) in run.stdout, run.stderr
    assert any(name.startswith("libm.") for name in _index_files(tmp_path / "cache")), "libm did not cache"


def test_a_directory_whose_modules_survive_as_pyc_alone_takes_the_fallback(tmp_path):
    """A directory whose modules are all present as ``.pyc`` alone makes the import warn once with the remedy for a
    source that is not a file on disk. The check finds that directory through its ``.pyc`` files, and the remedy does
    not offer ``NUMBA_CACHE_DIR``, which numba reads only for a source file on disk."""
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    sqlite = site / "numbox" / "core" / "bindings" / "sqlite"
    assert compileall.compile_dir(str(sqlite), quiet=1, legacy=True)
    for source in list(sqlite.glob("*.py")):
        source.unlink()
    env = dict(os.environ, PYTHONPATH=str(site), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_SQLITE_TYPEMAP], capture_output=True,
                         text=True, env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and str(sqlite) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "source is not a file on disk" in run.stderr and "Set NUMBA_CACHE_DIR" not in run.stderr


def test_a_sourceless_install_is_told_the_source_is_not_on_disk(tmp_path):
    """An install of ``.pyc`` files alone warns once with the remedy for a source that is not a file on disk, and
    numba writes nothing under ``NUMBA_CACHE_DIR``. A module's ``__file__`` names its ``.pyc``, but numba looks up
    the removed ``.py`` that the module's code names, so the check asks numba about the ``.py``."""
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    assert compileall.compile_dir(str(site), quiet=1, legacy=True)
    for source in list(site.rglob("*.py")):
        source.unlink()
    env = dict(os.environ, PYTHONPATH=str(site), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(site) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "source is not a file on disk" in run.stderr and "Set NUMBA_CACHE_DIR" not in run.stderr
    assert _index_files(tmp_path / "cache") == []


@pytest.mark.parametrize("tree", ["site", "numbox.zip.tree"])
def test_a_pyc_in_a_zip_asks_by_the_file_it_was_compiled_from(tmp_path, tree):
    """A ``.pyc`` member of a ``.zip``, compiled in a tree that no longer exists, makes the import warn once with the
    archive remedy. zipimport runs the ``.pyc``, whose code records the path that it was compiled from, and numba looks
    up that path, so the check asks numba about it."""
    # The second tree, numbox.zip.tree, lies beside the archive, and its path begins with the archive's path. numba's
    # .zip locator accepts the compiled-from path for its ".zip" and raises ValueError, because no part of the path
    # ends in .zip. The check compares whole parts of the path, so it does not mistake the tree for the archive.
    site = tmp_path / tree
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    libm = site / "numbox" / "core" / "bindings" / "libm.py"
    assert compileall.compile_file(str(libm), quiet=1, legacy=True)
    libm.unlink()
    archive = tmp_path / "numbox.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for member in sorted(path for path in site.rglob("*") if path.suffix in (".py", ".pyc")):
            zipped.write(member, str(member.relative_to(site)))
    shutil.rmtree(site)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_LIBM], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path))
    _skip_for_windows_separator_bug(run.stderr)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "source is not a file on disk" in run.stderr and "holding its source files" in run.stderr, run.stderr
    assert (repr(str(libm)) if tree == "site" else "No zip file found") in run.stderr, run.stderr


def test_a_stale_pyc_beside_its_source_in_a_zip_is_passed_over_as_zipimport_passes_it(tmp_path):
    """A stale ``.pyc`` beside its ``.py`` in a ``.zip`` gives no warning, and numba caches the module's functions from
    the archive. zipimport runs the ``.py`` when the ``.pyc`` beside it is stale by size or time, so the check does not
    ask numba about the path that the stale ``.pyc`` names."""
    # This test asserts the outcome alone, which a check that skipped every .pyc would also produce. The test
    # test_a_pyc_in_a_zip_asks_by_the_file_it_was_compiled_from asserts that the check asks about a .pyc that
    # zipimport runs.
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    libm = site / "numbox" / "core" / "bindings" / "libm.py"
    assert compileall.compile_file(str(libm), quiet=1, legacy=True)
    with libm.open("a") as source:
        source.write("\n# a line after the .pyc was compiled\n")
    archive = tmp_path / "numbox.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for member in sorted(path for path in site.rglob("*") if path.suffix in (".py", ".pyc")):
            zipped.write(member, str(member.relative_to(site)))
    shutil.rmtree(site)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_LIBM], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path))
    _skip_for_windows_separator_bug(run.stderr)
    assert run.returncode == 0 and run.stdout.strip().endswith("libm.py"), run.stderr
    assert "compiles without a cache" not in run.stderr, run.stderr
    if os.name != "nt":
        # On Windows, numba asks the system for the user's cache directory, so HOME and XDG_CACHE_HOME do not change it.
        assert _index_files(home), "libm's functions were not cached from the archive"


def test_a_stale_pyc_whose_source_in_the_zip_does_not_compile_is_passed_over(tmp_path):
    """A module of the ``.zip`` that nothing imports, with a stale ``.pyc`` and a ``.py`` that does not compile, neither
    fails the import of numbox nor makes it warn. zipimport would compile that ``.py`` and raise ``SyntaxError``, and
    the check skips a module that zipimport cannot run."""
    archive = _archive(tmp_path / "numbox.zip")
    stray = tmp_path / "stray.py"
    stray.write_text("compiled = True\n")
    assert compileall.compile_file(str(stray), quiet=1, legacy=True)
    with zipfile.ZipFile(archive, "a") as zipped:
        zipped.write(stray.with_suffix(".pyc"), "numbox/stray.pyc")
        zipped.writestr("numbox/stray.py", "def broken(:\n    pass\n")
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = _run(env, tmp_path)
    _skip_for_windows_separator_bug(run.stderr)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert "compiles without a cache" not in run.stderr, run.stderr


@pytest.mark.parametrize("damage", ["truncated", "lzma"])
def test_a_stray_pyc_in_a_zip_that_nothing_imports_is_passed_over(tmp_path, damage):
    """A ``.pyc`` member that zipimport cannot run, and that nothing imports, neither fails the import of numbox nor
    makes it warn. The check skips a module whose code zipimport cannot read."""
    # Both members begin with this interpreter's magic number. The truncated one fails to unmarshal, and zipimport
    # cannot decompress a member stored with LZMA compression.
    archive = _archive(tmp_path / "numbox.zip")
    with zipfile.ZipFile(archive, "a") as zipped:
        if damage == "truncated":
            zipped.writestr("numbox/stray.pyc", importlib.util.MAGIC_NUMBER + bytes(12) + b"\xe3\x00")
        else:
            data = importlib.util.MAGIC_NUMBER + bytes(12) + marshal.dumps(compile("stray = True\n", "stray.py", "exec"))
            zipped.writestr("numbox/stray.pyc", data, compress_type=zipfile.ZIP_LZMA)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = _run(env, tmp_path)
    _skip_for_windows_separator_bug(run.stderr)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert "compiles without a cache" not in run.stderr, run.stderr


@needs_a_directory_it_cannot_write
def test_a_zip_import_with_no_writable_user_cache_directory_compiles_uncached_with_a_warning(tmp_path):
    """An import from a ``.zip`` under a read-only home warns once, and the remedy asks for free space in the location
    or for write access to it. numba's ``.zip`` locator does not test that it can write in its location, so without the
    check numba's first save would raise ``PermissionError`` and fail the import."""
    archive = _archive(tmp_path / "numbox.zip")
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o555)
    try:
        env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
                   NUMBA_CACHE_DIR=str(tmp_path / "cache"))
        env.pop("NUMBOX_JIT_OPTIONS", None)
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        assert "NUMBA_CACHE_DIR has no effect here" in run.stderr
        assert "so make room there, or make it writable" in run.stderr, run.stderr
        # The archive remedy would offer a .zip, which the user already imports from.
        assert "import it from a .zip" not in run.stderr
    finally:
        home.chmod(0o755)


@needs_a_directory_it_cannot_write
def test_a_zip_whose_configurations_runs_from_a_pyc_compiled_from_a_tree_still_on_disk(tmp_path):
    """A ``.zip`` whose current ``configurations.pyc`` was compiled from a tree still on disk warns once when the home
    is read-only. The code of that ``.pyc`` names the tree, which numba could cache, so the check finds the archive
    through the module's ``__file__`` and asks numba about the archive's directories."""
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    configurations = site / "numbox" / "core" / "configurations.py"
    assert compileall.compile_file(str(configurations), quiet=1, legacy=True)
    archive = tmp_path / "numbox.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for member in sorted(path for path in site.rglob("*") if path.suffix in (".py", ".pyc")):
            zipped.write(member, str(member.relative_to(site)))
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o555)
    try:
        env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
                   NUMBA_CACHE_DIR=str(tmp_path / "cache"))
        env.pop("NUMBOX_JIT_OPTIONS", None)
        run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_LIBM], capture_output=True, text=True,
                             env=env, cwd=str(tmp_path))
        assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        assert "NUMBA_CACHE_DIR has no effect here" in run.stderr, run.stderr
    finally:
        home.chmod(0o755)


@pytest.mark.skipif(os.name == "nt", reason="numba's user cache directory on Windows is not under HOME")
def test_a_zip_import_whose_user_cache_directory_is_too_long_is_told_so(tmp_path):
    """An import from a ``.zip`` whose user cache directory has a component too long for the file system warns once,
    and the remedy asks for that directory at a shorter path. Write access would not let numba create the location,
    so the remedy does not ask for free space or for write access."""
    # HOME itself holds the long component, because numba puts the user's cache directory under HOME on macOS and
    # reads XDG_CACHE_HOME only on other systems.
    archive = _archive(tmp_path / "numbox.zip")
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(tmp_path / ("c" * 300)),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    for name in ("NUMBOX_JIT_OPTIONS", "XDG_CACHE_HOME"):
        env.pop(name, None)
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "too long for the file system, so put that directory" in run.stderr, run.stderr
    assert "make room there" not in run.stderr, run.stderr


def test_a_moved_zip_whose_pyc_members_name_its_old_path_compiles_uncached_and_is_told_why(tmp_path):
    """A ``.zip`` moved after its ``.pyc`` members were compiled to name its old path warns once, and the remedy names
    that path as missing. numba reads the archive's stamp at decoration, by the path that a member's code names, so
    the check reads the stamp too."""
    (tmp_path / "build").mkdir()
    archive = _archive(tmp_path / "build" / "numbox.zip", ("numbox/core/bindings",))
    moved = tmp_path / "numbox.zip"
    archive.rename(moved)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(moved), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_LIBM], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path))
    _skip_for_windows_separator_bug(run.stderr)
    assert run.returncode == 0 and str(moved) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert f"names {archive}" in run.stderr and "which is not there" in run.stderr, run.stderr


@pytest.mark.skipif(os.name == "nt", reason="a symlink needs a privilege on Windows; the cache directory is not under HOME")
def test_a_zip_import_whose_user_cache_directory_hangs_off_a_dangling_link_is_told_to_make_it(tmp_path):
    """A ``.zip`` import whose user cache directory lies under a dangling symbolic link warns once, and the remedy asks
    for the location to be made writable. The error is ``ENOENT``, as for a moved archive, but it names no path above
    the module, so the remedy does not mention ``.pyc`` members, of which this archive has none."""
    # HOME is the dangling link, so that the user's cache directory lies under it on macOS and on Linux alike.
    archive = _archive(tmp_path / "numbox.zip")
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere", target_is_directory=True)
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(tmp_path / "dangling"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    for name in ("NUMBOX_JIT_OPTIONS", "XDG_CACHE_HOME"):
        env.pop(name, None)
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "so make room there, or make it writable" in run.stderr, run.stderr
    assert "its .pyc was compiled" not in run.stderr, run.stderr


@needs_a_directory_it_cannot_write
def test_a_zip_import_whose_cache_directory_stopped_being_writable_compiles_uncached(tmp_path):
    """A ``.zip`` import whose user cache directory holds every entry but can no longer be written warns once. The
    check tests that the location can be written in, as numba tests every other kind of location, so that no function
    fails later at its first save there."""
    archive = _archive(tmp_path / "numbox.zip")
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    warm = _run(env, tmp_path)
    assert warm.returncode == 0, warm.stderr
    assert _index_files(home)
    read_only = [home, *(path for path in home.rglob("*") if path.is_dir())]
    for path in read_only:
        path.chmod(0o555)
    try:
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
    finally:
        for path in read_only:
            path.chmod(0o755)


@needs_a_directory_it_cannot_write
@pytest.mark.parametrize("parent, bytecode", [("", ()), ("container.zip", ()), ("", ("numbox/core/bindings",))],
                         ids=["sources", "under a directory named .zip", "bindings as .pyc naming the archive"])
def test_a_zip_import_whose_location_for_one_directory_stopped_being_writable_takes_the_fallback(
        tmp_path, parent, bytecode):
    """When the location of one directory of a ``.zip`` stops being writable, the import warns once and names that
    location. numba caches each directory of a ``.zip`` in a location of its own under the user's cache directory, so
    the check asks numba about each directory of the archive."""
    # The check takes the archive to be the first part of the path that ends in .zip and is a zip archive. It therefore
    # lists the archive's directories under container.zip too. A directory of .pyc members, compiled to name their own
    # paths in the archive as compileall -d compiles them, counts as a directory of the archive. numba 0.68 and later
    # read each member's source from the archive. They open container.zip as the archive in the second case and find
    # no source in the third, so in those cases the first import warns, and the test checks that remedy.
    (tmp_path / parent).mkdir(exist_ok=True)
    archive = _archive(tmp_path / parent / "numbox.zip", bytecode)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"))
    env.pop("NUMBA_CACHE_DIR", None)
    env.pop("NUMBOX_JIT_OPTIONS", None)
    warm = _run(env, tmp_path)
    assert warm.returncode == 0, warm.stderr
    if numba_version >= 68 and parent:
        assert warm.stderr.count("compiles without a cache") == 1, warm.stderr
        assert (f"numba takes the first part of the path ending in .zip, {tmp_path / parent}, for the archive, and it "
                "is a directory, which numba 0.68 and later open to read the member's source: put the archive under "
                "a path with no such directory above it, or NUMBOX_JIT_OPTIONS") in warm.stderr, warm.stderr
        return
    if numba_version >= 68 and bytecode:
        assert warm.stderr.count("compiles without a cache") == 1, warm.stderr
        assert ("numba 0.68 and later read a .zip member's source from the archive, and this one holds none under "
                "numbox/core/bindings/__init__.py, the module shipped as .pyc alone: to cache, install numbox with "
                "its source files on disk, unpacked from any archive, or import it from a .zip holding its source "
                "files, which numba caches in the user's cache directory; or NUMBOX_JIT_OPTIONS"
                ) in warm.stderr, warm.stderr
        return
    # numba's user cache directory is under XDG_CACHE_HOME on Linux and under Library/Caches in the home on macOS, so
    # the test searches the whole home for the location of the bindings directory.
    locations = [path for path in home.rglob("bindings_*") if path.is_dir()]
    assert len(locations) == 1, sorted(str(path.relative_to(home)) for path in home.rglob("*") if path.is_dir())
    locations[0].chmod(0o555)
    try:
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        # The remedy names the location that lost write access, not the user's cache directory above it.
        assert (f"no file can be written in that directory, {locations[0]} (Permission denied), so make room there, "
                "or make it writable") in run.stderr, run.stderr
    finally:
        locations[0].chmod(0o755)


def test_a_pyc_member_alone_beside_sourced_ones_in_a_zip_is_asked_for_itself(tmp_path):
    """A ``.zip`` member shipped as ``.pyc`` alone, beside members with their source, makes the import warn once from
    numba 0.68, and the import gives no warning under earlier releases. numba 0.68 and later read each member's source
    from the archive, so the check asks numba about such a member by itself and not only about its directory."""
    archive = _archive(tmp_path / "numbox.zip", ("numbox/core/bindings/libm.py",))
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"))
    env.pop("NUMBA_CACHE_DIR", None)
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_LIBM], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path))
    _skip_for_windows_separator_bug(run.stderr)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    if numba_version >= 68:
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        assert ("holds none under numbox/core/bindings/libm.py, the module shipped as .pyc alone: to cache, install "
                "numbox with its source files on disk") in run.stderr, run.stderr
    else:
        assert "compiles without a cache" not in run.stderr, run.stderr


def test_a_pyc_beside_its_source_in_a_zip_compiled_to_name_another_path_in_it_is_asked_for_itself(tmp_path):
    """A ``.pyc`` beside its ``.py`` in a ``.zip``, compiled to name a path that the archive does not hold, makes the
    import warn once from numba 0.68. The remedy asks for the ``.pyc`` to be compiled to name its own path or to be
    omitted, because numba 0.68 and later read the member's source by the name that the ``.pyc`` records."""
    # zipimport runs a .pyc before the .py beside it. An unchecked hash-based .pyc is never compared with the .py, so
    # zipimport runs it whatever the age of the .py.
    archive = tmp_path / "numbox.zip"
    libm = REPO / "numbox" / "core" / "bindings" / "libm.py"
    compiled = py_compile.compile(str(libm), cfile=str(tmp_path / "libm.pyc"),
                                  dfile=os.path.join(str(archive), "src", "numbox", "core", "bindings", "libm.py"),
                                  doraise=True, invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
    with zipfile.ZipFile(archive, "w") as zipped:
        for source in sorted((REPO / "numbox").rglob("*.py")):
            zipped.write(source, str(source.relative_to(REPO)))
        zipped.write(compiled, "numbox/core/bindings/libm.pyc")
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"))
    env.pop("NUMBA_CACHE_DIR", None)
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_LIBM], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path))
    _skip_for_windows_separator_bug(run.stderr)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    if numba_version >= 68:
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        # The archive holds the module's source, so the remedy must not call the module a .pyc shipped alone.
        assert ("holds none under src/numbox/core/bindings/libm.py" not in run.stderr
                or "shipped as .pyc alone" not in run.stderr), run.stderr
        assert ("by the name the module's code gives, src/numbox/core/bindings/libm.py, which "
                "numbox/core/bindings/libm.pyc was compiled to name, and this one holds none so named, while it holds "
                "the module's source, numbox/core/bindings/libm.py: to cache, compile the .pyc to name its path in the "
                "archive, or leave it out") in run.stderr, run.stderr
    else:
        assert "compiles without a cache" not in run.stderr, run.stderr


@pytest.mark.skipif(os.name == "nt", reason="a symlink needs a privilege on Windows")
def test_symlinks_that_cycle_inside_the_package_are_walked_once(tmp_path):
    """The import finishes without a warning when symbolic links in two subdirectories of numbox point to numbox's top
    directory. The check follows links but walks each real directory once, because the paths through two such links grow
    in number exponentially, up to the system's limit on links."""
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    (site / "numbox" / "core" / "up").symlink_to("..", target_is_directory=True)
    (site / "numbox" / "utils" / "up").symlink_to("..", target_is_directory=True)
    env = dict(os.environ, PYTHONPATH=str(site), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", IMPORT_LIBM], capture_output=True,
                         text=True, env=env, cwd=str(tmp_path), timeout=120)
    assert run.returncode == 0 and str(site) in run.stdout, run.stderr


@needs_a_directory_it_cannot_write
def test_a_symlinked_directory_of_the_package_answers_too(tmp_path):
    """A read-only directory of numbox reached through a symbolic link makes the import warn once and offer
    ``NUMBA_CACHE_DIR`` when the user's cache directory cannot be written either. The check follows symbolic links,
    because a linked directory can lie in a read-only tree."""
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    elsewhere = tmp_path / "elsewhere"
    shutil.move(str(site / "numbox" / "core" / "bindings"), str(elsewhere))
    (site / "numbox" / "core" / "bindings").symlink_to(elsewhere, target_is_directory=True)
    home = tmp_path / "home"
    home.mkdir()
    read_only = [home, elsewhere, *(path for path in elsewhere.rglob("*") if path.is_dir())]
    for path in read_only:
        path.chmod(0o555)
    try:
        env = dict(os.environ, PYTHONPATH=str(site), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"))
        env.pop("NUMBA_CACHE_DIR", None)
        env.pop("NUMBOX_JIT_OPTIONS", None)
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(site) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        assert "Set NUMBA_CACHE_DIR" in run.stderr
    finally:
        for path in read_only:
            path.chmod(0o755)


@needs_a_directory_it_cannot_write
def test_a_frozen_application_is_told_its_user_cache_directory(tmp_path):
    """A frozen application whose user cache directory can no longer be written warns once, and the remedy names that
    directory as one that numba could not use. numba caches a frozen application's modules in the user's cache
    directory, so the remedy does not offer the source files or a ``.zip``."""
    # The child sets sys.frozen, as PyInstaller does, so that numba treats the sourceless install as a frozen
    # application.
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    assert compileall.compile_dir(str(site), quiet=1, legacy=True)
    for source in list(site.rglob("*.py")):
        source.unlink()
    home = tmp_path / "home"
    home.mkdir()
    frozen = "import sys\nsys.frozen = True\n" + IMPORT_AND_USE
    env = dict(os.environ, PYTHONPATH=str(site), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"))
    env.pop("NUMBA_CACHE_DIR", None)
    env.pop("NUMBOX_JIT_OPTIONS", None)
    warm = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", frozen], capture_output=True,
                          text=True, env=env, cwd=str(tmp_path))
    assert warm.returncode == 0 and _index_files(home), warm.stderr
    read_only = [home, *(path for path in home.rglob("*") if path.is_dir())]
    for path in read_only:
        path.chmod(0o555)
    try:
        run = subprocess.run([sys.executable, "-W", "always", "-c", frozen], capture_output=True, text=True,
                             env=env, cwd=str(tmp_path))
        assert run.returncode == 0 and str(site) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        assert "frozen application" in run.stderr, run.stderr
        # For a frozen application, numba raises its error that no locator accepts the file. That error names no
        # directory and gives no reason, so the remedy names the user's cache directory, which lies under the home here.
        named = re.search(r"numba could not use that directory, (.+?): make it writable", run.stderr)
        assert named and Path(named.group(1)).is_relative_to(home), run.stderr
        assert "install numbox with its source files" not in run.stderr
    finally:
        for path in read_only:
            path.chmod(0o755)


@needs_a_directory_it_cannot_write
def test_a_read_only_install_warns_naming_numba_cache_dir_and_setting_it_caches(tmp_path):
    """A read-only install under a read-only home warns once and offers ``NUMBA_CACHE_DIR``, and with the variable set
    the import caches numbox without a warning. Here the source is on disk, and numba can write neither beside it nor
    in the user's cache directory."""
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    home = tmp_path / "home"
    home.mkdir()
    read_only = [home, *(path for path in site.rglob("*") if path.is_dir())]
    for path in read_only:
        path.chmod(0o555)
    try:
        env = dict(os.environ, PYTHONPATH=str(site), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"))
        env.pop("NUMBA_CACHE_DIR", None)
        env.pop("NUMBOX_JIT_OPTIONS", None)
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(site) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        assert "Set NUMBA_CACHE_DIR" in run.stderr
        cured = _run(dict(env, NUMBA_CACHE_DIR=str(tmp_path / "cache")), tmp_path, warnings="error")
        assert cured.returncode == 0, cured.stderr
        assert _index_files(tmp_path / "cache")
    finally:
        for path in read_only:
            path.chmod(0o755)


@pytest.mark.skipif(sys.platform != "linux", reason="a path of 4096 bytes is Linux's limit")
def test_a_location_too_close_to_the_path_limit_for_numbas_files_compiles_uncached_and_is_told_so(tmp_path):
    """A ``NUMBA_CACHE_DIR`` whose location for numbox ends 20 bytes short of Linux's path limit makes the import warn
    once and ask for a shorter ``NUMBA_CACHE_DIR``. numba's own test of the location passes there, because its
    temporary file usually has no name on Linux, but numba's cache files, with names of a hundred bytes or more, do not
    fit."""
    # The check creates a file with a name of LONGEST_CACHE_FILE_NAME bytes, which does not fit either. The first module
    # that the check asks about is configurations.py. Its location is the name of its directory and a hash of the
    # directory's path, under NUMBA_CACHE_DIR, so the test sets the length of NUMBA_CACHE_DIR from the length of that
    # name, whatever the checkout's path.
    from numba.core.caching import _CacheLocator
    location = _CacheLocator.get_suitable_cache_subpath(str(REPO / "numbox" / "core" / "configurations.py"))
    cache_dir = _directory_of_length(tmp_path, 4096 - 20 - 1 - len(location))
    env = dict(os.environ, PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(cache_dir))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(REPO / "numbox" / "core") in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "the path is too long for the file system: a shorter NUMBA_CACHE_DIR; or NUMBOX_JIT_OPTIONS" in (
        run.stderr), run.stderr
    assert not _index_files(cache_dir)


@needs_a_directory_it_cannot_write
@pytest.mark.skipif(sys.platform != "linux", reason="a path of 4096 bytes is Linux's limit")
def test_a_read_only_install_whose_user_cache_directory_is_too_long_is_told_to_shorten_that(tmp_path):
    """A read-only install whose location in the user's cache directory ends 20 bytes short of Linux's path limit warns
    once, and the remedy asks for the user's cache directory at a shorter path. With a short ``XDG_CACHE_HOME``, numba
    caches numbox there without a warning."""
    # NUMBA_CACHE_DIR is unset, and the __pycache__ beside each source cannot be written, so numba chooses its location
    # under the user's cache directory. The location of configurations.py, the first module that the check asks about,
    # ends 20 bytes short of the limit.
    from numba.core.caching import _CacheLocator
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    location = _CacheLocator.get_suitable_cache_subpath(str(site / "numbox" / "core" / "configurations.py"))
    xdg = _directory_of_length(tmp_path / "xdg", 4096 - 20 - 1 - len(location) - len("/numba"))
    read_only = [path for path in site.rglob("*") if path.is_dir()]
    for path in read_only:
        path.chmod(0o555)
    try:
        env = dict(os.environ, PYTHONPATH=str(site), XDG_CACHE_HOME=str(xdg))
        env.pop("NUMBA_CACHE_DIR", None)
        env.pop("NUMBOX_JIT_OPTIONS", None)
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(site) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        assert "at a shorter path, through XDG_CACHE_HOME or HOME" in run.stderr, run.stderr
        # NUMBA_CACHE_DIR is unset, so the remedy cannot ask for a shorter one. numbox installed at a shorter path would
        # not shorten the location under the user's cache directory.
        assert "a shorter NUMBA_CACHE_DIR" not in run.stderr and "numbox installed" not in run.stderr, run.stderr
        cured = _run(dict(env, XDG_CACHE_HOME=str(tmp_path / "short")), tmp_path, warnings="error")
        assert cured.returncode == 0, cured.stderr
        assert _index_files(tmp_path / "short")
    finally:
        for path in read_only:
            path.chmod(0o755)


# Each case gives the directory that NUMBA_CACHE_DIR names, or None if the variable is unset; the directory that holds
# the source; and the location that is too long. The directories are relative to tmp_path. The third value names the
# location: "NUMBA_CACHE_DIR" for the variable's location, "beside" for the __pycache__ beside the source, and "user"
# for numba's directory for the source under the user's cache directory.
LOCATION_CASES = {
    "NUMBA_CACHE_DIR": ("cache", "site", "NUMBA_CACHE_DIR"),
    "beside the source": (None, "site", "beside"),
    "the user's cache directory": (None, "site", "user"),
    "beside the source, NUMBA_CACHE_DIR passed over": ("cache", "site", "beside"),
    "beside a source under NUMBA_CACHE_DIR": ("cache", "cache/site", "beside"),
    "the user's cache directory under NUMBA_CACHE_DIR": ("user-cache", "site", "user"),
    "beside a source under the user's cache directory": (None, "user-cache/numba/site", "beside"),
}


@pytest.mark.parametrize("named", ["the location", "a file in it"])
@pytest.mark.parametrize("case", list(LOCATION_CASES))
def test_a_location_too_long_is_told_what_shortens_the_one_it_is(tmp_path, monkeypatch, case, named):
    """The remedy for a location too long for the file system names what shortens that location: a shorter
    ``NUMBA_CACHE_DIR``, the package at a shorter path, or the user's cache directory at a shorter path. The remedy
    compares whole paths in numba's order, so it distinguishes the locations where they nest, as for an install under a
    cache directory or for a ``NUMBA_CACHE_DIR`` above the user's cache directory."""
    # The user's cache directory lies under tmp_path, so that every case runs on every platform.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    cache_dir, source_dir, which = LOCATION_CASES[case]
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / cache_dir) if cache_dir else "")
    py_file = tmp_path / source_dir / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    subpath = _CacheLocator.get_suitable_cache_subpath(str(py_file))
    user_cache_dir = tmp_path / "user-cache" / "numba"
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    # If NUMBA_CACHE_DIR is set and numba chose another location, numba could not use the variable. The remedy then
    # names the variable as set instead of asking for it to be set.
    if cache_dir:
        instead = (f", or NUMBA_CACHE_DIR, which is set to {tmp_path / cache_dir} and numba could not use, made a "
                   "writable directory at a short path")
    else:
        instead = ", or NUMBA_CACHE_DIR set to a short path"
    where, cure = {
        "NUMBA_CACHE_DIR": (tmp_path / "cache" / subpath, "a shorter NUMBA_CACHE_DIR"),
        "beside": (py_file.parent / "__pycache__", f"numbduck installed at a shorter path{instead}"),
        "user": (user_cache_dir / subpath,
                 f"the user's cache directory, {user_cache_dir}, at a shorter path{configurations._moved_through()}"
                 f"{instead}"),
    }[which]
    # The check's error names the location. numba's first save names a file in it, and a caller can pass that error on.
    if named == "a file in it":
        where = where / "module.f-1.py312.nbi"
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(where))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence", package="numbduck")
    assert remedy == f"the path is too long for the file system: {cure}; or silence", remedy


def test_a_relative_numba_cache_dir_is_matched_as_the_location_numba_names(tmp_path, monkeypatch):
    """An error that names the location under a relative ``NUMBA_CACHE_DIR`` by its relative path gets the remedy for
    the variable's own location, a shorter ``NUMBA_CACHE_DIR``. numba joins the variable as it is set, so the location
    and its errors are relative, and the remedy reads both paths as absolute before it compares them."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "cache")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    location = os.path.join("cache", _CacheLocator.get_suitable_cache_subpath(str(py_file)))
    failure = OSError(errno.ENAMETOOLONG, "File name too long", location)
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == "the path is too long for the file system: a shorter NUMBA_CACHE_DIR; or silence", remedy


@pytest.mark.parametrize("locators, cure", [
    ("", "a shorter NUMBA_CACHE_DIR"),
    ("UserWideCacheLocator", "the user's cache directory, {user_cache_dir}, at a shorter path{moved_through}"),
    ("UserWideCacheLocator,UserProvidedCacheLocator",
     "the user's cache directory, {user_cache_dir}, at a shorter path{moved_through}"),
    ("InTreeCacheLocator,UserProvidedCacheLocator,UserWideCacheLocator", "a shorter NUMBA_CACHE_DIR"),
])
def test_a_location_too_long_that_is_numba_cache_dirs_and_the_users_cache_directorys_is_told_by_the_locator_order(
        tmp_path, monkeypatch, locators, cure):
    """When ``NUMBA_CACHE_DIR`` names the user's cache directory, the variable's location and the user-wide location
    are one path, and the remedy attributes an error there to the locator that numba tries first. It asks for a
    shorter ``NUMBA_CACHE_DIR`` if numba tries the variable first, and for the user's cache directory at a shorter path
    otherwise."""
    # numba's own test passed the variable's location, so numba continues to use the variable. A shorter path for the
    # user's cache directory would then not change what numba reads, and the remedy does not offer it.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    user_cache_dir = tmp_path / "user-cache" / "numba"
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(user_cache_dir))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    location = user_cache_dir / _CacheLocator.get_suitable_cache_subpath(str(py_file))
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(location))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    expected = cure.format(user_cache_dir=user_cache_dir, moved_through=configurations._moved_through())
    assert remedy == f"the path is too long for the file system: {expected}; or silence", remedy


@pytest.mark.parametrize("cache_dir_set", [False, True])
@pytest.mark.parametrize("locators, offered", [
    ("", True),
    ("IPythonCacheLocator,UserProvidedCacheLocator,InTreeCacheLocator,UserWideCacheLocator", True),
    ("InTreeCacheLocator,UserProvidedCacheLocator", False),
])
def test_a_location_too_long_that_is_none_of_numbas_is_named_with_the_variable_numba_tries_first(
        tmp_path, monkeypatch, locators, offered, cache_dir_set):
    """A caller's error that names a location that none of numba's locators would choose gets a remedy that names that
    location. The remedy offers ``NUMBA_CACHE_DIR`` only if numba tries the variable before every other locator that
    accepts the file, and it names a set variable without saying that numba could not use it."""
    # A caller's error that names such a location does not show what numba did with the variable. numba's IPython
    # locator, listed first in the second case, does not accept a file on disk, so numba still tries the variable
    # first. In the third case, the in-tree locator comes before the variable's.
    import numba
    from numbox.core.configurations import cache_remedy
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache_dir) if cache_dir_set else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = tmp_path / "package" / "module.py"
    py_file.parent.mkdir()
    py_file.write_text("")
    elsewhere = tmp_path / "elsewhere" / "cache"
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(elsewhere))
    remedy = cache_remedy(str(py_file), failure, "silence")
    if not offered:
        instead = ""
    elif cache_dir_set:
        instead = f", or NUMBA_CACHE_DIR, which is set to {cache_dir}, made a writable directory at a short path"
    else:
        instead = ", or NUMBA_CACHE_DIR set to a short path"
    assert remedy == (
        f"the path is too long for the file system: that location, {elsewhere}, at a shorter path{instead}; or silence"
    ), remedy


@pytest.mark.parametrize("filename", [None, ""])
@pytest.mark.parametrize("cache_dir", [None, "cache", "user-cache/numba"])
def test_a_location_too_long_that_the_error_does_not_name_is_told_the_locations_numba_could_have_taken(
        tmp_path, monkeypatch, cache_dir, filename):
    """A caller's ``ENAMETOOLONG`` that names no file, or an empty one, gets a remedy that lists every location that
    numba could have chosen, each once and in numba's order. The error does not show which location numba chose, and
    a ``NUMBA_CACHE_DIR`` set to the user's cache directory makes two of the locations one path."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    cache_dir = tmp_path / cache_dir if cache_dir else None
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache_dir) if cache_dir else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    # The working directory is the source's __pycache__. A remedy that read an empty name as the working directory would
    # attribute the error to the in-tree locator.
    (py_file.parent / "__pycache__").mkdir()
    monkeypatch.chdir(py_file.parent / "__pycache__")
    subpath = _CacheLocator.get_suitable_cache_subpath(str(py_file))
    locations = [str(py_file.parent / "__pycache__"), str(tmp_path / "user-cache" / "numba" / subpath)]
    if cache_dir:
        locations.insert(0, str(cache_dir / subpath))
        locations = list(dict.fromkeys(locations))
        instead = f", or NUMBA_CACHE_DIR, which is set to {cache_dir}, made a writable directory at a short path"
    else:
        instead = ", or NUMBA_CACHE_DIR set to a short path"
    failure = OSError(errno.ENAMETOOLONG, "File name too long", filename)
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == (
        f"the path is too long for the file system: the location numba took, one of {', '.join(locations[:-1])} or "
        f"{locations[-1]}, at a shorter path{instead}; or silence"), remedy


@pytest.mark.parametrize("failure, told", [
    pytest.param(OSError(errno.ENAMETOOLONG, "File name too long"),
                 "the path is too long for the file system: the location numba took, {location}, at a shorter path",
                 id="too long"),
    pytest.param(OSError(errno.EACCES, "Permission denied"),
                 "the location numba took, {location}, where no file can be written (Permission denied): make room "
                 "there, or make it writable", id="unwritable"),
])
def test_an_error_that_names_no_file_under_one_locator_is_told_its_one_location(tmp_path, monkeypatch, failure, told):
    """Under a ``NUMBA_CACHE_LOCATOR_CLASSES`` of one locator, an error that names no file gets a remedy that names the
    one location numba could have chosen, without the words "one of". The test covers a path too long and a location
    that cannot be written in."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "UserWideCacheLocator", raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    location = tmp_path / "user-cache" / "numba" / _CacheLocator.get_suitable_cache_subpath(str(py_file))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == f"{told.format(location=location)}; or silence", remedy


# Each case gives the directory that NUMBA_CACHE_DIR names, or None if the variable is unset, and the value of
# NUMBA_CACHE_LOCATOR_CLASSES. Each case also gives the location that the error names and the alternative that the
# remedy offers after it asks for free space or write access. The location "none" is none of numba's, and "unnamed"
# stands for an error that names no file.
UNWRITABLE_CASES = {
    "NUMBA_CACHE_DIR's": ("cache", "", "NUMBA_CACHE_DIR", ", or NUMBA_CACHE_DIR set to another writable directory"),
    "beside the source, NUMBA_CACHE_DIR unset": (None, "", "beside", ", or NUMBA_CACHE_DIR set to a writable directory"),
    "beside the source, NUMBA_CACHE_DIR passed over": (
        "cache", "", "beside",
        ", or NUMBA_CACHE_DIR, which is set to {cache_dir} and numba could not use, made a writable directory at a "
        "short path"),
    "the user's cache directory, another locator first": (None, "UserWideCacheLocator", "user", ""),
    "none of numba's": (None, "", "none", ", or NUMBA_CACHE_DIR set to a writable directory"),
    "unnamed, NUMBA_CACHE_DIR unset": (None, "", "unnamed", ", or NUMBA_CACHE_DIR set to a writable directory"),
    "unnamed, NUMBA_CACHE_DIR set": (
        "cache", "", "unnamed",
        ", or NUMBA_CACHE_DIR, which is set to {cache_dir}, made a writable directory at a short path"),
}


@pytest.mark.parametrize("error", [(errno.ENOSPC, "No space left on device"), (errno.EACCES, "Permission denied")])
@pytest.mark.parametrize("case", list(UNWRITABLE_CASES))
def test_a_location_numba_took_where_no_file_can_be_written_is_told_the_location_and_the_reason(
        tmp_path, monkeypatch, case, error):
    """An error from a location that refused a file, on a full disk or without permission, gets a remedy that names the
    location and the reason and asks for free space or write access. As the alternative, the remedy offers
    ``NUMBA_CACHE_DIR`` if numba tries the variable before that location, and the variable set to another directory if
    the location is the variable's own."""
    # numba chooses a location only if its own test can write a temporary file there. The check, or numba's first
    # save, writes a named file later, and a full disk, a quota or a change of permissions can make that write fail.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    cache_dir, locators, which, offer = UNWRITABLE_CASES[case]
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / cache_dir) if cache_dir else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    subpath = _CacheLocator.get_suitable_cache_subpath(str(py_file))
    location = {
        "NUMBA_CACHE_DIR": tmp_path / "cache" / subpath,
        "beside": py_file.parent / "__pycache__",
        "user": tmp_path / "user-cache" / "numba" / subpath,
        "none": tmp_path / "elsewhere" / "cache",
        "unnamed": None,
    }[which]
    number, reason = error
    # The remedy names a location that none of numba's locators would choose as the error gives it. For an error that
    # names no file, it lists the locations that numba could have chosen.
    if which == "unnamed":
        failure = OSError(number, reason)
        could_have_taken = [py_file.parent / "__pycache__", tmp_path / "user-cache" / "numba" / subpath]
        if cache_dir:
            could_have_taken.insert(0, tmp_path / "cache" / subpath)
        opening = f"the location numba took, one of {', '.join(map(str, could_have_taken[:-1]))} or {could_have_taken[-1]},"
    else:
        failure = OSError(number, reason, str(location))
        opening = f"that location, {location}," if which == "none" else f"numba caches this file in {location},"
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == (
        f"{opening} where no file can be written ({reason}): make room there, or make it "
        f"writable{offer.format(cache_dir=tmp_path / 'cache')}; or silence"), remedy


@pytest.mark.parametrize("placement", [
    "a source on disk", "a .zip member", "a .zip member in an ipykernel directory, naming IPython's location",
    "a .zip member, naming a location that is none of numba's",
])
def test_an_error_with_no_strerror_is_told_its_own_text_as_the_reason(tmp_path, monkeypatch, placement):
    """An ``OSError`` without a ``strerror`` gets a remedy that gives the error's own text as the reason. A caller can
    pass on such an error, built from a message alone or naming a file without a message, because
    ``is_a_cache_error`` counts every ``OSError``."""
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    # A stand-in for IPython.paths puts IPython's cache directory under tmp_path, whether or not IPython is installed.
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    failure = OSError("Disk quota exceeded on /home")
    user_cache_dir = tmp_path / "user-cache" / "numba"
    not_on_disk = (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        "effect here, because the source is not a file on disk: ")
    if placement == "a source on disk":
        py_file = tmp_path / "site" / "package" / "module.py"
        py_file.parent.mkdir(parents=True)
        py_file.write_text("")
        subpath = _CacheLocator.get_suitable_cache_subpath(str(py_file))
        expected = (
            f"the location numba took, one of {py_file.parent / '__pycache__'} or {user_cache_dir / subpath}, where no "
            f"file can be written ({failure}): make room there, or make it writable, or NUMBA_CACHE_DIR set to a "
            "writable directory; or silence")
    elif placement == "a .zip member":
        py_file = tmp_path / "bundle.zip" / "package" / "module.py"
        subpath = _CacheLocator.get_suitable_cache_subpath(str(py_file))
        expected = not_on_disk + (
            f"no file can be written in the location numba took, {user_cache_dir / subpath} ({failure}), so make "
            "room there, or make it writable, or silence")
    elif placement.startswith("a .zip member in an ipykernel directory"):
        py_file = tmp_path / "bundle.zip" / "ipykernel_123" / "cell.py"
        failure = OSError(errno.ENOSPC, None, str(ipython_dir / "numba_cache"))
        expected = (
            f"numba caches this file in {ipython_dir / 'numba_cache'}, where no file can be written ({failure}): make "
            "room there, or make it writable; or silence")
    else:
        py_file = tmp_path / "bundle.zip" / "package" / "module.py"
        failure = OSError(errno.ENOSPC, None, str(tmp_path / "elsewhere" / "cache"))
        expected = (
            "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: no file can be written at "
            f"that location, {tmp_path / 'elsewhere' / 'cache'} ({failure}), so make room "
            "there, or make it writable, or silence")
    assert failure.strerror is None and "None" not in expected.replace(str(failure), "")
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == expected, remedy


@pytest.mark.parametrize("locators, ipython, expected", [
    pytest.param(
        "IPythonCacheLocator,InTreeCacheLocator", "IPython.paths",
        "IPython's cache directory, {ipython_dir}, at a shorter path"
    ),
    ("", "IPython.paths",
     "IPython's cache directory, {ipython_dir}, at a shorter path, or NUMBA_CACHE_DIR set to a short path"),
    ("", "IPython.utils.path",
     "IPython's cache directory, {ipython_dir}, at a shorter path, or NUMBA_CACHE_DIR set to a short path"),
    ("", None, "that location, {named}, at a shorter path, or NUMBA_CACHE_DIR set to a short path"),
])
def test_a_cell_file_numbas_ipython_locator_takes_is_told_ipythons_cache_directory(
        tmp_path, monkeypatch, locators, ipython, expected):
    """A cell file on disk, in an ``ipykernel_`` directory, gets a remedy for a path too long that asks for IPython's
    cache directory at a shorter path. numba's IPython locator caches every such file in ``numba_cache`` under that
    directory. Without IPython, the remedy names the location as one that none of numba's locators chose."""
    # numba imports IPython to find its location and catches only OSError there. Without IPython, numba raises
    # ImportError at decoration, which is not a cache error, so a caller's error that names IPython's location names a
    # location that numba did not choose.
    import types
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    ipython_dir = tmp_path / "ipython"
    if ipython:
        # numba imports get_ipython_cache_dir from IPython.paths. If that module does not import, numba imports the
        # function from IPython.utils.path, where an older IPython defines it. The stand-in module gives a directory
        # under tmp_path.
        paths = types.ModuleType(ipython)
        paths.get_ipython_cache_dir = lambda: str(ipython_dir)
        monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
        monkeypatch.setitem(sys.modules, "IPython.paths", paths if ipython == "IPython.paths" else None)
        monkeypatch.setitem(sys.modules, "IPython.utils", types.ModuleType("IPython.utils"))
        monkeypatch.setitem(sys.modules, "IPython.utils.path", paths if ipython == "IPython.utils.path" else None)
    else:
        monkeypatch.setitem(sys.modules, "IPython", None)
        monkeypatch.setitem(sys.modules, "IPython.paths", None)
        monkeypatch.setitem(sys.modules, "IPython.utils.path", None)
    py_file = tmp_path / "ipykernel_123" / "cell.py"
    py_file.parent.mkdir()
    py_file.write_text("")
    named = ipython_dir / "numba_cache" / "cell-1.py312.nbi"
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(named))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    expected = expected.format(ipython_dir=ipython_dir, named=named)
    assert remedy == f"the path is too long for the file system: {expected}; or silence", remedy


@pytest.mark.parametrize("parent, name", [("", "bundle.zip"), ("container.zip", "bundle.zip"), ("", "bundle.egg")],
                         ids=["the archive", "under a directory named .zip", "an .egg"])
def test_a_zip_member_in_an_ipykernel_directory_is_asked_with_the_probes_source(tmp_path, monkeypatch, parent, name):
    """The check raises no error for a member of an archive in an ``ipykernel_`` directory, as numba raises none, and
    it creates ``numba_cache`` under IPython's cache directory. numba's IPython locator reads the function's source
    through the module's loader, so the probe's module reads the member's source as zipimport does."""
    # zipimport finds the archive under a directory whose name ends in .zip, and it reads an archive of any name, such
    # as an .egg. The check finds the archive in the same way.
    import linecache
    import types
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    (tmp_path / parent).mkdir(exist_ok=True)
    archive = tmp_path / parent / name
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ipykernel_123/cell.py", "def f():\n    pass\n")
    py_file = str(archive / "ipykernel_123" / "cell.py")
    configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert (ipython_dir / "numba_cache").is_dir()
    # After the check, linecache holds the member's own lines, as it does after any read of the member.
    assert linecache.getlines(py_file) == ["def f():\n", "    pass\n"]
    linecache.cache.pop(py_file, None)


@pytest.mark.parametrize("longest", [LONGEST_CACHE_FILE_NAME, 0],
                         ids=["with the package's bound on names", "with no bound, as a package may ask"])
def test_a_zip_member_in_an_ipykernel_directory_under_a_temporary_ipython_directory_takes_the_fallback(
        tmp_path, monkeypatch, longest):
    """The check raises ``FileNotFoundError`` for a member in an ``ipykernel_`` directory if IPython gives a new
    temporary directory on every call, and the remedy names that cause and ``IPYTHONDIR``. numba would create
    ``numba_cache`` in one directory and save in another, so numba's first save would fail in the same way."""
    # IPython gives such directories when it can write neither its cache directory nor its own. In both cases, the
    # check raises when it asks IPython for the location once more, because that directory does not exist. With a
    # bound on names, the check raises before it creates the file with a long name.
    import tempfile
    import types
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: tempfile.mkdtemp(dir=tmp_path)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ipykernel_123/cell.py", "def f():\n    pass\n")
    py_file = str(archive / "ipykernel_123" / "cell.py")
    with pytest.raises(FileNotFoundError) as raised:
        configurations.check_cache_location(py_file, longest)
    assert raised.value.errno == errno.ENOENT
    assert raised.value.filename.startswith(str(tmp_path)) and raised.value.filename.endswith("numba_cache")
    assert not os.path.exists(raised.value.filename)
    assert configurations.is_a_cache_error(raised.value)

    def left_behind():
        return len([path for path in tmp_path.iterdir() if path.name.startswith("tmp")])

    # Each request for IPython's cache directory creates a temporary directory that remains on disk. The check makes
    # three requests: numba's set-up of the cache, numba's test that the location can be written in, and the check's own
    # request for the location that numba would save in. The remedy makes two requests, and it makes the second one to
    # determine whether IPython gives a new directory on every call.
    assert left_behind() == 3
    assert configurations.cache_remedy(py_file, raised.value, "silence") == (
        "numba caches this file in numba_cache under IPython's cache directory, and IPython, able to write neither "
        "that directory nor its own, gives a new temporary directory on every call, so numba makes numba_cache in one "
        "and saves in another: set IPYTHONDIR, or XDG_CACHE_HOME on a POSIX system, to a writable directory; or silence")
    assert left_behind() == 5


@pytest.mark.parametrize("locators", [
    pytest.param("IPythonCacheLocator,InTreeCacheLocator", id="the IPython locator listed first"),
    pytest.param("", id="numba's order, the locators before IPython's passing the file over",
                 marks=needs_a_directory_it_cannot_write),
])
def test_a_cell_file_on_disk_under_a_temporary_ipython_directory_is_told_so(tmp_path, monkeypatch, locators):
    """A cell file on disk that reaches numba's IPython locator, under an IPython that gives a new temporary directory
    on every call, gets the remedy that names that cause and ``IPYTHONDIR``. numba reaches that locator if the list
    puts it first, or once the in-tree and user-wide locators skip the file."""
    import tempfile
    import types
    import numba
    import numba.core.caching
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "read-only" / "user-cache" / "numba")

    # numba's user-wide locator reads AppDirs from numba's caching module, and the remedy reads it from configurations,
    # so the test replaces both.
    monkeypatch.setattr(numba.core.caching, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: tempfile.mkdtemp(dir=tmp_path)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    kernel = tmp_path / "read-only" / "ipykernel_123"
    kernel.mkdir(parents=True)
    py_file = kernel / "cell.py"
    py_file.write_text("def f():\n    pass\n")
    read_only = [tmp_path / "read-only", kernel]
    if not locators:
        # The in-tree locator then cannot create the __pycache__ beside the cell, and the user-wide locator cannot
        # create the user's cache directory, which lies under the read-only directory.
        for path in read_only:
            path.chmod(0o555)
    try:
        with pytest.raises(FileNotFoundError) as raised:
            configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
        assert raised.value.filename.startswith(str(tmp_path)) and raised.value.filename.endswith("numba_cache")
        remedy = configurations.cache_remedy(str(py_file), raised.value, "silence")
    finally:
        for path in read_only:
            path.chmod(0o755)
    assert remedy == (
        "numba caches this file in numba_cache under IPython's cache directory, and IPython, able to write neither "
        "that directory nor its own, gives a new temporary directory on every call, so numba makes numba_cache in one "
        "and saves in another: set IPYTHONDIR, or XDG_CACHE_HOME on a POSIX system, "
        "to a writable directory; or silence"), remedy


@pytest.mark.parametrize("named", ["the directory", "a file numba writes in it"])
def test_a_cell_file_on_disk_refused_its_pycache_is_told_that_under_a_temporary_ipython_directory_too(
        tmp_path, monkeypatch, named):
    """A cell file on disk whose ``__pycache__`` refused a file gets the remedy for that ``__pycache__``, also under an
    IPython that gives a new temporary directory on every call. The in-tree locator accepted the file, so numba never
    asked IPython, and the remedy does not offer ``IPYTHONDIR``."""
    # The check's error names the __pycache__, and numba's first save names a file in it.
    import tempfile
    import types
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: tempfile.mkdtemp(dir=tmp_path)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    kernel = tmp_path / "ipykernel_123"
    kernel.mkdir()
    py_file = kernel / "cell.py"
    py_file.write_text("def f():\n    pass\n")
    refused = kernel / "__pycache__"
    if named != "the directory":
        refused = refused / "cell-1.py312.nbi.tmp.0123456789ab"
    failure = OSError(errno.EACCES, "Permission denied", str(refused))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == (
        f"numba caches this file in {kernel / '__pycache__'}, where no file can be written (Permission denied): make "
        "room there, or make it writable, or NUMBA_CACHE_DIR set to a writable directory; or silence"), remedy
    # The remedy asks IPython for its location once, to compare it with the path that the error names. It makes no
    # second request, which would test whether IPython gives a new directory on every call.
    assert len([path for path in tmp_path.iterdir() if path.name.startswith("tmp")]) == 1


def test_a_cell_under_a_locator_list_without_the_ipython_locator_is_told_to_list_it(tmp_path, monkeypatch):
    """A cell under a ``NUMBA_CACHE_LOCATOR_CLASSES`` without the IPython locator raises numba's error that no locator
    accepts the file, and the remedy asks for that locator to be listed. numba's IPython locator alone accepts a cell,
    so source files on disk or a ``.zip`` would not help."""
    # IPython holds a cell's source in linecache while the cell runs, and the test stores the cell's source there.
    import linecache
    import types
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "ZipCacheLocator", raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(tmp_path / "ipython")
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    cell = "<ipython-input-3-0123456789ab>"
    source = "def f():\n    pass\n"
    monkeypatch.setitem(linecache.cache, cell, (len(source), None, source.splitlines(True), cell))
    with pytest.raises(RuntimeError, match="no locator available") as raised:
        configurations.check_cache_location(cell, configurations.LONGEST_CACHE_FILE_NAME)
    assert configurations.is_a_cache_error(raised.value)
    assert configurations.cache_remedy(cell, raised.value, "silence") == (
        "numba caches a notebook cell through IPythonCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, "
        "ZipCacheLocator, leaves it out: list it; or silence")


@needs_a_directory_it_cannot_write
@pytest.mark.parametrize("ipython", ["left alone", "giving a temporary directory per call"])
def test_a_source_on_disk_numba_cannot_read_is_told_to_make_it_readable(tmp_path, monkeypatch, ipython):
    """A source on disk that cannot be read gets a remedy that asks for it to be made readable, also under an IPython
    that gives temporary directories. The remedy does not treat the source as a location that refused a file, and it
    does not offer ``IPYTHONDIR``."""
    # numba 0.68 and later read the source on disk to hash it into the stamp, so under those releases the check raises
    # numba's PermissionError, which names the source. numba's IPython locator reads the source in every release, but
    # here the in-tree locator accepts the file first, so under earlier releases the test builds the error itself.
    import tempfile
    import types
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    if ipython != "left alone":
        paths = types.ModuleType("IPython.paths")
        paths.get_ipython_cache_dir = lambda: tempfile.mkdtemp(dir=tmp_path)
        monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
        monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    kernel = tmp_path / "ipykernel_123"
    kernel.mkdir()
    py_file = kernel / "cell.py"
    py_file.write_text("def f():\n    pass\n")
    py_file.chmod(0)
    try:
        if numba_version >= 68:
            with pytest.raises(PermissionError) as raised:
                configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
            assert raised.value.filename == str(py_file)
            failure = raised.value
        else:
            failure = PermissionError(errno.EACCES, "Permission denied", str(py_file))
        assert configurations.is_a_cache_error(failure)
        remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    finally:
        py_file.chmod(0o644)
    assert remedy == (
        f"numba reads the function's source from its file, {py_file}, which cannot be read (Permission denied): make it "
        "readable; or silence"), remedy


@pytest.mark.parametrize("locators", [
    pytest.param("", id="numba's order"),
    pytest.param("ZipCacheLocator,InTreeCacheLocator", id="the .zip locator listed first"),
    pytest.param("UserProvidedCacheLocator,InTreeCacheLocator,UserWideCacheLocator", id="no .zip locator listed"),
])
def test_a_source_on_disk_under_a_directory_named_zip_is_told_the_directory_once_the_locators_before_passed(
        tmp_path, monkeypatch, locators):
    """A source on disk under a directory named ``data.zip`` gets a remedy for the ``IsADirectoryError`` that names
    that directory as the archive of numba's ``.zip`` locator. That locator takes the directory for the archive, and
    numba 0.68 and later open it for the source, so the remedy does not ask for free space in a directory that is not a
    cache location."""
    import numba
    import numba.core.caching
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "read-only" / "user-cache" / "numba")

    monkeypatch.setattr(numba.core.caching, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    container = tmp_path / "read-only" / "data.zip"
    package = container / "pkg"
    package.mkdir(parents=True)
    py_file = package / "m.py"
    py_file.write_text("x = 1\n")
    if locators and "ZipCacheLocator" not in locators:
        # Under a list with no .zip locator, the error that names the directory is one that a caller passes on, and the
        # remedy names the location that the error gives.
        expected = (
            f"that location, {container}, where no file can be written (Is a directory): make room there, or make it "
            "writable, or NUMBA_CACHE_DIR set to a writable directory; or silence")
    elif locators:
        # Listed first, the .zip locator accepts the file before any locator for a file on disk, so the remedy asks for
        # such a locator to be listed before it.
        expected = (
            "numba's .zip locator, which NUMBA_CACHE_LOCATOR_CLASSES puts before every locator for a source file on "
            f"disk, takes the first part of the path ending in .zip, {container}, for an archive, and it is a "
            "directory, which numba 0.68 and later open for the source: list UserProvidedCacheLocator, with "
            "NUMBA_CACHE_DIR set, InTreeCacheLocator or UserWideCacheLocator before it, or put the file under a path "
            "with no such directory; or silence")
    else:
        # In numba's order, the .zip locator accepts the file once the locators before it skip the file.
        expected = (
            "numba's .zip locator, reached once the locators before it passed this file over, takes the first part of "
            f"the path ending in .zip, {container}, for an archive, and it is a directory, which numba 0.68 and later "
            "open for the source: make one of those locators' locations writable, or NUMBA_CACHE_DIR set to a "
            "writable directory, or put the file under a path with no such directory; or silence")
    failure = IsADirectoryError(errno.EISDIR, "Is a directory", str(container))
    assert configurations.cache_remedy(str(py_file), failure, "silence") == expected
    # The check raises the error itself only under numba 0.68 and later. The test runs the check under numba's order
    # alone, and only where it can make directories read-only.
    if numba_version < 68 or os.name == "nt" or os.geteuid() == 0 or locators:
        return
    # With these directories read-only, the in-tree locator cannot create the __pycache__ beside the source, and the
    # user-wide locator cannot create the user's cache directory under them.
    read_only = [tmp_path / "read-only", package]
    for path in read_only:
        path.chmod(0o555)
    try:
        with pytest.raises(IsADirectoryError) as raised:
            configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
        assert raised.value.filename == str(container)
        assert configurations.is_a_cache_error(raised.value)
        assert configurations.cache_remedy(str(py_file), raised.value, "silence") == expected
    finally:
        for path in read_only:
            path.chmod(0o755)


def test_an_ipython_that_raises_for_its_cache_directory_leaves_the_remedy_a_remedy(tmp_path, monkeypatch):
    """If IPython raises an error other than ``OSError`` for its cache directory, the remedy for a cell still returns,
    and it asks for IPython's home to be made writable. The remedy asks IPython for the directory only to compose its
    text, so it treats any error there as no directory."""
    import types
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    paths = types.ModuleType("IPython.paths")

    def raises():
        raise RuntimeError("IPython misbehaved")

    paths.get_ipython_cache_dir = raises
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    cell = "<ipython-input-3-0123456789ab>"
    failure = RuntimeError(f"cannot cache function 'f': no locator available for file '{cell}'")
    remedy = configurations.cache_remedy(cell, failure, "silence")
    assert remedy == (
        "numba caches this file in numba_cache under IPython's cache directory, which IPython raised for, under a "
        "home it cannot write: make that home writable; or silence"), remedy


def test_an_error_naming_no_file_for_an_ipykernel_source_on_disk_is_not_blamed_on_ipython(tmp_path, monkeypatch):
    """An error that names no file, for a cell file on disk under an IPython that gives temporary directories, gets the
    remedy that lists the locations numba could have chosen, without ``IPYTHONDIR``. numba's first save can raise such
    an error on a full disk, and the error does not show which locator accepted the file."""
    import tempfile
    import types
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: tempfile.mkdtemp(dir=tmp_path)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    kernel = tmp_path / "ipykernel_123"
    kernel.mkdir()
    py_file = kernel / "cell.py"
    py_file.write_text("def f():\n    pass\n")
    remedy = configurations.cache_remedy(str(py_file), OSError(errno.ENOSPC, "No space left on device"), "silence")
    assert remedy.startswith("the location numba took, one of "), remedy
    assert str(kernel / "__pycache__") in remedy and "IPYTHONDIR" not in remedy, remedy
    assert remedy.endswith("where no file can be written (No space left on device): make room there, or make it "
                           "writable, or NUMBA_CACHE_DIR set to a writable directory; or silence"), remedy


def test_a_name_too_long_under_the_zip_locator_listed_before_the_in_tree_one_is_not_offered_the_pycache(
        tmp_path, monkeypatch):
    """If the list puts the ``.zip`` locator before the in-tree one, a directory name too long gets a remedy that asks
    for the in-tree locator to be listed before the ``.zip`` one, and not for a writable ``__pycache__``. The ``.zip``
    locator accepts a source on disk whose path contains ``.zip``, and numba reaches no locator after it."""
    import numba
    from numba.core.caching import _CacheLocator
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "ZipCacheLocator,InTreeCacheLocator", raising=False)
    # The source's directory has a name 30 characters shorter than the file system's limit. numba's directory named
    # after it is 41 characters longer, so its name exceeds the limit.
    limit = configurations._name_limit(str(tmp_path))
    parent = tmp_path / ("d" * (limit - 34) + ".zip")
    parent.mkdir()
    py_file = parent / "module.py"
    py_file.write_text("def f():\n    pass\n")
    location = os.path.join(str(tmp_path / "user-cache" / "numba"), _CacheLocator.get_suitable_cache_subpath(str(py_file)))
    remedy = configurations.cache_remedy(str(py_file), OSError(errno.ENAMETOOLONG, "File name too long", location), "silence")
    assert remedy.startswith("numba names the directory it caches this file in after the source's directory, "), remedy
    assert remedy.endswith(
        ": put the file in a directory of a shorter name, or list InTreeCacheLocator before ZipCacheLocator, which "
        "caches beside the source, in its __pycache__, under no such name; or silence"), remedy
    assert "writable" not in remedy


def test_a_path_too_long_the_user_wide_and_zip_locators_share_is_offered_the_variable_listed_between_them(
        tmp_path, monkeypatch):
    """When the user-wide and ``.zip`` locators choose the same location for a source on disk, a path too long there
    gets a remedy that offers ``NUMBA_CACHE_DIR`` too, if the list puts the variable's locator between them. The error
    cannot show which of the two locators accepted the file, and numba tries the variable before the ``.zip`` locator.
    """
    import numba
    from numba.core.caching import _CacheLocator
    import numbox.core.configurations as configurations

    user_cache_dir = tmp_path / "user-cache" / "numba"

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(user_cache_dir)

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES",
                        "UserWideCacheLocator,UserProvidedCacheLocator,ZipCacheLocator", raising=False)
    py_file = tmp_path / "data.zip" / "pkg" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("def f():\n    pass\n")
    location = user_cache_dir / _CacheLocator.get_suitable_cache_subpath(str(py_file))
    remedy = configurations.cache_remedy(
        str(py_file), OSError(errno.ENAMETOOLONG, "File name too long", str(location)), "silence")
    assert remedy == (
        f"the path is too long for the file system: the user's cache directory, {user_cache_dir}, at a shorter "
        f"path{configurations._moved_through()}, or NUMBA_CACHE_DIR set to a short path; or silence"), remedy


def test_a_name_too_long_behind_the_zip_locators_error_for_a_path_with_zip_in_it_is_told_the_name(
        tmp_path, monkeypatch):
    """A source on disk in a long directory whose name contains ``.zip`` without ending in it gets the remedy for a
    directory name too long, not one for the ``.zip`` locator's ``ValueError`` alone. The remedy offers a writable
    ``__pycache__`` and does not offer ``NUMBA_CACHE_DIR``, whose directory numba would name in the same way."""
    # The directory's name is 11 characters shorter than the file system's limit, and numba's directory named after
    # it is 41 characters longer, so the user-wide locator skips the file. The in-tree locator skips it too. numba's
    # .zip locator then accepts the file for the ".zip" in its path and raises ValueError, because no part of the path
    # ends in .zip.
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(numba.core.caching, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    limit = configurations._name_limit(str(tmp_path))
    package = tmp_path / ("proj.zip_" + "x" * (limit - 20))
    package.mkdir()
    py_file = package / "module.py"
    py_file.write_text("def f():\n    pass\n")
    # A file named __pycache__ prevents the in-tree locator from creating that directory beside the source.
    (package / "__pycache__").write_text("")
    with pytest.raises(ValueError, match="No zip file found") as raised:
        configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
    assert configurations.is_a_cache_error(raised.value)
    remedy = configurations.cache_remedy(str(py_file), raised.value, "silence")
    assert remedy.startswith("numba names the directory it caches this file in after the source's directory, "), remedy
    assert remedy.endswith(
        f": put the file in a directory of a shorter name, or make the __pycache__ beside the source, "
        f"{package / '__pycache__'}, writable, where InTreeCacheLocator caches it under no such name; or silence"), remedy
    assert "NUMBA_CACHE_DIR" not in remedy


@pytest.mark.skipif(not hasattr(os, "statvfs"), reason="the file system's name limit is asked through statvfs")
def test_the_name_limit_is_the_file_systems_own_asked_of_the_nearest_directory_that_exists(tmp_path, monkeypatch):
    """``_name_limit`` reads the file system's limit on names through ``os.statvfs``, from the nearest existing path at
    or above the location. It returns 255 if ``os.statvfs`` is missing or raises, if the file system reports 0, and for
    no location at all."""
    # The stand-in for os.statvfs reports 80, a limit that no common file system has, so that the result shows that the
    # limit comes from os.statvfs.
    import numbox.core.configurations as configurations
    asked = []
    real = os.statvfs

    class Limit:
        f_namemax = 80

    def statvfs(path):
        asked.append(path)
        return Limit()

    monkeypatch.setattr(os, "statvfs", statvfs)
    assert configurations._name_limit(str(tmp_path / "not" / "yet" / "made")) == 80
    assert asked == [str(tmp_path)]
    monkeypatch.setattr(os, "statvfs", real)
    monkeypatch.delattr(os, "statvfs")
    assert configurations._name_limit(str(tmp_path)) == 255

    def refusing(path):
        raise OSError(errno.EACCES, "Permission denied", path)

    # os.statvfs raises for a directory that the user cannot search, and the limit is then 255.
    monkeypatch.setattr(os, "statvfs", refusing, raising=False)
    assert configurations._name_limit(str(tmp_path)) == 255
    assert configurations._name_limit(None) == 255

    class NoLimit:
        f_namemax = 0

    # A file system with no limit of its own reports 0. The limit is then 255, because a limit of 0 would make every
    # name too long.
    monkeypatch.setattr(os, "statvfs", lambda path: NoLimit())
    assert configurations._name_limit(str(tmp_path)) == 255


def test_is_a_cache_error_admits_numbas_errors_by_their_text_and_no_other():
    """``is_a_cache_error`` admits any ``OSError``, and admits numba's ``ValueError`` and ``RuntimeError`` and, from
    numba 0.68, zipfile's ``KeyError`` only by their messages. Any other error, such as a dictionary's ``KeyError``, is
    not a cache error, so the check raises it unchanged instead of warning."""
    from numbox.core.configurations import is_a_cache_error
    assert is_a_cache_error(OSError(errno.EACCES, "Permission denied", "/x"))
    assert is_a_cache_error(ValueError("No zip file found in path"))
    assert not is_a_cache_error(ValueError("invalid literal for int()"))
    assert is_a_cache_error(RuntimeError("cannot cache function 'f': no locator available for file 'x.py'"))
    assert not is_a_cache_error(RuntimeError("cannot cache function 'f': something else"))
    assert is_a_cache_error(KeyError("There is no item named 'pkg/m.py' in the archive")) == (numba_version >= 68)
    assert not is_a_cache_error(KeyError("jit"))
    assert not is_a_cache_error(KeyError("There is no item named 'pkg/m.py' in the archive")) or numba_version >= 68
    assert not is_a_cache_error(TypeError("x"))


def test_a_reader_of_the_member_meanwhile_reads_the_members_own_lines_for_a_file_not_on_disk(tmp_path, monkeypatch):
    """A reader of a member's source while the check runs, as in another thread, reads the member's own lines and not
    the probe's. The probe belongs to a module of its own, whose loader reads the member's source from the archive, so
    the check never stores the probe's two lines in linecache under the member's name."""
    import inspect
    import linecache
    import types
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    source = "import os\n\n\ndef cell():\n    return os.sep\n"
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ipykernel_123/cell.py", source)
    py_file = str(archive / "ipykernel_123" / "cell.py")
    linecache.cache.pop(py_file, None)

    class ReadingTheMember:
        # A stand-in for zipimport, the loader of the member's own module.
        def get_source(self, name):
            return source

    member = types.ModuleType("ipykernel_123_cell")
    member.__file__ = py_file
    member.__loader__ = ReadingTheMember()
    exec(compile(source, py_file, "exec"), member.__dict__)
    monkeypatch.setitem(sys.modules, member.__name__, member)
    # The member's function starts at line 4, after the probe's two lines, so a read of the probe's lines would raise
    # inspect's error, "lineno is out of bounds".
    assert member.cell.__code__.co_firstlineno == 4
    seen = []
    real = configurations.CompileResultCacheImpl

    # The stand-in reads the member's source just before and just after numba picks the locator for the probe, as
    # another thread could.
    def reading_meanwhile(probe):
        seen.append(inspect.getsource(member.cell))
        locator = real(probe)
        seen.append(inspect.getsource(member.cell))
        return locator

    monkeypatch.setattr(configurations, "CompileResultCacheImpl", reading_meanwhile)
    configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert seen == ["def cell():\n    return os.sep\n"] * 2, seen
    assert linecache.getlines(py_file) == source.splitlines(True)
    assert (ipython_dir / "numba_cache").is_dir()
    linecache.cache.pop(py_file, None)


@pytest.mark.parametrize("given", ["Path", "bytes"])
def test_a_zip_member_in_an_ipykernel_directory_given_as_bytes_or_a_path_object_is_checked_as_its_string_is(
        tmp_path, monkeypatch, given):
    """``check_cache_location`` checks a member given as a path object or as bytes as it checks the member's string.
    compile stores the probe's file name as a string, so the check decodes the path once and gives the probe's module
    and loader that string."""
    import linecache
    import types
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ipykernel_123/cell.py", "def cell():\n    return 1\n")
    py_file = archive / "ipykernel_123" / "cell.py"
    linecache.cache.pop(str(py_file), None)
    as_given = py_file if given == "Path" else os.fsencode(py_file)
    configurations.check_cache_location(as_given, configurations.LONGEST_CACHE_FILE_NAME)
    assert (ipython_dir / "numba_cache").is_dir()
    assert linecache.getlines(str(py_file)) == ["def cell():\n", "    return 1\n"]
    linecache.cache.pop(str(py_file), None)


@pytest.mark.parametrize("placement", ["a .zip member", "on disk"])
def test_an_error_naming_no_file_under_a_temporary_ipython_directory_is_told_the_locations_with_its_reason(
        tmp_path, monkeypatch, placement):
    """An error with a reason and no file name, under an IPython that gives temporary directories, gets the remedy that
    lists the locations numba could have chosen, with the reason, and not the one that names ``IPYTHONDIR``. numba's
    first save under such an IPython raises an error that names the ``numba_cache`` it could not save in."""
    import tempfile
    import types
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: tempfile.mkdtemp(dir=tmp_path)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    if placement == "a .zip member":
        archive = tmp_path / "bundle.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("ipykernel_123/cell.py", "def cell():\n    return 1\n")
        py_file = archive / "ipykernel_123" / "cell.py"
    else:
        py_file = tmp_path / "ipykernel_123" / "cell.py"
        py_file.parent.mkdir()
        py_file.write_text("def cell():\n    return 1\n")
    failure = OSError(errno.ENOSPC, "No space left on device")
    assert failure.filename is None
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert "IPYTHONDIR" not in remedy and "temporary" not in remedy, remedy
    assert "the location numba took, " in remedy and "(No space left on device)" in remedy, remedy
    assert "numba_cache" in remedy, remedy
    # The list also holds a location under the user's cache directory: the .zip locator's location for the member, and
    # the user-wide locator's location for the file on disk.
    assert str(tmp_path / "user-cache" / "numba") in remedy, remedy


@pytest.mark.parametrize("placement", ["a .zip member", "on disk"])
def test_a_locator_of_none_of_numbas_families_listed_before_the_ipython_one_is_told_its_own_location(
        tmp_path, monkeypatch, placement):
    """A class outside numba's locator families that accepts the file, listed before the IPython locator, gets a remedy
    that names the location in the class's error, not ``IPYTHONDIR``. numba reaches no locator listed after one that
    accepts the file, so IPython's temporary directories cannot cause that error."""
    # OurLocator accepts every file, and its test of the location raises an error that names its own location.
    import tempfile
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    where = tmp_path / "our-cache"

    class OurLocator(_CacheLocator):
        def __init__(self, py_func, py_file):
            self._py_file = py_file

        def get_cache_path(self):
            return str(where)

        def get_source_stamp(self):
            return 0

        def get_disambiguator(self):
            return "x"

        def ensure_cache_path(self):
            raise PermissionError(errno.EACCES, "Permission denied", str(where))

        @classmethod
        def from_function(cls, py_func, py_file):
            return cls(py_func, py_file)

    ours = types.ModuleType("our_locators")
    ours.OurLocator = OurLocator
    monkeypatch.setitem(sys.modules, "our_locators", ours)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "our_locators.OurLocator,IPythonCacheLocator",
                        raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: tempfile.mkdtemp(dir=tmp_path)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    if placement == "a .zip member":
        archive = tmp_path / "bundle.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("ipykernel_123/cell.py", "def cell():\n    return 1\n")
        py_file = archive / "ipykernel_123" / "cell.py"
    else:
        py_file = tmp_path / "ipykernel_123" / "cell.py"
        py_file.parent.mkdir()
        py_file.write_text("def cell():\n    return 1\n")
    with pytest.raises(PermissionError) as raised:
        configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
    assert raised.value.filename == str(where)
    remedy = configurations.cache_remedy(str(py_file), raised.value, "silence")
    if placement == "a .zip member":
        assert remedy == (
            "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: no file can be written at "
            f"that location, {where} (Permission denied), so make room there, or make it writable, or silence"), remedy
    else:
        assert remedy == (
            f"that location, {where}, where no file can be written (Permission denied): make room there, or make it "
            "writable; or silence"), remedy


@needs_a_directory_it_cannot_write
def test_a_zip_members_location_that_cannot_be_made_is_told_the_directory_refusing_it(tmp_path, monkeypatch):
    """A ``.zip`` member whose location cannot be created gets a remedy that names the location and the existing
    directory under which numba cannot create it. numba's ``.zip`` locator does not create its location when it
    accepts the file, so the check's error names a directory that does not exist, as the error of numba's first save
    does."""
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderHome:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "home" / ".cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderHome)
    monkeypatch.setattr(numba.core.caching, "AppDirs", UserCacheUnderHome)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("package/module.py", "def f():\n    pass\n")
    py_file = str(archive / "package" / "module.py")
    refusing = tmp_path / "home" / ".cache"
    refusing.mkdir(parents=True)
    refusing.chmod(0o555)
    try:
        with pytest.raises(PermissionError) as raised:
            configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
        location = raised.value.filename
        assert location.startswith(str(refusing / "numba") + os.sep) and not os.path.isdir(location)
        remedy = configurations.cache_remedy(py_file, raised.value, "silence")
    finally:
        refusing.chmod(0o755)
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        f"effect here, because the source is not a file on disk: no file can be written in that directory, {location}, "
        f"which cannot be made under {refusing} (Permission denied), so make room there, or make it writable, or "
        "silence"), remedy


@pytest.mark.parametrize("placement", ["a .zip member", "on disk"])
def test_a_locator_of_none_of_numbas_families_that_passes_the_file_over_leaves_the_ipython_one_reached(
        tmp_path, monkeypatch, placement):
    """A class outside numba's families whose ``from_function`` skips the file, listed before the IPython locator, does
    not prevent numba from reaching that locator, and the remedy names IPython's temporary directories and
    ``IPYTHONDIR``. The remedy asks such a class about the file with a probe, as numba asks it, and does not assume that
    the class accepts every file."""
    import tempfile
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class OurPasser(_CacheLocator):
        @classmethod
        def from_function(cls, py_func, py_file):
            return None

    ours = types.ModuleType("our_locators")
    ours.OurPasser = OurPasser
    monkeypatch.setitem(sys.modules, "our_locators", ours)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "our_locators.OurPasser,IPythonCacheLocator",
                        raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: tempfile.mkdtemp(dir=tmp_path)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    if placement == "a .zip member":
        archive = tmp_path / "bundle.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("ipykernel_123/cell.py", "def cell():\n    return 1\n")
        py_file = archive / "ipykernel_123" / "cell.py"
    else:
        py_file = tmp_path / "ipykernel_123" / "cell.py"
        py_file.parent.mkdir()
        py_file.write_text("def cell():\n    return 1\n")
    with pytest.raises(FileNotFoundError) as raised:
        configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
    assert raised.value.filename.endswith(os.sep + "numba_cache")
    remedy = configurations.cache_remedy(str(py_file), raised.value, "silence")
    assert remedy == (
        "numba caches this file in numba_cache under IPython's cache directory, and IPython, able to write neither "
        "that directory nor its own, gives a new temporary directory on every call, so numba makes numba_cache in one "
        "and saves in another: set IPYTHONDIR, or XDG_CACHE_HOME on a POSIX system, "
        "to a writable directory; or silence"), remedy


@pytest.mark.parametrize("placement", ["a .zip member", "on disk"])
def test_a_location_of_a_locator_listed_after_a_class_of_none_of_numbas_families_that_took_the_file_is_none_of_numbas(
        tmp_path, monkeypatch, placement):
    """A class outside numba's families that accepts the file is the last locator numba reaches, so an error that names
    the location of a locator listed after the class names a location that numba did not choose. The remedy names that
    location as the error gives it and does not say that numba caches the file there."""
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    where = tmp_path / "our-cache"

    class OurLocator(_CacheLocator):
        def __init__(self, py_func, py_file):
            self._py_file = py_file

        def get_cache_path(self):
            return str(where)

        def get_source_stamp(self):
            return 0

        def get_disambiguator(self):
            return "x"

        @classmethod
        def from_function(cls, py_func, py_file):
            return cls(py_func, py_file)

    ours = types.ModuleType("our_locators")
    ours.OurLocator = OurLocator
    monkeypatch.setitem(sys.modules, "our_locators", ours)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "our_locators.OurLocator,IPythonCacheLocator",
                        raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    if placement == "a .zip member":
        archive = tmp_path / "bundle.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("ipykernel_123/cell.py", "def cell():\n    return 1\n")
        py_file = archive / "ipykernel_123" / "cell.py"
    else:
        py_file = tmp_path / "ipykernel_123" / "cell.py"
        py_file.parent.mkdir()
        py_file.write_text("def cell():\n    return 1\n")
    configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
    assert where.is_dir() and not ipython_dir.exists()
    named = ipython_dir / "numba_cache"
    remedy = configurations.cache_remedy(str(py_file), PermissionError(errno.EACCES, "Permission denied", str(named)),
                                         "silence")
    assert "numba caches this file in" not in remedy, remedy
    if placement == "a .zip member":
        assert remedy == (
            "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: no file can be written at "
            f"that location, {named} (Permission denied), so make room there, or make it writable, or silence"), remedy
    else:
        assert remedy == (
            f"that location, {named}, where no file can be written (Permission denied): make room there, or make it "
            "writable; or silence"), remedy


@pytest.mark.parametrize("placement", ["a .zip member", "on disk"])
def test_a_locator_of_none_of_numbas_families_whose_from_function_raises_takes_the_file(tmp_path, monkeypatch, placement):
    """A class outside numba's families whose ``from_function`` raises counts as accepting the file, because numba
    raises that error at decoration and reaches no locator listed after the class. The remedy names the location that
    the error gives and does not offer ``IPYTHONDIR`` for the IPython locator listed after the class."""
    import tempfile
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    where = tmp_path / "our-cache"

    class OurRaiser(_CacheLocator):
        @classmethod
        def from_function(cls, py_func, py_file):
            raise PermissionError(errno.EACCES, "Permission denied", str(where))

    ours = types.ModuleType("our_locators")
    ours.OurRaiser = OurRaiser
    monkeypatch.setitem(sys.modules, "our_locators", ours)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "our_locators.OurRaiser,IPythonCacheLocator",
                        raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: tempfile.mkdtemp(dir=tmp_path)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    if placement == "a .zip member":
        archive = tmp_path / "bundle.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("ipykernel_123/cell.py", "def cell():\n    return 1\n")
        py_file = archive / "ipykernel_123" / "cell.py"
    else:
        py_file = tmp_path / "ipykernel_123" / "cell.py"
        py_file.parent.mkdir()
        py_file.write_text("def cell():\n    return 1\n")
    assert configurations._takes(OurRaiser, str(py_file))
    with pytest.raises(PermissionError) as raised:
        configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
    assert raised.value.filename == str(where)
    remedy = configurations.cache_remedy(str(py_file), raised.value, "silence")
    if placement == "a .zip member":
        assert remedy == (
            "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: no file can be written at "
            f"that location, {where} (Permission denied), so make room there, or make it writable, or silence"), remedy
    else:
        assert remedy == (
            f"that location, {where}, where no file can be written (Permission denied): make room there, or make it "
            "writable; or silence"), remedy


@on_windows_under_numba_0_68_0
def test_a_zip_members_location_under_a_file_is_told_the_file_in_its_path(tmp_path, monkeypatch):
    """A ``.zip`` member whose location lies under a file gets a remedy that names that file and asks for it to be put
    elsewhere. The user's cache directory lies under a file if ``XDG_CACHE_HOME`` names a path through one, and free
    space or write access in the directory above that file would not help."""
    import numba
    import numbox.core.configurations as configurations

    a_file = tmp_path / "home" / "afile"
    a_file.parent.mkdir()
    a_file.write_text("not a directory\n")

    class UserCacheUnderTheFile:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(a_file / "c" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTheFile)
    monkeypatch.setattr(numba.core.caching, "AppDirs", UserCacheUnderTheFile)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("package/module.py", "def f():\n    pass\n")
    py_file = str(archive / "package" / "module.py")
    with pytest.raises(OSError) as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    location = raised.value.filename
    assert location.startswith(str(a_file) + os.sep) and not os.path.exists(location)
    remedy = configurations.cache_remedy(py_file, raised.value, "silence")
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        f"effect here, because the source is not a file on disk: no file can be written in that directory, {location}, "
        f"which cannot be made, {a_file} being a file in its path ({raised.value.strerror}), so put that file "
        "elsewhere, or silence"), remedy


@pytest.mark.parametrize("error", ["no archive in the path", "a directory named .zip"])
def test_the_zip_locators_errors_after_a_class_of_none_of_numbas_families_that_took_the_file_are_a_callers(
        tmp_path, monkeypatch, error):
    """If the list puts the ``.zip`` locator after a class outside numba's families that accepts the file, the
    ``.zip`` locator's errors are errors that a caller passes on, and the remedy does not attribute them to that
    locator. numba reaches no locator listed after the class, and the check passes under that list."""
    # The errors are the ValueError for a path that contains ".zip" with no part that ends in .zip, and the
    # IsADirectoryError for a directory named .zip above a source on disk.
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    where = tmp_path / "our-cache"

    class OurLocator(_CacheLocator):
        def __init__(self, py_func, py_file):
            self._py_file = py_file

        def get_cache_path(self):
            return str(where)

        def get_source_stamp(self):
            return 0

        def get_disambiguator(self):
            return "x"

        @classmethod
        def from_function(cls, py_func, py_file):
            return cls(py_func, py_file)

    ours = types.ModuleType("our_locators")
    ours.OurLocator = OurLocator
    monkeypatch.setitem(sys.modules, "our_locators", ours)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "our_locators.OurLocator,ZipCacheLocator",
                        raising=False)
    if error == "no archive in the path":
        py_file = tmp_path / "not.zipped" / "module.py"
        py_file.parent.mkdir()
        py_file.write_text("x = 1\n")
        configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
        remedy = configurations.cache_remedy(str(py_file), ValueError("No zip file found in path"), "silence")
        assert "passed the file over" not in remedy and ".zip locator" not in remedy, remedy
        assert remedy == (
            "numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, our_locators.OurLocator,ZipCacheLocator, says: make "
            "one of those locations writable, or silence"), remedy
    else:
        container = tmp_path / "data.zip"
        py_file = container / "package" / "module.py"
        py_file.parent.mkdir(parents=True)
        py_file.write_text("x = 1\n")
        configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
        remedy = configurations.cache_remedy(str(py_file), OSError(errno.EISDIR, "Is a directory", str(container)),
                                             "silence")
        assert "passed this file over" not in remedy and ".zip locator" not in remedy, remedy
        assert remedy == (
            f"that location, {container}, where no file can be written (Is a directory): make room there, or make it "
            "writable; or silence"), remedy
    assert where.is_dir()


@pytest.mark.parametrize("beside", ["its source beside it", "alone"])
def test_a_keyerror_for_a_pyc_compiled_to_name_another_path_in_the_archive_is_told_the_pyc(tmp_path, beside):
    """A ``KeyError`` for a member whose ``.pyc`` was compiled to name another path in the archive gets a remedy that
    names that ``.pyc``, if the archive holds the module's source beside it. The remedy then asks for the ``.pyc`` to be
    compiled to name its own path or omitted from the archive, and it asks for source files only for a ``.pyc`` shipped
    alone."""
    # numba 0.68 and later read a member's source by the name that the module's code gives, and zipfile raises
    # KeyError for a name that the archive does not hold. zipimport runs an unchecked hash-based .pyc without comparing
    # it with the .py beside it.
    import numbox.core.configurations as configurations

    archive = tmp_path / "bundle.zip"
    source = tmp_path / "b.py"
    source.write_text("def f():\n    pass\n")
    compiled = py_compile.compile(str(source), cfile=str(tmp_path / "b.pyc"),
                                  dfile=os.path.join(str(archive), "src", "pkg", "b.py"), doraise=True,
                                  invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("pkg/__init__.py", "")
        if beside == "its source beside it":
            zipped.write(source, "pkg/b.py")
        zipped.write(compiled, "pkg/b.pyc")
    py_file = str(archive / "src" / "pkg" / "b.py")
    assert configurations._compiled_from(str(archive), "pkg", "b") == py_file
    failure = KeyError("There is no item named 'src/pkg/b.py' in the archive")
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    if beside == "its source beside it":
        assert remedy == (
            "numba 0.68 and later read a .zip member's source from the archive by the name the module's code gives, "
            "src/pkg/b.py, which pkg/b.pyc was compiled to name, and this one holds none so named, while it holds the "
            "module's source, pkg/b.py: to cache, compile the .pyc to name its path in the archive, or leave it out, "
            "so that Python compiles the source and numba reads it under that name; or silence"), remedy
    else:
        assert remedy.startswith(
            "numba 0.68 and later read a .zip member's source from the archive, and this one holds none under "
            "src/pkg/b.py, the module shipped as .pyc alone: to cache, "), remedy


def test_an_archive_numba_cannot_read_is_told_to_make_it_readable(tmp_path, monkeypatch):
    """If numba cannot read the archive above a member, the error that names the archive gets a remedy that asks for the
    archive to be made readable, and not for free space or write access. numba reads the archive for the member's stamp
    and, from 0.68, for the member's source."""
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("package/module.py", "def f():\n    pass\n")
    py_file = str(archive / "package" / "module.py")
    failure = PermissionError(errno.EACCES, "Permission denied", str(archive))
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    assert "writable" not in remedy and "make room" not in remedy, remedy
    assert remedy == (
        "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: numba reads the archive, "
        f"{archive}, for the member's stamp, and from 0.68 for the member's source, and cannot (Permission denied): "
        "make it readable, or silence"), remedy


def test_an_error_naming_a_members_moved_archive_under_a_temporary_ipython_directory_is_the_archives(
        tmp_path, monkeypatch):
    """An error that names the moved archive of a member in an ``ipykernel_`` directory gets the archive's remedy, also
    under an IPython that gives temporary directories. The remedy that names ``IPYTHONDIR`` needs an error that names a
    ``numba_cache``, such as the error that numba's first save raises under such an IPython."""
    # In a real check, inspect reads no source from an archive that has moved and raises first. A caller can still pass
    # on an error that names the archive.
    import tempfile
    import types
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: tempfile.mkdtemp(dir=tmp_path)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    archive = tmp_path / "bundle.zip"
    py_file = str(archive / "ipykernel_123" / "cell.py")
    failure = FileNotFoundError(errno.ENOENT, "No such file or directory", str(archive))
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    assert "IPYTHONDIR" not in remedy, remedy
    assert remedy.startswith(f"the module's code names {archive}, which is not there: its .pyc was compiled to name "
                             "that path, so "), remedy


@pytest.mark.parametrize("name", ["APP.ZIP", "bundle.egg", "my.zipdir/bundle.egg"])
def test_an_archive_numbas_zip_locator_does_not_read_is_told_to_be_named_zip(tmp_path, monkeypatch, name):
    """An archive that numba's ``.zip`` locator does not accept, such as ``APP.ZIP`` or an ``.egg``, gets a remedy that
    asks for the archive to be renamed to end in ``.zip``, lower-case. Python imports from such an archive, so the
    remedy names the extension instead of offering a ``.zip``, which the reader may already use."""
    # The .zip locator looks for ".zip", lower-case, in the path. Under my.zipdir, the locator accepts the .egg and
    # raises ValueError, because no part of the path ends in .zip, and the remedy is the same rename.
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    archive = tmp_path / name
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("pkg/__init__.py", "")
        zipped.writestr("pkg/mod.py", "def f():\n    pass\n")
    py_file = str(archive / "pkg" / "mod.py")
    error, text = (ValueError, "No zip file found") if ".zip" in name else (RuntimeError, "no locator available")
    with pytest.raises(error, match=text) as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert configurations.is_a_cache_error(raised.value)
    remedy = configurations.cache_remedy(py_file, raised.value, "silence")
    assert "import it from a .zip" not in remedy, remedy
    assert remedy == (
        "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, install numbox with "
        f"its source files on disk, unpacked from any archive, or rename the archive, {archive}, to end in .zip, "
        "lower-case, which numba's .zip locator looks for in the path, with its source files in it, so that numba "
        "caches it in the user's cache directory; or silence"), remedy


def test_the_archive_above_a_file_is_found_with_os_name_set_to_windows(tmp_path, monkeypatch):
    """``_archive_above`` finds the archive above a file with ``os.name`` set to Windows's value, as the tests of
    numba's separator bug set it. It splits the path through the pure path classes, because the concrete
    ``pathlib.Path`` follows ``os.name``, and Python before 3.12 refuses to build a Windows path on Linux."""
    import numbox.core.configurations as configurations
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("pkg/mod.py", "x = 1\n")
    py_file = str(archive / "pkg" / "mod.py")
    monkeypatch.setattr(os, "name", "nt")
    assert configurations._archive_above(py_file) == (str(archive), "pkg/mod.py")
    assert configurations._archive_above(str(tmp_path / "elsewhere" / "mod.py")) is None


def test_a_relative_members_moved_archive_under_a_temporary_ipython_directory_is_the_archives(tmp_path, monkeypatch):
    """A member named by a relative path, whose error names its moved archive by a relative path too, gets the
    archive's remedy, also under an IPython that gives temporary directories. The remedy reads both paths as absolute
    before it compares them."""
    # numba names the member by a relative path if the process imports the archive from its own directory.
    import tempfile
    import types
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: tempfile.mkdtemp(dir=tmp_path)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    monkeypatch.chdir(tmp_path)
    py_file = os.path.join("bundle.zip", "ipykernel_123", "cell.py")
    failure = FileNotFoundError(errno.ENOENT, "No such file or directory", "bundle.zip")
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    assert "IPYTHONDIR" not in remedy, remedy
    assert remedy.startswith("the module's code names bundle.zip, which is not there: its .pyc was compiled to name "
                             "that path, so "), remedy


def test_a_relative_members_archive_named_absolute_by_the_error_is_told_as_moved(tmp_path, monkeypatch):
    """A member named by a relative path, whose error names its moved archive by an absolute path, as the stamp's error
    does, gets the archive's remedy and not the remedy for a location that refused a file. The remedy reads both paths
    as absolute before it compares them."""
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.chdir(tmp_path)
    py_file = os.path.join("build", "numbox.zip", "numbox", "module.py")
    archive = str(tmp_path / "build" / "numbox.zip")
    failure = FileNotFoundError(errno.ENOENT, "No such file or directory", archive)
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    assert "make room" not in remedy and "writable" not in remedy, remedy
    assert remedy.startswith(f"the module's code names {archive}, which is not there: its .pyc was compiled to name "
                             "that path, so "), remedy


def test_a_name_too_long_names_the_locator_the_in_tree_one_is_to_be_listed_before(tmp_path, monkeypatch):
    """For a directory name too long, the remedy asks for the in-tree locator to be listed before the locator that
    accepted the file, here a class outside numba's families, and does not name the ``.zip`` locator, which the list
    omits. The in-tree locator caches in the ``__pycache__`` beside the source, under no name that numba derives from
    the source's directory."""
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    class OurLocator(_CacheLocator):
        def __init__(self, py_func, py_file):
            self._py_file = py_file

        def get_cache_path(self):
            return str(tmp_path / "our-cache")

        def get_source_stamp(self):
            return 0

        def get_disambiguator(self):
            return "x"

        @classmethod
        def from_function(cls, py_func, py_file):
            return cls(py_func, py_file)

    ours = types.ModuleType("our_locators")
    ours.OurLocator = OurLocator
    monkeypatch.setitem(sys.modules, "our_locators", ours)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES",
                        "UserWideCacheLocator,our_locators.OurLocator,InTreeCacheLocator", raising=False)
    # A limit of 20 makes numba's directory name for the source too long on any file system.
    monkeypatch.setattr(configurations, "_name_limit", lambda location: 20)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("x = 1\n")
    subpath = _CacheLocator.get_suitable_cache_subpath(str(py_file))
    named = str(tmp_path / "user-cache" / "numba" / subpath)
    failure = OSError(errno.ENAMETOOLONG, "File name too long", named)
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert "ZipCacheLocator" not in remedy, remedy
    # The remedy counts the name as the file system does: in UTF-16 units, which it calls characters, on Windows and
    # macOS, and in bytes elsewhere.
    counted_in = "characters" if os.name == "nt" or sys.platform == "darwin" else "bytes"
    assert remedy == (
        f"numba names the directory it caches this file in after the source's directory, {subpath}, "
        f"{len(subpath.encode())} {counted_in}, which is longer than a name can be on that file system (20): put the "
        "file in a directory of a shorter name, or list InTreeCacheLocator before OurLocator, which caches beside the "
        "source, in its __pycache__, under no such name; or silence"), remedy


@pytest.mark.skipif(os.name == "nt", reason="Windows decodes any byte of a path")
@pytest.mark.skipif(sys.platform == "darwin", reason="APFS takes no name that is not UTF-8")
@pytest.mark.parametrize("error", ["a full __pycache__", "numba's no-locator error"])
def test_a_source_under_a_directory_that_is_not_utf8_is_not_offered_numba_cache_dir(tmp_path, monkeypatch, error):
    """A source under a directory whose path is not UTF-8 gets a remedy that does not offer ``NUMBA_CACHE_DIR``. numba
    names its directory under the variable with a hash of the source's directory encoded as UTF-8, so with the variable
    set numba would raise ``UnicodeEncodeError`` for this path, and the warning would remain."""
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    directory = tmp_path / os.fsdecode(b"caf\xe9") / "package"
    directory.mkdir(parents=True)
    py_file = str(directory / "module.py")
    with open(py_file, "w", encoding="utf-8") as handle:
        handle.write("x = 1\n")
    with pytest.raises(UnicodeEncodeError):
        py_file.encode()
    if error == "a full __pycache__":
        failure = OSError(errno.ENOSPC, "No space left on device", str(directory / "__pycache__"))
    else:
        failure = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    assert "NUMBA_CACHE_DIR set to" not in remedy and "Set NUMBA_CACHE_DIR" not in remedy, remedy
    if error == "a full __pycache__":
        assert remedy.endswith(
            "make room there, or make it writable, and NUMBA_CACHE_DIR cures nothing, numba hashing the source's "
            "directory as UTF-8 for its directory there, which this one is not; or silence"), remedy
    else:
        # The test builds numba's error that no locator accepts the file. In numba's order, the in-tree locator comes
        # before every locator that hashes the path, and it hashes no path, so the remedy also offers a writable
        # __pycache__.
        assert remedy == (
            "numba hashes the source's directory as UTF-8 for the directory it caches this file in under "
            f"NUMBA_CACHE_DIR or the user's cache directory, and {directory} is not: put the file under a UTF-8 "
            f"path, or make the __pycache__ beside the source, {directory / '__pycache__'}, writable, where "
            "InTreeCacheLocator caches it hashing no path; or silence"), remedy


def test_a_zip_member_in_an_ipykernel_directory_that_is_not_utf8_is_told_so(tmp_path, monkeypatch):
    """A member in an ``ipykernel_`` directory whose source is not UTF-8 makes the check raise inspect's ``OSError``, a
    cache error, and the remedy asks for the member to be shipped as UTF-8. zipimport raises ``UnicodeDecodeError`` for
    such a source, which is not a cache error, so the probe's loader returns no source instead."""
    # The remedy reads the member from the archive to report the reason and the byte at which the member stops being
    # valid UTF-8.
    import types
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ipykernel_123/cell.py", b"x = 'caf\xe9'\n")
    py_file = str(archive / "ipykernel_123" / "cell.py")
    with pytest.raises(OSError, match="could not get source code") as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert configurations.is_a_cache_error(raised.value)
    remedy = configurations.cache_remedy(py_file, raised.value, "silence")
    assert remedy == (
        f"numba's IPython locator reads the member's source as UTF-8, which {py_file} is not (invalid continuation "
        "byte at byte 8): ship it as UTF-8; or silence"), remedy


def test_a_listed_locator_whose_module_raises_on_import_is_no_locator_for_the_remedy(tmp_path, monkeypatch):
    """``_locators`` omits a listed class whose module raises on import, with any error, so the remedy for a cache error
    does not raise the module's error. numba raises that error at decoration, and the check raises it before any remedy
    is composed."""
    import numba
    import numbox.core.configurations as configurations
    (tmp_path / "badmod.py").write_text("raise RuntimeError('boom on import')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "badmod", raising=False)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "badmod.SomeClass,InTreeCacheLocator", raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("x = 1\n")
    failure = OSError(errno.EACCES, "Permission denied", str(py_file.parent / "__pycache__"))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert "boom" not in remedy and remedy.endswith("; or silence"), remedy
    from numba.core.caching import InTreeCacheLocator
    assert configurations._locators("badmod.SomeClass,InTreeCacheLocator") == [InTreeCacheLocator]


def test_an_empty_path_object_as_an_errors_name_is_the_working_directory_not_an_archive_above(tmp_path, monkeypatch):
    """An error whose file name is a path object built from an empty string names the working directory, and the remedy
    names ``.`` as a location that numba did not choose. The working directory lies above a member under it, but the
    remedy does not read it as the member's moved archive."""
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.chdir(tmp_path)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("pkg/mod.py", "x = 1\n")
    py_file = str(archive / "pkg" / "mod.py")
    failure = FileNotFoundError(errno.ENOENT, "No such file or directory", Path(""))
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    assert "which is not there" not in remedy and "compiled to name" not in remedy, remedy
    assert remedy == (
        "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: no file can be written at "
        "that location, . (No such file or directory), so make room there, or make it writable, or silence"), remedy


@pytest.mark.parametrize("placement", ["on disk", "a .zip member"])
def test_an_error_naming_no_numba_cache_under_a_temporary_ipython_directory_is_not_told_ipythondir(
        tmp_path, monkeypatch, placement):
    """A caller's error that names an unrelated location, under an IPython that gives temporary directories, gets the
    remedy for that location and not ``IPYTHONDIR``. The remedy names ``IPYTHONDIR`` only for an error that names a
    ``numba_cache``, as the check's error and numba's first save's error under such an IPython do."""
    import tempfile
    import types
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: tempfile.mkdtemp(dir=tmp_path)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    if placement == "on disk":
        # The in-tree locator accepts the cell file on disk, so numba never reaches the IPython locator.
        py_file = tmp_path / "ipykernel_123" / "cell.py"
        py_file.parent.mkdir()
        py_file.write_text("def cell():\n    return 1\n")
        configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
        assert (py_file.parent / "__pycache__").is_dir()
    else:
        archive = tmp_path / "bundle.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("ipykernel_123/cell.py", "def cell():\n    return 1\n")
        py_file = archive / "ipykernel_123" / "cell.py"
    elsewhere = str(tmp_path / "elsewhere" / "x")
    remedy = configurations.cache_remedy(str(py_file), OSError(errno.EACCES, "Permission denied", elsewhere), "silence")
    assert "IPYTHONDIR" not in remedy and "temporary" not in remedy, remedy
    assert f"that location, {elsewhere}" in remedy and "(Permission denied)" in remedy, remedy


def test_the_windows_separator_skip_fails_the_test_on_any_other_platform(monkeypatch):
    """``_skip_for_windows_separator_bug`` skips a test for the remedy of numba issue 10889 only on Windows, and fails
    the test if that remedy appears on any other platform. Other text in ``stderr`` neither skips nor fails the test."""
    stderr = f"RuntimeWarning: numba cannot cache numbox here (...). {windows_path_sep_bug}, x, which no archive holds"
    monkeypatch.setattr(os, "name", "posix")
    with pytest.raises(AssertionError, match="the Windows separator remedy on posix"):
        _skip_for_windows_separator_bug(stderr)
    monkeypatch.setattr(os, "name", "nt")
    with pytest.raises(pytest.skip.Exception):
        _skip_for_windows_separator_bug(stderr)
    _skip_for_windows_separator_bug("RuntimeWarning: numba cannot cache numbox here, another remedy")


def test_an_ipython_that_gives_its_directory_once_and_then_raises_is_not_read_as_temporary(tmp_path, monkeypatch):
    """A member's error that names IPython's ``numba_cache`` gets the remedy for that location, without temporary
    directories or ``IPYTHONDIR``, if IPython gives its cache directory once and raises at the next request."""
    # The remedy asks IPython a second time only to determine whether IPython gives a new temporary directory on every
    # call, and it makes that request only for an error with ENOENT. This error has EACCES, so the remedy asks IPython
    # once, and the stand-in never raises.
    import types
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    answers = [str(tmp_path / "ipython")]

    def once():
        if answers:
            return answers.pop()
        raise RuntimeError("IPython misbehaved")

    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = once
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ipykernel_123/cell.py", "def f():\n    pass\n")
    py_file = str(archive / "ipykernel_123" / "cell.py")
    named = tmp_path / "ipython" / "numba_cache"
    remedy = configurations.cache_remedy(py_file, OSError(errno.EACCES, "Permission denied", str(named)), "silence")
    assert "temporary" not in remedy and "IPYTHONDIR" not in remedy, remedy
    assert remedy == (
        f"numba caches this file in {named}, where no file can be written (Permission denied): make room there, or "
        "make it writable; or silence"), remedy


@pytest.mark.parametrize("placement", ["on disk", "a .zip member"])
def test_a_callers_error_with_no_errno_and_no_name_is_not_read_as_inspects(tmp_path, monkeypatch, placement):
    """A caller's ``OSError`` with no errno and no file name, for a file that numba's IPython locator accepts, is not
    read as inspect's error, which has no errno and no file name either. The remedy identifies inspect's errors by their
    messages, so it gives the caller's message as the reason and does not ask for the file to be made readable."""
    import types
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(tmp_path / "ipython")
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    if placement == "on disk":
        py_file = tmp_path / "ipykernel_123" / "cell.py"
        py_file.parent.mkdir()
        py_file.write_text("def f():\n    pass\n")
    else:
        archive = tmp_path / "bundle.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("ipykernel_123/cell.py", "def f():\n    pass\n")
        py_file = archive / "ipykernel_123" / "cell.py"
    failure = OSError("Disk quota exceeded on /home")
    assert failure.errno is None and failure.filename is None
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert "readable" not in remedy and "linecache" not in remedy, remedy
    assert "the location numba took" in remedy and "Disk quota exceeded on /home" in remedy, remedy


def test_the_check_reads_the_lines_linecache_holds_for_a_file_not_on_disk(tmp_path, monkeypatch):
    """The check does not change a member's linecache entry, and numba reads the member's lines from that entry while it
    picks the locator. A traceback through a member, or a read by inspect, leaves such an entry, and inspect reads the
    entry before it asks the module's loader."""
    import linecache
    import types
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ipykernel_123/cell.py", "def cell():\n    return 1\n")
    py_file = str(archive / "ipykernel_123" / "cell.py")
    held = (24, None, ["def cell():\n", "    return 1\n"], py_file)
    monkeypatch.setitem(linecache.cache, py_file, held)
    seen = []
    real = configurations.CompileResultCacheImpl

    def recording(probe):
        seen.append(linecache.getlines(py_file))
        return real(probe)

    monkeypatch.setattr(configurations, "CompileResultCacheImpl", recording)
    configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert seen == [["def cell():\n", "    return 1\n"]], seen
    assert linecache.cache[py_file] is held
    assert (ipython_dir / "numba_cache").is_dir()


@pytest.mark.parametrize("locators, made", [
    pytest.param("", "__pycache__", id="numba's order, the in-tree locator takes it"),
    pytest.param("IPythonCacheLocator,InTreeCacheLocator", "numba_cache", id="the IPython locator first"),
])
def test_a_cell_file_on_disk_is_asked_without_touching_linecache(tmp_path, monkeypatch, locators, made):
    """The check of a cell file on disk leaves linecache's lines for the file unchanged while numba picks the locator
    and afterward. inspect reads a file on disk from the file itself, so a reader of the file's source during the
    check reads the file's own lines."""
    # In numba's order, the in-tree locator accepts the cell file before the IPython locator, which accepts the file
    # only if the list puts it first.
    import linecache
    import types
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    py_file = tmp_path / "ipykernel_123" / "cell.py"
    py_file.parent.mkdir()
    py_file.write_text("def cell():\n    return 1\n")
    lines = linecache.getlines(str(py_file))
    assert lines == ["def cell():\n", "    return 1\n"]
    seen = []
    real = configurations.CompileResultCacheImpl

    def recording(probe):
        seen.append(linecache.getlines(str(py_file)))
        return real(probe)

    monkeypatch.setattr(configurations, "CompileResultCacheImpl", recording)
    configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
    assert seen == [lines] and linecache.getlines(str(py_file)) == lines, seen
    location = py_file.parent / "__pycache__" if made == "__pycache__" else ipython_dir / "numba_cache"
    assert location.is_dir()


def test_two_checks_of_a_file_not_on_disk_overlap_and_each_reads_the_members_source(tmp_path, monkeypatch):
    """Two checks of one member can run at once in two threads, and each probe reads the member's source. Each probe
    reads the source through a module of its own, so neither check waits for the other. Afterward linecache holds the
    member's own lines, and no probe module remains in ``sys.modules``."""
    import inspect
    import linecache
    import threading
    import types
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    with zipfile.ZipFile(tmp_path / "bundle.zip", "w") as zipped:
        zipped.writestr("ipykernel_123/cell.py", "def cell():\n    pass\n")
    py_file = str(tmp_path / "bundle.zip" / "ipykernel_123" / "cell.py")
    linecache.cache.pop(py_file, None)
    real = configurations.CompileResultCacheImpl
    inside = {"first": threading.Event(), "second": threading.Event()}
    hold = threading.Event()
    seen = {}

    def holding(probe):
        name = threading.current_thread().name
        seen[name] = inspect.getsource(probe)
        inside[name].set()
        if name == "first":
            hold.wait(5)
        return real(probe)

    monkeypatch.setattr(configurations, "CompileResultCacheImpl", holding)
    checks = [threading.Thread(target=configurations.check_cache_location, args=(py_file,), name=name)
              for name in ("first", "second")]
    checks[0].start()
    assert inside["first"].wait(5)
    checks[1].start()
    # The second check reaches numba's locator while the first check waits there.
    assert inside["second"].wait(5)
    hold.set()
    for check in checks:
        check.join(5)
    assert seen == {"first": "def cell():\n    pass\n", "second": "def cell():\n    pass\n"}, seen
    assert linecache.getlines(py_file) == ["def cell():\n", "    pass\n"]
    assert not [name for name in sys.modules if name.startswith("_numbox_cache_probe_")]
    linecache.cache.pop(py_file, None)


@pytest.mark.parametrize("refused, told", [
    pytest.param(("cell-1.py312.nbi.tmp.0123456789abcdef", errno.ENAMETOOLONG, "File name too long"),
                 "the path is too long for the file system: IPython's cache directory, {ipython_dir}, at a shorter path",
                 id="too long, a file in it"),
    pytest.param(("", errno.EACCES, "Permission denied"),
                 "numba caches this file in {numba_cache}, where no file can be written (Permission denied): make room "
                 "there, or make it writable", id="unwritable, the location"),
])
def test_a_zip_member_in_an_ipykernel_directory_refused_ipythons_location_is_told_ipythons_cache_directory(
        tmp_path, monkeypatch, refused, told):
    """A member in an ``ipykernel_`` directory whose error names IPython's ``numba_cache``, or a file in it, gets the
    remedy for IPython's location, as a cell file on disk does, but without ``NUMBA_CACHE_DIR``. numba's IPython
    locator accepts such a member before numba reaches the ``.zip`` locator."""
    import types
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    py_file = str(tmp_path / "bundle.zip" / "ipykernel_123" / "cell.py")
    name, number, strerror = refused
    named = ipython_dir / "numba_cache" / name if name else ipython_dir / "numba_cache"
    remedy = configurations.cache_remedy(py_file, OSError(number, strerror, str(named)), "silence")
    told = told.format(ipython_dir=ipython_dir, numba_cache=ipython_dir / "numba_cache")
    assert remedy == f"{told}; or silence", remedy


def test_a_zip_members_error_is_answered_without_asking_ipython_for_its_cache_directory(tmp_path, monkeypatch):
    """The remedy for a ``.zip`` member outside an ``ipykernel_`` directory never asks IPython for its cache directory.
    Under a read-only home, IPython warns and creates a temporary directory each time it is asked, so the remedy asks
    only if numba reaches its IPython locator and that locator accepts the file."""
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    def never():
        raise AssertionError("IPython was asked for its cache directory")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = never
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    py_file = str(tmp_path / "bundle.zip" / "package" / "module.py")
    location = tmp_path / "user-cache" / "numba" / _CacheLocator.get_suitable_cache_subpath(py_file)
    # The location exists and refuses the file. For a location that does not exist, the remedy would also name the
    # directory under which numba cannot create it.
    location.mkdir(parents=True)
    remedy = configurations.cache_remedy(py_file, OSError(errno.EACCES, "Permission denied", str(location)), "silence")
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        f"effect here, because the source is not a file on disk: no file can be written in that directory, {location} "
        "(Permission denied), so make room there, or make it writable, or silence"), remedy


@pytest.mark.parametrize("platform, moved_through", [
    ("linux", ", through XDG_CACHE_HOME or HOME"), ("darwin", ", through HOME"), ("win32", ""),
])
def test_the_user_cache_directory_is_told_what_moves_it_on_the_platform(tmp_path, monkeypatch, platform, moved_through):
    """The remedy that asks for the user's cache directory at a shorter path names the variables that set that
    directory on the platform: ``XDG_CACHE_HOME`` or ``HOME`` on Linux, ``HOME`` on macOS, and none on Windows. It
    names them alike for a source on disk and for a ``.zip`` member."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = tmp_path / "package" / "module.py"
    py_file.parent.mkdir()
    py_file.write_text("")
    user_cache_dir = configurations.AppDirs(appname="numba", appauthor=False).user_cache_dir
    location = os.path.join(user_cache_dir, _CacheLocator.get_suitable_cache_subpath(str(py_file)))
    # The test sets sys.platform after it reads the user's cache directory, so that the platform changes only the
    # remedy's text.
    monkeypatch.setattr(sys, "platform", platform)
    on_disk = configurations.cache_remedy(
        str(py_file), OSError(errno.ENAMETOOLONG, "File name too long", location), "silence")
    assert f"the user's cache directory, {user_cache_dir}, at a shorter path{moved_through}, or NUMBA_CACHE_DIR" in (
        on_disk), on_disk
    zip_member = str(tmp_path / "package.zip" / "package" / "module.py")
    zip_location = os.path.join(user_cache_dir, _CacheLocator.get_suitable_cache_subpath(zip_member))
    zipped = configurations.cache_remedy(
        zip_member, OSError(errno.ENAMETOOLONG, "File name too long", zip_location), "silence")
    assert f"put that directory, {zip_location}, at a shorter path{moved_through}, or silence" in zipped, zipped


@pytest.mark.parametrize("locators, offered", [
    ("InTreeCacheLocator,UserWideCacheLocator", False),
    ("numba.core.caching.UserProvidedCacheLocator, numba.core.caching.InTreeCacheLocator", True),
    ("IPythonCacheLocator,UserProvidedCacheLocator,InTreeCacheLocator,UserWideCacheLocator", True),
    ("InTreeCacheLocator,UserProvidedCacheLocator,UserWideCacheLocator", False),
    pytest.param("UserProvidedCacheLocator,InTreeCacheLocatorFsAgnostic,UserWideCacheLocator", True),
    pytest.param("InTreeCacheLocatorFsAgnostic", False),
])
def test_numba_cache_dir_is_offered_only_where_numba_tries_it_before_the_location_it_took(
        tmp_path, monkeypatch, locators, offered):
    """For a ``__pycache__`` too long, the remedy offers ``NUMBA_CACHE_DIR`` only if numba tries the variable's locator
    before the in-tree one, whether the list names classes by name or by dotted path. The IPython and ``.zip`` locators
    accept no plain file on disk, so listing one of them first does not change the remedy, and
    ``InTreeCacheLocatorFsAgnostic`` counts as the in-tree locator, because it caches in the same ``__pycache__``."""
    # If numba tries a set variable before the location that it chose, numba skipped the variable, so the remedy names
    # the variable as one that numba could not use. numba strips the space after a comma in the list.
    import numba
    from numbox.core.configurations import cache_remedy
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = tmp_path / "package" / "module.py"
    py_file.parent.mkdir()
    py_file.write_text("")
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(py_file.parent / "__pycache__"))
    remedy = cache_remedy(str(py_file), failure, "silence", package="numbduck")
    instead = (f", or NUMBA_CACHE_DIR, which is set to {tmp_path / 'cache'} and numba could not use, made a writable "
               "directory at a short path") if offered else ""
    assert remedy == f"the path is too long for the file system: numbduck installed at a shorter path{instead}; or silence", (
        remedy)


# Each case gives whether NUMBA_CACHE_DIR is set, the value of NUMBA_CACHE_LOCATOR_CLASSES, and the remedy up to its
# last clause, "or silence".
NO_LOCATOR_CASES = {
    "unset": (False, "", "Set NUMBA_CACHE_DIR to a writable directory"),
    "set, passed over": (
        True, "",
        "NUMBA_CACHE_DIR is set to {cache_dir}, which numba could not use: set it to a writable directory at a short "
        "path"),
    "without the user-provided locator": (
        False, "InTreeCacheLocator,UserWideCacheLocator",
        "numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, {locators}, says: make one of those locations writable"),
    "unset, the user-provided locator after another": (
        False, "InTreeCacheLocator,UserProvidedCacheLocator", "Set NUMBA_CACHE_DIR to a writable directory"),
    "set, passed over, the user-provided locator after another": (
        True, "InTreeCacheLocator,UserProvidedCacheLocator",
        "NUMBA_CACHE_DIR is set to {cache_dir}, which numba could not use: set it to a writable directory at a short "
        "path"),
    "only locators that take no plain file on disk": (
        False, "IPythonCacheLocator,ZipCacheLocator",
        "numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, {locators}, says, and none of those locators takes a "
        "source file on disk: list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or "
        "UserWideCacheLocator"),
}


@pytest.mark.parametrize("setting", list(NO_LOCATOR_CASES))
def test_no_locator_for_a_source_on_disk_is_told_what_numba_would_take(tmp_path, monkeypatch, setting):
    """numba's error that no locator accepts a source on disk gets a remedy that depends on the settings of
    ``NUMBA_CACHE_DIR`` and ``NUMBA_CACHE_LOCATOR_CLASSES``. The user-provided locator's place in the list does not
    change the remedy, because numba reads the variable once the locators before it skip the file."""
    # A set variable that numba skipped names a directory that numba could not write in, or one too deep for numba's
    # own directory. A writable directory at the same depth would not help, so the remedy asks for a short path. The
    # variable here names a long path.
    import numba
    from numbox.core.configurations import cache_remedy
    py_file = tmp_path / "package" / "module.py"
    py_file.parent.mkdir()
    py_file.write_text("")
    cache_dir = str(tmp_path / ("d" * 200))
    cache_dir_set, locators, opening = NO_LOCATOR_CASES[setting]
    monkeypatch.setattr(numba.config, "CACHE_DIR", cache_dir if cache_dir_set else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    failure = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
    remedy = cache_remedy(str(py_file), failure, "silence")
    assert remedy == f"{opening.format(cache_dir=cache_dir, locators=locators)}, or silence", remedy


def test_a_location_too_long_under_the_zip_locator_listed_first_is_the_users_cache_directorys(tmp_path, monkeypatch):
    """A source on disk under a directory named ``bundle.zip``, with the ``.zip`` locator listed first, gets a remedy
    for a path too long that names the user's cache directory and does not offer ``NUMBA_CACHE_DIR``. The ``.zip``
    locator accepts any path that contains ``.zip`` and caches it under the user's cache directory, so numba never reads
    the variable for that path."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        numba.config, "CACHE_LOCATOR_CLASSES",
        "ZipCacheLocator,UserProvidedCacheLocator,InTreeCacheLocator,UserWideCacheLocator", raising=False)
    py_file = tmp_path / "bundle.zip" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    user_cache_dir = tmp_path / "user-cache" / "numba"
    location = user_cache_dir / _CacheLocator.get_suitable_cache_subpath(str(py_file))
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(location))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == (
        f"the path is too long for the file system: the user's cache directory, {user_cache_dir}, at a shorter path"
        f"{configurations._moved_through()}; or silence"), remedy


# Each case gives whether NUMBA_CACHE_DIR is set, the value of NUMBA_CACHE_LOCATOR_CLASSES, and the remedy up to its
# last clause, "or silence".
@pytest.mark.parametrize("cache_dir_set, locators, expected", [
    (False, "", "Set NUMBA_CACHE_DIR to a writable directory"),
    pytest.param(
        False, "ZipCacheLocator,InTreeCacheLocator,UserWideCacheLocator",
        'numba\'s .zip locator, which NUMBA_CACHE_LOCATOR_CLASSES puts before every locator for a source file on disk, '
        'takes this file for the ".zip" in its path and finds no archive there: list UserProvidedCacheLocator, with '
        "NUMBA_CACHE_DIR set, InTreeCacheLocator or UserWideCacheLocator before it",
    ),
    pytest.param(
        True, "InTreeCacheLocator,ZipCacheLocator,UserProvidedCacheLocator",
        'numba\'s .zip locator takes this file for the ".zip" in its path and finds no archive there, after every '
        "locator NUMBA_CACHE_LOCATOR_CLASSES, {locators}, puts before it passed the file over: make one of those "
        "locations writable, or list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or "
        "UserWideCacheLocator before it",
    ),
    pytest.param(
        True, "UserProvidedCacheLocator,ZipCacheLocator,InTreeCacheLocator",
        "NUMBA_CACHE_DIR is set to {cache_dir}, which numba could not use: set it to a writable directory at a short "
        "path",
    ),
    pytest.param(
        False, "UserProvidedCacheLocator,ZipCacheLocator,InTreeCacheLocator",
        "Set NUMBA_CACHE_DIR to a writable directory"),
])
def test_a_source_on_disk_with_zip_in_its_path_and_no_archive_is_told_by_the_zip_locators_place(
        tmp_path, monkeypatch, cache_dir_set, locators, expected):
    """The ``.zip`` locator's ``ValueError``, for a source on disk whose path contains ``.zip`` but has no part that
    ends in it, gets a remedy that depends on the locator's place in the list. numba tries the locators before the
    ``.zip`` one and none after it, so the error means that those before it skipped the file."""
    # In numba's order, every locator for a file on disk comes before the .zip one. The remedy is then the one for
    # numba's error that no locator accepts the file. Listed first, the .zip locator accepts the file before any
    # locator for a file on disk, and the remedy asks for one of those to be listed before it. Listed before the
    # user-provided locator, it raises before numba reads NUMBA_CACHE_DIR, so the remedy does not describe a set
    # variable as one that numba could not use. Listed after that locator, it raises once numba has skipped the
    # variable, and the remedy is the one for the variable. The user-provided locator accepts a file only while
    # NUMBA_CACHE_DIR is set, so the remedy asks for that locator to be listed with the variable set.
    import numba
    from numbox.core.configurations import cache_remedy
    cache_dir = str(tmp_path / "cache")
    monkeypatch.setattr(numba.config, "CACHE_DIR", cache_dir if cache_dir_set else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = tmp_path / "not.zipped" / "module.py"
    py_file.parent.mkdir()
    py_file.write_text("")
    remedy = cache_remedy(str(py_file), ValueError("No zip file found in path"), "silence")
    assert remedy == f"{expected.format(cache_dir=cache_dir, locators=locators)}, or silence", remedy


@pytest.mark.parametrize("failure", [
    pytest.param(OSError(errno.ENAMETOOLONG, "File name too long"), id="too long, unnamed"),
    pytest.param(OSError(errno.EACCES, "Permission denied"), id="unwritable, unnamed"),
    pytest.param(OSError(errno.ENAMETOOLONG, "File name too long", "/elsewhere/cache"), id="too long, named"),
])
def test_an_error_under_a_locator_list_with_no_locator_that_takes_the_file_is_told_the_list(
        tmp_path, monkeypatch, failure):
    """If no listed locator accepts a source file on disk, a caller's ``OSError``, with or without a file name, gets the
    remedy that names the list, as numba's error that no locator accepts the file does. numba chooses no location under
    such a list, so the remedy neither names a location nor asks for a shorter path."""
    import numba
    from numbox.core.configurations import cache_remedy
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "IPythonCacheLocator,ZipCacheLocator", raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    assert cache_remedy(str(py_file), failure, "silence") == (
        "numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, IPythonCacheLocator,ZipCacheLocator, says, and none of "
        "those locators takes a source file on disk: list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, "
        "InTreeCacheLocator or UserWideCacheLocator, or silence"), cache_remedy(str(py_file), failure, "silence")


@pytest.mark.parametrize("failure", [
    pytest.param(OSError(errno.ENAMETOOLONG, "File name too long"), id="too long, unnamed"),
    pytest.param(OSError(errno.EACCES, "Permission denied"), id="unwritable, unnamed"),
    pytest.param(OSError(errno.ENAMETOOLONG, "File name too long", "/elsewhere/cache"), id="too long, named"),
])
def test_an_error_under_the_user_provided_locator_alone_with_numba_cache_dir_unset_is_told_to_set_it(
        tmp_path, monkeypatch, failure):
    """If the list holds the user-provided locator alone and ``NUMBA_CACHE_DIR`` is unset, numba's error that no locator
    accepts the file and a caller's ``OSError`` both get the remedy that asks for the variable to be set. That locator
    accepts nothing while the variable is unset, so numba chooses no location."""
    import numba
    from numbox.core.configurations import cache_remedy
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "UserProvidedCacheLocator", raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    no_locator = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
    told = cache_remedy(str(py_file), no_locator, "silence")
    assert told == "Set NUMBA_CACHE_DIR to a writable directory, or silence", told
    assert cache_remedy(str(py_file), failure, "silence") == told, cache_remedy(str(py_file), failure, "silence")


# These are the clauses of the archive remedy under NUMBA_CACHE_LOCATOR_CLASSES, whose value {locators} stands for.
# They are the install on disk and the .zip, and the clause that is added to each if the list lacks the locator that it
# needs.
ON_DISK = ("install numbox with its source files on disk, unpacked from any archive")
LIST_ON_DISK = (", and list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or UserWideCacheLocator "
                "in NUMBA_CACHE_LOCATOR_CLASSES, {locators}, which has none of them")
A_ZIP = ("import it from a .zip holding its source files, which numba caches in the user's cache directory")
LIST_A_ZIP = " through ZipCacheLocator, once NUMBA_CACHE_LOCATOR_CLASSES, {locators}, lists it"
SET_THE_VARIABLE = (", with NUMBA_CACHE_DIR set, which UserProvidedCacheLocator, the one locator for a file on disk in "
                    "NUMBA_CACHE_LOCATOR_CLASSES, {locators}, takes nothing without")


@pytest.mark.parametrize("locators, cache_dir_set, on_disk, a_zip", [
    pytest.param("", False, ON_DISK, A_ZIP, id="numba's order"),
    pytest.param("UserProvidedCacheLocator,InTreeCacheLocator,UserWideCacheLocator", False, ON_DISK,
                 A_ZIP + LIST_A_ZIP, id="no .zip locator"),
    pytest.param("ZipCacheLocator", False, ON_DISK + LIST_ON_DISK, A_ZIP, id="no locator for a file on disk"),
    pytest.param("IPythonCacheLocator", False, ON_DISK + LIST_ON_DISK, A_ZIP + LIST_A_ZIP, id="neither"),
    pytest.param("UserProvidedCacheLocator,ZipCacheLocator", False, ON_DISK + SET_THE_VARIABLE, A_ZIP,
                 id="the user-provided locator alone for a file on disk, NUMBA_CACHE_DIR unset"),
    pytest.param("UserProvidedCacheLocator,ZipCacheLocator", True, ON_DISK, A_ZIP,
                 id="the user-provided locator alone for a file on disk, NUMBA_CACHE_DIR set"),
    pytest.param("InTreeCacheLocator,ZipCacheLocator", False, ON_DISK, A_ZIP,
                 id="the in-tree locator alone for a file on disk"),
    pytest.param("UserWideCacheLocator,ZipCacheLocator", False, ON_DISK, A_ZIP,
                 id="the user-wide locator alone for a file on disk"),
])
def test_an_import_from_an_egg_under_a_locator_list_is_offered_each_way_with_the_locator_it_needs(
        tmp_path, monkeypatch, locators, cache_dir_set, on_disk, a_zip):
    """The archive remedy for an ``.egg`` under a ``NUMBA_CACHE_LOCATOR_CLASSES`` offers each way to cache with the
    locator that it needs: the install on disk with a locator for a file on disk, and a ``.zip`` with
    ``ZipCacheLocator``. If the user-provided locator is the only listed one for a file on disk, the remedy also asks
    for ``NUMBA_CACHE_DIR`` to be set, because that locator accepts nothing without it."""
    import numba
    from numbox.core.configurations import cache_remedy
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache") if cache_dir_set else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = tmp_path / "numbox-0.0.0-py3.12.egg" / "numbox" / "module.py"
    failure = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
    remedy = cache_remedy(str(py_file), failure, "silence")
    assert remedy == (
        "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, "
        f"{on_disk.format(locators=locators)}, or {a_zip.format(locators=locators)}; or silence"), remedy


@pytest.mark.parametrize("locators, expected", [
    ("", "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, install numbox with "
         "its source files on disk, unpacked from any archive, or import it from a .zip holding its source files, "
         "which numba caches in the user's cache directory"),
    ("UserProvidedCacheLocator,InTreeCacheLocator,UserWideCacheLocator",
     "numba caches a .zip through ZipCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, {locators}, leaves it out: "
     "list it"),
])
def test_a_zip_import_under_a_locator_list_without_the_zip_locator_is_told_to_list_it(
        tmp_path, monkeypatch, locators, expected):
    """A ``.zip`` member under a list without ``ZipCacheLocator`` gets the remedy that asks for that locator to be
    listed, and not the archive remedy, which would offer the ``.zip`` that the reader already uses. numba caches a
    ``.zip`` through that locator alone."""
    import numba
    from numbox.core.configurations import cache_remedy
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = str(tmp_path / "bundle.zip" / "numbox" / "module.py")
    failure = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
    remedy = cache_remedy(py_file, failure, "silence")
    assert remedy == f"{expected.format(locators=locators)}; or silence", remedy


@pytest.mark.parametrize("locators, expected", [
    ("", "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
         "effect here, because the source is not a file on disk: numba could not use that directory, {user_cache_dir}: "
         "make it writable, or put it at a shorter path{moved_through}"),
    ("UserProvidedCacheLocator,InTreeCacheLocator",
     "numba caches a frozen application through UserWideCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, "
     "{locators}, leaves it out: list it"),
])
def test_a_frozen_application_under_a_locator_list_without_the_user_wide_locator_is_told_to_list_it(
        tmp_path, monkeypatch, locators, expected):
    """A frozen application under a list without ``UserWideCacheLocator`` gets the remedy that asks for that locator to
    be listed, and not for the user's cache directory to be made writable. numba caches a frozen application through
    that locator alone, so under such a list it never tries the user's cache directory."""
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = str(tmp_path / "frozen" / "numbox" / "module.py")
    failure = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    expected = expected.format(locators=locators, user_cache_dir=tmp_path / "user-cache" / "numba",
                               moved_through=configurations._moved_through())
    assert remedy == f"{expected}, or silence", remedy


# NONE_OF_NUMBAS_OFF_DISK holds the remedies, by errno, for a source not on disk whose error names a location that
# none of numba's locators chose.
NONE_OF_NUMBAS_OFF_DISK = {
    "too long": ((errno.ENAMETOOLONG, "File name too long"),
                 "the path is too long for the file system, so put that location, {named}, at a shorter path"),
    "unwritable": ((errno.EACCES, "Permission denied"),
                   "no file can be written at that location, {named} (Permission denied), so make room there, or make it "
                   "writable"),
}


@pytest.mark.parametrize("locators, in_numba_cache, refused", [
    pytest.param("ZipCacheLocator,IPythonCacheLocator", "cell-1.py312.nbi", "too long",
                 id="the .zip locator first, a file in numba_cache, too long"),
    pytest.param("ZipCacheLocator,IPythonCacheLocator", "", "unwritable",
                 id="the .zip locator first, numba_cache, unwritable"),
    pytest.param("", "subdir/cell-1.py312.nbi", "too long", id="numba's order, a directory numba never makes in it"),
])
def test_an_error_naming_ipythons_location_numbas_ipython_locator_did_not_take_is_told_the_location_as_named(
        tmp_path, monkeypatch, locators, in_numba_cache, refused):
    """A member's error that names a path under IPython's ``numba_cache`` that numba's IPython locator did not choose
    gets the remedy for a location that no locator chose, as the error names it. numba never reaches the IPython
    locator if the list puts the ``.zip`` locator first, and the IPython locator writes nothing in a subdirectory of
    ``numba_cache``."""
    # Listed first, the .zip locator accepts the member without trying its location, and numba caches the member under
    # the user's cache directory. The error names a path outside that directory, so the remedy does not mention the
    # directory.
    import types
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    py_file = str(tmp_path / "bundle.zip" / "ipykernel_123" / "cell.py")
    named = ipython_dir / "numba_cache" / in_numba_cache if in_numba_cache else ipython_dir / "numba_cache"
    error, told = NONE_OF_NUMBAS_OFF_DISK[refused]
    remedy = configurations.cache_remedy(py_file, OSError(*error, str(named)), "silence")
    assert remedy == (
        "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: "
        f"{told.format(named=named)}, or silence"), remedy


def test_a_zip_members_error_naming_the_zips_location_after_ipythons_locator_passed_it_over_is_told_the_location(
        tmp_path, monkeypatch):
    """A member's error that names the ``.zip`` locator's location, after numba's IPython locator skipped the member,
    gets the ``.zip`` remedy for that location. The IPython locator skips a location that it cannot create or write in,
    and the ``.zip`` locator listed after it, as in numba's own order, then caches the member in the user's cache
    directory."""
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "IPythonCacheLocator,ZipCacheLocator", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    py_file = str(tmp_path / "bundle.zip" / "ipykernel_123" / "cell.py")
    location = tmp_path / "user-cache" / "numba" / _CacheLocator.get_suitable_cache_subpath(py_file)
    remedy = configurations.cache_remedy(
        py_file, OSError(errno.EACCES, "Permission denied", str(location / "cell-1.py312.nbi.tmp.0123456789abcdef")),
        "silence")
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        f"effect here, because the source is not a file on disk: no file can be written in that directory, {location} "
        "(Permission denied), so make room there, or make it writable, or silence"), remedy


@pytest.mark.parametrize("locators, member", [
    pytest.param("", "bundle.zip/package/module.py", id="numba's order, a .zip member in no ipykernel directory"),
    pytest.param("ZipCacheLocator", "bundle.zip/ipykernel_123/cell.py", id="a list without the IPython locator"),
    pytest.param("ZipCacheLocator,IPythonCacheLocator", "bundle.zip/ipykernel_123/cell.py",
                 id="the .zip locator listed first, which takes the member before the IPython one is reached"),
])
@on_windows_under_numba_0_68_0
def test_a_file_not_on_disk_numbas_ipython_locator_does_not_read_is_asked_without_touching_linecache(
        tmp_path, monkeypatch, locators, member):
    """The check adds no entry to linecache for a file not on disk if no IPython locator that numba reaches accepts the
    file, and IPython's directory is not created. numba's IPython locator alone reads the function's source, so for such
    a file nothing reads the member while numba picks the locator."""
    # The files are a member outside an ipykernel_ directory, and a member under a list that omits the IPython locator
    # or puts the .zip locator first.
    import linecache
    import types
    import numba
    import numba.core.caching
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(numba.core.caching, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(member.split("/", 1)[1], "def f():\n    pass\n")
    py_file = str(tmp_path / member)
    seen = []
    real = configurations.CompileResultCacheImpl

    def recording(probe):
        seen.append((py_file in linecache.cache, linecache.getlines(py_file)))
        return real(probe)

    monkeypatch.setattr(configurations, "CompileResultCacheImpl", recording)
    configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert seen == [(False, [])], seen
    assert py_file not in linecache.cache
    assert (tmp_path / "user-cache" / "numba").is_dir()
    assert not ipython_dir.exists()


@pytest.mark.parametrize("refused, told", [
    pytest.param(("__pycache__", errno.ENAMETOOLONG, "File name too long"),
                 "the path is too long for the file system: that location, {named}, at a shorter path",
                 id="too long, beside the source"),
    pytest.param(("__pycache__", errno.EACCES, "Permission denied"),
                 "that location, {named}, where no file can be written (Permission denied): make room there, or make it "
                 "writable", id="unwritable, beside the source"),
    pytest.param(("", errno.ENAMETOOLONG, "File name too long"),
                 "the path is too long for the file system: the user's cache directory, {user_cache_dir}, at a shorter "
                 "path{moved_through}", id="too long, the .zip locator's location"),
])
def test_a_file_on_disk_under_a_zip_directory_with_the_zip_locator_first_is_answered_for_the_locators_numba_reaches(
        tmp_path, monkeypatch, refused, told):
    """If the list puts the ``.zip`` locator first, the remedy matches the error of a source on disk under a directory
    named ``d.zip`` against the ``.zip`` locator's location alone. That locator accepts the file without trying its
    location, so numba never reaches the in-tree locator listed after it, and the ``__pycache__`` beside the source is
    a location that numba did not choose."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "ZipCacheLocator,InTreeCacheLocator", raising=False)
    py_file = tmp_path / "d.zip" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    user_cache_dir = tmp_path / "user-cache" / "numba"
    location = user_cache_dir / _CacheLocator.get_suitable_cache_subpath(str(py_file))
    which, number, strerror = refused
    named = py_file.parent / which if which else location
    remedy = configurations.cache_remedy(str(py_file), OSError(number, strerror, str(named)), "silence")
    told = told.format(named=named, user_cache_dir=user_cache_dir, moved_through=configurations._moved_through())
    assert remedy == f"{told}; or silence", remedy


def test_an_empty_path_object_names_the_working_directory_as_pathlib_reads_it(tmp_path, monkeypatch):
    """An error whose file name is ``Path("")`` names the working directory, ``.``, and gets the remedy that the string
    ``"."`` gets, because pathlib reads an empty string as ``.``. An empty string or empty bytes does not count as a
    name, and such an error gets the remedy that lists the locations numba could have chosen."""
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")

    def remedy(name):
        return configurations.cache_remedy(
            str(py_file), OSError(errno.ENAMETOOLONG, "File name too long", name), "silence")

    assert Path("") == Path(".") and os.fspath(Path("")) == "."
    assert remedy(Path("")) == remedy(".") == (
        "the path is too long for the file system: that location, ., at a shorter path, or NUMBA_CACHE_DIR set to a "
        "short path; or silence")
    assert remedy("").startswith("the path is too long for the file system: the location numba took, one of ")
    assert remedy(b"") == remedy("")


def test_a_locator_entry_naming_no_class_is_left_to_numba_and_its_error_raised_as_it_was(tmp_path, monkeypatch):
    """A dotted entry of ``NUMBA_CACHE_LOCATOR_CLASSES`` that names no class makes the check raise numba's own
    ``AttributeError``, which is not a cache error, and the remedy reads the list without that entry. numba resolves the
    entry to the object that the module holds under that name, and it fails when it calls the entry's ``from_function``.
    """
    import types
    import numba
    import numbox.core.configurations as configurations
    ours = types.ModuleType("our_locators")

    def not_a_locator_class():
        pass

    ours.not_a_locator_class = not_a_locator_class
    monkeypatch.setitem(sys.modules, "our_locators", ours)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "our_locators.not_a_locator_class,InTreeCacheLocator",
                        raising=False)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ipykernel_123/cell.py", "def f():\n    pass\n")
    py_file = str(archive / "ipykernel_123" / "cell.py")
    with pytest.raises(AttributeError, match="from_function") as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert not configurations.is_a_cache_error(raised.value)
    on_disk = tmp_path / "site" / "package" / "module.py"
    on_disk.parent.mkdir(parents=True)
    on_disk.write_text("")
    failure = OSError(errno.EACCES, "Permission denied", str(on_disk.parent / "__pycache__"))
    with_the_entry = configurations.cache_remedy(str(on_disk), failure, "silence")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "InTreeCacheLocator", raising=False)
    assert with_the_entry == configurations.cache_remedy(str(on_disk), failure, "silence")
    assert "numba caches this file in" in with_the_entry, with_the_entry


@pytest.mark.parametrize("refused", ["CacheLocator", "SourceFileBackedLocatorMixin"])
def test_a_listed_name_numba_refuses_is_left_out_as_numba_leaves_it(tmp_path, monkeypatch, refused):
    """A plain entry of ``NUMBA_CACHE_LOCATOR_CLASSES`` that names nothing in numba's caching module is omitted from the
    remedy's list, as numba refuses it. The check raises numba's ``RuntimeError``, which is not a cache error, and the
    remedy for a caller's error names the list as holding no locator for a file on disk."""
    # The two names are those of numba's base classes, _CacheLocator and _SourceFileBackedLocatorMixin, without their
    # leading underscore.
    import numba
    import numbox.core.configurations as configurations
    from numba.core import caching
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    assert configurations._locators(f"{refused},InTreeCacheLocator") == [caching.InTreeCacheLocator]
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", refused, raising=False)
    with pytest.raises(RuntimeError, match="Unknown cache locator class") as raised:
        configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
    assert not configurations.is_a_cache_error(raised.value)
    failure = OSError(errno.EACCES, "Permission denied", str(py_file.parent / "__pycache__"))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == (
        f"numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, {refused}, says, and none of those locators takes a "
        "source file on disk: list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or "
        "UserWideCacheLocator, or silence"), remedy


def test_a_locator_entry_with_an_empty_module_path_is_left_out_and_numbas_error_raised_as_it_was(tmp_path, monkeypatch):
    """A dotted entry with an empty module path, such as ``.InTreeCacheLocator``, makes the check raise importlib's
    ``ValueError``, which is not a cache error, and the remedy reads the list without that entry. numba does not catch
    that error, so the first function that numba decorates raises it."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core import caching
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    assert configurations._locators(".InTreeCacheLocator,InTreeCacheLocator") == [caching.InTreeCacheLocator]
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", ".InTreeCacheLocator,InTreeCacheLocator", raising=False)
    with pytest.raises(ValueError, match="Empty module name") as raised:
        configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
    assert not configurations.is_a_cache_error(raised.value)
    failure = OSError(errno.EACCES, "Permission denied", str(py_file.parent / "__pycache__"))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy.startswith(f"numba caches this file in {py_file.parent / '__pycache__'},"), remedy


@pytest.mark.parametrize("placement", ["a cell file on disk", "a .zip member in an ipykernel directory"])
def test_an_ipython_that_raises_for_its_cache_directory_leaves_numba_no_location_of_ipythons(
        tmp_path, monkeypatch, placement):
    """If IPython raises ``OSError`` for its cache directory, numba's IPython locator skips the file, and the remedy
    treats IPython as giving no location instead of raising IPython's error. The test covers numba's error that no
    locator accepts a cell file on disk, and an error that names the ``.zip`` locator's location for a member."""
    # IPython raises such an error if it cannot create its directory in any of the places that it tries. numba's IPython
    # locator asks for the directory when it creates its location, and it skips the file on that error as on any
    # location that it cannot create.
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    def nowhere():
        raise PermissionError(errno.EACCES, "Permission denied", str(tmp_path / "home"))

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = nowhere
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    assert configurations._ipython_numba_cache() is None
    if placement == "a cell file on disk":
        monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "IPythonCacheLocator", raising=False)
        py_file = tmp_path / "ipykernel_123" / "cell.py"
        py_file.parent.mkdir()
        py_file.write_text("def cell():\n    pass\n")
        with pytest.raises(RuntimeError, match="no locator available") as raised:
            configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
        assert configurations.is_a_cache_error(raised.value)
        remedy = configurations.cache_remedy(str(py_file), raised.value, "silence")
        assert remedy == (
            "numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, IPythonCacheLocator, says: make one of those "
            "locations writable, or silence"), remedy
    else:
        monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
        py_file = tmp_path / "bundle.zip" / "ipykernel_123" / "cell.py"
        location = tmp_path / "user-cache" / "numba" / _CacheLocator.get_suitable_cache_subpath(str(py_file))
        # The location exists and refuses the file. For a location that does not exist, the remedy would also name
        # the directory under which numba cannot create it.
        location.mkdir(parents=True)
        failure = OSError(errno.EACCES, "Permission denied", str(location))
        remedy = configurations.cache_remedy(str(py_file), failure, "silence")
        assert remedy == (
            "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has "
            "no effect here, because the source is not a file on disk: no file can be written in that directory, "
            f"{location} (Permission denied), so make room there, or make it writable, or silence"), remedy


# Each case gives the value of NUMBA_CACHE_LOCATOR_CLASSES, the source's path under tmp_path, the error's errno and
# message, and the remedy.
NO_LOCATOR_TAKES_IT_OFF_DISK = {
    "a .zip member, too long": (
        "IPythonCacheLocator", "bundle.zip/package/module.py", (errno.ENAMETOOLONG, "File name too long"),
        "numba caches a .zip through ZipCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, IPythonCacheLocator, "
        "leaves it out: list it; or silence"),
    "a .zip member, unwritable": (
        "IPythonCacheLocator,InTreeCacheLocator", "bundle.zip/package/module.py", (errno.EACCES, "Permission denied"),
        "numba caches a .zip through ZipCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, "
        "IPythonCacheLocator,InTreeCacheLocator, leaves it out: list it; or silence"),
    "a frozen application, unwritable": (
        "InTreeCacheLocator", "frozen/numbox/module.py", (errno.EACCES, "Permission denied"),
        "numba caches a frozen application through UserWideCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, "
        "InTreeCacheLocator, leaves it out: list it, or silence"),
}


@pytest.mark.parametrize("case", list(NO_LOCATOR_TAKES_IT_OFF_DISK))
@pytest.mark.parametrize("names", ["the location", "nothing"])
def test_an_error_for_a_source_not_on_disk_under_a_list_with_no_locator_that_takes_it_is_told_the_list(
        tmp_path, monkeypatch, case, names):
    """If the list lacks the one locator that accepts a source not on disk, a caller's ``OSError``, with or without a
    file name, gets the remedy that asks for that locator to be listed. That locator is ``ZipCacheLocator`` for a member
    outside an ``ipykernel_`` directory and ``UserWideCacheLocator`` for a frozen application, and numba chooses no
    location without it."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    locators, member, error, told = NO_LOCATOR_TAKES_IT_OFF_DISK[case]
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    monkeypatch.setattr(sys, "frozen", case.startswith("a frozen"), raising=False)
    py_file = str(tmp_path / member)
    location = tmp_path / "user-cache" / "numba" / _CacheLocator.get_suitable_cache_subpath(py_file)
    failure = OSError(*error, str(location)) if names == "the location" else OSError(*error)
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    assert remedy == told, remedy
    no_locator = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
    assert configurations.cache_remedy(py_file, no_locator, "silence") == told


# Each case gives the value of NUMBA_CACHE_LOCATOR_CLASSES, the source's path under tmp_path, the error's errno and
# message, and the remedy up to its last clause.
NO_NAME_OFF_DISK = {
    "a .zip member in an ipykernel directory, numba's order, a full disk": (
        "", "bundle.zip/ipykernel_123/cell.py", (errno.ENOSPC, "No space left on device"),
        "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: no file can be written in the "
        "location numba took, one of {numba_cache} or {location} (No space left on device), so make room there, or "
        "make it writable"),
    "a .zip member in an ipykernel directory, numba's order, too long": (
        "", "bundle.zip/ipykernel_123/cell.py", (errno.ENAMETOOLONG, "File name too long"),
        "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: the path is too long for the "
        "file system, so put the location numba took, one of {numba_cache} or {location}, at a shorter path"),
    "a .zip member in an ipykernel directory, the .zip locator first, a full disk": (
        "ZipCacheLocator,IPythonCacheLocator", "bundle.zip/ipykernel_123/cell.py",
        (errno.ENOSPC, "No space left on device"),
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        "effect here, because the source is not a file on disk: no file can be written in the location numba took, "
        "{location} (No space left on device), so make room there, or make it writable"),
    "a .zip member, a full disk": (
        "", "bundle.zip/package/module.py", (errno.ENOSPC, "No space left on device"),
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        "effect here, because the source is not a file on disk: no file can be written in the location numba took, "
        "{location} (No space left on device), so make room there, or make it writable"),
    "a .zip member, too long": (
        "", "bundle.zip/package/module.py", (errno.ENAMETOOLONG, "File name too long"),
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        "effect here, because the source is not a file on disk: the path is too long for the file system, so put the "
        "location numba took, {location}, at a shorter path{moved_through}"),
    "a frozen application, a full disk": (
        "", "frozen/numbox/module.py", (errno.ENOSPC, "No space left on device"),
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        "effect here, because the source is not a file on disk: no file can be written in the location numba took, "
        "{location} (No space left on device), so make room there, or make it writable"),
}


@pytest.mark.parametrize("case", [pytest.param(case, id=case) for case in NO_NAME_OFF_DISK])
def test_an_error_naming_no_file_for_a_source_not_on_disk_is_told_the_locations_numba_could_have_taken(
        tmp_path, monkeypatch, case):
    """An ``OSError`` that names no file, for a source not on disk, gets a remedy that lists the locations numba could
    have chosen, each once and in numba's order. If IPython's ``numba_cache`` is among them, the remedy does not say
    that numba caches the file in the user's cache directory."""
    # numba's first save raises such an error when a write fails, as on a full disk, and a caller can pass one on.
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    locators, member, error, told = NO_NAME_OFF_DISK[case]
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    monkeypatch.setattr(sys, "frozen", case.startswith("a frozen"), raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    py_file = str(tmp_path / member)
    location = tmp_path / "user-cache" / "numba" / _CacheLocator.get_suitable_cache_subpath(py_file)
    remedy = configurations.cache_remedy(py_file, OSError(*error), "silence")
    told = told.format(numba_cache=ipython_dir / "numba_cache", location=location,
                       moved_through=configurations._moved_through())
    assert remedy == f"{told}, or silence", remedy


@pytest.mark.parametrize("refused", list(NONE_OF_NUMBAS_OFF_DISK))
def test_an_error_naming_ipythons_location_with_ipython_not_importable_is_told_the_location_as_none_of_numbas(
        tmp_path, monkeypatch, refused):
    """Without IPython, a member's error that names IPython's ``numba_cache`` gets the remedy for a location that no
    locator chose, without saying where numba caches a ``.zip``. numba's IPython locator accepts the member and imports
    IPython for its location, catching only ``OSError``, so numba raises ``ImportError`` at decoration and does not
    cache the member."""
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setitem(sys.modules, "IPython.paths", None)
    monkeypatch.setitem(sys.modules, "IPython.utils.path", None)
    with zipfile.ZipFile(tmp_path / "bundle.zip", "w") as zipped:
        zipped.writestr("ipykernel_123/cell.py", "def cell():\n    pass\n")
    py_file = str(tmp_path / "bundle.zip" / "ipykernel_123" / "cell.py")
    with pytest.raises(ImportError) as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert not configurations.is_a_cache_error(raised.value)
    named = tmp_path / "ipython" / "numba_cache"
    error, told = NONE_OF_NUMBAS_OFF_DISK[refused]
    remedy = configurations.cache_remedy(py_file, OSError(*error, str(named)), "silence")
    assert remedy == (
        "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: "
        f"{told.format(named=named)}, or silence"), remedy


@pytest.mark.parametrize("refused, told", [
    ("too long", "the path is too long for the file system: IPython's cache directory, {ipython_dir}, at a shorter path"),
    ("unwritable", "numba caches this file in {numba_cache}, where no file can be written (Permission denied): make "
                   "room there, or make it writable"),
])
def test_a_cell_numbas_ipython_locator_takes_by_its_name_is_asked_and_told_as_a_cell_file_is(
        tmp_path, monkeypatch, refused, told):
    """The check raises no error for a cell named ``<ipython-...>``, whose source is in linecache, and it creates
    ``numba_cache`` under IPython's cache directory, as numba does. An error there gets the remedy that a cell file on
    disk gets for IPython's location."""
    # numba's IPython locator accepts a cell by its name and reads the cell's source from linecache, where IPython
    # holds it while the cell runs. The check does not change linecache's entry for the cell.
    import linecache
    import types
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    cell = "<ipython-input-3-0123456789ab>"
    source = "def f():\n    pass\n"
    monkeypatch.setitem(linecache.cache, cell, (len(source), None, source.splitlines(True), cell))
    configurations.check_cache_location(cell, configurations.LONGEST_CACHE_FILE_NAME)
    assert (ipython_dir / "numba_cache").is_dir()
    assert linecache.cache[cell][2] == source.splitlines(True)
    error = (errno.ENAMETOOLONG, "File name too long") if refused == "too long" else (errno.EACCES, "Permission denied")
    remedy = configurations.cache_remedy(cell, OSError(*error, str(ipython_dir / "numba_cache")), "silence")
    told = told.format(ipython_dir=ipython_dir, numba_cache=ipython_dir / "numba_cache")
    assert remedy == f"{told}; or silence", remedy


def test_a_frozen_application_whose_user_cache_directory_is_too_deep_is_told_a_shorter_path(tmp_path, monkeypatch):
    """A frozen application whose user cache directory cannot be created, because a component of its path is too long,
    gets a remedy that offers a shorter path as well as write access. numba's user-wide locator skips such a location as
    it skips an unwritable one, and numba's error that no locator accepts the file gives no reason."""
    import numba
    from numba.core import caching
    import numbox.core.configurations as configurations

    class UserCacheTooDeep:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / ("x" * 300) / "numba")

    monkeypatch.setattr(caching, "AppDirs", UserCacheTooDeep)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheTooDeep)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = str(tmp_path / "frozen" / "numbox" / "module.py")
    with pytest.raises(RuntimeError, match="no locator available") as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert configurations.is_a_cache_error(raised.value)
    remedy = configurations.cache_remedy(py_file, raised.value, "silence")
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        "effect here, because the source is not a file on disk: numba could not use that directory, "
        f"{tmp_path / ('x' * 300) / 'numba'}: make it writable, or put it at a shorter "
        f"path{configurations._moved_through()}, or silence"), remedy


# Each case gives the value of NUMBA_CACHE_LOCATOR_CLASSES and the remedy up to its last clause.
A_FROZEN_ZIP_PATH = {
    "the .zip locator first": (
        "ZipCacheLocator,UserWideCacheLocator",
        "numba's .zip locator, listed in NUMBA_CACHE_LOCATOR_CLASSES, {locators}, with no UserWideCacheLocator before "
        'it, takes this file for the ".zip" in its path and finds no archive there: list UserWideCacheLocator, the one '
        "locator for a frozen application, before it"),
    "the .zip locator and no user-wide one": (
        "ZipCacheLocator,InTreeCacheLocator",
        "numba's .zip locator, listed in NUMBA_CACHE_LOCATOR_CLASSES, {locators}, with no UserWideCacheLocator before "
        'it, takes this file for the ".zip" in its path and finds no archive there: list UserWideCacheLocator, the one '
        "locator for a frozen application, before it"),
    "numba's order, the user-wide locator passed over": (
        "",
        'numba\'s .zip locator takes this file for the ".zip" in its path and finds no archive there, after '
        "UserWideCacheLocator, which numba tries before it, could not use the user's cache directory, "
        "{user_cache_dir}: make it writable, or put it at a shorter path{moved_through}"),
}


@pytest.mark.parametrize("case", [pytest.param(case, id=case) for case in A_FROZEN_ZIP_PATH])
def test_a_frozen_applications_path_with_zip_in_it_and_no_archive_is_answered_for_the_zip_locators_error(
        tmp_path, monkeypatch, case):
    """A frozen application whose path contains ``.zip`` but has no part that ends in it gets a remedy for the ``.zip``
    locator's ``ValueError`` that depends on the list. If the list puts that locator before the user-wide one, which
    alone accepts the file, the remedy asks for the user-wide locator to be listed first. In numba's order, the remedy
    names the user's cache directory, which the user-wide locator could not use."""
    import numba
    from numba.core import caching
    import numbox.core.configurations as configurations

    class UserCacheTooDeep:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / ("x" * 300) / "numba")

    locators, told = A_FROZEN_ZIP_PATH[case]
    monkeypatch.setattr(caching, "AppDirs", UserCacheTooDeep)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheTooDeep)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = str(tmp_path / "not.zipped" / "frozen" / "numbox" / "module.py")
    with pytest.raises(ValueError, match="No zip file found") as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert configurations.is_a_cache_error(raised.value)
    remedy = configurations.cache_remedy(py_file, raised.value, "silence")
    told = told.format(locators=locators, user_cache_dir=tmp_path / ("x" * 300) / "numba",
                       moved_through=configurations._moved_through())
    assert remedy == f"{told}, or silence", remedy


@pytest.mark.parametrize("cell_file, order", [
    pytest.param("empty", "the IPython locator alone", marks=[pytest.mark.skipif(
        sys.version_info >= (3, 13), reason="linecache gives an empty file a blank line from Python 3.13")]),
    pytest.param("unreadable", "the IPython locator alone", marks=[needs_a_directory_it_cannot_write]),
    pytest.param("unreadable", "numba's order", marks=needs_a_directory_it_cannot_write),
])
def test_the_check_raises_inspects_error_naming_no_file_for_a_cell_file_on_disk_it_reads_no_source_in(
        tmp_path, monkeypatch, cell_file, order):
    """The check raises inspect's ``OSError``, with no errno and no file name, for a cell file on disk in which inspect
    reads no source, before numba's IPython locator tries its location. The remedy asks for the file to be made
    readable, with the source in it, and does not name a location that refused a file."""
    # inspect reads no source in a file that cannot be read, or before Python 3.13 in an empty one. From 3.13,
    # linecache holds an empty file as one blank line. is_a_cache_error counts inspect's error, as it counts every
    # OSError.
    import types
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    listed = "IPythonCacheLocator" if order == "the IPython locator alone" else ""
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", listed, raising=False)
    if order == "numba's order":
        # In numba's order, the in-tree and user-wide locators try their locations first and skip the file: the cell's
        # directory is read-only, and the user's cache directory lies under a file.
        from numba.core import caching
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("")

        class UserCacheUnderAFile:
            def __init__(self, appname, appauthor):
                self.user_cache_dir = str(blocker / "user-cache" / "numba")

        monkeypatch.setattr(caching, "AppDirs", UserCacheUnderAFile)
        monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderAFile)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    cell = tmp_path / "ipykernel_123" / "__init__.py"
    cell.parent.mkdir()
    cell.touch()
    if cell_file == "unreadable":
        cell.write_text("def f():\n    pass\n")
        cell.chmod(0)
    if order == "numba's order":
        cell.parent.chmod(0o555)
    try:
        with pytest.raises(OSError, match="could not get source code") as raised:
            configurations.check_cache_location(str(cell), configurations.LONGEST_CACHE_FILE_NAME)
    finally:
        cell.parent.chmod(0o755)
        cell.chmod(0o644)
    assert raised.value.filename is None and raised.value.errno is None
    assert configurations.is_a_cache_error(raised.value)
    assert not (ipython_dir / "numba_cache").exists() and not (cell.parent / "__pycache__").exists()
    remedy = configurations.cache_remedy(str(cell), raised.value, "silence")
    assert remedy == (
        f"numba's IPython locator reads the function's source from its file before trying its location, and read "
        f"none in {cell} (could not get source code): make the file readable, with the source in it; or silence"), remedy
    # inspect's errors name no file, so the remedy reads an error with no errno that names a file as one that a caller
    # passes on, and it names the location that the error gives.
    named = OSError("could not get source code")
    named.filename = str(tmp_path / "elsewhere")
    remedy = configurations.cache_remedy(str(cell), named, "silence")
    assert remedy.startswith(f"that location, {tmp_path / 'elsewhere'}, where no file can be written"), remedy


def test_a_source_not_on_disk_under_a_directory_named_for_a_zip_is_in_no_zip(tmp_path, monkeypatch):
    """A source in an ``.egg`` under a directory named ``my.zipped`` is in no ``.zip``, so it gets the archive remedy,
    under a list without ``ZipCacheLocator`` and under numba's order alike. The ``.zip`` locator finds the archive only
    by a part of the path that ends in ``.zip``, and under a list of that locator alone, numba raises its
    ``ValueError``."""
    # Under the list, the archive remedy names the locator that each of its two ways needs. Under numba's order, the
    # .zip locator accepts the path for its ".zip" and raises, because no part of the path ends in .zip.
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    py_file = str(tmp_path / "my.zipped" / "package.egg" / "numbox" / "module.py")
    archive_remedy = (
        "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, install numbox with "
        "its source files on disk, unpacked from any archive, or import it from a .zip holding its source files, "
        "which numba caches in the user's cache directory")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "InTreeCacheLocator", raising=False)
    with pytest.raises(RuntimeError, match="no locator available") as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    remedy = configurations.cache_remedy(py_file, raised.value, "silence")
    assert remedy == (
        f"{archive_remedy} through ZipCacheLocator, once NUMBA_CACHE_LOCATOR_CLASSES, InTreeCacheLocator, lists it; "
        "or silence"), remedy
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    with pytest.raises(ValueError, match="No zip file found") as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert configurations.is_a_cache_error(raised.value)
    assert configurations.cache_remedy(py_file, raised.value, "silence") == f"{archive_remedy}; or silence"


@pytest.mark.parametrize("entry", ["..InTreeCacheLocator", ".a.InTreeCacheLocator"])
def test_a_locator_entry_with_a_relative_module_path_is_left_out_and_numbas_error_raised_as_it_was(
        tmp_path, monkeypatch, entry):
    """A dotted entry with a relative module path, such as ``..InTreeCacheLocator``, makes the check raise importlib's
    ``TypeError``, which is not a cache error, and the remedy reads the list without that entry. numba does not catch
    that error, as it does not catch the ``ValueError`` for an empty module path, so the first function that numba
    decorates raises it."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core import caching
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    assert configurations._locators(f"{entry},InTreeCacheLocator") == [caching.InTreeCacheLocator]
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", f"{entry},InTreeCacheLocator", raising=False)
    with pytest.raises(TypeError, match="relative import") as raised:
        configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
    assert not configurations.is_a_cache_error(raised.value)
    failure = OSError(errno.EACCES, "Permission denied", str(py_file.parent / "__pycache__"))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy.startswith(f"numba caches this file in {py_file.parent / '__pycache__'},"), remedy


@pytest.mark.parametrize("entry", ["CacheImpl", "pathlib.PurePath"])
def test_a_locator_entry_naming_a_class_with_no_from_function_is_left_out_as_one_naming_no_class_is(
        tmp_path, monkeypatch, entry):
    """A listed class without ``from_function``, such as numba's ``CacheImpl``, is omitted from the remedy's list, in
    the same way as an entry that names no class. The check raises numba's ``AttributeError``, which is not a cache
    error, and the remedy with the entry equals the remedy without it, so the class neither counts as a locator nor
    removes the offer of ``NUMBA_CACHE_DIR``."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core import caching
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    assert configurations._locators(f"{entry},InTreeCacheLocator") == [caching.InTreeCacheLocator]
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", f"{entry},InTreeCacheLocator", raising=False)
    with pytest.raises(AttributeError, match="from_function") as raised:
        configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
    assert not configurations.is_a_cache_error(raised.value)
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(tmp_path / "elsewhere"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", f"{entry},UserProvidedCacheLocator,InTreeCacheLocator",
                        raising=False)
    with_the_entry = configurations.cache_remedy(str(py_file), failure, "silence")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "UserProvidedCacheLocator,InTreeCacheLocator",
                        raising=False)
    assert with_the_entry == configurations.cache_remedy(str(py_file), failure, "silence")
    assert "NUMBA_CACHE_DIR set to a short path" in with_the_entry, with_the_entry
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", entry, raising=False)
    alone = configurations.cache_remedy(str(py_file), failure, "silence")
    assert alone == (
        f"numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, {entry}, says, and none of those locators takes a "
        "source file on disk: list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or "
        "UserWideCacheLocator, or silence"), alone


@pytest.mark.parametrize("refused, told", [
    ("a full disk", "no file can be written in the location numba took, {location} (No space left on device), so make "
                    "room there, or make it writable"),
    ("too long", "the path is too long for the file system, so put the location numba took, {location}, at a shorter "
                 "path"),
])
def test_an_error_naming_no_file_with_ipython_not_importable_says_nothing_of_the_user_cache_directory(
        tmp_path, monkeypatch, refused, told):
    """Without IPython, an error that names no file for a member in an ``ipykernel_`` directory gets a remedy that does
    not say that numba caches the member in the user's cache directory. numba's IPython locator comes before the
    ``.zip`` locator for such a member, so the error may belong to the IPython locator, whether or not IPython gives a
    directory."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setitem(sys.modules, "IPython.paths", None)
    monkeypatch.setitem(sys.modules, "IPython.utils.path", None)
    py_file = str(tmp_path / "bundle.zip" / "ipykernel_123" / "cell.py")
    location = tmp_path / "user-cache" / "numba" / _CacheLocator.get_suitable_cache_subpath(py_file)
    error = ((errno.ENOSPC, "No space left on device") if refused == "a full disk" else
             (errno.ENAMETOOLONG, "File name too long"))
    remedy = configurations.cache_remedy(py_file, OSError(*error), "silence")
    assert remedy == (
        "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: "
        f"{told.format(location=location)}, or silence"), remedy


@pytest.mark.parametrize("locators, told", [
    ("IPythonCacheLocator",
     "numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, IPythonCacheLocator, says: make one of those locations "
     "writable, or silence"),
    ("IPythonCacheLocator,InTreeCacheLocator",
     "the path is too long for the file system: that location, {named}, at a shorter path; or silence"),
])
def test_an_error_naming_ipythons_location_with_ipython_not_importable_is_told_the_list_where_no_locator_has_a_place(
        tmp_path, monkeypatch, locators, told):
    """Without IPython, a caller's error that names IPython's location for a cell file on disk gets the remedy for a
    location that no locator chose, if another listed locator has a place for the file. If no listed locator has one,
    the error gets the remedy that names the list. numba's IPython locator does not catch the ``ImportError`` at
    decoration, so numba never chooses IPython's location."""
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setitem(sys.modules, "IPython.paths", None)
    monkeypatch.setitem(sys.modules, "IPython.utils.path", None)
    py_file = tmp_path / "ipykernel_123" / "cell.py"
    py_file.parent.mkdir()
    py_file.write_text("def cell():\n    pass\n")
    with pytest.raises(ImportError) as raised:
        configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
    assert not configurations.is_a_cache_error(raised.value)
    named = tmp_path / "ipython" / "numba_cache" / "cell-1.py312.nbi"
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(named))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == told.format(named=named), remedy


@pytest.mark.parametrize("placement, refused", [
    pytest.param("a source on disk", "too long", marks=pytest.mark.skipif(
        os.name == "nt", reason="Windows refuses a 300-character directory name as invalid, not as too long")),
    pytest.param("a source on disk", "unwritable", marks=needs_a_directory_it_cannot_write),
    pytest.param("a .zip member", "unwritable", marks=needs_a_directory_it_cannot_write),
    ("a source on disk", "naming no file"), ("a .zip member", "naming no file"),
    ("a source on disk, the in-tree locator after", "naming no file"),
])
def test_a_listed_class_of_none_of_numbas_families_is_told_the_location_the_error_names(
        tmp_path, monkeypatch, placement, refused):
    """A reader's own locator class, outside numba's families, gets remedies that name the location in the check's
    error, with the cause, for a source on disk and for a ``.zip`` member. For an error that names no file, the remedy
    describes the location that numba chose as the class's own, because only the class can name it."""
    # A reader can list such a class in NUMBA_CACHE_LOCATOR_CLASSES. The class has a from_function and a place of its
    # own, numba caches the file in that place, and the check's error names it.
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    where = tmp_path / ("x" * 300) if refused == "too long" else tmp_path / "our-cache"

    class OurLocator(_CacheLocator):
        def __init__(self, py_func, py_file):
            self._py_file = py_file
            self._cache_path = str(where / self.get_suitable_cache_subpath(py_file))

        def get_cache_path(self):
            return self._cache_path

        def get_source_stamp(self):
            return 0

        def get_disambiguator(self):
            return "x"

        @classmethod
        def from_function(cls, py_func, py_file):
            return cls(py_func, py_file)

    ours = types.ModuleType("our_locators")
    ours.OurLocator = OurLocator
    monkeypatch.setitem(sys.modules, "our_locators", ours)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    listed = "our_locators.OurLocator,InTreeCacheLocator" if placement.endswith("after") else "our_locators.OurLocator"
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", listed, raising=False)
    assert configurations._locators("our_locators.OurLocator") == [OurLocator]
    if placement.startswith("a source on disk"):
        py_file = tmp_path / "site" / "package" / "module.py"
        py_file.parent.mkdir(parents=True)
        py_file.write_text("")
    else:
        py_file = tmp_path / "bundle.zip" / "package" / "module.py"
    if refused == "naming no file":
        remedy = configurations.cache_remedy(str(py_file), OSError(errno.ENOSPC, "No space left on device"), "silence")
        if placement.startswith("a source on disk"):
            # OurLocator accepts the file when the remedy asks it, so numba never reaches the in-tree locator listed
            # after it, and the remedy does not name the __pycache__.
            assert remedy == (
                "the location numba took, OurLocator's own, where no file can be written (No space left on device): "
                "make room there, or make it writable; or silence"), remedy
        else:
            assert remedy == (
                "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: no file can be written "
                "in the location numba took, OurLocator's own (No space left on device), so make room there, or make "
                "it writable, or silence"), remedy
        return
    if refused == "unwritable":
        where.mkdir()
        where.chmod(0o555)
    try:
        with pytest.raises(OSError) as raised:
            configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
    finally:
        if refused == "unwritable":
            where.chmod(0o755)
    failure = raised.value
    assert configurations.is_a_cache_error(failure)
    named = os.fsdecode(failure.filename)
    assert named.startswith(str(where)), named
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    if refused == "too long":
        assert failure.errno == errno.ENAMETOOLONG
        assert remedy == (
            f"the path is too long for the file system: that location, {named}, at a shorter path; or silence"), remedy
    elif placement == "a source on disk":
        assert remedy == (
            f"that location, {named}, where no file can be written (Permission denied): make room there, or make it "
            "writable; or silence"), remedy
    else:
        assert remedy == (
            "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: no file can be written at "
            f"that location, {named} (Permission denied), so make room there, or make it writable, or silence"), remedy


def test_a_frozen_applications_no_locator_error_under_a_class_of_none_of_numbas_families_is_told_the_user_wide_locator(
        tmp_path, monkeypatch):
    """A frozen application under a list of a reader's own locator alone, which skips the file, gets the remedy that
    asks for ``UserWideCacheLocator`` to be listed. numba never tries the user's cache directory under that list, so
    the remedy does not ask for that directory to be made writable."""
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class OurLocator(_CacheLocator):
        @classmethod
        def from_function(cls, py_func, py_file):
            return None

    ours = types.ModuleType("our_locators")
    ours.OurLocator = OurLocator
    monkeypatch.setitem(sys.modules, "our_locators", ours)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "our_locators.OurLocator", raising=False)
    py_file = str(tmp_path / "frozen" / "numbox" / "module.py")
    with pytest.raises(RuntimeError, match="no locator available") as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert configurations.is_a_cache_error(raised.value)
    remedy = configurations.cache_remedy(py_file, raised.value, "silence")
    assert remedy == (
        "numba caches a frozen application through UserWideCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, "
        "our_locators.OurLocator, leaves it out: list it, or silence"), remedy


def test_an_error_naming_a_file_descriptor_names_no_file(tmp_path, monkeypatch):
    """An ``OSError`` whose file name is an integer descriptor, as ``os.stat`` raises for one, gets the remedy for an
    error that names no file. The remedy reads only a string, bytes or a path object as a file name, so it lists the
    locations numba could have chosen and does not name the descriptor."""
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    with pytest.raises(OSError) as raised:
        os.stat(987654)
    failure = raised.value
    assert failure.filename == 987654 and configurations.is_a_cache_error(failure)
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == configurations.cache_remedy(str(py_file), OSError(failure.errno, failure.strerror), "silence")
    assert remedy.startswith("the location numba took, one of "), remedy
    assert "987654" not in remedy


@pytest.mark.parametrize("placement", [
    pytest.param("a cell", id="a cell, numba's order"),
    pytest.param("a cell, a class of none of numba's families listed too",
                 id="a cell, the IPython locator and a class of the reader's own"),
    pytest.param("a cell, a class of none of numba's families listed first",
                 id="a cell, a class of the reader's own and the IPython locator"),
    pytest.param("a .zip member in an ipykernel directory", id="a .zip member, the IPython locator alone"),
])
@pytest.mark.parametrize("ipython", ["cannot make its directory", "raises for its directory"])
def test_a_file_numbas_ipython_locator_alone_takes_and_passes_over_is_told_ipythons_location(
        tmp_path, monkeypatch, placement, ipython):
    """A cell, or a member in an ``ipykernel_`` directory under a list of the IPython locator alone, gets the remedy for
    IPython's location when that locator skips the file. The remedy asks for the location to be made writable, or for
    IPython's home to be made writable if IPython raises for its directory. If the list also holds a reader's own
    locator, the remedy names that locator as having skipped the file."""
    # numba's IPython locator skips its location if it cannot create it or write in it, or if IPython raises OSError
    # for its directory. No other locator accepts a cell, so numba then raises its error that no locator accepts the
    # file. The archive remedy, with source files on disk or a .zip, would not help a cell.
    import linecache
    import types
    import numba
    import numbox.core.configurations as configurations

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    ipython_dir = str(blocker / "ipython")

    def nowhere():
        raise PermissionError(errno.EACCES, "Permission denied", str(tmp_path / "home"))

    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = (lambda: ipython_dir) if ipython == "cannot make its directory" else nowhere
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    py_file = "<ipython-input-3-0123456789ab>" if placement.startswith("a cell") else str(
        tmp_path / "bundle.zip" / "ipykernel_123" / "cell.py")
    if placement.startswith("a cell"):
        # IPython holds a cell's source in linecache while the cell runs, and numba's IPython locator reads the source
        # there.
        source = "def f():\n    pass\n"
        monkeypatch.setitem(linecache.cache, py_file, (len(source), None, source.splitlines(True), py_file))
    else:
        # inspect reads a .zip member's source through its module's loader. The probe's loader reads the member from
        # the archive, as zipimport does, so the test writes the member's source into the archive.
        with zipfile.ZipFile(tmp_path / "bundle.zip", "w") as zipped:
            zipped.writestr("ipykernel_123/cell.py", "def cell():\n    pass\n")
    if placement == "a cell":
        monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    elif placement.startswith("a cell"):
        from numba.core.caching import _CacheLocator

        class OurLocator(_CacheLocator):
            @classmethod
            def from_function(cls, py_func, py_file):
                return None

        ours = types.ModuleType("our_locators")
        ours.OurLocator = OurLocator
        monkeypatch.setitem(sys.modules, "our_locators", ours)
        listed = ("IPythonCacheLocator,our_locators.OurLocator" if placement.endswith("too")
                  else "our_locators.OurLocator,IPythonCacheLocator")
        monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", listed, raising=False)
    else:
        monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "IPythonCacheLocator", raising=False)
    passed_over = ", and OurLocator, listed too, passed the file over" if "none of numba's" in placement else ""
    with pytest.raises(RuntimeError, match="no locator available") as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert configurations.is_a_cache_error(raised.value)
    remedy = configurations.cache_remedy(py_file, raised.value, "silence")
    if ipython == "cannot make its directory":
        assert remedy == (
            f"numba caches this file in {os.path.join(ipython_dir, 'numba_cache')}, which it could not make or write "
            f"in{passed_over}: "
            f"make it writable, or put IPython's cache directory, {ipython_dir}, at a shorter path; or silence"), remedy
    else:
        assert remedy == (
            "numba caches this file in numba_cache under IPython's cache directory, which IPython raised for, under a "
            f"home it cannot write{passed_over}: make that home writable; or silence"), remedy


def test_the_check_raises_the_stamps_error_naming_the_archive_a_moved_zips_members_code_names(tmp_path, monkeypatch):
    """The check raises ``FileNotFoundError`` naming the archive that a member's code names, if that archive has moved,
    before numba tries any location. numba reads the archive's stamp at decoration, and the remedy names the archive as
    missing and asks for the source files to be shipped."""
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    archive = tmp_path / "build" / "numbox.zip"
    py_file = str(archive / "numbox" / "module.py")
    with pytest.raises(FileNotFoundError) as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert raised.value.filename == str(archive)
    assert configurations.is_a_cache_error(raised.value)
    assert not (tmp_path / "user-cache").exists()
    remedy = configurations.cache_remedy(py_file, raised.value, "silence")
    # Under numba 0.67 and earlier, numba caches members that were compiled to name the archive's present path, so the
    # remedy offers that compilation. numba 0.68 and later also read the source from the archive, so under those
    # releases the remedy asks for the source files alone.
    recompile = "" if numba_version >= 68 else "compile the archive's .pyc members to name its path now, or "
    assert remedy == (
        f"the module's code names {archive}, which is not there: its .pyc was compiled to name that path, so "
        f"{recompile}ship its source files; or silence"), remedy


# A_NAME_TOO_LONG maps each case of the test below to the marks that it runs under. A case's name gives the source's
# place, the settings and the error.
A_NAME_TOO_LONG = {
    "a .zip member, numba's order": [on_windows_under_numba_0_68_0],
    "on disk, NUMBA_CACHE_DIR set, the user-provided locator alone": [],
    "on disk, numba's order, the __pycache__ unwritable": needs_a_directory_it_cannot_write,
    "on disk, an error naming the location under NUMBA_CACHE_DIR, the user-provided locator alone": [],
    "on disk under a directory named for a .zip, a caller's no-locator error, the .zip locator then the user-wide one":
        [],
    "a frozen application": [],
    "a frozen application, a name that just fits, the user's cache directory too deep": [],
    "on disk, an error naming the location under NUMBA_CACHE_DIR as an invalid name, the user-provided locator alone":
        [],
    "a .zip member, an error naming the location as an invalid name": [],
    "on disk, NUMBA_CACHE_DIR set, the user-provided locator alone, a name of CJK characters": [
        pytest.mark.skipif(
            os.name == "nt" or sys.platform == "darwin",
            reason="Windows and macOS count a name in UTF-16 units, where it fits",
        )
    ],
    "a .zip member, numba's order, a name of astral characters, counted as macOS counts": [
        on_windows_under_numba_0_68_0
    ],
}


@pytest.mark.parametrize("case", [pytest.param(case, marks=marks, id=case) for case, marks in A_NAME_TOO_LONG.items()])
def test_a_source_directory_whose_name_is_too_long_for_numbas_cache_directory_name_is_told_a_shorter_name(
        tmp_path, monkeypatch, case):
    """If numba's directory name for a source, the source directory's name and 41 bytes more, is too long for the file
    system, the remedy names that name and its length and asks for a directory of a shorter name. No path is short
    enough for such a name, so for a source on disk the remedy offers the ``__pycache__`` beside the source, whose name
    holds no hash, instead of a shorter path."""
    # numba names its directory under NUMBA_CACHE_DIR or the user's cache directory with the source directory's name
    # and a hash of its path. numba skips a location whose name is too long, and for a .zip member the first save fails
    # there. A name whose length equals the file system's limit gets the remedy for a location that numba could not use.
    import numba
    from numba.core import caching
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    just_fits = "just fits" in case
    if "astral" in case:
        monkeypatch.setattr(sys, "platform", "darwin")
    limit = os.statvfs(tmp_path).f_namemax if hasattr(os, "statvfs") else 255
    too_deep = tmp_path / ("x" * 300) / "numba" if "too deep" in case else None

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(too_deep or tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(caching, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(sys, "frozen", case.startswith("a frozen"), raising=False)
    cache_dir = str(tmp_path / "cache") if "NUMBA_CACHE_DIR" in case else ""
    if cache_dir:
        os.mkdir(cache_dir)
    monkeypatch.setattr(numba.config, "CACHE_DIR", cache_dir)
    listed = "UserProvidedCacheLocator" if "user-provided locator alone" in case else ""
    if "the .zip locator then the user-wide one" in case:
        listed = "ZipCacheLocator,UserWideCacheLocator"
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", listed, raising=False)
    directory = "d" * (limit - 41 if just_fits else limit - 35)
    # The remedy counts a name as the file system does: in bytes, or in UTF-16 units on Windows and macOS. A name of
    # CJK characters fits in UTF-16 units and not in bytes. A name of astral characters, each two UTF-16 units and four
    # bytes, fits in code points and not in UTF-16 units.
    if "CJK" in case:
        directory = "\u5b57" * ((limit - 41) // 3 + 1)
    if "astral" in case:
        directory = "\U0001f600" * ((limit - 41) // 2 + 1)
    if case.startswith("a .zip"):
        archive = tmp_path / "app.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr(f"{directory}/module.py", "x = 1\n")
        py_file = str(archive / directory / "module.py")
    elif case.startswith("on disk"):
        site = tmp_path / "site" / "a.zip" if "named for a .zip" in case else tmp_path / "site"
        py_file = str(site / directory / "module.py")
        os.makedirs(os.path.dirname(py_file))
        with open(py_file, "w", encoding="utf-8") as handle:
            handle.write("x = 1\n")
    else:
        py_file = str(tmp_path / "frozen" / directory / "module.py")
    subpath = _CacheLocator.get_suitable_cache_subpath(py_file)
    counted_in = "characters" if os.name == "nt" or sys.platform == "darwin" else "bytes"
    length = len(subpath.encode("utf-16-le")) // 2 if counted_in == "characters" else len(subpath.encode())
    assert length == limit if just_fits else length > limit
    if "CJK" in case or "astral" in case:
        assert len(subpath) < limit
    clause = (
        f"numba names the directory it caches this file in after the source's directory, {subpath}, {length} "
        f"{counted_in}, which is longer than a name can be on that file system ({limit})")
    if "unwritable" in case:
        Path(py_file).parent.chmod(0o555)
    try:
        if "an error naming" in case:
            # Windows refuses a name too long as an invalid one, with EINVAL, and the remedy measures the name
            # whatever the errno.
            refused = ((errno.EINVAL, "Invalid argument") if "invalid name" in case
                       else (errno.ENAMETOOLONG, "File name too long"))
            named = os.path.join(cache_dir or str(tmp_path / "user-cache" / "numba"), subpath)
            failure = OSError(*refused, named)
        elif "a caller's no-locator error" in case:
            # numba's .zip locator accepts the path without trying its location, so numba raises no such error here,
            # and the test builds it. The remedy measures the name only for the locators that numba reaches, and the
            # user-wide locator listed after the .zip one is not among them.
            failure = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
        else:
            expected = OSError if case.startswith("a .zip") else RuntimeError
            with pytest.raises(expected) as raised:
                configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
            failure = raised.value
        assert configurations.is_a_cache_error(failure)
        remedy = configurations.cache_remedy(py_file, failure, "silence")
    finally:
        if "unwritable" in case:
            Path(py_file).parent.chmod(0o755)
    if just_fits:
        assert remedy == (
            "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
            f"effect here, because the source is not a file on disk: numba could not use that directory, {too_deep}: "
            f"make it writable, or put it at a shorter path{configurations._moved_through()}, or silence"), remedy
    elif case.startswith("on disk, numba's order"):
        assert failure.filename is None if isinstance(failure, OSError) else True
        assert remedy == (
            f"{clause}: put the file in a directory of a shorter name, or make the __pycache__ beside the source, "
            f"{os.path.join(os.path.dirname(py_file), '__pycache__')}, writable, where InTreeCacheLocator caches it "
            "under no such name; "
            "or silence"), remedy
    elif case.startswith("on disk"):
        # The user-provided locator alone, with the variable unset, chose no location and measured no name. The remedy
        # then names the variable, without which that locator accepts nothing, beside the shorter name.
        variable = (", with NUMBA_CACHE_DIR set, which UserProvidedCacheLocator takes nothing without"
                    if listed == "UserProvidedCacheLocator" and not cache_dir else "")
        assert remedy == (
            f"{clause}: put the file in a directory of a shorter name{variable}, or list InTreeCacheLocator, which "
            "caches beside the source, in its __pycache__, under no such name; or silence"), remedy
    else:
        assert remedy == (
            "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
            f"effect here, because the source is not a file on disk: {clause}, so put the file in a directory of a "
            "shorter name, or silence"), remedy


@pytest.mark.skipif(os.name == "nt", reason="Windows decodes any byte of a path")
@pytest.mark.parametrize("placement", [
    "an .egg member, numba's order",
    pytest.param("on disk, the in-tree locator alone, its __pycache__ unwritable", marks=[
        needs_a_directory_it_cannot_write,
        pytest.mark.skipif(sys.platform == "darwin", reason="APFS takes no name that is not UTF-8")]),
])
def test_a_source_whose_path_holds_bytes_that_are_not_utf8_is_answered_where_numba_hashes_no_path(
        tmp_path, monkeypatch, placement):
    """A source with bytes in its path that are not UTF-8 gets a remedy, and not a ``UnicodeEncodeError``, if numba
    hashes no path for it. The test covers an ``.egg`` member, which no locator accepts, and a source on disk under the
    in-tree locator alone. numba encodes the path as UTF-8 only to name its directory under ``NUMBA_CACHE_DIR`` or the
    user's cache directory."""
    # Python decodes such bytes in a path to surrogates, which UTF-8 cannot encode.
    import numba
    import numbox.core.configurations as configurations

    directory = tmp_path / os.fsdecode(b"caf\xe9")
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    if placement.startswith("an .egg"):
        monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
        py_file = str(directory / "numbox-0.0.0-py3.12.egg" / "numbox" / "module.py")
        failure = RuntimeError(f"cannot cache function 'f': no locator available for file '{py_file}'")
        expected = f"{ARCHIVE_REMEDY}; or silence"
    else:
        monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "InTreeCacheLocator", raising=False)
        (directory / "package").mkdir(parents=True)
        py_file = str(directory / "package" / "module.py")
        with open(py_file, "w", encoding="utf-8") as handle:
            handle.write("x = 1\n")
        (directory / "package").chmod(0o555)
        try:
            with pytest.raises(RuntimeError, match="no locator available") as raised:
                configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
        finally:
            (directory / "package").chmod(0o755)
        failure = raised.value
        expected = ("numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, InTreeCacheLocator, says: make one of those "
                    "locations writable, or silence")
    with pytest.raises(UnicodeEncodeError):
        py_file.encode()
    assert configurations.is_a_cache_error(failure)
    assert configurations.cache_remedy(py_file, failure, "silence") == expected


@pytest.mark.parametrize("ipython", ["giving its directory", "giving a temporary directory per call"])
def test_a_cell_whose_source_linecache_does_not_hold_raises_inspects_error_as_numba_does(tmp_path, monkeypatch, ipython):
    """The check raises inspect's ``OSError`` for a cell whose source is not in linecache, before numba's IPython
    locator tries its location, as numba raises it at decoration. The remedy says that numba caches a cell only while
    IPython holds its source, also under an IPython that gives temporary directories."""
    # IPython holds a cell's source in linecache while the cell runs, and numba's IPython locator reads it there.
    import linecache
    import tempfile
    import types
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    temporary = ipython != "giving its directory"
    paths.get_ipython_cache_dir = (lambda: tempfile.mkdtemp(dir=tmp_path)) if temporary else (lambda: str(ipython_dir))
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    cell = "<ipython-input-7-0123456789ab>"
    linecache.cache.pop(cell, None)
    with pytest.raises(OSError, match="could not get source code") as raised:
        configurations.check_cache_location(cell, configurations.LONGEST_CACHE_FILE_NAME)
    assert raised.value.errno is None and raised.value.filename is None
    assert temporary or not ipython_dir.exists()
    assert configurations.is_a_cache_error(raised.value)
    remedy = configurations.cache_remedy(cell, raised.value, "silence")
    assert remedy == (
        f"numba's IPython locator reads a cell's source from linecache, which holds none for {cell} (could not get "
        "source code), and numba caches a cell only while IPython holds its source; or silence"), remedy


@pytest.mark.parametrize("ipython", ["giving its directory", "giving a temporary directory per call"])
@pytest.mark.parametrize("name", ["bundle.zip", "bundle.egg"])
def test_a_zip_member_in_an_ipykernel_directory_compiled_to_pyc_alone_raises_inspects_error_as_numba_does(
        tmp_path, monkeypatch, name, ipython):
    """The check raises inspect's ``OSError`` for a member in an ``ipykernel_`` directory shipped as ``.pyc`` alone, as
    numba raises it at decoration, and the remedy is the archive remedy, which asks for the source files. numba's
    IPython locator reads the source through the module's loader, and zipimport gives none for such a member."""
    import linecache
    import tempfile
    import types
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    temporary = ipython != "giving its directory"
    paths.get_ipython_cache_dir = (lambda: tempfile.mkdtemp(dir=tmp_path)) if temporary else (lambda: str(ipython_dir))
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    source = tmp_path / "cell.py"
    source.write_text("def cell():\n    pass\n")
    archive = tmp_path / name
    compiled = py_compile.compile(str(source), cfile=str(tmp_path / "cell.pyc"),
                                  dfile=str(archive / "ipykernel_123" / "cell.py"), doraise=True)
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.write(compiled, "ipykernel_123/cell.pyc")
    py_file = str(archive / "ipykernel_123" / "cell.py")
    linecache.cache.pop(py_file, None)
    with pytest.raises(OSError, match="could not get source code") as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert raised.value.errno is None and raised.value.filename is None
    # linecache stores the probe's loader in an entry of one item under the member's name, to ask it again later, and
    # the check removes that entry.
    assert py_file not in linecache.cache
    assert temporary or not ipython_dir.exists()
    assert configurations.is_a_cache_error(raised.value)
    remedy = configurations.cache_remedy(py_file, raised.value, "silence")
    if name.endswith(".egg"):
        # The remedy asks for an archive that numba's .zip locator never accepts to be renamed to end in .zip, with its
        # source files in it.
        assert remedy == (
            "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, install numbox "
            f"with its source files on disk, unpacked from any archive, or rename the archive, {archive}, to end in "
            ".zip, lower-case, which numba's .zip locator looks for in the path, with its source files in it, so that "
            "numba caches it in the user's cache directory; or silence"), remedy
    else:
        assert remedy == f"{ARCHIVE_REMEDY}; or silence", remedy


# A_ZIP_REFUSED holds the remedies, by errno, for a .zip member whose error names its location under the user's cache
# directory, without the opening that every such remedy shares.
A_ZIP_REFUSED = {
    "unwritable": ((errno.EACCES, "Permission denied"),
                   "no file can be written in that directory, {location} (Permission denied), so make room there, or "
                   "make it writable"),
    "too long": ((errno.ENAMETOOLONG, "File name too long"),
                 "the path is too long for the file system, so put that directory, {location}, at a shorter "
                 "path{moved_through}"),
}


@pytest.mark.parametrize("refused", list(A_ZIP_REFUSED))
def test_a_zip_import_refused_its_location_under_a_list_without_the_user_wide_locator_is_told_the_location(
        tmp_path, monkeypatch, refused):
    """A ``.zip`` member's error that names its location, under a list with the ``.zip`` locator and without the
    user-wide one, gets the remedy for that location, as under numba's order. The remedy that asks for
    ``UserWideCacheLocator`` to be listed is for a frozen application, which only that locator accepts."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "InTreeCacheLocator,ZipCacheLocator", raising=False)
    py_file = str(tmp_path / "bundle.zip" / "package" / "module.py")
    location = tmp_path / "user-cache" / "numba" / _CacheLocator.get_suitable_cache_subpath(py_file)
    error, told = A_ZIP_REFUSED[refused]
    # The location exists and refuses the file. For a location that does not exist, the remedy would also name the
    # directory under which numba cannot create it.
    location.mkdir(parents=True)
    remedy = configurations.cache_remedy(py_file, OSError(*error, str(location)), "silence")
    told = told.format(location=location, moved_through=configurations._moved_through())
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        f"effect here, because the source is not a file on disk: {told}, or silence"), remedy


@pytest.mark.parametrize("refused", list(A_ZIP_REFUSED))
def test_a_zip_error_naming_a_file_numba_writes_in_its_location_is_told_the_location(tmp_path, monkeypatch, refused):
    """A ``.zip`` member's error that names a file in its location, as numba's first save names its temporary file,
    gets the remedy for the location, a directory, and not for the file. The remedy compares the error's path and that
    path's directory with the location, as it does for a source on disk."""
    # The check's error names the location instead, and a caller can pass on the first save's error.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = str(tmp_path / "bundle.zip" / "package" / "module.py")
    location = tmp_path / "user-cache" / "numba" / _CacheLocator.get_suitable_cache_subpath(py_file)
    named = location / "module-1.py312.nbi.tmp.0123456789abcdef"
    error, told = A_ZIP_REFUSED[refused]
    remedy = configurations.cache_remedy(py_file, OSError(*error, str(named)), "silence")
    told = told.format(location=location, moved_through=configurations._moved_through())
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        f"effect here, because the source is not a file on disk: {told}, or silence"), remedy


@pytest.mark.parametrize("refused", list(A_ZIP_REFUSED))
def test_a_frozen_applications_error_naming_a_file_numba_writes_in_its_location_is_told_the_location(
        tmp_path, monkeypatch, refused):
    """A frozen application's error that names a file in its location gets the remedy for that location under the
    user's cache directory, as a ``.zip`` member's error does. numba's user-wide locator accepts a frozen application's
    file, which is not on disk, and caches it in a directory of its own there."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = str(tmp_path / "frozen" / "numbox" / "module.py")
    location = tmp_path / "user-cache" / "numba" / _CacheLocator.get_suitable_cache_subpath(py_file)
    named = location / "module-1.py312.nbi.tmp.0123456789abcdef"
    error, told = A_ZIP_REFUSED[refused]
    remedy = configurations.cache_remedy(py_file, OSError(*error, str(named)), "silence")
    told = told.format(location=location, moved_through=configurations._moved_through())
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        f"effect here, because the source is not a file on disk: {told}, or silence"), remedy


def test_a_file_not_found_error_that_names_no_file_for_a_source_not_on_disk_gets_the_archive_remedy(tmp_path, monkeypatch):
    """A caller's ``FileNotFoundError`` that names no file, for a source that is not on disk, gets the archive remedy
    and not the remedy for a moved archive. No locator accepts a source that is not on disk, not in a ``.zip`` and not a
    frozen application's source, so the error gets the same remedy as numba's error that no locator accepts the file."""
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = str(tmp_path / "site" / "package" / "module.py")
    remedy = configurations.cache_remedy(py_file, FileNotFoundError(errno.ENOENT, "No such file or directory"), "silence")
    assert remedy == f"{ARCHIVE_REMEDY}; or silence", remedy


def test_a_zip_member_whose_source_the_archive_does_not_hold_is_told_the_source_files_naming_the_member(
        tmp_path, monkeypatch):
    """zipfile's ``KeyError`` for a member whose source the archive does not hold is a cache error from numba 0.68, and
    the remedy names the member as a module shipped as ``.pyc`` alone and asks for the source files. numba 0.68 and
    later read the member's source from the archive for its stamp."""
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = str(tmp_path / "numbox.zip" / "numbox" / "core" / "bindings" / "libm.py")
    failure = KeyError("There is no item named 'numbox/core/bindings/libm.py' in the archive")
    assert configurations.is_a_cache_error(failure) == (numba_version >= 68)
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    assert remedy == (
        "numba 0.68 and later read a .zip member's source from the archive, and this one holds none under "
        "numbox/core/bindings/libm.py, the module shipped as .pyc alone: to cache, install numbox with its source "
        "files on disk, unpacked from any archive, or import it from a .zip holding its source files, which numba "
        "caches in the user's cache directory; or silence"), remedy


def test_a_zip_members_name_built_with_windows_separator_is_told_numba_0_68_1(tmp_path, monkeypatch):
    """On Windows, a ``KeyError`` for a member named with backslashes, which the archive holds under forward slashes,
    gets the remedy for numba issue 10889: the install on disk or a numba release with the fix. numba 0.68.0 builds a
    member's name with the platform's separator, and numba 0.68.1 reads the name as the archive records it."""
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setattr(os, "name", "nt")
    archive = tmp_path / "numbox.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("numbox/core/bindings/libm.py", "")
    py_file = str(archive / "numbox" / "core" / "bindings" / "libm.py")
    # zipfile quotes the name in its message as repr does, so each backslash is doubled.
    failure = KeyError("There is no item named 'numbox\\\\core\\\\bindings\\\\libm.py' in the archive")
    assert configurations.is_a_cache_error(failure) == (numba_version >= 68)
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    assert remedy == (
        "numba 0.68.0 reads a .zip member's source by a name built with Windows's separator, "
        "numbox\\core\\bindings\\libm.py, which no archive holds (numba issue "
        "https://github.com/numba/numba/issues/10889, fixed for numba 0.68.1): to cache, install numbox with its "
        "source files on disk, unpacked from any archive, which no separator touches, or a numba release that "
        "carries the fix, which reads the member as the archive names it; or silence"), remedy


def test_a_pyc_alone_in_a_zip_on_windows_is_told_the_source_files_not_numba_0_68_1(tmp_path, monkeypatch):
    """On Windows, a ``KeyError`` for a member shipped as ``.pyc`` alone gets the remedy that asks for the source files,
    although numba 0.68.0 named the member with backslashes. The archive holds no source under the name that numba
    0.68.1 reads either, so a release with the fix would not cache the module."""
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setattr(os, "name", "nt")
    archive = tmp_path / "numbox.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("numbox/core/bindings/libm.pyc", b"")
    py_file = str(archive / "numbox" / "core" / "bindings" / "libm.py")
    failure = KeyError("There is no item named 'numbox\\\\core\\\\bindings\\\\libm.py' in the archive")
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    assert remedy == (
        "numba 0.68 and later read a .zip member's source from the archive, and this one holds none under "
        "numbox/core/bindings/libm.py, the module shipped as .pyc alone: to cache, install numbox with its source "
        "files on disk, unpacked from any archive, or import it from a .zip holding its source files, which numba "
        "caches in the user's cache directory; or silence"), remedy


def test_a_directory_named_zip_above_the_archive_is_told_as_the_directory_from_numba_0_68(tmp_path, monkeypatch):
    """A member of an archive under a directory named ``container.zip`` gets a remedy for the ``IsADirectoryError`` that
    names that directory and asks for the archive to be put under a path without it. numba takes the first part of the
    path that ends in ``.zip`` for the archive, and numba 0.68 and later open that part to read the member's source."""
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    container = tmp_path / "container.zip"
    container.mkdir()
    py_file = str(container / "numbox.zip" / "numbox" / "core" / "configurations.py")
    failure = IsADirectoryError(errno.EISDIR, "Is a directory", str(container))
    assert configurations.is_a_cache_error(failure)
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    assert remedy == (
        "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: numba takes the first part of "
        f"the path ending in .zip, {container}, for the archive, and it is a directory, which numba 0.68 and later "
        "open to read the member's source: put the archive under a path with no such directory above it, or silence"
    ), remedy


# SUBCLASSED_LOCATORS is a module of two locator classes that subclass numba's user-provided and .zip locators, and the
# test lists them by their dotted paths.
SUBCLASSED_LOCATORS = (
    "from numba.core.caching import UserProvidedCacheLocator, ZipCacheLocator\n"
    "class OurUserProvided(UserProvidedCacheLocator):\n    pass\n"
    "class OurZip(ZipCacheLocator):\n    pass\n"
)

# Each case gives the value of NUMBA_CACHE_LOCATOR_CLASSES, whether NUMBA_CACHE_DIR is set, the source's directory,
# the location that the error names, and the remedy up to its last clause. The location "none" stands for numba's
# error that no locator accepts the file.
SUBCLASS_CASES = {
    "the user-provided subclass took NUMBA_CACHE_DIR's location": (
        "our_locators.OurUserProvided,InTreeCacheLocator,UserWideCacheLocator", True, "site", "NUMBA_CACHE_DIR",
        "the path is too long for the file system: a shorter NUMBA_CACHE_DIR"),
    "the .zip subclass took the user's cache directory's location": (
        "our_locators.OurZip,UserProvidedCacheLocator,InTreeCacheLocator,UserWideCacheLocator", True, "bundle.zip",
        "user", "the path is too long for the file system: the user's cache directory, {user_cache_dir}, at a "
        "shorter path{moved_through}"),
    "the user-provided subclass before the in-tree locator that took the location": (
        "our_locators.OurUserProvided,InTreeCacheLocator", True, "site", "beside",
        "the path is too long for the file system: numbox installed at a shorter path, or NUMBA_CACHE_DIR, which "
        "is set to {cache_dir} and numba could not use, made a writable directory at a short path"),
    "no locator, the user-provided subclass passed over": (
        "our_locators.OurUserProvided,InTreeCacheLocator", True, "site", "none",
        "NUMBA_CACHE_DIR is set to {cache_dir}, which numba could not use: set it to a writable directory at a short "
        "path"),
}


@pytest.mark.parametrize("case", list(SUBCLASS_CASES))
def test_a_subclass_of_one_of_numbas_locators_is_told_as_its_parent_is(tmp_path, monkeypatch, case):
    """A subclass of one of numba's locators, listed by its dotted path, gets the remedy that its parent gets, and a
    subclass of the user-provided locator counts as reading ``NUMBA_CACHE_DIR``. A subclass accepts what its parent
    accepts and caches where its parent does."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    (tmp_path / "our_locators.py").write_text(SUBCLASSED_LOCATORS)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "our_locators", raising=False)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    locators, cache_dir_set, source_dir, which, expected = SUBCLASS_CASES[case]
    cache_dir = tmp_path / "cache"
    user_cache_dir = tmp_path / "user-cache" / "numba"
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache_dir) if cache_dir_set else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = tmp_path / source_dir / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    subpath = _CacheLocator.get_suitable_cache_subpath(str(py_file))
    if which == "none":
        failure = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
        ending = ", or silence"
    else:
        location = {
            "NUMBA_CACHE_DIR": cache_dir / subpath, "beside": py_file.parent / "__pycache__", "user": user_cache_dir / subpath,
        }[which]
        failure = OSError(errno.ENAMETOOLONG, "File name too long", str(location))
        ending = "; or silence"
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    expected = expected.format(cache_dir=cache_dir, user_cache_dir=user_cache_dir,
                               moved_through=configurations._moved_through())
    assert remedy == f"{expected}{ending}", remedy


@pytest.mark.parametrize("given", ["Path", "bytes"])
@pytest.mark.parametrize("placement", ["no locator", "too long", "archive"])
def test_a_source_given_as_a_path_object_or_bytes_gets_the_remedy_its_string_gets(
        tmp_path, monkeypatch, placement, given):
    """``cache_remedy`` gives a source passed as a path object or as bytes the remedy that the source's string gets,
    for numba's error that no locator accepts the source, for a path too long and for a moved archive.
    ``check_cache_location`` accepts any path-like, and the remedy reads the source as a string, as the check does."""
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    if placement == "archive":
        py_file = tmp_path / "bundle.zip" / "package" / "module.py"
        failure = FileNotFoundError(errno.ENOENT, "No such file or directory", str(tmp_path / "bundle.zip"))
    else:
        py_file = tmp_path / "site" / "package" / "module.py"
        py_file.parent.mkdir(parents=True)
        py_file.write_text("")
        if placement == "too long":
            failure = OSError(errno.ENAMETOOLONG, "File name too long", str(py_file.parent / "__pycache__"))
        else:
            failure = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
    as_given = py_file if given == "Path" else os.fsencode(py_file)
    as_a_string = configurations.cache_remedy(str(py_file), failure, "silence")
    assert configurations.cache_remedy(as_given, failure, "silence") == as_a_string


@pytest.mark.parametrize("given", ["Path", "bytes"])
@pytest.mark.parametrize("placement", ["too long", "unwritable", "a .zip's location", "a moved archive"])
def test_an_error_naming_its_file_as_a_path_object_or_bytes_gets_the_remedy_a_string_name_gets(
        tmp_path, monkeypatch, placement, given):
    """An error that names its file as a path object or as bytes gets the remedy that the same error with a string name
    gets, for a source on disk and for a ``.zip`` member. A call that receives bytes raises an error that names its file
    as bytes, and a caller can build an error with a path object, so the remedy reads the name as a string."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    refused = (errno.EACCES, "Permission denied")
    if placement in ("a .zip's location", "a moved archive"):
        py_file = tmp_path / "bundle.zip" / "package" / "module.py"
        if placement == "a moved archive":
            error, named = (errno.ENOENT, "No such file or directory"), tmp_path / "bundle.zip"
        else:
            subpath = _CacheLocator.get_suitable_cache_subpath(str(py_file))
            error, named = refused, tmp_path / "user-cache" / "numba" / subpath
    else:
        py_file = tmp_path / "site" / "package" / "module.py"
        py_file.parent.mkdir(parents=True)
        py_file.write_text("")
        error = (errno.ENAMETOOLONG, "File name too long") if placement == "too long" else refused
        named = py_file.parent / "__pycache__"
    as_given = named if given == "Path" else os.fsencode(named)
    as_a_string = configurations.cache_remedy(str(py_file), OSError(*error, str(named)), "silence")
    assert configurations.cache_remedy(str(py_file), OSError(*error, as_given), "silence") == as_a_string


@pytest.mark.parametrize("placement", ["too long", "archive"])
def test_the_remedies_that_name_a_package_name_the_one_given(tmp_path, monkeypatch, placement):
    """The two remedies that ask for a package to be installed again, at a shorter path or with its source files on
    disk, name the package passed as ``package``, and numbox by default. A package built on numbox runs the check on
    its own files and passes its own name."""
    import numba
    from numbox.core.configurations import cache_remedy
    if placement == "too long":
        monkeypatch.setattr(numba.config, "CACHE_DIR", "")
        py_file = tmp_path / "ducklib.py"
        py_file.write_text("")
        failure = OSError(errno.ENAMETOOLONG, "File name too long", str(tmp_path / "__pycache__"))
        named = "{} installed at a shorter path"
    else:
        py_file = tmp_path / "numbduck-0.0.0-py3-none-any.whl" / "numbduck" / "ducklib.py"
        failure = RuntimeError(f"cannot cache function 'f': no locator available for file '{py_file}'")
        named = "install {} with its source files on disk"
    remedy = cache_remedy(str(py_file), failure, "silence", package="numbduck")
    assert named.format("numbduck") in remedy and "numbox" not in remedy, remedy
    assert named.format("numbox") in cache_remedy(str(py_file), failure, "silence")


def _function_names(tree):
    """Yield the qualified name and the line of each function in ``tree``.

    Each name has the form that numba uses in cache file names. numba removes the angle brackets of ``<locals>``, so the
    name of a nested function is ``outer.locals.inner``. The line is that of the ``def``. numba writes the line of the
    first decorator, which is at or before the ``def``, so the line given here has at least as many digits.
    """
    def walk(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield ".".join(scope + [child.name]), child.lineno
                yield from walk(child, scope + [child.name, "locals"])
            elif isinstance(child, ast.ClassDef):
                yield from walk(child, scope + [child.name])
            else:
                yield from walk(child, scope)
    return walk(tree, [])


def test_numbas_longest_file_name_for_the_package_is_within_the_probes_bound():
    """The longest cache file name that numba can write for a function of numbox, or for the graph maker that
    ``make_graph`` compiles as code of ``builder.py``, fits within ``LONGEST_CACHE_FILE_NAME``. The check creates a file
    with a name of that length in each location, so a longer name could still fail at numba's first save."""
    # Each name holds the module's stem, the function's qualified name and line, and the interpreter tag. The tag
    # carries t, the abiflags of a free-threaded build. Each name also holds the data file's number at the width that
    # numbox reserves for it, and numba's temporary suffix of 21 bytes. The test counts every function, decorated or
    # not. The graph maker's name holds a SHA-256 digest of 64 hexadecimal digits, and its line is given four digits.
    from numbox.core.configurations import LONGEST_CACHE_FILE_NAME, _OVERLOAD_DIGITS
    tag = f".py{sys.version_info[0]}{sys.version_info[1]}t.{'9' * _OVERLOAD_DIGITS}.nbc.tmp.0123456789abcdef"
    names = [f"{path.stem}.{qualname}-{lineno}{tag}"
             for path in sorted((REPO / "numbox").rglob("*.py"))
             for qualname, lineno in _function_names(ast.parse(path.read_text(encoding="utf-8")))]
    names.append(f"builder._make_{'0' * 64}-9999{tag}")
    longest = max(names, key=len)
    assert len(names) > 500
    assert len(longest) <= LONGEST_CACHE_FILE_NAME, longest


# ANOTHER_ERROR_AT_THE_CACHE_SET_UP is a child that makes numba's cache set-up raise a RuntimeError of another kind,
# and then imports numbox's configurations module.
ANOTHER_ERROR_AT_THE_CACHE_SET_UP = (
    "import numba.core.caching as caching\n"
    "def refuse(self, py_func):\n"
    "    raise RuntimeError('a locator of another kind refused')\n"
    "caching.CompileResultCacheImpl.__init__ = refuse\n"
    "import numbox.core.configurations\n"
)


def test_an_error_that_is_not_the_caches_is_raised_as_it_was(tmp_path):
    """A ``RuntimeError`` other than numba's error that no locator accepts the file, raised while numba sets up the
    cache, fails the import of numbox unchanged and without a warning. ``is_a_cache_error`` admits only numba's cache
    errors, so the check does not turn another fault into an uncached import with a remedy that does not apply."""
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", ANOTHER_ERROR_AT_THE_CACHE_SET_UP],
                         capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert run.returncode != 0 and "a locator of another kind refused" in run.stderr, run.stderr
    assert "compiles without a cache" not in run.stderr


# Each child runs code that numbox generates at run time. The five are a struct, an SQLite aggregate, a kernel for a
# graph of formulas, a work graph with a derive function, and a table-valued function. numbox writes each piece of
# generated code to an anchor, a file under NUMBA_CACHE_DIR or the user's cache directory. It compiles the code under
# that file's name. Each child prints a word once its code has run. The two SQLite children import their helpers from
# the test modules of those bindings.
MAKE_A_STRUCTREF = (
    "from numba.core.types import StructRef, float32\n"
    "from numba.experimental.structref import register\n"
    "from numbox.utils.highlevel import make_structref\n"
    "@register\n"
    "class TypeClass(StructRef):\n"
    "    pass\n"
    "Struct = make_structref('Struct', {'value': float32}, TypeClass)\n"
    "assert Struct(2.5).value == 2.5\n"
    "print('made')\n"
)

REGISTER_AN_AGGREGATE = (
    "from test.core.test_sqlite_udf_helpers import (_open_memory, _make_table, _read1_int64, sum_state_type,\n"
    "                                               sum_init, sum_step, sum_finalize)\n"
    "from numbox.core.bindings.sqlite.udf_helpers import register_aggregate\n"
    "from numbox.core.bindings.sqlite.conn import sqlite3_close\n"
    "db = _open_memory()\n"
    "_make_table(db, [1, 2, 3, 4, 5])\n"
    "register_aggregate(db, 'my_sum', 1, sum_state_type, sum_init, sum_step, sum_finalize)\n"
    "value, _ = _read1_int64(db, 'SELECT __cap(my_sum(v)) FROM t')\n"
    "sqlite3_close(db)\n"
    "assert value == 15, value\n"
    "print('summed')\n"
)

COMPILE_A_KERNEL = (
    "from numbox.core.variable.variable import Graph\n"
    "from numbox.core.variable.compile_kernel import compile_kernel\n"
    "def f(x):\n"
    "    return x + 1.0\n"
    "graph = Graph({'calc': [{'name': 'y', 'inputs': {'x': 'ext'}, 'formula': f}]}, ['ext'])\n"
    "kernel = compile_kernel(graph, 'calc.y')\n"
    "assert kernel.execute({'ext': {'x': 1.0}})['calc.y'] == 2.0\n"
    "print('executed')\n"
)

BUILD_A_DERIVE = (
    "from numpy import isclose\n"
    "from numbox.core.work.builder import Derived, End, make_graph\n"
    "x = End(name='x', init_value=3.14)\n"
    "def twice(x):\n"
    "    return 2 * x\n"
    "y = Derived(name='y', init_value=0.0, derive=twice, sources=(x,))\n"
    "access = make_graph(y)\n"
    "access.y.calculate()\n"
    "assert isclose(access.y.data, 6.28), access.y.data\n"
    "print('derived')\n"
)

REGISTER_A_TVF = (
    "import numpy as np\n"
    "from test.core.test_sqlite_tvf import _open, _select_int, _series, _OUT\n"
    "from numbox.core.bindings.sqlite.tvf import register_tvf\n"
    "from numbox.core.bindings.sqlite.conn import sqlite3_close\n"
    "db = _open()\n"
    "handle = register_tvf(db.value, 'series', (np.int64, np.int64), _OUT, _series)\n"
    "rc, rows = _select_int(db, 'SELECT n FROM series(2, 5)')\n"
    "assert rc == 0 and [row[0] for row in rows] == [2, 3, 4], (rc, rows)\n"
    "sqlite3_close(db.value)\n"
    "print('selected')\n"
)


@needs_a_directory_it_cannot_write
@pytest.mark.parametrize("child, word", [
    (MAKE_A_STRUCTREF, "made"), (REGISTER_AN_AGGREGATE, "summed"), (REGISTER_A_TVF, "selected"),
], ids=["make_structref", "register_aggregate", "register_tvf"])
def test_generated_code_compiles_uncached_where_its_anchor_cannot_be_written(tmp_path, child, word):
    """Generated code whose anchor cannot be written, under a read-only home with ``NUMBA_CACHE_DIR`` unset, compiles
    without a cache after a warning that offers ``NUMBA_CACHE_DIR``. numbox's own functions cache beside their sources
    here, and with the variable set to a writable directory, the anchor is written there and no warning appears."""
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o555)
    try:
        env = dict(os.environ, PYTHONPATH=str(REPO), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"))
        env.pop("NUMBA_CACHE_DIR", None)
        env.pop("NUMBOX_JIT_OPTIONS", None)
        run = subprocess.run([sys.executable, "-W", "always", "-c", child], capture_output=True, text=True,
                             env=env, cwd=str(tmp_path))
        assert run.returncode == 0 and word in run.stdout, run.stderr
        assert "compiles without a cache" in run.stderr and "Set NUMBA_CACHE_DIR" in run.stderr, run.stderr
        cured = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", child], capture_output=True,
                               text=True, env=dict(env, NUMBA_CACHE_DIR=str(tmp_path / "cache")), cwd=str(tmp_path))
        assert cured.returncode == 0 and word in cured.stdout, cured.stderr
        assert list((tmp_path / "cache").rglob("*.py")), "no anchor under NUMBA_CACHE_DIR"
    finally:
        home.chmod(0o755)


@needs_a_directory_it_cannot_write
@pytest.mark.parametrize("child, word, warns", [
    (MAKE_A_STRUCTREF, "made", True), (REGISTER_AN_AGGREGATE, "summed", True), (REGISTER_A_TVF, "selected", True),
    (COMPILE_A_KERNEL, "executed", True), (BUILD_A_DERIVE, "derived", False),
], ids=["make_structref", "register_aggregate", "register_tvf", "compile_kernel", "derive"])
def test_a_warm_anchor_in_a_directory_that_stopped_being_writable(tmp_path, child, word, warns):
    """Generated code whose anchor exists, in a ``NUMBA_CACHE_DIR`` that can no longer be written, is cached in the
    user's cache directory, because the check asks numba where it would cache. Once the user's cache directory is
    read-only too, the code compiles without a cache after one warning, and a derive function compiles without a
    cache and without a warning."""
    # A check of the anchor's own directory alone would disable caching in the second run, although numba has a
    # location in the user's cache directory. In the third run the anchor exists, so numbox does not write it again,
    # and only the check finds that numba has no location. The work builder compiles a derive function without a
    # cache whenever numba has no location for its anchor, and it gives no warning.
    home = tmp_path / "home"
    home.mkdir()
    cache = tmp_path / "cache"
    env = dict(os.environ, PYTHONPATH=str(REPO), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(cache))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    warm = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", child], capture_output=True,
                          text=True, env=env, cwd=str(tmp_path))
    assert warm.returncode == 0 and word in warm.stdout, warm.stderr
    anchors = list(cache.rglob("*.py"))
    assert anchors, "no anchor under NUMBA_CACHE_DIR"
    read_only = [cache, *(path for path in cache.rglob("*") if path.is_dir())]
    for path in read_only:
        path.chmod(0o555)
    try:
        in_user_cache = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", child],
                                       capture_output=True, text=True, env=env, cwd=str(tmp_path))
        assert in_user_cache.returncode == 0 and word in in_user_cache.stdout, in_user_cache.stderr
        assert _index_files(home), "numba did not cache the generated code in the user's cache directory"
        home_tree = [home, *(path for path in home.rglob("*") if path.is_dir())]
        read_only.extend(home_tree)
        for path in home_tree:
            path.chmod(0o555)
        uncached = subprocess.run([sys.executable, "-W", "always", "-c", child], capture_output=True, text=True,
                                  env=env, cwd=str(tmp_path))
        assert uncached.returncode == 0 and word in uncached.stdout, uncached.stderr
        if warns:
            assert "compiles without a cache" in uncached.stderr and "Set NUMBA_CACHE_DIR" in uncached.stderr
        else:
            assert "compiles without a cache" not in uncached.stderr, uncached.stderr
    finally:
        for path in read_only:
            path.chmod(0o755)


@pytest.mark.parametrize("child, word, warns", [
    (MAKE_A_STRUCTREF, "made", True), (REGISTER_AN_AGGREGATE, "summed", True), (REGISTER_A_TVF, "selected", True),
    (COMPILE_A_KERNEL, "executed", True), (BUILD_A_DERIVE, "derived", False),
], ids=["make_structref", "register_aggregate", "register_tvf", "compile_kernel", "derive"])
@pytest.mark.parametrize("placement", ["a file", "a component too long"])
def test_generated_code_compiles_uncached_under_a_numba_cache_dir_numba_cannot_use(
        tmp_path, placement, child, word, warns):
    """Generated code compiles without a cache after a warning whose remedy matches the error, if ``NUMBA_CACHE_DIR``
    names a file or has a component too long for the file system. numba skips such a location, so numbox's own functions
    cache beside their sources, while the anchor's directory cannot be created."""
    # numbox counts any OSError from writing the anchor as an error of the location. The derive function compiles
    # without a cache and without a warning.
    if placement == "a file":
        cache_dir = tmp_path / "file"
        cache_dir.write_text("not a directory\n")
        remedy = "Set NUMBA_CACHE_DIR to a writable directory"
    else:
        cache_dir = tmp_path / ("c" * 300)
        # Windows refuses the long component as an invalid name, with WinError 123 and EINVAL, and the error gives no
        # length, so the warning gives the general remedy there.
        remedy = ("Set NUMBA_CACHE_DIR to a writable directory" if os.name == "nt"
                  else "too long for the file system: NUMBA_CACHE_DIR at a shorter path")
    env = dict(os.environ, PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(cache_dir))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", child], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path))
    assert run.returncode == 0 and word in run.stdout, run.stderr
    if warns:
        assert "compiles without a cache" in run.stderr and remedy in run.stderr, run.stderr
    else:
        assert "compiles without a cache" not in run.stderr, run.stderr


def test_make_graph_under_a_callers_cache_option_falls_back_from_an_archive(tmp_path):
    """``make_graph`` from an ``.egg``, with a caller's ``{"cache": True}``, warns once for the graph maker, which it
    compiles as code of ``builder.py``, and the derive function still caches under its own anchor. numba receives a
    caller's options without the check that numbox's options pass at import, so ``make_graph`` runs that check for
    ``builder.py``."""
    archive = _archive(tmp_path / "numbox-0.0.0-py3.12.egg")
    child = BUILD_A_DERIVE.replace("make_graph(y)", "make_graph(y, jit_options={'cache': True})")
    assert child != BUILD_A_DERIVE
    env = dict(os.environ, PYTHONPATH=str(archive), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", child], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path))
    assert run.returncode == 0 and "derived" in run.stdout, run.stderr
    assert "the code generated at builder.py" in run.stderr and "source is not a file on disk" in run.stderr
    assert '"cache" off in the jit options this code was given' in run.stderr, run.stderr
    # The derive function caches under an anchor of its own, under NUMBA_CACHE_DIR, which the check of builder.py does
    # not affect.
    assert _index_files(tmp_path / "cache"), "the derive did not cache under NUMBA_CACHE_DIR"


# A_STRUCTREF_WITH_A_TYPING_ERROR is a child whose struct has a method that adds a string to a float32, so that numba
# raises a TypingError at the call.
A_STRUCTREF_WITH_A_TYPING_ERROR = MAKE_A_STRUCTREF.replace(
    "Struct = make_structref('Struct', {'value': float32}, TypeClass)\n",
    "def bad(self):\n"
    "    return self.value + 'x'\n"
    "Struct = make_structref('Struct', {'value': float32}, TypeClass, struct_methods={'bad': bad})\n"
    "Struct(2.5).bad()\n",
)


def test_a_typing_error_in_generated_code_quotes_the_source_with_caching_off(tmp_path):
    """A typing error in generated code quotes the offending line with caching disabled too, because numbox writes the
    anchor, from which numba reads the line, whether or not caching is enabled. numba writes no cache file then."""
    assert A_STRUCTREF_WITH_A_TYPING_ERROR != MAKE_A_STRUCTREF
    script = tmp_path / "probe.py"
    script.write_text(A_STRUCTREF_WITH_A_TYPING_ERROR)
    env = dict(os.environ, PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(tmp_path / "cache"),
               NUMBOX_JIT_OPTIONS='{"cache": false}')
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert run.returncode != 0 and "TypingError" in run.stderr, run.stderr
    assert "source missing" not in run.stderr and "return self.value + 'x'" in run.stderr, run.stderr
    assert list((tmp_path / "cache").rglob("*.py")) and not _index_files(tmp_path / "cache")


@pytest.mark.parametrize("child, word", [
    (MAKE_A_STRUCTREF.replace("TypeClass)\n", "TypeClass, jit_options={'cache': True})\n"), "made"),
    (COMPILE_A_KERNEL.replace("'calc.y')", "'calc.y', jit_options={'cache': False}, cache=True)"), "executed"),
], ids=["make_structref with jit_options", "compile_kernel with cache"])
def test_the_anchor_warning_names_the_options_the_caller_gave(tmp_path, child, word):
    """Generated code whose caller enables caching, under ``NUMBOX_JIT_OPTIONS='{"cache": false}'``, warns with a remedy
    that names the caller's jit options and the ``cache`` argument as ways to disable the cache. A caller's ``"cache"``
    option takes precedence over the variable, and the ``cache`` argument of ``compile_kernel`` takes precedence over
    both."""
    assert child not in (MAKE_A_STRUCTREF, COMPILE_A_KERNEL)
    cache_dir = tmp_path / "file"
    cache_dir.write_text("not a directory\n")
    env = dict(os.environ, PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(cache_dir),
               NUMBOX_JIT_OPTIONS='{"cache": false}')
    run = subprocess.run([sys.executable, "-W", "always", "-c", child], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path))
    assert run.returncode == 0 and word in run.stdout, run.stderr
    assert "compiles without a cache" in run.stderr, run.stderr
    assert '"cache" off in the jit options this code was given, or in its cache argument where it takes one' in run.stderr


# The type class is defined in a module of its own, as the documentation of make_structref requires, so that a second
# process can load the struct's cache entries.
A_TYPE_CLASS = (
    "from numba.core.types import StructRef\n"
    "from numba.experimental.structref import register\n"
    "@register\n"
    "class TypeClass(StructRef):\n"
    "    pass\n"
)

MAKE_A_LONG_NAMED_STRUCTREF = (
    "from numba.core.types import float32\n"
    "from numbox.utils.highlevel import make_structref\n"
    "from numbox.utils.preprocessing import bounded_stem\n"
    "from long_named_type_class import TypeClass\n"
    "def dddddddddddddddddddddddddddddddddddddddd(self):\n"
    "    return self.value * 2\n"
    "def " + "m" * 200 + "(self):\n"
    "    return self.value * 3\n"
    "name = NAME\n"
    "field = 'f' + name[1:]\n"
    "fields = {'value': float32}\n"
    "if bounded_stem(field) != field:\n"
    "    fields[bounded_stem(field)] = float32\n"
    "fields[field] = float32\n"
    "methods = {'d' * 40: dddddddddddddddddddddddddddddddddddddddd, 'm' * 200: " + "m" * 200 + "}\n"
    "Struct = make_structref(name, fields, TypeClass, struct_methods=methods)\n"
    "values = [1.5 * (index + 1) for index in range(len(fields))]\n"
    "struct = Struct(*values)\n"
    "assert [getattr(struct, each) for each in fields] == values and getattr(struct, 'd' * 40)() == 3.0\n"
    "assert getattr(struct, 'm' * 200)() == 4.5\n"
    "assert Struct.__name__ == name and Struct.__qualname__ == name and repr(struct).startswith(name + '(')\n"
    "print('made', len(name))\n"
)


@pytest.mark.parametrize("name", ["S" * 40, "S" * 41, "S" * 150, "S" * 300, "é" * 40, "結" * 100],
                         ids=["40 ascii", "41 ascii", "150 ascii", "300 ascii", "40 accented", "100 cjk"])
def test_a_struct_name_of_any_length_caches(tmp_path, name):
    """A struct whose name has any length, in ASCII, accented or CJK characters, caches without a warning, and a second
    process loads its entries without adding any. numbox bounds each name that it puts into a file name to 40 bytes of
    UTF-8, so every cache file name stays under 230 bytes, and the struct's class still has its full name."""
    # numba names a cache file after the anchor's stem and the jitted function's qualified name, and both hold the
    # struct's name. A name of 40 accented characters is 80 bytes, so numbox bounds it, and it truncates a name of 100
    # CJK characters at a whole character. The struct also has a field whose name is as long as the struct's name, and a
    # field named after that field's bounded name, defined first. numbox bounds the long field's getter as it bounds the
    # struct's name, and the short field retains its own property. One method's name is 40 bytes, the longest that a
    # bounded name can be, so the files of its jitted helper are the longest that numba writes for any struct. The other
    # method's name, of 200 bytes, must be bounded in the names of its helper and its overload. The bound of 230 bytes
    # allows for the 21 bytes of numba's temporary name.
    (tmp_path / "long_named_type_class.py").write_text(A_TYPE_CLASS)
    script = tmp_path / "make.py"
    script.write_text(MAKE_A_LONG_NAMED_STRUCTREF.replace("NAME", repr(name)), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", str(script)],
                         capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and f"made {len(name)}" in run.stdout, run.stderr
    cache_files = [path.name for path in (tmp_path / "cache").rglob("*.nb*")]
    assert cache_files and all(len(each.encode()) < 230 for each in cache_files), cache_files
    indexes = _index_files(tmp_path / "cache")
    again = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", str(script)],
                           capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert again.returncode == 0, again.stderr
    assert _index_files(tmp_path / "cache") == indexes


@pytest.mark.skipif(sys.platform != "linux", reason="a path of 4096 bytes and a name of 255 are Linux's limits")
@pytest.mark.parametrize("directory, remedy", [
    ("HOME", "too long for the file system: NUMBA_CACHE_DIR at a shorter path"),
    ("NUMBA_CACHE_DIR", "the path is too long for the file system: a shorter NUMBA_CACHE_DIR; or NUMBOX_JIT_OPTIONS"),
])
def test_an_anchor_path_too_long_for_the_file_system_compiles_uncached_and_the_warning_says_so(
        tmp_path, directory, remedy):
    """A cache directory too close to Linux's path limit makes generated code compile without a cache after one warning
    whose remedy says that the path is too long, under a deep ``HOME`` and under a deep ``NUMBA_CACHE_DIR``. Under the
    deep variable, numbox's own check fails first, and the struct then compiles under numbox's options without a second
    warning."""
    # Under the deep HOME, the anchor's directory fits the path limit and the anchor's own name does not. The anchor's
    # remedy then asks for NUMBA_CACHE_DIR at a shorter path. The deep NUMBA_CACHE_DIR leaves numbox's location for
    # configurations.py 20 bytes short of the limit. numba's own test passes there, also with a named temporary file on
    # a file system that does not support unnamed temporary files, and numbox's check fails, so numbox warns first. At
    # the depth of the HOME case, numba's own test would fail on such a file system, numba would cache numbox in the
    # __pycache__ beside its sources, and the anchor would warn instead.
    if directory == "NUMBA_CACHE_DIR":
        from numba.core.caching import _CacheLocator
        location = _CacheLocator.get_suitable_cache_subpath(str(REPO / "numbox" / "core" / "configurations.py"))
        deep = _directory_of_length(tmp_path, 4096 - 20 - 1 - len(location))
    else:
        deep = _directory_of_length(tmp_path, 4096 - 50)
    env = dict(os.environ, PYTHONPATH=str(REPO))
    for name in ("NUMBOX_JIT_OPTIONS", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"):
        env.pop(name, None)
    env[directory] = str(deep)
    run = subprocess.run([sys.executable, "-W", "always", "-c", MAKE_A_STRUCTREF], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and "made" in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert remedy in run.stderr, run.stderr


# A_CACHE_FILE_NAME_TOO_LONG lists the cases of the test below. A case's name gives the source's place, the settings,
# and whether some file system in use allows the name.
A_CACHE_FILE_NAME_TOO_LONG = (
    "on disk, numba's order",
    "on disk, numba's order, an invalid name",
    "on disk, NUMBA_CACHE_DIR set",
    "on disk, the in-tree locator alone",
    "a .zip member in an ipykernel directory",
    "a .zip member, numba's order",
    "on disk, numba's order, a name a file system in use takes",
    "on disk, NUMBA_CACHE_DIR set, a name a file system in use takes",
    "on disk, the in-tree locator alone, a name a file system in use takes",
    "a .zip member in an ipykernel directory, a name a file system in use takes",
    "a .zip member, numba's order, a name a file system in use takes",
)


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=case) for case in A_CACHE_FILE_NAME_TOO_LONG])
def test_a_cache_file_whose_name_is_too_long_for_the_file_system_is_told_the_name_which_no_shorter_path_cures(
        tmp_path, monkeypatch, case):
    """A cache file name too long for the file system gets a remedy that names the file and its length. No shorter path
    avoids the error, so the remedy asks for a shorter name for the function or its file. If some file system in use
    allows the name, the remedy for a source on disk also offers ``NUMBA_CACHE_DIR`` on such a file system."""
    # numba names the files that it caches a function in after the source's name and the function's qualified name. Its
    # first read of the index fails on a name that is too long. The remedy counts the name as the file system counts it.
    # It offers the variable on the terms on which it offers the variable for a path too long. It names the variable as
    # set if the variable's own locator accepted the file.
    import types
    import numba
    from numba.core import caching
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(caching, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    cache_dir = str(tmp_path / "cache") if "NUMBA_CACHE_DIR set" in case else ""
    if cache_dir:
        os.mkdir(cache_dir)
    monkeypatch.setattr(numba.config, "CACHE_DIR", cache_dir)
    listed = "InTreeCacheLocator" if "the in-tree locator alone" in case else ""
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", listed, raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    if case.startswith("on disk"):
        py_file = str(tmp_path / "site" / "module.py")
        os.makedirs(os.path.dirname(py_file))
        with open(py_file, "w", encoding="utf-8") as handle:
            handle.write("x = 1\n")
    elif "ipykernel" in case:
        py_file = str(tmp_path / "bundle.zip" / "ipykernel_123" / "cell.py")
    else:
        py_file = str(tmp_path / "bundle.zip" / "package" / "module.py")
    subpath = _CacheLocator.get_suitable_cache_subpath(py_file)
    if "ipykernel" in case:
        location = ipython_dir / "numba_cache"
    elif case.startswith("on disk") and not cache_dir:
        location = Path(py_file).parent / "__pycache__"
    else:
        location = Path(cache_dir or str(tmp_path / "user-cache" / "numba")) / subpath
    if "a file system in use takes" in case:
        # A limit of 60 fits numba's directory name for the source, the directory's name and 41 bytes more, but not the
        # cache file's name.
        monkeypatch.setattr(configurations, "_name_limit", lambda location: 60)
    limit = configurations._name_limit(str(location))
    name = f"{Path(py_file).stem}.{'f' * (limit + 5)}-1.py{sys.version_info[0]}{sys.version_info[1]}.nbi"
    counted_in = "characters" if os.name == "nt" or sys.platform == "darwin" else "bytes"
    length = len(name.encode("utf-16-le")) // 2 if counted_in == "characters" else len(os.fsencode(name))
    assert length > limit
    assert (length <= 255) == ("a file system in use takes" in case)
    # Windows refuses a name too long as an invalid one, with EINVAL, and the remedy measures the name whatever the
    # errno.
    refused = ((errno.EINVAL, "Invalid argument") if "invalid name" in case
               else (errno.ENAMETOOLONG, "File name too long"))
    failure = OSError(*refused, str(location / name))
    assert configurations.is_a_cache_error(failure)
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    clause = (
        f"numba names the cache files for a function after the source's name and the function's qualified name, "
        f"{name}, {length} {counted_in}, which is longer than a name can be on that file system ({limit})")
    fits = "directory on a file system that takes a name that long"
    cure = f"{clause}, and no shorter path cures it: a shorter name for the function or its file, which is numbox's to give"
    # 255 is the limit on names of ext4, NTFS and APFS, so the remedy offers another file system only for a name of 255
    # or fewer.
    if length > 255:
        offer = ""
    elif case.startswith("on disk, numba's order"):
        offer = f", or NUMBA_CACHE_DIR set to a {fits}"
    elif cache_dir:
        offer = f", or NUMBA_CACHE_DIR, which is set to {cache_dir}, pointed at a {fits}"
    else:
        offer = ""
    if case.startswith("on disk") or "ipykernel" in case:
        assert remedy == f"{cure}{offer}; or silence", remedy
    else:
        assert remedy == (
            "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
            f"effect here, because the source is not a file on disk: {cure}, or silence"), remedy


def test_a_function_whose_cache_file_name_is_too_long_for_the_file_system_is_told_the_name_by_numbas_own_error(
        tmp_path, monkeypatch):
    """numba's own error for a cache file name too long, which numba raises at the function's first call when it opens
    the index, gets the remedy that names the file and asks for a shorter name. The function's name is five characters
    longer than the file system's limit, so no path is short enough for its cache files."""
    # Windows refuses such a name as an invalid one, with EINVAL, and the remedy reads the name whatever the errno.
    import importlib
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    limit = configurations._name_limit(str(tmp_path))
    function = "f" * (limit + 5)
    source = tmp_path / "module_named_long.py"
    source.write_text(
        f"from numba import njit\n\n\n@njit(cache=True)\ndef {function}(x):\n    return x + 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "module_named_long", raising=False)
    module = importlib.import_module("module_named_long")
    with pytest.raises(OSError) as raised:
        getattr(module, function)(1)
    failure = raised.value
    assert configurations.is_a_cache_error(failure)
    assert Path(failure.filename).parent == tmp_path / "__pycache__"
    name = Path(failure.filename).name
    assert name.startswith(f"module_named_long.{function}-")
    counted_in = "characters" if os.name == "nt" or sys.platform == "darwin" else "bytes"
    length = len(name.encode("utf-16-le")) // 2 if counted_in == "characters" else len(os.fsencode(name))
    assert length > limit
    offer = ", or NUMBA_CACHE_DIR set to a directory on a file system that takes a name that long" if length <= 255 else ""
    assert configurations.cache_remedy(str(source), failure, "silence") == (
        f"numba names the cache files for a function after the source's name and the function's qualified name, "
        f"{name}, {length} {counted_in}, which is longer than a name can be on that file system ({limit}), and no "
        f"shorter path cures it: a shorter name for the function or its file, which is numbox's to give{offer}; or "
        "silence")


def test_a_cell_file_on_disk_the_in_tree_locator_takes_is_not_told_ipythondir_for_an_error_naming_another_numba_cache(
        tmp_path, monkeypatch):
    """A caller's error that names an unrelated ``numba_cache``, for a cell file on disk that the in-tree locator
    accepts, gets the remedy for that location and not ``IPYTHONDIR``, also under an IPython that gives temporary
    directories. The in-tree locator accepts such a file if it can write its ``__pycache__``, and numba then never
    reaches the IPython locator."""
    import tempfile
    import types
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: tempfile.mkdtemp(dir=tmp_path)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    kernel = tmp_path / "ipykernel_123"
    kernel.mkdir()
    py_file = str(kernel / "cell.py")
    with open(py_file, "w", encoding="utf-8") as handle:
        handle.write("def cell():\n    return 1\n")
    configurations.check_cache_location(py_file, 0)
    assert (kernel / "__pycache__").is_dir()
    unrelated = str(tmp_path / "elsewhere" / "numba_cache")
    remedy = configurations.cache_remedy(py_file, OSError(errno.EACCES, "Permission denied", unrelated), "silence")
    assert remedy == (
        f"that location, {unrelated}, where no file can be written (Permission denied): make room there, or make it "
        "writable, or NUMBA_CACHE_DIR set to a writable directory; or silence"), remedy


@pytest.mark.skipif(os.name == "nt", reason="Windows decodes any byte of a path")
@pytest.mark.skipif(sys.platform == "darwin", reason="APFS takes no name that is not UTF-8")
@needs_a_directory_it_cannot_write
def test_a_source_under_a_directory_that_is_not_utf8_whose_pycache_is_unwritable_is_a_cache_error_told_a_utf8_path(
        tmp_path, monkeypatch):
    """The check raises numba's ``UnicodeEncodeError`` for a source under a directory whose path is not UTF-8, once the
    in-tree locator skips its unwritable ``__pycache__``, and ``is_a_cache_error`` counts that error. The remedy asks
    for a UTF-8 path or a writable ``__pycache__``. A ``UnicodeEncodeError`` for another reason is not a cache error."""
    # numba raises the error when its user-wide locator hashes the path to name its directory under the user's cache
    # directory. Because the error counts as a cache error, a package under such a directory imports without a cache
    # instead of failing.
    import numba
    from numba.core import caching
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(caching, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    directory = tmp_path / os.fsdecode(b"site\xff")
    directory.mkdir()
    py_file = str(directory / "module.py")
    with open(py_file, "w", encoding="utf-8") as handle:
        handle.write("x = 1\n")
    pycache = directory / "__pycache__"
    pycache.mkdir()
    pycache.chmod(0o555)
    try:
        with pytest.raises(UnicodeEncodeError) as raised:
            configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
        failure = raised.value
        assert failure.reason == "surrogates not allowed"
        assert configurations.is_a_cache_error(failure)
        remedy = configurations.cache_remedy(py_file, failure, "silence")
    finally:
        pycache.chmod(0o755)
    assert remedy == (
        "numba hashes the source's directory as UTF-8 for the directory it caches this file in under NUMBA_CACHE_DIR "
        f"or the user's cache directory, and {directory} is not: put the file under a UTF-8 path, or make the "
        f"__pycache__ beside the source, {pycache}, writable, where InTreeCacheLocator caches it hashing no path; or "
        "silence"), remedy
    assert not configurations.is_a_cache_error(UnicodeEncodeError("ascii", "\u00e9", 0, 1, "ordinal not in range(128)"))


@pytest.mark.skipif(os.name == "nt", reason="Windows decodes any byte of a path")
def test_an_error_naming_a_file_with_a_byte_that_is_not_utf8_is_answered_where_names_are_counted_in_utf16(
        tmp_path, monkeypatch):
    """``_measured`` counts a name with a byte that is not UTF-8 in UTF-16 units, as Windows and macOS count names, and
    the remedy for an error that names such a file returns normally. Python decodes such a byte to a lone surrogate, and
    Windows accepts one in a name. Python's UTF-16 codec raises on a lone surrogate unless it is given the
    ``surrogatepass`` error handler."""
    # The test sets sys.platform to macOS's value, so that the remedy counts the name in UTF-16 units.
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    site = tmp_path / "site"
    site.mkdir()
    py_file = str(site / "module.py")
    with open(py_file, "w", encoding="utf-8") as handle:
        handle.write("x = 1\n")
    named = str(site / "__pycache__" / os.fsdecode(b"module.f-1.py312.nbi\xff"))
    with pytest.raises(UnicodeEncodeError):
        os.path.basename(named).encode("utf-16-le")
    assert configurations._measured(os.path.basename(named)) == (21, "characters")
    remedy = configurations.cache_remedy(py_file, OSError(errno.EACCES, "Permission denied", named), "silence")
    assert remedy == (
        f"numba caches this file in {site / '__pycache__'}, where no file can be written (Permission denied): make "
        "room there, or make it writable, or NUMBA_CACHE_DIR set to a writable directory; or silence"), remedy


@pytest.mark.skipif(os.name == "nt", reason="Windows decodes any byte of a path")
@pytest.mark.skipif(sys.platform == "darwin", reason="APFS takes no name that is not UTF-8")
def test_a_zip_under_a_directory_that_is_not_utf8_is_a_cache_error_told_the_archive_under_a_utf8_path(
        tmp_path, monkeypatch):
    """A ``.zip`` under a directory whose path is not UTF-8 makes the check raise numba's ``UnicodeEncodeError``, a
    cache error, and the remedy asks for the archive to be put under a UTF-8 path. numba's ``.zip`` locator hashes the
    member's directory path, which contains the archive's path, to name its directory under the user's cache
    directory."""
    import numba
    from numba.core import caching
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(caching, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    directory = tmp_path / os.fsdecode(b"caf\xe9")
    directory.mkdir()
    archive = directory / "app.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("package/module.py", "x = 1\n")
    py_file = str(archive / "package" / "module.py")
    with pytest.raises(UnicodeEncodeError) as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    failure = raised.value
    assert configurations.is_a_cache_error(failure)
    assert configurations.cache_remedy(py_file, failure, "silence") == (
        "numba caches a .zip in the user's cache directory, and NUMBA_CACHE_DIR has no effect here, because the "
        "source is not a file on disk: numba hashes the source's directory as UTF-8 for the directory it caches this "
        f"file in there, and {archive / 'package'} is not: put the archive under a UTF-8 path, or silence")


@pytest.mark.skipif(os.name == "nt", reason="Windows decodes any byte of a path")
@pytest.mark.skipif(sys.platform == "darwin", reason="APFS takes no name that is not UTF-8")
def test_an_anchor_under_a_cache_directory_that_is_not_utf8_compiles_uncached_and_the_warning_asks_for_a_utf8_path(
        tmp_path, monkeypatch):
    """An anchor under a ``NUMBA_CACHE_DIR`` whose path is not UTF-8 compiles without a cache after one warning that
    asks for ``NUMBA_CACHE_DIR`` at a UTF-8 path. numba raises ``UnicodeEncodeError`` when it hashes the anchor's
    directory path, and that directory is writable, so a writable directory would not help."""
    import warnings
    import numba
    from numbox.utils import preprocessing

    cache_dir = tmp_path / os.fsdecode(b"caf\xe9")
    cache_dir.mkdir()
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    code_txt = "def f():\n    return 1\n"
    path = preprocessing._structref_anchor_path("S", code_txt)
    assert path.parent == cache_dir / "numbox-structref"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        options = preprocessing._anchored_or_uncached(path, code_txt, {"cache": True})
    assert options == {"cache": False}
    assert path.is_file()
    assert len(caught) == 1, [str(w.message) for w in caught]
    message = str(caught[0].message)
    assert "surrogates not allowed" in message
    assert message.endswith(
        f"numba hashes the anchor's directory path as UTF-8 to name its cache directory, and {path.parent} is not: "
        f"NUMBA_CACHE_DIR at a UTF-8 path, or {preprocessing._SILENCE}"), message


@pytest.mark.skipif(os.name == "nt", reason="Windows decodes any byte of a path")
@pytest.mark.skipif(sys.platform == "darwin", reason="APFS takes no name that is not UTF-8")
def test_a_source_under_a_directory_that_is_not_utf8_with_numba_cache_dir_set_is_told_to_unset_it(
        tmp_path, monkeypatch):
    """A source under a directory whose path is not UTF-8, with ``NUMBA_CACHE_DIR`` set, gets a remedy that asks for the
    variable to be unset. The variable's locator hashes the path before numba reaches the in-tree locator, which hashes
    no path, and with the variable unset the check passes and numba caches beside the source."""
    import numba
    import numbox.core.configurations as configurations

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    directory = tmp_path / os.fsdecode(b"site\xff")
    directory.mkdir()
    py_file = str(directory / "module.py")
    with open(py_file, "w", encoding="utf-8") as handle:
        handle.write("x = 1\n")
    with pytest.raises(UnicodeEncodeError) as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    failure = raised.value
    assert configurations.is_a_cache_error(failure)
    assert configurations.cache_remedy(py_file, failure, "silence") == (
        "numba hashes the source's directory as UTF-8 for the directory it caches this file in under NUMBA_CACHE_DIR "
        f"or the user's cache directory, and {directory} is not, and NUMBA_CACHE_DIR, which is set to {cache_dir}, is "
        f"hashed for first: unset it, and numba caches beside the source, in its __pycache__, {directory / '__pycache__'}, "
        "hashing no path; or put the file under a UTF-8 path; or silence")
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert (directory / "__pycache__").is_dir()


@pytest.mark.skipif(os.name == "nt", reason="Windows decodes any byte of a path")
@pytest.mark.skipif(sys.platform == "darwin", reason="APFS takes no name that is not UTF-8")
def test_a_source_under_a_directory_that_is_not_utf8_under_the_user_wide_locator_alone_is_told_a_utf8_path(
        tmp_path, monkeypatch):
    """If the list holds the user-wide locator alone, a source under a directory whose path is not UTF-8 gets a remedy
    that asks for a UTF-8 path or for the in-tree locator to be listed first. The user-wide locator hashes the path, so
    no change to its location helps, and the in-tree locator caches beside the source without hashing a path."""
    import numba
    from numba.core import caching
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(caching, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "UserWideCacheLocator", raising=False)
    directory = tmp_path / os.fsdecode(b"site\xff")
    directory.mkdir()
    py_file = str(directory / "module.py")
    with open(py_file, "w", encoding="utf-8") as handle:
        handle.write("x = 1\n")
    with pytest.raises(UnicodeEncodeError) as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    failure = raised.value
    assert configurations.is_a_cache_error(failure)
    assert configurations.cache_remedy(py_file, failure, "silence") == (
        "numba hashes the source's directory as UTF-8 for the directory it caches this file in under NUMBA_CACHE_DIR "
        f"or the user's cache directory, and {directory} is not: put the file under a UTF-8 path, or list "
        "InTreeCacheLocator before UserWideCacheLocator, and numba caches beside the source, in its __pycache__, "
        f"{directory / '__pycache__'}, hashing no path; or silence")


@needs_a_directory_it_cannot_write
@pytest.mark.skipif(os.name == "nt", reason="Windows decodes any byte of a path")
@pytest.mark.skipif(sys.platform == "darwin", reason="APFS takes no name that is not UTF-8")
def test_a_source_under_a_zip_named_directory_on_a_path_that_is_not_utf8_is_told_the_path_not_a_missing_archive(
        tmp_path, monkeypatch):
    """A source on disk under a directory named ``proj.zip``, on a path that is not UTF-8, gets the remedy for the path
    and not the one for a missing archive. Once the in-tree locator skips the unwritable ``__pycache__``, the ``.zip``
    locator listed after it accepts the file and raises ``UnicodeEncodeError`` when it hashes the path."""
    # A UnicodeEncodeError is a ValueError, but it is not the .zip locator's error for a path with no archive.
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "InTreeCacheLocator,ZipCacheLocator", raising=False)
    directory = tmp_path / os.fsdecode(b"site\xff") / "proj.zip"
    directory.mkdir(parents=True)
    py_file = str(directory / "module.py")
    with open(py_file, "w", encoding="utf-8") as handle:
        handle.write("x = 1\n")
    pycache = directory / "__pycache__"
    pycache.mkdir()
    pycache.chmod(0o555)
    try:
        with pytest.raises(UnicodeEncodeError) as raised:
            configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
        failure = raised.value
        assert configurations.is_a_cache_error(failure)
        remedy = configurations.cache_remedy(py_file, failure, "silence")
    finally:
        pycache.chmod(0o755)
    assert remedy == (
        "numba hashes the source's directory as UTF-8 for the directory it caches this file in under NUMBA_CACHE_DIR "
        f"or the user's cache directory, and {directory} is not: put the file under a UTF-8 path, or make the "
        f"__pycache__ beside the source, {pycache}, writable, where InTreeCacheLocator caches it hashing no path; or "
        "silence"), remedy


@pytest.mark.skipif(os.name == "nt", reason="Windows decodes any byte of a path")
@pytest.mark.skipif(sys.platform == "darwin", reason="APFS takes no name that is not UTF-8")
def test_a_frozen_application_under_a_path_that_is_not_utf8_is_told_the_application_under_a_utf8_path(
        tmp_path, monkeypatch):
    """A frozen application under a path that is not UTF-8 makes the check raise numba's ``UnicodeEncodeError``, and the
    remedy asks for the application to be put under a UTF-8 path. numba's user-wide locator, the one locator of numba's
    that accepts a frozen application's file, hashes the path to name its directory under the user's cache directory."""
    import numba
    from numba.core import caching
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(caching, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    directory = tmp_path / os.fsdecode(b"app\xff") / "numbox" / "core"
    py_file = str(directory / "configurations.py")
    with pytest.raises(UnicodeEncodeError) as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    failure = raised.value
    assert configurations.is_a_cache_error(failure)
    assert configurations.cache_remedy(py_file, failure, "silence") == (
        "numba caches a frozen application in the user's cache directory, and NUMBA_CACHE_DIR has no effect here, "
        "because the source is not a file on disk: numba hashes the source's directory as UTF-8 for the directory it "
        f"caches this file in there, and {directory} is not: put the application under a UTF-8 path, or silence")


@pytest.mark.skipif(os.name == "nt", reason="Windows decodes any byte of a path")
@pytest.mark.skipif(sys.platform == "darwin", reason="APFS takes no name that is not UTF-8")
@pytest.mark.parametrize("listed", [
    pytest.param("UserProvidedCacheLocator,UserWideCacheLocator", id="the in-tree locator left out, the variable set"),
    pytest.param("UserWideCacheLocator,InTreeCacheLocator", id="the in-tree locator after the user-wide one"),
])
def test_a_source_under_a_directory_that_is_not_utf8_is_told_the_in_tree_locator_before_the_first_that_hashes(
        tmp_path, monkeypatch, listed):
    """A source under a directory whose path is not UTF-8 gets a remedy that asks for the in-tree locator to be listed
    before the first listed locator that hashes the path. The check passes under the list that the remedy asks for. The
    in-tree locator hashes no path, but numba never reaches it after a locator that raises for the path."""
    # The locators that hash the path are the variable's locator while NUMBA_CACHE_DIR is set, and the user-wide one.
    import numba
    from numba.core import caching
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(caching, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache_dir) if "UserProvided" in listed else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", listed, raising=False)
    directory = tmp_path / os.fsdecode(b"site\xff")
    directory.mkdir()
    (directory / "__pycache__").mkdir()
    py_file = str(directory / "module.py")
    with open(py_file, "w", encoding="utf-8") as handle:
        handle.write("x = 1\n")
    with pytest.raises(UnicodeEncodeError) as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    failure = raised.value
    assert configurations.is_a_cache_error(failure)
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    hashed = (
        "numba hashes the source's directory as UTF-8 for the directory it caches this file in under NUMBA_CACHE_DIR "
        f"or the user's cache directory, and {directory} is not")
    beside = f"numba caches beside the source, in its __pycache__, {directory / '__pycache__'}, hashing no path"
    if "UserProvided" in listed:
        assert remedy == (
            f"{hashed}: put the file under a UTF-8 path, or list InTreeCacheLocator before UserProvidedCacheLocator, "
            f"and {beside}; or silence"), remedy
        monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "InTreeCacheLocator," + listed, raising=False)
    else:
        assert remedy == (
            f"{hashed}, and NUMBA_CACHE_LOCATOR_CLASSES puts UserWideCacheLocator, which hashes it, before "
            f"InTreeCacheLocator: list InTreeCacheLocator before it, and {beside}; or put the file under a UTF-8 path; "
            "or silence"), remedy
        monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "InTreeCacheLocator,UserWideCacheLocator",
                            raising=False)
    # The check passes once the list is changed as the remedy says.
    configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)


@pytest.mark.skipif(os.name == "nt", reason="Windows decodes any byte of a path")
@pytest.mark.skipif(sys.platform == "darwin", reason="APFS takes no name that is not UTF-8")
def test_a_frozen_application_under_a_zip_named_directory_on_a_path_that_is_not_utf8_is_told_the_application(
        tmp_path, monkeypatch):
    """A frozen application under a directory whose name ends in ``.zip``, on a path that is not UTF-8, gets the remedy
    that asks for the application, not an archive, to be put under a UTF-8 path. In numba's order, the user-wide
    locator comes before the ``.zip`` one, and it raises first, when it hashes the path."""
    import numba
    from numba.core import caching
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(caching, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    directory = tmp_path / os.fsdecode(b"app\xff.zip") / "frozen" / "numbox" / "core"
    directory.mkdir(parents=True)
    py_file = str(directory / "module.py")
    with pytest.raises(UnicodeEncodeError) as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    failure = raised.value
    assert configurations.is_a_cache_error(failure)
    assert configurations.cache_remedy(py_file, failure, "silence") == (
        "numba caches a frozen application in the user's cache directory, and NUMBA_CACHE_DIR has no effect here, "
        "because the source is not a file on disk: numba hashes the source's directory as UTF-8 for the directory it "
        f"caches this file in there, and {directory} is not: put the application under a UTF-8 path, or silence")


def test_a_source_that_is_nowhere_under_a_directory_named_zip_is_not_told_to_list_the_zip_locator(tmp_path, monkeypatch):
    """A source that does not exist, under a real directory named ``proj.zip``, gets the remedy that asks for the source
    files, on disk or in a ``.zip``, and not one that asks for ``ZipCacheLocator`` to be listed. That locator treats the
    directory as an archive and caches nothing under it, so listing it would not help."""
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "InTreeCacheLocator", raising=False)
    directory = tmp_path / "proj.zip" / "numbox" / "core"
    directory.mkdir(parents=True)
    py_file = str(directory / "module.py")
    with pytest.raises(RuntimeError, match="no locator available") as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    failure = raised.value
    assert configurations.is_a_cache_error(failure)
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    assert "list it" not in remedy and "ZipCacheLocator alone" not in remedy, remedy
    assert remedy.startswith("NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, ")
    assert "install numbox with its source files on disk" in remedy and remedy.endswith("; or silence"), remedy


def test_a_cell_under_a_directory_named_zip_whose_source_linecache_does_not_hold_is_told_the_cell_remedy(
        tmp_path, monkeypatch):
    """A cell file that does not exist, under a directory named ``proj.zip``, makes the check raise inspect's error, and
    the remedy is the one for a cell whose source is not in linecache. The directory is not an archive, so the remedy
    does not ask for source files to be shipped, as it does for a member of an archive."""
    import types
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(tmp_path / "ipython")
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    directory = tmp_path / "proj.zip" / "ipykernel_123"
    directory.mkdir(parents=True)
    py_file = str(directory / "cell.py")
    with pytest.raises(OSError, match="could not get source code") as raised:
        configurations.check_cache_location(py_file, 0)
    failure = raised.value
    assert failure.errno is None and failure.filename is None
    assert configurations.cache_remedy(py_file, failure, "silence") == (
        f"numba's IPython locator reads a cell's source from linecache, which holds none for {py_file} ({failure}), "
        "and numba caches a cell only while IPython holds its source; or silence")


@on_windows_under_numba_0_68_0
def test_an_egg_under_a_directory_named_zip_is_told_to_put_the_renamed_archive_out_from_under_it(tmp_path, monkeypatch):
    """An ``.egg`` under a directory named ``dir.zip`` gets a remedy that asks for the archive to be renamed to end in
    ``.zip`` and put under a path without that directory, and the check passes once both changes are made. numba's
    ``.zip`` locator takes the first part of the path that ends in ``.zip`` for the archive, so after the rename alone,
    numba would still take the directory for the archive."""
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "InTreeCacheLocator", raising=False)
    in_the_way = tmp_path / "dir.zip"
    in_the_way.mkdir()
    egg = in_the_way / "numbox.egg"
    with zipfile.ZipFile(egg, "w") as zipped:
        zipped.writestr("pkg/m.py", "x = 1\n")
    py_file = str(egg / "pkg" / "m.py")
    with pytest.raises(RuntimeError, match="no locator available") as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert configurations.cache_remedy(py_file, raised.value, "silence") == (
        "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, install numbox with "
        f"its source files on disk, unpacked from any archive, or rename the archive, {egg}, to end in .zip, "
        "lower-case, which numba's .zip locator looks for in the path, with its source files in it, and put it under a "
        f"path with no part ending in .zip above it, {in_the_way} being one, which that locator takes for the archive "
        "first, so that numba caches it in the user's cache directory through ZipCacheLocator, once "
        "NUMBA_CACHE_LOCATOR_CLASSES, InTreeCacheLocator, lists it; or silence")
    # With the archive renamed and moved, and the .zip locator listed, the check passes.
    moved = tmp_path / "numbox.zip"
    egg.rename(moved)
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "InTreeCacheLocator,ZipCacheLocator", raising=False)
    configurations.check_cache_location(str(moved / "pkg" / "m.py"), configurations.LONGEST_CACHE_FILE_NAME)


@pytest.mark.parametrize("units", [230, 231, 251, 252])
def test_the_offer_of_a_file_system_that_takes_a_cache_file_name_holds_at_the_usual_limit(tmp_path, monkeypatch, units):
    """The remedy offers ``NUMBA_CACHE_DIR`` on a file system that allows the name only if the longest name that numba
    writes for the function fits in 255. That name is the index's base, a data file's suffix ``.999.nbc`` with three
    digits for its number, and the 21 bytes of numba's temporary name. The test uses names at that bound and one byte
    past it."""
    # A limit of 143, as on eCryptfs, makes each of these names too long for the file system that holds the location.
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setattr(configurations, "_name_limit", lambda location: 143)
    site = tmp_path / "site"
    site.mkdir()
    py_file = str(site / "module.py")
    with open(py_file, "w", encoding="utf-8") as handle:
        handle.write("x = 1\n")
    tail = f"-1.py{sys.version_info[0]}{sys.version_info[1]}.nbi"
    if units > 240:
        # The longer names are those of numba's temporary file, the index's name and 21 bytes more.
        tail += ".tmp.0123456789abcdef"
    name = f"module.{'f' * (units - len('module.') - len(tail))}{tail}"
    counted_in = "characters" if os.name == "nt" or sys.platform == "darwin" else "bytes"
    assert configurations._measured(name) == (units, counted_in)
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(site / "__pycache__" / name))
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    base = units - len(".nbi") - (len(".tmp.0123456789abcdef") if units > 240 else 0)
    fits = base + len(".999.nbc") + 21 <= 255
    assert fits == (units in (230, 251))
    offer = ", or NUMBA_CACHE_DIR set to a directory on a file system that takes a name that long" if fits else ""
    assert remedy == (
        "numba names the cache files for a function after the source's name and the function's qualified name, "
        f"{name}, {units} {counted_in}, which is longer than a name can be on that file system (143), and no shorter "
        f"path cures it: a shorter name for the function or its file, which is numbox's to give{offer}; or silence"), remedy


def test_a_callers_error_naming_another_numba_cache_for_a_member_is_told_that_location_under_a_temporary_ipython_too(
        tmp_path, monkeypatch):
    """A caller's ``EACCES`` that names an unrelated ``numba_cache``, for a member in an ``ipykernel_`` directory, gets
    the remedy for that location, whether or not IPython gives temporary directories. Only an ``ENOENT`` that names a
    ``numba_cache``, such as the error that numba's first save raises under such an IPython, gets the remedy that names
    ``IPYTHONDIR``."""
    import tempfile
    import types
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    stable = tmp_path / "stable-cache"
    stable.mkdir()
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(stable)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ipykernel_123/cell.py", "def cell():\n    return 1\n")
    py_file = str(archive / "ipykernel_123" / "cell.py")
    unrelated = str(tmp_path / "elsewhere" / "numba_cache")
    refused = OSError(errno.EACCES, "Permission denied", unrelated)
    told = (
        "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: no file can be written at that "
        f"location, {unrelated} (Permission denied), so make room there, or make it writable, or silence")
    assert configurations.cache_remedy(py_file, refused, "silence") == told
    paths.get_ipython_cache_dir = lambda: tempfile.mkdtemp(dir=tmp_path)
    assert configurations.cache_remedy(py_file, refused, "silence") == told
    # numba's first save under such an IPython raises ENOENT, and that error gets the remedy that names IPYTHONDIR.
    missing = FileNotFoundError(errno.ENOENT, "No such file or directory", unrelated)
    assert configurations.cache_remedy(py_file, missing, "silence") == (
        "numba caches this file in numba_cache under IPython's cache directory, and IPython, able to write neither "
        "that directory nor its own, gives a new temporary directory on every call, so numba makes numba_cache in one "
        "and saves in another: set IPYTHONDIR, or XDG_CACHE_HOME on a POSIX system, to a writable directory; or silence")


@needs_a_directory_it_cannot_write
@pytest.mark.skipif(os.name == "nt", reason="Windows decodes any byte of a path")
@pytest.mark.skipif(sys.platform == "darwin", reason="APFS takes no name that is not UTF-8")
def test_a_source_under_a_directory_that_is_not_utf8_with_the_variable_set_and_its_pycache_unwritable_is_told_both(
        tmp_path, monkeypatch):
    """A source under a directory whose path is not UTF-8, with ``NUMBA_CACHE_DIR`` set and the ``__pycache__`` beside
    it read-only, gets a remedy that asks for both changes: the variable unset and the ``__pycache__`` made writable.
    numba's error does not show whether the in-tree locator could cache the file, so the remedy asks that locator."""
    import numba
    import numbox.core.configurations as configurations

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    directory = tmp_path / os.fsdecode(b"site\xff")
    directory.mkdir()
    py_file = str(directory / "module.py")
    with open(py_file, "w", encoding="utf-8") as handle:
        handle.write("x = 1\n")
    pycache = directory / "__pycache__"
    pycache.mkdir()
    pycache.chmod(0o555)
    try:
        with pytest.raises(UnicodeEncodeError) as raised:
            configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
        remedy = configurations.cache_remedy(py_file, raised.value, "silence")
        assert remedy == (
            "numba hashes the source's directory as UTF-8 for the directory it caches this file in under "
            f"NUMBA_CACHE_DIR or the user's cache directory, and {directory} is not, and NUMBA_CACHE_DIR, which is set "
            f"to {cache_dir}, is hashed for first: unset it, and make the __pycache__ beside the source, {pycache}, "
            "writable, and numba caches there, hashing no path; or put the file under a UTF-8 path; or silence"), remedy
        # With the variable unset alone, the in-tree locator skips the read-only __pycache__, and the user-wide locator
        # raises the same error. With both changes made, the check passes.
        monkeypatch.setattr(numba.config, "CACHE_DIR", "")
        with pytest.raises(UnicodeEncodeError):
            configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    finally:
        pycache.chmod(0o755)
    configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)


# MAKE_A_STRUCTREF_OF_FORTY_CHARACTER_NAMES is a child whose struct and field have names of 40 characters, the longest
# names that numbox puts into a file name unchanged.
MAKE_A_STRUCTREF_OF_FORTY_CHARACTER_NAMES = MAKE_A_STRUCTREF.replace(
    "make_structref('Struct', {'value': float32}, TypeClass)",
    "make_structref('S' * 40, {'v' * 40: float32}, TypeClass)").replace(
    "assert Struct(2.5).value == 2.5", "assert getattr(Struct(2.5), 'v' * 40) == 2.5")


@pytest.mark.skipif(sys.platform != "linux", reason="a path of 4096 bytes is Linux's limit")
@pytest.mark.parametrize("short_of_the_limit", [230, 190])
def test_an_anchor_within_numbas_longest_name_of_the_path_limit_compiles_uncached_and_the_warning_says_so(
        tmp_path, short_of_the_limit):
    """A ``NUMBA_CACHE_DIR`` too close to the path limit for the longest cache file name of a struct makes the struct
    compile without a cache after one warning that asks for ``NUMBA_CACHE_DIR`` at a shorter path. The anchor's check
    creates a file with a name as long as the longest name that numba writes for the struct's code, so the check fails
    for the location before numba's first save can fail there."""
    deep = _directory_of_length(tmp_path, 4096 - short_of_the_limit)
    env = dict(os.environ, PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(deep))
    for name in ("NUMBOX_JIT_OPTIONS", "XDG_CACHE_HOME"):
        env.pop(name, None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", MAKE_A_STRUCTREF_OF_FORTY_CHARACTER_NAMES],
                         capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and "made" in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "The path is too long for the file system: NUMBA_CACHE_DIR at a shorter path" in run.stderr, run.stderr


def test_an_error_naming_a_zips_location_by_a_relative_path_under_nothing_yet_names_the_directory_it_is_made_under(
        tmp_path, monkeypatch):
    """A ``.zip`` member's error that names its location by a relative path, under directories that do not exist yet,
    gets a remedy that names the working directory as the one under which numba cannot create the location. The remedy
    reads the path as absolute before it walks up to the nearest existing directory."""
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderNothingYet:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "home" / ".cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderNothingYet)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.chdir(tmp_path)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("package/module.py", "def f():\n    pass\n")
    py_file = str(archive / "package" / "module.py")
    location = os.path.join("home", ".cache", "numba", _CacheLocator.get_suitable_cache_subpath(py_file))
    remedy = configurations.cache_remedy(py_file, OSError(errno.EACCES, "Permission denied", location), "silence")
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        f"effect here, because the source is not a file on disk: no file can be written in that directory, {location}, "
        f"which cannot be made under {tmp_path} (Permission denied), so make room there, or make it writable, or "
        "silence"), remedy


def test_a_name_limit_below_the_longest_name_the_check_asks_for_is_told_the_length_not_a_shorter_path(
        tmp_path, monkeypatch):
    """If the file system allows no name as long as the one that the check creates, the check's error carries that
    length, and the remedy asks for a file system that allows it instead of a shorter path. The anchor's warning does
    the same for generated code, with the longest name that numba writes for that code."""
    # The stand-in for tempfile's NamedTemporaryFile refuses a name longer than the stand-in limit, as such a file
    # system would.
    import tempfile
    import warnings
    import numba
    import numbox.core.configurations as configurations
    from numbox.utils import preprocessing

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setattr(configurations, "_name_limit", lambda location: 100)
    real = tempfile.NamedTemporaryFile

    def refusing_a_long_name(*args, **kwargs):
        if len(kwargs.get("prefix", "")) + 8 > configurations._name_limit(kwargs["dir"]):
            raise OSError(errno.ENAMETOOLONG, "File name too long", os.path.join(kwargs["dir"], kwargs["prefix"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(configurations.tempfile, "NamedTemporaryFile", refusing_a_long_name)
    py_file = tmp_path / "site" / "module.py"
    py_file.parent.mkdir()
    py_file.write_text("x = 1\n")
    with pytest.raises(OSError) as raised:
        configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
    failure = raised.value
    assert failure.errno == errno.ENAMETOOLONG and failure.filename == str(py_file.parent / "__pycache__")
    assert failure.longest_file_name == configurations.LONGEST_CACHE_FILE_NAME
    assert configurations.is_a_cache_error(failure)
    counted_in = "characters" if os.name == "nt" or sys.platform == "darwin" else "bytes"
    assert configurations.cache_remedy(str(py_file), failure, "silence") == (
        f"numba's cache files for numbox's functions have names of up to {configurations.LONGEST_CACHE_FILE_NAME} "
        f"{counted_in}, longer than a name can be on that file system (100), so no shorter path cures it, or "
        "NUMBA_CACHE_DIR set to a directory on a file system that takes a name that long; or silence")
    # The anchor's check asks for the longest name that numba writes for the functions of the code, which numbox reads
    # from the code.
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    code_txt = "@njit(**jit_options)\ndef f():\n    return 1\n"
    path = preprocessing._structref_anchor_path("S", code_txt)
    longest = preprocessing._longest_cache_file_name(path.stem, code_txt)
    assert 40 < longest < 100
    monkeypatch.setattr(configurations, "_name_limit", lambda location: 40)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        options = preprocessing._anchored_or_uncached(path, code_txt, {"cache": True})
    assert options == {"cache": False}
    assert len(caught) == 1, [str(w.message) for w in caught]
    assert str(caught[0].message).endswith(
        f"numba's cache files for the functions anchored here have names of up to {longest} bytes, longer than a "
        f"name can be on that file system (40): NUMBA_CACHE_DIR at a directory on a file system that takes a name "
        f"that long, or {preprocessing._SILENCE}"), str(caught[0].message)


def test_the_anchors_check_asks_for_the_longest_name_the_code_gets_not_a_bound_for_any_struct(tmp_path, monkeypatch):
    """The anchor's check asks for the longest name that numba writes for the code's own functions, read from the code,
    and not for a bound that fits any struct. Code with short names therefore caches on a file system with a limit of
    143 bytes, such as eCryptfs, and code with a longer name warns there."""
    # The stand-in for tempfile's NamedTemporaryFile refuses a name longer than 143 bytes.
    import tempfile
    import warnings
    import numba
    import numbox.core.configurations as configurations
    from numbox.utils import preprocessing

    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setattr(configurations, "_name_limit", lambda location: 143)
    real = tempfile.NamedTemporaryFile

    def refusing_a_long_name(*args, **kwargs):
        if len(kwargs.get("prefix", "")) + 8 > 143:
            raise OSError(errno.ENAMETOOLONG, "File name too long", os.path.join(kwargs["dir"], kwargs["prefix"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(configurations.tempfile, "NamedTemporaryFile", refusing_a_long_name)
    short = "@njit(**jit_options)\ndef f():\n    return 1\n"
    path = preprocessing._structref_anchor_path("Point", short)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert preprocessing._anchored_or_uncached(path, short, {"cache": True}) == {"cache": True}
    assert not caught, [str(w.message) for w in caught]
    assert preprocessing._longest_cache_file_name(path.stem, short) < 143
    long = f"@njit(**jit_options)\ndef {'f' * 120}():\n    return 1\n"
    path = preprocessing._structref_anchor_path("Point", long)
    longest = preprocessing._longest_cache_file_name(path.stem, long)
    assert longest > 143
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert preprocessing._anchored_or_uncached(path, long, {"cache": True}) == {"cache": False}
    assert len(caught) == 1 and f"names of up to {longest} bytes, longer than a name can be on that file system (143)" in str(
        caught[0].message), [str(w.message) for w in caught]
    # numbox counts each decorated function, whatever the decorator, and each function defined inside one, with
    # numba's "locals" for the nesting. That includes an overload and the constructor that it returns, and a derive
    # function under a decorator of numbox's own. numba writes no cache file for a plain method under its own name, and
    # numbox does not count it. Each name has three digits for the data file's number and numba's temporary suffix.
    version = f"py{sys.version_info[0]}{sys.version_info[1]}{getattr(sys, 'abiflags', '')}"
    code_txt = ("class C:\n    @njit(**jit_options)\n    def m(self):\n        def inner():\n"
                "            return 1\n        return inner\n\n    def plain_but_long_" + "p" * 200 + "(self):\n"
                "        return 2\n\n\n@overload(S, strict=False, jit_options=jit_options)\ndef ol_s(x):\n"
                "    def a_constructor_named_" + "c" * 60 + "(x):\n        return 3\n    return ctor\n\n\n"
                "@_cres\ndef derive_" + "d" * 50 + "(x):\n    return 4\n")
    counted = [f"stem.C.m-3.{version}.999.nbc.tmp.", f"stem.C.m.locals.inner-4.{version}.999.nbc.tmp.",
               f"stem.ol_s-13.{version}.999.nbc.tmp.",
               f"stem.ol_s.locals.a_constructor_named_{'c' * 60}-14.{version}.999.nbc.tmp.",
               f"stem.derive_{'d' * 50}-20.{version}.999.nbc.tmp."]
    assert preprocessing._longest_cache_file_name("stem", code_txt) == max(len(each) for each in counted) + 16
    assert preprocessing._longest_cache_file_name("stem", "x = 1\n") == 0
    assert preprocessing._longest_cache_file_name("stem", "def plain():\n    return 1\n") == 0


@pytest.mark.parametrize("number, base, offered", [
    pytest.param(1, 226, True, id="a one-digit number, three reserved, the base at the limit"),
    pytest.param(100, 227, False, id="a hundredth signature at the base two digits left room for"),
    pytest.param(1000, 226, False, id="a thousandth signature, wider than the reserve, read by its width"),
])
def test_a_cache_file_name_with_a_number_wider_than_the_reserve_is_read_by_its_own_width(
        tmp_path, monkeypatch, number, base, offered):
    """The remedy measures a data file's name with three digits for its number, or with the width of the number that the
    error names if that is wider, before it offers a file system that allows the name. numba numbers a function's data
    files from one without bound, so the remedy must not offer a file system on which a later data file's name would not
    fit."""
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setattr(configurations, "_name_limit", lambda location: 143)
    site = tmp_path / "site"
    site.mkdir()
    py_file = str(site / "module.py")
    with open(py_file, "w", encoding="utf-8") as handle:
        handle.write("x = 1\n")
    name = f"module.{'f' * (base - len('module.'))}.{number}.nbc.tmp.0123456789abcdef"
    counted_in = "characters" if os.name == "nt" or sys.platform == "darwin" else "bytes"
    units = len(name)
    assert (base + len(".") + max(3, len(str(number))) + len(".nbc") + 21 <= 255) == offered
    remedy = configurations.cache_remedy(
        py_file, OSError(errno.ENAMETOOLONG, "File name too long", str(site / "__pycache__" / name)), "silence")
    offer = ", or NUMBA_CACHE_DIR set to a directory on a file system that takes a name that long" if offered else ""
    assert remedy == (
        "numba names the cache files for a function after the source's name and the function's qualified name, "
        f"{name}, {units} {counted_in}, which is longer than a name can be on that file system (143), and no shorter "
        f"path cures it: a shorter name for the function or its file, which is numbox's to give{offer}; or silence")
