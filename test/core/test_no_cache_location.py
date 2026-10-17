"""Importing numbox where numba can write no cache for it.

numba sets a cached function up when it is decorated and raises there when no
cache location can be written, so ``import numbox`` died at its first module
under an archive import or a read-only install. Every check here runs in a
subprocess with its own tree and cache directory, so the placement under test
is the one the subprocess sees and nothing else.
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

# chmod takes write access from a directory neither on Windows nor from root.
needs_a_directory_it_cannot_write = pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="needs a directory this user cannot write to")

# A lazily compiled helper, an eagerly compiled one and a proxied binding, each
# decorated at import under the package's options, then used.
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


ARCHIVE_REMEDY = (
    "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, install numbox with its "
    "source files on disk, unpacked from any archive, or import it from a .zip holding its source files, which numba "
    "caches in the user's cache directory"
)


def _archive(path, bytecode_naming_the_archive=()):
    """numbox's modules zipped into ``path``, which goes on PYTHONPATH as it is.

    A directory, or a file, named in ``bytecode_naming_the_archive`` goes in as ``.pyc`` alone, each compiled to
    name its path inside the archive, as ``compileall -d`` and ``py_compile``'s ``dfile`` do.
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
    # From a neutral cwd the child imports the placement PYTHONPATH names,
    # not the tree the test runs from.
    return subprocess.run([sys.executable, "-W", warnings, "-c", IMPORT_AND_USE],
                          capture_output=True, text=True, env=env, cwd=str(cwd))


def _index_files(cache_dir):
    return sorted(path.name for path in Path(cache_dir).rglob("*.nbi"))


def _directory_of_length(root, length):
    """A directory made under ``root`` whose path is ``length`` bytes, whatever ``root``'s length.

    Components of 200 while one more leaves room for a last, then the last of the length that lands there, so
    the directory itself can always be made.
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
    # numba's cache locators need the source file on disk, so an import from
    # an .egg, .whl or .pyz archive, which Spark's --py-files ships, raised
    # RuntimeError at the first decorated function, naming neither the way to
    # a cache nor the option that turns caching off. The package answers the
    # question once, so there is one warning, not one per function.
    # NUMBA_CACHE_DIR is no way to a cache here: it is set and writable, the
    # warning fires all the same and nothing is written there, so the warning
    # says so instead of offering it. Under a directory whose name holds .zip
    # numba's .zip locator takes the file by that substring and raises
    # ValueError finding no .zip in it; that is the cache's error too.
    (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
    archive = _archive(tmp_path / name)
    env = dict(os.environ, PYTHONPATH=str(archive), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "NUMBA_CACHE_DIR has no effect here" in run.stderr and "Set NUMBA_CACHE_DIR" not in run.stderr
    assert _index_files(tmp_path / "cache") == []
    quiet = _run(dict(env, NUMBOX_JIT_OPTIONS='{"cache": false}'), tmp_path, warnings="error")
    assert quiet.returncode == 0, quiet.stderr


def test_options_without_a_cache_key_are_asked_for_the_sites_that_cache_under_them(tmp_path):
    # NUMBOX_JIT_OPTIONS without "cache" leaves njit's default, off, but the
    # sqlite virtual-table and table-valued-function callbacks cache unless it
    # says no, and a check that read the missing key as off let the first
    # callback die at numba's set-up from an archive.
    archive = _archive(tmp_path / "numbox-0.0.0-py3.12.egg")
    env = dict(os.environ, PYTHONPATH=str(archive), NUMBA_CACHE_DIR=str(tmp_path / "cache"),
               NUMBOX_JIT_OPTIONS='{"boundscheck": false}')
    child = "import numbox.core.bindings.sqlite.vtable as m; print(m.__file__)"
    run = subprocess.run([sys.executable, "-W", "always", "-c", child], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path))
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr


windows_path_sep_bug = "numba 0.68.0 reads a .zip member's source by a name built with Windows's separator"


# numba 0.68.0 on Windows reads a .zip member's source by a name built with
# Windows's separator and raises KeyError before any location is tried, numba
# issue 10889, fixed for 0.68.1: a check of a member there meets the bug, not
# the case under test, where the fork's matrix runs Windows on 0.67.0 and
# upstream's on the latest numba.
on_windows_under_numba_0_68_0 = pytest.mark.skipif(
    os.name == "nt" and version("numba") == "0.68.0", reason=windows_path_sep_bug)


def _skip_for_windows_separator_bug(stderr):
    """Skip a test whose import met numba issue 10889, which numba 0.68.0 has on Windows alone: the remedy's text for
    it in ``stderr`` on any other platform is a regression, which the skip hid wherever it keyed on the text alone."""
    if windows_path_sep_bug in stderr:
        assert os.name == "nt", f"the Windows separator remedy on {os.name}: {stderr}"
        pytest.skip(windows_path_sep_bug)


def test_a_zip_import_is_cached_by_numba(tmp_path):
    # The warning sends an archive's user to a .zip, which numba caches
    # in the user's cache directory whatever NUMBA_CACHE_DIR says.
    # Before that a .zip is one more archive. This pins the remedy the warning
    # names, numba's behaviour, not the fallback, which has nothing to do here.
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
        # On Windows numba asks the system for the user's cache directory, and no variable set here moves it.
        assert _index_files(home)


@needs_a_directory_it_cannot_write
def test_one_read_only_directory_among_writable_ones_takes_the_fallback(tmp_path):
    # numba's in-tree cache is a __pycache__ beside each source, so the
    # package's directories answer separately. A check on configurations.py's
    # directory alone passed here, and the import died at libm's first binding.
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


IMPORT_SQLITE_TYPEMAP = "import numbox.core.bindings.sqlite._typemap as m; print(m.__file__)"
IMPORT_LIBM = "import numbox.core.bindings.libm as m; print(m.__file__)"


def test_a_module_that_survives_as_pyc_alone_beside_sourced_ones_takes_the_fallback(tmp_path):
    # numba finds a location by each module's own source, so one module without
    # its .py fails where its neighbours, __init__.py among them, pass. A walk
    # that asked one module per directory passed here, and libm died.
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
    # The sourceless loader keeps the file a .pyc was compiled from on its code,
    # and numba looks that up: here a tree elsewhere, still on disk, so libm's
    # functions cache under NUMBA_CACHE_DIR by it. A probe that asked by the .py
    # beside the .pyc, gone, turned caching off for the package and said the
    # source is not on disk.
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
    # The rest of the package keeps its sources, so its directories pass; the
    # one directory numba cannot cache from has no .py to name, and a walk that
    # named directories by their .py files skipped it. The remedy is the one
    # for a source that is not on disk, not NUMBA_CACHE_DIR, which numba reads
    # for a source on disk only.
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
    # A .pyc-only install: numba looks the source up by the code's co_filename,
    # which names the .py that was removed, and finds no locator. The module's
    # __file__ is the .pyc, on disk, and a check on it offered NUMBA_CACHE_DIR,
    # which numba ignores without the source.
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
    # zipimport takes a .pyc before the .py beside it and keeps the file it was
    # compiled from on its code, which is what numba looks up for the module's
    # functions; here that file is gone. A listing of the archive's .py members
    # said every directory caches and libm's first binding died at numba's set-up.
    # A tree named after the archive, beside it, passed a test by string prefix
    # for the archive's own files, and its .pyc went unasked: numba, seeing ".zip"
    # in the file's path and no part named so, died with its ValueError.
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
    # zipimport runs the .py where the .pyc beside it is stale against it, by
    # size or time, so the module's functions are cached from the archive like
    # the rest; a probe that asked by the stale .pyc's compile-time file, gone
    # here, turned caching off for the package and offered a .zip holding the
    # sources, which this is. This pins the outcome; that a .pyc zipimport
    # would run is asked at all is pinned by the sourceless case above, since
    # passing every .pyc over gives this outcome too.
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
        # On Windows numba's user cache directory is the shell's known folder,
        # which no variable moves under the home.
        assert _index_files(home), "libm's functions were not cached from the archive"


def test_a_stale_pyc_whose_source_in_the_zip_does_not_compile_is_passed_over(tmp_path):
    # zipimport, passing the stale .pyc over, compiles the .py beside it, and a
    # syntax error there, uncaught, killed the import of configurations where
    # nothing imports the module; its own import would die on it, as it should.
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
    # A .pyc member of this interpreter's magic that zipimport could not run,
    # truncated, or stored with a compression zipimport does not read, and that
    # no import reaches: the probe's unmarshal of the first died with EOFError
    # at the import of configurations, where main imported, and zipimport's
    # own read of the second with zlib.error, past the exceptions then caught.
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
    # numba takes the user's cache directory for a .zip without checking that
    # it can be written, so where it cannot, an executor with a read-only
    # home, the first save raised PermissionError and the import died on it.
    # The remedy is that directory made writable, not the .zip the archive
    # warning offers, which is what the user already has.
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
        assert "import it from a .zip" not in run.stderr
    finally:
        home.chmod(0o755)


@needs_a_directory_it_cannot_write
def test_a_zip_whose_configurations_runs_from_a_pyc_compiled_from_a_tree_still_on_disk(tmp_path):
    # configurations.pyc in the archive, current, compiled from a tree that is
    # still on disk: zipimport runs it with that tree's file on its code, and a
    # probe that took the package's place from its own code's file walked the
    # tree, found it cacheable, and never asked the archive, whose first
    # function died at numba's first save with the home read-only.
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
    # numba's location for a .zip is under the user's cache directory, and
    # with a component of that too long for the file system the location
    # cannot be made: the warning said to make it writable, which cannot help;
    # the path is the thing, through XDG_CACHE_HOME or HOME. The home itself
    # carries the component, since macOS keeps the directory under
    # ~/Library/Caches and reads no XDG_CACHE_HOME.
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
    # numba reads the source's stamp at decoration, the archive's for a .zip,
    # by the path the module's code names: .pyc members compiled to name the
    # archive, then the archive moved, name a path that is not there, and the
    # import died there with FileNotFoundError past a check that read no stamp.
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
    # numba's location cannot be made under a link to nowhere, ENOENT, which is
    # the error a moved archive's stamp gives too: the warning blamed the
    # archive's .pyc members, of which this archive has none. The moved archive
    # is the one whose path the module's code lies under. The home is the link,
    # for macOS's ~/Library/Caches as for Linux's ~/.cache.
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
    # Every entry is in the user's cache directory from an earlier import, and
    # the directory can no longer be written. A probe that only loaded its own
    # entry would have said the cache works and left the import to die at
    # numba's first write for anything not there; the writability check numba
    # runs for every other placement is run here for the .zip too.
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
    # numba caches a .zip per directory of it, each in a location of its own
    # under the user's cache directory, so the directories answer separately
    # there too; a check on configurations.py's location alone passed here,
    # and libm died at its first save. The archive's directories are listed,
    # the archive being the first part of the path named .zip that is one: a
    # listing that took a directory so named above it listed nothing. A
    # directory of .pyc members compiled to name the archive, as compileall -d
    # does, stands in the listing like one of .py members: a listing that let
    # only .py members stand never asked for it. From numba 0.68 the archive
    # is opened for each member's source, and the directory so named above
    # it, or a directory of .pyc members, takes the fallback instead, told so.
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
    # Under XDG_CACHE_HOME on Linux, under Library/Caches on macOS.
    locations = [path for path in home.rglob("bindings_*") if path.is_dir()]
    assert len(locations) == 1, sorted(str(path.relative_to(home)) for path in home.rglob("*") if path.is_dir())
    locations[0].chmod(0o555)
    try:
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        # The directory to make writable is the location that lost it, not the
        # user's cache directory above it, which the warning named.
        assert (f"no file can be written in that directory, {locations[0]} (Permission denied), so make room there, "
                "or make it writable") in run.stderr, run.stderr
    finally:
        locations[0].chmod(0o755)


def test_a_pyc_member_alone_beside_sourced_ones_in_a_zip_is_asked_for_itself(tmp_path):
    # numba 0.68 and later read each member's source from the archive by the
    # name its code gives, so a module shipped as .pyc alone fails where the
    # .py members beside it pass: a listing that let one member stand for the
    # directory never asked for it, and libm died at its first binding.
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
    # zipimport runs a .pyc before the .py beside it, a hash-based one it
    # does not check whatever the .py's age, and numba reads the member's
    # source by the name the .pyc's code gives, from numba 0.68 the source
    # too; compiled to name another path in the archive, compileall's -d
    # with a path the archive does not hold, that name is of no member. The
    # listing took a .pyc naming the archive for one standing for its
    # directory, like its .py, and never asked: libm died at decoration.
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
    # Two links pointing up the package, followed with no memory of where the
    # walk had been, gave an exponential number of paths before the file
    # system's link limit, and the import did not finish; the base imported at
    # once. Each real directory is walked once.
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
    # A directory of the package reached through a symlink was not walked, so a
    # read-only one behind the link, with no user cache, passed the check and
    # died at libm's first binding.
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
    # With sys.frozen set numba caches a source that is not on disk in the
    # user's cache directory, as a .zip; where that directory cannot be written
    # the remedy is to make it so, not the sources or a .zip the archive text
    # offered.
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
        # The frozen error is the no-locator one, which names no directory
        # and gives no reason, so the warning names numba's, the user cache
        # directory under the home, as one numba could not use.
        named = re.search(r"numba could not use that directory, (.+?): make it writable", run.stderr)
        assert named and Path(named.group(1)).is_relative_to(home), run.stderr
        assert "install numbox with its source files" not in run.stderr
    finally:
        for path in read_only:
            path.chmod(0o755)


@needs_a_directory_it_cannot_write
def test_a_read_only_install_warns_naming_numba_cache_dir_and_setting_it_caches(tmp_path):
    # The other way to have no cache location: the source is on disk, and
    # neither its directory nor the user's cache directory can be written.
    # NUMBA_CACHE_DIR is the remedy there, and nothing showed that the warning
    # names it or that setting it works.
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
    # numba's own check of a location makes a temporary file, one without a
    # name on Linux, so a NUMBA_CACHE_DIR whose location for a module fits the
    # path limit while numba's files, of a hundred bytes and more, do not
    # passed it, and the import died at the first save. The probe makes a file
    # named as long as numba's longest for the package, and the warning says
    # what the length's remedy is. The location of the first module asked,
    # the directory's name and a hash of its path under NUMBA_CACHE_DIR, ends
    # 20 bytes short of the limit, whatever the checkout's own path.
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
    # Where the source's __pycache__ cannot be written and NUMBA_CACHE_DIR is
    # not set, numba caches under the user's cache directory, and a location
    # there too long for numba's files was told "a shorter NUMBA_CACHE_DIR, or
    # none, where it is set, else numbox installed at a shorter path": a
    # variable that was not set and a reinstall that changes nothing. The first
    # module asked has its location there end 20 bytes short of the limit.
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
        assert "a shorter NUMBA_CACHE_DIR" not in run.stderr and "numbox installed" not in run.stderr, run.stderr
        cured = _run(dict(env, XDG_CACHE_HOME=str(tmp_path / "short")), tmp_path, warnings="error")
        assert cured.returncode == 0, cured.stderr
        assert _index_files(tmp_path / "short")
    finally:
        for path in read_only:
            path.chmod(0o755)


# Each case: where NUMBA_CACHE_DIR is, where the source is, and which of numba's
# locations for it is too long; "user" stands for the user's cache directory.
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
    # numba takes a directory under NUMBA_CACHE_DIR where that is set, else the
    # __pycache__ beside the source, else a directory under the user's cache
    # directory, and the error names the one that is too long. One remedy for
    # all three named a variable that was not set and a reinstall that changes
    # nothing where the location was under the user's cache directory, and
    # telling them apart by which directory a location starts with took one for
    # another where they nest: an install under either cache directory, or
    # NUMBA_CACHE_DIR above the user's. The user's cache directory is one under
    # tmp_path, so that every case runs on every platform. numbox's own check
    # names the location; numba's first save names a file in it, and a caller
    # can pass that error on.
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
    # A location other than NUMBA_CACHE_DIR's where the variable is set means
    # numba passed it over, so the warning names it rather than telling the
    # reader to set what is set.
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
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
    if named == "a file in it":
        where = where / "module.f-1.py312.nbi"
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(where))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence", package="numbduck")
    assert remedy == f"the path is too long for the file system: {cure}; or silence", remedy


def test_a_relative_numba_cache_dir_is_matched_as_the_location_numba_names(tmp_path, monkeypatch):
    # numba joins NUMBA_CACHE_DIR as set to the name it gives the source's
    # directory, so set relative, to the directory the process runs from, the
    # location is relative, and so is the path numba's error, and the
    # package's own, names. The remedy writes the locations absolute, and the
    # path as named matched none of them, so it was told as a location that
    # is none of numba's, with the variable to be made a writable directory
    # at a short path, where a shorter NUMBA_CACHE_DIR shortens it.
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
    # NUMBA_CACHE_DIR set to the user's cache directory makes one path of two
    # of numba's locations, and which locator took it is the order's to say:
    # numba's own order tries the variable first, so a shorter NUMBA_CACHE_DIR
    # shortens it, and nothing else does: numba's own check passed the
    # location, so numba keeps taking the variable, and the user's cache
    # directory at a shorter path, once offered beside it, moved nothing numba
    # reads; a list that tries the user's cache directory before the
    # variable, or never reads the variable, took it as the user's cache
    # directory, and the variable's remedy, matched first whatever the order,
    # told the reader to shorten a variable numba did not read.
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
    # A caller can pass on an error naming a location that is none of the three
    # numba builds from the file. The remedy names it and offers NUMBA_CACHE_DIR
    # where numba tries the variable before any other location, the IPython
    # locator ahead taking no file on disk; it read "NUMBA_CACHE_DIR set to a
    # short path, which numba takes first", the alternative's clause cut from
    # its "or" and nothing said of the location. Set, the variable is named
    # without the claim that numba passed it over: a location that is none of
    # numba's shows nothing of what numba did with the variable, where the
    # remedy said it was one "numba could not use", as it says of a set
    # variable passed over for a location numba took. With another locator
    # ahead of the variable's, the variable is not offered, and no case here
    # held that: the offer with the position dropped passed every one.
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
    # numba's errors and the package's own check name the file refused, but a
    # caller can pass on an ENAMETOOLONG that names nothing. The remedy then
    # cannot tell which location numba took, and it lists the locations numba
    # could have taken for the file, in numba's order, for the reader to put
    # the one that is too long at a shorter path; it read "that location,
    # None, at a shorter path", and matched the empty name against the working
    # directory and its parent, which could have taken one location for
    # another. NUMBA_CACHE_DIR set to the user's cache directory makes one
    # path of two locations, and the list named it twice. An empty name is no
    # name either, and it read "that location, , at a shorter path".
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
    # The working directory is the source's __pycache__, so an empty name
    # matched against it would take the in-tree location for the one named.
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
    # A list of one locator leaves numba one location to take for the file, so
    # an error that names no file is told that location. The list of the
    # locations numba could have taken read "one of" before the one path.
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


# Each case: where NUMBA_CACHE_DIR is, NUMBA_CACHE_LOCATOR_CLASSES, which of
# numba's locations the error names, and the alternative offered after it.
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
    # numba takes a location its own check could write a temporary file in,
    # and the package's check then writes a named file there, which a full
    # disk, a quota or permissions changed since can refuse. The location is
    # the one numba took, so "NUMBA_CACHE_DIR is set to X, which numba could
    # not use: set it to a writable directory at a short path", the answer for
    # a variable numba passed over, was false on both counts; the remedy names
    # the location and the reason, and offers the variable as the alternative
    # where numba tries it before that location, or set elsewhere where the
    # location is the variable's own. A location that is none of numba's for
    # the file, which a caller can pass on, is named as the error names it,
    # as the too-long remedy names one; "numba caches this file in" claimed it
    # for numba. An error naming no file, a caller's too, is told the
    # locations numba could have taken, as the too-long remedy tells them; it
    # fell past this branch to the no-locator remedy, which dropped the reason
    # and said the variable, set and writable, was one numba could not use.
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
    # is_a_cache_error admits any OSError, and a caller can pass on one built
    # from a message alone, with no errno, no strerror and no filename, or one
    # naming a file with no strerror. The reason read "(None)" and the message
    # was lost, for a source on disk; the remedies for a source not on disk
    # give the reason the same way, and nothing held them to it.
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
    # numba's IPython locator takes a file on disk in an ipykernel directory,
    # a notebook's cell file, and caches it in numba_cache under IPython's
    # cache directory, with no directory of its own per file. That location
    # too long for the file system got "that location, X", the wording for a
    # location that is none of numba's. numba imports IPython for the location
    # when it makes it and catches only OSError there, so without IPython it
    # raises ImportError at decoration, no cache error; an error a caller
    # passes on naming that location is then told it as none of numba's.
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
        # The module numba imports the location from: IPython.paths, or
        # IPython.utils.path in an older IPython, which numba falls back to
        # when IPython.paths does not import.
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
    # numba's IPython locator takes a file in an ipykernel directory, on disk
    # or not, and reads the function's source when it does: from the file, or
    # through the module's loader for a member of a .zip. The probe the check
    # compiles has no module, so for such a member inspect found no source,
    # and the check raised the OSError from reading it, which is no cache
    # error and which the remedy, taking an OSError for a source not on disk
    # as the .zip's, answered with the user's cache directory made writable,
    # a directory numba never tried. The probe's module reads the member's
    # source for inspect, as the member's own does, and the question goes on
    # to IPython's cache directory, as numba's does. zipimport finds the archive
    # below a directory named .zip, and whatever its name, an .egg as Spark
    # ships; a check that took the first part named .zip for the archive found
    # no member in the directory, or no archive in the .egg, and raised there.
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
    # What linecache holds from then on is the member's own, as any reader leaves it.
    assert linecache.getlines(py_file) == ["def f():\n", "    pass\n"]
    linecache.cache.pop(py_file, None)


@pytest.mark.parametrize("longest", [LONGEST_CACHE_FILE_NAME, 0],
                         ids=["with the package's bound on names", "with no bound, as a package may ask"])
def test_a_zip_member_in_an_ipykernel_directory_under_a_temporary_ipython_directory_takes_the_fallback(
        tmp_path, monkeypatch, longest):
    # numba's IPython locator asks IPython for its cache directory at every
    # save, and IPython, able to write neither that directory nor its own,
    # gives a new temporary directory on every call: numba makes numba_cache
    # in one and saves in another, dying with FileNotFoundError at the
    # function's first call. The check made numba_cache the same way and
    # passed the file, where before the probe's source was given it raised
    # inspect's error and the package fell back. With a bound on names the
    # file the check makes in the location raises first, under the next
    # directory IPython gives; without one the location itself is asked for.
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

    # Each call leaves a directory behind: numba asks IPython twice making
    # the directory, the check once more for the one numba would save in,
    # and the remedy twice, the second to tell a new answer from the first.
    # The check asked at every step, and left four.
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
    # A cell file on disk in an ipykernel directory reaches numba's IPython
    # locator where it is listed first, or once the in-tree and user-wide
    # locators passed the file over, and IPython giving a new temporary
    # directory on every call dies the same way a .zip member does; the
    # remedy, written for the member alone, told the file on disk to make
    # room in the temporary directory, and offered NUMBA_CACHE_DIR.
    import tempfile
    import types
    import numba
    import numba.core.caching
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "read-only" / "user-cache" / "numba")

    # numba's own user-wide locator reads the directory through its caching
    # module, the remedy through this one.
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
        # The in-tree locator cannot make __pycache__ beside the cell, the
        # user-wide one cannot make the user's cache directory.
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
    # The in-tree locator took the cell file and the __pycache__ beside it
    # refused a file: that is the remedy, whatever IPython would give for its
    # directory, which numba never asked; a remedy that asked IPython first
    # told IPYTHONDIR for an error naming the __pycache__, or a file in it,
    # as the first save's error does, which nothing told apart from one
    # naming the directory itself.
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
    # IPython is asked for its location once, to tell it from the error's,
    # and never whether it gives a new one on every call.
    assert len([path for path in tmp_path.iterdir() if path.name.startswith("tmp")]) == 1


def test_a_cell_under_a_locator_list_without_the_ipython_locator_is_told_to_list_it(tmp_path, monkeypatch):
    # numba's IPython locator alone takes a cell, so a list without it leaves
    # numba no locator for one, and the remedy offered source files on disk,
    # locators for a file on disk and a .zip, none of which takes a cell.
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
    # numba 0.68 and later read a source on disk for the stamp, and its
    # IPython locator reads the function's source in every version, so a file
    # that cannot be read raises PermissionError naming the source; the
    # remedy told it as the location numba took, to make room in or make
    # writable, and, under an IPython giving temporary directories, told
    # IPYTHONDIR, the error naming no location of another locator's.
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
    # numba's .zip locator takes a file on disk for the ".zip" in its path once
    # the in-tree and user-wide locators passed it over, takes the directory so
    # named for the archive and, from 0.68, opens it for the source, which a
    # directory refuses; the remedy told the reader to make room in that
    # directory, writable already and no cache location, where the member's
    # remedy knew the case.
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
        # Under a list with no .zip locator the error naming the directory is
        # a caller's, and is told as the location it names; the remedy spoke
        # of the list putting the .zip locator first.
        expected = (
            f"that location, {container}, where no file can be written (Is a directory): make room there, or make it "
            "writable, or NUMBA_CACHE_DIR set to a writable directory; or silence")
    elif locators:
        # Listed first, the .zip locator takes the file before any locator for
        # a file on disk; the remedy spoke of locators before it passing the
        # file over, where there were none.
        expected = (
            "numba's .zip locator, which NUMBA_CACHE_LOCATOR_CLASSES puts before every locator for a source file on "
            f"disk, takes the first part of the path ending in .zip, {container}, for an archive, and it is a "
            "directory, which numba 0.68 and later open for the source: list UserProvidedCacheLocator, with "
            "NUMBA_CACHE_DIR set, InTreeCacheLocator or UserWideCacheLocator before it, or put the file under a path "
            "with no such directory; or silence")
    else:
        expected = (
            "numba's .zip locator, reached once the locators before it passed this file over, takes the first part of "
            f"the path ending in .zip, {container}, for an archive, and it is a directory, which numba 0.68 and later "
            "open for the source: make one of those locators' locations writable, or NUMBA_CACHE_DIR set to a "
            "writable directory, or put the file under a path with no such directory; or silence")
    failure = IsADirectoryError(errno.EISDIR, "Is a directory", str(container))
    assert configurations.cache_remedy(str(py_file), failure, "silence") == expected
    if numba_version < 68 or os.name == "nt" or os.geteuid() == 0 or locators:
        return
    # The in-tree locator cannot make __pycache__ beside the source, the
    # user-wide one cannot make the user's cache directory.
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
    # The remedy asks IPython for its cache directory as numba's locator does,
    # and took OSError and ImportError for no directory; anything else escaped,
    # and the import died with IPython's error where the warning was due.
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
    # A cell file on disk in an ipykernel directory that the in-tree locator
    # took, with an error naming no file, a full disk at the first save, under
    # an IPython giving temporary directories: the error shows nothing of
    # which locator took the file, and the remedy, finding no other location
    # named, blamed IPython's temporary directory and told IPYTHONDIR.
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
    # numba's .zip locator, listed first, takes a source on disk with .zip in
    # its path and reaches nothing after it, so the in-tree locator listed
    # after it never caches beside the source; the remedy for a name too long
    # offered the __pycache__ beside the source all the same, reading the
    # whole list for the locators numba reaches.
    import numba
    from numba.core.caching import _CacheLocator
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "ZipCacheLocator,InTreeCacheLocator", raising=False)
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
    # A source on disk with .zip in its path is one the user-wide and .zip
    # locators give one path under the user's cache directory; with the
    # user-provided locator listed between them, a path too long that the
    # user-wide one passed over and the .zip one took is cured by
    # NUMBA_CACHE_DIR too, which numba tried between. The remedy read the
    # location as the user-wide locator's, the first to match, and left the
    # variable out.
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
    # A source on disk in a directory whose name holds .zip without ending in
    # it, long enough that the directory numba names after it is too long for
    # the file system: the locators that name theirs so pass the file over,
    # the in-tree one cannot cache beside the source, and numba's .zip locator
    # raises for the ".zip" in the path. The remedy read the .zip locator's
    # error alone and asked for NUMBA_CACHE_DIR, whose directory is named the
    # same way and cannot be made either.
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
    # The in-tree locator cannot make __pycache__ beside the source.
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
    # The longest name a file can have is the file system's to say, 255 nearly
    # everywhere; the remedy reads it from the nearest directory above the
    # location that exists, and 255 where the system cannot say. Nothing told
    # a limit the remedy made up from one the file system gave: the tests
    # took their expected limit from the remedy's own answer.
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

    # A statvfs that raises, for a directory the user cannot search, leaves
    # 255 too; no test held the fallback, and one answering 0 passed.
    monkeypatch.setattr(os, "statvfs", refusing, raising=False)
    assert configurations._name_limit(str(tmp_path)) == 255
    assert configurations._name_limit(None) == 255

    class NoLimit:
        f_namemax = 0

    # A file system that reports 0, no limit of its own, left 0 for the
    # limit, and every name was too long for it.
    monkeypatch.setattr(os, "statvfs", lambda path: NoLimit())
    assert configurations._name_limit(str(tmp_path)) == 255


def test_is_a_cache_error_admits_numbas_errors_by_their_text_and_no_other():
    # The line between numba's cache errors and the rest: any OSError; the
    # .zip locator's ValueError by its text; numba's no-locator RuntimeError
    # by its text; and, from numba 0.68, zipfile's KeyError for a member the
    # archive does not hold, by its text, so that a KeyError of the code's
    # own, a dictionary's, is raised as it was and not turned into the
    # warning. Nothing pinned the texts: a classifier admitting every
    # KeyError passed every test.
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
    # The probe's two lines stood in the process-wide linecache under the
    # member's name while numba picked the locator, and numba's IPython
    # locator reading another function's source from the member meanwhile, in
    # another thread, read the probe's two lines for it and raised inspect's
    # error for a function past them, "lineno is out of bounds". The probe is
    # a function of a module of its own, whose loader reads the member's
    # source from the archive as the member's own module's does, and the lines
    # linecache holds from then on are the member's own.
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
        # zipimport, for the member's own module.
        def get_source(self, name):
            return source

    member = types.ModuleType("ipykernel_123_cell")
    member.__file__ = py_file
    member.__loader__ = ReadingTheMember()
    exec(compile(source, py_file, "exec"), member.__dict__)
    monkeypatch.setitem(sys.modules, member.__name__, member)
    assert member.cell.__code__.co_firstlineno == 4
    seen = []
    real = configurations.CompileResultCacheImpl

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
    # check_cache_location takes any path-like, bytes too, and the probe's
    # code carries the name as a string, as compile makes it; the probe's
    # module and loader were given the path as os.fspath left it, bytes for
    # bytes, and the loader raised TypeError, through inspect, at the first
    # path operation asked of it for a member numba's IPython locator reads.
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
    # An error naming no file is told the locations numba could have taken,
    # with its reason, as the docstring says; the remedy for an IPython that
    # gives a new temporary directory on every call, to set IPYTHONDIR, took
    # a caller's error with a reason and no name too, a full disk's, for a
    # .zip member in an ipykernel directory, and dropped the reason. The error
    # numba's first save raises under such an IPython names the directory it
    # could not save in.
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
    # The .zip locator's location, under the user's cache directory, is among
    # them; a file on disk has the user-wide locator's there under every version.
    assert str(tmp_path / "user-cache" / "numba") in remedy, remedy


@pytest.mark.parametrize("placement", ["a .zip member", "on disk"])
def test_a_locator_of_none_of_numbas_families_listed_before_the_ipython_one_is_told_its_own_location(
        tmp_path, monkeypatch, placement):
    # A class of none of numba's families takes any file for the remedy, so
    # numba reaches none listed after it; the remedy for an IPython that gives
    # a new temporary directory on every call, to set IPYTHONDIR, asked only
    # whether a locator of IPython's family was listed before the .zip one,
    # and blamed IPython for the error such a class raised naming its own
    # location, for a file in an ipykernel directory, where the location the
    # error names is the remedy.
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
    # numba's .zip locator takes a member without making its location, and
    # the first save makes it under the user's cache directory, as the check
    # does; where a directory on the way is not writable, the error names the
    # location, which is not there, and the remedy asked for room in it, or
    # for it to be made writable, and never named the directory refusing.
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
    # A class of none of numba's families takes what its from_function takes,
    # which the remedy asks it with a probe as numba does; the remedy took
    # such a class for taking any file, so for one listed before
    # IPythonCacheLocator that passed the file over, numba going on to
    # IPython's locator, it read IPython's locator as never reached, and told
    # the error numba's first save raises under an IPython giving a temporary
    # directory per call as a location that is none of numba's, to make
    # writable, where IPYTHONDIR is the remedy.
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
    # A class of none of numba's families that takes the file, asked with a
    # probe as numba asks it, is the last locator numba reaches; the remedy
    # read the ones listed after it as reached too, and told an error naming
    # IPython's numba_cache as numba caching the file there, a location numba
    # never reached. It is a location that is none of numba's for the file.
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
    # A class of none of numba's families whose from_function raises takes
    # the file, the error being its own, as numba raises it at decoration:
    # numba reaches none listed after it. Read as passing the file over, it
    # would leave IPython's locator, listed after it, reached, and the error
    # it raised naming its own location would be told as IPython's, to set
    # IPYTHONDIR for. No test held the rule.
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
    # The user's cache directory can lie under a file, XDG_CACHE_HOME set to
    # a path through one; numba's first save, making the member's location,
    # raises for it, and the remedy named the first directory above the
    # location that is there, writable, and asked for room in it: the file
    # in the way is what to move.
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
    # numba's .zip locator listed after a class of none of numba's families
    # that takes the file, asked, is not reached, so its errors under the
    # list are a caller's: the ValueError for a path with .zip in it and no
    # part ending in it, and the IsADirectoryError for a directory named
    # .zip above a source on disk. The remedy told both as the .zip
    # locator's, the locators before it having passed the file over, where
    # one of them took it.
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
    # numba 0.68 and later read a .zip member's source by the name the
    # module's code gives, and zipfile raises for a name the archive does not
    # hold; the remedy read every such error as a module shipped as .pyc
    # alone and asked for source files, where the archive held the module's
    # source beside a .pyc compiled to name another path in it, compileall's
    # -d, which zipimport runs: the .pyc is what to recompile or leave out.
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
    # numba reads the archive for the member's stamp, and from 0.68 opens it
    # for the member's source; one it cannot read raises naming the archive,
    # and the remedy told the archive as a location that refused a file, to
    # make room in or make writable, which reads nothing.
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
    # The error numba's first save raises under an IPython giving a temporary
    # directory per call names a numba_cache under that directory; one naming
    # the archive a member's code names, where that has since moved, is the
    # stamp's, told as the archive's, and the remedy for the temporary
    # directory took it, for a member in an ipykernel directory, and said
    # IPYTHONDIR. In the flow itself inspect reads no source from an archive
    # that is gone, and that error is told first; a caller's error can name
    # the archive.
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
    # numba's .zip locator takes a path with ".zip" in it, lower-case, so an
    # archive named APP.ZIP, or an .egg, which Python imports from all the
    # same, leaves numba no locator; the remedy told the reader to import
    # from a .zip holding the source files, which they did, and never named
    # the extension numba does not read. An .egg under a directory with
    # ".zip" in its name is taken by that locator, which finds no part ending
    # in .zip and raises its ValueError; the remedy held the rename back for
    # the ".zip" in the path, and told the same.
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
    # The archive above a file was found through pathlib.Path, which makes a
    # path of the platform os.name says: set to Windows's, as the tests for
    # numba's separator bug set it, Python before 3.12 refuses to make one on
    # Linux, and the remedy for such a member, finding the archive for the
    # rename it offers, died on it. The path is read by the separator in use.
    import numbox.core.configurations as configurations
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("pkg/mod.py", "x = 1\n")
    py_file = str(archive / "pkg" / "mod.py")
    monkeypatch.setattr(os, "name", "nt")
    assert configurations._archive_above(py_file) == (str(archive), "pkg/mod.py")
    assert configurations._archive_above(str(tmp_path / "elsewhere" / "mod.py")) is None


def test_a_relative_members_moved_archive_under_a_temporary_ipython_directory_is_the_archives(tmp_path, monkeypatch):
    # The file numba names can be relative, run from the archive's directory,
    # and the error naming the archive above it relative too; the remedy for
    # an IPython giving a temporary directory per call compared the file as
    # given against the archive's name made absolute, matched nothing, and
    # said IPYTHONDIR, where the archive's remedy is the one.
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
    # The file numba names can be relative, run from the archive's directory,
    # while the error names the archive absolute, as the stamp's does; the
    # archive's remedy compared the file as given against the name, matched
    # nothing, and told the archive that is gone as a location refusing a
    # file, to make room in or make writable.
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
    # The in-tree locator caches beside the source under no directory named
    # after the source's, so a name too long for the user's cache directory
    # is told to list it before the locator that took the file; the remedy
    # named numba's .zip locator for any, where a class of none of numba's
    # families, listed before the in-tree one, took the file and the .zip
    # locator was not listed at all.
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
    monkeypatch.setattr(configurations, "_name_limit", lambda location: 20)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("x = 1\n")
    subpath = _CacheLocator.get_suitable_cache_subpath(str(py_file))
    named = str(tmp_path / "user-cache" / "numba" / subpath)
    failure = OSError(errno.ENAMETOOLONG, "File name too long", named)
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert "ZipCacheLocator" not in remedy, remedy
    # The name is counted as the file system counts it, in UTF-16 units on
    # Windows and macOS, bytes elsewhere; the test said bytes everywhere.
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
    # numba names the directory it caches a file in under NUMBA_CACHE_DIR by
    # hashing the source's directory as UTF-8, and raises UnicodeEncodeError
    # for one that is not, no cache error; the remedy offered the variable
    # for such a source, and set as advised, the import died where the
    # warning had stood.
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
        # numba's no-locator error here is the in-tree locator passing an
        # unwritable __pycache__ over before the user-wide one raised, so
        # that directory writable is a cure too, hashing no path.
        assert remedy == (
            "numba hashes the source's directory as UTF-8 for the directory it caches this file in under "
            f"NUMBA_CACHE_DIR or the user's cache directory, and {directory} is not: put the file under a UTF-8 "
            f"path, or make the __pycache__ beside the source, {directory / '__pycache__'}, writable, where "
            "InTreeCacheLocator caches it hashing no path; or silence"), remedy


def test_a_zip_member_in_an_ipykernel_directory_that_is_not_utf8_is_told_so(tmp_path, monkeypatch):
    # numba's IPython locator reads a member's source through zipimport,
    # which decodes it as UTF-8 and raises for one that is not, as numba then
    # does at decoration; the probe's loader read it the same way, and the
    # UnicodeDecodeError came out of the check as its own, no cache error, so
    # the import died where the warning should have stood. inspect's error
    # tells such a member, and the remedy reads why.
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
    # An entry of NUMBA_CACHE_LOCATOR_CLASSES whose module raises on import,
    # anything, not importlib's error alone, numba raises as its own at
    # decoration, and the check before any remedy; the remedy, reached for
    # an error of the cache's, imported the module for the entry and raised
    # the module's error in place of the remedy.
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
    # A path object built from an empty string names the working directory,
    # which the remedy tells as a location that is none of numba's; for a
    # member under that directory it stood as a path above the file, and a
    # FileNotFoundError naming it was told as the member's moved archive.
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
    # The error numba's first save raises under an IPython giving a temporary
    # directory per call names the numba_cache it could not save in, as the
    # check's does; the remedy for that case took any error naming a location
    # that is none of numba's, and told IPYTHONDIR for a cell file on disk
    # that the in-tree locator took, IPython never reached, on a caller's
    # error naming an unrelated place.
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
    # The end-to-end .zip tests skip where the import met numba issue 10889,
    # numba 0.68.0's on Windows alone; keyed on the remedy's text in stderr,
    # the skip hid that remedy wrongly given on Linux, to a module shipped
    # as .pyc alone for one, as a skip. The skip holds on Windows alone.
    stderr = f"RuntimeWarning: numba cannot cache numbox here (...). {windows_path_sep_bug}, x, which no archive holds"
    monkeypatch.setattr(os, "name", "posix")
    with pytest.raises(AssertionError, match="the Windows separator remedy on posix"):
        _skip_for_windows_separator_bug(stderr)
    monkeypatch.setattr(os, "name", "nt")
    with pytest.raises(pytest.skip.Exception):
        _skip_for_windows_separator_bug(stderr)
    _skip_for_windows_separator_bug("RuntimeWarning: numba cannot cache numbox here, another remedy")


def test_an_ipython_that_gives_its_directory_once_and_then_raises_is_not_read_as_temporary(tmp_path, monkeypatch):
    # The remedy asks IPython for its cache directory twice to tell a new
    # temporary directory from a steady one; an IPython that answers once and
    # raises on the second call gave no second directory, and the remedy,
    # reading the raise as a new answer, blamed a temporary directory.
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
    # inspect's error for a source it cannot read has no errno and no name,
    # and so can a caller's, OSError("Disk quota exceeded on /home"); the
    # remedy read every such error for a file numba's IPython locator takes
    # as inspect's and told the reader to make the file readable. inspect's
    # are known by their words.
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
    # A traceback through a .zip member, or inspect, leaves the member's lines
    # in the process-wide linecache under its name, and inspect reads those
    # for the probe before it asks the module's loader; the check gave the
    # probe's source to inspect under that name instead for the time numba
    # picked the locator, and owed what was there back.
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
    # inspect reads a file on disk itself, and asks the probe's module's
    # loader, which reads a .zip member from the archive, for no file on disk.
    # The probe's source, given to inspect for every file once, stood in for the
    # real file's lines in the process-wide linecache while numba picked the
    # locator, and a locator reading another function's source from the file
    # meanwhile, as numba's IPython locator reads the function's own, read the
    # probe's two lines instead. numba's own order tries the in-tree locator
    # before the IPython one, which takes a cell file on disk only listed
    # ahead of it.
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
    # The probe's source was given under a lock, one check at a time, so that
    # each check put back what it had found in the process-wide linecache,
    # and a second check waited at the lock while the first held it. Each
    # probe reads the member's source through a module of its own now, and
    # leaves the member's lines, so there is nothing to put back and nothing
    # to wait for.
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
    # The second check goes in while the first is still there.
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
    # numba's IPython locator takes a member of a .zip in an ipykernel directory
    # before the .zip locator is reached, and caches it in numba_cache under
    # IPython's cache directory, which the error names, or a file numba writes
    # in it. The remedy, taking every OSError for a source not on disk as the
    # .zip's, told that location as the user's cache directory, with what moves
    # that directory, where it is IPython's and the remedy is the one a cell
    # file on disk gets there, with no NUMBA_CACHE_DIR to offer for a source
    # not on disk.
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
    # numba's IPython locator takes a file in an ipykernel directory alone, so
    # a .zip member anywhere else is never cached in IPython's cache directory;
    # the remedy asked IPython for that directory all the same, to compare the
    # error's path against it, and IPython, with its own directory unwritable,
    # under a read-only home, warned on every import of the package and left a
    # temporary directory behind. The directory is asked for where numba's
    # IPython locator takes the file and is listed.
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
    # The location is there and refuses, as a directory made can; one not
    # there is told the directory it cannot be made under as well.
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
    # numba's appdirs puts the user's cache directory under XDG_CACHE_HOME or
    # HOME on Linux, under HOME alone on macOS, and asks the system on Windows,
    # so "through XDG_CACHE_HOME or HOME" sent a macOS reader to a variable that
    # moves nothing, for a source on disk and for a .zip alike.
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
    # numba takes NUMBA_CACHE_LOCATOR_CLASSES as the locators and
    # their order, each entry a class of numba's caching module by its name or
    # its dotted path. NUMBA_CACHE_DIR is an alternative to the __pycache__
    # numba took only where numba tries the variable before that: a set one
    # was passed over, an unset one would be taken. The IPython and .zip
    # locators take no plain file on disk, so one of them ahead changes
    # nothing, where the first entry's name alone said another locator was
    # first. numba's InTreeCacheLocatorFsAgnostic caches in the __pycache__ as
    # its parent does, and keyed by name alone it was no in-tree locator, so
    # its too-long __pycache__ got "that location" and no cure.
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


# Each case: whether NUMBA_CACHE_DIR is set, NUMBA_CACHE_LOCATOR_CLASSES, and
# the remedy before "or silence".
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
    # numba raises "no locator available" where it passed every location over.
    # "Set NUMBA_CACHE_DIR to a writable directory" was the answer whatever the
    # setting: to a NUMBA_CACHE_DIR too deep for numba to make its directory in,
    # where a writable one as deep cures nothing, and to a locator list numba
    # reads from NUMBA_CACHE_LOCATOR_CLASSES without the
    # user-provided locator, where numba never reads the variable. With that
    # locator anywhere in the list numba reads the variable once the locators
    # before it have passed, so its place in the list changes nothing here. A
    # list of numba's IPython and .zip locators alone has no locator for a
    # plain file on disk, and "make one of those locations writable" asked for
    # what no listing can give.
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
    # numba's .zip locator takes any path with ".zip" in it before every
    # locator after it, and caches one with a part ending in ".zip", a
    # directory of that name included, under the user's cache directory.
    # Listed first it leaves
    # NUMBA_CACHE_DIR unread for such a path, where the remedy, holding that
    # locator to take no file on disk, left it out of the order and offered
    # the variable.
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


# Each case: whether NUMBA_CACHE_DIR is set, NUMBA_CACHE_LOCATOR_CLASSES, and
# the remedy before "or silence".
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
    # numba's .zip locator takes a path with ".zip" in it and raises ValueError
    # where no part of the path ends in .zip. numba tries the locators before
    # it and none after, so the error means the ones before it passed the file
    # over: in numba's own order that is every locator for a file on disk, and
    # the no-locator remedy applies; listed first it takes the file first, and
    # the remedy offered NUMBA_CACHE_DIR, which numba never reaches; listed
    # after one locator for a file on disk and before the user-provided one,
    # it raised before numba read NUMBA_CACHE_DIR, and the remedy, answering
    # the error only where the .zip locator came first, said the variable, set
    # and writable, was one numba could not use. The user-provided locator
    # takes a file only with NUMBA_CACHE_DIR set, so listing it is asked for
    # with the variable set, as the no-locator remedy asks; without that, the
    # reader who listed it first saw the same error again.
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
    # A list with no locator that takes a source file on disk leaves numba no
    # location for it, and numba's error is the no-locator one, which is told
    # the list. An OSError a caller passes on under such a list was answered
    # as if numba had taken a location: unnamed, "the location numba took"
    # with no location to list after it, and named, that location at a
    # shorter path, though the list keeps numba from caching the file at all.
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
    # numba's user-provided locator takes a file only with NUMBA_CACHE_DIR set,
    # so a list of that locator alone leaves numba no locator while the
    # variable is unset, and numba's error is the no-locator one, told to set
    # the variable. An OSError a caller passes on under that list counted the
    # locator as taking the file, and was told "the location numba took" with
    # no location after it, or the location it named at a shorter path, where
    # numba took none.
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
    # The remedy for an archive numba does not cache offers the source files
    # on disk, which the three locators for a file on disk take, or a .zip
    # holding them, which the .zip locator alone takes. It offered both under
    # any list, so under one without the .zip locator it offered a .zip numba
    # then had no locator for, as under one without a locator for a file on
    # disk it offered an install numba would not cache either; and the
    # user-provided locator, alone of the three in the list, takes nothing
    # without NUMBA_CACHE_DIR, so unset it is asked for with the install,
    # where the in-tree or the user-wide one alone takes a file on disk as
    # it is, and the install is offered as it is.
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
    # numba caches a .zip through its .zip locator alone, so a list that leaves
    # that locator out gives numba no locator for a source in a .zip, and the
    # error is the no-locator one. The archive remedy, which reads the list for
    # a source on disk only, told the reader to import from a .zip holding the
    # source files, which they had done, and never named the list.
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
    # numba's user-wide locator alone takes a frozen application, whose source
    # is not on disk, so a list that leaves it out gives numba no locator and
    # the error is the no-locator one. The frozen remedy, reading no list,
    # asked for the user's cache directory to be made writable, which it was
    # and which numba never tried.
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
    # numba's .zip locator takes a file without trying its location, so a
    # list with it before the IPython one leaves that one unreached for a
    # .zip member in an ipykernel directory, and numba caches the member in
    # the user's cache directory; the remedy answered an error a caller passes
    # on naming IPython's numba_cache with IPython's cache directory all the
    # same, a directory numba never used. And numba's IPython locator writes
    # its files in numba_cache itself, with no directory under it, so a path
    # under one is none of numba's, and was told as the .zip's directory to
    # put at a shorter path through what moves the user's cache directory,
    # which moves nothing of it. A location that is none of numba's is told
    # as the error names it, as for a source on disk, and with nothing of
    # where numba caches the file, which it is none of.
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
    # numba's IPython locator passes a location it cannot make or write over,
    # and the .zip locator listed after it then takes the member and caches
    # it in the user's cache directory: an error naming that location is the
    # .zip's, under numba's own order and under one that lists the IPython
    # locator first alike.
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
    # numba's IPython locator reads the function's source when it takes the
    # file, through the probe's module's loader for a .zip member, which
    # leaves the member's lines in the process-wide linecache; the probe's own
    # source stood there instead, for every file not on disk, while numba
    # picked the locator, and a reader in another thread, formatting a
    # traceback through that file meanwhile, got the probe's two lines for it,
    # for a .zip member in no ipykernel directory too, which that locator
    # never takes. The linecache is untouched for a file no locator of that
    # family numba reaches takes.
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
    # numba's .zip locator takes a path with .zip in it, a file on disk under
    # a directory named so included, without trying its location, so numba
    # reaches no locator listed after it. The remedy matched the error's path
    # against the locations of those too, and an error naming the __pycache__
    # beside the source, under a list with the in-tree locator after the .zip
    # one, was told the package installed at a shorter path, for a location
    # numba never used. The path is matched against the locations of the
    # locators numba reaches alone, and any other is told as the error names
    # it.
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
    # An error that names no file, or an empty name, is told the locations
    # numba could have taken; a path object has no empty name, pathlib
    # reading an empty string as the working directory, ".", so one built
    # from it names that directory, which is none of numba's, and is told as
    # named, as the string "." is.
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
    # numba resolves a dotted entry of NUMBA_CACHE_LOCATOR_CLASSES to whatever
    # the module holds under the name and fails, for want of from_function,
    # at a function it decorates where it reaches that entry; the check
    # read the entry as numba does and asked issubclass of it first, which
    # raised a TypeError naming neither the entry nor the variable, ahead of
    # numba's error. An entry that is no class is no locator here, numba's
    # own error is raised as it was, and the remedy reads the list without it.
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
    # numba resolves a plain entry of NUMBA_CACHE_LOCATOR_CLASSES
    # as a name of the caching module and refuse any other before a location
    # is tried; the remedy read the two names of numba's base
    # classes as those classes and answered for a list numba refuses, with
    # one of their locations to make writable.
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
    # numba imports the module of a dotted entry and lets the ValueError for
    # an empty module path, ".InTreeCacheLocator", through at the first
    # function it decorates; the remedy, reading the list, raised that
    # ValueError itself instead of answering an error a caller passes on
    # under such a list.
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
    # numba's IPython locator asks IPython for its cache directory when it
    # makes its location, and passes the file over on OSError there as it does
    # a location it cannot make; IPython raises one where it can make the
    # directory nowhere. The remedy, asking IPython the same for the location
    # to match the error's path against, raised that error instead of
    # answering, for numba's no-locator error and for the .zip's alike.
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
        # The location is there and refuses, as a directory made can; one not
        # there is told the directory it cannot be made under as well.
        location.mkdir(parents=True)
        failure = OSError(errno.EACCES, "Permission denied", str(location))
        remedy = configurations.cache_remedy(str(py_file), failure, "silence")
        assert remedy == (
            "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has "
            "no effect here, because the source is not a file on disk: no file can be written in that directory, "
            f"{location} (Permission denied), so make room there, or make it writable, or silence"), remedy


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
    # numba's .zip locator alone takes a .zip member in no ipykernel directory,
    # and its user-wide one alone a frozen application's file, so a list
    # without the one leaves numba no locator for the file, and its error is
    # the no-locator one, told to list it. An OSError a caller passes on under
    # such a list, naming a location or none, was answered as a .zip's or a
    # frozen application's: that numba caches the file in the user's cache
    # directory, and that location, or that directory, to make writable or put
    # at a shorter path, while numba took none. It is told the list, as
    # numba's no-locator error is, as for a source on disk.
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
    # numba's first save raises an OSError that names no file where a write
    # fails, as on a full disk, and a caller can pass one on. Which location
    # numba took cannot be told from it, so the remedy for a source on disk
    # lists the ones numba could have taken; the remedy for a source not on
    # disk told the user's cache directory, which for a .zip member in an
    # ipykernel directory under numba's order, cached through its IPython
    # locator in numba_cache under IPython's cache directory, is a directory
    # numba never used, and for a .zip or a frozen application is above the
    # directory numba took. It is told the locations numba could have taken,
    # in numba's order and each once, and where IPython's is among them
    # nothing of the user's cache directory as where numba caches the file.
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
    # numba's IPython locator takes a .zip member in an ipykernel directory
    # and imports IPython for its location, catching OSError alone, so with
    # IPython not importable numba raises ImportError at decoration, no cache
    # error, and caches the file nowhere; an error a caller passes on naming
    # IPython's location is told it as none of numba's. The remedy opened with
    # where numba caches a .zip, or a frozen application, the user's cache
    # directory, which numba did not cache this file in.
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
    # numba's IPython locator takes a cell, "<ipython-...>", by its name, as it
    # takes a file in an ipykernel directory, reads its source from linecache,
    # where IPython holds it while it runs the cell, and caches it in
    # numba_cache under IPython's cache directory: the check leaves linecache
    # to the locator, which reads the cell as numba does, and the remedy is
    # the one a cell file on disk gets. Nothing held the remedy to the cell:
    # it told IPython's location as none of numba's.
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
    # numba's user-wide locator passes a location it cannot make over, one too
    # deep for the file system as an unwritable one, and numba's error for a
    # frozen application is then its no-locator one, which gives no reason;
    # the remedy asked for the user's cache directory to be made writable
    # alone, which cures no path too long.
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
    # numba's .zip locator takes a path with ".zip" in it without trying its
    # location and raises where no part of the path ends in .zip, so a frozen
    # application's file at such a path gets that error where the list puts
    # the .zip locator before the user-wide one, which alone takes the file,
    # or once the user-wide one passed its location over. The remedy answered
    # with the user's cache directory to make writable, a directory numba never
    # tried under such a list; it is the user-wide locator to list before the
    # .zip one, or the location that one passed over, as for a source on disk.
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
    # numba's IPython locator reads the function's source, and for a file on
    # disk inspect reads it from the file itself and raises OSError, naming no
    # file and with no errno, where it reads no source there, a file it cannot
    # read or, before Python 3.13, an empty one, which linecache gives a blank
    # line from then on; the check raises that error before that locator
    # tries its location, the locators numba tries before it having passed
    # the file over, and is_a_cache_error admits it as it admits every
    # OSError. The
    # check's docstring said every OSError from it names the location numba
    # picked, and the remedy told the location numba took, which it had not,
    # as refusing a file.
    import types
    import numba
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    listed = "IPythonCacheLocator" if order == "the IPython locator alone" else ""
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", listed, raising=False)
    if order == "numba's order":
        # The in-tree and user-wide locators try their locations first and
        # pass the file over, the cell's directory unwritable and the user's
        # cache directory under a file.
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
    # An error with no errno that names a file is a caller's, not inspect's,
    # and is told the location it names.
    named = OSError("could not get source code")
    named.filename = str(tmp_path / "elsewhere")
    remedy = configurations.cache_remedy(str(cell), named, "silence")
    assert remedy.startswith(f"that location, {tmp_path / 'elsewhere'}, where no file can be written"), remedy


def test_a_source_not_on_disk_under_a_directory_named_for_a_zip_is_in_no_zip(tmp_path, monkeypatch):
    # numba's .zip locator takes a path with ".zip" in it and finds the archive
    # by a part of the path that ends in .zip, so a source under a directory
    # named my.zipped is in no .zip: under a list without that locator, numba's
    # no-locator error is told the source files on disk or a .zip holding them,
    # with the locator each needs, and under numba's order the .zip locator
    # takes the path and raises for the archive it finds no part of it to be,
    # which is told the same. Nothing held the remedy to the part: with the
    # substring enough, the no-locator error was told to list the .zip locator
    # alone, which numba then raises through, and the suite passed.
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
    # numba imports the module of a dotted entry and lets importlib's TypeError
    # for a relative module path, "..InTreeCacheLocator", through at the first
    # function it decorates, as it lets the ValueError for an empty one; the
    # remedy, reading the list, caught the ValueError and raised the TypeError
    # itself instead of answering an error a caller passes on under such a list.
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
    # numba fails on a listed class with no from_function, its own CacheImpl or
    # any other, as it fails on an entry naming no class, for want of that
    # method at a function it decorates where it reaches the entry; the
    # remedy left out the entry naming no class and kept the class, as a
    # locator of its own that takes any file on disk, so the remedy with the
    # entry lost the NUMBA_CACHE_DIR offer, and the entry alone was told to
    # make one of its locations writable, a list with no locator at all.
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
    # numba's IPython locator takes a .zip member in an ipykernel directory
    # before its .zip locator can, so an error naming no file for the member
    # may be IPython's location's, and the remedy says nothing of the user's
    # cache directory as where numba caches the file; it did so only where
    # IPython gave its directory, and with IPython not importable, which
    # numba raises ImportError for at decoration, opened with that directory
    # as where numba caches a .zip, or a frozen application, against its own
    # comment.
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
    # numba's IPython locator imports IPython for its location and lets the
    # ImportError through at decoration, so with IPython not importable numba
    # caches a cell file nowhere and raises no cache error; an error a caller
    # passes on naming IPython's location is told it as none of numba's where
    # another listed locator has a place for the file, and the list where
    # none has, as numba's no-locator error is told it. The docstring promised
    # the location as named under any list.
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
    # NUMBA_CACHE_LOCATOR_CLASSES exists for a locator of the reader's own, a
    # class of none of numba's families, with a from_function and a place of
    # its own that the remedy cannot know; numba caches through it, and the
    # check's error names its location. The remedy read the class as taking
    # files on disk alone and as having no place, so a path too long on disk
    # was told to make one of the list's locations writable, with the cause
    # dropped, and a .zip member's refused file to list the .zip locator, with
    # the location dropped; upstream main named the cause for the one and the
    # location for the other. The class takes any file for the remedy, and the
    # location the error names, which no locator of numba's gives the file, is
    # told as named, with the cause; an error naming no file is told the
    # location numba took as the class's own, with no path to name, among
    # numba's locators' places where those are reached.
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
            # OurLocator takes the file, asked, so the in-tree locator listed
            # after it is not reached: the remedy named its __pycache__ too.
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
    # A locator of the reader's own listed alone can pass a frozen
    # application's file over, and numba's error is then its no-locator one,
    # which names nothing; the remedy, counting the class as having a place
    # of its own, answered with the user's cache directory as one numba could
    # not use, a directory numba never tried, where under a list of numba's
    # own locators without the user-wide one it asks for that locator.
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
    # os.stat(fd) and the other calls that take a descriptor raise OSError with
    # the integer as the file name, and is_a_cache_error admits every OSError;
    # the remedy read every name as a path and raised TypeError on the integer
    # instead of answering. A name of another kind than str, bytes or a path
    # object is none, and the error is told the locations numba could have
    # taken, as one naming no file is.
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
    # numba's IPython locator passes its location over where it cannot make or
    # write in it, or where IPython raises for its directory, and a cell, which
    # no other locator takes, then gets numba's no-locator error, as does a
    # .zip member in an ipykernel directory under a list with no .zip locator
    # after the IPython one; the remedy answered with the archive remedy, the
    # source files on disk or a .zip holding them, which touch no cell, and
    # answered so again for a cell under a list of the IPython locator and a
    # class of none of numba's families, which passed the cell over too and
    # is named as having.
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
        # IPython holds a cell's source in linecache while it runs it, and
        # numba's IPython locator reads it there.
        source = "def f():\n    pass\n"
        monkeypatch.setitem(linecache.cache, py_file, (len(source), None, source.splitlines(True), py_file))
    else:
        # inspect reads a .zip member's source through the module's loader,
        # which the probe has none of, so the check gives it the probe's
        # where the archive holds the member's source.
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
    # numba reads the source's stamp at decoration, the archive's for a .zip
    # member, and the check reads it as numba does, so for a member whose code
    # names an archive since moved it raises FileNotFoundError naming that
    # archive, no location of numba's, before any location is tried; the
    # check's docstring excepted inspect's error alone from naming the
    # location numba picked.
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
    # Members compiled to name the archive's path now cache under numba 0.67
    # and earlier; from 0.68 the archive must hold the sources too.
    recompile = "" if numba_version >= 68 else "compile the archive's .pyc members to name its path now, or "
    assert remedy == (
        f"the module's code names {archive}, which is not there: its .pyc was compiled to name that path, so "
        f"{recompile}ship its source files; or silence"), remedy


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
    # numba names the directory it caches a file in, under NUMBA_CACHE_DIR or
    # the user's cache directory, after the source's directory, by its name
    # and a hash of its path, forty-one bytes more, and a name longer than the
    # file system takes can be made at no path: numba passes the location
    # over, or the first save dies there for a .zip member, and the remedy
    # asked for the user's cache directory at a shorter path, or a writable
    # NUMBA_CACHE_DIR at a short path, which cure nothing there. It tells the
    # name and its length, and asks for the file in a directory of a shorter
    # name, with the __pycache__ beside a source on disk, which the in-tree
    # locator names after nothing, made writable or that locator listed; a
    # name that just fits is told as before. The name is counted as the
    # file system counts it, in bytes, or in UTF-16 units on Windows and
    # macOS, where a name of CJK characters that fits was counted in bytes
    # and told too long; Windows refuses a name too long as an invalid one,
    # errno EINVAL, and the check runs whatever the errno. A name of astral
    # characters, each one code point, two UTF-16 units and four bytes, is
    # too long in units where it fits in code points: the count by units
    # was held by no test, and one by code points passed them all.
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
            refused = ((errno.EINVAL, "Invalid argument") if "invalid name" in case
                       else (errno.ENAMETOOLONG, "File name too long"))
            named = os.path.join(cache_dir or str(tmp_path / "user-cache" / "numba"), subpath)
            failure = OSError(*refused, named)
        elif "a caller's no-locator error" in case:
            # numba's .zip locator takes the path without trying its
            # location, so numba raises no no-locator error here, and a
            # caller's reached a class listed after that locator, unreached,
            # for its location and raised KeyError.
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
        # The user-provided locator alone, with the variable unset, took no
        # location and measured no name; the variable it takes nothing
        # without is named beside the shorter name it would then need.
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
    # numba names the directory it caches a file in under NUMBA_CACHE_DIR or
    # the user's cache directory with a SHA-1 of the source's directory path
    # encoded as UTF-8, and raises UnicodeEncodeError for a path holding bytes
    # that are not, decoded with surrogates, where a locator that names its
    # directory so reaches the file; the remedy computed that name for every
    # file and raised the same where numba hashed no path, an .egg member no
    # locator takes and a source on disk under the in-tree locator alone, and
    # the import died where upstream main warned.
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
    # numba's IPython locator reads a cell's source from linecache, where
    # IPython holds it while it runs the cell, and inspect raises, with no
    # errno and no name, for a cell linecache does not hold, before the
    # locator tries its location; the check gave the probe's source to every
    # file the locator takes by name and returned for such a cell, where numba
    # raises at decoration, and the remedy told the location numba took as
    # refusing a file, or, under an IPython giving temporary directories,
    # blamed the directory the locator never tried.
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
    # numba's IPython locator reads the function's source through the module's
    # loader for a .zip member, and zipimport gives none for a member compiled
    # to .pyc alone, so numba raises inspect's error at decoration; the check
    # gave the probe's source to every member a locator of that family takes
    # and returned, where upstream main's check raised as numba does, and the
    # remedy told the location numba took as refusing a file where the
    # archive's remedy, its source files to ship, is the one, and, under an
    # IPython giving temporary directories, blamed the directory the locator
    # never tried.
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
    # linecache kept the probe's loader under the member's name, to ask again
    # for a file whose source it found none of; the check takes it out.
    assert py_file not in linecache.cache
    assert temporary or not ipython_dir.exists()
    assert configurations.is_a_cache_error(raised.value)
    remedy = configurations.cache_remedy(py_file, raised.value, "silence")
    if name.endswith(".egg"):
        # An archive numba's .zip locator never takes is told to be renamed
        # to end in .zip, with its source files in it.
        assert remedy == (
            "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, install numbox "
            f"with its source files on disk, unpacked from any archive, or rename the archive, {archive}, to end in "
            ".zip, lower-case, which numba's .zip locator looks for in the path, with its source files in it, so that "
            "numba caches it in the user's cache directory; or silence"), remedy
    else:
        assert remedy == f"{ARCHIVE_REMEDY}; or silence", remedy


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
    # The remedy that asks for the user-wide locator to be listed is for
    # numba's no-locator error, which a frozen application gets under a list
    # without that locator. A .zip's error names the location numba took
    # through its .zip locator, which such a list can hold, and the remedy is
    # that location made writable or put at a shorter path, as under numba's
    # own order; held to any error under such a list, the frozen remedy would
    # tell a .zip's reader to list a locator numba takes no .zip through.
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
    # The location is there and refuses, as a directory made can; one not
    # there is told the directory it cannot be made under as well.
    location.mkdir(parents=True)
    remedy = configurations.cache_remedy(py_file, OSError(*error, str(location)), "silence")
    told = told.format(location=location, moved_through=configurations._moved_through())
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        f"effect here, because the source is not a file on disk: {told}, or silence"), remedy


@pytest.mark.parametrize("refused", list(A_ZIP_REFUSED))
def test_a_zip_error_naming_a_file_numba_writes_in_its_location_is_told_the_location(tmp_path, monkeypatch, refused):
    # The package's own check names the location numba took for a .zip, a
    # directory of its own under the user's cache directory; numba's first
    # save names the file it was writing there, under a temporary name, and a
    # caller can pass that error on. The remedy quoted the file as the
    # directory to make writable or to put at a shorter path, which the
    # remedy for a source on disk, matching the file's directory too, did not.
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
    # numba's user-wide locator takes a frozen application's file, which is
    # not on disk, besides any file on disk, and caches it in a directory of
    # its own under the user's cache directory, as the .zip locator does a
    # .zip; numba's first save names the file it was writing there, and the
    # remedy is the directory, to make writable or put at a shorter path, as
    # for a .zip. Nothing held the remedy to that locator taking the file:
    # with the frozen application left to the locators for a file on disk, no
    # locator took it, and the error was answered with the list, the archive
    # remedy, instead of the location numba took.
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
    # numba's errors name the file they are about, and so does the package's
    # own check, but a caller can pass on a FileNotFoundError that names none.
    # The check for a moved archive, which asks whether the source lies under
    # the file the error names, added os.sep to that name before asking, and
    # raised TypeError on None instead of returning a remedy.
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
    # No locator takes a source neither on disk, in a .zip nor a frozen
    # application's, so the error gets the remedy numba's no-locator one does.
    assert remedy == f"{ARCHIVE_REMEDY}; or silence", remedy


def test_a_zip_member_whose_source_the_archive_does_not_hold_is_told_the_source_files_naming_the_member(
        tmp_path, monkeypatch):
    # numba 0.68 and later read a member's source from the archive for the
    # stamp, and zipfile's KeyError names the member it did not find, a module
    # shipped as .pyc alone: the error was not the cache's, so the import died
    # at the first decorated function, and the archive remedy told the source
    # files without the member.
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
    # numba 0.68.0 builds the member's name with the platform's separator, so
    # on Windows no archive holds it (numba issue 10889, fixed in 0.68.1): the
    # archive's user was told to add source files that were there.
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
    # zipfile quotes the name as Python does, each backslash doubled.
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
    # The separator is in the name numba 0.68.0 builds for every member on
    # Windows, a module shipped as .pyc alone among them, and that one numba
    # 0.68.1 cannot cache either, the archive holding no source under the name
    # it reads; the remedy read the backslash alone and promised that release.
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
    # numba takes the first part of the path ending in .zip for the archive, a
    # directory so named above it included, and from 0.68 opens it to read the
    # member's source, which the directory refuses; the remedy told the reader
    # to make room in the directory, or make it writable.
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


SUBCLASSED_LOCATORS = (
    "from numba.core.caching import UserProvidedCacheLocator, ZipCacheLocator\n"
    "class OurUserProvided(UserProvidedCacheLocator):\n    pass\n"
    "class OurZip(ZipCacheLocator):\n    pass\n"
)

# Each case: NUMBA_CACHE_LOCATOR_CLASSES, whether NUMBA_CACHE_DIR is set, the
# source's directory, which location the error names ("none" for numba's
# no-locator error), and the remedy before "; or silence" or ", or silence".
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
    # NUMBA_CACHE_LOCATOR_CLASSES takes a dotted path to any class, and a
    # subclass of one of numba's locators caches where its parent does; numba
    # takes it as it takes the parent. The remedy knew a subclass of the
    # in-tree locator for one, and matched the user-provided and .zip
    # locators by identity, so their subclasses got "that location, X" with no
    # cure, and the variable the subclass reads counted as unread.
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
    # check_cache_location takes any path-like through os.fspath, and the
    # remedy took one too until it asked '".zip" in py_file' of it, which a
    # Path cannot answer: TypeError for every source on disk under numba's own
    # order, and AttributeError at 'py_file.startswith' for a source not on disk.
    # A bytes path, which check_cache_location takes as well, raised TypeError
    # at the first 'startswith' asked of it with a str, os.fspath passing bytes
    # through as bytes.
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
    # An error names the file it refused as the call that raised it was given
    # it, bytes for a call given bytes, or as a caller built it, a path object
    # too, and the remedy read the name as a string: bytes matched none of
    # the locations, which the remedy writes as strings, and were told as a
    # location that is none of numba's, and for a source not on disk the
    # check for a moved archive, adding os.sep to the name, raised TypeError
    # on bytes and on a path object alike.
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
    # A package built on numbox puts the question for its own files and takes
    # the remedy from cache_remedy; the two remedies that tell the reader to
    # install a package again named numbox whichever package had asked.
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
    """Each function's qualified name as numba names its cache files, angle brackets dropped, with its line."""
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
    # The probe reserves LONGEST_CACHE_FILE_NAME bytes in a location for the
    # files numba writes there: the module's stem, the function's qualified
    # name, a line number, the interpreter tag with its abiflags and an index
    # number, written under a 21-byte temporary name. Every function of the
    # package must fit, decorated or not, and so must the builder's generated
    # kernel, anchored to its file under a name of 64 hex characters.
    from numbox.core.configurations import LONGEST_CACHE_FILE_NAME, _OVERLOAD_DIGITS
    tag = f".py{sys.version_info[0]}{sys.version_info[1]}t.{'9' * _OVERLOAD_DIGITS}.nbc.tmp.0123456789abcdef"
    names = [f"{path.stem}.{qualname}-{lineno}{tag}"
             for path in sorted((REPO / "numbox").rglob("*.py"))
             for qualname, lineno in _function_names(ast.parse(path.read_text(encoding="utf-8")))]
    names.append(f"builder._make_{'0' * 64}-9999{tag}")
    longest = max(names, key=len)
    assert len(names) > 500
    assert len(longest) <= LONGEST_CACHE_FILE_NAME, longest


ANOTHER_ERROR_AT_THE_CACHE_SET_UP = (
    "import numba.core.caching as caching\n"
    "def refuse(self, py_func):\n"
    "    raise RuntimeError('a locator of another kind refused')\n"
    "caching.CompileResultCacheImpl.__init__ = refuse\n"
    "import numbox.core.configurations\n"
)


def test_an_error_that_is_not_the_caches_is_raised_as_it_was(tmp_path):
    # The fallback answers two errors of numba's cache set-up, no locator and
    # a directory that cannot be written. A RuntimeError of another kind at
    # the same step, a locator of the user's own refusing, say, is not its to
    # turn into an uncached import with a remedy that does not apply.
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", ANOTHER_ERROR_AT_THE_CACHE_SET_UP],
                         capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert run.returncode != 0 and "a locator of another kind refused" in run.stderr, run.stderr
    assert "compiles without a cache" not in run.stderr


# The code numbox generates at run time, each child importing the helpers of
# the test module that exercises it.
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
    # The code numbox generates at run time is anchored to a file under
    # NUMBA_CACHE_DIR or the user's cache directory, and the anchor was written
    # whatever the cache option said. So with the tree writable, where the
    # package's own functions cache, and the user's cache directory not,
    # make_structref and the sqlite registrations died at the anchor's
    # directory. The anchor is written to be cached from, and where it cannot
    # be, the code compiles without a cache after a warning; NUMBA_CACHE_DIR at
    # a writable directory cures it and holds the anchor.
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
    # The anchor is on disk from an earlier run, in a NUMBA_CACHE_DIR that can
    # no longer be written. numba then caches the code in the user's cache
    # directory, and a check on the anchor's own directory would have turned
    # caching off where numba had a location. With the user's cache directory
    # unwritable too there is none, and a check that only wrote the anchor
    # would have let the first decorated function die at numba's set-up. The
    # builder's derive falls back without a word, as it did for an anchor it
    # could not write.
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
    # NUMBA_CACHE_DIR pointing into a file, or with a component longer than the
    # file system allows: the package's own functions cache beside their
    # sources, numba passing such a location over, and the anchor's directory
    # cannot be made. Any error there means no cache here, so the code compiles
    # without one, as it did before the anchors asked numba, and the warning's
    # remedy is the error's: a directory numba can use, or a shorter path.
    if placement == "a file":
        cache_dir = tmp_path / "file"
        cache_dir.write_text("not a directory\n")
        remedy = "Set NUMBA_CACHE_DIR to a writable directory"
    else:
        cache_dir = tmp_path / ("c" * 300)
        # Windows reports the component as a syntax error (WinError 123, EINVAL),
        # naming no length, and the warning offers the directory then.
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
    # make_graph's kernel is anchored to builder.py itself and cached beside it,
    # and a caller's options reach numba as they are: with cache on from an
    # archive the kernel died at numba's set-up, where the package, the
    # structref, the kernel compiler and the derives had fallen back.
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
    # The derives cache under anchors of their own, written here, and the
    # builder's answer for its file is not theirs.
    assert _index_files(tmp_path / "cache"), "the derive did not cache under NUMBA_CACHE_DIR"


A_STRUCTREF_WITH_A_TYPING_ERROR = MAKE_A_STRUCTREF.replace(
    "Struct = make_structref('Struct', {'value': float32}, TypeClass)\n",
    "def bad(self):\n"
    "    return self.value + 'x'\n"
    "Struct = make_structref('Struct', {'value': float32}, TypeClass, struct_methods={'bad': bad})\n"
    "Struct(2.5).bad()\n",
)


def test_a_typing_error_in_generated_code_quotes_the_source_with_caching_off(tmp_path):
    # numba quotes the offending line from the file the code names, the
    # anchor, which with caching off was no longer written: the message then
    # pointed at a file that was not there, with "source missing", where main
    # showed the line. The anchor is written where it can be, cache or no.
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
    # make_structref, compile_kernel and the builder take jit options of the
    # caller's, which NUMBOX_JIT_OPTIONS does not reach, and compile_kernel's
    # cache argument overrides those too; the warning offered the variable
    # alone, here set to turn caching off already, and then the jit options,
    # here off as well.
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


# The type class lives in a module of its own, as the docs ask, so that the
# struct's cache entries load in a second process.
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
    # numba names a cache file after the anchor's stem and the jitted
    # function's qualname, both of which carried the struct's name, so a
    # name of about 93 characters overflowed the file system's 255 bytes in
    # numba's own files, past the anchor's check. The stems and the generated
    # names are bounded now: as they are up to 40 bytes, a prefix and a digest
    # beyond, and the class takes its full name back once compiled. The file
    # system counts bytes, so a name of 40 accented characters (80 bytes) is
    # bounded, and 100 CJK characters (300 bytes) are cut by whole characters.
    # A field named with the struct's length is bounded in its getter the same
    # way, and a field named with that bounded name, defined before it, keeps
    # its property: the long field's getter took the name and the hand-over
    # deleted the short field's. One method's name is 40 bytes, the most a
    # bounded name can be, so its thunk's files are the longest numba writes
    # for any struct: under the 230 bytes bounded_stem promises, which leave
    # room for numba's temporary name at the write; the other's is 200, which
    # the thunk's and the overload's names must bound.
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
    # A cache directory deep enough that the anchor's directory fits the path
    # limit and the anchor's own name, of fixed length, does not: the user's
    # cache directory under a deep home, which the warning met with "a shorter
    # NUMBA_CACHE_DIR" where none was set. There is no cache here, and the
    # warning names the length rather than offering a writable directory, which
    # this one is; NUMBA_CACHE_DIR at a short path cures it. A NUMBA_CACHE_DIR
    # is made deep enough for the package's own files instead, whose location
    # under it, a directory named for theirs by its name and a hash of its
    # path, ends 20 bytes short of the limit: numba's own check passes there,
    # with a named temporary file too where the file system makes no unnamed
    # one, and the package's does not, so the package answers first, with its
    # remedy, and the struct compiles under its answer without a word of its
    # own. At the home's depth numba's check failed the location on such a file
    # system and moved on to the tree, and the anchor answered instead.
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
    # numba names the files it caches a function in after the source's name
    # and the function's qualified name, and a name longer than the file
    # system takes can be made at no path: numba's first read of the index
    # dies on it, and the remedy told the path as too long, asking for the
    # package, the user's cache directory, IPython's or NUMBA_CACHE_DIR at a
    # shorter path, which move nothing of the name. It tells the name and its
    # length, counted as the file system counts it, that no shorter path
    # cures it, and the cure, a shorter name for the function or its file,
    # the package's to give, with the variable, where numba tries it before
    # the locator that took the file and a file system in use takes the name,
    # 255 being their limit, pointed at a directory on one, and the
    # variable's own location named where it took the file; the remedy
    # offered such a file system for a name none takes, and never the
    # shorter name. Windows refuses a name too long as an invalid one, errno
    # EINVAL, and the check runs whatever the errno.
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
        # A limit the directory numba names after the source's, its name and
        # forty-one bytes, fits, and the file's name does not.
        monkeypatch.setattr(configurations, "_name_limit", lambda location: 60)
    limit = configurations._name_limit(str(location))
    name = f"{Path(py_file).stem}.{'f' * (limit + 5)}-1.py{sys.version_info[0]}{sys.version_info[1]}.nbi"
    counted_in = "characters" if os.name == "nt" or sys.platform == "darwin" else "bytes"
    length = len(name.encode("utf-16-le")) // 2 if counted_in == "characters" else len(os.fsencode(name))
    assert length > limit
    assert (length <= 255) == ("a file system in use takes" in case)
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
    # numba's own error for a cache file whose name is too long, which its
    # first read of the index raises at the first call, names the file, and
    # the remedy told the path as too long and asked for the package installed
    # at a shorter path, or NUMBA_CACHE_DIR set to a short one, which cure
    # nothing of the name. Windows refuses the name as an invalid one, errno
    # EINVAL, and the remedy reads the name whatever the errno.
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
    # numba's in-tree locator takes a cell file on disk in an ipykernel
    # directory whose __pycache__ it can write, and IPython's locator, listed
    # after it, is never reached; the remedy read IPython's as reached for any
    # file its locator takes by name, and told a caller's error naming another
    # numba_cache, a location that is none of numba's, as IPython giving a
    # temporary directory per call, with IPYTHONDIR for it.
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
    # numba hashes the source's directory path as UTF-8 for the name of its
    # directory under NUMBA_CACHE_DIR or the user's cache directory, and
    # raises UnicodeEncodeError for bytes that are not, decoded with
    # surrogates, at decoration, where a locator that names its directory so
    # reaches the file: the user-wide one, once the in-tree locator passed an
    # unwritable __pycache__ over. The check raises it as numba does, and
    # is_a_cache_error did not admit it, a ValueError that is not the .zip
    # locator's, so a package installed under such a directory died at import
    # on it, where with caching off it imports, and the remedy for it stood
    # unreached. A UnicodeEncodeError of another reason is no cache error.
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
    # A name an error carries decoded with surrogateescape holds a surrogate
    # for a byte that is not UTF-8, and Windows takes a lone surrogate in a
    # name of its own; counting the name in UTF-16 units, as Windows and
    # macOS count one, raised UnicodeEncodeError on it, and the remedy died on
    # an error it admits instead of answering it.
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
    # numba's .zip locator names its directory under the user's cache
    # directory after the source's directory, the archive's path in it, hashed
    # as UTF-8, and raises UnicodeEncodeError for bytes that are not, at
    # decoration and in the check alike; the remedy took the error for a
    # caller's with no place for the file, and offered the .zip the reader was
    # importing from, with nothing of the path.
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
    # numba hashes the anchor's directory path as UTF-8 to name its cache
    # directory under NUMBA_CACHE_DIR or the user's cache directory, and
    # raises UnicodeEncodeError for bytes that are not, which is a cache
    # error; the anchor's warning offered a writable directory, which the
    # anchor's is, and said nothing of the path.
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
    # With NUMBA_CACHE_DIR set, numba's user-provided locator hashes the
    # source's directory path as UTF-8 for its directory under the variable
    # and raises for bytes that are not, before the in-tree locator is
    # reached, which caches beside the source and hashes no path: the same
    # install imports cached with the variable unset. The remedy said the
    # variable cures nothing and asked for a UTF-8 path alone.
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
    # Under a list of the user-wide locator alone, numba hashes the source's
    # directory path as UTF-8 for its directory under the user's cache
    # directory and raises for bytes that are not; the remedy told the list
    # and its locations to make writable, which hashing refuses whatever
    # they are, with nothing of the path or of the in-tree locator, which
    # caches beside the source hashing no path.
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
    # numba's .zip locator, listed after the in-tree one, takes a source on
    # disk under a directory named .zip once an unwritable __pycache__ left
    # it, and hashes the source's directory path as UTF-8 for its directory,
    # raising UnicodeEncodeError for bytes that are not; the remedy read
    # every ValueError the .zip locator reaches as its own for a path with
    # no archive, and told a missing archive and locations to make writable.
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
    # numba's user-wide locator, the one of numba's that takes a frozen
    # application's file, hashes the file's directory path as UTF-8 for its
    # directory under the user's cache directory and raises for bytes that
    # are not; the remedy took the error for a caller's with no place for
    # the file and offered the source files on disk, or a .zip holding them,
    # with nothing of the path.
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
    # The in-tree locator caches beside the source hashing no path, and
    # listed after one that hashes, the variable's set or the user-wide one,
    # it is never reached: the remedy asked for it to be listed, with no
    # place, where appended it leaves numba raising the same error, and, for
    # one listed after the user-wide locator, for its __pycache__ to be made
    # writable, which it was.
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
    # Listed as the remedy says, numba caches the file.
    configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)


@pytest.mark.skipif(os.name == "nt", reason="Windows decodes any byte of a path")
@pytest.mark.skipif(sys.platform == "darwin", reason="APFS takes no name that is not UTF-8")
def test_a_frozen_application_under_a_zip_named_directory_on_a_path_that_is_not_utf8_is_told_the_application(
        tmp_path, monkeypatch):
    # numba's user-wide locator, before the .zip one in numba's order, takes
    # a frozen application's file and hashes its directory path as UTF-8,
    # raising for bytes that are not; the remedy read the .zip in the path
    # and told an archive to put under a UTF-8 path, where there is none.
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
    # A list without numba's .zip locator was told to list it for any path
    # with .zip in it, where a directory so named above a source that is
    # nowhere is no archive: listed, that locator takes the directory for one
    # and caches nothing, numba's check raising for it. Such a source is told
    # the source files, on disk or in a .zip.
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
    # inspect raises for a cell whose source linecache does not hold, before
    # numba's IPython locator tries its location, and the remedy tells it as
    # such, but for a member of an archive, told the source files to ship: a
    # directory named .zip above the cell was read as an archive, and the
    # cell was told to ship source files.
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
    # numba's .zip locator takes the first part of the path ending in .zip for
    # the archive, a directory so named above an .egg included, and opens it:
    # the remedy told the .egg renamed to end in .zip, under that directory
    # still, where numba then raises IsADirectoryError for the directory.
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
    # Done as told, numba caches the member.
    moved = tmp_path / "numbox.zip"
    egg.rename(moved)
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "InTreeCacheLocator,ZipCacheLocator", raising=False)
    configurations.check_cache_location(str(moved / "pkg" / "m.py"), configurations.LONGEST_CACHE_FILE_NAME)


@pytest.mark.parametrize("units", [230, 231, 251, 252])
def test_the_offer_of_a_file_system_that_takes_a_cache_file_name_holds_at_the_usual_limit(tmp_path, monkeypatch, units):
    # NUMBA_CACHE_DIR on a file system that takes the name is offered where
    # the longest name numba writes for the function fits 255, the file
    # systems in use's limit: the index's base, ".999.nbc" for a data file
    # with the three digits reserved for its number, and the 21 of the
    # temporary name numba writes each file through, ".tmp." and sixteen hex
    # digits; and not for one over. No test held the boundary, and a count of
    # less than 255 passed them all; then the offer went to a name of 255,
    # whose temporary name numba's save of it failed on; then to an index of
    # 234, whose data file's did; then two digits left a hundredth
    # signature's data file a byte over.
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
        # numba's save names the temporary file, the name and 21 more.
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
    # The error numba's first save raises under an IPython giving a temporary
    # directory per call is FileNotFoundError, naming the numba_cache it could
    # not save in; the remedy for it took any OSError naming a numba_cache,
    # a caller's refused directory of its own among them, for a member in an
    # ipykernel directory, and told IPYTHONDIR where, with IPython giving its
    # directory, the same error is told as the location it names.
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
    # The shape numba's own first save has there is told IPYTHONDIR still.
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
    # With NUMBA_CACHE_DIR set, its locator hashes the source's directory path
    # and raises before the in-tree locator is tried, so the remedy cannot
    # read from numba's error whether the __pycache__ beside the source can
    # be written; it asks numba, and tells the variable to unset and the
    # __pycache__ to make writable where it cannot be: the remedy hedged
    # "where that directory can be written" and never asked for it, and the
    # variable unset as told left numba raising the same error.
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
        # Unset alone, numba raises the same error; both as told, it caches.
        monkeypatch.setattr(numba.config, "CACHE_DIR", "")
        with pytest.raises(UnicodeEncodeError):
            configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    finally:
        pycache.chmod(0o755)
    configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)


MAKE_A_STRUCTREF_OF_FORTY_CHARACTER_NAMES = MAKE_A_STRUCTREF.replace(
    "make_structref('Struct', {'value': float32}, TypeClass)",
    "make_structref('S' * 40, {'v' * 40: float32}, TypeClass)").replace(
    "assert Struct(2.5).value == 2.5", "assert getattr(Struct(2.5), 'v' * 40) == 2.5")


@pytest.mark.skipif(sys.platform != "linux", reason="a path of 4096 bytes is Linux's limit")
@pytest.mark.parametrize("short_of_the_limit", [230, 190])
def test_an_anchor_within_numbas_longest_name_of_the_path_limit_compiles_uncached_and_the_warning_says_so(
        tmp_path, short_of_the_limit):
    # The anchor's check asked numba's location for a name of eight, which
    # numba's own check makes too, so a NUMBA_CACHE_DIR within the longest
    # name numba writes for a struct of the path limit passed both, and the
    # struct's making died at numba's first save with no warning; the check
    # asks for a name as long as that, and the code compiles without a cache
    # after the warning naming the remedy.
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
    # The first path at or above the location that is there is the directory
    # it cannot be made under; walked up a relative path with no ancestor on
    # the way, it came to nothing, and the remedy said a file with no name
    # was in the way.
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
    # The check makes a file of the longest name the package's functions get
    # in the location, and where the file system takes no name as long, its
    # error, re-raised as the location's, was read as a path too long and a
    # shorter NUMBA_CACHE_DIR offered, which cures nothing of a name; the
    # error carries the length now, and the remedy asks for a file system
    # that takes it. The anchor's warning does the same for a struct's.
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
    # The anchor, whose check asks for numba's longest name for the code's
    # functions, read off the code.
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
    # The anchor's check asked numba's location for a name of 251, a bound
    # for any struct, so a file system whose name limit is under it, eCryptfs's
    # 143, turned caching off for every struct, one whose names were 58
    # included; the check asks for the longest name numba writes for the
    # code's own functions, read off the code: the stem, the qualified name
    # with the angle brackets numba drops, the line, Python's version, two
    # digits for the overload and the temporary name, of the functions the
    # code jits alone, a struct's method under its own name, which calls the
    # jitted thunk, and an overload's getting no cache file.
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
    # The names counted: a function decorated itself, with njit or a decorator
    # of the package's own that jits, or defined within a decorated one, an
    # overload's constructor among them, with numba's "locals" for the
    # nesting; and not a plain method's under its own name. Counted by five
    # decorators' names alone, the builder's derive function and the
    # constructor overload's inner function were left out.
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
    # numba numbers a function's data files from one without bound, so a
    # hundredth signature's name is a byte longer than two reserved digits
    # allowed for, and the offer of a file system that takes the name went to
    # a name whose data file that file system then refused; three digits are
    # reserved now, and a number the error names that is wider is read by its
    # own width.
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
