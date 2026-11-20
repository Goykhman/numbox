"""Tests of loading numba's cache from a directory that can be read but not written.

:mod:`numbox.core.read_only_cache` holds a locator that takes such a directory, and the function that recognizes the
error of a write in one. The first tests call both in this process. They also call the check of a cache location,
numbox's wrappers around numba's save and flush and the writer of generated code. The other tests run child processes
with a copy of numbox, a small package and cache directories that each test creates. The first child builds the
cache while every directory can be written. The test then removes write access from the cache, the sources, the home
and the copy of numbox, and the later child processes run from those directories. Those tests are skipped on Windows
and when the tests run as root, because chmod does not prevent a write there. A write in such a directory fails with
``EACCES``. A write on a file system that is mounted read-only fails with ``EROFS``, which chmod cannot produce. The
tests that replace a function of ``tempfile``, numba's save or numba's flush raise that error number at the locator,
at the check, at the save, at the flush and at the writer of generated code.
"""
import contextlib
import errno
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numba
import pytest
from numba.core import caching

from numbox.core import cache_key
from numbox.core.configurations import LONGEST_CACHE_FILE_NAME, check_cache_location
from numbox.core.read_only_cache import ReadOnlyCacheLocator, cannot_be_written
from numbox.utils.preprocessing import _anchored_or_uncached

REPO = Path(__file__).resolve().parent.parent.parent

needs_a_directory_it_cannot_write = pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="needs a directory this user cannot write to")

# The value of NUMBA_CACHE_LOCATOR_CLASSES that the configurations page gives: numbox's locator in place of
# numba's UserProvidedCacheLocator, followed by numba's other locators in numba's order.
SHIPPED = ("numbox.core.read_only_cache.ReadOnlyCacheLocator,InTreeCacheLocator,UserWideCacheLocator,"
           "IPythonCacheLocator,ZipCacheLocator")


class NeverTestsLocator(caching.UserProvidedCacheLocator):
    """A user-defined locator that never tests whether it can write in its location."""

    def ensure_cache_path(self):
        pass


class TakesUntestedLocator(caching.UserProvidedCacheLocator):
    """A user-defined locator that takes a file without a test and leaves numba's test in place for the saves."""

    @classmethod
    def from_function(cls, py_func, py_file):
        return cls(py_func, py_file)


class UserProvidedCacheLocator(TakesUntestedLocator):
    """A user-defined locator with the same name as numba's class."""


@contextlib.contextmanager
def _read_only(*roots):
    """Remove write access from every directory under ``roots`` for the block, and restore it afterward."""
    directories = [path for root in roots for path in [root, *root.rglob("*")] if path.is_dir()]
    for path in directories:
        path.chmod(0o555)
    try:
        yield
    finally:
        for path in directories:
            path.chmod(0o755)


def _files(root):
    """Return each file under ``root`` with its size and modification time, to compare a tree before and after."""
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def _function_in_a_file_of_its_own(tmp_path, name="kernels", function="add_one"):
    """Return ``function``, decorated with ``njit(cache=True)`` in a new module file under ``tmp_path``.

    numba keys a cache by the source file, so a function in a file of its own has a cache that no other test and no
    earlier run has filled.
    """
    source = tmp_path / f"{name}.py"
    source.write_text(f"from numba import njit\n\n\n@njit(cache=True)\ndef {function}(x):\n    return x + 1\n")
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function)


def _module_whose_function_adds_a_global(tmp_path, name):
    """Return a new module ``name`` loaded from ``scaled.py`` under ``tmp_path``.

    The module's cached function ``add_g`` returns its argument plus the module's global ``G``, which is 1 when the
    module is loaded. numba keys a cache by the source file, so the modules that the helper loads from one
    ``tmp_path`` share one cache. The helper writes the file once, because numba discards the cache of a file whose
    modification time changed.
    """
    source = tmp_path / "scaled.py"
    if not source.exists():
        source.write_text("from numba import njit\n\nG = 1\n\n\n@njit(cache=True)\ndef add_g(x):\n    return x + G\n")
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _refusing(number, root):
    """Return a replacement for a function of ``tempfile`` that fails with ``number`` for a file under ``root``.

    The replacement raises the error associated with a directory that does not permit the write, as a read-only mount
    raises ``EROFS``. No test expects a call for a file elsewhere, so such a call fails an assertion.
    """
    def refuse(*args, **kwargs):
        directory = str(kwargs["dir"])
        assert directory.startswith(str(root)), directory
        raise OSError(number, os.strerror(number), os.path.join(directory, "tmp0123"))
    return refuse


@pytest.mark.parametrize("number", [errno.EROFS, errno.EACCES, errno.EPERM])
def test_a_write_refused_in_an_existing_directory_is_recognized(tmp_path, number):
    assert cannot_be_written(OSError(number, os.strerror(number), str(tmp_path / "x.tmp")), tmp_path)
    assert cannot_be_written(PermissionError(number, os.strerror(number)), str(tmp_path))


@pytest.mark.parametrize("number", [errno.ENOSPC, errno.ENAMETOOLONG, errno.ENOENT, errno.ENOTDIR, errno.EIO, None])
def test_another_error_of_a_write_is_not_recognized(tmp_path, number):
    """The other errors, such as a full disk or a name that is too long, do not mean that the directory holds a
    cache that cannot be written."""
    error = OSError("no error number") if number is None else OSError(number, os.strerror(number))
    assert not cannot_be_written(error, tmp_path)


def test_a_refused_write_where_no_directory_exists_is_not_recognized(tmp_path):
    """A location that does not exist, or that is a file, holds no cache to load."""
    refused = PermissionError(errno.EACCES, "Permission denied")
    assert not cannot_be_written(refused, tmp_path / "missing")
    (tmp_path / "a-file").write_text("")
    # The file is executable, so the permission that allows a process to search a directory does not exclude it.
    (tmp_path / "a-file").chmod(0o755)
    assert not cannot_be_written(refused, tmp_path / "a-file")
    assert not cannot_be_written(RuntimeError("no locator available"), tmp_path)


@needs_a_directory_it_cannot_write
@pytest.mark.parametrize("mode", [0o000, 0o600])
def test_a_refused_write_in_a_directory_that_cannot_be_searched_is_not_recognized(tmp_path, mode):
    """numba opens a cache file by its path, which fails in a directory without search permission."""
    location = tmp_path / "location"
    location.mkdir()
    refused = PermissionError(errno.EACCES, "Permission denied")
    location.chmod(0o500)
    try:
        assert cannot_be_written(refused, location)
        location.chmod(mode)
        assert not cannot_be_written(refused, location)
    finally:
        location.chmod(0o755)


def test_the_locator_behaves_as_numbas_where_the_directory_can_be_written(tmp_path, monkeypatch):
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    add_one = _function_in_a_file_of_its_own(tmp_path).py_func
    source = str(tmp_path / "kernels.py")
    theirs = caching.UserProvidedCacheLocator.from_function(add_one, source)
    shutil.rmtree(tmp_path / "cache")
    ours = ReadOnlyCacheLocator.from_function(add_one, source)
    assert ours.get_cache_path() == theirs.get_cache_path()
    # The locator created its directory, as numba's does, and left no file in it.
    assert os.path.isdir(ours.get_cache_path()) and not os.listdir(ours.get_cache_path())
    assert ours.get_source_stamp() == theirs.get_source_stamp()
    assert ours.get_disambiguator() == theirs.get_disambiguator()


def test_the_locator_takes_no_file_while_numba_cache_dir_is_unset(tmp_path, monkeypatch):
    add_one = _function_in_a_file_of_its_own(tmp_path).py_func
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    assert ReadOnlyCacheLocator.from_function(add_one, str(tmp_path / "kernels.py")) is None


@needs_a_directory_it_cannot_write
def test_the_locator_takes_a_directory_that_exists_and_cannot_be_written(tmp_path, monkeypatch):
    """numba's own locator for ``NUMBA_CACHE_DIR`` skips such a directory, which is why a cache in it is never read."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    add_one = _function_in_a_file_of_its_own(tmp_path).py_func
    source = str(tmp_path / "kernels.py")
    built = ReadOnlyCacheLocator.from_function(add_one, source).get_cache_path()
    with _read_only(tmp_path / "cache"):
        assert caching.UserProvidedCacheLocator.from_function(add_one, source) is None
        ours = ReadOnlyCacheLocator.from_function(add_one, source)
        assert ours is not None and ours.get_cache_path() == built
        # numba calls the method again before each save, and it must not raise there either.
        ours.ensure_cache_path()


@pytest.mark.parametrize("number", [errno.EROFS, errno.EACCES, errno.EPERM])
def test_the_locator_takes_a_directory_whose_write_test_fails_with_each_error_of_a_refused_write(
        tmp_path, monkeypatch, number):
    """``EROFS`` is the error on a file system that is mounted read-only."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    add_one = _function_in_a_file_of_its_own(tmp_path).py_func
    source = str(tmp_path / "kernels.py")
    monkeypatch.setattr(caching.tempfile, "TemporaryFile", _refusing(number, tmp_path / "cache"))
    assert caching.UserProvidedCacheLocator.from_function(add_one, source) is None
    assert ReadOnlyCacheLocator.from_function(add_one, source) is not None


@needs_a_directory_it_cannot_write
def test_the_locator_skips_a_directory_that_cannot_be_searched(tmp_path, monkeypatch):
    """numba cannot open the cache files of such a directory, so the locator skips the file, as numba's own locator
    does."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    add_one = _function_in_a_file_of_its_own(tmp_path).py_func
    source = str(tmp_path / "kernels.py")
    built = Path(ReadOnlyCacheLocator.from_function(add_one, source).get_cache_path())
    built.chmod(0o000)
    try:
        assert caching.UserProvidedCacheLocator.from_function(add_one, source) is None
        assert ReadOnlyCacheLocator.from_function(add_one, source) is None
    finally:
        built.chmod(0o755)


@needs_a_directory_it_cannot_write
def test_the_locator_takes_a_directory_that_can_be_searched_but_not_listed(tmp_path, monkeypatch):
    """numba opens a cache file by its path, which needs search permission on the directory and does not need read
    permission."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    add_one = _function_in_a_file_of_its_own(tmp_path).py_func
    source = str(tmp_path / "kernels.py")
    built = Path(ReadOnlyCacheLocator.from_function(add_one, source).get_cache_path())
    built.chmod(0o111)
    try:
        assert caching.UserProvidedCacheLocator.from_function(add_one, source) is None
        ours = ReadOnlyCacheLocator.from_function(add_one, source)
        assert ours is not None and ours.get_cache_path() == str(built)
    finally:
        built.chmod(0o755)


@needs_a_directory_it_cannot_write
def test_an_index_file_that_cannot_be_read_raises_numbas_error_at_the_first_call(tmp_path, monkeypatch):
    """The locator does not test whether the process can read the files of a directory that it takes."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "numbox.core.read_only_cache.ReadOnlyCacheLocator",
                        raising=False)
    assert _module_whose_function_adds_a_global(tmp_path, "scaled_build").add_g(1) == 2
    (index,) = (tmp_path / "cache").rglob("*.nbi")
    index.chmod(0o000)
    try:
        with _read_only(tmp_path / "cache"):
            run = _module_whose_function_adds_a_global(tmp_path, "scaled_run")
            with pytest.raises(PermissionError) as raised:
                run.add_g(1)
    finally:
        index.chmod(0o644)
    assert raised.value.filename == str(index)


def test_an_index_file_that_numba_is_denied_raises_at_the_first_call_except_on_windows(tmp_path, monkeypatch):
    """On Windows, numba ignores a load and a save that fail with ``EACCES``, so the function compiles."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    add_one = _function_in_a_file_of_its_own(tmp_path)

    def denied(self):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), self._index_path)

    monkeypatch.setattr(caching.IndexDataCacheFile, "_load_index", denied)
    if os.name == "nt":
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert add_one(1) == 2
        assert caught == []
        assert (sum(add_one.stats.cache_hits.values()), sum(add_one.stats.cache_misses.values())) == (0, 1)
        return
    with pytest.raises(PermissionError) as raised:
        add_one(1)
    assert raised.value.errno == errno.EACCES


@needs_a_directory_it_cannot_write
def test_a_data_file_that_cannot_be_read_is_compiled_again_and_its_save_is_skipped(tmp_path, monkeypatch):
    """numba treats a data file that it cannot open as an entry that the directory does not hold."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "numbox.core.read_only_cache.ReadOnlyCacheLocator",
                        raising=False)
    monkeypatch.setattr(cache_key, "_unsaved_reported", False)
    assert _module_whose_function_adds_a_global(tmp_path, "scaled_build").add_g(1) == 2
    (data,) = (tmp_path / "cache").rglob("*.nbc")
    before = _files(tmp_path / "cache")
    data.chmod(0o000)
    try:
        with _read_only(tmp_path / "cache"):
            run = _module_whose_function_adds_a_global(tmp_path, "scaled_run")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                assert run.add_g(1) == 2
    finally:
        data.chmod(0o644)
    assert (sum(run.add_g.stats.cache_hits.values()), sum(run.add_g.stats.cache_misses.values())) == (0, 1)
    assert len(caught) == 1, [str(warning.message) for warning in caught]
    assert str(caught[0].message).startswith("numbox: numba compiled add_g but cannot save its cache entry")
    assert _files(tmp_path / "cache") == before


@needs_a_directory_it_cannot_write
def test_the_locator_skips_a_directory_that_it_cannot_create(tmp_path, monkeypatch):
    """A source directory that the cache was not built with has no subdirectory, so there is nothing to load."""
    # The function is decorated before the variable is set, so that numba's own locator does not create the
    # subdirectory.
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    add_one = _function_in_a_file_of_its_own(tmp_path).py_func
    (tmp_path / "cache").mkdir()
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    with _read_only(tmp_path / "cache"):
        assert ReadOnlyCacheLocator.from_function(add_one, str(tmp_path / "kernels.py")) is None
    assert not os.listdir(tmp_path / "cache")


def test_the_locator_does_not_take_a_directory_that_fails_for_another_reason(tmp_path, monkeypatch):
    """A full disk is not a read-only cache: the locator skips the directory, as numba's does."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    add_one = _function_in_a_file_of_its_own(tmp_path).py_func

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(caching.tempfile, "TemporaryFile", full)
    assert ReadOnlyCacheLocator.from_function(add_one, str(tmp_path / "kernels.py")) is None


@needs_a_directory_it_cannot_write
@pytest.mark.parametrize("locator", [
    "numbox.core.read_only_cache.ReadOnlyCacheLocator",
    f"{__name__}.NeverTestsLocator",
    f"{__name__}.TakesUntestedLocator",
])
@pytest.mark.parametrize("longest_file_name", [0, LONGEST_CACHE_FILE_NAME])
def test_the_check_passes_a_location_that_exists_and_cannot_be_written(
        tmp_path, monkeypatch, locator, longest_file_name):
    """The check returns for a location that a locator took and that refuses a write, whichever step writes first.

    numbox's locator and numba's test fail first under two of the locators. Under the locator that tests nothing, the
    first write is the check's own file with a name of ``longest_file_name`` bytes. For a length of zero, the check
    writes nothing under that locator.
    """
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    # The location is created under numba's own locators, as it is when a cache is built, because one of the locators
    # under test never creates its directory.
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    source = tmp_path / "kernels.py"
    source.write_text("")
    check_cache_location(source, longest_file_name)
    assert [path.is_dir() for path in (tmp_path / "cache").iterdir()] == [True]
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locator, raising=False)
    with _read_only(tmp_path / "cache"):
        assert check_cache_location(source, longest_file_name) is None
    assert _files(tmp_path / "cache") == {}


@pytest.mark.parametrize("number", [errno.EROFS, errno.EACCES, errno.EPERM])
@pytest.mark.parametrize("locator, function", [
    ("TakesUntestedLocator", "TemporaryFile"),
    ("NeverTestsLocator", "NamedTemporaryFile"),
])
def test_the_check_passes_a_location_whose_write_fails_with_each_error_of_a_refused_write(
        tmp_path, monkeypatch, number, locator, function):
    """numba's test creates its file with ``TemporaryFile``, and the check creates its own file with
    ``NamedTemporaryFile``. Under the locator that tests nothing, the check's own file is the first write."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    source = tmp_path / "kernels.py"
    source.write_text("")
    check_cache_location(source, LONGEST_CACHE_FILE_NAME)
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", f"{__name__}.{locator}", raising=False)
    monkeypatch.setattr(caching.tempfile, function, _refusing(number, tmp_path / "cache"))
    assert check_cache_location(source, LONGEST_CACHE_FILE_NAME) is None


@needs_a_directory_it_cannot_write
@pytest.mark.parametrize("locator", ["NeverTestsLocator", "TakesUntestedLocator"])
def test_the_check_raises_for_a_location_that_cannot_be_searched(tmp_path, monkeypatch, locator):
    """numba cannot open the index file in such a location, so the check raises the error caused by the failed
    write."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    source = tmp_path / "kernels.py"
    source.write_text("")
    check_cache_location(source, LONGEST_CACHE_FILE_NAME)
    (location,) = (tmp_path / "cache").iterdir()
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", f"{__name__}.{locator}", raising=False)
    location.chmod(0o000)
    try:
        with pytest.raises(PermissionError) as raised:
            check_cache_location(source, LONGEST_CACHE_FILE_NAME)
    finally:
        location.chmod(0o755)
    assert raised.value.filename == str(location)


@needs_a_directory_it_cannot_write
def test_the_check_still_raises_for_a_location_that_does_not_exist_and_cannot_be_created(tmp_path, monkeypatch):
    (tmp_path / "cache").mkdir()
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", f"{__name__}.TakesUntestedLocator", raising=False)
    source = tmp_path / "kernels.py"
    source.write_text("")
    with _read_only(tmp_path / "cache"):
        with pytest.raises(PermissionError) as raised:
            check_cache_location(source, LONGEST_CACHE_FILE_NAME)
    assert raised.value.filename.startswith(str(tmp_path / "cache"))


def test_the_check_still_raises_for_a_location_that_fails_for_another_reason(tmp_path, monkeypatch):
    """The check raises the error if the creation of its own file fails because the disk is full: a full disk is
    not a read-only cache."""
    import numbox.core.configurations as configurations

    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    source = tmp_path / "kernels.py"
    source.write_text("")

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(configurations.tempfile, "NamedTemporaryFile", full)
    with pytest.raises(OSError) as raised:
        check_cache_location(source, LONGEST_CACHE_FILE_NAME)
    assert raised.value.errno == errno.ENOSPC


@pytest.mark.parametrize("number", [errno.EROFS, errno.EACCES, errno.EPERM])
def test_a_save_that_the_directory_refuses_is_skipped_with_one_warning_in_a_process(tmp_path, monkeypatch, number):
    """The call that compiled the function returns its result, and numbox skips the second refused save of the
    process without a warning."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setattr(cache_key, "_unsaved_reported", False)
    add_one = _function_in_a_file_of_its_own(tmp_path)
    location = add_one.stats.cache_path
    saves = []

    def refused(self, sig, data):
        saves.append(sig)
        raise OSError(number, os.strerror(number), os.path.join(location, "kernels.add_one-4.nbi.tmp.0123"))

    monkeypatch.setattr(caching.Cache, "_save_overload", refused)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert add_one(1) == 2
        assert add_one(1.5) == 2.5
    assert len(saves) == 2
    if os.name == "nt" and number == errno.EACCES:
        # On Windows, numba itself ignores a save that fails with this error, so the wrapper never sees the error.
        assert caught == []
        return
    assert [str(warning.message) for warning in caught] == [
        f"numbox: numba compiled add_one but cannot save its cache entry in the cache directory {location} "
        f"({os.strerror(number)}), so the function compiles again in every process. numbox skips each save that "
        "fails in this way. It warns only once per process, for the first skipped save."
    ]
    assert caught[0].category is RuntimeWarning


def test_two_functions_whose_saves_are_skipped_give_one_warning_in_a_process(tmp_path, monkeypatch):
    """The warning names the function whose save was skipped first."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setattr(cache_key, "_unsaved_reported", False)
    first = _function_in_a_file_of_its_own(tmp_path, "first_kernels", "first")
    second = _function_in_a_file_of_its_own(tmp_path, "second_kernels", "second")

    def refused(self, sig, data):
        raise OSError(errno.EROFS, os.strerror(errno.EROFS), os.path.join(self.cache_path, "x.nbi.tmp.0123"))

    monkeypatch.setattr(caching.Cache, "_save_overload", refused)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert first(1) == 2
        assert second(1) == 2
    assert len(caught) == 1, [str(warning.message) for warning in caught]
    assert str(caught[0].message).startswith("numbox: numba compiled first but cannot save its cache entry")


# The program refuses every save with EROFS, so numbox skips each save of its two functions. It prints the functions
# named in numbox's warnings in the parent, in a child forked after the parent's warning, and in the parent again.
FORKING = '''import errno
import json
import os
import re
import warnings

from numba import njit
from numba.core import caching

import numbox.core  # noqa: F401


def refused(self, sig, data):
    raise OSError(errno.EROFS, os.strerror(errno.EROFS), os.path.join(self.cache_path, "x.nbi.tmp.0123"))


caching.Cache._save_overload = refused


@njit(cache=True)
def first(x):
    return x + 1


@njit(cache=True)
def second(x):
    return x + 1


def named(function, argument):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        function(argument)
    messages = " ".join(str(warning.message) for warning in caught)
    return re.findall(r"numba compiled (\\w+) but cannot save its cache entry", messages)


seen = {"parent": named(first, 1)}
read_end, write_end = os.pipe()
pid = os.fork()
if pid == 0:
    try:
        os.write(write_end, json.dumps(named(second, 1)).encode())
    finally:
        os._exit(0)
os.close(write_end)
seen["forked child"] = json.loads(os.read(read_end, 1000))
os.waitpid(pid, 0)
seen["parent after the fork"] = named(second, 1)
print(json.dumps(seen))
'''


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_process_forked_after_the_warning_warns_of_its_own_first_skipped_save(tmp_path):
    """A forked child starts with a copy of its parent's memory, including numbox's warning state from before the fork.
    The parent does not issue another warning after the fork."""
    (tmp_path / "forking.py").write_text(FORKING)
    (tmp_path / "cache").mkdir()
    env = dict(os.environ, PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    for name in ("NUMBOX_JIT_OPTIONS", "NUMBA_CACHE_LOCATOR_CLASSES", "NUMBOX_PROXY_CACHE_STRICT"):
        env.pop(name, None)
    done = subprocess.run([sys.executable, str(tmp_path / "forking.py")], capture_output=True, text=True, env=env,
                          cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.splitlines()[-1]) == {
        "parent": ["first"], "forked child": ["second"], "parent after the fork": []}


def test_a_function_whose_save_was_skipped_still_loads_the_entries_that_the_directory_holds(tmp_path, monkeypatch):
    """The build caches two signatures. The run calls a third signature before the two cached signatures, and numbox
    skips saving the third signature."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setattr(cache_key, "_unsaved_reported", False)
    build = _module_whose_function_adds_a_global(tmp_path, "scaled_build")
    assert build.add_g(1) == 2
    assert build.add_g(1.5) == 2.5
    run = _module_whose_function_adds_a_global(tmp_path, "scaled_run")

    def refused(self, sig, data):
        raise OSError(errno.EROFS, os.strerror(errno.EROFS), os.path.join(self.cache_path, "x.nbi.tmp.0123"))

    monkeypatch.setattr(caching.Cache, "_save_overload", refused)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert run.add_g(True) == 2
        assert run.add_g(1) == 2
        assert run.add_g(1.5) == 2.5
    assert len(caught) == 1, [str(warning.message) for warning in caught]
    assert sum(run.add_g.stats.cache_hits.values()) == 2


@pytest.mark.parametrize("number", [errno.ENOSPC, errno.EIO, errno.ENAMETOOLONG])
def test_a_save_that_fails_for_another_reason_still_raises(tmp_path, monkeypatch, number):
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setattr(cache_key, "_unsaved_reported", False)
    add_one = _function_in_a_file_of_its_own(tmp_path)

    def failed(self, sig, data):
        raise OSError(number, os.strerror(number))

    monkeypatch.setattr(caching.Cache, "_save_overload", failed)
    with pytest.raises(OSError) as raised:
        add_one(1)
    assert raised.value.errno == number
    assert cache_key._unsaved_reported is False


def test_a_refused_save_where_the_directory_is_gone_still_raises(tmp_path, monkeypatch):
    """The error does not indicate a read-only cache, because no directory exists to load from."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setattr(cache_key, "_unsaved_reported", False)
    add_one = _function_in_a_file_of_its_own(tmp_path)

    def refused(self, sig, data):
        shutil.rmtree(tmp_path / "cache")
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(caching.Cache, "_save_overload", refused)
    with pytest.raises(OSError) as raised:
        add_one(1)
    assert raised.value.errno == errno.EROFS


@needs_a_directory_it_cannot_write
def test_a_recompile_in_a_directory_that_cannot_be_written_compiles_without_the_cache(tmp_path, monkeypatch):
    """numba's ``recompile`` empties the function's index before it compiles the function again, and that write
    fails in such a directory. Without numbox's wrapper, ``recompile`` raises the error, and the next call loads the
    entry that the build saved."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "numbox.core.read_only_cache.ReadOnlyCacheLocator",
                        raising=False)
    monkeypatch.setattr(cache_key, "_unsaved_reported", False)
    assert _module_whose_function_adds_a_global(tmp_path, "scaled_build").add_g(1) == 2
    before = _files(tmp_path / "cache")
    with _read_only(tmp_path / "cache"):
        run = _module_whose_function_adds_a_global(tmp_path, "scaled_run")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert run.add_g(1) == 2
            assert sum(run.add_g.stats.cache_hits.values()) == 1
            run.G = 10
            run.add_g.recompile()
            assert run.add_g(1) == 11
            # The function's cache is disabled, so numba attempts no save for a new signature either.
            assert run.add_g(1.5) == 11.5
        assert caught == []
        assert _files(tmp_path / "cache") == before


@pytest.mark.parametrize("number", [errno.EROFS, errno.EACCES, errno.EPERM])
def test_a_recompile_whose_index_the_directory_refuses_to_empty_compiles_without_the_cache(
        tmp_path, monkeypatch, number):
    """The directory is writable in this test, so a file that changes shows that the function's cache was left
    enabled."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    module = _module_whose_function_adds_a_global(tmp_path, "scaled")
    assert module.add_g(1) == 2
    before = _files(tmp_path / "cache")

    def refused(self):
        raise OSError(number, os.strerror(number))

    monkeypatch.setattr(caching.IndexDataCacheFile, "flush", refused)
    module.G = 10
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module.add_g.recompile()
        assert module.add_g(1) == 11
    assert caught == []
    assert _files(tmp_path / "cache") == before


def test_explain_cache_key_raises_for_a_function_whose_cache_a_recompile_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    source = tmp_path / "keyed.py"
    source.write_text("from numbox.core.cache_key import numbox_njit\n\n\n"
                      "@numbox_njit(cache=True)\ndef add_one(x):\n    return x + 1\n")
    spec = importlib.util.spec_from_file_location("keyed", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.add_one(1) == 2
    assert set(cache_key.explain_cache_key(module.add_one)) == {"numbox", "flags", "settings"}

    def refused(self):
        raise OSError(errno.EROFS, os.strerror(errno.EROFS))

    monkeypatch.setattr(caching.IndexDataCacheFile, "flush", refused)
    module.add_one.recompile()
    with pytest.raises(ValueError, match="compiles without an on-disk cache: its cache is disabled"):
        cache_key.explain_cache_key(module.add_one)


@needs_a_directory_it_cannot_write
def test_a_recompile_in_a_directory_that_cannot_be_searched_still_raises(tmp_path, monkeypatch):
    """numba cannot open a cache file in such a directory, so the error does not indicate a cache that numba can
    load and cannot write."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    module = _module_whose_function_adds_a_global(tmp_path, "scaled")
    assert module.add_g(1) == 2
    location = Path(module.add_g.stats.cache_path)
    location.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            module.add_g.recompile()
    finally:
        location.chmod(0o755)


@pytest.mark.parametrize("number", [errno.ENOSPC, errno.EIO])
def test_a_recompile_whose_index_cannot_be_emptied_for_another_reason_still_raises(tmp_path, monkeypatch, number):
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    module = _module_whose_function_adds_a_global(tmp_path, "scaled")
    assert module.add_g(1) == 2

    def failed(self):
        raise OSError(number, os.strerror(number))

    monkeypatch.setattr(caching.IndexDataCacheFile, "flush", failed)
    with pytest.raises(OSError) as raised:
        module.add_g.recompile()
    assert raised.value.errno == number
    # The function's cache remains enabled, so the next call loads the entry that the index still lists.
    assert module.add_g(1) == 2
    assert sum(module.add_g.stats.cache_hits.values()) == 1


@needs_a_directory_it_cannot_write
@pytest.mark.parametrize("built_with_such_code", [True, False])
@pytest.mark.parametrize("locators", [
    "numbox.core.read_only_cache.ReadOnlyCacheLocator",
    f"InTreeCacheLocator,{__name__}.TakesUntestedLocator",
    f"{__name__}.UserProvidedCacheLocator",
], ids=["numbox's locator", "a user-defined locator", "a user-defined locator with the same name as numba's class"])
def test_generated_code_whose_file_is_missing_from_a_read_only_directory_is_told_to_build_again(
        tmp_path, monkeypatch, built_with_such_code, locators):
    """The warning names the directory that refused the write: the directory of the generated files, or the cache
    directory if the cache was built without any generated code of this kind. numbox gives this remedy if
    ``NUMBA_CACHE_LOCATOR_CLASSES`` lists a class that is not one of numba's own."""
    cache = tmp_path / "cache"
    anchors = cache / "numbox-structref"
    (anchors if built_with_such_code else cache).mkdir(parents=True)
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    path = anchors / "Struct_0123456789abcdef.py"
    with _read_only(cache):
        with pytest.warns(RuntimeWarning) as caught:
            options = _anchored_or_uncached(path, "x = 1\n", {"cache": True})
    assert options == {"cache": False} and not path.exists()
    assert len(caught) == 1
    message = str(caught[0].message)
    assert message.startswith("numba cannot cache Struct_0123456789abcdef.py here ([Errno 13] Permission denied: ")
    refused = anchors if built_with_such_code else cache
    assert (f"it compiles without a cache. {refused} is not writable and holds no file for this code. Build the "
            "cache again with a run that generates this code, set NUMBA_CACHE_DIR to a writable directory, or "
            "compile without a cache to silence this warning") in message


@pytest.mark.parametrize("number", [errno.EROFS, errno.EACCES, errno.EPERM])
def test_generated_code_whose_write_fails_with_each_error_of_a_refused_write_is_told_to_build_again(
        tmp_path, monkeypatch, number):
    """``EROFS`` is the error on a file system that is mounted read-only."""
    import numbox.utils.preprocessing as preprocessing

    cache = tmp_path / "cache"
    anchors = cache / "numbox-structref"
    anchors.mkdir(parents=True)
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "numbox.core.read_only_cache.ReadOnlyCacheLocator",
                        raising=False)
    monkeypatch.setattr(preprocessing.tempfile, "mkstemp", _refusing(number, cache))
    path = anchors / "Struct_0123456789abcdef.py"
    with pytest.warns(RuntimeWarning) as caught:
        options = _anchored_or_uncached(path, "x = 1\n", {"cache": True})
    assert options == {"cache": False} and not path.exists()
    assert len(caught) == 1
    assert (f"it compiles without a cache. {anchors} is not writable and holds no file for this code. Build the "
            "cache again with a run that generates this code") in str(caught[0].message)


@needs_a_directory_it_cannot_write
@pytest.mark.parametrize("cache_dir_is_set, locators", [
    (True, ""),
    (True, "UserProvidedCacheLocator,InTreeCacheLocator"),
    (True, "numba.core.caching.UserProvidedCacheLocator,numba.core.caching.InTreeCacheLocator"),
    (True, "UserProvidedCacheLocator,InTreeCacheLocatorFsAgnostic"),
    (False, "numbox.core.read_only_cache.ReadOnlyCacheLocator"),
], ids=["numba's own locators", "numba's own locators, listed", "numba's own locators, listed by their paths",
        "a locator of numba's that numba tries only when it is listed", "NUMBA_CACHE_DIR unset"])
def test_generated_code_outside_the_setup_of_a_read_only_cache_keeps_its_remedy(
        tmp_path, monkeypatch, cache_dir_is_set, locators):
    """numba's own locators do not load the cache of a generated file from a directory that cannot be written, so numba
    would not load a cache that is built there again. That holds whether numba tries them in its own order or
    ``NUMBA_CACHE_LOCATOR_CLASSES`` lists them. numbox's locator takes no file while ``NUMBA_CACHE_DIR`` is unset."""
    cache = tmp_path / "cache"
    anchors = cache / "numbox-structref"
    anchors.mkdir(parents=True)
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache) if cache_dir_is_set else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    path = anchors / "Struct_0123456789abcdef.py"
    with _read_only(cache):
        with pytest.warns(RuntimeWarning) as caught:
            options = _anchored_or_uncached(path, "x = 1\n", {"cache": True})
    assert options == {"cache": False} and not path.exists()
    assert len(caught) == 1
    message = str(caught[0].message)
    assert message.startswith("numba cannot cache Struct_0123456789abcdef.py here ([Errno 13] Permission denied: ")
    assert "holds no file for this code" not in message
    assert "it compiles without a cache. Set NUMBA_CACHE_DIR to a writable directory, or compile" in message


@needs_a_directory_it_cannot_write
def test_generated_code_keeps_its_remedy_under_a_user_defined_locator_that_the_variable_does_not_list(
        tmp_path, monkeypatch):
    """A program can add its own locator class to the list that numba uses while ``NUMBA_CACHE_LOCATOR_CLASSES`` is
    unset. numbox gives the remedy of building the cache again only if the variable lists such a class."""
    cache = tmp_path / "cache"
    anchors = cache / "numbox-structref"
    anchors.mkdir(parents=True)
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    monkeypatch.setattr(caching.CacheImpl, "_locator_classes",
                        [TakesUntestedLocator, *caching.CacheImpl._locator_classes])
    path = anchors / "Struct_0123456789abcdef.py"
    with _read_only(cache):
        with pytest.warns(RuntimeWarning) as caught:
            options = _anchored_or_uncached(path, "x = 1\n", {"cache": True})
    assert options == {"cache": False} and not path.exists()
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "holds no file for this code" not in message
    assert "it compiles without a cache. Set NUMBA_CACHE_DIR to a writable directory, or compile" in message


@needs_a_directory_it_cannot_write
def test_generated_code_whose_file_was_written_is_not_told_that_the_directory_holds_none(tmp_path, monkeypatch):
    """In this test, numbox writes the file, and numba's location for the file cannot be created. That error is
    also the error of a refused write, and the directory of the generated files exists, but the directory holds the
    file."""
    cache = tmp_path / "cache"
    anchors = cache / "numbox-structref"
    anchors.mkdir(parents=True)
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", f"{__name__}.TakesUntestedLocator", raising=False)
    path = anchors / "Struct_0123456789abcdef.py"
    # The test removes write access from the cache directory alone. numbox can write in the directory of the generated
    # files below it, and numba cannot create its own subdirectory beside that one.
    cache.chmod(0o555)
    try:
        with pytest.warns(RuntimeWarning) as caught:
            options = _anchored_or_uncached(path, "x = 1\n", {"cache": True})
    finally:
        cache.chmod(0o755)
    assert options == {"cache": False} and path.read_text() == "x = 1\n"
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "holds no file for this code" not in message
    assert "it compiles without a cache. Set NUMBA_CACHE_DIR to a writable directory, or compile" in message


@pytest.mark.parametrize("failure", ["a full disk", "a file in the place of the cache directory"])
def test_generated_code_whose_file_fails_for_another_reason_keeps_its_remedy(tmp_path, monkeypatch, failure):
    """The file is also missing in this test, but the error is not that of a directory that refuses a write."""
    import numbox.utils.preprocessing as preprocessing

    cache = tmp_path / "cache"
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "numbox.core.read_only_cache.ReadOnlyCacheLocator",
                        raising=False)
    path = cache / "numbox-structref" / "Struct_0123456789abcdef.py"
    if failure == "a full disk":
        def full(*args, **kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(preprocessing.tempfile, "mkstemp", full)
    else:
        cache.write_text("")
    with pytest.warns(RuntimeWarning) as caught:
        options = _anchored_or_uncached(path, "x = 1\n", {"cache": True})
    assert options == {"cache": False} and not path.exists()
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "holds no file for this code" not in message
    assert "it compiles without a cache. Set NUMBA_CACHE_DIR to a writable directory, or compile" in message


@needs_a_directory_it_cannot_write
def test_generated_code_in_a_directory_that_cannot_be_searched_keeps_its_remedy(tmp_path, monkeypatch):
    """numbox cannot look up a file in a directory without search permission, so the warning does not state that the
    directory holds no file for the code."""
    cache = tmp_path / "cache"
    anchors = cache / "numbox-structref"
    anchors.mkdir(parents=True)
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "numbox.core.read_only_cache.ReadOnlyCacheLocator",
                        raising=False)
    path = anchors / "Struct_0123456789abcdef.py"
    anchors.chmod(0o600)
    try:
        with pytest.warns(RuntimeWarning) as caught:
            options = _anchored_or_uncached(path, "x = 1\n", {"cache": True})
    finally:
        anchors.chmod(0o755)
    assert options == {"cache": False} and not path.exists()
    assert len(caught) == 1
    message = str(caught[0].message)
    assert message.startswith("numba cannot cache Struct_0123456789abcdef.py here ([Errno 13] Permission denied: ")
    assert "holds no file for this code" not in message
    assert "it compiles without a cache. Set NUMBA_CACHE_DIR to a writable directory, or compile" in message


@needs_a_directory_it_cannot_write
def test_generated_code_in_a_read_only_directory_that_holds_its_file_stays_cached(tmp_path, monkeypatch):
    """The file was written while the cache was built, so nothing is written now and no warning is given."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    path = tmp_path / "cache" / "numbox-structref" / "Struct_0123456789abcdef.py"
    assert _anchored_or_uncached(path, "x = 1\n", {"cache": True}) == {"cache": True}
    before = _files(tmp_path / "cache")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "numbox.core.read_only_cache.ReadOnlyCacheLocator",
                        raising=False)
    with _read_only(tmp_path / "cache"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _anchored_or_uncached(path, "x = 1\n", {"cache": True}) == {"cache": True}
    assert _files(tmp_path / "cache") == before


# The package that the child processes import in addition to numbox. It makes one struct at import, whose anchor the
# build writes, and it can make a second one on request, whose anchor the build never writes. f is cached by numba
# alone, and g compiles under numbox's own options and calls a binding of numbox.
PACKAGE = '''from numba import njit
from numba.core.types import StructRef
from numba.experimental.structref import register

from numbox.core.bindings.libm import cos
from numbox.core.configurations import jit_options
from numbox.utils.highlevel import make_structref


@register
class BuiltTypeClass(StructRef):
    pass


@register
class LaterTypeClass(StructRef):
    pass


make_built = make_structref("Built", ("x", "y"), BuiltTypeClass)


def make_later():
    return make_structref("Later", ("a", "b", "c"), LaterTypeClass)


@njit(cache=True)
def f(x):
    return x + 1


@njit(**jit_options)
def g(x):
    return cos(x)
'''

# The child prints one JSON object: what the calls returned, how often each function was loaded from the cache and
# compiled, and every warning. "misses" calls f with two signatures that the build did not compile, and "later" makes
# the struct that the build did not make.
CHILD = '''import json
import sys
import warnings


def counts(function):
    stats = function.stats
    return [sum(stats.cache_hits.values()), sum(stats.cache_misses.values()), stats.cache_path]


seen = {}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from numbox.core.bindings.libm import cos
    from numbox.core.configurations import jit_options
    from numbox.utils.lowlevel import get_str_from_p_as_int
    from userpkg import mod
    seen["options"] = jit_options
    seen["f"] = mod.f(1)
    seen["g"] = mod.g(0.0)
    built = mod.make_built(3.5, 2)
    seen["built"] = [built.x, built.y]
    if "misses" in sys.argv:
        seen["misses"] = [mod.f(1.5), mod.f(True)]
    if "later" in sys.argv:
        later = mod.make_later()(1.5, 2, 3)
        seen["later"] = [later.a, later.b, later.c]
    seen["f counts"] = counts(mod.f)
    seen["g counts"] = counts(mod.g)
    seen["cos counts"] = counts(cos)
    seen["get_str counts"] = counts(get_str_from_p_as_int)
seen["warnings"] = [str(warning.message) for warning in caught]
print(json.dumps(seen))
'''

# A module that uses numba alone, and a child that imports nothing else. The child lists the modules of numbox that
# its process imported. The list is empty unless numba imports numbox's locator.
PLAIN = '''from numba import njit


@njit(cache=True)
def k(x):
    return x * 2
'''

PLAIN_CHILD = '''import json
import sys
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from userpkg import plain
    seen = {"k": plain.k(1)}
    if "miss" in sys.argv:
        seen["miss"] = plain.k(1.5)
    seen["k counts"] = [sum(plain.k.stats.cache_hits.values()), sum(plain.k.stats.cache_misses.values())]
seen["numbox modules"] = sorted(name for name in sys.modules if name.split(".")[0] == "numbox")
seen["warnings"] = [str(warning.message) for warning in caught]
print(json.dumps(seen))
'''

THEIR_LOCATORS = '''from numba.core.caching import UserProvidedCacheLocator


class NeverTestsLocator(UserProvidedCacheLocator):

    def ensure_cache_path(self):
        pass


class TakesUntestedLocator(UserProvidedCacheLocator):

    @classmethod
    def from_function(cls, py_func, py_file):
        return cls(py_func, py_file)
'''


class _Built:
    """A copy of numbox, the package and a cache built from them, with the children that run from them."""

    def __init__(self, tmp_path, locators=None):
        self.root = tmp_path
        self.site, self.home, self.cache = tmp_path / "site", tmp_path / "home", tmp_path / "cache"
        shutil.copytree(REPO / "numbox", self.site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
        (self.site / "userpkg").mkdir()
        (self.site / "userpkg" / "__init__.py").write_text("")
        (self.site / "userpkg" / "mod.py").write_text(PACKAGE)
        (self.site / "userpkg" / "plain.py").write_text(PLAIN)
        (self.site / "their_locators.py").write_text(THEIR_LOCATORS)
        (tmp_path / "child.py").write_text(CHILD)
        (tmp_path / "plain_child.py").write_text(PLAIN_CHILD)
        (tmp_path / "cwd").mkdir()
        self.home.mkdir()
        self.built = self.run(locators=locators)
        assert self.built["options"] == {"cache": True} and not self.built["warnings"], self.built
        # numba wrote into the cache directory alone: the sources and the home hold no cache file.
        assert not list(self.site.rglob("*.nbi")) and not list(self.home.rglob("*.nbi"))
        self.files = _files(self.cache)

    def run(self, *arguments, locators=None, script="child.py"):
        env = dict(os.environ, PYTHONPATH=str(self.site), HOME=str(self.home),
                   XDG_CACHE_HOME=str(self.home / "cache"), NUMBA_CACHE_DIR=str(self.cache))
        for name in ("NUMBOX_JIT_OPTIONS", "NUMBA_CACHE_LOCATOR_CLASSES", "NUMBOX_PROXY_CACHE_STRICT"):
            env.pop(name, None)
        if locators:
            env["NUMBA_CACHE_LOCATOR_CLASSES"] = locators
        # The working directory is outside the checkout, so the child imports numbox from the copy.
        done = subprocess.run([sys.executable, str(self.root / script), *arguments],
                              capture_output=True, text=True, env=env, cwd=str(self.root / "cwd"))
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout.splitlines()[-1])

    def read_only(self):
        return _read_only(self.cache, self.site, self.home)

    def unchanged(self):
        """Return whether the cache holds exactly the files that the build left, and whether the sources and the home
        hold no cache file."""
        return (_files(self.cache) == self.files and not list(self.site.rglob("*.nbi"))
                and not list(self.home.rglob("*.nbi")))


@needs_a_directory_it_cannot_write
@pytest.mark.parametrize("built_with", [None, SHIPPED], ids=["built with numba's locators", "built with the locator"])
def test_a_cache_built_writable_is_loaded_from_the_same_directory_read_only(tmp_path, built_with):
    """Under numbox's locator, every function loads from the directory, no warning is issued and nothing is
    written."""
    built = _Built(tmp_path, locators=built_with)
    assert built.built["f counts"][:2] == [0, 1] and built.built["cos counts"][:2] == [0, 1]
    with built.read_only():
        run = built.run(locators=SHIPPED)
        assert built.unchanged()
    assert run["warnings"] == []
    assert run["options"] == {"cache": True}
    assert (run["f"], run["g"], run["built"]) == (2, 1.0, [3.5, 2])
    for function in ("f", "g", "cos", "get_str"):
        hits, misses, location = run[f"{function} counts"]
        assert (hits, misses) == (1, 0), (function, run)
        assert location.startswith(str(built.cache) + os.sep), (function, location)


@needs_a_directory_it_cannot_write
def test_a_subdirectory_that_cannot_be_searched_is_skipped_and_numba_caches_elsewhere(tmp_path):
    """numba cannot open the cache files of the subdirectory that holds numbox's bindings. The locator skips that
    subdirectory, and numba caches the bindings beside their sources, which can be written in this test."""
    built = _Built(tmp_path)
    (bindings,) = [path for path in built.cache.iterdir() if path.name.startswith("bindings_")]
    with _read_only(built.cache):
        bindings.chmod(0o000)
        try:
            run = built.run(locators=SHIPPED)
        finally:
            bindings.chmod(0o555)
        assert _files(built.cache) == built.files
    assert run["warnings"] == [] and run["options"] == {"cache": True}
    assert (run["f"], run["g"]) == (2, 1.0)
    assert run["f counts"][:2] == [1, 0] and run["f counts"][2].startswith(str(built.cache) + os.sep)
    assert run["cos counts"][:2] == [0, 1]
    assert run["cos counts"][2] == str(built.site / "numbox" / "core" / "bindings" / "__pycache__")


@needs_a_directory_it_cannot_write
def test_a_signature_that_the_cache_lacks_compiles_and_warns_once(tmp_path):
    """Without numbox's wrapper, numba raises the error of the save from the call that compiled the function."""
    built = _Built(tmp_path)
    with built.read_only():
        run = built.run("misses", locators=SHIPPED)
        assert built.unchanged()
    assert run["misses"] == [2.5, 2]
    assert run["f counts"][:2] == [1, 2]
    location = run["f counts"][2]
    assert run["warnings"] == [
        f"numbox: numba compiled f but cannot save its cache entry in the cache directory {location} (Permission "
        "denied), so the function compiles again in every process. numbox skips each save that fails in this way. "
        "It warns only once per process, for the first skipped save."
    ]


@needs_a_directory_it_cannot_write
def test_the_locator_brings_the_skipped_save_to_a_program_that_never_imports_numbox(tmp_path):
    """numba imports the locator's module to resolve the class, and that import installs numbox's wrapper around
    numba's save. The program here imports numba alone, and a signature that the cache lacks still compiles and
    returns. The import does not check numbox's own files, because another module runs that check."""
    built = _Built(tmp_path)
    first = built.run(script="plain_child.py")
    assert first["k counts"] == [0, 1] and first["numbox modules"] == []
    built.files = _files(built.cache)
    with built.read_only():
        run = built.run("miss", locators=SHIPPED, script="plain_child.py")
        assert built.unchanged()
    assert (run["k"], run["miss"], run["k counts"]) == (2, 3.0, [1, 1])
    assert "numbox.core.cache_key" in run["numbox modules"]
    assert "numbox.core.configurations" not in run["numbox modules"]
    assert len(run["warnings"]) == 1, run
    assert run["warnings"][0].startswith(
        "numbox: numba compiled k but cannot save its cache entry in the cache directory ")


@needs_a_directory_it_cannot_write
@pytest.mark.parametrize("locator", ["NeverTestsLocator", "TakesUntestedLocator"])
def test_a_user_defined_locator_keeps_numbox_cached(tmp_path, locator):
    """A user can list a user-defined locator that takes the directory, as numba's maintainers suggest. numbox then
    loads its own functions from the directory too."""
    built = _Built(tmp_path)
    with built.read_only():
        run = built.run("misses", locators=f"their_locators.{locator}")
        assert built.unchanged()
    assert run["options"] == {"cache": True}
    assert (run["f"], run["g"], run["built"], run["misses"]) == (2, 1.0, [3.5, 2], [2.5, 2])
    for function in ("g", "cos", "get_str"):
        assert run[f"{function} counts"][:2] == [1, 0], (function, run)
    assert len(run["warnings"]) == 1, run
    assert "numbox: numba compiled f but cannot save its cache entry" in run["warnings"][0], run


@needs_a_directory_it_cannot_write
def test_numbas_own_locators_never_read_a_cache_directory_that_cannot_be_written(tmp_path):
    """Without a locator that takes the directory, numba skips it, and numbox compiles without a cache.

    numba raises an error when it decorates the package's own function, which numba alone caches, so this child
    imports numbox alone.
    """
    built = _Built(tmp_path)
    (tmp_path / "numbox_alone.py").write_text(
        "import json\nimport warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    from numbox.core.bindings.libm import cos\n"
        "    from numbox.core.configurations import jit_options\n"
        "print(json.dumps({'options': jit_options, 'cos': cos(0.0),\n"
        "                  'warnings': [str(warning.message) for warning in caught]}))\n")
    with built.read_only():
        run = built.run(script="numbox_alone.py")
        assert built.unchanged()
    assert run["options"] == {"cache": False} and run["cos"] == 1.0
    assert len(run["warnings"]) == 1, run
    assert "it compiles without a cache" in run["warnings"][0]
    assert f"NUMBA_CACHE_DIR is set to {built.cache}, which numba could not use" in run["warnings"][0]


@needs_a_directory_it_cannot_write
def test_generated_code_that_the_cache_was_not_built_with_compiles_uncached_and_says_so(tmp_path):
    """The struct that the build generated loads from the cache. numbox cannot write the other struct's anchor."""
    built = _Built(tmp_path)
    with built.read_only():
        run = built.run("later", locators=SHIPPED)
        assert built.unchanged()
    assert run["built"] == [3.5, 2] and run["later"] == [1.5, 2, 3]
    assert len(run["warnings"]) == 1, run
    anchors = built.cache / "numbox-structref"
    warning = run["warnings"][0]
    assert warning.startswith("numba cannot cache Later_") and "it compiles without a cache" in warning
    assert (f"{anchors} is not writable and holds no file for this code. Build the cache again with a run that "
            "generates this code, set NUMBA_CACHE_DIR to a writable directory, or compile without a cache to "
            "silence this warning") in warning


@needs_a_directory_it_cannot_write
def test_only_the_cache_read_only_under_numbas_locators_caches_elsewhere(tmp_path):
    """numba's locators skip the directory, and numba uses the next location, so numba does not load from the cache
    that was built."""
    built = _Built(tmp_path)
    with _read_only(built.cache):
        run = built.run()
        assert _files(built.cache) == built.files
    assert run["options"] == {"cache": True} and run["warnings"] == []
    assert run["f counts"][:2] == [0, 1] and run["cos counts"][:2] == [0, 1]
    assert not run["f counts"][2].startswith(str(built.cache))
    assert list(built.site.rglob("*.nbi"))
