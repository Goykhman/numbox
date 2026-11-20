"""Loading numba's cache from a directory that can be read but not written.

A cache that was built in ``NUMBA_CACHE_DIR`` can be deployed read-only, for example in a container image, so that no
process compiles when it starts. numba alone does not load from such a cache. Its locator for ``NUMBA_CACHE_DIR``
creates a temporary file in the directory when numba decorates a function, and numba skips the directory if the file
cannot be created. numba also raises the ``OSError`` of a failed save from the call that compiled the function, except
for ``EACCES`` on Windows, which it ignores.

:class:`ReadOnlyCacheLocator` is a subclass of that locator that also takes a directory that exists and cannot be
written. To use it, list it in numba's ``NUMBA_CACHE_LOCATOR_CLASSES`` as
``numbox.core.read_only_cache.ReadOnlyCacheLocator``, followed by the locators that numba should try for a file that
this locator does not take. The configurations page states the complete value.

numbox recognizes such a directory by the error of a write that fails in it (:func:`cannot_be_written`). Its check of
a cache location passes a location in which a write fails with such an error
(:func:`~numbox.core.configurations.check_cache_location`). The wrapper that
:mod:`numbox.core.cache_key` installs around numba's save skips a save that fails with such an error, for every
function that numba caches in the process. numba's ``Dispatcher.recompile`` empties the function's index before it
compiles the function again. A second wrapper disables the function's cache if that write fails with such an error,
and numba then compiles the function without loading or saving an entry. numba resolves the locator by importing this
module, which installs both wrappers.
"""
import errno
import os

from numba.core.caching import UserProvidedCacheLocator


# A write fails with one of these errors in a directory that the process may read. EROFS is the error on a file system
# that is mounted read-only. EACCES and EPERM are the errors where the directory's permissions or flags deny the write.
_CANNOT_WRITE = (errno.EROFS, errno.EACCES, errno.EPERM)


def cannot_be_written(error, location):
    """Return whether ``error`` is the error of a write that the operating system denied in the directory ``location``.

    ``error`` is such an error if it is an ``OSError`` whose ``errno`` is ``EROFS``, ``EACCES`` or ``EPERM``. The
    function reads the error number and does not test whether the process can write in ``location``, so the caller
    must pass the error of a write that failed there. A single write to a directory that the process can otherwise
    write in can fail with one of these errors, for example the replacement of another user's file in a directory
    that has the sticky bit. The result is True for that error as well.

    A location that does not exist, or that is not a directory, holds no cache to load, so the result is False for it.
    The result is also False for a directory that the process cannot search, because numba cannot open a cache file
    in it. Any other ``OSError``, such as a full disk or a name that is too long, also gives False.
    """
    return (isinstance(error, OSError) and error.errno in _CANNOT_WRITE and os.path.isdir(location)
            and os.access(location, os.X_OK))


class ReadOnlyCacheLocator(UserProvidedCacheLocator):
    """numba's locator for ``NUMBA_CACHE_DIR``, which also takes a directory that exists and cannot be written.

    numba's ``UserProvidedCacheLocator`` caches a source file in a subdirectory of ``NUMBA_CACHE_DIR`` that numba
    names after the file's directory. It skips the file if it cannot create that subdirectory or cannot create a
    temporary file in it. This locator names the same subdirectory. It differs in one case: it takes the file if the
    subdirectory exists, the process can search it, and the creation of the temporary file fails with an error that
    :func:`cannot_be_written` recognizes. numba then loads the entries that the subdirectory holds. A function that
    the subdirectory holds no entry for compiles in every process. numbox skips the save that numba attempts for it
    and issues one warning per process.

    Where the subdirectory can be written, the locator behaves as numba's does, so the same value of
    ``NUMBA_CACHE_LOCATOR_CLASSES`` serves to build the cache and to run from it. The locator skips a file whose
    subdirectory does not exist and cannot be created, and numba then tries the next locator in the list. The
    locator also skips a file whose subdirectory the process cannot search. numba creates the subdirectory when it
    decorates the first cached function of a source directory, so a cache holds a subdirectory for each source
    directory that was imported while the cache was built.
    """

    def ensure_cache_path(self):
        try:
            super().ensure_cache_path()
        except OSError as error:
            if not cannot_be_written(error, self.get_cache_path()):
                raise
