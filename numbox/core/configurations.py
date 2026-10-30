"""numbox's njit options, and the check that turns caching off where numba cannot cache numbox.

``jit_options`` holds the keyword arguments that numbox passes to numba's ``njit`` for its own functions. When this
module is first imported, it reads them from the ``NUMBOX_JIT_OPTIONS`` environment variable. Unless they set
``"cache"`` to ``false``, it then asks numba whether it could cache a function in each directory of numbox. If numba
could not, ``jit_options`` has ``"cache"`` set to ``False``, and one ``RuntimeWarning`` says what to change.
:func:`check_cache_location`, :func:`is_a_cache_error` and :func:`cache_remedy` run the same check for a package built
on numbox. The module also holds ``MAX_STR_LENGTH``, ``function_struct_size`` and the switch for strict validation of
``@proxy`` cache loads, which other numbox modules read.
"""
import ast
import contextlib
import errno
import importlib.machinery
import inspect
import itertools
import os
import pathlib
import json
import linecache
import re
import sys
import tempfile
import types
import warnings
import zipfile
import zipimport

from importlib.metadata import version

import numba.experimental.function_type  # noqa: F401  registers `FunctionModel` against `FunctionType`
from numba.core.caching import CompileResultCacheImpl
from numba.core.datamodel import default_manager
from numba.misc.appdirs import AppDirs
from numba.core.types import FunctionType, void


numba_version = int(version("numba").split(".")[1])
assert numba_version >= 63, numba_version


def get_jit_options():
    """Return the njit options that ``NUMBOX_JIT_OPTIONS`` sets, or ``{"cache": True}`` if the variable is unset.

    The variable holds a JSON object, whose keys numbox passes to ``njit`` as keyword arguments, as in::

        export NUMBOX_JIT_OPTIONS='{"cache": false}'

    Raises ``ValueError`` if the value is not valid JSON, is not a JSON object, or holds a ``"cache"`` that is not
    ``true`` or ``false``. The variable is read on every call; this module calls the function once, when it is first
    imported, to set ``jit_options``.
    """
    as_str = os.environ.get("NUMBOX_JIT_OPTIONS")
    if as_str is None:
        return {"cache": True}
    try:
        as_json = json.loads(as_str)
    except json.JSONDecodeError:
        raise ValueError("NUMBOX_JIT_OPTIONS must be valid JSON")
    if not isinstance(as_json, dict):
        # njit takes the keys as keyword arguments, and uncached_where_no_cache_can_be_written reads "cache" from the
        # object. A JSON value of another shape is therefore refused here, with a message that shows the expected form.
        raise ValueError('NUMBOX_JIT_OPTIONS must be a JSON object, e.g. {"cache": false}')
    if "cache" in as_json and not isinstance(as_json["cache"], bool):
        # numba tests the option for truth, so the string "false" would leave caching enabled.
        raise ValueError('NUMBOX_JIT_OPTIONS "cache" must be true or false, e.g. {"cache": false}')
    return as_json


def _cache_probe():
    """Never called or compiled: the check asks numba about the file that this function's code records."""


# numba names a cache file after the module's stem, the function's qualified name, the line of its first decorator,
# the interpreter tag and, for a data file, the signature's number. It writes each file first under a temporary name
# 21 bytes longer. A test measures these names against this bound for every function of numbox, and for the graph
# maker that make_graph compiles as a function of builder.py. By that measure, the longest name for a function of
# numbox is 107 bytes, and the graph maker's is 119.
#: The bound, in bytes, on the length of the cache file names that numba writes for numbox's own functions, numba's
#: temporary suffix included. The check creates and removes a file with a name this long in each location that it asks
#: numba about.
LONGEST_CACHE_FILE_NAME = 128

# The digits reserved for the number that numba gives a function's data file. numba writes one data file for each
# signature and numbers the files from 1 without bound. Three digits are reserved, because no function of numbox or of
# its generated code reaches a thousand signatures. If an error names a file with a wider number, cache_remedy reads
# that number at its own width. Two digits would be too few: the name of the hundredth signature's file is one byte
# longer than a two-digit reserve allows.
_OVERLOAD_DIGITS = 3

# Each check puts the probe's module in sys.modules under a name of its own while numba picks the locator. This
# counter makes the names unique.
_probes = itertools.count()


class _MemberSource:
    """The loader of the probe's module: inspect reads a ``.zip`` member's source through it, as through zipimport."""

    def __init__(self, py_file):
        self._py_file = py_file

    def get_source(self, name):
        """Return the source of the ``.zip`` member at the probe's file, decoded as UTF-8, or None.

        zipimport decodes a member's source as UTF-8 too. ``name`` is the module name that linecache passes; the
        probe's file names the member instead. The result is None if no archive above the file holds the member, if
        the archive cannot be read, or if the member is not UTF-8.
        """
        found = _archive_above(self._py_file)
        if found is None:
            return None
        zip_path, member = found
        try:
            with zipfile.ZipFile(zip_path) as archive:
                return archive.read(member).decode()
        except (KeyError, OSError, zipfile.BadZipFile, UnicodeDecodeError):
            # zipimport raises UnicodeDecodeError for a member that is not UTF-8, and linecache lets that error
            # through, so it would end the import that runs the check. None makes inspect raise its own OSError
            # instead. That error is a cache error and marks the member, and cache_remedy reads the reason from the
            # archive.
            return None


@contextlib.contextmanager
def _probing(py_file):
    """Yield a probe, a function that is never compiled and whose code names ``py_file`` as its file.

    ``py_file`` can be a string, bytes or a path object. The caller asks a locator about the file with the probe, as
    numba asks at decoration. The probe belongs to a module of its own, which is in ``sys.modules`` while the block
    runs. When the block ends, the module leaves ``sys.modules``, and a linecache entry that the block left holding the
    probe's loader is removed, so that no loader of the probe outlives the block.
    """
    # compile stores the file name in the probe's code as a string, even when it is given bytes, so py_file is read as
    # a string throughout.
    file = os.fsdecode(py_file)
    # A locator reads only the probe's file name, except numba's IPython locator, which also reads the function's source
    # when it takes the file, as inspect reads it. inspect reads a file on disk from the file itself, and a cell from
    # linecache, where IPython holds the cell while it runs. It reads a .zip member through the loader of the function's
    # module. That loader is zipimport's for a real module, and it gives the member's source from the archive, or None
    # for a member compiled to .pyc alone. The probe's loader reads the member as zipimport's does, and inspect finds
    # the probe's module in sys.modules.
    #
    # The probe's source is never put in linecache under the member's name. A reader of the member in another thread
    # would then read the probe's two lines instead of the member's own.
    module = types.ModuleType(f"_numbox_cache_probe_{next(_probes)}")
    module.__file__ = file
    module.__loader__ = _MemberSource(file)
    exec(compile("def _cache_probe():\n    pass\n", file, "exec"), module.__dict__)  # nosec B102 - fixed source
    held = file in linecache.cache
    sys.modules[module.__name__] = module
    try:
        yield module._cache_probe
    finally:
        sys.modules.pop(module.__name__, None)
        # After a read, linecache holds the member's own lines for the file, as any reader would leave them. linecache
        # keeps the probe's loader instead, in an entry of one item, for a member whose archive holds no source, so
        # that it can ask again later. That entry is removed here, unless the file was in linecache before the block.
        if not held and len(linecache.cache.get(file, ())) == 1:
            linecache.cache.pop(file, None)


def _takes(cls, py_file):
    """Return whether the locator class ``cls`` takes ``py_file`` when asked with a probe, as numba asks at decoration.

    ``from_function`` returns a locator for a file that the class takes and None for a file that it passes over. A
    class whose ``from_function`` raises counts as taking the file, because numba raises that error at decoration."""
    with _probing(py_file) as probe:
        try:
            return cls.from_function(probe, os.fsdecode(py_file)) is not None
        except Exception:
            return True


def check_cache_location(py_file, longest_file_name=0):
    """Raise the error that numba would raise if it could not cache a function defined in ``py_file``; else return.

    ``py_file`` can be a string, bytes or a path object; it is read as a string. The check repeats the steps that numba
    runs when it decorates a function with ``cache=True`` and when it first saves the function. It runs them for a
    probe whose code names ``py_file`` as its file:

    - It sets up numba's cache, which picks a location for the file. numba raises ``RuntimeError`` if no locator takes
      the file.
    - It reads the source's stamp, as numba does at decoration. For a ``.zip`` member, numba reads the archive's
      modification time and size, and from numba 0.68 it also reads the member's source from the archive. zipfile
      raises ``KeyError`` if the archive holds no source under the member's name. From numba 0.68, numba also reads a
      source on disk, to hash it.
    - It runs numba's test that the location can be written in. numba runs that test at decoration for every location
      except a ``.zip``'s, and at the first save for every location.
    - It asks numba for the location again, as numba asks at every save, and raises ``FileNotFoundError`` if the
      location is not there.
    - If ``longest_file_name`` is not 0, it creates and removes a file with a name of that many bytes in the location.

    Nothing is compiled, and no file is left behind. The directories that numba creates while it tries the locations
    stay. An IPython that can write neither its cache directory nor its own also creates a new temporary directory on
    every call, and the check causes three such calls.

    An ``OSError`` from the check names the location that numba picked, except in three cases:

    - inspect's error names no file. numba's IPython locator reads the function's source before it tries its location,
      and inspect raises if it finds no source. That happens for a cell that linecache does not hold, for a ``.zip``
      member compiled to ``.pyc`` alone or whose source is not UTF-8, and for a cell file on disk that cannot be read.
      An empty cell file raises too before Python 3.13; from 3.13, linecache holds an empty file as one blank line.
    - The stamp's error names the archive that a ``.zip`` member's code names, if that archive has moved or cannot be
      read. From numba 0.68, it can also name the source, if numba cannot read it, or a directory named ``.zip`` above a
      file on disk, which numba's ``.zip`` locator takes for the archive and opens.
    - If IPython gives a new temporary directory on every call, numba creates ``numba_cache`` in one directory and
      would save in another. The ``FileNotFoundError`` then names ``numba_cache`` under the last directory that IPython
      gave, which numba never created. numba's first save would instead name a file in ``numba_cache`` under the
      directory that IPython gave numba when it set the cache up.

    numba's own test creates a temporary file, which on Linux is usually unnamed and elsewhere has a short name. The
    cache files that numba saves later have names that reach a hundred bytes and more. A location close to the path
    limit therefore passes numba's test and fails at the first save. ``longest_file_name`` closes that gap; numbox
    passes ``LONGEST_CACHE_FILE_NAME``. If a name of that length is longer than the file system holding the location
    allows, no shorter path helps. The ``OSError`` then carries the length as its ``longest_file_name`` attribute, and
    :func:`cache_remedy` asks for a file system that allows such a name.
    """
    # numba is asked about a probe, a function in a module of its own, as it is asked about a decorated function.
    with _probing(py_file) as probe:
        locator = CompileResultCacheImpl(probe).locator
    # numba reads the source's stamp at decoration too. For a .zip member, the stamp reads the archive, which raises if
    # the member's code names an archive that has since moved. The stamp hashes the source from numba 0.68, so it also
    # raises for a source that cannot be read.
    locator.get_source_stamp()
    try:
        locator.ensure_cache_path()
    except OSError as error:
        # The error names whatever failed, possibly a temporary file. The reader can act on the location that holds
        # it, so the error raised here names the location.
        raise OSError(error.errno, error.strerror, locator.get_cache_path()) from error
    # The location is asked for once more, as numba asks at every save. numba's IPython locator asks IPython for it.
    # An IPython that can write neither its cache directory nor its own gives a new temporary directory on every call,
    # so numba creates numba_cache in one and saves in another. That save fails although numba's own test passed, and
    # without this question the check would pass the file the same way. numba's ensure_cache_path asks once at
    # decoration and once more above, and the check asks a third time here; each answer is a new temporary directory.
    location = locator.get_cache_path()
    if not os.path.isdir(location):
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", location)
    if longest_file_name:
        try:
            # tempfile appends eight random characters to the prefix.
            with tempfile.NamedTemporaryFile(dir=location, prefix="x" * (longest_file_name - 8)):
                pass
        except OSError as error:
            raised = OSError(error.errno, error.strerror, location)
            if error.errno == errno.ENAMETOOLONG and _name_limit(location) < longest_file_name:
                # The name alone is too long for the file system, whatever the path. The error carries the name's
                # length, so that cache_remedy does not read the error as the location's and offer a shorter path.
                raised.longest_file_name = longest_file_name
            raise raised from error


def is_a_cache_error(error):
    """Return whether ``error`` is one of numba's errors for a cache that numba cannot set up.

    Those errors are:

    - any ``OSError``;
    - numba's ``RuntimeError`` for a file that no locator takes, whose message contains ``no locator available``;
    - numba's ``ValueError`` ``No zip file found in path``, which numba's ``.zip`` locator raises for a path that
      contains ``.zip`` but has no part that ends in ``.zip``;
    - from numba 0.68, zipfile's ``KeyError`` ``There is no item named ...``, which reaches the caller when the archive
      holds no source under a ``.zip`` member's name. numba reads that source for the member's stamp. The name is
      missing for a module shipped as ``.pyc`` alone, and on Windows under numba 0.68.0 for every member, because that
      release builds the name with the platform's separator (numba issue 10889, fixed in 0.68.1);
    - the ``UnicodeEncodeError`` ``surrogates not allowed``, which numba raises for a source directory whose path is
      not UTF-8, when it hashes the path to name its directory under ``NUMBA_CACHE_DIR`` or the user's cache directory.

    Any other error, a ``UnicodeEncodeError`` for another reason included, is not a cache error.
    """
    if isinstance(error, OSError):
        # numba itself passes over a location on any OSError from creating its directory or writing a file there:
        # permission denied, a read-only file system, a path through a file, a component too long, a full disk.
        return True
    if isinstance(error, UnicodeEncodeError):
        # A UnicodeEncodeError is a ValueError, so it is tested first. A path whose bytes are not UTF-8 is decoded
        # with surrogates, which UTF-8 cannot encode. numba names its directory under NUMBA_CACHE_DIR or the user's
        # cache directory with a hash of the path encoded as UTF-8. A locator that names its directory that way raises
        # at decoration when numba reaches it, after the locators before it passed the file over, so no location takes
        # the file. Counting the error lets a package installed under such a directory, with its __pycache__
        # unwritable, import uncached instead of failing at import.
        return error.reason == "surrogates not allowed"
    if isinstance(error, ValueError):
        # numba's .zip locator takes any path that contains ".zip", and it raises this error when no part of the path
        # ends in ".zip", as for an archive under a directory whose name contains ".zip".
        return "No zip file found" in str(error)
    if isinstance(error, KeyError):
        # numba 0.68 and later hash a .zip member's source into the member's stamp, and zipfile raises this error,
        # naming the member, for a name that the archive does not hold.
        return numba_version >= 68 and "There is no item named" in str(error)
    return isinstance(error, RuntimeError) and "no locator available" in str(error)


def _module_files():
    """Yield the files to ask numba about for numbox, this module's own file first.

    For numbox installed in a directory, the files are the first source file of each directory that holds a module,
    and the file that each ``.pyc`` file without its ``.py`` was compiled from. For numbox imported from a ``.zip``,
    they are the files that ``_archived_module_files`` yields. One directory that numba cannot cache in turns caching
    off for all of numbox, even if every directory that holds a cached function is fine. Compiling without a cache
    costs compile time but never gives a wrong result.
    """
    # This module's code can name a file other than its __file__, as any module's code can, and numba looks that file
    # up for a function of this module. That file is asked about first.
    yield inspect.getfile(_cache_probe)
    # numbox is found by this module's __file__, which is where it was imported from, an archive or a directory. An
    # archive has no directories to walk, so its members are listed instead.
    package = os.path.dirname(os.path.dirname(__file__))
    if not os.path.isdir(package):
        yield from _archived_module_files(__file__)
        return
    walked = set()
    for directory, subdirectories, files in os.walk(package, followlinks=True):
        # Symlinks are followed, and a directory reached twice through links, as in a cycle, is walked once.
        real = os.path.realpath(directory)
        if real in walked:
            subdirectories[:] = []
            continue
        walked.add(real)
        subdirectories[:] = sorted(name for name in subdirectories if name != "__pycache__")
        # numba's in-tree cache is the __pycache__ beside each source, so each directory answers on its own, and one
        # can be writable while another is not. The first source file of a directory represents the directory, whether
        # or not its modules cache anything.
        stems = sorted({name.rsplit(".", 1)[0] for name in files if name.endswith((".py", ".pyc"))})
        sources = [os.path.join(directory, stem + ".py") for stem in stems]
        on_disk = [source for source in sources if os.path.exists(source)]
        if on_disk:
            yield on_disk[0]
        # numba finds a location for a module by its code's file. A module present as .pyc alone, beside modules with
        # their source, is asked about by the file that it was compiled from, which its code records. That file is the
        # .py that is gone, if the .pyc was compiled in place, or a file in a tree elsewhere, on disk or not.
        for source in sources:
            if source not in on_disk:
                compiled_from = _sourceless_compiled_from(source[:-3] + ".pyc")
                if compiled_from is not None:
                    yield compiled_from


def _sourceless_compiled_from(pyc_path):
    """Return the file that the ``.pyc`` file at ``pyc_path`` was compiled from, or None if Python cannot run it.

    The import system runs a ``.pyc`` file that has no ``.py`` beside it through its sourceless loader. The loader keeps
    on the code the file name that the ``.pyc`` file was compiled with. It fails the import of a ``.pyc`` file that it
    cannot run, such as one with another interpreter's magic number or an unreadable one, and such a module has nothing
    to ask numba about.
    """
    loader = importlib.machinery.SourcelessFileLoader(os.path.basename(pyc_path)[:-4], pyc_path)
    try:
        return loader.get_code(loader.name).co_filename
    except Exception:
        return None


def _archived_module_files(own):
    """Yield the files to ask numba about for numbox in the ``.zip`` that holds ``own``, this module's ``__file__``.

    The function yields, for each directory of numbox in the archive, the file that one module's code names. It also
    yields the file that the code of a ``.pyc`` member names, if that file is outside the archive or is a member that
    the archive does not hold. Such a ``.pyc`` stands without its ``.py``, or was compiled to name another path.
    Nothing is yielded if no part of ``own`` that ends in ``.zip`` is a zip archive.
    """
    # The archive is the first part of the path that ends in .zip and is a zip archive; a directory so named above it is
    # not the archive. Any other kind of archive has no cache location at all, and the probe's own file, which is asked
    # about first, already fails for it.
    parts = own.split(os.sep)
    for depth, part in enumerate(parts):
        zip_path = os.sep.join(parts[:depth + 1])
        if part.endswith(".zip") and zipfile.is_zipfile(zip_path):
            break
    else:
        return
    package = "/".join(parts[depth + 1:-2])
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(name for name in archive.namelist()
                       if name.startswith(package + "/") and name.endswith((".py", ".pyc"))
                       and "/__pycache__/" not in name)
    held = set(names)
    seen = set()
    # numba caches a .zip per directory, each directory in a location of its own under the user's cache directory.
    # Each directory of the archive therefore answers on its own. A .pyc run from the archive keeps the file that it
    # was compiled from as its code's file. numba looks that file up for the module's functions, not the archive's own
    # path for the module. Such a member is therefore asked about by that file, whether the file exists or not.
    for name in names:
        directory, _, file = name.rpartition("/")
        if file.endswith(".pyc"):
            # If the archive holds the file that the code names, that file represents its directory, as a .py member
            # does. It is the archive's own path for the module when zipimport compiled the .py, or a member that the
            # .pyc was compiled to name. A file outside the archive, on disk or not, is asked about for itself.
            member = _compiled_from(zip_path, directory, file[:-4])
            if member is None:
                continue
            if not member.startswith(zip_path + os.sep):
                yield member
                continue
            named = member[len(zip_path) + 1:].replace(os.sep, "/")
            if named not in held:
                # The member that the code names is not in the archive. A .pyc alone names the .py that it was compiled
                # from, which is gone. A .pyc compiled to name another path in the archive (compileall -d) names a
                # member that was never there, with or without the .py beside it. numba 0.68 and later read the
                # member's source from the archive by that name and find none, so the .pyc is asked about for itself,
                # whichever member represents its directory. If a .py member represented the directory alone, the .pyc
                # would never be asked about, and its module would fail at decoration. A .pyc beside its .py would go
                # unasked the same way if it were assumed to name the archive's own path.
                seen.add(directory)
                yield member
                continue
        else:
            member = os.path.join(zip_path, *name.split("/"))
        if directory not in seen:
            seen.add(directory)
            yield member


def _compiled_from(zip_path, directory, stem):
    """Return the file named by the code that zipimport runs for module ``stem`` of ``directory``, or None.

    zipimport runs the ``.pyc`` rather than the ``.py`` beside it, unless the ``.pyc`` has another interpreter's magic
    number, is stale against the ``.py``, or cannot be unmarshalled. zipimport compiles the source instead in the first
    two cases, and the module's import fails in the last. The file that the code names is the archive's own path for
    the module if zipimport compiled the ``.py``, and otherwise the path that the ``.pyc`` was compiled to name. The
    result is None if zipimport runs no code: if it cannot compile the source, cannot decompress the member, or fails
    for any other reason.
    """
    # zipimport finds a package's __init__ by the package's name, in the directory above the package.
    if stem == "__init__":
        directory, _, stem = directory.rpartition("/")
    importer = zipimport.zipimporter(os.path.join(zip_path, *directory.split("/")))
    try:
        return importer.get_code(stem).co_filename
    except Exception:
        return None


def _moved_through():
    """Return the clause that names what moves numba's user cache directory on this platform; empty on Windows.

    numba's ``appdirs`` asks the system for the directory on Windows, where no environment variable moves it. On macOS
    it puts the directory under ``HOME``. Elsewhere it reads ``XDG_CACHE_HOME``, or puts the directory under ``HOME``
    if that variable is unset.
    """
    if sys.platform == "win32":
        return ""
    if sys.platform == "darwin":
        return ", through HOME"
    return ", through XDG_CACHE_HOME or HOME"


def _measured(name):
    """Return the length of ``name`` as the file system counts it, and the name of the unit.

    Windows and macOS count a name in UTF-16 units, which the remedy calls characters; the file systems elsewhere count
    bytes. A surrogate counts as one unit."""
    if os.name == "nt" or sys.platform == "darwin":
        # A name decoded with surrogateescape carries a surrogate for each byte that is not UTF-8, and Windows accepts
        # one in a name. Encoding without surrogatepass would raise on it, and cache_remedy would raise for a name that
        # numba's error carried instead of returning a remedy.
        return len(name.encode("utf-16-le", "surrogatepass")) // 2, "characters"
    return len(os.fsencode(name)), "bytes"


def _name_limit(location):
    """Return the longest file name that the file system holding ``location`` allows.

    The result is 255 if the system cannot say, if it says 0, as a file system with no limit of its own does, or if
    ``location`` is None. The count is in bytes outside Windows and macOS, and in UTF-16 units on those two. The
    system is asked about the nearest path at or above ``location`` that exists, because the location itself may not
    exist yet.
    """
    if location is None or not hasattr(os, "statvfs"):
        return 255
    path = os.path.abspath(location)
    while not os.path.exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    try:
        return os.statvfs(path).f_namemax or 255
    except OSError:
        return 255


def _locators(listed):
    """Return numba's cache locator classes in the order that numba tries them.

    ``listed`` is the value of ``NUMBA_CACHE_LOCATOR_CLASSES``. Each entry names a class of numba's caching module, or
    any class by its dotted path, and is resolved as numba resolves it. If ``listed`` is empty, the order is numba's
    own.

    An entry that numba cannot resolve is left out. numba refuses such an entry with its own ``RuntimeError`` before it
    tries any location, or lets importlib's error through, such as the ``TypeError`` for a relative module path. An
    entry that names no class, or a class with no ``from_function``, is left out too: numba fails on it, for want of
    ``from_function``, when a decorated function reaches the entry. In every case the check raises numba's own error as
    numba raised it.
    """
    from numba.core import caching
    if not listed:
        return list(caching.CacheImpl._locator_classes)
    classes = []
    for entry in listed.split(","):
        entry = entry.strip()
        if "." in entry:
            module_path, class_name = entry.rsplit(".", 1)
            try:
                cls = getattr(importlib.import_module(module_path), class_name, None)
            except Exception:
                # The clause catches importlib's errors, the TypeError for a relative module path and any error that the
                # module raises on import. numba raises each of them at decoration, an ImportError wrapped in its own
                # RuntimeError, and the check raises them before any remedy. The remedy, reached only for a cache error,
                # must not raise the module's error instead.
                cls = None
        else:
            cls = getattr(caching, entry, None)
        if isinstance(cls, type) and callable(getattr(cls, "from_function", None)):
            classes.append(cls)
    return classes


def _existing_above(path):
    """Return the nearest path at or above ``path`` that exists, with ``path`` read as absolute.

    The result is the directory that would hold the first missing directory on the way to ``path``. It can also be a
    file in the way, under which no directory can be created, or, at the last, the root."""
    # A relative path with no existing ancestor would walk up to an empty string, and the remedy would say that a file
    # with no name is in the way.
    path = os.path.abspath(path)
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def _ipython_numba_cache():
    """Return ``numba_cache`` under IPython's cache directory, or None if IPython is not importable or raises for it.

    That is numba's cache location for every file that its IPython locator takes, with no directory per file."""
    try:
        try:
            from IPython.paths import get_ipython_cache_dir
        except ImportError:
            from IPython.utils.path import get_ipython_cache_dir
        return os.path.join(get_ipython_cache_dir(), "numba_cache")
    except Exception:
        # Without IPython, numba's locator raises ImportError at decoration, which is not a cache error, so the location
        # is then none of numba's. For an OSError from IPython, numba's locator passes the file over, as it does for a
        # location that it cannot create, so numba took no location of IPython's. numba's locator would raise any other
        # error at decoration. The remedy asks the same question only for the words of a warning, and it takes such an
        # error as no location given.
        return None


# inspect gives these messages when it reads no source. It reads none for a cell that linecache does not hold, a member
# whose archive holds no source, a function whose line lies past the lines that inspect read, and a function of no
# module.
_INSPECTS_ERRORS = ("could not get source code", "source code not available", "lineno is out of bounds")


def _is_inspects(failure):
    """Return whether ``failure`` is inspect's error for a source that inspect could not read.

    Such an error is an ``OSError`` with no errno, no file name and one of inspect's own messages. The message tells it
    apart from a caller's error with no errno and no file name."""
    return (isinstance(failure, OSError) and failure.errno is None and not getattr(failure, "filename", None)
            and str(failure) in _INSPECTS_ERRORS)


def _ipython_cache_is_temporary(first):
    """Return whether IPython gives a new temporary directory as its cache directory on every call.

    IPython does that, with a warning, if it can write neither its cache directory nor its own. numba's IPython locator
    asks for the directory at every save, so numba then creates ``numba_cache`` in one directory and saves in another.
    ``first`` is a location that ``_ipython_numba_cache`` already returned. The function asks IPython for one more, and
    each temporary directory that IPython creates stays on disk. A second answer of None does not count as a new
    directory, because an IPython that gives a directory once and raises the next time gives no temporary directory."""
    if first is None:
        return False
    second = _ipython_numba_cache()
    return second is not None and second != first


def _taken_by_ipython(py_file):
    """Return whether numba's IPython locator takes ``py_file`` by its name, whether or not the file is on disk.

    The locator takes a cell, named ``<ipython-...>``, and a file in an ``ipykernel_`` directory, and it reads the
    function's source when it takes the file."""
    return py_file.startswith("<ipython-") or os.path.basename(os.path.dirname(py_file)).startswith("ipykernel_")


def _archive_above(py_file):
    """Return the archive that holds ``py_file`` and the member's name in it, as zipimport finds them; or None.

    The first part of the path that is not a directory must be a zip archive, whatever its name. The rest of the path
    names the member. The result is None if every part above the file is a directory, or if the first other part is
    not a zip archive.
    """
    # The path is split by the platform's separator, os.sep, through the pure path classes. The concrete pathlib.Path
    # builds a path of the platform that os.name names, which a test can set to Windows's, and Python before 3.12
    # refuses to build such a path on Linux. The remedy for a member that numba names with Windows's separator would
    # then raise.
    pure = pathlib.PureWindowsPath if os.sep == "\\" else pathlib.PurePosixPath
    parts = pure(os.fsdecode(py_file)).parts
    for at in range(len(parts) - 1):
        prefix = str(pure(*parts[:at + 1]))
        if os.path.isdir(prefix):
            continue
        if zipfile.is_zipfile(prefix):
            return prefix, "/".join(parts[at + 1:])
        return None
    return None


def _ipython_locator_reads(py_file):
    """Return whether numba reaches a locator of its IPython family that takes ``py_file`` by its name.

    Such a locator reads the function's source when it takes the file. numba tries the locators in its own order, or in
    the order that ``NUMBA_CACHE_LOCATOR_CLASSES`` gives, and stops at the first locator that takes the file."""
    from numba import config
    from numba.core import caching
    if not _taken_by_ipython(py_file):
        return False
    for cls in _locators(config.CACHE_LOCATOR_CLASSES):
        if issubclass(cls, caching.IPythonCacheLocator):
            return True
        if issubclass(cls, caching.ZipCacheLocator) and ".zip" in py_file:
            # numba's .zip locator takes a path that contains ".zip" without trying its location, so numba reaches no
            # locator listed after it.
            return False
        if _takes(cls, py_file):
            # A locator before IPython's took the file when asked with a probe, as numba asks it, so numba does not
            # reach the IPython locator. The in-tree locator, for one, takes a cell file on disk whose __pycache__ it
            # can write. A locator that passes the file over leaves the ones after it reached. Without this question,
            # the remedy would read a caller's error that names some other numba_cache as the IPython locator's error,
            # for a cell file on disk whose __pycache__ the in-tree locator writes.
            return False
    return False


def _numbas_archive_holds(py_file, name):
    """Return whether the archive that numba's ``.zip`` locator takes for ``py_file`` holds a member named ``name``.

    That archive is the first part of the path that ends in ``.zip``. The result is False if no part does, or if that
    part is not a zip archive or cannot be read."""
    parts = py_file.split(os.sep)
    at = next((i for i, part in enumerate(parts) if part.endswith(".zip")), None)
    if at is None:
        return False
    try:
        with zipfile.ZipFile(os.sep.join(parts[:at + 1])) as archive:
            return name in archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def _pyc_naming(py_file):
    """Return the ``.pyc`` member whose code names ``py_file`` and whether its ``.py`` is beside it, or None.

    The function searches the archive that numba's ``.zip`` locator takes for ``py_file``, and it reads the code that
    zipimport runs for each ``.pyc`` member. The result is None if no member's code names ``py_file``, or if the archive
    cannot be read. The function finds a ``.pyc`` compiled to name another path in the archive (``compileall -d``):
    numba 0.68 and later read the source by that name and find none, while the archive holds the source beside the
    ``.pyc``."""
    parts = py_file.split(os.sep)
    at = next((i for i, part in enumerate(parts) if part.endswith(".zip")), None)
    if at is None:
        return None
    zip_path = os.sep.join(parts[:at + 1])
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return None
    for name in sorted(names):
        if name.endswith(".pyc") and "/__pycache__/" not in name:
            directory, _, file = name.rpartition("/")
            if _compiled_from(zip_path, directory, file[:-4]) == py_file:
                return name, name[:-1] in names
    return None


def _member_not_utf8(py_file):
    """Return why the member of the archive above ``py_file`` is not UTF-8, as the reason and the byte offset, or None.

    zipimport decodes a member's source as UTF-8, and numba's IPython locator reads the source as zipimport decodes
    it. The result is None if no archive above the file holds the member, if the archive cannot be read, or if the
    member decodes as UTF-8."""
    found = _archive_above(py_file)
    if found is None:
        return None
    zip_path, member = found
    try:
        with zipfile.ZipFile(zip_path) as archive:
            data = archive.read(member)
    except (KeyError, OSError, zipfile.BadZipFile):
        return None
    try:
        data.decode()
    except UnicodeDecodeError as error:
        return f"{error.reason} at byte {error.start}"
    return None


def _missing_member(error):
    """Return the member that zipfile's ``KeyError`` names, or the error's whole message if it names none.

    The message reads "There is no item named 'x' in the archive". zipfile quotes the name as Python's ``repr`` does,
    so a backslash in it is doubled, and the function reads the name back. numba 0.68.0 builds a member's name with
    backslashes on Windows.
    """
    message = str(error.args[0]) if error.args else str(error)
    found = re.search(r"^There is no item named (.+) in the archive$", message)
    if not found:
        return message
    try:
        return str(ast.literal_eval(found.group(1)))
    except (ValueError, SyntaxError):
        return found.group(1)


def cache_remedy(py_file, failure, silence, package="numbox"):
    """Return the remedy for ``failure``, numba's cache error for a function whose source file is ``py_file``.

    The remedy is one sentence. It names what numba tried and why that failed, says what to change, and ends in
    ``silence``, the clause that says how to turn caching off. ``py_file`` can be a string, bytes or a path object, and
    so can the file name that ``failure`` carries; both are read as strings. ``package`` names the package in the
    remedies that ask for it to be installed again, at a shorter path or with its source files on disk. A package built
    on numbox passes its own name, as it passes its own ``longest_file_name`` to :func:`check_cache_location`;
    ``LONGEST_CACHE_FILE_NAME`` is numbox's.

    The remedy reads ``NUMBA_CACHE_DIR`` and ``NUMBA_CACHE_LOCATOR_CLASSES`` from ``numba.config``, the user's cache
    directory, ``sys.frozen``, and the archive above ``py_file``, if there is one. It asks IPython for its cache
    directory only if numba reaches a locator of its IPython family that takes the file. IPython under a home
    directory that it cannot write warns each time it is asked, and it leaves a temporary directory behind.

    Most of the remedy depends on the locator that took the file. The remedy compares the path that the error names,
    and that path's directory, with the location that each locator numba reaches would give the file. It compares
    whole paths, in numba's order, and reads the error as the first matching locator's. numba's own errors mean that
    the locators before that one passed the file over; an error that a caller passes on need not mean that. If the
    error names a path that no locator gives the file, the remedy names that path as a location that is none of
    numba's.

    If the error names no file, the remedy names every location that numba could have taken, each once, in numba's
    order. Such an error has no file name, an empty one, or a name of another kind, such as an integer file
    descriptor. numba's first save raises one when a write fails, as on a full disk. A path object never counts as
    empty: pathlib reads an empty string as the working directory, ``.``, and the remedy names that directory as a
    location that is none of numba's.

    ``NUMBA_CACHE_LOCATOR_CLASSES`` sets the order. Each entry is a class of numba's caching module; a subclass of one,
    named by its dotted path, which the remedy treats as its parent; or a class outside numba's families. The remedy
    asks a class outside them about the file with a probe, as numba asks it, through the class's ``from_function``.
    Such a class caches in a place that only the class can name, and the remedy calls that place the class's own. An
    error that names a path of none of numba's locators may name that place. numba reaches no locator listed after one
    that takes the file. Its ``.zip`` locator takes every path that contains ``.zip`` without trying its location, so
    numba reaches no locator listed after it for such a path.

    The remedy distinguishes these cases.

    A source file on disk
        numba reads ``NUMBA_CACHE_DIR`` only for a source file on disk, so the remedies for such a file offer it. The
        remedy for numba's error that no locator takes the file names the variable as one that numba could not use, if
        it is set, and asks for a writable directory at a short path. If the variable is unset, the remedy asks for it
        to be set to a writable directory. If ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves the user-provided locator out,
        numba never reads the variable, and the remedy asks for the listed locations to be made writable.

        An ``OSError`` that names the location that numba took, or a file in it, means that the location refused a
        file after numba's own test passed, as on a full disk or with permissions changed since. The remedy names the
        location and the reason, and asks for free space or a writable directory there. As the alternative, it offers
        ``NUMBA_CACHE_DIR`` if numba tries the variable before that location's locator, or the variable set to another
        directory if the location is the variable's own. The user-wide and ``.zip`` locators give a file the same
        location, so for that location the remedy offers the variable if numba tries it before the last of the two.
        The remedy offers the variable for a location that is none of numba's only if numba tries it first. In these
        two cases the remedy does not say that numba could not use the variable.

    A path too long for the file system
        The remedy names what shortens the location that numba took. That is a shorter ``NUMBA_CACHE_DIR`` for the
        variable's location, the package installed at a shorter path for the ``__pycache__`` beside the source, and
        IPython's cache directory at a shorter path for IPython's location. The remedy for a location under the user's
        cache directory is that directory at a shorter path: through ``XDG_CACHE_HOME`` or ``HOME``, through ``HOME``
        alone on macOS, and through nothing on Windows. For a source file on disk, the remedy also offers
        ``NUMBA_CACHE_DIR`` set to a short path, on the terms on which it offers the variable for a refused file.

    A directory name too long for the file system
        numba names its directory under ``NUMBA_CACHE_DIR`` or the user's cache directory after the source's
        directory. If that name is longer than the file system holding the location allows, no path helps. The remedy
        then names the directory's name and its length, and asks for the file to be put in a directory with a shorter
        name. It does so for an error that names such a location, and for a frozen application's error. It does so too
        for numba's error that no locator takes a file on disk, if every locator tried that names its directory this
        way found the name too long. The ``.zip`` locator's error once the locators before it passed the file over
        counts as such an error. If the user-provided locator is the only one tried and ``NUMBA_CACHE_DIR`` is unset,
        the remedy also asks for the variable to be set.

        For a file on disk, the remedy adds the ``__pycache__`` beside the source, whose name holds no hash. It asks
        for that directory to be made writable if numba reaches the in-tree locator. Otherwise it asks for the in-tree
        locator to be listed, before the ``.zip`` locator or the outside class that took the file if the list puts it
        after one.

    A cache file name too long for the file system
        numba names the files that it writes for a function after the source's name and the function's qualified
        name. If numba could not create or open such a file in the location that it took, because the name is too
        long for that file system, the remedy names the file and its length. It says that no shorter path helps, and
        asks for a shorter name for the function or its file, which only the package can give. If some file system in
        use allows the name, the remedy also offers ``NUMBA_CACHE_DIR`` pointed at a directory on such a file system.
        It offers the variable as set if the variable's own location took the file, and otherwise on the terms on
        which it offers the variable for a path too long.

        The remedy for the check's error about a name of ``longest_file_name`` bytes, on a file system that allows no
        name that long, names that length. It offers ``NUMBA_CACHE_DIR`` on the same terms, whatever the length. The
        remedy offers nothing for a source that is not a file on disk, because numba reads the variable only for a file
        on disk.

    A path that is not UTF-8
        numba names its directory under ``NUMBA_CACHE_DIR`` or the user's cache directory with a hash of the source's
        directory path, encoded as UTF-8. A locator that names its directory this way raises at decoration for a path
        whose bytes are not UTF-8, when numba reaches it. For such a path, the remedy names no location under either
        directory, and where it would offer ``NUMBA_CACHE_DIR`` it says that the variable does not help. For a ``.zip``
        under such a path, the remedy asks for the archive to be put under a UTF-8 path, because the ``.zip`` locator
        hashes the path the same way. For a frozen application, which the user-wide locator hashes, it asks the same
        of the application. For a source file on disk, it also offers the ``__pycache__`` beside the source, because
        the in-tree locator hashes no path:

        - If the list leaves the in-tree locator out, or puts it after a locator that hashes other than the
          variable's, the remedy asks for the in-tree locator to be listed before the first locator that hashes.
        - If the variable's locator is the only one that hashes before the in-tree locator, the remedy asks for
          ``NUMBA_CACHE_DIR`` to be unset.
        - If the in-tree locator comes before every locator that hashes and passed the file over, the remedy asks for
          the ``__pycache__`` to be made writable.

        With either of the first two, the remedy also asks for the ``__pycache__`` to be made writable if numba cannot
        write in it.

    A file that numba's IPython locator takes
        numba's IPython locator takes a cell, named ``<ipython-...>``, and any file in an ``ipykernel_`` directory. It
        caches every such file in ``numba_cache`` under IPython's cache directory, with no directory per file. It
        passes that location over if it cannot create it or write in it. No other locator of numba's takes a cell, so
        a cell then gets numba's error that no locator takes the file. So does a ``.zip`` member in an ``ipykernel_``
        directory if the list puts no ``.zip`` locator after the IPython one. The remedy asks for that location to be
        made writable, or for IPython's cache directory to be put at a shorter path. If IPython raised for its
        directory, the remedy asks for IPython's home to be made writable. It names a class outside numba's families
        listed with the IPython locator as having passed the file over too. For a path too long, it asks for IPython's
        cache directory at a shorter path. That holds for a cell file on disk and for a ``.zip`` member in an
        ``ipykernel_`` directory alike, if the IPython locator is listed before the ``.zip`` one.

        If IPython is not importable, numba raises ``ImportError`` at decoration for a file that the IPython locator
        takes, which is not a cache error. The remedy then treats an error that a caller passes on and that names
        IPython's location as an error of a location that is none of numba's, if another locator that numba reaches
        has a place for the file. Otherwise, for a file on disk, the remedy names the list, as for numba's error that
        no locator takes the file. If IPython raises an ``OSError`` for its directory, numba passes the locator over,
        and the remedy has no location of IPython's to compare the error with.

        If IPython can write neither its cache directory nor its own, it gives a new temporary directory as its cache
        directory on every call. numba then creates ``numba_cache`` in one such directory and saves in another. Its
        first save raises ``FileNotFoundError`` for a file in the ``numba_cache`` that it could not save in. If the
        IPython locator takes the file, the remedy for such an error says so, and asks for ``IPYTHONDIR``, or
        ``XDG_CACHE_HOME`` on a POSIX system, to be set to a writable directory. An error with another errno that names
        a ``numba_cache`` is a caller's, and the remedy treats it as an error of the location that it names.

        inspect's error, which has no errno and no file name, means that the IPython locator read no source before it
        tried its location. For a cell, the remedy says that numba caches a cell only while IPython holds its source,
        which numba reads from linecache. For a cell file on disk, the remedy asks for the file to be made readable,
        with the source in it; inspect raises for a file that cannot be read and, before Python 3.13, for an empty
        one. The remedy for a ``.zip`` member whose archive holds no source for it, a ``.pyc`` alone, is the archive's:
        ship the source files. If a member's source is not UTF-8, the remedy asks for it to be shipped as UTF-8.

    A ``.zip`` or a frozen application
        numba caches both in the user's cache directory, and ``NUMBA_CACHE_DIR`` has no effect on them. The remedy asks
        for free space or a writable directory in the location that numba took, with the reason that the error gives,
        such as a full disk or permissions. A ``.zip``'s error names that location, a directory of numba's under the
        user's cache directory, or a file in it, as the first save's error does; the remedy names the directory. If the
        directory does not exist, the remedy names the existing directory above it, under which it cannot be created.
        If a file is in the way, the remedy asks for that file to be put elsewhere. If the error names a location that
        is none of numba's, the remedy names that location as the error gives it, and says nothing of the user's cache
        directory.

        A frozen application's error is numba's error that no locator takes the file. It gives no reason, because numba
        passed the location over on its own error, for an unwritable directory and for one too deep alike. The remedy
        therefore names the user's cache directory as one that numba could not use, to be made writable or put at a
        shorter path. A frozen application whose path contains ``.zip`` but has no part that ends in ``.zip`` gets the
        ``.zip`` locator's error instead, if the list puts that locator before the user-wide one, or once the user-wide
        one passed its location over. The remedy then asks for the user-wide locator to be listed before the ``.zip``
        one, or for the user-wide location to be made writable or put at a shorter path.

        An error that names the archive that a member's code names is the stamp's error. If the archive is not there,
        because the member's ``.pyc`` was compiled to name a path where no archive is now, the remedy asks for the
        source files to be shipped. Before numba 0.68, it also offers compiling the ``.pyc`` members to name the
        archive's present path. If the archive cannot be read, the remedy asks for it to be made readable.

        If the error names no file, the remedy names the locations that numba could have taken, as for a source file
        on disk. If numba reaches a locator of IPython's family or a class outside its families for a member, the
        remedy does not say that numba caches the member in the user's cache directory. IPython's ``numba_cache`` is
        among the locations whenever IPython gives its directory.

    A list without the locator that a source needs
        numba caches a ``.zip`` through its ``.zip`` locator alone, a cell through its IPython locator alone, and a
        frozen application through its user-wide locator alone. If ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves out the
        locator that the source needs, the remedy asks for that locator to be listed. It does so for numba's error that
        no locator takes the file, and for any error that a caller passes on under such a list, because numba took no
        location. A ``.zip`` counts only if no part of the path that ends in ``.zip`` is a directory: numba caches
        nothing under a directory so named.

        For a source file on disk, a list in which no locator takes the file is named, with the locators for a file on
        disk to be listed. The user-provided locator takes nothing while ``NUMBA_CACHE_DIR`` is unset, so the remedy
        names it together with the variable. A ``.zip`` locator listed before every locator for a file on disk takes a
        path that contains ``.zip``. If no part of that path ends in ``.zip``, the locator finds no archive and raises,
        and the remedy asks for a locator for a file on disk to be listed before it. A ``.zip`` locator listed after
        some of them raises the same error once those passed the file over, and numba tries no locator after it. The
        remedy then asks for one of their locations to be made writable, or for such a locator to be listed before the
        ``.zip`` one.

    Another archive, or a ``.pyc`` alone
        A module in an ``.egg``, ``.whl``, ``.pyz`` or other archive that numba does not cache has no cache location,
        and neither has a module present as ``.pyc`` alone. The remedy for either asks for the package to be installed
        with its source files on disk. As the alternative, the remedy offers a ``.zip`` that holds them, which numba
        caches in the user's cache directory. If the module is already in an archive whose name does not end in
        ``.zip``, lower-case, the remedy asks for that archive to be renamed to end so. If a directory above that
        archive has a name that ends in ``.zip``, the remedy also asks for the archive to be moved out from under it,
        because numba takes the first such part of the path for the archive. Each option names the locator that it
        needs, if ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves that locator out. A source that is neither in a ``.zip`` nor a
        frozen application's, and that no locator takes, gets this remedy for any error.

    A ``.zip`` member under numba 0.68 and later
        From numba 0.68, the stamp of a ``.zip`` member hashes the member's source. numba reads the source from the
        archive by the name that the module's code gives under the archive's path, and zipfile raises ``KeyError`` for
        a name that the archive does not hold. For a module shipped as ``.pyc`` alone, the remedy names the member and
        asks for the source files, on disk or in a ``.zip``. If a ``.pyc`` was compiled to name another path in the
        archive (``compileall -d``) and the archive holds its source beside it, the remedy asks for the ``.pyc`` to be
        compiled to name its own path, or to be left out.

        On Windows, numba 0.68.0 builds the name with the platform's separator, so every member raises that error
        (numba issue 10889, fixed in 0.68.1). If the archive holds the member's source under the name with forward
        slashes, which numba 0.68.1 reads, the remedy is the install on disk or a numba release that carries the fix.
        Otherwise it is the remedy for any other member.

        numba takes the first part of the path that ends in ``.zip`` for the archive, a directory so named included,
        and from 0.68 it opens that part to read the source. The remedy for the ``IsADirectoryError`` that a directory
        gives asks for the archive to be put under a path with no such directory above it. If a file on disk lies
        under such a directory, the ``.zip`` locator takes it once the locators before it passed the file over. The
        remedy then asks for one of their locations to be made writable. If the ``.zip`` locator is listed first, the
        remedy asks instead for a locator for a file on disk to be listed before it. Either remedy also offers putting
        the file under a path with no such directory.
    """
    from numba import config
    from numba.core import caching
    from numba.core.caching import _CacheLocator
    # check_cache_location takes any path-like object, bytes included, and so does cache_remedy. The remedy reads a str;
    # os.fspath would leave bytes as bytes.
    py_file = os.fsdecode(py_file)
    # The error names the refused file in the form that the raising call was given it: bytes for a call given bytes, or
    # the form that a caller built the error with, a path object included. The remedy reads the name as a str, as it
    # reads the source's. An empty str or bytes is no name. A path object built from an empty string names the working
    # directory, ".", as pathlib reads it. A name of another kind, such as the integer file descriptor that os.stat(fd)
    # raises with, is no name.
    filename = getattr(failure, "filename", None)
    filename = os.fsdecode(filename) if filename and isinstance(filename, (str, bytes, os.PathLike)) else None
    listed = config.CACHE_LOCATOR_CLASSES
    user_provided = caching.UserProvidedCacheLocator
    in_tree = caching.InTreeCacheLocator
    user_wide = caching.UserWideCacheLocator
    for_ipython = caching.IPythonCacheLocator
    for_a_zip = caching.ZipCacheLocator

    def one_of(cls, base):
        """Return whether ``cls`` is ``base`` or a subclass of it."""
        # A subclass listed by its dotted path takes what its parent takes and caches where its parent does, as numba's
        # InTreeCacheLocatorFsAgnostic does, so each of numba's locators represents its whole family.
        return issubclass(cls, base)

    def of_numbas(cls):
        """Return whether ``cls`` belongs to one of numba's locator families, whose places the remedy knows."""
        # A class outside them has a from_function and caches in a place of its own, which only the class can name.
        return any(one_of(cls, base) for base in (user_provided, in_tree, user_wide, for_ipython, for_a_zip))

    def could_have_taken(locations):
        """Return the clause that names each location in ``locations`` that numba could have taken, each once."""
        # An error that names no file cannot show which location numba took, so the clause lists the values of
        # locations, in numba's order. Each is listed once: on disk, NUMBA_CACHE_DIR set to the user's cache directory
        # makes one path of two. A class outside numba's families has a place that the remedy cannot name, so the
        # clause calls that place the class's own.
        known = list(dict.fromkeys(
            location if location else f"{cls.__name__}'s own"
            for cls, location in locations.items() if location or not of_numbas(cls)))
        if len(known) > 1:
            return f"the location numba took, one of {', '.join(known[:-1])} or {known[-1]}"
        return f"the location numba took, {known[0]}"

    def listed_without(base):
        """Return whether ``NUMBA_CACHE_LOCATOR_CLASSES`` is set and names no class of ``base``'s family."""
        return bool(listed) and not any(one_of(cls, base) for cls in _locators(listed))

    try:
        subpath = _CacheLocator.get_suitable_cache_subpath(py_file)
    except UnicodeEncodeError:
        # numba hashes the source's directory path as UTF-8 to name its directory. A locator that names its directory
        # this way raises at decoration for bytes that are not UTF-8, when numba reaches it. The remedy then has no
        # such location to name.
        subpath = None

    def too_long_a_name(location):
        """Return the clause saying that numba's directory name for the file is too long at ``location``, or None.

        The result is None if the name fits the file system that holds ``location``, or if numba has no name to
        give."""
        # numba names the directory that it creates for the file, under NUMBA_CACHE_DIR or the user's cache directory,
        # after the source's directory. The name is the source directory's name, an underscore and the 40 hexadecimal
        # digits of a SHA-1 hash of the directory's path: 41 bytes more than the directory's name. A name longer than
        # the file system holding the location allows cannot be created at any path, so no shorter path helps.
        if subpath is None:
            return None
        limit = _name_limit(location)
        length, counted_in = _measured(subpath)
        if length <= limit:
            return None
        return (
            f"numba names the directory it caches this file in after the source's directory, {subpath}, {length} "
            f"{counted_in}, which is longer than a name can be on that file system ({limit})"
        )

    a_shorter_name = f"a shorter name for the function or its file, which is {package}'s to give"

    def a_name_limit_below(location):
        """Return the clause for the check's error about a name of ``longest_file_name`` bytes, or None.

        The check raises that error, with the name's length as its ``longest_file_name``, on a file system that allows
        no name that long. The result is None for any other error, or if the file system at ``location`` allows the
        name."""
        # Without this clause, the remedy would read the error as the location's error and offer a shorter path, which
        # does nothing for a name that is too long.
        longest = getattr(failure, "longest_file_name", 0)
        if not longest or location is None:
            return None
        limit = _name_limit(location)
        if limit >= longest:
            return None
        length, counted_in = _measured("x" * longest)
        return (
            f"numba's cache files for {package}'s functions have names of up to {length} {counted_in}, longer than "
            f"a name can be on that file system ({limit})"
        )

    def too_long_a_file_name(location):
        """Return the clause saying that the cache file the error names is too long for its file system, or None.

        The result is None if the error names no file in ``location``, or if the file's name fits. Otherwise it is the
        clause and whether some file system in use allows the name at all."""
        # numba names the files that it caches a function in after the source's name and the function's qualified
        # name. A name longer than the file system holding the location allows cannot be created at any path, so no
        # shorter path helps; only a shorter name does. Without this clause, the remedy would ask for a shorter path
        # for the package, the user's cache directory, IPython's cache directory or the variable, none of which
        # shortens a name.
        #
        # The limit of ext4, NTFS and APFS alike is 255, in the units that each counts. The name is measured against
        # the longest name that numba writes for the function. numba writes the index, ".nbi", and a data file for
        # each signature, ".<n>.nbc", numbered from 1 without bound. It writes each through a temporary name 21 bytes
        # longer, ".tmp." and 16 hexadecimal digits. The data file's number takes the digits reserved for it, or the
        # width of the number that the error names if that is wider. With three digits, that adds 29 bytes to the
        # name's base. A file system in use allows the name if the result fits in 255; the remedy offers such a file
        # system only then, and names the shorter name always.
        if not filename or location is None:
            return None
        named = os.path.abspath(filename)
        if os.path.dirname(named) != os.path.abspath(location):
            return None
        name = os.path.basename(named)
        length, counted_in = _measured(name)
        limit = _name_limit(location)
        if length <= limit:
            return None
        clause = (
            f"numba names the cache files for a function after the source's name and the function's qualified "
            f"name, {name}, {length} {counted_in}, which is longer than a name can be on that file system ({limit})"
        )
        stripped = re.sub(r"\.tmp\.[0-9a-f]{16}$", "", name)
        numbered = re.search(r"\.(\d+)\.nbc$", stripped)
        digits = max(_OVERLOAD_DIGITS, len(numbered.group(1))) if numbered else _OVERLOAD_DIGITS
        base = re.sub(r"\.nbi$|\.\d+\.nbc$", "", stripped)
        return clause, _measured(base)[0] + len(".") + digits + len(".nbc") + 21 <= 255

    a_file_on_disk = os.path.exists(py_file)

    def may_take(cls):
        """Return whether the locator class ``cls`` takes the file.

        A locator of numba's is judged by the file's name alone, before any location is tried. A class outside numba's
        families is asked with a probe, as numba asks it."""
        # numba's IPython locator takes a cell, or a file in an ipykernel_ directory, on disk or not. Its .zip locator
        # takes a path that contains ".zip"; it caches the file if a part of the path ends in .zip, an archive or a
        # directory, and raises if none does. So listing either of the two ahead of the rest changes nothing for any
        # other file. numba's user-wide locator takes a frozen application's file as well as any file on disk, and the
        # rest take any file on disk. Without asking a class outside numba's families, the remedy would take it for one
        # that takes every file. It would report a file that the class passed over as the class's own, although numba
        # went on to the next locator listed.
        if one_of(cls, for_ipython):
            return _taken_by_ipython(py_file)
        if one_of(cls, for_a_zip):
            return ".zip" in py_file
        if one_of(cls, user_wide):
            return a_file_on_disk or bool(getattr(sys, "frozen", False))
        if of_numbas(cls):
            return a_file_on_disk
        return _takes(cls, py_file)

    def names_a_path_above():
        """Return whether the error names a path above the file, such as the archive that a member's code names.

        A directory on the way to the file counts too; the working directory does not."""
        # Both paths are read as absolute. A relative file name matches no absolute name, and the remedy would
        # otherwise report a member's archive that has since moved as a location that refused the file. A path object
        # built from an empty string names the working directory, and the remedy reports that path as a location that
        # is none of numba's. If the working directory counted, it would stand above a file under it as a moved archive.
        above = os.path.abspath(filename) if filename else ""
        return bool(above) and above != os.getcwd() and os.path.abspath(py_file).startswith(above + os.sep)

    def names_another_location():
        """Return whether the error names the source, or a location of numba's other locators for a file on disk.

        Those locations are ``NUMBA_CACHE_DIR``, the ``__pycache__`` beside the source and the user's cache directory,
        with the files under them. A location that is none of those, such as a temporary directory that IPython gave,
        counts as the IPython locator's."""
        if not filename:
            return False
        named = os.path.abspath(filename)
        if named == os.path.abspath(py_file):
            # numba reads the source itself. If the source cannot be read, the remedy says to make it readable,
            # whatever directory IPython gives as its cache directory.
            return True
        others = [os.path.abspath(config.CACHE_DIR)] if config.CACHE_DIR else []
        others.append(os.path.abspath(os.path.join(os.path.dirname(py_file), "__pycache__")))
        others.append(os.path.abspath(AppDirs(appname="numba", appauthor=False).user_cache_dir))
        return any(named == other or named.startswith(other + os.sep) for other in others)

    asked = []

    def ipython_numba_cache():
        """Return what ``_ipython_numba_cache`` returns, asking IPython once for the whole remedy."""
        # IPython creates a temporary directory on every call, and warns, under a home directory that it cannot write.
        if not asked:
            asked.append(_ipython_numba_cache())
        return asked[0]

    def names_a_numba_cache():
        """Return whether the error is ``ENOENT`` for a ``numba_cache`` directory, or for a file in one."""
        # numba's first save raises FileNotFoundError under an IPython that gives a temporary directory on every call.
        # The error names a file in the numba_cache directory that numba could not save in, because numba created that
        # directory in another temporary directory. The check's error names the directory. A caller's error that names
        # another location shows nothing of IPython's locator. Without this test, the remedy would offer IPYTHONDIR for
        # a cell on disk that the in-tree locator took, and for a member under such an IPython on any errno, a refused
        # directory of the caller's own included.
        return (
            bool(filename) and failure.errno == errno.ENOENT
            and "numba_cache" in os.path.abspath(filename).split(os.sep))

    if isinstance(failure, OSError) and names_a_numba_cache() and not names_a_path_above() and (
            _ipython_locator_reads(py_file)) and (
            not a_file_on_disk or not names_another_location()
    ) and _ipython_cache_is_temporary(ipython_numba_cache()):
        # numba reaches its IPython locator first for a member in an ipykernel_ directory, and for a cell file on disk
        # in such a directory once the locators before it passed the file over. The locator asks IPython for its cache
        # directory at every save. An IPython that can write neither that directory nor its own gives a new temporary
        # directory on every call, so numba creates numba_cache in one and fails to save in the second. Without this
        # case, the remedy would report the directory that the error names, a temporary one, as refusing a file. For a
        # file on disk it would offer NUMBA_CACHE_DIR beside the source.
        #
        # Three errors stay out of this case. The remedy for an error that names no file names the locations that
        # numba could have taken, with the error's reason, whatever directory IPython gives. For a file on disk, such
        # an error also shows nothing of which locator took the file. A caller's error with a reason and no name, such
        # as a full disk's, gets a remedy with its reason, not IPYTHONDIR. The error of numba's first save here names a
        # file in the numba_cache that numba could not save in. inspect's error, with no errno and no name, came before
        # the locator tried any location. The remedy below treats it as inspect's for a cell or a file on disk and as
        # the archive's for a member shipped as .pyc, whatever directory IPython gives. IPython's directory is not the
        # cause for a cell whose source linecache does not hold, or for such a member. An error that names a path above
        # the file, the archive that a member's code names when that archive has since moved, is the stamp's error,
        # and the remedy below handles it. The temporary directory's error names a numba_cache under the temporary
        # directory.
        #
        # From IPython 8, IPython reads XDG_CACHE_HOME for its cache directory on every POSIX system, macOS included;
        # IPython 7 reads it on POSIX systems other than macOS. The message therefore says POSIX, not Linux, so that it
        # applies to a reader on macOS as well.
        return (
            "numba caches this file in numba_cache under IPython's cache directory, and IPython, able to write neither "
            "that directory nor its own, gives a new temporary directory on every call, so numba makes numba_cache "
            "in one and saves in another: set IPYTHONDIR, or XDG_CACHE_HOME on a POSIX system, to a writable "
            "directory; or "
            f"{silence}"
        )

    if a_file_on_disk:
        def reads_the_variable(cls):
            """Return whether ``cls`` is a locator of the family that reads ``NUMBA_CACHE_DIR``."""
            return one_of(cls, user_provided)

        def caches_beside_the_source(cls):
            """Return whether ``cls`` is a locator of the family that caches beside the source."""
            return one_of(cls, in_tree)

        def caches_under_the_user_cache_dir(cls):
            """Return whether ``cls`` is a locator of a family that caches under the user's cache directory."""
            return one_of(cls, user_wide) or one_of(cls, for_a_zip)

        order = [cls for cls in _locators(listed) if may_take(cls)]
        # numba's .zip locator takes a path that contains ".zip" without trying its location, and a class outside
        # numba's families takes the file when asked. Either stops the order, and numba reaches no locator listed after
        # it. A location that such a locator would give is therefore none of numba's. Without this cut, the remedy would
        # read the locators after such a class as reached. It would then report an error that names IPython's location
        # as one where numba caches the file.
        zip_at = next((at for at, cls in enumerate(order) if one_of(cls, for_a_zip)), None)
        stops_at = next((at for at, cls in enumerate(order) if one_of(cls, for_a_zip) or not of_numbas(cls)), None)
        reached = order if stops_at is None else order[:stops_at + 1]
        # A .zip locator listed after such a class is not reached, so under that list the errors that it would raise
        # are a caller's. Without this, the remedy would report those errors as the .zip locator's own, on the ground
        # that the locators before it passed the file over, when one of them had taken the file.
        zip_at = zip_at if zip_at == stops_at else None
        variable_at = next((at for at, cls in enumerate(order) if reads_the_variable(cls)), None)
        cache_dir_read = variable_at is not None

        def instead(taken, asks="a short path", shared=None, made="a writable directory at a short path"):
            """Return the clause that offers ``NUMBA_CACHE_DIR`` instead of the location of ``taken``, or ``""``.

            The clause offers the variable if numba tries it before the locator ``taken``. ``taken`` is None for a
            location that is none of numba's, which counts as the first locator in the order other than the
            variable's own. If the variable is set, the clause names it and ends with ``made``, what the reader should
            make of its directory. If it is unset, the clause asks for it to be set to ``asks``. ``shared`` is the last
            locator that numba reaches whose place the error names too."""
            # If the variable is set and numba tried it before the location that it took, numba passed the variable
            # over. Its directory was unwritable or too deep, and the clause says that numba could not use it. A
            # location that is none of numba's shows nothing of what numba did with the variable. The clause then names
            # a set variable without that claim. The user-wide and .zip locators give the file one path under the
            # user's cache directory, and the error cannot show which of them took the file. The clause therefore
            # offers the variable if numba tries it before the last of them, again without the claim. Reading the first
            # of those two locators as the one that took the file would leave the variable out when the variable sits
            # between them.
            if subpath is None:
                # numba names its directory under NUMBA_CACHE_DIR with a hash of the source's directory path, encoded
                # as UTF-8, and raises UnicodeEncodeError for a path that is not UTF-8. A reader who set the variable
                # would make numba raise that error for this path, which is_a_cache_error counts as a cache error: the
                # warning would come back, and the variable would not help. The clause says so instead.
                return (
                    ", and NUMBA_CACHE_DIR cures nothing, numba hashing the source's directory as UTF-8 for its "
                    "directory there, which this one is not"
                )
            if taken in order:
                position = order.index(taken)
                passed_over = " and numba could not use"
            else:
                position = next((at for at, cls in enumerate(order) if not reads_the_variable(cls)), len(order))
                passed_over = ""
            if shared is not None and shared is not taken and shared in order:
                position = max(position, order.index(shared))
                passed_over = ""
            if not cache_dir_read or variable_at >= position:
                return ""
            if config.CACHE_DIR:
                return f", or NUMBA_CACHE_DIR, which is set to {config.CACHE_DIR}{passed_over}, made {made}"
            return f", or NUMBA_CACHE_DIR set to {asks}"

        user_cache_dir = AppDirs(appname="numba", appauthor=False).user_cache_dir

        def beside_the_source():
            """Return the clause that offers the ``__pycache__`` beside the source, or an empty string.

            The clause asks for that directory to be made writable if numba reaches the in-tree locator. Otherwise,
            under a list, it asks for the in-tree locator to be listed."""
            # The in-tree locator caches beside the source, in its __pycache__, under no directory named after the
            # source's directory. It is therefore the one locator of numba's for a file on disk that a directory name
            # too long leaves usable. The clause asks for the in-tree locator to be listed before the .zip one if the
            # .zip locator, listed first, took a path that contains ".zip" and left the in-tree locator unreached.
            # Offering the writable __pycache__ under a list that never reaches the in-tree locator would not help.
            if any(caches_beside_the_source(cls) for cls in reached):
                beside = os.path.abspath(os.path.join(os.path.dirname(py_file), "__pycache__"))
                return (
                    f", or make the __pycache__ beside the source, {beside}, writable, where InTreeCacheLocator "
                    "caches it under no such name"
                )
            if listed:
                # If the in-tree locator is listed after the locator that took the file, the .zip one or a class
                # outside numba's families, the in-tree locator has to go before that locator. The remedy names the
                # locator to go before, which is not always the .zip one.
                before = ""
                if stops_at is not None and any(caches_beside_the_source(cls) for cls in order):
                    stopper = order[stops_at]
                    before = f" before {'ZipCacheLocator' if one_of(stopper, for_a_zip) else stopper.__name__}"
                return (
                    f", or list InTreeCacheLocator{before}, which caches beside the source, in its __pycache__, under "
                    "no such name"
                )
            return ""

        def location_of(cls):
            """Return the location where the locator class ``cls`` places the file, or None if the remedy has none."""
            # The .zip locator places the file under the user's cache directory, as the user-wide one does. The result
            # is None for a class that is not numba's, whose place is its own. It is None for the user-provided locator
            # without NUMBA_CACHE_DIR, which takes nothing without the variable, and for a locator that names its
            # directory with a hash when the path is not UTF-8. It is None for the IPython locator when IPython is not
            # importable, or when IPython raises OSError for its directory and numba passes the locator over.
            if reads_the_variable(cls):
                return os.path.abspath(os.path.join(config.CACHE_DIR, subpath)) if config.CACHE_DIR and subpath else None
            if caches_beside_the_source(cls):
                return os.path.abspath(os.path.join(os.path.dirname(py_file), "__pycache__"))
            if caches_under_the_user_cache_dir(cls):
                return os.path.abspath(os.path.join(user_cache_dir, subpath)) if subpath else None
            if one_of(cls, for_ipython):
                ipython_cache = ipython_numba_cache()
                return os.path.abspath(ipython_cache) if ipython_cache else None
            return None

        locations = {cls: location_of(cls) for cls in reached}
        if _is_inspects(failure) and any(one_of(cls, for_ipython) for cls in reached):
            # inspect's error has no errno and no file name, because no call into the file system raised it. numba's
            # IPython locator reads the function's source, from the file itself for a file on disk. inspect raises if it
            # reads no source there: in a file that it cannot read or, before Python 3.13, an empty one. That happens
            # before the locator tries its location, after the locators before it passed the file over. No location
            # that numba took refused the file.
            return (
                f"numba's IPython locator reads the function's source from its file before trying its location, and "
                f"read none in {py_file} ({failure}): make the file readable, with the source in it; or {silence}"
            )
        if isinstance(failure, OSError) and filename and os.path.abspath(filename) == os.path.abspath(py_file):
            # numba reads the source itself: its IPython locator reads the function's source, and from 0.68 numba
            # reads the file for the stamp, which hashes the source. The remedy reports a file that cannot be read as
            # such, not as the location that numba took refusing the file.
            reason = failure.strerror or failure
            return (
                f"numba reads the function's source from its file, {py_file}, which cannot be read ({reason}): make "
                f"it readable; or {silence}"
            )
        # numba took no location if no locator in the list takes the file or has a place for it. A list of the
        # user-provided locator alone with NUMBA_CACHE_DIR unset is one such list. The list itself is then the remedy,
        # below, for an error that a caller passes on, as it is for numba's error that no locator takes the file. A
        # class outside numba's families has a place of its own, which an error can name.
        if isinstance(failure, OSError) and (any(locations.values()) or not all(of_numbas(cls) for cls in reached)):
            # The error names the location that numba took, or a file that numba writes in that location. numba's own
            # test passed the location because its temporary file fit where numba's cache files do not. Or the file
            # that the error names, which the package's check created, was refused later, by a full disk or by
            # permissions changed since. The locator that took the location decides the remedy: the first in numba's
            # order whose place the error names. Paths are matched whole, because the places can nest, as they do for
            # an install under either cache directory, or for NUMBA_CACHE_DIR set above the user's cache directory.
            # They are matched in order, because NUMBA_CACHE_DIR set to the user's cache directory makes one path of
            # two.
            if filename:
                named = os.path.abspath(filename)
                named = {named, os.path.dirname(named)}
            else:
                # A caller can pass on an error that names nothing, and numba's first save raises one when a write
                # fails, as on a full disk.
                named = set()
            taken = next((cls for cls in reached if locations[cls] and locations[cls] in named), None)
            shared = next((cls for cls in reversed(reached) if locations[cls] and locations[cls] in named), None)
            # A name too long is refused whatever the errno, EINVAL on Windows.
            named_after_the_source = taken is not None and (
                reads_the_variable(taken) or caches_under_the_user_cache_dir(taken))
            too_long = too_long_a_name(locations[taken]) if named_after_the_source else None
            if too_long:
                return f"{too_long}: put the file in a directory of a shorter name{beside_the_source()}; or {silence}"
            too_long_file = too_long_a_file_name(locations[taken]) if taken is not None else None
            if too_long_file:
                # The clause offers the variable, pointed at a directory on a file system that allows the name, on two
                # conditions. A file system in use must allow the name, and numba must try the variable before the
                # locator that took the file. Those are the terms on which the variable is offered at a short path for
                # a path too long. The variable's own location took the file only because the variable is set, so the
                # variable is named as set.
                clause, a_file_system_takes_it = too_long_file
                offer = ""
                if a_file_system_takes_it:
                    fits = "directory on a file system that takes a name that long"
                    if reads_the_variable(taken):
                        offer = f", or NUMBA_CACHE_DIR, which is set to {config.CACHE_DIR}, pointed at a {fits}"
                    else:
                        offer = instead(taken, asks=f"a {fits}", shared=shared, made=f"a writable {fits}")
                return f"{clause}, and no shorter path cures it: {a_shorter_name}{offer}; or {silence}"
            too_long_for_the_names = a_name_limit_below(locations[taken]) if taken is not None else None
            if too_long_for_the_names:
                fits = "directory on a file system that takes a name that long"
                if reads_the_variable(taken):
                    offer = f", or NUMBA_CACHE_DIR, which is set to {config.CACHE_DIR}, pointed at a {fits}"
                else:
                    offer = instead(taken, asks=f"a {fits}", shared=shared, made=f"a writable {fits}")
                return f"{too_long_for_the_names}, so no shorter path cures it{offer}; or {silence}"
            if failure.errno == errno.ENAMETOOLONG:
                if taken is not None and reads_the_variable(taken):
                    # numba's own test passed the variable's location, and the first save's longer name overflows it.
                    # numba keeps taking the variable as it is, so the user's cache directory at a shorter path would
                    # move nothing that numba reads, even if the variable is set to that directory.
                    cure = "a shorter NUMBA_CACHE_DIR"
                elif taken is not None and caches_beside_the_source(taken):
                    cure = f"{package} installed at a shorter path{instead(taken)}"
                elif taken is not None and caches_under_the_user_cache_dir(taken):
                    cure = (
                        f"the user's cache directory, {user_cache_dir}, at a shorter path{_moved_through()}"
                        f"{instead(taken, shared=shared)}"
                    )
                elif taken is not None and one_of(taken, for_ipython):
                    cure = f"IPython's cache directory, {os.path.dirname(locations[taken])}, at a shorter path{instead(taken)}"
                elif not filename:
                    cure = f"{could_have_taken(locations)}, at a shorter path{instead(None)}"
                else:
                    cure = f"that location, {filename}, at a shorter path{instead(None)}"
                return f"the path is too long for the file system: {cure}; or {silence}"
            if (failure.errno == errno.EISDIR and zip_at is not None and filename and os.path.isdir(filename)
                    and filename.endswith(".zip") and names_a_path_above()):
                # numba's .zip locator takes a file on disk because the path contains ".zip", after the locators before
                # it passed the file over. It takes the first part of the path that ends in .zip for the archive, a
                # directory so named included. numba 0.68 and later open that part to read the source, which a
                # directory refuses. That directory is no cache location, so the remedy does not ask for free space in
                # it. The remedy for a member handles the same case. The error is a caller's under a list with no .zip
                # locator, and the remedy then names the location that the error gives.
                if zip_at:
                    reached_how = "reached once the locators before it passed this file over"
                    cure = f"make one of those locators' locations writable{instead(None, 'a writable directory')}, or "
                else:
                    # The .zip locator, listed first, takes the file before any locator for a file on disk, so no
                    # locator before it can be named.
                    reached_how = "which NUMBA_CACHE_LOCATOR_CLASSES puts before every locator for a source file on disk"
                    cure = "list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or " \
                           "UserWideCacheLocator before it, or "
                return (
                    f"numba's .zip locator, {reached_how}, takes the first part of the path ending in .zip, {filename}, "
                    "for an archive, and it is a directory, which numba 0.68 and later open for the source: "
                    f"{cure}put the file under a path with no such directory; or {silence}"
                )
            # If the error names the variable's own location, numba took that location and did not pass the variable
            # over; set to another directory, the variable moves the cache off a full disk. The remedy names a location
            # that is none of numba's as the error gives it. For an error that names none, it names the locations that
            # numba could have taken.
            if taken is not None and reads_the_variable(taken):
                alternative = "" if subpath is None else ", or NUMBA_CACHE_DIR set to another writable directory"
            else:
                alternative = instead(taken, "a writable directory", shared=shared)
            if taken:
                opening = f"numba caches this file in {locations[taken]},"
            elif filename:
                opening = f"that location, {filename},"
            else:
                opening = f"{could_have_taken(locations)},"
            # An error that a caller builds from a message alone has no strerror.
            reason = failure.strerror or failure
            return (
                f"{opening} where no file can be written ({reason}): make room there, or make it "
                f"writable{alternative}; or {silence}"
            )
        # numba itself passes over a location for a source on disk that it cannot create or write in. An OSError
        # therefore reaches this point only if numba took no location, as under a list of the user-provided locator
        # alone with NUMBA_CACHE_DIR unset. Any other error here is one of four. It can be numba's error that no locator
        # takes the file: numba tried every locator, and each passed the file over. It can be the .zip locator's
        # ValueError for a path that contains ".zip" and has no part that ends in ".zip". numba raises that error when
        # it reaches the .zip locator, after the locators before it passed the file over, and tries none after it. It
        # can be the UnicodeEncodeError of a locator that hashes a path that is not UTF-8, or an error that a caller
        # passes on.
        #
        # zip_raised reads any ValueError as the .zip locator's error once that locator is reached, and a
        # UnicodeEncodeError is a ValueError. Such an error still reaches the case for a path that is not UTF-8 below,
        # unless the .zip locator is listed first. Then no locator is tried before it, and a source on disk under a
        # directory named .zip, on a path that is not UTF-8, gets the .zip locator's remedy, "finds no archive there".
        zip_raised = isinstance(failure, ValueError) and zip_at is not None
        tried = order[:zip_at] if zip_raised else order
        if not tried:
            if zip_raised:
                return (
                    "numba's .zip locator, which NUMBA_CACHE_LOCATOR_CLASSES puts before every locator for a source "
                    'file on disk, takes this file for the ".zip" in its path and finds no archive there: list '
                    "UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or UserWideCacheLocator "
                    f"before it, or {silence}"
                )
            return (
                f"numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, {listed}, says, and none of those locators takes "
                "a source file on disk: list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator "
                f"or UserWideCacheLocator, or {silence}"
            )
        named_after_the_source = [
            cls for cls in tried
            if cls in locations and (reads_the_variable(cls) or caches_under_the_user_cache_dir(cls))]
        if named_after_the_source and all(too_long_a_name(locations[cls]) for cls in named_after_the_source):
            # Every locator that numba tried and that names its directory after the source's directory found that name
            # too long for its file system. Each passed the file over for that reason. A class listed after the .zip
            # locator is unreached for a path that the .zip locator takes. The variable's locator measures the name by
            # the file system of the directory that the variable names. If the variable is unset, it measures by 255,
            # the limit of nearly every file system. No writable directory at a short path helps with a name too long,
            # so the remedy asks for none. If the .zip locator then raised for a path that contains ".zip" and has no
            # part that ends in ".zip", the remedy does not ask for the variable without naming the name either.
            too_long = too_long_a_name(locations[named_after_the_source[0]])
            # Under the user-provided locator alone, with the variable unset, numba took no location at all and never
            # measured the name. That locator takes nothing without the variable, and the name would stop it once the
            # variable is set, so the remedy names both.
            variable = (
                ", with NUMBA_CACHE_DIR set, which UserProvidedCacheLocator takes nothing without"
                if not config.CACHE_DIR and all(reads_the_variable(cls) for cls in tried) else "")
            return (
                f"{too_long}: put the file in a directory of a shorter name{variable}{beside_the_source()}; or {silence}"
            )
        hashing = [
            cls for cls in tried if (reads_the_variable(cls) and config.CACHE_DIR) or caches_under_the_user_cache_dir(cls)]
        if subpath is None and (isinstance(failure, UnicodeEncodeError) or hashing):
            # numba hashes the source's directory path as UTF-8 for its directory under NUMBA_CACHE_DIR or the user's
            # cache directory, and the locator that names its directory with that hash raises. That is the variable's
            # locator, if the variable is set, before numba reaches the in-tree locator. The in-tree locator caches
            # beside the source and hashes no path. Or it is the user-wide locator, once the in-tree one passed an
            # unwritable __pycache__ over. The same install imports cached with the variable unset, so the remedy must
            # say more than that the variable does not help and a UTF-8 path is needed. Under a list without the
            # variable's locator, making the listed locations writable does not help either, because hashing refuses
            # them however they are set up. numba hashed the path if the error is its own UnicodeEncodeError, or if it
            # tried a locator that names its directory with the hash. Those are the variable's locator when the
            # variable is set, the user-wide one and the .zip one. The in-tree locator alone hashes nothing, and under
            # it numba's error that no locator takes the file means that numba could not use the __pycache__.
            hashed = (
                "numba hashes the source's directory as UTF-8 for the directory it caches this file in under "
                f"NUMBA_CACHE_DIR or the user's cache directory, and {os.path.dirname(py_file)} is not"
            )

            def hashes(cls):
                """Return whether the locator class ``cls`` names its directory with a hash of the source's path."""
                # The locators that hash are the variable's locator when the variable is set, the user-wide locator,
                # and the .zip locator for a path that contains ".zip". The in-tree locator caches beside the source,
                # in its __pycache__, and hashes no path.
                return ((reads_the_variable(cls) and bool(config.CACHE_DIR)) or caches_under_the_user_cache_dir(cls)
                        or (one_of(cls, for_a_zip) and ".zip" in py_file))

            beside = os.path.abspath(os.path.join(os.path.dirname(py_file), "__pycache__"))
            # Listed after any locator that hashes, the in-tree locator is never reached, so making the __pycache__
            # writable or listing the in-tree locator after the hashing one does not help: numba still raises. The fix
            # is to list the in-tree locator before the first locator that hashes. If the variable's locator is the
            # only hashing one before it, the fix is to unset the variable. If the in-tree locator came first and
            # passed the file over, the fix is to make the __pycache__ writable.
            in_tree_at = next((at for at, cls in enumerate(order) if caches_beside_the_source(cls)), None)
            before_in_tree = [cls for cls in (order if in_tree_at is None else order[:in_tree_at]) if hashes(cls)]
            # The in-tree locator is asked whether it takes the file, as numba asks it: it takes a file whose
            # __pycache__ it can write. If the remedy said "where that directory can be written" instead of asking the
            # locator, and never asked for the directory to be made writable, numba would raise the same error after
            # the reader unset the variable.
            if _takes(in_tree, py_file):
                caches_beside = f"numba caches beside the source, in its __pycache__, {beside}, hashing no path"
            else:
                caches_beside = (
                    f"make the __pycache__ beside the source, {beside}, writable, and numba caches there, hashing no "
                    "path")
            if in_tree_at is None:
                where = f" before {before_in_tree[0].__name__}" if before_in_tree else ""
                return (
                    f"{hashed}: put the file under a UTF-8 path, or list InTreeCacheLocator{where}, and "
                    f"{caches_beside}; or {silence}"
                )
            if before_in_tree and all(reads_the_variable(cls) for cls in before_in_tree):
                return (
                    f"{hashed}, and NUMBA_CACHE_DIR, which is set to {config.CACHE_DIR}, is hashed for first: unset "
                    f"it, and {caches_beside}; or put the file under a UTF-8 path; or {silence}"
                )
            if before_in_tree:
                return (
                    f"{hashed}, and NUMBA_CACHE_LOCATOR_CLASSES puts {before_in_tree[0].__name__}, which hashes it, "
                    f"before InTreeCacheLocator: list InTreeCacheLocator before it, and {caches_beside}; or put the "
                    f"file under a UTF-8 path; or {silence}"
                )
            return (
                f"{hashed}: put the file under a UTF-8 path, or make the __pycache__ beside the source, {beside}, "
                f"writable, where InTreeCacheLocator caches it hashing no path; or {silence}"
            )
        if not any(reads_the_variable(cls) for cls in tried):
            if zip_raised:
                return (
                    'numba\'s .zip locator takes this file for the ".zip" in its path and finds no archive there, after '
                    f"every locator NUMBA_CACHE_LOCATOR_CLASSES, {listed}, puts before it passed the file over: make "
                    "one of those locations writable, or list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, "
                    f"InTreeCacheLocator or UserWideCacheLocator before it, or {silence}"
                )
            return (
                f"numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, {listed}, says: make one of those locations "
                f"writable, or {silence}"
            )
        if config.CACHE_DIR:
            # The variable is set and numba passed it over: the directory that it names is unwritable, or too deep for
            # numba to create its own directory there, and a writable directory at the same depth would not fix the
            # depth.
            return (
                f"NUMBA_CACHE_DIR is set to {config.CACHE_DIR}, which numba could not use: set it to a writable "
                f"directory at a short path, or {silence}"
            )
        return f"Set NUMBA_CACHE_DIR to a writable directory, or {silence}"

    def the_source_files():
        """Return the clauses that ask for the source files, installed on disk and in a ``.zip``.

        Each clause names the locator that it needs, if ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves that locator out."""
        on_disk = f"install {package} with its source files on disk, unpacked from any archive"
        if listed:
            # The install on disk caches through a locator for a file on disk that the list names. If the list names
            # no such locator, the install caches no more than the archive does. If the list names the user-provided
            # locator alone, that locator takes nothing without NUMBA_CACHE_DIR.
            for_a_file_on_disk = [
                cls for cls in _locators(listed)
                if any(one_of(cls, base) for base in (user_provided, in_tree, user_wide))
            ]
            if not for_a_file_on_disk:
                on_disk += (
                    ", and list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or "
                    f"UserWideCacheLocator in NUMBA_CACHE_LOCATOR_CLASSES, {listed}, which has none of them"
                )
            elif all(one_of(cls, user_provided) for cls in for_a_file_on_disk) and not config.CACHE_DIR:
                on_disk += (
                    ", with NUMBA_CACHE_DIR set, which UserProvidedCacheLocator, the one locator for a file on disk in "
                    f"NUMBA_CACHE_LOCATOR_CLASSES, {listed}, takes nothing without"
                )
        a_zip = (
            "import it from a .zip holding its source files, which numba caches in the user's cache "
            "directory"
        )
        found = _archive_above(py_file)
        if found is not None and not found[0].endswith(".zip"):
            # The file is already in an archive that numba's .zip locator never takes, such as APP.ZIP or an .egg. The
            # locator looks for a part of the path that ends in ".zip", lower-case. The remedy asks for such an archive
            # to be renamed, naming the extension that numba does not read, instead of offering a .zip that the reader
            # already uses. The remedy offers the rename even if a directory above the archive has ".zip" in its name,
            # which makes the locator take the file and find no archive. numba takes the first part of the path that
            # ends in .zip for the archive, so a directory so named above the archive is in the way. The rename alone
            # would leave numba opening the directory, so the remedy asks the reader to move the archive out from under
            # that directory.
            archive_parts = pathlib.Path(found[0]).parts
            in_the_way = next(
                (str(pathlib.Path(*archive_parts[:at + 1])) for at, part in enumerate(archive_parts[:-1])
                 if part.endswith(".zip")), None)
            moved = (
                f", and put it under a path with no part ending in .zip above it, {in_the_way} being one, which that "
                "locator takes for the archive first" if in_the_way else "")
            a_zip = (
                f"rename the archive, {found[0]}, to end in .zip, lower-case, which numba's .zip locator looks for in "
                f"the path, with its source files in it{moved}, so that numba caches it in the user's "
                "cache directory"
            )
        if listed_without(for_a_zip):
            # numba's .zip locator alone takes a .zip, so a list without it would leave numba no locator for the .zip
            # that the remedy offers either.
            a_zip += f" through ZipCacheLocator, once NUMBA_CACHE_LOCATOR_CLASSES, {listed}, lists it"
        return on_disk, a_zip

    if isinstance(failure, KeyError):
        # numba 0.68 and later read a .zip member's source from the archive for the stamp, by the name that the
        # module's code gives under the archive's path. zipfile raises KeyError for a name that the archive does not
        # hold. That happens for a module shipped as .pyc alone. It happens on Windows under numba 0.68.0 for every
        # member, because that release builds the name with the platform's separator (numba issue 10889, fixed in
        # 0.68.1). The archive is searched for the member, so that the remedy does not ask for source files that are
        # there. A backslash alone does not decide the Windows case either: a module shipped as .pyc alone has no
        # source under the name that numba 0.68.1 reads, so that release does not fix it.
        member = _missing_member(failure)
        held = member.replace("\\", "/")
        on_disk, a_zip = the_source_files()
        if os.name == "nt" and "\\" in member and _numbas_archive_holds(py_file, held):
            # The install on disk fixes the case, and so does a numba release that carries the fix, such as numba
            # 0.68.1: it reads the member under the name with forward slashes, as the archive names it.
            return (
                f"numba 0.68.0 reads a .zip member's source by a name built with Windows's separator, {member}, which "
                "no archive holds (numba issue https://github.com/numba/numba/issues/10889, fixed for numba 0.68.1): "
                f"to cache, {on_disk}, which no separator touches, or a numba release that carries the fix, which "
                f"reads the member as the archive names it; or {silence}"
            )
        naming = _pyc_naming(py_file)
        if naming is not None and naming[1]:
            # The archive holds the module's source beside a .pyc compiled to name another path in the archive.
            # zipimport runs that .pyc, and numba reads the source by the name that the .pyc gives. The module is not
            # shipped as .pyc alone, so the remedy does not ask for source files that the archive already holds.
            compiled, source = naming[0], naming[0][:-1]
            return (
                f"numba 0.68 and later read a .zip member's source from the archive by the name the module's code "
                f"gives, {held}, which {compiled} was compiled to name, and this one holds none so named, while it "
                f"holds the module's source, {source}: to cache, compile the .pyc to name its path in the archive, "
                f"or leave it out, so that Python compiles the source and numba reads it under that name; or {silence}"
            )
        return (
            f"numba 0.68 and later read a .zip member's source from the archive, and this one holds none under "
            f"{held}, the module shipped as .pyc alone: to cache, {on_disk}, or {a_zip}; or {silence}"
        )
    # order lists the locators that take a file not on disk, in numba's order. numba's .zip locator takes the file
    # without trying its location, and a class outside numba's families that takes the file when asked stops the
    # order too, so numba reaches no locator listed after either. numba's IPython and user-wide locators pass over a
    # location that they cannot create or write in, and numba tries the next locator.
    order = [cls for cls in _locators(listed) if may_take(cls)]
    zip_at = next((at for at, cls in enumerate(order) if one_of(cls, for_a_zip)), None)
    stops_at = next((at for at, cls in enumerate(order) if one_of(cls, for_a_zip) or not of_numbas(cls)), None)
    reached = order if stops_at is None else order[:stops_at + 1]
    # A .zip locator listed after such a class is not reached, and under that list its error for a path with no part
    # that ends in .zip is a caller's.
    zip_at = zip_at if zip_at == stops_at else None
    user_cache_dir = AppDirs(appname="numba", appauthor=False).user_cache_dir
    in_the_user_cache = os.path.join(user_cache_dir, subpath) if subpath else None

    def location_of(cls):
        """Return the location where the locator class ``cls`` places a file not on disk, or None if there is none."""
        # The IPython locator places the file in numba_cache under IPython's cache directory. The result is None
        # without IPython, or when IPython raises OSError for its directory and numba passes the locator over. The
        # remedy asks IPython for the directory only when numba reaches a locator of that family, never for any other
        # file. If it asked for every .zip member, an IPython whose own directory is unwritable would warn on every
        # import of the package and leave a temporary directory behind. The .zip and user-wide locators place the file
        # in a directory of their own under the user's cache directory. A class outside numba's families has a place
        # of its own, which is None here.
        if one_of(cls, for_ipython):
            ipython_cache = ipython_numba_cache()
            return os.path.abspath(ipython_cache) if ipython_cache else None
        if not of_numbas(cls) or in_the_user_cache is None:
            return None
        return os.path.abspath(in_the_user_cache)

    locations = {cls: location_of(cls) for cls in reached}
    took_no_location = not any(locations.values()) and all(of_numbas(cls) for cls in reached)
    if getattr(sys, "frozen", False) and listed_without(user_wide) and (
            took_no_location or not isinstance(failure, (OSError, ValueError))):
        # Of numba's locators, only the user-wide one takes a frozen application, so a list without it leaves numba no
        # locator of its own for one. This case covers numba's error that no locator takes the file, whatever classes
        # of the reader's own the list holds, since those passed the file over. It also covers any error that a caller
        # passes on when the list has no place for the file. Asking for a writable directory would name one that numba
        # never tried.
        return (
            "numba caches a frozen application through UserWideCacheLocator alone, and "
            f"NUMBA_CACHE_LOCATOR_CLASSES, {listed}, leaves it out: list it, or {silence}"
        )
    # numba took no location if no locator in the list takes the file or has a place for it. The list is then the
    # remedy, below, for an error that a caller passes on, as for numba's error that no locator takes the file and as
    # for a source on disk. A class outside numba's families has a place of its own.
    parts = pathlib.Path(py_file).parts
    in_a_zip = any(part.endswith(".zip") for part in parts)
    # a_directory_named_zip is true if a directory named .zip lies above the file. numba's .zip locator takes such a
    # directory for an archive all the same, and caches nothing under it. A cell under such a directory, whose source
    # linecache does not hold, must not get the archive remedy, the source files to ship.
    a_directory_named_zip = any(
        os.path.isdir(os.path.join(*parts[:at + 1])) for at, part in enumerate(parts) if part.endswith(".zip"))
    # in_an_archive is true for a member of any archive, an .egg or a .whl as well as a .zip. zipimport runs such a
    # member, and inspect reads its source through its loader. A .zip is the one kind that numba's .zip locator takes,
    # and the only kind that it caches.
    in_an_archive = (in_a_zip and not a_directory_named_zip) or _archive_above(py_file) is not None
    # A caller's error with no errno and no name looks like inspect's, so inspect's errors are known by their words,
    # and the remedy never treats a caller's error as inspect's.
    inspects = _is_inspects(failure) and any(one_of(cls, for_ipython) for cls in reached)
    if inspects and not in_an_archive:
        # This is inspect's error, with no errno and no name, for a cell whose source linecache does not hold. numba's
        # IPython locator reads a cell's source from linecache, where IPython holds it while the cell runs. inspect
        # raises before the locator tries its location, so no location that numba took refused a file. inspect's error
        # for a member of an archive that holds no source for it, a .pyc alone, gets the archive remedy below, the
        # source files to ship. The archive is found by looking, not by its name, so a member of an .egg gets that
        # remedy too.
        return (
            f"numba's IPython locator reads a cell's source from linecache, which holds none for {py_file} "
            f"({failure}), and numba caches a cell only while IPython holds its source; or {silence}"
        )
    not_utf8 = _member_not_utf8(py_file) if inspects else None
    if not_utf8:
        # This is inspect's error for a member of an archive that holds its source, which is not UTF-8. zipimport
        # decodes a member's source as UTF-8 and raises for one that is not. The probe's loader reads the member as
        # zipimport does but returns None, so the check raises inspect's error, a cache error, instead of the decoding
        # error. The decoding error is not a cache error and would end the import, as numba's would at decoration.
        return (
            f"numba's IPython locator reads the member's source as UTF-8, which {py_file} is not ({not_utf8}): ship "
            f"it as UTF-8; or {silence}"
        )
    hashed_by = next((cls for cls in reached if (one_of(cls, user_wide) and getattr(sys, "frozen", False))
                      or (one_of(cls, for_a_zip) and in_a_zip)), None)
    if isinstance(failure, UnicodeEncodeError) and hashed_by is not None:
        # numba's .zip locator names its directory under the user's cache directory after the source's directory, with
        # the archive's path in it hashed as UTF-8. It raises for bytes that are not UTF-8, at decoration and in the
        # check alike. The user-wide locator does the same for a frozen application's file. The error is not a
        # caller's with no place for the file, so the remedy names the path, not the .zip that the reader imports from
        # or the source files on disk. The first of the two locators that numba reaches is the one that raised. The
        # user-wide locator comes before the .zip locator in numba's order, so for a frozen application under a
        # directory named .zip the remedy names the application, not an archive.
        what, where = (("a .zip", "the archive") if one_of(hashed_by, for_a_zip)
                       else ("a frozen application", "the application"))
        return (
            f"numba caches {what} in the user's cache directory, and NUMBA_CACHE_DIR has no effect here, because the "
            "source is not a file on disk: numba hashes the source's directory as UTF-8 for the directory it caches "
            f"this file in there, and {os.path.dirname(py_file)} is not: put {where} under a UTF-8 path, or "
            f"{silence}"
        )
    if not took_no_location and not inspects and (isinstance(failure, OSError) or getattr(sys, "frozen", False)):
        if isinstance(failure, ValueError) and zip_at is not None:
            # numba's .zip locator takes a frozen application's file for the ".zip" in its path, without trying its
            # location. It raises if no part of the path ends in .zip, an archive or a directory. Listed before the
            # user-wide locator, which alone takes the file, it leaves that locator unreached, and the remedy is to
            # list the user-wide locator before the .zip one. Listed after the user-wide locator, the .zip locator
            # raises once the user-wide one passed its location over, as unwritable or too deep. That location is then
            # the remedy, as for a source on disk.
            if not any(one_of(cls, user_wide) for cls in order[:zip_at]):
                return (
                    f"numba's .zip locator, listed in NUMBA_CACHE_LOCATOR_CLASSES, {listed}, with no "
                    'UserWideCacheLocator before it, takes this file for the ".zip" in its path and finds no archive '
                    f"there: list UserWideCacheLocator, the one locator for a frozen application, before it, or {silence}"
                )
            return (
                'numba\'s .zip locator takes this file for the ".zip" in its path and finds no archive there, after '
                "UserWideCacheLocator, which numba tries before it, could not use the user's cache directory, "
                f"{user_cache_dir}: make it writable, or put it at a shorter path{_moved_through()}, or {silence}"
            )
        if filename:
            named = os.path.abspath(filename)
            named = {named, os.path.dirname(named)}
        else:
            named = set()
        # The error names the location that numba took, or a file that numba writes in it, as the first save's error
        # does. The locator that took the location decides the remedy: the first reached whose place the error names.
        # The frozen application's error that no locator takes the file names nothing.
        taken = next((cls for cls in reached if locations[cls] and locations[cls] in named), None)
        if taken is not None and one_of(taken, for_ipython):
            # numba caches the member in numba_cache under IPython's cache directory. The remedy is the one that a cell
            # file on disk gets there, with no NUMBA_CACHE_DIR to offer for a source not on disk.
            ipython_cache = locations[taken]
            too_long_file = too_long_a_file_name(ipython_cache)
            if too_long_file:
                return f"{too_long_file[0]}, and no shorter path cures it: {a_shorter_name}; or {silence}"
            too_long_for_the_names = a_name_limit_below(ipython_cache)
            if too_long_for_the_names:
                return f"{too_long_for_the_names}, so no shorter path cures it; or {silence}"
            if failure.errno == errno.ENAMETOOLONG:
                return (
                    f"the path is too long for the file system: IPython's cache directory, "
                    f"{os.path.dirname(ipython_cache)}, at a shorter path; or {silence}"
                )
            reason = failure.strerror or failure
            return (
                f"numba caches this file in {ipython_cache}, where no file can be written ({reason}): make room "
                f"there, or make it writable; or {silence}"
            )
        # The .zip's location, a directory of numba's own under the user's cache directory, is what a reader can act
        # on. The error names it, or names a file in it. For an error that names no file, the remedy names the
        # locations that numba could have taken, IPython's numba_cache among them if numba reaches a locator of that
        # family. The frozen application's error that no locator takes the file names nothing, and the remedy names
        # the user's cache directory.
        if taken is not None:
            location = filename if os.path.abspath(filename) == locations[taken] else os.path.dirname(filename)
        else:
            location = filename or user_cache_dir
        # in_the_user_cache_alone decides whether the remedy says where numba caches the file. A .zip's or a frozen
        # application's location is under the user's cache directory, and a location that is none of numba's is not. If
        # numba reaches a locator of IPython's family or a class outside numba's families for a member, an error that
        # names no file may be that locator's, whether or not IPython gave its directory. So nothing is said of the
        # user's cache directory for any of these.
        in_the_user_cache_alone = taken is not None or (
            not filename and all(of_numbas(cls) and not one_of(cls, for_ipython) for cls in reached))
        where = (
            "numba caches a .zip, or a frozen application, in the user's cache directory, and "
            if in_the_user_cache_alone else "")
        opening = f"{where}NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: "
        if isinstance(failure, FileNotFoundError) and names_a_path_above():
            # numba reads the stamp of the archive that the code names at decoration: here, .pyc members compiled to
            # name an archive that has since moved. A location that cannot be created, under a dangling link, raises
            # ENOENT too, but it does not lie above the code's file.
            suggestion = "" if numba_version >= 68 else "compile the archive's .pyc members to name its path now, or "
            return (
                f"the module's code names {filename}, which is not there: its .pyc was compiled to name "
                f"that path, so {suggestion}ship its source files; "
                f"or {silence}"
            )
        if (isinstance(failure, OSError) and failure.errno == errno.EISDIR and filename and os.path.isdir(filename)
                and filename.endswith(".zip") and names_a_path_above()):
            # numba takes the first part of the path that ends in .zip for the archive, a directory so named above the
            # file included. numba 0.68 and later open that part to read the member's source, which a directory
            # refuses. The directory is no cache location, so the remedy does not ask for free space in it.
            return (
                f"{opening}numba takes the first part of the path ending in .zip, {filename}, for the archive, and it "
                "is a directory, which numba 0.68 and later open to read the member's source: put the archive under "
                f"a path with no such directory above it, or {silence}"
            )
        if isinstance(failure, OSError) and names_a_path_above():
            # The archive that the member's code names cannot be read. numba reads it for the stamp, and from 0.68 it
            # opens the archive for the member's source. The remedy reports the archive as unreadable; making it
            # writable, as for a location that refused a file, would let numba read nothing.
            reason = failure.strerror or failure
            return (
                f"{opening}numba reads the archive, {filename}, for the member's stamp, and from 0.68 for the "
                f"member's source, and cannot ({reason}): make it readable, or {silence}"
            )
        if isinstance(failure, OSError) and filename and taken is None:
            # The error names a location that is none of numba's, under the list and in its order, as an error that a
            # caller passes on can. The remedy names this location as the error gives it, as for a source on disk. It
            # says nothing of what moves the user's cache directory, because moving that directory moves nothing of this
            # location. It says nothing of where numba caches the file either, because this location is none of numba's.
            # One such case is a member that numba's IPython locator takes while IPython is not importable, which numba
            # caches nowhere.
            if failure.errno == errno.ENAMETOOLONG:
                return (
                    f"{opening}the path is too long for the file system, so put that location, {filename}, at a "
                    f"shorter path, or {silence}"
                )
            reason = failure.strerror or failure
            return (
                f"{opening}no file can be written at that location, {filename} ({reason}), so make room there, or "
                f"make it writable, or {silence}"
            )
        too_long = too_long_a_name(locations[taken]) if isinstance(failure, OSError) and taken is not None else None
        if too_long:
            # The .zip's directory under the user's cache directory, named after the source's directory, is refused
            # whatever the errno, which is EINVAL on Windows. The user's cache directory at a shorter path does not help
            # with a name that is too long, so the remedy does not ask for a shorter path.
            return f"{opening}{too_long}, so put the file in a directory of a shorter name, or {silence}"
        too_long_file = (
            too_long_a_file_name(locations[taken]) if isinstance(failure, OSError) and taken is not None else None)
        if too_long_file:
            return f"{opening}{too_long_file[0]}, and no shorter path cures it: {a_shorter_name}, or {silence}"
        too_long_for_the_names = (
            a_name_limit_below(locations[taken]) if isinstance(failure, OSError) and taken is not None else None)
        if too_long_for_the_names:
            return f"{opening}{too_long_for_the_names}, so no shorter path cures it, or {silence}"
        if isinstance(failure, OSError) and failure.errno == errno.ENAMETOOLONG:
            if taken is None:
                # The error names no file, so the remedy cannot know which of the locations is too long. The user's
                # cache directory moves them all if numba reaches no locator of IPython's family and no class outside
                # numba's families.
                moved_through = _moved_through() if in_the_user_cache_alone else ""
                return (
                    f"{opening}the path is too long for the file system, so put {could_have_taken(locations)}, at a "
                    f"shorter path{moved_through}, or {silence}"
                )
            # The names that numba adds under the user's cache directory are bounded, so that directory is what makes
            # the location too long.
            return (
                f"{opening}the path is too long for the file system, so put that directory, {location}, at a shorter "
                f"path{_moved_through()}, or {silence}"
            )
        if isinstance(failure, OSError):
            # The location refused a file for another reason, a full disk or permissions, which the error gives, and
            # the remedy repeats it as it does for a source on disk. The frozen application's error that no locator
            # takes the file gives no reason, because numba passed the location over on its own error.
            reason = failure.strerror or failure
            if taken is None:
                return (
                    f"{opening}no file can be written in {could_have_taken(locations)} ({reason}), so make room "
                    f"there, or make it writable, or {silence}"
                )
            # numba's .zip locator takes the file without creating its location. The first save creates it, or the
            # check does, under the first directory above it that exists, and that directory refused. The remedy
            # therefore names that directory: asking for free space in a directory that is not there, or for it to be
            # made writable, would name the wrong one. An error that names a file in the location found the location
            # there.
            made_under = ""
            if os.path.abspath(filename) == locations[taken] and not os.path.isdir(location):
                above = _existing_above(location)
                if not os.path.isdir(above):
                    # A file is in the way, and the remedy names it: free space in the directory above it would not
                    # help.
                    return (
                        f"{opening}no file can be written in that directory, {location}, which cannot be made, "
                        f"{above} being a file in its path ({reason}), so put that file elsewhere, or {silence}"
                    )
                made_under = f", which cannot be made under {above}"
            return (
                f"{opening}no file can be written in that directory, {location}{made_under} ({reason}), so make room "
                f"there, or make it writable, or {silence}"
            )
        # numba's error that no locator takes a frozen application's file gives no reason, because numba passed the
        # user-wide location over on its own error. The location may be an unwritable directory, one too deep to
        # create, or one whose name, after the source's directory, is too long for the file system. The remedy names
        # the last as such, because no shorter path helps.
        too_long = too_long_a_name(in_the_user_cache)
        if too_long:
            return f"{opening}{too_long}, so put the file in a directory of a shorter name, or {silence}"
        return (
            f"{opening}numba could not use that directory, {location}: make it writable, or put it at a shorter "
            f"path{_moved_through()}, or {silence}"
        )
    of_ipythons = [cls for cls in reached if one_of(cls, for_ipython)]
    if not isinstance(failure, (OSError, ValueError)) and of_ipythons and all(
            one_of(cls, for_ipython) or not of_numbas(cls) for cls in reached):
        # This is numba's error that no locator takes a file that only numba's IPython locator takes: a cell, or a .zip
        # member in an ipykernel_ directory under a list with no .zip locator after the IPython one. The locator passed
        # its location over, as numba's user-wide one passes over a frozen application's. The location was unwritable or
        # too deep to create, or IPython raised for its directory and gave none. A class outside numba's families listed
        # with the IPython locator passed the file over too, as numba's error shows, and the remedy names it as such.
        # Once asked, such a class drops out of the order, so the remedy names it from the list. If the remedy took it
        # for one of numba's locators, the remedy would be the archive's.
        ipython_cache = locations[of_ipythons[0]]
        passed_over = [cls.__name__ for cls in _locators(listed) if not of_numbas(cls)]
        passed_over = f", and {' and '.join(passed_over)}, listed too, passed the file over" if passed_over else ""
        if ipython_cache:
            return (
                f"numba caches this file in {ipython_cache}, which it could not make or write in{passed_over}: make "
                f"it writable, or put IPython's cache directory, {os.path.dirname(ipython_cache)}, at a shorter path; "
                f"or {silence}"
            )
        return (
            "numba caches this file in numba_cache under IPython's cache directory, which IPython raised for, under a "
            f"home it cannot write{passed_over}: make that home writable; or {silence}"
        )
    if _taken_by_ipython(py_file) and not in_a_zip and listed_without(for_ipython):
        # numba's IPython locator alone takes a cell, or a file in an ipykernel_ directory that is not on disk, so a
        # list without it leaves numba no locator for one. The archive remedy would offer source files on disk and
        # locators that take none of these.
        return (
            f"numba caches a notebook cell through IPythonCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, {listed}, "
            f"leaves it out: list it; or {silence}"
        )
    if in_a_zip and not a_directory_named_zip and listed_without(for_a_zip):
        # numba's .zip locator alone takes a source in a .zip, so a list without it leaves numba no locator for one,
        # and the archive remedy would offer the .zip that the reader already imports from. A source that is nowhere,
        # under a directory named .zip, is left out, because that locator takes the directory for an archive and
        # caches nothing.
        return (
            f"numba caches a .zip through ZipCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, {listed}, leaves "
            f"it out: list it; or {silence}"
        )
    on_disk, a_zip = the_source_files()
    return (
        f"NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, {on_disk}, or "
        f"{a_zip}; or {silence}"
    )


def uncached_where_no_cache_can_be_written(options):
    """Return ``options``, or a copy with ``"cache"`` set to ``False`` after one warning if numba cannot cache numbox.

    numba sets a cached function's cache up when it decorates the function, and raises ``RuntimeError`` there if it
    finds no location that it can write in. That happens in a read-only install whose user cache directory cannot be
    written either, and for a module imported from an ``.egg``, ``.whl`` or ``.pyz`` archive. Spark ships ``.egg``
    files, as well as ``.zip`` and ``.py`` files, to its executors through the ``--py-files`` option of
    ``spark-submit``. For a ``.zip``, numba uses the user's cache directory without testing at decoration that it can
    write there, and the first save raises ``OSError`` instead. Either error would end numbox's import at its first
    decorated function, with nothing to say what to change.

    numbox decorates every function that it caches under the one ``jit_options``, so this function runs the check
    once, on a source file in each directory of numbox that holds a module, and answers for all of numbox. If numba
    could not cache a function of one of those files, the options come back with ``"cache"`` set to ``False``. One
    ``RuntimeWarning`` then gives numba's error and the remedy that :func:`cache_remedy` composes for it. The remedy
    ends with ``NUMBOX_JIT_OPTIONS='{"cache": false}'``, which turns caching off and silences the warning. An error
    that is not a cache error, as :func:`is_a_cache_error` defines it, is raised as it was.

    Options with ``"cache"`` set to ``False`` come back as they are, and the check does not run.
    ``NUMBOX_JIT_OPTIONS`` can give options without a ``"cache"`` key; such options leave ``njit`` at its default,
    which is not to cache. The check still runs for them, because
    :func:`~numbox.core.variable.compile_kernel.compile_kernel` caches its kernels under such options, and so do
    numbox's SQLite callbacks for virtual tables and table-valued functions. If numba cannot cache numbox, these
    options come back with ``"cache"`` set to ``False`` explicitly.

    The check is stricter than numba for a ``.zip`` whose cache directory still holds every entry but can no longer be
    written: numbox compiles without a cache there, where numba alone would load the entries. The test that the
    location can be written in is the rule that numba applies to every other location, and the one that its ``.zip``
    locator lacks, as https://github.com/numba/numba/issues/10888 reports.
    """
    if not options.get("cache", True):
        return options
    for py_file in _module_files():
        try:
            check_cache_location(py_file, LONGEST_CACHE_FILE_NAME)
        # numba's cache errors belong to these four classes; a UnicodeEncodeError is a ValueError.
        except (RuntimeError, OSError, KeyError, ValueError) as error:
            if not is_a_cache_error(error):
                raise
            failure = error
            break
    else:
        return options
    silence = "NUMBOX_JIT_OPTIONS='{\"cache\": false}' to turn caching off and silence this warning"
    remedy = cache_remedy(py_file, failure, silence)
    warnings.warn(
        f"numba cannot cache numbox here ({failure}); it compiles without a cache. {remedy}",
        RuntimeWarning, stacklevel=2,
    )
    return {**options, "cache": False}


#: The keyword arguments that numbox passes to numba's ``njit`` for its own functions. They are the options that
#: ``NUMBOX_JIT_OPTIONS`` sets, or ``{"cache": True}`` if it is unset, read when this module is first imported, with
#: ``"cache"`` set to ``False`` if numba cannot cache numbox. A package's own functions compile under the same options
#: with ``@njit(**jit_options)``.
jit_options = uncached_where_no_cache_can_be_written(get_jit_options())


_PROXY_CACHE_STRICT_ENV = "NUMBOX_PROXY_CACHE_STRICT"


def _strict_cache_mode():
    """Return whether ``NUMBOX_PROXY_CACHE_STRICT`` selects strict validation of ``@proxy`` cache loads.

    In strict mode, the cache guard of ``numbox.core.proxy.proxy`` raises on each path where it would otherwise load
    unchecked. A payload that it cannot read, or an object whose container it cannot parse, ends the load with
    ``UnvalidatedProxyCacheError``. A stale alias raises ``StaleProxyCacheError`` before the guard discards the entry
    and recompiles, so the stale entry stays on disk. Strict mode is a debugging aid and is off by default. Any value
    enables it except an empty one and ``0``, ``false``, ``no`` or ``off``, in any letter case and with surrounding
    white space ignored. Unlike ``NUMBOX_JIT_OPTIONS``, the variable is read on every call, so a process can switch the
    mode; the read costs one environment lookup, against a cache load of several milliseconds.
    """
    value = os.environ.get(_PROXY_CACHE_STRICT_ENV)
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


MAX_STR_LENGTH = 2 ** 31 - 1

#: numba's ``FunctionModel``, looked up once. A data-model lookup is type machinery and compiles nothing, so this module
#: reads the layout without the compilation that importing ``numbox.utils.lowlevel`` causes: that module compiles a
#: cached eager ``@njit`` helper at import. ``numbox.utils.lowlevel`` reads the field count, ``function_struct_size``.
_function_model = default_manager.lookup(FunctionType(void()))

#: The number of fields in numba's ``FunctionModel`` struct: ``c_addr``, ``py_addr`` and ``jit_addr``.
function_struct_size = _function_model.field_count
