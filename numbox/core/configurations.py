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
    """
    E.g., export NUMBOX_JIT_OPTIONS='{"cache": false}'
    """
    as_str = os.environ.get("NUMBOX_JIT_OPTIONS")
    if as_str is None:
        return {"cache": True}
    try:
        as_json = json.loads(as_str)
    except json.JSONDecodeError:
        raise ValueError("NUMBOX_JIT_OPTIONS must be valid JSON")
    if not isinstance(as_json, dict):
        # The keys go to njit as keyword arguments, and the fallback below
        # reads "cache" from the object, so a value of another shape is
        # refused here, with a message that says what is expected.
        raise ValueError('NUMBOX_JIT_OPTIONS must be a JSON object, e.g. {"cache": false}')
    if "cache" in as_json and not isinstance(as_json["cache"], bool):
        # numba tests the option for truth, so the string "false" would turn caching on.
        raise ValueError('NUMBOX_JIT_OPTIONS "cache" must be true or false, e.g. {"cache": false}')
    return as_json


def _cache_probe():
    """A function of this module that is never compiled; numba looks up its code's file as the module's file."""


# An upper bound, with room to spare, on the length of a cache file name numba writes for a function of numbox's
# own files. The name holds the module's stem, the function's qualified name, numba's line number, the interpreter
# tag and the index number, plus the 21 bytes of the temporary name numba writes the file under. The longest-named
# function of the package comes to 105 bytes, and the builder's generated kernel, anchored to its file, to 117. A
# test holds every function of the package under this bound.
LONGEST_CACHE_FILE_NAME = 128

# The digits reserved for the number numba gives a function's data file. numba writes one data file per signature
# and numbers them from one without bound. Three digits are reserved, because no function of the package or of its
# generated code reaches a thousand signatures; if an error carries a name with a wider number, that number is read
# at its own width. Two digits were not enough: the hundredth signature's name is a byte longer than a two-digit
# reserve allows.
_OVERLOAD_DIGITS = 3

# Each check puts the probe's module in sys.modules under a unique name while
# numba picks the locator; this counter supplies those names.
_probes = itertools.count()


class _MemberSource:
    """The loader of the probe's module. inspect reads a ``.zip`` member's source through it, as it would read the
    member's own module through zipimport.

    ``get_source`` returns the member's bytes decoded as UTF-8, which is how zipimport decodes them. It returns None
    if no archive above the file holds the member, and None if the member is not UTF-8. zipimport raises for such a
    member and linecache lets that error through; None makes inspect raise its own error instead. That error marks
    the member, and the remedy then reads the reason."""

    def __init__(self, py_file):
        self._py_file = py_file

    def get_source(self, name):
        found = _archive_above(self._py_file)
        if found is None:
            return None
        zip_path, member = found
        try:
            with zipfile.ZipFile(zip_path) as archive:
                return archive.read(member).decode()
        except (KeyError, OSError, zipfile.BadZipFile, UnicodeDecodeError):
            return None


@contextlib.contextmanager
def _probing(py_file):
    """A probe for the block: a function that is never compiled, whose code names ``py_file`` as its file.

    The probe is a function of a module of its own, which the block puts in ``sys.modules``. The block asks a locator
    about the file with the probe, as numba asks at decoration.

    ``py_file`` can be a string, bytes or a path object. It is read as a string, because that is what the probe's code
    carries as its file name; ``compile`` makes a string of bytes too. numba's IPython locator reads the function's
    source when it takes the file, the way inspect reads it. inspect reads a file on disk from the file itself, a cell
    from linecache, where IPython holds the cell while it runs, and a ``.zip`` member through the loader of the
    function's module. For a real module that loader is zipimport's, which gives the member's source from the archive,
    or None for a member compiled to ``.pyc`` alone. The probe's module's loader reads the member as zipimport's does,
    and inspect finds the module in ``sys.modules``. After the read, the lines linecache holds for the file are the
    member's own, as any reader would leave them. For a member whose archive holds no source, linecache keeps the
    probe's loader to ask again later; the block removes that linecache entry when it ends, so that no loader of the
    probe outlives the block.
    """
    file = os.fsdecode(py_file)
    # The probe's source is never put in linecache under the member's name:
    # a reader of the member in another thread would then read the probe's
    # two lines instead of the member's own. The loader below gives the
    # member's own source.
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
        if not held and len(linecache.cache.get(file, ())) == 1:
            linecache.cache.pop(file, None)


def _takes(cls, py_file):
    """Whether ``cls``, a locator class outside numba's families, takes ``py_file``.

    Its ``from_function`` is asked with a probe, as numba asks it at decoration. It returns a locator for a file it
    takes and None for a file it passes over. A class whose ``from_function`` raises counts as taking the file: the
    error is its own, and numba raises it at decoration. Without asking, the remedy would take such a class for one
    that takes every file. It would read a file as the class's own when the class had passed the file over and
    numba had gone on to the locators listed after that class."""
    with _probing(py_file) as probe:
        try:
            return cls.from_function(probe, os.fsdecode(py_file)) is not None
        except Exception:
            return True


def check_cache_location(py_file, longest_file_name=0):
    """Raise the error numba would raise if a function whose source file is ``py_file`` cannot be cached; else return.

    ``py_file`` can be a string, bytes or a path object; it is read as a string.

    The question is put to numba the way decoration puts it, in three steps. First the cache set-up, which picks the
    location for the file, or raises ``RuntimeError`` if no locator takes it. Then the source's stamp, which decoration
    reads too: for a ``.zip`` it stats the archive, and from numba 0.68 it also reads the member's source from the
    archive, as from 0.68 it reads a source on disk for its hash. Then the writability check, which decoration runs
    for every location but a ``.zip``'s and the first save runs for all; it raises ``OSError``.

    The probe is compiled with ``py_file`` as its file. That is all a locator reads of it, except numba's IPython
    locator, which also reads the function's source, the way inspect reads it. inspect reads a file on disk from the
    file itself, a cell from linecache, where IPython holds the cell while it runs, and a ``.zip`` member through the
    loader of the function's module; for a real module that loader is zipimport's, which gives the member's source from
    the archive. So the probe is a function of a module of its own, whose loader reads the member as zipimport's does,
    and that module is in ``sys.modules`` while numba picks the locator. After the read, the lines linecache holds for
    the file are the member's own, as any reader would leave them. For a member whose archive holds no source, linecache
    keeps the probe's loader to ask again later; the check removes that linecache entry, so that no loader of the probe
    outlives the check. A cell linecache does not hold, or a member compiled to ``.pyc`` alone, raises inspect's error,
    as it does for numba at decoration.

    Nothing is compiled. Nothing is written but the cache directory itself, with one exception: an IPython that can
    write neither its cache directory nor its own makes a temporary directory on every call, and the check causes
    three such calls, two inside numba's making of the directory and one of its own.

    An ``OSError`` from the check names the location numba picked, with three exceptions. inspect's error names no
    file. For a cell file on disk that numba's IPython locator takes, inspect reads the source from the file itself
    before the locator tries its location, and raises if it finds no source there: a file it cannot read or, before
    Python 3.13, an empty one, which linecache then holds as a blank line. The stamp's error names the archive
    that a ``.zip`` member's code names, if that archive has since moved or cannot be read; the stamp is read once the
    locator is picked, as numba reads it at decoration. From numba 0.68, where the stamp hashes the source, the stamp's
    error can also name the source itself, if it cannot be read. It can also name a directory named ``.zip`` above a
    file on disk, which numba's ``.zip`` locator takes for the archive and opens.

    A location can be absent once made. numba's IPython locator asks IPython for its directory at every save, and an
    IPython that can write neither its cache directory nor its own gives a new temporary directory on every call. So
    the directory numba made is not the one it is told next. The check then raises ``FileNotFoundError`` naming
    ``numba_cache`` under the last directory IPython gave it, which numba never made. numba's first save would
    instead name a file in ``numba_cache`` under the directory IPython gave numba when it set the cache up, a
    different temporary directory.

    numba's writability check makes a temporary file with a short name, or no file at all on Linux, while the files it
    saves have names of up to a hundred bytes and more. So a location within that length of the path limit passes
    numba's check, and the first save dies. Given ``longest_file_name``, the check also makes and removes a file with a
    name that long in the location, and the ``OSError`` it raises names the location. The package asks with
    ``LONGEST_CACHE_FILE_NAME``, the bound on the names numba gives the package's own files. If the name is longer than
    the file system holding the location allows, no shorter path cures that, so the error carries the length as
    ``longest_file_name`` and the remedy asks for a file system that takes the name.
    """
    # What a locator is asked about: a probe, in a module of its own.
    with _probing(py_file) as probe:
        locator = CompileResultCacheImpl(probe).locator
    # numba reads the source's stamp at decoration too. For a .zip that is the
    # archive's stamp, which raises if the code names an archive since moved.
    locator.get_source_stamp()
    try:
        locator.ensure_cache_path()
    except OSError as error:
        # The error names whatever failed, possibly a temporary file. The
        # location it is in is what a reader can act on, so the error raised
        # here names the location.
        raise OSError(error.errno, error.strerror, locator.get_cache_path()) from error
    # The location is asked for once more, as numba asks at every save. numba's
    # IPython locator asks IPython for it, and an IPython that can write neither
    # its cache directory nor its own gives a new temporary directory on every
    # call: numba makes numba_cache in one and saves in another, and dies there
    # past its own check. Without this question the check would pass the file
    # the same way. numba asks twice while making the directory, and the check
    # asks once more here; each answer is a temporary directory of its own.
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
                # The name alone is too long for the file system, not the
                # path. The error carries the name's length, so that the remedy
                # does not read the error as the location's and offer a
                # shorter path.
                raised.longest_file_name = longest_file_name
            raise raised from error


def is_a_cache_error(error):
    """Whether ``error`` is numba's error for a cache it cannot set up.

    That is one of: no locator takes the file, a location numba cannot use, or, from numba 0.68, a ``.zip`` member
    whose source numba cannot read from the archive.

    numba itself passes over a location on any ``OSError`` from making its directory or writing a file there: permission
    denied, a read-only file system, a path through a file, a component too long, a full disk. So any ``OSError``
    counts. So does the ``ValueError`` numba raises for an archive under a directory whose name holds ``.zip``: its
    ``.zip`` locator takes the file for that substring and then finds no ``.zip`` in the path. So does the ``KeyError``
    numba 0.68 and later raise when they read a ``.zip`` member's source from the archive, which they hash into the
    archive's stamp. It is zipfile's error, naming the member, for a name the archive does not hold. The name is
    missing for a module shipped as ``.pyc`` alone, and on Windows under numba 0.68.0 for any member, because that
    release builds the name with the platform's separator (numba issue 10889, fixed in 0.68.1). And so does the
    ``UnicodeEncodeError`` numba raises when it hashes the source's directory path as UTF-8 for the name of its
    directory under ``NUMBA_CACHE_DIR`` or the user's cache directory. numba raises it for a path that holds bytes
    which are not UTF-8, decoded with surrogates. A locator that names its cache directory by that hash raises it at
    decoration when numba reaches it, the locators before it having passed the file over, so no location takes the
    file. Without this clause, a package installed under such a directory, with its ``__pycache__`` unwritable, would
    die at import, while with caching off it imports.
    """
    if isinstance(error, OSError):
        return True
    if isinstance(error, UnicodeEncodeError):
        return error.reason == "surrogates not allowed"
    if isinstance(error, ValueError):
        return "No zip file found" in str(error)
    if isinstance(error, KeyError):
        return numba_version >= 68 and "There is no item named" in str(error)
    return isinstance(error, RuntimeError) and "no locator available" in str(error)


def _module_files():
    """The source files that numba's answer for numbox rests on, this module's own file first.

    numba's in-tree cache is a ``__pycache__`` beside each source, so the directories answer separately, and one can
    be writable while another is not. One module of each directory stands for the directory, whether or not that
    directory's modules cache anything. So one directory numba cannot cache in turns caching off for the whole
    package, even if every cached function's own directory is fine. The answer errs toward uncached, which is never
    wrong.

    numba finds a location for a module by its code's file. So a module that survives as ``.pyc`` alone, beside
    sourced ones, answers for itself, by the file it was compiled from, which its code keeps: the ``.py`` that is
    gone, if it was compiled in place, or a file in a tree elsewhere, on disk or not.

    The package is found by this module's ``__file__``, which is where it was imported from, an archive or a
    directory. This module's code can name a file elsewhere in the same way; numba looks that file up for the probe,
    and it is asked first. An archive shows no directories to walk, so its members are listed instead.
    """
    yield inspect.getfile(_cache_probe)
    package = os.path.dirname(os.path.dirname(__file__))
    if not os.path.isdir(package):
        yield from _archived_module_files(__file__)
        return
    walked = set()
    for directory, subdirectories, files in os.walk(package, followlinks=True):
        # Symlinks are followed, and a directory reached twice through links,
        # as in a cycle, is walked once.
        real = os.path.realpath(directory)
        if real in walked:
            subdirectories[:] = []
            continue
        walked.add(real)
        subdirectories[:] = sorted(name for name in subdirectories if name != "__pycache__")
        stems = sorted({name.rsplit(".", 1)[0] for name in files if name.endswith((".py", ".pyc"))})
        sources = [os.path.join(directory, stem + ".py") for stem in stems]
        on_disk = [source for source in sources if os.path.exists(source)]
        if on_disk:
            yield on_disk[0]
        for source in sources:
            if source not in on_disk:
                compiled_from = _sourceless_compiled_from(source[:-3] + ".pyc")
                if compiled_from is not None:
                    yield compiled_from


def _sourceless_compiled_from(pyc_path):
    """The file the ``.pyc`` at ``pyc_path``, alone beside sourced modules, was compiled from, or None.

    The import system runs such a ``.pyc`` through its sourceless loader, which keeps the compile-time file name on
    the code. The loader fails the module's import for a ``.pyc`` it cannot run, one with another interpreter's magic
    number or an unreadable one. So the loader's answer is read, and if it has none there is nothing to ask about.
    """
    loader = importlib.machinery.SourcelessFileLoader(os.path.basename(pyc_path)[:-4], pyc_path)
    try:
        return loader.get_code(loader.name).co_filename
    except Exception:
        return None


def _archived_module_files(own):
    """The files to ask about for numbox inside the ``.zip`` that holds ``own``, this module's ``__file__``.

    They are: one module per directory of numbox in the archive, given as the file its code names; and every
    ``.pyc`` whose code names a member the archive does not hold, which asks for itself. Such a ``.pyc`` stands alone
    with no ``.py`` beside it, or was compiled to name another path in the archive. Nothing is given for any other
    kind of archive.

    numba caches a ``.zip`` per directory of it, each in a location of its own under the user's cache directory, so
    the directories answer separately there too. A ``.pyc`` run from the archive keeps the file it was compiled from
    as its code's file. numba looks that file up for the module's functions, not the archive's own path for the
    module, so such a member asks by that file, whether it is there or gone. numba 0.68 and later read the member's
    source from the archive by the name its code gives. A ``.pyc`` alone has no source there, and a ``.pyc``
    compiled to name another path in the archive (``compileall -d``) has no source there either, whether or not its
    ``.py`` is beside it. So either kind of ``.pyc`` asks for itself, whichever member stands for its directory. The
    archive is the first part of the path
    that is named ``.zip`` and is an archive; a directory so named above it is no archive. Any other kind of archive
    has no cache location at all, and the probe's own file has already asked.
    """
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
    for name in names:
        directory, _, file = name.rpartition("/")
        if file.endswith(".pyc"):
            member = _compiled_from(zip_path, directory, file[:-4])
            if member is None:
                continue
            if not member.startswith(zip_path + os.sep):
                yield member
                continue
            named = member[len(zip_path) + 1:].replace(os.sep, "/")
            if named not in held:
                # The member the code names is not in the archive. A .pyc alone
                # names the .py it was compiled from, which is gone; a .pyc
                # compiled to name another path in the archive (compileall -d)
                # names a member that was never there, with or without the .py
                # beside it. numba 0.68 and later read the member's source from
                # the archive by that name and find none, so the .pyc is asked
                # for itself, whichever member stands for the .pyc's directory.
                # If a .py member stood for the directory alone, the .pyc would
                # never be asked about, and its module would die at decoration;
                # the same if a .pyc beside its .py were treated as naming the
                # archive.
                seen.add(directory)
                yield member
                continue
        else:
            member = os.path.join(zip_path, *name.split("/"))
        if directory not in seen:
            seen.add(directory)
            yield member


def _compiled_from(zip_path, directory, stem):
    """The file that is named by the code which ``zipimport`` runs for module ``stem`` of ``directory``, or None if it
    runs no code.

    zipimport decides which code runs. It runs the ``.pyc`` rather than the ``.py`` beside it, unless the ``.pyc`` has
    another interpreter's magic number, is stale against the ``.py``, or cannot be unmarshalled. In the first two
    cases it compiles the source instead; in the last, the module's import fails. So its answer, the code, is read
    here. The file that code names is the archive's own path for the module if zipimport compiled the ``.py``, or if
    the ``.pyc`` was compiled to name a member the archive holds; such a file stands for its directory as a ``.py``
    member does. Otherwise the file is one the archive does not hold, elsewhere, or a path in the archive that the
    ``.pyc`` was compiled to name, there or gone; it then asks for itself. If zipimport runs no code, because it
    cannot compile the source, cannot decompress the member, or fails for any other reason, there is nothing to ask
    about.
    """
    if stem == "__init__":
        directory, _, stem = directory.rpartition("/")
    importer = zipimport.zipimporter(os.path.join(zip_path, *directory.split("/")))
    try:
        return importer.get_code(stem).co_filename
    except Exception:
        return None


def _moved_through():
    """A clause for a remedy naming what moves numba's user cache directory on this platform; empty on Windows.

    numba's ``appdirs`` asks the system for the directory on Windows, where no variable moves it. It puts the
    directory under ``HOME`` on macOS. Elsewhere it takes ``XDG_CACHE_HOME``, or a directory under ``HOME`` if that
    is unset.
    """
    if sys.platform == "win32":
        return ""
    if sys.platform == "darwin":
        return ", through HOME"
    return ", through XDG_CACHE_HOME or HOME"


def _measured(name):
    """The length of ``name`` as the file system counts it, and the unit's name.

    Windows and macOS count a name in UTF-16 units; the file systems elsewhere count bytes. A surrogate counts as one
    unit: a name decoded with surrogateescape carries one for each byte that is not UTF-8, and Windows accepts one in
    a name. Encoding without ``surrogatepass`` would raise on it, and the remedy would die on a name that numba's
    error carried instead of answering."""
    if os.name == "nt" or sys.platform == "darwin":
        return len(name.encode("utf-16-le", "surrogatepass")) // 2, "characters"
    return len(os.fsencode(name)), "bytes"


def _name_limit(location):
    """The longest name a file can have where ``location`` is.

    It is 255 when the system cannot say, when the system says 0, or when ``location`` is None; a file system with no
    limit of its own reports 0. The count is in bytes outside Windows and macOS, and in UTF-16 units on those two.

    The question goes to the nearest directory above ``location`` that exists, because the location itself may not be
    there yet.
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
    """numba's cache locator classes in the order numba tries them.

    ``listed`` is ``NUMBA_CACHE_LOCATOR_CLASSES``. Each entry is a class of numba's
    caching module, by its name, or any class by its dotted path, resolved as numba resolves it. If ``listed`` is
    empty, the order is numba's own.

    An entry numba could not resolve is not a locator here. numba has refused such an entry already, before any
    location was tried, or raised importlib's error for it, or a ``TypeError`` for a relative module path. An entry
    that names no class, or a class with no ``from_function``, is not a locator here either: numba fails on it, for
    want of ``from_function``, at a decorated function that reaches the entry. In every case numba's own error
    is raised as it was.
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
                # importlib's errors, a relative path's TypeError and whatever
                # the module raises on import. numba raises these as its own
                # at decoration, and the check raises them before any remedy;
                # the remedy, reached only for a cache error, must not raise
                # the module's error instead.
                cls = None
        else:
            cls = getattr(caching, entry, None)
        if isinstance(cls, type) and callable(getattr(cls, "from_function", None)):
            classes.append(cls)
    return classes


def _existing_above(path):
    """The first path at or above ``path`` that exists.

    That is the directory under which the first missing directory on the way to ``path`` would be made; if the
    location is not there, that directory is the one that refused. Or it is a file in the way, under which no
    directory can be made; or it is the root, at the last. ``path`` is read as absolute: a relative path with no
    existing ancestor would walk up to an empty
    string, and the remedy would say that a file with no name is in the way."""
    path = os.path.abspath(path)
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def _ipython_numba_cache():
    """numba's cache location for a cell file that its IPython locator takes, or None.

    The location is ``numba_cache`` under IPython's cache directory, with no directory per file. It is None if
    IPython is not importable: numba's locator raises ``ImportError`` for that at decoration, which is no cache
    error, so the location is then none of numba's. It is None too if IPython raises for its directory. For an
    ``OSError`` numba's locator passes the file over, as it does for a location it cannot make, so numba took no
    location of IPython's. Anything else numba's locator would raise at decoration; the remedy, which asks the same
    question for the words of a warning, takes it as no location given."""
    try:
        try:
            from IPython.paths import get_ipython_cache_dir
        except ImportError:
            from IPython.utils.path import get_ipython_cache_dir
        return os.path.join(get_ipython_cache_dir(), "numba_cache")
    except Exception:
        return None


# What inspect says when it reads no source: for a cell that linecache does not hold, for a member whose archive
# holds no source, for a function whose line lies past the lines inspect read, and for a function of no module.
_INSPECTS_ERRORS = ("could not get source code", "source code not available", "lineno is out of bounds")


def _is_inspects(failure):
    """Whether ``failure`` is inspect's error for a source it could not read.

    Such an error is an ``OSError`` with no errno, no file name and one of inspect's own messages. A caller's error
    with no errno and no file name looks the same, but its message tells it apart."""
    return (isinstance(failure, OSError) and failure.errno is None and not getattr(failure, "filename", None)
            and str(failure) in _INSPECTS_ERRORS)


def _ipython_cache_is_temporary(first):
    """Whether IPython gives a new temporary directory as its cache directory on every call.

    IPython does that, with a warning, if it can write neither its cache directory nor its own. numba's IPython
    locator asks for the directory at every save, so numba then makes ``numba_cache`` in one directory and saves in
    another. ``first`` is an answer IPython already gave; one more is asked for here, and each is a directory IPython
    leaves behind. An IPython that gives a directory once and raises the next time gives no temporary directory, so
    a second answer of None does not count as a new directory."""
    if first is None:
        return False
    second = _ipython_numba_cache()
    return second is not None and second != first


def _taken_by_ipython(py_file):
    """Whether numba's IPython locator takes ``py_file`` by its name alone, whether or not the file is on disk.

    It takes a cell, ``<ipython-...>``, and a file in an ipykernel directory, and reads the function's source when it
    does."""
    return py_file.startswith("<ipython-") or os.path.basename(os.path.dirname(py_file)).startswith("ipykernel_")


def _archive_above(py_file):
    """The archive that ``py_file`` is a member of, and the member's name in it, as zipimport finds them; or None.

    The first part of the path that is a file, whatever its name, must be a zip archive, and the rest of the path is
    the member. The result is None if no part of the path is a file, or if the first part that is one is no archive.

    The path is split by the platform's separator, ``os.sep``, through the pure path classes. The concrete
    ``pathlib.Path`` would build a path of the platform ``os.name`` names, which a test can set to Windows's, and
    Python before 3.12 refuses to build such a path on Linux; the remedy for a member that numba names with Windows's
    separator would die on that.
    """
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
    """Whether a locator of numba's IPython family that numba reaches takes ``py_file`` by its name.

    Such a locator reads the function's source when it takes the file. numba reaches the locators in its own order, or
    in the order ``NUMBA_CACHE_LOCATOR_CLASSES`` lists them. It reaches none listed after its ``.zip`` locator for a
    path with ``.zip`` in it, because that locator takes the file without trying its location. Nor does it reach any
    listed after a locator that takes the file, one of its own or a class outside its families, asked with a probe as
    numba asks it; the in-tree locator, for one, takes a cell file on disk whose ``__pycache__`` it can write. A
    locator that passes the file over leaves the ones after it reached."""
    from numba import config
    from numba.core import caching
    if not _taken_by_ipython(py_file):
        return False
    for cls in _locators(config.CACHE_LOCATOR_CLASSES):
        if issubclass(cls, caching.IPythonCacheLocator):
            return True
        if issubclass(cls, caching.ZipCacheLocator) and ".zip" in py_file:
            return False
        if _takes(cls, py_file):
            # A locator before IPython's took the file, so the IPython locator
            # is not reached. Without this question the remedy would read a
            # caller's error that names some other numba_cache as the IPython
            # locator's error, for a cell file on disk whose __pycache__ the
            # in-tree locator writes.
            return False
    return False


def _numbas_archive_holds(py_file, name):
    """Whether the archive that numba's ``.zip`` locator takes for ``py_file`` holds a member named ``name``.

    That archive is the first part of the path ending in ``.zip``; the answer is False if that is no archive."""
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
    """The ``.pyc`` member whose code names ``py_file``, and whether a ``.py`` stands beside it; or None.

    The archive searched is the one numba's ``.zip`` locator takes for ``py_file``, and the code is what zipimport
    runs for the member. The result is None if no member's code names ``py_file``. This finds a ``.pyc`` compiled to
    name another path in the archive (``compileall -d``): numba 0.68 and later read the source by that name and find
    none under it, while the archive holds the source beside the ``.pyc``."""
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
    """Why the member of the archive above ``py_file`` is not UTF-8: the byte and the reason; or None.

    zipimport decodes a member's source as UTF-8, and numba's IPython locator reads the source as zipimport decodes
    it. The result is None if no archive above the file holds the member, or if the member decodes."""
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
    """The member zipfile's ``KeyError`` names, "There is no item named 'x' in the archive", or its whole message.

    zipfile quotes the name as Python does, so a backslash in it is doubled there and has to be read back; numba
    0.68.0 builds a member's name with backslashes on Windows.
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
    """The remedy for ``failure``, numba's error for a function whose file is ``py_file``, ending in ``silence``.

    The remedy is one sentence. It names what numba tried, why that failed and what moves it, and ends in
    ``silence``, the clause that says how to turn caching off. ``py_file`` can be a string, bytes or a path object,
    and so can the file name the error carries; both are read as strings. ``package`` is the package the remedy
    tells the reader to install again, at a shorter path or with its source files on disk. A package built on numbox
    that asks the question for its own files, with ``check_cache_location`` and ``is_a_cache_error``, passes its own
    name, just as it passes ``check_cache_location`` a ``longest_file_name`` for its own functions;
    ``LONGEST_CACHE_FILE_NAME`` is numbox's.

    Most of the remedy turns on which locator took the file. numba caches a source file on disk in one of three
    places: a directory under ``NUMBA_CACHE_DIR`` if that is set, else the ``__pycache__`` beside the source, else a
    directory under the user's cache directory. The first and the third are under a cache directory named after the
    source's directory, by its name and a hash of its path. The three can nest, and ``NUMBA_CACHE_DIR`` set to the
    user's cache directory makes one path of two. So the remedy matches the whole path the error names against the
    location each locator numba reaches would give the file, in the order numba tries the locators, and reads the
    error as the first matching locator's. numba's own errors mean that the locators before that one passed the
    file over; an error a caller passes on, naming a location numba did not take, need not mean that. numba reaches
    no locator listed after its ``.zip`` locator for a path with ``.zip`` in it, because that locator takes the file
    without trying its location, so a location that one of those would give is none of numba's.

    ``NUMBA_CACHE_LOCATOR_CLASSES`` decides the order. Each entry is one of three
    things: a class of numba's caching module; a subclass of such a class, by its dotted path, which takes what its
    parent takes, caches where its parent does, and is told as its parent is; or a class outside numba's families,
    with a ``from_function`` and a place of its own, which takes whatever its ``from_function`` takes when asked with
    a probe, as numba asks it, and caches in a place that only the class can name. An error naming a location that
    no locator of numba's gives the file is told that location as the error names it; the location is then that
    class's. An error naming no file is told the location numba took as the outside class's own, with no path to
    name, among the places of numba's locators that are reached. numba's IPython locator takes a file on disk only
    in an ipykernel directory, and its ``.zip`` locator only a path with ``.zip`` in it, so listing either ahead of
    the rest changes nothing for any other file. The ``.zip`` locator listed first takes such a path before any
    other locator does. It caches the file under the user's cache directory if a part of the path ends in ``.zip``,
    an archive or a directory; if no part does, it finds no archive and raises, and the remedy asks for a locator
    for a file on disk listed before it. The user-provided locator counts only with ``NUMBA_CACHE_DIR`` set, because
    it takes nothing without. Listed after some of the locators for a file on disk, the ``.zip`` locator finds no
    archive and raises once those have passed the file over, and numba tries none after it; the remedy then asks
    for one of their locations to be made writable, or for such a locator to be listed before it. A list with no
    locator that takes the file is told so, and so is a list whose only locator for the file is the user-provided
    one with ``NUMBA_CACHE_DIR`` unset, which is told to set the variable. Both are told the same for an error a
    caller passes on under such a list, as for numba's no-locator error, because numba took no location under it.

    For a source file on disk, ``NUMBA_CACHE_DIR`` is offered, because that is the only kind of source numba reads
    the variable for. If the variable is set and numba passed it over, the remedy names it and asks for a writable
    directory at a short path. numba passes over a directory too deep to make its own directory in, as it does an
    unwritable one. If ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves the user-provided locator out, numba never
    reads the variable, and the remedy asks for the locations the list names to be made writable. If the location
    numba took refuses a file for another reason, a full disk or permissions changed since numba's own check, the
    remedy names the location and the reason and asks for room or a writable directory there. It offers
    ``NUMBA_CACHE_DIR`` as the alternative if numba tries the variable before that location's locator, or the
    variable set to another directory if the location is the variable's own. numba passes over a location it cannot
    make or write in, so only its no-locator error means the variable was passed over. A location that is none of
    numba's is named as the error names it. If numba tries ``NUMBA_CACHE_DIR`` before any other locator, the
    variable is named as set, to be made a writable directory at a short path, or, unset, asked for at a short path
    for a path too long and as a writable directory otherwise; nothing is said of numba passing it over, because
    such a location cannot show that.

    For a path too long for the file system, the error names the location, and the remedy fits the one it is: a
    shorter ``NUMBA_CACHE_DIR`` for the first of the three; the package at a shorter path for the second; the user's
    cache directory at a shorter path for the third, through ``XDG_CACHE_HOME`` or ``HOME``, ``HOME`` alone on macOS
    and nothing on Windows. For either of the last two the remedy also offers ``NUMBA_CACHE_DIR`` set to a short
    path. It makes that offer if numba tries the variable before the locator that took the location, or before the
    last of the locators that give the file that one path, as the user-wide and ``.zip`` locators do. If the
    variable is set and numba passed it over, the remedy names the variable and asks for it to be made a writable
    directory at a short path.

    A name can be too long for the file system. The directory numba names after the source's directory, under
    ``NUMBA_CACHE_DIR`` or the user's cache directory, is that directory's name plus forty-one bytes. A name longer
    than the file system holding the location allows can be made at no path, and is refused whatever the errno,
    ``EINVAL`` on Windows. The limit is asked of the system, 255 if it cannot say, in bytes, or in the UTF-16 units
    Windows and macOS count a name in. Three errors are told the name and its length, and to put the file in a
    directory of a shorter name. The first is an error naming such a location. The second is numba's no-locator
    error for a file on disk, or its ``.zip`` locator's error once the locators before it passed the file over, if
    every locator numba tried that names its cache directory by that hash found the name too long; the variable's
    locator measures by the limit of the directory the variable is set to, or by 255 if it is unset, and under that
    locator alone, unset, the remedy also asks for the variable to be set. The third is a frozen application's
    error. For a file on disk the remedy adds the ``__pycache__`` beside the source, which caches under no such name:
    made writable if numba reaches the in-tree locator; otherwise, under a list, that locator listed, before the
    ``.zip`` locator or the outside class that took the file if the list puts it after one.

    The files numba writes for a function are named after the source's name and the function's qualified name. A
    cache file numba could not make or open in the location it took, because its name is longer than that file
    system allows, is told the name and its length, which no shorter path cures. It is told a shorter name for the
    function or its file, which is the package's to give. If a file system in use takes the name, the remedy also
    offers ``NUMBA_CACHE_DIR`` pointed at a directory on such a file system: the variable as set, if its own location
    took the file, and otherwise provided numba tries the variable before the locator that took the file, as the
    variable is offered at a short path for a path too long. A file system in use takes the name if the name's base
    fits in 255, the common limit, less the 29 bytes numba adds for the longest name it writes for the function: a
    data file's, with three digits for its number, which numba counts from one without bound, written through a
    temporary name; a wider number in the error's name counts at its own width. The check's error for a name as long
    as the package's functions get, on a file system that takes no name that long, is told the length. It is offered
    ``NUMBA_CACHE_DIR`` pointed at a directory on a file system that takes such a name, on the same terms, whatever
    the name's length. Nothing is offered for a ``.zip`` member, which the variable does nothing for.

    A path can hold bytes that are not UTF-8. numba hashes the source's directory path as UTF-8 for the name of its
    directory, and raises at decoration for such bytes when a locator that names its cache directory by that hash
    reaches the file. For such a path the remedy names no location under ``NUMBA_CACHE_DIR`` or the user's cache
    directory. A ``.zip`` under such a path, whose archive path numba's ``.zip`` locator hashes the same way for its
    directory under the user's cache directory, is told to put the archive under a UTF-8 path; a frozen application
    under one, whose path the user-wide locator hashes the same way, is told to put the application under a UTF-8
    path. A file on disk under such a path is also told the ``__pycache__`` beside it, because the in-tree locator
    hashes no path for that directory. If the list leaves the in-tree locator out, or puts it after a locator that
    hashes other than the variable's, the remedy says to list the in-tree locator before the first locator that
    hashes. If only the variable's locator hashes before the in-tree locator, the remedy says to unset
    ``NUMBA_CACHE_DIR``. If the in-tree locator came before every locator that hashes and passed the directory over,
    the remedy says to make the ``__pycache__`` writable. With any of these, the remedy also says to make the
    ``__pycache__`` writable if it cannot be written.

    numba's IPython locator caches a cell file in ``numba_cache`` under IPython's cache directory, with no directory per
    file, and passes the location over if it cannot make or write in it. No other locator of numba's takes a cell, so a
    cell then gets numba's no-locator error. It is told that location, to be made writable or put at a shorter path, or
    IPython's cache directory to be given a home if IPython raised for it; a class outside numba's families listed with
    the locator is named as having passed the cell over too. That location too long is told as IPython's cache directory
    at a shorter path, for a cell file on disk and for a ``.zip`` member in an ipykernel directory alike, if the IPython
    locator is listed before the ``.zip`` one; no ``NUMBA_CACHE_DIR`` is offered for a source not on disk. IPython is
    asked for its directory only if a locator of that family is listed and takes the file, never for any other file, a
    plain ``.zip`` member among them. The reason is that IPython warns when asked under a home it cannot write, and
    leaves a temporary directory behind. numba imports IPython when it makes the location and catches only ``OSError``
    there. So if IPython is not importable, numba raises ``ImportError`` at decoration for a file that the IPython
    locator takes, which is no cache error. An error a caller passes on naming that location is told as none of numba's
    if another locator numba reaches has a place for the file; if none has, the list is told, as for numba's no-locator
    error. If IPython raises ``OSError`` for its directory, numba passes the locator over, as it does a location it
    cannot make, and the remedy, which asks the same question, has no location of IPython's to match the error against.

    A file in an ipykernel directory that numba's IPython locator takes, a cell file on disk or a ``.zip`` member, can
    meet an IPython that gives a new temporary directory as its cache directory on every call. IPython does that if it
    can write neither that directory nor its own. An error naming a file is then told so, with ``IPYTHONDIR`` or
    ``XDG_CACHE_HOME`` at a writable path as the remedy; the error numba's first save raises there is
    ``FileNotFoundError``, naming a file in the ``numba_cache`` directory it could not save in, because numba makes
    ``numba_cache`` in one such directory and saves in another. An error of another errno naming a ``numba_cache`` is a
    caller's, and is told as the location it names. An error naming no file is told the locations numba could have
    taken, with its reason.

    An error can name no file, carry an empty string or empty bytes as its name, or carry a name of another kind, an
    integer file descriptor, which counts as none. A caller can build such an error, and numba's first save raises
    one when a write fails, as on a full disk. So does inspect when, for numba's IPython locator, it reads no source
    in a cell file on disk, one that cannot be read or, before Python 3.13, an empty one; that error has no errno
    either, because no call into the file system raised it. For a file on disk that a locator of IPython's family
    takes, numba having reached it, that error is told as inspect's, the locator having tried no location: make the
    file readable, with the source in it. For a cell, whose source numba reads from linecache while IPython holds
    it to run the cell, the remedy says that numba caches a cell only while IPython holds its source. For a ``.zip``
    member whose archive holds no source, a ``.pyc`` alone, the remedy is the archive's: ship its source files.
    Every other ``OSError`` that ``check_cache_location`` raises names a file. An error with no name, or an empty
    one, is told the locations numba could have taken, in numba's order and each once: for a path too long, to put
    whichever of them is too long at a shorter path; for a file refused for another reason, such as a full disk,
    the reason, and room or a writable directory there. A path object is never empty, because pathlib reads an
    empty string as the working directory, ``.``, which the remedy names as a location that is none of numba's.

    numba caches a ``.zip`` and a frozen application under the user's cache directory, and the remedy for either is
    that location made writable or given room, with the reason the error gives, a full disk or permissions. The
    ``.zip``'s error names the location, a directory of numba's under the user's cache directory, or a file numba
    writes in it, as the first save's error does, and the directory is told. An error naming a location that is none
    of numba's, under the list and in its order, is told that location as the error names it, as for a source on
    disk. Nothing is said of what moves the user's cache directory, or of where numba caches the file, which that
    location is none of. The frozen application's error is numba's no-locator one, which gives no reason, because
    numba passed the location over on its own error, unwritable or too deep alike; so the user's cache directory is
    named as one numba could not use, to be made writable or put at a shorter path. If the name numba gives the
    directory after the source's directory is too long for the file system, that is told instead; so is a cache
    file's name too long for the file system, in that location or in IPython's ``numba_cache`` for a member in an
    ipykernel directory, which no shorter path cures. A ``.zip``'s location too long for the file system is the
    user's cache directory's doing, because the names numba makes are bounded, and the remedy is that directory at a
    shorter path, through what moves it on the platform. A frozen application's path with ``.zip`` in it and no part
    ending in ``.zip`` gets the ``.zip`` locator's error instead, if the list puts that locator before the user-wide
    one, or once the user-wide one passed its location over; the remedy is the user-wide locator listed before the
    ``.zip`` one, or the user-wide location made writable or put at a shorter path. An error that names no file, a
    caller's or the first save's on a full disk, is told the locations numba could have taken, as for a source on
    disk, in order and each once. It is told the reason, or, for a path too long, to put whichever is too long at a
    shorter path. For such an error, for a member in an ipykernel directory where a locator of IPython's family is
    reached, the user's cache directory is not named as where numba caches the file. IPython's ``numba_cache`` is
    among the possible locations whenever IPython gives its directory.

    numba caches a ``.zip`` through its ``.zip`` locator alone, so a list that leaves that locator out gives numba no
    locator for a source in a ``.zip``, and the remedy is to list it. That holds for a member of an archive that
    exists; a directory named ``.zip`` above a source that is nowhere is no archive. The same holds for a notebook
    cell and numba's IPython locator, and for a frozen application and numba's user-wide locator, which alone takes
    one. Either is told for an error a caller passes on under such a list, as for numba's no-locator error, because
    numba took no location. For any other kind of archive, or a module without its source, the remedy is the source
    files on disk or a ``.zip`` holding them, each with the locator it needs listed if ``NUMBA_CACHE_LOCATOR_CLASSES``
    leaves it out. The source files on disk need one of the three locators for a file on disk, and
    ``NUMBA_CACHE_DIR`` set if the user-provided locator is the one listed; the ``.zip`` needs the ``.zip`` locator.
    A source that is neither in a ``.zip`` nor a frozen application's, which no locator takes, is told the same for
    any error.

    From numba 0.68 the archive's stamp hashes the member's source, which numba reads from the archive by the name
    the module's code gives under the archive's path. zipfile raises ``KeyError`` for a name the archive does not
    hold. For a module shipped as ``.pyc`` alone the remedy names the member and asks for the source files, on disk
    or in a ``.zip``; for a ``.pyc`` compiled to name another path in the archive (``compileall -d``), whose source
    the archive holds beside it, the remedy is to compile the ``.pyc`` to name its own path, or to leave it out. On
    Windows under numba 0.68.0 every member gets that error, because that release builds the name with the
    platform's separator (numba issue 10889, fixed for 0.68.1). If the archive holds the member's source under that
    name with forward slashes, the name a release with the fix reads, the remedy is the install on disk, or a numba
    release carrying the fix; otherwise it is the remedy for any other member. numba takes the first part of the
    path ending in ``.zip`` for the archive, a directory so named above the file included, and from 0.68 opens it to
    read the source. The ``IsADirectoryError`` that gives is told the directory: put the archive under a path with
    no such directory above it. A file on disk under such a directory, which the ``.zip`` locator takes once the
    locators before it passed the file over, is told to make one of their locations writable, or, with the ``.zip``
    locator listed first, to list a locator for a file on disk before it; or to put the file elsewhere.
    """
    from numba import config
    from numba.core import caching
    from numba.core.caching import _CacheLocator
    # check_cache_location takes any path-like, bytes included, and so does
    # cache_remedy. The remedy reads a str; os.fspath would leave bytes as
    # bytes.
    py_file = os.fsdecode(py_file)
    # The error names the refused file in the form the raising call was given
    # it, bytes for a call given bytes, or in the form a caller built the
    # error with, a path object included. The remedy reads the name as str, as it reads the
    # source's. An empty str or bytes is no name. A path object built from an
    # empty string names the working directory, ".", as pathlib reads it. A
    # name of another kind, an integer file descriptor as os.stat(fd) raises
    # it, is no name.
    filename = getattr(failure, "filename", None)
    filename = os.fsdecode(filename) if filename and isinstance(filename, (str, bytes, os.PathLike)) else None
    listed = config.CACHE_LOCATOR_CLASSES
    user_provided = caching.UserProvidedCacheLocator
    in_tree = caching.InTreeCacheLocator
    user_wide = caching.UserWideCacheLocator
    for_ipython = caching.IPythonCacheLocator
    for_a_zip = caching.ZipCacheLocator

    def one_of(cls, base):
        # Whether cls is base or a subclass of it. A subclass listed by its
        # dotted path takes what its parent takes and caches where its parent
        # does, as numba's InTreeCacheLocatorFsAgnostic does, so each of
        # numba's locators stands for its whole family.
        return issubclass(cls, base)

    def of_numbas(cls):
        # Whether cls belongs to one of numba's locator families, whose places
        # the remedy knows. A class outside them has a from_function and a
        # place of its own, and caches in a place that only the class can name.
        return any(one_of(cls, base) for base in (user_provided, in_tree, user_wide, for_ipython, for_a_zip))

    def could_have_taken(locations):
        # An error that names no file cannot tell which location numba took,
        # so the locations numba could have taken, the values of locations in
        # numba's order, are listed, each once: on disk, NUMBA_CACHE_DIR set
        # to the user's cache directory makes one path of two. A class outside
        # numba's families has a place the remedy cannot name, so the list
        # calls that place the class's own.
        known = list(dict.fromkeys(
            location if location else f"{cls.__name__}'s own"
            for cls, location in locations.items() if location or not of_numbas(cls)))
        if len(known) > 1:
            return f"the location numba took, one of {', '.join(known[:-1])} or {known[-1]}"
        return f"the location numba took, {known[0]}"

    def listed_without(base):
        # Whether NUMBA_CACHE_LOCATOR_CLASSES is set and names no class of
        # base's family.
        return bool(listed) and not any(one_of(cls, base) for cls in _locators(listed))

    try:
        subpath = _CacheLocator.get_suitable_cache_subpath(py_file)
    except UnicodeEncodeError:
        # numba hashes the source's directory path as UTF-8 for the name of
        # its directory, and raises at decoration for bytes that are not
        # UTF-8, when a locator that names its cache directory by that hash reaches the
        # file. The remedy then has no such location to name.
        subpath = None

    def too_long_a_name(location):
        # numba names the cache directory it makes for the file, under
        # NUMBA_CACHE_DIR or the user's cache directory, with the source's
        # directory name and a hash of the source's directory path. A name
        # longer than the file system that holds the location allows can be
        # made at no path, so no shorter path cures it. The result is the
        # clause that says so, or None if the name fits or numba has no name
        # to give.
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
        # The check's error for a name as long as the package's functions get,
        # on a file system that takes no name that long: the error carries the
        # name's length, and the result is the clause that says so, or None
        # for any other error. Without it the remedy would read the error as
        # the location's error and offer a shorter path, which does nothing
        # for a name that is too long.
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
        # numba names the files it caches a function in after the source's
        # name and the function's qualified name. A name longer than the file
        # system that holds the location allows can be made at no path, so no
        # shorter path cures it; only a shorter name does. The result is the
        # clause that says so for the file the error names in the location,
        # and whether any file system in use takes the name at all; or None
        # if the error names no such file, or the file's name fits. The limit
        # of ext4, NTFS and APFS alike is 255, in the units each counts, and
        # the name is measured against the longest name numba writes for the
        # function: the index, ".nbi", and a data file per signature,
        # ".<n>.nbc", numbered from one without bound, each written through a
        # temporary name 21 longer, ".tmp." and sixteen hex digits. The data
        # file's number takes the digits reserved for it, or the width of the
        # number the error names if that is wider. Without this clause the
        # remedy would ask for a shorter path for the package, the user's
        # cache directory, IPython's cache directory or the variable, none of
        # which shortens a name. A file system that takes the name is offered
        # only if one in use does, counting the temporary name and a data file
        # name with three digits for its number, or the number the error names
        # if that is wider; and the shorter name is always named.
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
        # What a locator takes by the file's name alone, before any location
        # is tried. numba's IPython locator takes a cell, or a file in an
        # ipykernel directory, on disk or not. Its .zip locator takes a path
        # with .zip in it; it caches the file if a part of the path ends in
        # .zip, an archive or a directory, and raises if none does. Its
        # user-wide locator takes a frozen application's file as well as any
        # file on disk. The rest take any file on disk. A class outside
        # numba's families, with a from_function and a place of its own,
        # takes what its from_function takes when asked with a probe, as
        # numba asks it, and caches in a place that only the class can name.
        # Without asking, the remedy would take such a class for one that
        # takes every file, and would report a file the class passed over as
        # the class's own when numba went on to the next locator listed.
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
        # Whether the error names a path above the file: the archive that a
        # member's code names, or a directory on the way. Both are read as
        # absolute: a relative file name matches no absolute name, and the
        # remedy would otherwise tell a member's archive that has since moved
        # as a location that refused the file. The working directory does not
        # count: a path object built from an empty string names it, and the
        # remedy tells that path as a location that is none of numba's;
        # otherwise the working directory would stand above a file under it
        # as a moved archive.
        above = os.path.abspath(filename) if filename else ""
        return bool(above) and above != os.getcwd() and os.path.abspath(py_file).startswith(above + os.sep)

    def names_another_location():
        # Whether the error names a location of one of numba's locators for a
        # file on disk other than IPython's, or a file in such a location:
        # NUMBA_CACHE_DIR's, the __pycache__ beside the source, or the user's
        # cache directory. A location that is none of those, for example a
        # temporary directory IPython gave, is taken as the IPython locator's.
        if not filename:
            return False
        named = os.path.abspath(filename)
        if named == os.path.abspath(py_file):
            # The source itself, which numba reads. If the source cannot be
            # read, the remedy says to make it readable, whatever directory
            # IPython gives as its cache directory.
            return True
        others = [os.path.abspath(config.CACHE_DIR)] if config.CACHE_DIR else []
        others.append(os.path.abspath(os.path.join(os.path.dirname(py_file), "__pycache__")))
        others.append(os.path.abspath(AppDirs(appname="numba", appauthor=False).user_cache_dir))
        return any(named == other or named.startswith(other + os.sep) for other in others)

    asked = []

    def ipython_numba_cache():
        # IPython's cache directory, asked for once: under a home directory
        # that IPython cannot write, IPython makes a temporary directory per
        # call and warns.
        if not asked:
            asked.append(_ipython_numba_cache())
        return asked[0]

    def names_a_numba_cache():
        # Whether the error names IPython's location for numba, a numba_cache
        # directory or a file in one, as not there. Under an IPython that gives
        # a temporary directory per call, the error numba's first save raises
        # is FileNotFoundError, naming a file in the numba_cache directory it
        # could not save in, because that directory was made in a different
        # temporary directory; the check's error names the directory. A caller's error that
        # names another location shows nothing of IPython's locator. Without
        # this test the remedy would offer IPYTHONDIR for a cell on disk that
        # the in-tree locator took, and for a member under such an IPython on
        # any errno, a refused directory of the caller's own included.
        return (
            bool(filename) and failure.errno == errno.ENOENT
            and "numba_cache" in os.path.abspath(filename).split(os.sep))

    if isinstance(failure, OSError) and names_a_numba_cache() and not names_a_path_above() and (
            _ipython_locator_reads(py_file)) and (
            not a_file_on_disk or not names_another_location()
    ) and _ipython_cache_is_temporary(ipython_numba_cache()):
        # numba's IPython locator is reached first for a member in an ipykernel
        # directory. For a cell file on disk in such a directory it is reached
        # once the locators before it passed the file over. It asks IPython
        # for its cache directory at every save, and an IPython that can write
        # neither that directory nor its own gives a new temporary directory
        # on every call: numba makes numba_cache in one and saves in another,
        # and dies in the second. Without this case the remedy would tell the
        # directory the error names, a temporary one, as refusing a file, and
        # for a file on disk would offer NUMBA_CACHE_DIR beside the source.
        #
        # Three errors in particular stay out. An error that names no file is
        # told the locations numba could have taken, with its reason, as the
        # docstring says, whatever directory IPython gives; for a file on
        # disk such an error also shows nothing of which locator took the
        # file. The error numba's first save raises here names a file in the
        # numba_cache directory it could not save in, whereas a caller's
        # error with a reason and no name, a full disk's, keeps its reason
        # instead of being told IPYTHONDIR. inspect's error, with no errno
        # and no name, came before the locator tried any location, and is
        # told below, as inspect's for a cell or a file on disk and as the
        # archive's for a member shipped as .pyc, whatever directory IPython
        # gives; IPython's directory is not to blame for a cell whose source
        # linecache does not hold, or for such a member. An error naming a
        # path above the file, the archive a
        # member's code names when that archive has since moved, is the
        # stamp's error, and is told below; the temporary directory's error
        # names a numba_cache under the temporary directory.
        #
        # IPython reads XDG_CACHE_HOME for its cache directory on every POSIX
        # system from IPython 8, macOS included, which IPython 7 left out, so
        # the message says POSIX, not Linux, and does not steer a macOS reader
        # off the variable.
        return (
            "numba caches this file in numba_cache under IPython's cache directory, and IPython, able to write neither "
            "that directory nor its own, gives a new temporary directory on every call, so numba makes numba_cache "
            "in one and saves in another: set IPYTHONDIR, or XDG_CACHE_HOME on a POSIX system, to a writable "
            "directory; or "
            f"{silence}"
        )

    if a_file_on_disk:
        def reads_the_variable(cls):
            return one_of(cls, user_provided)

        def caches_beside_the_source(cls):
            return one_of(cls, in_tree)

        def caches_under_the_user_cache_dir(cls):
            return one_of(cls, user_wide) or one_of(cls, for_a_zip)

        order = [cls for cls in _locators(listed) if may_take(cls)]
        # numba's .zip locator takes a path with .zip in it without trying
        # its location, and a class outside numba's families takes the file
        # when asked, which stops the order too; no locator listed after
        # either is reached. Without this cut the remedy would read the
        # locators after such a class as reached, and would tell an error
        # that names IPython's location as one where numba caches the file.
        zip_at = next((at for at, cls in enumerate(order) if one_of(cls, for_a_zip)), None)
        stops_at = next((at for at, cls in enumerate(order) if one_of(cls, for_a_zip) or not of_numbas(cls)), None)
        reached = order if stops_at is None else order[:stops_at + 1]
        # A .zip locator listed after such a class is not reached, so under
        # that list the errors it would raise are a caller's. Without this the
        # remedy would tell those errors as the .zip locator's own, on the
        # ground that the locators before it passed the file over, when one
        # of them had taken the file.
        zip_at = zip_at if zip_at == stops_at else None
        variable_at = next((at for at, cls in enumerate(order) if reads_the_variable(cls)), None)
        cache_dir_read = variable_at is not None

        def instead(taken, asks="a short path", shared=None, made="a writable directory at a short path"):
            # NUMBA_CACHE_DIR as the alternative to the location numba took,
            # offered if numba tries the variable before that location's
            # locator. If the variable is set, numba passed it over because
            # the directory was unwritable or too deep, and the made argument
            # says what the reader is asked to make of the directory. If the
            # variable is unset, numba would take it, and the asks argument
            # says what the reader is asked to set the variable to. The taken
            # argument is None for a location that is none of numba's; that
            # counts as the first locator in the order other than the
            # variable's own. Such a location shows nothing of what numba did
            # with the variable, so a set variable is named without the claim
            # that numba passed it over. The shared argument is the last
            # locator numba reaches whose place the error names too. The
            # user-wide and .zip locators give the file one path under the
            # user's cache directory, and the error cannot show which of them
            # took the file; so the variable is offered if numba tries it
            # before the last of them, without the claim. Reading the first of
            # those two locators as the one that took the file would leave the
            # variable out when the variable sits between them.
            if subpath is None:
                # numba names its directory under NUMBA_CACHE_DIR by hashing
                # the source's directory path as UTF-8, and raises for a
                # path that is not UTF-8. If the remedy offered the variable
                # here, the warning would turn into an import error: a reader
                # who set the variable would make numba hash the path and
                # raise at decoration.
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
            # The in-tree locator caches beside the source, in its __pycache__,
            # under no directory named after the source's directory, so it is
            # the one locator of numba's for a file on disk that a directory
            # name too long leaves usable. The remedy offers the __pycache__
            # made writable if numba reaches the in-tree locator and the
            # locator passed the file over. It offers the in-tree locator
            # listed if NUMBA_CACHE_LOCATOR_CLASSES leaves it out. It offers
            # the in-tree locator listed before the .zip one if the .zip
            # locator, listed first, took a path with .zip in it and left the
            # in-tree locator unreached. Offering the writable __pycache__
            # under a list that never reaches the in-tree locator would cure
            # nothing.
            if any(caches_beside_the_source(cls) for cls in reached):
                beside = os.path.abspath(os.path.join(os.path.dirname(py_file), "__pycache__"))
                return (
                    f", or make the __pycache__ beside the source, {beside}, writable, where InTreeCacheLocator "
                    "caches it under no such name"
                )
            if listed:
                # If the in-tree locator is listed after the locator that took
                # the file, the .zip one or a class outside numba's families,
                # the in-tree locator has to go before that locator. The
                # remedy names the locator to go before, which is not always
                # the .zip one.
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
            # Where the class places the file. The .zip locator places it under
            # the user's cache directory, like the user-wide one. The result is
            # None for a class that is not numba's, whose place is its own; for
            # the user-provided locator without NUMBA_CACHE_DIR, which takes
            # nothing without the variable; and for the IPython locator when
            # IPython is not importable, or when IPython raises OSError for
            # its directory, and numba then passes the locator over.
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
            # inspect's error, which no call into the file system raised, so
            # it has no errno and no file name. numba's IPython locator reads
            # the function's source, from the file itself for a file on disk,
            # and inspect raises if it reads no source there, in a file it
            # cannot read or, before Python 3.13, an empty one. That happens
            # before the locator tries its location, the locators numba tried
            # before it having passed the file over, so there is no location
            # numba took to tell as refusing the file.
            return (
                f"numba's IPython locator reads the function's source from its file before trying its location, and "
                f"read none in {py_file} ({failure}): make the file readable, with the source in it; or {silence}"
            )
        if isinstance(failure, OSError) and filename and os.path.abspath(filename) == os.path.abspath(py_file):
            # numba reads the source itself: its IPython locator reads the
            # function's source, and from 0.68 every locator reads the file
            # for the stamp, which hashes the source. A file that cannot be
            # read is told as such, not as the location numba took refusing
            # the file.
            reason = failure.strerror or failure
            return (
                f"numba reads the function's source from its file, {py_file}, which cannot be read ({reason}): make "
                f"it readable; or {silence}"
            )
        # Under a list with no locator that takes the file, or with no locator
        # that has a place for it, numba took no location; a list of the
        # user-provided locator alone with NUMBA_CACHE_DIR unset is one such
        # list. The list itself is then the remedy, below, for an error a
        # caller passes on, just as it is for numba's no-locator error. A
        # class outside numba's families has a place of its own, which an
        # error can name.
        if isinstance(failure, OSError) and (any(locations.values()) or not all(of_numbas(cls) for cls in reached)):
            # The error names the location numba took, or a file numba writes
            # in that location. numba's own check passed the location because
            # its temporary file fit where numba's cache files would not. Or
            # the file the error names, which the package's check made, was
            # refused later, by a full disk or by permissions changed since.
            # The locator that took the location decides the remedy: the first in
            # numba's order whose place the error names. Paths are matched
            # whole, because the places can nest, as they do for an install
            # under either cache directory, or for NUMBA_CACHE_DIR set above
            # the user's cache directory; and in order, because
            # NUMBA_CACHE_DIR set to the user's cache directory makes one path
            # of two.
            if filename:
                named = os.path.abspath(filename)
                named = {named, os.path.dirname(named)}
            else:
                # A caller can pass on an error that names nothing, and numba's
                # first save raises one when a write fails, as on a full disk.
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
                # The variable, pointed at a directory on a file system that
                # takes the name. It is offered if numba tries the variable
                # before the locator that took the file and a file system in
                # use takes the name, as the variable is offered at a short
                # path for a path too long. The variable's own location took
                # the file only because the variable is set, so the variable
                # is named.
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
                    # The variable's location, which numba's own check passed
                    # and the first save's longer name overflows. numba keeps
                    # taking the variable as it is, so the user's cache
                    # directory at a shorter path would move nothing numba
                    # reads, even if the variable is set to that directory.
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
                # numba's .zip locator takes a file on disk because the path
                # contains ".zip", after the locators before it have passed
                # the file over. It takes the first part of the path ending in
                # .zip for the archive, a directory so named included, and
                # from 0.68 opens that archive to read the source, which a
                # directory refuses. That directory is no cache location, so
                # the reader is not told to make room in it; the remedy for a
                # member knows the same case. Under a list with no .zip locator
                # the error is a caller's, and is told as the location the
                # error names.
                if zip_at:
                    reached_how = "reached once the locators before it passed this file over"
                    cure = f"make one of those locators' locations writable{instead(None, 'a writable directory')}, or "
                else:
                    # Listed first, the .zip locator takes the file before any
                    # locator for a file on disk, so there are no locators
                    # before it to name.
                    reached_how = "which NUMBA_CACHE_LOCATOR_CLASSES puts before every locator for a source file on disk"
                    cure = "list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or " \
                           "UserWideCacheLocator before it, or "
                return (
                    f"numba's .zip locator, {reached_how}, takes the first part of the path ending in .zip, {filename}, "
                    "for an archive, and it is a directory, which numba 0.68 and later open for the source: "
                    f"{cure}put the file under a path with no such directory; or {silence}"
                )
            # numba took the location, so the variable was not passed over for
            # it; set elsewhere, the variable moves the cache off a disk that
            # is full. A location that is none of numba's is named as the error
            # names it, and an error that names none is told the locations
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
            # An error a caller builds from a message alone has no strerror.
            reason = failure.strerror or failure
            return (
                f"{opening} where no file can be written ({reason}): make room there, or make it "
                f"writable{alternative}; or {silence}"
            )
        # For a source on disk numba itself passes over a location it cannot
        # make or write in. So the error reaching this point is one of two. It
        # is the no-locator error: numba tried every locator and each passed
        # the file over. Or it is the .zip locator's error for a path with
        # .zip in it and no part ending in .zip, which numba raises when it
        # reaches that locator: the locators before it passed the file over,
        # and numba tried none after it.
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
            # Every locator numba tried that names its directory after the
            # source's directory found that name too long for its file system,
            # and passed the file over for that reason. A class listed after
            # the .zip locator is unreached for a path that the .zip locator
            # takes. The variable's locator measures by the file system of the
            # directory
            # the variable is set to, or, if the variable is unset, by the
            # limit nearly every file system has. No writable directory at a
            # short path cures a name too long, so none is asked for; and if
            # the .zip locator then raised for a path with .zip in it and no
            # part ending in .zip, the variable is not asked for with no
            # mention of the name either.
            too_long = too_long_a_name(locations[named_after_the_source[0]])
            # Under the user-provided locator alone, with the variable unset,
            # numba took no location at all and never measured the name. The
            # variable is what that locator takes nothing without, and the
            # name is what would stop that locator once the variable is set,
            # so both are told.
            variable = (
                ", with NUMBA_CACHE_DIR set, which UserProvidedCacheLocator takes nothing without"
                if not config.CACHE_DIR and all(reads_the_variable(cls) for cls in tried) else "")
            return (
                f"{too_long}: put the file in a directory of a shorter name{variable}{beside_the_source()}; or {silence}"
            )
        hashing = [
            cls for cls in tried if (reads_the_variable(cls) and config.CACHE_DIR) or caches_under_the_user_cache_dir(cls)]
        if subpath is None and (isinstance(failure, UnicodeEncodeError) or hashing):
            # numba hashes the source's directory path as UTF-8 for its
            # directory under NUMBA_CACHE_DIR or the user's cache directory,
            # and the locator that names its cache directory by that hash
            # raises: the variable's locator, if the variable is set, before
            # the in-tree locator is reached, which caches beside the source
            # and hashes no path; or the user-wide locator, once the in-tree
            # one passed an unwritable __pycache__ over. With the variable
            # unset the same install imports cached, so the remedy must say
            # more than that the variable cures nothing and a UTF-8 path is
            # needed. Under a list without the variable's locator, making the
            # listed locations writable cures nothing either, because hashing
            # refuses them however they are set up. numba hashed the path if
            # the error is its own UnicodeEncodeError, or if a locator that
            # names its cache directory by that hash was tried: the variable's
            # locator, when set, the user-wide one, or the .zip one. The in-tree locator
            # alone hashes nothing, and its no-locator error means the
            # __pycache__ was refused.
            hashed = (
                "numba hashes the source's directory as UTF-8 for the directory it caches this file in under "
                f"NUMBA_CACHE_DIR or the user's cache directory, and {os.path.dirname(py_file)} is not"
            )

            # The locators that hash the source's directory path are the
            # variable's locator, when the variable is set, the user-wide
            # locator, and the .zip locator for a path with .zip in it. The
            # in-tree locator caches beside the source, in its __pycache__,
            # and hashes no path. Listed after any locator that hashes, the
            # in-tree locator is never reached, so making the __pycache__
            # writable or listing the in-tree locator after the hashing one
            # cures nothing: numba still raises. The cure is to list the
            # in-tree locator before the first locator that hashes; if the
            # variable's locator is the only hashing one before it, the cure
            # is to unset the variable; and if the in-tree locator came first
            # and passed the file over, the cure is to make the __pycache__
            # writable.
            def hashes(cls):
                return ((reads_the_variable(cls) and bool(config.CACHE_DIR)) or caches_under_the_user_cache_dir(cls)
                        or (one_of(cls, for_a_zip) and ".zip" in py_file))

            beside = os.path.abspath(os.path.join(os.path.dirname(py_file), "__pycache__"))
            in_tree_at = next((at for at, cls in enumerate(order) if caches_beside_the_source(cls)), None)
            before_in_tree = [cls for cls in (order if in_tree_at is None else order[:in_tree_at]) if hashes(cls)]
            # Whether the in-tree locator takes the file, asked as numba asks
            # it: the in-tree locator takes a file whose __pycache__ it can
            # write. Hedging with "where that directory can be written"
            # instead of asking the locator, and never asking the reader to
            # make the directory writable, would leave numba raising the same
            # error after the reader unsets the variable as told.
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
            # The variable is set and numba passed it over: the directory it
            # points at is unwritable, or too deep for numba to make its own
            # directory there, and a writable directory at the same depth
            # would not cure the depth.
            return (
                f"NUMBA_CACHE_DIR is set to {config.CACHE_DIR}, which numba could not use: set it to a writable "
                f"directory at a short path, or {silence}"
            )
        return f"Set NUMBA_CACHE_DIR to a writable directory, or {silence}"

    def the_source_files():
        # The source files, on disk or in a .zip, with the locator each needs
        # listed if NUMBA_CACHE_LOCATOR_CLASSES leaves that locator out.
        on_disk = f"install {package} with its source files on disk, unpacked from any archive"
        if listed:
            # The install on disk caches through a locator for a file on disk
            # that the list names. If the list names no such locator, the
            # install caches no more than the archive does; if the list names
            # the user-provided locator alone, that locator takes nothing
            # without NUMBA_CACHE_DIR.
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
            # The file is in an archive already, APP.ZIP or an .egg, which
            # numba's .zip locator never takes, because the locator looks for
            # a part of the path ending in ".zip", lower-case. The remedy tells
            # a reader importing from such an archive to rename the archive,
            # naming the extension numba does not read, rather than offering
            # a .zip the reader already uses. The remedy offers the rename even
            # if a directory above the archive has ".zip" in its name, which
            # makes the locator take the file and find no archive. numba takes
            # the first part of the path ending in .zip for the archive, so a
            # directory so named above the archive is in the way: the rename
            # alone would leave numba opening the directory. The reader is
            # told to move the archive out from under that directory.
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
            # numba's .zip locator alone takes a .zip, so a list without it
            # would leave numba no locator for the .zip offered either.
            a_zip += f" through ZipCacheLocator, once NUMBA_CACHE_LOCATOR_CLASSES, {listed}, lists it"
        return on_disk, a_zip

    if isinstance(failure, KeyError):
        # numba 0.68 and later read a .zip member's source from the archive
        # for the stamp, by the name the module's code gives under the
        # archive's path, and zipfile raises for a name the archive does not
        # hold: a module shipped as .pyc alone, or, on Windows under 0.68.0,
        # any member. The Windows case happens because that release builds
        # the name with the platform's separator (numba issue 10889, fixed in
        # 0.68.1). The archive is checked for the member, so that a user is
        # not told to add source files that are there. A backslash alone does
        # not decide the case either: a module shipped as .pyc alone has no
        # source under the name numba 0.68.1 reads, so that release does not
        # cure it.
        member = _missing_member(failure)
        held = member.replace("\\", "/")
        on_disk, a_zip = the_source_files()
        if os.name == "nt" and "\\" in member and _numbas_archive_holds(py_file, held):
            # The install on disk is a cure now. A numba release with the fix,
            # none out as of 0.68.0, reads the member as the archive names it;
            # it is named by the fix, not as a release to install.
            return (
                f"numba 0.68.0 reads a .zip member's source by a name built with Windows's separator, {member}, which "
                "no archive holds (numba issue https://github.com/numba/numba/issues/10889, fixed for numba 0.68.1): "
                f"to cache, {on_disk}, which no separator touches, or a numba release that carries the fix, which "
                f"reads the member as the archive names it; or {silence}"
            )
        naming = _pyc_naming(py_file)
        if naming is not None and naming[1]:
            # The archive holds the module's source beside a .pyc compiled to
            # name another path in the archive. zipimport runs that .pyc, and
            # numba reads the source by the name the .pyc gives. The module is
            # not shipped as .pyc alone, so the remedy does not ask for source
            # files the archive already holds.
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
    # The locators numba reaches for a file not on disk, in numba's order.
    # numba's .zip locator takes the file without trying its location. A class
    # outside numba's families that takes the file when asked stops the order
    # too, so no locator listed after either is reached. numba's IPython and
    # user-wide locators pass over a location they cannot make or write in,
    # and numba tries the next locator.
    order = [cls for cls in _locators(listed) if may_take(cls)]
    zip_at = next((at for at, cls in enumerate(order) if one_of(cls, for_a_zip)), None)
    stops_at = next((at for at, cls in enumerate(order) if one_of(cls, for_a_zip) or not of_numbas(cls)), None)
    reached = order if stops_at is None else order[:stops_at + 1]
    # The .zip locator listed after such a class is not reached, and its
    # error for a path with no part ending in .zip is a caller's under the list.
    zip_at = zip_at if zip_at == stops_at else None
    user_cache_dir = AppDirs(appname="numba", appauthor=False).user_cache_dir
    in_the_user_cache = os.path.join(user_cache_dir, subpath) if subpath else None

    def location_of(cls):
        # Where the class places the file. The IPython locator places it in
        # numba_cache under IPython's cache directory. The result is None
        # without IPython, or when IPython raises OSError for its directory,
        # and numba then passes the locator over. The remedy asks IPython for
        # the directory only when a locator of that family is reached, never
        # for any other file. If it asked for any .zip member, an IPython with
        # its own directory unwritable would warn on every import of the
        # package and leave a temporary directory behind. The .zip and
        # user-wide locators place the file in a directory of their own under
        # the user's cache directory. A class outside numba's families has a
        # place of its own, which is None here.
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
        # Of numba's locators only the user-wide one takes a frozen
        # application, so a list without it leaves numba no locator of its
        # own for one. This covers numba's no-locator error, whatever classes
        # of the reader's own the list holds, those having passed the file
        # over, and any error a caller passes on when the list has no place
        # for the file. Asking for a writable directory would name one numba
        # never tried.
        return (
            "numba caches a frozen application through UserWideCacheLocator alone, and "
            f"NUMBA_CACHE_LOCATOR_CLASSES, {listed}, leaves it out: list it, or {silence}"
        )
    # Under a list with no locator that takes the file, or none with a place
    # for it, numba took no location. The list is then the remedy, below, for
    # an error a caller passes on, as for numba's no-locator error and as for
    # a source on disk. A class outside numba's families has a place of its
    # own.
    parts = pathlib.Path(py_file).parts
    in_a_zip = any(part.endswith(".zip") for part in parts)
    # A directory named .zip above the file. numba's .zip locator takes it for
    # an archive all the same, and caches nothing under it. A cell under such
    # a directory, whose source linecache does not hold, must not get the
    # archive remedy, the source files to ship.
    a_directory_named_zip = any(
        os.path.isdir(os.path.join(*parts[:at + 1])) for at, part in enumerate(parts) if part.endswith(".zip"))
    # A member of any archive, an .egg or .whl as well as a .zip, which
    # zipimport runs and whose source inspect reads through its loader. The
    # .zip is the one kind numba's .zip locator takes, and the only kind it
    # caches.
    in_an_archive = (in_a_zip and not a_directory_named_zip) or _archive_above(py_file) is not None
    # A caller's error with no errno and no name looks like inspect's, so
    # inspect's errors are known by their words, and a caller's error is
    # never told as inspect's.
    inspects = _is_inspects(failure) and any(one_of(cls, for_ipython) for cls in reached)
    if inspects and not in_an_archive:
        # inspect's error, with no errno and no name, for a cell whose source
        # linecache does not hold. numba's IPython locator reads a cell's
        # source from linecache, where IPython holds it while the cell runs,
        # and inspect raises before the locator tries its location, so there
        # is no location numba took to tell as refusing a file. For a member of
        # an archive that holds no source for it, a .pyc alone, inspect's error
        # gets the archive remedy below, the source files to ship; the archive
        # is found by looking, not by its name, so a member of an .egg gets it
        # too.
        return (
            f"numba's IPython locator reads a cell's source from linecache, which holds none for {py_file} "
            f"({failure}), and numba caches a cell only while IPython holds its source; or {silence}"
        )
    not_utf8 = _member_not_utf8(py_file) if inspects else None
    if not_utf8:
        # inspect's error for a member of an archive that holds its source,
        # which is not UTF-8. zipimport decodes a member's source as UTF-8 and
        # raises for one that is not. The probe's loader reads the member as
        # zipimport does but answers None, so the check raises inspect's
        # error, a cache error, instead of the decoding error, which is none
        # and would kill the import, as numba's would at decoration.
        return (
            f"numba's IPython locator reads the member's source as UTF-8, which {py_file} is not ({not_utf8}): ship "
            f"it as UTF-8; or {silence}"
        )
    hashed_by = next((cls for cls in reached if (one_of(cls, user_wide) and getattr(sys, "frozen", False))
                      or (one_of(cls, for_a_zip) and in_a_zip)), None)
    if isinstance(failure, UnicodeEncodeError) and hashed_by is not None:
        # numba's .zip locator names its directory under the user's cache
        # directory after the source's directory, with the archive's path in
        # it hashed as UTF-8, and raises for bytes that are not UTF-8, at
        # decoration and in the check alike; the user-wide locator does the
        # same for a frozen application's file. The error is not a caller's
        # with no place for the file, so the path is named, not the .zip the
        # reader imports from or the source files on disk. The first of the
        # two locators numba reaches is the one that raised. For a frozen
        # application under a directory named .zip, the user-wide locator
        # comes before the .zip locator in numba's order, so the application
        # is named, not an archive.
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
            # numba's .zip locator takes a frozen application's file for the
            # ".zip" in its path, without trying its location, and raises if
            # no part of the path ends in .zip, an archive or a directory.
            # Listed before the user-wide locator, which alone takes the file,
            # it leaves that one unreached, and the remedy is to list the
            # user-wide locator before the .zip one. Listed after it, the
            # user-wide locator passed its location over, unwritable or too
            # deep, and that location is the remedy, as for a source on disk.
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
        # The error names the location numba took, or a file numba writes in
        # it, as the first save's error does. The locator that took the
        # location decides the remedy: the first reached whose place the error
        # names. The frozen application's no-locator error names nothing.
        taken = next((cls for cls in reached if locations[cls] and locations[cls] in named), None)
        if taken is not None and one_of(taken, for_ipython):
            # numba caches the member in numba_cache under IPython's cache
            # directory. The remedy is the one a cell file on disk gets there,
            # with no NUMBA_CACHE_DIR to offer for a source not on disk.
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
        # The .zip's location, a directory of numba's own under the user's
        # cache directory, is what a reader can act on. It is named by the
        # error, or it is the directory of the file the error names. An error
        # that names no file is told the locations numba could have taken,
        # IPython's numba_cache among them if a locator of that family is
        # reached. The frozen application's no-locator error names nothing,
        # and the user's cache directory is told.
        if taken is not None:
            location = filename if os.path.abspath(filename) == locations[taken] else os.path.dirname(filename)
        else:
            location = filename or user_cache_dir
        # Whether to say where numba caches the file. A .zip's or a frozen
        # application's location is under the user's cache directory. A
        # location that is none of numba's is not. For a member that a locator
        # of IPython's family or a class outside numba's families is reached
        # for, an error that names no file may be that locator's, whether or
        # not IPython gave its directory. So nothing is said of the user's
        # cache directory for any of these.
        in_the_user_cache_alone = taken is not None or (
            not filename and all(of_numbas(cls) and not one_of(cls, for_ipython) for cls in reached))
        where = (
            "numba caches a .zip, or a frozen application, in the user's cache directory, and "
            if in_the_user_cache_alone else "")
        opening = f"{where}NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: "
        if isinstance(failure, FileNotFoundError) and names_a_path_above():
            # The stamp numba reads at decoration, of the archive the code
            # names: .pyc members compiled to name an archive that has since
            # moved. A location that cannot be made, under a dangling link, is
            # ENOENT too, but it does not lie above the code's file.
            suggestion = "" if numba_version >= 68 else "compile the archive's .pyc members to name its path now, or "
            return (
                f"the module's code names {filename}, which is not there: its .pyc was compiled to name "
                f"that path, so {suggestion}ship its source files; "
                f"or {silence}"
            )
        if (isinstance(failure, OSError) and failure.errno == errno.EISDIR and filename and os.path.isdir(filename)
                and filename.endswith(".zip") and names_a_path_above()):
            # numba takes the first part of the path ending in .zip for the
            # archive, a directory so named above the file included. From
            # 0.68 numba opens that archive to read the member's source,
            # which a directory refuses. The directory is no cache location,
            # so the reader is not told to make room in it.
            return (
                f"{opening}numba takes the first part of the path ending in .zip, {filename}, for the archive, and it "
                "is a directory, which numba 0.68 and later open to read the member's source: put the archive under "
                f"a path with no such directory above it, or {silence}"
            )
        if isinstance(failure, OSError) and names_a_path_above():
            # The archive that the member's code names cannot be read; numba
            # reads it for the stamp, and from 0.68 opens it for the member's
            # source. The remedy tells the archive as unreadable; making it
            # writable, as for a location that refused a file, would read
            # nothing.
            reason = failure.strerror or failure
            return (
                f"{opening}numba reads the archive, {filename}, for the member's stamp, and from 0.68 for the "
                f"member's source, and cannot ({reason}): make it readable, or {silence}"
            )
        if isinstance(failure, OSError) and filename and taken is None:
            # A location that is none of numba's, under the list and in its
            # order, which an error a caller passes on can name. The remedy
            # tells this location as the error names it, as for a source on
            # disk. It says nothing of what moves the user's cache directory,
            # because moving that directory moves nothing of this location,
            # and nothing of where numba caches the file, because this
            # location is none of numba's. One case: a member numba's IPython
            # locator takes, with IPython not importable, which numba caches
            # nowhere.
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
            # The .zip's directory under the user's cache directory, named
            # after the source's directory, is refused whatever the errno,
            # EINVAL on Windows. The user's cache directory at a shorter path
            # cures nothing of a name that is too long, so the remedy does not
            # ask for a shorter path.
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
                # The error names no file, and which of the locations is too
                # long cannot be told; the user's cache directory moves them
                # all if no locator of IPython's family is reached.
                moved_through = _moved_through() if in_the_user_cache_alone else ""
                return (
                    f"{opening}the path is too long for the file system, so put {could_have_taken(locations)}, at a "
                    f"shorter path{moved_through}, or {silence}"
                )
            return (
                f"{opening}the path is too long for the file system, so put that directory, {location}, at a shorter "
                f"path{_moved_through()}, or {silence}"
            )
        if isinstance(failure, OSError):
            # The location refused a file for another reason, a full disk or
            # permissions, which the error gives, and the remedy repeats it as
            # it does for a source on disk. The frozen application's
            # no-locator error gives no reason, because numba passed the
            # location over on its own error.
            reason = failure.strerror or failure
            if taken is None:
                return (
                    f"{opening}no file can be written in {could_have_taken(locations)} ({reason}), so make room "
                    f"there, or make it writable, or {silence}"
                )
            # numba's .zip locator takes the file without making its location.
            # The first save makes it, or the check does, under the first
            # directory above it that exists, and that directory refused. So
            # that directory is named: asking for room in a directory that is
            # not there, or for it to be made writable, would name the wrong
            # one. An error naming a file in the location found the location
            # there.
            made_under = ""
            if os.path.abspath(filename) == locations[taken] and not os.path.isdir(location):
                above = _existing_above(location)
                if not os.path.isdir(above):
                    # A file in the way. It is named; the directory above it
                    # is writable, and room in it would cure nothing.
                    return (
                        f"{opening}no file can be written in that directory, {location}, which cannot be made, "
                        f"{above} being a file in its path ({reason}), so put that file elsewhere, or {silence}"
                    )
                made_under = f", which cannot be made under {above}"
            return (
                f"{opening}no file can be written in that directory, {location}{made_under} ({reason}), so make room "
                f"there, or make it writable, or {silence}"
            )
        # numba's no-locator error for a frozen application gives no reason,
        # because numba passed the user-wide location over on its own error:
        # an unwritable directory, one too deep to make, or one whose name,
        # after the source's directory, is too long for the file system. The
        # last is told as such, because no shorter path cures it.
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
        # numba's no-locator error for a file that only numba's IPython locator
        # takes: a cell, or a .zip member in an ipykernel directory under a
        # list with no .zip locator after the IPython one. The locator passed
        # its location over, as numba's user-wide one does a frozen
        # application's, because it was unwritable or too deep to make, or
        # IPython raised for its directory and gave none. A class outside
        # numba's families listed with the IPython locator passed the file
        # over too, as numba's no-locator error shows, and the remedy names
        # it as such. Once asked, such a class is no longer in the order;
        # taking it for one of numba's locators would answer with the archive
        # remedy.
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
        # numba's IPython locator alone takes a cell, or a file in an ipykernel
        # directory that is not on disk, so a list without it leaves numba no
        # locator for one. The archive remedy would offer source files on disk
        # and locators that take none of these.
        return (
            f"numba caches a notebook cell through IPythonCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, {listed}, "
            f"leaves it out: list it; or {silence}"
        )
    if in_a_zip and not a_directory_named_zip and listed_without(for_a_zip):
        # numba's .zip locator alone takes a source in a .zip, so a list
        # without it leaves numba no locator for one; the archive remedy would
        # offer the .zip the reader already imports from. A source that is
        # nowhere, under a directory named .zip, is left out, because that
        # locator takes the directory for an archive and caches nothing.
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
    """``options`` as given, or with ``cache`` off, after one warning, if numba can write no cache for numbox.

    numba sets a cached function up when the function is decorated, and raises ``RuntimeError`` there if no cache
    location can be written: in a read-only install whose user cache directory cannot be written either, or on an
    import from an ``.egg``, ``.whl`` or ``.pyz`` archive, which Spark's ``--py-files`` ships. For a ``.zip`` numba
    takes the user's cache directory without checking at decoration that it can be written, and the first write
    raises ``OSError`` instead. Either way the import would die at the first decorated function, with nothing to
    name the way out.

    Every function numbox caches is decorated under the one ``jit_options``, so the question is put here, once, for
    a source file in each directory of the package that holds a module, and answered for the whole package. If numba
    can cache no function of one of those files, the options come back with ``cache`` off, and one warning names the
    remedy, as ``cache_remedy`` sets it out. For a source file on disk the remedy is ``NUMBA_CACHE_DIR``, and, if the
    location is too long for the file system, whatever shortens the location numba took. For a ``.zip`` it is the
    location numba took under the user's cache directory, made writable or given room, with the reason; or that
    directory at a shorter path, if the path is too long; or the ``.zip`` locator listed, if
    ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves it out. numba reads ``NUMBA_CACHE_DIR`` only for a source file on disk.
    For any other kind of archive, a ``.pyc``-only install or a ``.pyc`` in a ``.zip``, the remedy is the source
    files on disk, or a ``.zip`` holding them, which numba caches in the user's cache directory. The
    same goes, from numba 0.68, for a ``.zip`` member whose archive holds no source for it, a ``.pyc`` alone, because
    numba then reads the member's source from the archive for the stamp; the member is named. On Windows under numba
    0.68.0 every member is such a member, because the name is built with the platform's separator (numba issue
    10889), and the remedy is the install on disk, or a numba release carrying the fix.
    ``NUMBOX_JIT_OPTIONS='{"cache": false}'`` turns caching off and silences the warning. An error that is not the
    cache's, as ``is_a_cache_error`` draws the line, is raised as it was.

    A ``.zip`` whose cache directory holds every entry but can no longer be written takes the fallback too, where
    numba alone would have loaded the entries. The writability check is the rule numba applies to every other
    placement, and the one its ``.zip`` locator is missing, reported as https://github.com/numba/numba/issues/10888.

    Options without a ``cache`` key, which ``NUMBOX_JIT_OPTIONS`` can give, leave ``njit`` to its default, which is
    off. But the sqlite virtual-table and table-valued-function callbacks cache under those options, so they are
    asked about too, and come back with ``cache`` off, explicitly, if the answer is no.
    """
    if not options.get("cache", True):
        return options
    for py_file in _module_files():
        try:
            check_cache_location(py_file, LONGEST_CACHE_FILE_NAME)
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


jit_options = uncached_where_no_cache_can_be_written(get_jit_options())


_PROXY_CACHE_STRICT_ENV = "NUMBOX_PROXY_CACHE_STRICT"


def _strict_cache_mode():
    """True when ``NUMBOX_PROXY_CACHE_STRICT`` selects strict validation of ``@proxy`` cache loads.

    Strict mode makes every fail-open path in the ``numbox.core.proxy.proxy`` cache guard loud: a payload
    that cannot be read, or an object whose container cannot be parsed, aborts the load with
    ``UnvalidatedProxyCacheError`` instead of loading unchecked; and a detected stale alias raises
    ``StaleProxyCacheError`` before the heal, leaving the stale entry on disk. It is a debugging aid, off by
    default. Unlike ``jit_options`` above, the value is read on each call rather than once at import, so the
    knob can be toggled within a process; the cost is one environment lookup against a multi-millisecond
    cache load. Anything other than the unset/empty/``0``/``false``/``no``/``off`` set (case-insensitive)
    enables it.
    """
    value = os.environ.get(_PROXY_CACHE_STRICT_ENV)
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


MAX_STR_LENGTH = 2 ** 31 - 1

#: numba's `FunctionModel`, resolved once. Looking a data model up is type machinery and
#: compiles nothing, which is what lets the layout be read here: `numbox.utils.lowlevel` and
#: `numbox.utils.derive_wap` both need it on paths that must stay free of compilation side
#: effects, and importing the former compiles a cached eager `@njit` helper.
_function_model = default_manager.lookup(FunctionType(void()))

#: Slots in the `FunctionModel` struct: (c_addr, py_addr, jit_addr)
function_struct_size = _function_model.field_count
