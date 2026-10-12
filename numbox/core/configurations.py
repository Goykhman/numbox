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
assert numba_version >= 60, numba_version


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
        # Its keys go to njit as keyword arguments, and the fallback below
        # reads "cache" from it, so a value of another shape stops here, named.
        raise ValueError('NUMBOX_JIT_OPTIONS must be a JSON object, e.g. {"cache": false}')
    if "cache" in as_json and not isinstance(as_json["cache"], bool):
        # numba reads the option's truth, so the string "false" turned caching on.
        raise ValueError('NUMBOX_JIT_OPTIONS "cache" must be true or false, e.g. {"cache": false}')
    return as_json


def _cache_probe():
    """A function of this module, never compiled: its code's file is the one numba looks up for the module."""


# The longest file numba writes for a function of numbox's own files, with room to spare: the module's stem,
# the function's qualified name, numba's line number, interpreter tag and index number, and the 21 bytes of the
# temporary name it is written under. The longest-named function of the package comes to 105 bytes and the
# builder's generated kernel, anchored to its file, to 117; a test holds every function of the package under
# this bound.
LONGEST_CACHE_FILE_NAME = 128

# The digits reserved for the number numba gives a function's data file, one per signature, counted from one
# without bound: three, a thousandth signature of one function being beyond what the package and its generated
# code make; a name an error carries with a wider number is read by its own width. Two digits, a ninety-ninth
# signature's, left the hundredth's name a byte longer than the reserve.
_OVERLOAD_DIGITS = 3

# The probe's module, one per check, named apart from every other's, stands in
# sys.modules while numba picks the locator.
_probes = itertools.count()


class _MemberSource:
    """The loader of the check's probe's module, which inspect reads a ``.zip`` member's source through, as zipimport
    for the member's own module: the member's bytes decoded as zipimport decodes them, as UTF-8, or None where no
    archive above the file holds the member, or it is not UTF-8, which zipimport raises for and linecache lets
    through: inspect's error tells such a member, and the remedy reads why."""

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
    """A probe, a function never compiled whose code names ``py_file``, in ``sys.modules`` as a function of a module
    of its own for the block, which asks a locator about the file as numba asks it at decoration.

    ``py_file``, a string, bytes or a path object, is read as a string, the name the probe's code carries, as compile
    makes it of bytes. numba's IPython locator reads the function's source when it takes the file, as inspect reads
    it: a file on disk itself, a cell from linecache, where IPython holds it while it runs the cell, and a ``.zip``
    member through the function's module's loader, zipimport's, which gives the member's source from the archive, or
    None for a member compiled to ``.pyc`` alone; the probe's module's loader reads the member as zipimport's does,
    and inspect looks the function's module up in ``sys.modules``. The lines linecache holds from then on are the
    member's own, as any reader leaves them; for a member the archive holds no source for linecache keeps the probe's
    loader to ask again, and the block takes it out at its end, a loader of the probe's own staying for no file.
    """
    file = os.fsdecode(py_file)
    # The probe's own two lines, put in linecache under the member's name
    # instead while numba picked the locator, were what a reader of the
    # member meanwhile, in another thread, read for it.
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
    """Whether ``cls``, a locator class of none of numba's families, takes ``py_file``: its ``from_function``, asked
    with a probe as numba asks it at decoration, gives a locator for a file it takes and None for one it passes over.
    A class whose ``from_function`` raises takes the file, the error being its own, as numba raises it at decoration;
    the remedy took such a class for taking any file, and read the file as its own where it had passed it over and
    numba had gone on to the locators listed after it."""
    with _probing(py_file) as probe:
        try:
            return cls.from_function(probe, os.fsdecode(py_file)) is not None
        except Exception:
            return True


def check_cache_location(py_file, longest_file_name=0):
    """Raise as numba would where a function whose source is ``py_file``, a string, bytes or a path object, read as a
    string, cannot be cached; else return.

    The question is put the way numba puts it: the cache set-up that decoration runs, which picks the location
    for the file or raises ``RuntimeError`` with no locator, then the source's stamp, which decoration reads
    and which stats the archive for a ``.zip``, reading the member's source from it too from numba 0.68, as it then
    reads a source on disk for its hash, then the
    writability check, which decoration runs for every
    location but a ``.zip``'s and the first save runs for all, raising ``OSError``. The probe is compiled with
    ``py_file`` as its file, which is all a locator reads of it but for numba's IPython locator, which reads the
    function's source too, as inspect reads it: a file on disk itself, a cell from linecache, where IPython holds it
    while it runs the cell, and a ``.zip`` member through the function's module's loader, zipimport's, which gives the
    member's source from the archive; the probe is a function of a module of its own, whose loader reads the member
    as zipimport's does, in ``sys.modules`` while numba picks the locator, and the lines linecache holds from then on
    are the member's own, as any reader leaves them; for a member the archive holds no source for linecache keeps the
    probe's loader to ask again, and the check takes it out, a loader of its own staying for no file. A cell linecache
    does not hold, or a member compiled to ``.pyc`` alone, raises inspect's error, as numba does at decoration.
    Nothing is
    compiled, and nothing is written but the cache directory itself and, asked for it under an IPython that can write
    neither its cache directory nor its own, the temporary directories IPython makes, one per call, three for the
    check, two of them numba's own making the directory. An ``OSError`` from the check names the
    location numba picked, but for three: inspect's, which names no file, for a cell file on disk that numba's
    IPython locator takes, where inspect reads the source from the file itself, before that locator tries its
    location, and finds none there, in a file it cannot read or, before Python 3.13, an empty one, which linecache
    gives a blank line from then on; the stamp's, which names the archive a ``.zip`` member's code names where that
    has since moved or cannot be read, read once the locator is picked, as numba's own does at decoration; and, from
    numba 0.68, where
    the stamp hashes the source, the stamp's naming the source itself, which cannot be read, or the directory named
    ``.zip`` above a file on disk, which numba's ``.zip`` locator takes for the archive and opens. A location that
    is not there once made, numba's IPython locator asking IPython for its directory
    at every save and IPython giving a new temporary one on every call where it can write neither its cache
    directory nor its own, raises ``FileNotFoundError`` naming the last directory IPython gave the check, which is
    not there, as numba's first save names the one IPython gave it, a temporary directory in either case and a
    different one.

    numba's writability check makes a temporary file whose name is short, or none at all on Linux, and the
    files it saves have names of up to a hundred bytes and more, so a location within their length of the path
    limit passes the check and the first save dies. Given ``longest_file_name``, a file with a name that long
    is made and removed in the location too, and the ``OSError`` is the location's: the package asks with
    ``LONGEST_CACHE_FILE_NAME``, the bound on numba's names for its own files. Where that name is longer than the
    file system holding the location takes, the error carries the length as ``longest_file_name``, no shorter
    path curing it, and the remedy asks for a file system that takes it.
    """
    # What a locator is asked about: a probe, in a module of its own.
    with _probing(py_file) as probe:
        locator = CompileResultCacheImpl(probe).locator
    # numba reads the source's stamp at decoration too, the archive's for a
    # .zip, which is not there where the code names an archive since moved.
    locator.get_source_stamp()
    try:
        locator.ensure_cache_path()
    except OSError as error:
        # The error names what failed, a temporary file's name among the
        # possibilities; the location it is in is what a reader can act on.
        raise OSError(error.errno, error.strerror, locator.get_cache_path()) from error
    # The location is asked for once more, as numba asks at every save, and
    # its IPython locator asks IPython, which gives a new temporary directory
    # on every call where it can write neither its cache directory nor its
    # own: numba makes numba_cache in one and saves in another, dying there
    # past its own check; the check passed the file the same way, and asked
    # for the location at every step, a temporary directory each, where numba
    # asks twice making it and the check once more.
    location = locator.get_cache_path()
    if not os.path.isdir(location):
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", location)
    if longest_file_name:
        try:
            # tempfile's name is the prefix and eight characters of its own.
            with tempfile.NamedTemporaryFile(dir=location, prefix="x" * (longest_file_name - 8)):
                pass
        except OSError as error:
            raised = OSError(error.errno, error.strerror, location)
            if error.errno == errno.ENAMETOOLONG and _name_limit(location) < longest_file_name:
                # The name alone is too long for the file system, not the
                # path: the error carries the length, for the remedy, which
                # read it as the location's and offered a shorter path.
                raised.longest_file_name = longest_file_name
            raise raised from error


def is_a_cache_error(error):
    """Whether ``error`` is numba's for a cache it cannot set up: no locator, a location it cannot use, or, from
    numba 0.68, a ``.zip`` member whose source it cannot read from the archive.

    numba itself passes over a location on any ``OSError`` from making its directory or writing a file there,
    permission denied, a read-only file system, a path into a file, a component too long, a full disk, so any
    ``OSError`` counts. So does the ``ValueError`` numba raises for an archive under a directory whose name
    holds ``.zip``: its ``.zip`` locator takes the file by that substring and then finds no ``.zip`` in it. And
    so does the ``KeyError`` numba 0.68 and later raise reading a ``.zip`` member's source from the archive,
    which they hash into the archive's stamp: zipfile's, naming the member, for a name the archive does not
    hold, a module shipped as ``.pyc`` alone, or, on Windows under numba 0.68.0, any member, the name built with
    the platform's separator (numba issue 10889, fixed in 0.68.1). And so does the ``UnicodeEncodeError`` numba
    raises hashing the source's directory path as UTF-8 for the name of its directory under ``NUMBA_CACHE_DIR`` or
    the user's cache directory, for a path holding bytes that are not, decoded with surrogates: a locator that
    names its directory so raises it at decoration where it is reached, the ones before it having passed the file
    over, and no location takes the file; a package installed under such a directory, its ``__pycache__``
    unwritable, died at import on it where, with caching off, it imports.
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
    """The source files numba's answer for numbox rests on, this module's own first.

    numba's in-tree cache is a ``__pycache__`` beside each source, so the directories answer separately and one
    can be writable while another is not: one module of each directory stands for the directory, whether or not
    that directory's modules cache anything, so a directory numba cannot cache in turns caching off for the
    package even where every cached function's own directory is fine; the answer errs toward uncached, which
    is never wrong. And numba finds a location for a module by its code's file, so a module that survives as
    ``.pyc`` alone, beside sourced ones, answers for itself, by the file it was compiled from, which its code
    keeps: the ``.py`` that is gone where it was compiled in place, or a tree elsewhere, on disk or not. The
    package is found by this module's ``__file__``, which is where it was imported from, an archive or a
    directory: its code's file, which numba looks up for the probe and which is asked first, can be elsewhere
    the same way. An archive shows no directories to walk, so there its members are listed instead.
    """
    yield inspect.getfile(_cache_probe)
    package = os.path.dirname(os.path.dirname(__file__))
    if not os.path.isdir(package):
        yield from _archived_module_files(__file__)
        return
    walked = set()
    for directory, subdirectories, files in os.walk(package, followlinks=True):
        # A symlink is followed, and a directory reached twice through links,
        # a cycle among them, is walked once.
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

    The import system runs such a ``.pyc`` through its sourceless loader, which keeps the compile-time file on
    the code, and fails the module's import on one it cannot run, of another interpreter's magic number or
    unreadable; so the loader's answer is read, and none leaves nothing to ask.
    """
    loader = importlib.machinery.SourcelessFileLoader(os.path.basename(pyc_path)[:-4], pyc_path)
    try:
        return loader.get_code(loader.name).co_filename
    except Exception:
        return None


def _archived_module_files(own):
    """One module per directory of numbox inside the ``.zip`` that holds ``own``, this module as imported, the
    file each ``.pyc`` in it that zipimport would run was compiled from, and each ``.pyc`` whose code names a member
    the archive does not hold, alone with no ``.py`` beside it or compiled to name another path in the archive, for
    itself; nothing for any other archive.

    numba caches a ``.zip`` per directory of it, each in a location of its own under the user's cache directory,
    so the directories answer separately there too. A ``.pyc`` run from the archive keeps the file it was compiled
    from as its code's file, and that is what numba looks up for its functions, the archive's own path for the
    module is not; so such a member asks by that file, there or gone. numba 0.68 and later read the member's source
    from the archive by the name its code gives, and a ``.pyc`` alone has none there, nor has one compiled to name
    another path in the archive, ``compileall -d``, beside its ``.py`` or not, so either asks for itself whichever
    member stands for its directory. The archive is the first part of the path named ``.zip`` that is one, a
    directory so named above it being no archive. Any other archive has no location at all, and the probe's own
    file has already asked.
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
                # The member the code names is not in the archive: a .pyc
                # alone names the .py it was compiled from, gone, and one
                # compiled to name another path in the archive, compileall's
                # -d, names a member that was never there, beside its .py or
                # not. numba 0.68 and later read the member's source from the
                # archive by that name, and find none, so it is asked for
                # itself whichever member stands for its directory; a listing
                # that let a .py member stand for the directory never asked,
                # and the module died at decoration, as it did for a .pyc
                # beside its .py that a listing took for naming the archive.
                seen.add(directory)
                yield member
                continue
        else:
            member = os.path.join(zip_path, *name.split("/"))
        if directory not in seen:
            seen.add(directory)
            yield member


def _compiled_from(zip_path, directory, stem):
    """The file the code zipimport runs module ``stem`` of ``directory`` from names, or None where it runs none.

    zipimport decides: it runs the ``.pyc`` before the ``.py`` beside it, but not one of another interpreter's
    magic number, nor one stale against that ``.py``, which it passes over for the source, nor one it cannot
    unmarshal, and the module's import fails with it; so its answer, its code, is read. The file that names is
    the archive's own path for the module where zipimport compiled the ``.py``, or where the ``.pyc`` was
    compiled to name a member the archive holds, and stands for the directory as a ``.py`` member does; else it
    is a file the archive does not hold, elsewhere or a path in it the ``.pyc`` was compiled to name, there or
    gone, and asks for itself. No code at all, for whatever
    stops zipimport, a source it cannot compile or a member it cannot decompress among the reasons, leaves
    nothing to ask.
    """
    if stem == "__init__":
        directory, _, stem = directory.rpartition("/")
    importer = zipimport.zipimporter(os.path.join(zip_path, *directory.split("/")))
    try:
        return importer.get_code(stem).co_filename
    except Exception:
        return None


def _moved_through():
    """What moves numba's user cache directory on this platform, as a clause for a remedy; nothing on Windows.

    numba's ``appdirs`` asks the system for it on Windows, which no variable changes, puts it under ``HOME`` on
    macOS, and takes ``XDG_CACHE_HOME``, else a directory under ``HOME``, elsewhere.
    """
    if sys.platform == "win32":
        return ""
    if sys.platform == "darwin":
        return ", through HOME"
    return ", through XDG_CACHE_HOME or HOME"


def _measured(name):
    """``name``'s length as the file system counts it, in bytes, or in the UTF-16 units Windows and macOS count a name
    in, and the unit's name. A surrogate, which a name decoded with surrogateescape carries for a byte that is not
    UTF-8, and which Windows takes in a name, is one unit; the count raised on it, and the remedy died on a name
    numba's error carried instead of answering."""
    if os.name == "nt" or sys.platform == "darwin":
        return len(name.encode("utf-16-le", "surrogatepass")) // 2, "characters"
    return len(os.fsencode(name)), "bytes"


def _name_limit(location):
    """The longest name a file can have where ``location`` is: 255 where the system cannot say, or says 0, as a file
    system with no limit of its own reports it, or for none.

    In bytes, which the file systems elsewhere count a name in, or in the UTF-16 units Windows and macOS count one in.

    Asked of the nearest directory above ``location`` that exists, since the location itself may not be there yet.
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


def _class_named(module, name):
    """The class of this name in ``module``, as numba spells its locators from 0.62, or, under 0.60 and 0.61, with the
    underscore it put before them until then; None where the module has no such class. The underscore is tried under
    those two alone, which read no ``NUMBA_CACHE_LOCATOR_CLASSES``, so that a listed name numba refuses from 0.62,
    ``CacheLocator`` for its base class, is left out as numba leaves it."""
    cls = getattr(module, name, None)
    if cls is None and numba_version < 62:
        cls = getattr(module, "_" + name, None)
    return cls


def _numba_locator(name):
    """numba's cache locator class of this name, or None where this numba has none: the ``.zip`` locator arrived in
    0.61."""
    from numba.core import caching
    return _class_named(caching, name)


def _locators(listed):
    """numba's cache locator classes in the order it tries them.

    ``listed`` is ``NUMBA_CACHE_LOCATOR_CLASSES`` as numba 0.62 and later read it, each entry a class of numba's
    caching module by its name or its dotted path, resolved as numba resolves it, with the underscore numba put before
    its locators' names until 0.62 allowed in either; an entry numba could not resolve
    it has refused already, before any location was tried, or raised importlib's error for, a relative module path's
    TypeError, and one that names no class, or a class with no ``from_function``, it fails on, for want of
    ``from_function``, at a function it decorates where it reaches the entry, so neither is a locator here, and
    numba's own error for either is raised as it was. Empty, the order is numba's own.
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
                cls = _class_named(importlib.import_module(module_path), class_name)
            except Exception:
                # importlib's errors, a relative path's TypeError, and
                # whatever the module raises on import, which numba raises
                # as its own at decoration and the check as its own before
                # any remedy: the remedy, reached for an error of the
                # cache's, raised the module's for the entry.
                cls = None
        else:
            cls = _numba_locator(entry)
        if isinstance(cls, type) and callable(getattr(cls, "from_function", None)):
            classes.append(cls)
    return classes


def _existing_above(path):
    """The first path at or above ``path`` that is there: the directory a directory on the way to ``path`` is made
    under, which refused where the location is not there, or a file in the way, which no directory can be made under;
    the root at the last. ``path`` is read absolute: a relative one with no ancestor on the way walked up to nothing,
    and the remedy said a file with no name was in the way."""
    path = os.path.abspath(path)
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def _ipython_numba_cache():
    """numba's cache location for a cell file its IPython locator takes: ``numba_cache`` under IPython's cache
    directory, with no directory per file; None where IPython is not importable, which numba's locator raises
    ImportError for at decoration, no cache error, so the location is then none of numba's, and None where IPython
    raises for its directory: OSError, which numba's locator passes the file over on, as it does a location it cannot
    make, so that numba took no location of IPython's, or anything else, which numba's locator would raise at
    decoration and which the remedy, asked the same for the words of a warning, takes as no location given."""
    try:
        try:
            from IPython.paths import get_ipython_cache_dir
        except ImportError:
            from IPython.utils.path import get_ipython_cache_dir
        return os.path.join(get_ipython_cache_dir(), "numba_cache")
    except Exception:
        return None


# What inspect says where it reads no source: a cell linecache does not hold, a member the archive holds no source
# for, a function whose line the lines it read do not reach, and a function of no module.
_INSPECTS_ERRORS = ("could not get source code", "source code not available", "lineno is out of bounds")


def _is_inspects(failure):
    """Whether ``failure`` is inspect's, for a source it could not read: an ``OSError`` with no errno and no name and
    one of inspect's own messages; a caller's error with no errno and no name is not, though it looks the same."""
    return (isinstance(failure, OSError) and failure.errno is None and not getattr(failure, "filename", None)
            and str(failure) in _INSPECTS_ERRORS)


def _ipython_cache_is_temporary(first):
    """Whether IPython gives a new temporary directory as its cache directory on every call, which it does, warning
    so, where it can write neither its cache directory nor its own; numba's IPython locator asks for the directory at
    every save, so numba makes ``numba_cache`` in one and saves in another. ``first`` is one answer already given,
    and one more is asked for; each is a directory IPython leaves behind. An IPython that gives a directory once and
    raises the next time gives no temporary directory: the second answer, none, counted as a new one."""
    if first is None:
        return False
    second = _ipython_numba_cache()
    return second is not None and second != first


def _taken_by_ipython(py_file):
    """Whether numba's IPython locator takes ``py_file`` by its name alone, on disk or not: a cell, ``<ipython-...>``, or
    a file in an ipykernel directory. It reads the function's source when it does."""
    return py_file.startswith("<ipython-") or os.path.basename(os.path.dirname(py_file)).startswith("ipykernel_")


def _archive_above(py_file):
    """The archive ``py_file`` is a member of and the member's name in it, as zipimport finds them: the first part of
    the path that is a file, whatever its name, must be a zip archive, and the rest is the member; None where no part
    is a file, or the first that is one is no archive.

    The path is read as the platform's, by its separator: a path of the platform ``os.name`` says, which a test sets to
    Windows's, Python before 3.12 refuses to make on Linux, and the remedy for a member numba names with Windows's
    separator died on it."""
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


_FAMILIES = ("UserProvidedCacheLocator", "InTreeCacheLocator", "UserWideCacheLocator", "IPythonCacheLocator",
             "ZipCacheLocator")


def _of_numbas(cls):
    """Whether ``cls`` is of one of numba's locator families, whose places the remedy knows; a class of none of them,
    with a ``from_function`` and a place of its own, takes what its ``from_function``, asked with a probe as numba asks
    it, takes, and caches where it alone can say."""
    return any(base is not None and issubclass(cls, base) for base in map(_numba_locator, _FAMILIES))


def _ipython_locator_reads(py_file):
    """Whether a locator of numba's IPython family that numba reaches, in its own order or where
    ``NUMBA_CACHE_LOCATOR_CLASSES`` lists one, takes ``py_file`` by its name, and so reads the function's source; numba
    reaches none listed after its ``.zip`` locator for a path with ``.zip`` in it, which that locator takes without
    trying its location, nor after a locator of its own, or a class of none of its families, that takes the file,
    asked with a probe as numba asks it, the in-tree one for a cell file on disk whose ``__pycache__`` it can write
    among them; one that passes the file over leaves the ones after it reached."""
    from numba import config
    for_ipython = _numba_locator("IPythonCacheLocator")
    for_a_zip = _numba_locator("ZipCacheLocator")
    if for_ipython is None or not _taken_by_ipython(py_file):
        return False
    for cls in _locators(getattr(config, "CACHE_LOCATOR_CLASSES", "")):
        if issubclass(cls, for_ipython):
            return True
        if for_a_zip is not None and issubclass(cls, for_a_zip) and ".zip" in py_file:
            return False
        if _takes(cls, py_file):
            # The remedy read IPython's locator as reached past a locator of
            # numba's own that took the file, and told a caller's error
            # naming another numba_cache, for a cell file on disk whose
            # __pycache__ the in-tree locator writes, as IPython's.
            return False
    return False


def _numbas_archive_holds(py_file, name):
    """Whether the archive numba's ``.zip`` locator takes for ``py_file``, the first part of the path ending in
    ``.zip``, holds a member named ``name``; False where that is no archive."""
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
    """The ``.pyc`` member of the archive numba's ``.zip`` locator takes for ``py_file`` whose code zipimport runs
    names ``py_file``, and whether a ``.py`` stands beside it, or None where no member's does: a ``.pyc`` compiled to
    name another path in the archive, ``compileall -d``, which numba 0.68 and later read the source by, and find none
    under, while the archive holds the source beside the ``.pyc``."""
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
    """Why the member of the archive above ``py_file`` is not UTF-8, which zipimport decodes a member's source as and
    numba's IPython locator reads it by: the byte and the reason; None where no archive above holds the member, or it
    decodes."""
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

    zipfile quotes the name as Python does, so a backslash in it, which numba 0.68.0 builds a member's name with on
    Windows, is doubled there and read back.
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
    """The remedy for ``failure``, numba's for a function whose file is ``py_file``, ending in ``silence``.

    ``NUMBA_CACHE_DIR`` for a source file on disk, which is the only kind numba reads the variable for; where it is
    set and numba passed it over, the variable named and a writable directory at a short path asked for, since
    numba passes over one too deep to make its directory in as it does an unwritable one; and where
    ``NUMBA_CACHE_LOCATOR_CLASSES``, from numba 0.62, leaves the user-provided locator out, the locations it names
    made writable, numba never reading the variable. Where the
    location is too long for the file system, the error names it, and the remedy is for the one it is: numba takes
    a directory under ``NUMBA_CACHE_DIR`` where that is set, else the ``__pycache__`` beside the source, else a
    directory under the user's cache directory, each of the two under a cache directory named for the source's
    directory, by its name and a hash of its path; the path the error names is matched whole against each location a
    locator numba reaches gives it, in the order numba tries them, since the three can nest and ``NUMBA_CACHE_DIR`` set
    to the user's cache directory makes one path of two, and the error is read as that locator's, the ones numba tries
    before it having passed the file over, as numba's own errors mean and an error a caller passes on naming a
    location numba did not take need not; numba reaches no locator listed after its ``.zip`` one for a
    path with ``.zip`` in it, which that locator takes without trying its location, so a location one of those would
    give is none of numba's. So a shorter ``NUMBA_CACHE_DIR`` for the first; the
    package at a shorter path for the second; the user's cache directory at a shorter path, through
    ``XDG_CACHE_HOME`` or ``HOME``, ``HOME`` alone on macOS and nothing on Windows, for the third; and for either
    of the last two ``NUMBA_CACHE_DIR`` set to a short path, where numba tries it before the locator that took the
    location, or before the last of the locators that give the file that one path, where the user-wide and ``.zip``
    ones do, or, where it is set and numba passed it over, named and made a writable directory at a short path. The
    directory numba names after the source's, under ``NUMBA_CACHE_DIR`` or the user's cache directory, is that
    directory's name and forty-one bytes, and one longer than a name can be on the file system holding the location,
    which the system is asked for, 255 where it cannot say, in bytes, or in the UTF-16 units Windows and macOS count
    a name in, can be made at no path, and is refused whatever the errno, EINVAL on Windows: an error naming such a
    location, numba's no-locator error for a file on disk where every locator it tried that names its directory so
    found the name too long for its file system, the variable's unset by that limit, and a frozen application's are
    told the name and its length, and the file in a directory of a shorter name, with, for a file on disk, the
    ``__pycache__`` beside the source made writable where the in-tree locator is listed, or that locator listed where
    it is not, which caches under no such name. A cache file numba could not make or open in the location it took,
    whose name, given after the source's name and the function's qualified name, is longer than a name can be on that
    file system, is told the name and its length, which no shorter path cures, and a shorter name for the function
    or its file, the package's to give, and the check's error for a name as long as the package's functions get,
    which that file system takes no name as long as, is told the length, with ``NUMBA_CACHE_DIR`` pointed at a
    directory on a file system that takes a
    name that long offered where a file system in use takes the name, 255 being their limit, less the 29 of the
    longest name numba writes for the function, a data file's with three digits for its number, numba numbering
    them from one without bound, through the temporary name it writes each file through, and numba tries the
    variable before the locator that took the file, as it is offered at a short path for a path too long, and
    nothing offered for a ``.zip`` member, which the variable does nothing for. numba hashes the source's directory
    path as UTF-8 for that name and
    raises for bytes that are not, at decoration, where a locator that names its directory so reaches the file, so
    for such a path the remedy names no location under ``NUMBA_CACHE_DIR`` or the user's cache directory, and a
    ``.zip`` under such a path, whose archive's path numba's ``.zip`` locator hashes the same way for its directory
    under the user's cache directory, is told the archive under a UTF-8 path, a frozen
    application under one, whose path numba's user-wide locator hashes the same way, the application under a UTF-8
    path, and a file on disk under one is told
    the ``__pycache__`` beside it too, which numba's in-tree locator hashes no path for: that locator to list before
    the first that hashes,
    where the list leaves it out or puts it after one, ``NUMBA_CACHE_DIR`` to unset where its locator alone hashes
    before the in-tree one, or that directory to make writable where the in-tree locator came first and passed it
    over, and with any of these where it cannot be written. A
    location that is none of numba's is named as the error names it, with the variable, where numba tries it before
    any other locator, named as set or asked for at a short path, and nothing said of numba passing it over, which
    that location cannot show. An error can name no file, or carry an empty string or empty bytes as its name, or a name of
    another kind, an integer file descriptor's, which is none. A caller
    can build such an error, and numba's first save raises one where a write fails, as on a full disk. So does inspect
    where, for numba's IPython locator, it reads no source in a cell file on disk, unreadable or, before Python 3.13,
    empty, with no errno
    either, no call of the file system's having raised it, and that error, for a file on disk that a locator of
    IPython's family numba reaches takes, is told as inspect's, that locator having tried no location: the file to
    make readable, with the source in it; for a cell, whose source numba reads from linecache, where IPython holds
    it while it runs the cell, as that, numba caching a cell only while IPython holds its source; and for a ``.zip``
    member whose archive holds no source, a ``.pyc`` alone, as the archive's, its source files to ship. Every other
    OSError that
    ``check_cache_location`` raises names a file. An error with no name, or an empty one, is told the locations numba
    could have taken, in numba's order and each once. For a path too long, the remedy is to put whichever of them is too
    long at a shorter path. For a file refused for another reason, such as a full disk, the remedy gives the reason and
    asks for room or a writable directory there. A path object is never empty: pathlib reads an empty string as the
    working directory, ``.``, which the remedy names as a location that is none of numba's.
    ``NUMBA_CACHE_LOCATOR_CLASSES`` decides that order, each entry a class of numba's caching module or, by its dotted
    path, a subclass of one, which takes what its parent takes, caches where it does and is told as it is, or a class of
    none of numba's families, with a ``from_function`` and a place of its own, which takes what its ``from_function``,
    asked with a probe as numba asks it, takes, where it caches being its own to say: an error naming a location that
    no locator of numba's gives the
    file is told that location as the error names it, which is then that class's, and one naming no file is told the
    location numba took as the class's own, with no path to name, among numba's locators' places where those are
    reached. numba's
    IPython locator caches a cell file it takes in ``numba_cache`` under IPython's cache directory, with no directory
    per file, and passes the location over where it cannot make or write in it, so that a cell, which no other
    locator of numba's takes, gets numba's no-locator error and is told that location, to make writable or put at a
    shorter path, or IPython's cache directory to make a home for where IPython raised for it, with a class of none
    of numba's families listed with it named as having passed the cell over too; that location too long is
    told as IPython's cache directory at a shorter path, for a cell file on
    disk and for a ``.zip`` member in an ipykernel directory alike, where that locator is listed before the
    ``.zip`` one, which takes a file without trying its location, so that none listed after it is reached, with no
    ``NUMBA_CACHE_DIR`` offered for a source not on disk; IPython is asked for that directory
    where a locator of that family is listed and takes the file, and left alone for every other, a plain ``.zip``
    member among them, since it warns when asked under a home it cannot write and leaves a temporary directory
    behind; numba imports
    IPython for the location when it makes it and catches only OSError there, so where IPython is not importable it
    raises ImportError at decoration for a file that locator takes, no cache error, and an error a caller passes on
    naming that location is told it as none of numba's where another locator numba reaches has a place for the file,
    and the list where none has, as numba's no-locator error is told it; and where IPython raises OSError for its
    directory numba passes the locator over, as it does a location it cannot make, and the remedy, asked the same, has
    no location of IPython's to match the error against. numba's IPython locator takes a file on disk only in
    an ipykernel directory and its ``.zip`` locator only a path with ``.zip`` in it, so either ahead of the rest
    changes nothing for any other file; the ``.zip`` locator ahead of them all takes such a path first and caches it
    under the user's cache directory where a part of the path ends in ``.zip``, an archive or a directory, or, where
    none does, finds no archive in it, and the remedy
    then asks for a locator for a file on disk listed before it, the user-provided one with ``NUMBA_CACHE_DIR`` set,
    which alone it takes nothing without; after some of them, it finds no archive once those have passed the file
    over, numba trying none after it, and the remedy asks for one of their locations made writable or such a
    locator listed before it. A list with no locator that takes the file is told so, for an error a caller
    passes on under it as for numba's no-locator one, and so is one whose only locator for the file is the
    user-provided one with ``NUMBA_CACHE_DIR`` unset, which takes nothing without it: the variable to set, as
    numba's no-locator error is told there. Where the location numba took refuses a file for another reason, a
    full disk or permissions changed since numba's own check, the remedy names the location and the reason and asks for room
    or a writable directory there, with ``NUMBA_CACHE_DIR`` as the alternative where numba tries it before that
    location, or set to another directory where the location is the variable's own; numba passes over a location it
    cannot make or write in, so only its no-locator error means the variable was passed over. The error can name the
    location or a file numba writes in it, as a string, bytes or a path object, read as a string; one that names a
    location that is none of numba's is told that location as the error names it, and one that names no file is told
    the locations numba could have taken. For a ``.zip`` or
    a frozen application, both cached under the user's cache directory, the location made writable or room made
    there, with the reason the error gives, a full disk or permissions: the ``.zip``'s error names it, a directory of
    numba's under the user's cache directory, or a file numba writes in it, as the first save's does, and the
    directory is told; an error naming a location that is none of numba's, under the list and in its order, is told
    that location as the error names it, as for a source on disk, with nothing of what moves the user's cache
    directory and nothing of where numba caches the file, which that location is none of; the frozen application's is
    the no-locator one, which gives no reason, numba having passed
    the location over on its error, unwritable or too deep alike, so the user's cache directory is named as one numba
    could not use, to be made writable or put at a shorter path, or, where the name numba gives the directory after
    the source's is too long for the file system, told so, as is a cache file's name in that location, or in
    IPython's ``numba_cache`` for a member in an ipykernel directory, too long for the file system, which no shorter
    path cures; a frozen application's path with ``.zip`` in it and
    no part ending in it gets the ``.zip`` locator's error instead where the list puts that locator before the
    user-wide one, or once the user-wide one passed its location over, and the remedy is the user-wide locator listed
    before it, or that location made writable or put at a shorter path; an error that
    names no file, a caller's, or the first save's on a full disk, is told the locations numba could have taken, as
    for a source on disk, in its order and each once, with the reason, or, for a path too long, to put whichever is
    too long at a shorter path, and for a member in an ipykernel directory where a locator of IPython's family is
    reached nothing is said of the user's cache directory as where numba caches the file, IPython's ``numba_cache``
    being among them where IPython gives its directory. For
    any other archive, or a
    module without its source, the source files on disk or a ``.zip`` holding them, each with the locator it needs
    listed where ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves it out: one of the three for a file on disk for the first,
    the ``.zip`` locator for the second, and ``NUMBA_CACHE_DIR`` set where the user-provided locator is the one
    listed for a file on disk, which takes nothing without it. A ``.zip``'s location too long for the file system is
    the user's cache directory's doing, with the names numba makes bounded, and the remedy is that directory at a
    shorter path, through what moves it on the platform. numba caches a ``.zip`` through its ``.zip`` locator alone, so a
    ``NUMBA_CACHE_LOCATOR_CLASSES`` that leaves that locator out gives numba no locator for a source in a ``.zip``,
    and the remedy is to list it, for a member of an archive that is there, a directory named ``.zip`` above a source
    that is nowhere being no archive; so for a notebook cell and numba's IPython locator, and for a frozen application and
    numba's user-wide locator, which alone takes one;
    either is told for an error a caller passes on under such a list as for numba's no-locator one, since numba
    took no location under it, and a source neither in a ``.zip`` nor a frozen application's, which no locator
    takes, is told the source files on disk or a ``.zip`` holding them for any error.

    ``package`` is the one the remedy tells the reader to install again, at a shorter path or with its source
    files on disk: a package built on numbox that puts the question for its own files with
    ``check_cache_location`` and ``is_a_cache_error`` passes its own name, as it passes ``check_cache_location`` a
    ``longest_file_name`` for its own functions, ``LONGEST_CACHE_FILE_NAME`` being numbox's.

    From numba 0.68 the archive's stamp hashes the member's source, read from the archive by the name the
    module's code gives under the archive's path, and the ``KeyError`` zipfile raises for a name the archive does
    not hold is told the source files, on disk or in a ``.zip``, naming the member, a module shipped as ``.pyc``
    alone; on Windows under numba 0.68.0 every member gets that error, its name built with the platform's
    separator (numba issue 10889, fixed for 0.68.1), and the remedy is the install on disk, or a release carrying
    the fix, where the archive holds the member's source under the name it reads, else the source files as for any
    member. The
    first part of the path ending in ``.zip`` is what numba takes for the archive, a directory so named above it
    included, and from 0.68
    opens to read the source: the ``IsADirectoryError`` is told the directory, to put the archive under a path
    with none such above it; a file on disk under such a directory, which that locator takes once the ones before it
    passed the file over, is told to make one of their locations writable or to put the file elsewhere. A file in an
    ipykernel directory that numba's IPython locator takes, a cell file on disk
    or a ``.zip`` member, where IPython gives a new temporary directory as its cache directory on every call, able
    to write neither that directory nor its own, is told so for an error naming a file, as the one numba's first save
    raises there is ``FileNotFoundError`` naming the directory it could not save in, with ``IPYTHONDIR`` or
    ``XDG_CACHE_HOME`` at a writable path as the remedy, and an error of another errno naming a ``numba_cache`` is a
    caller's, told as the location it names: numba makes ``numba_cache`` in one such directory and saves in another;
    an error naming no
    file is told the locations numba could have taken, with its reason, as above.
    """
    from numba import config
    from numba.core.caching import _CacheLocator
    # check_cache_location takes any path-like, bytes too, and so does this:
    # the str the remedy reads, which os.fspath would leave bytes as bytes.
    py_file = os.fsdecode(py_file)
    # The error names the file refused as the call that raised it was given
    # it, bytes for one given bytes, or as a caller built it, a path object
    # too, and the remedy reads the name as str, as it reads the source's. An
    # empty str or bytes is no name; a path object built from an empty string
    # names the working directory, ".", as pathlib reads it; a name of another
    # kind, an integer file descriptor's as os.stat(fd) raises it, is none.
    filename = getattr(failure, "filename", None)
    filename = os.fsdecode(filename) if filename and isinstance(filename, (str, bytes, os.PathLike)) else None
    listed = getattr(config, "CACHE_LOCATOR_CLASSES", "")
    user_provided = _numba_locator("UserProvidedCacheLocator")
    in_tree = _numba_locator("InTreeCacheLocator")
    user_wide = _numba_locator("UserWideCacheLocator")
    for_ipython = _numba_locator("IPythonCacheLocator")
    for_a_zip = _numba_locator("ZipCacheLocator")

    def one_of(cls, base):
        # Whether ``cls`` is ``base`` or a subclass of it: a subclass, listed
        # by its dotted path, takes what its parent takes and caches where it
        # does, as numba's InTreeCacheLocatorFsAgnostic does, so each of
        # numba's locators stands for its family.
        return base is not None and issubclass(cls, base)

    def of_numbas(cls):
        # Whether ``cls`` is of one of numba's locator families, whose places
        # the remedy knows; a class of none of them, with a from_function and
        # a place of its own, caches where it alone can say.
        return any(one_of(cls, base) for base in (user_provided, in_tree, user_wide, for_ipython, for_a_zip))

    def could_have_taken(locations):
        # Which location numba took cannot be told from an error that names
        # no file, so the ones it could have taken, the values of
        # ``locations`` in numba's order, are listed each once: on disk,
        # NUMBA_CACHE_DIR set to the user's cache directory makes one path of
        # two. A class of none of numba's families has a place the remedy
        # cannot name, listed as the class's own.
        known = list(dict.fromkeys(
            location if location else f"{cls.__name__}'s own"
            for cls, location in locations.items() if location or not of_numbas(cls)))
        if len(known) > 1:
            return f"the location numba took, one of {', '.join(known[:-1])} or {known[-1]}"
        return f"the location numba took, {known[0]}"

    def listed_without(base):
        # Whether NUMBA_CACHE_LOCATOR_CLASSES is set and names no class of
        # ``base``'s family.
        return base is not None and bool(listed) and not any(one_of(cls, base) for cls in _locators(listed))

    try:
        subpath = _CacheLocator.get_suitable_cache_subpath(py_file)
    except UnicodeEncodeError:
        # numba hashes the source's directory path as UTF-8 for the name,
        # and raises for bytes that are not, at decoration, where a locator
        # that names its directory so reaches the file; asked for a file
        # none did, the remedy has no such location to name.
        subpath = None

    def too_long_a_name(location):
        # numba names the directory it caches the file in, under
        # NUMBA_CACHE_DIR or the user's cache directory, after the source's
        # directory, by its name and a hash of its path, and a name longer
        # than the file system holding ``location`` takes can be made at no
        # path, so no shorter one cures it: the clause that says so, or None
        # where the name fits, or numba has none to give.
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
        # The check's error for the file of the longest name the package's
        # functions get, which the file system holding ``location`` takes no
        # name as long as: the error carries the length, and the clause
        # says so, or None for any other error. The remedy read the error as
        # the location's and offered a shorter path, which cures nothing of
        # a name.
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
        # name and the function's qualified name, and a name longer than the
        # file system holding ``location`` takes can be made at no path, so
        # no shorter one cures it, a shorter name alone: the clause that says
        # so for the file the error names in ``location``, and whether a file
        # system in use takes the name at all, 255 being the limit of ext4,
        # NTFS and APFS alike, in the units each counts, for the longest name
        # numba writes for the function: the index, ".nbi", and a data file
        # per signature, ".<n>.nbc", numbered from one without bound, each
        # through a temporary name 21 longer, ".tmp." and sixteen hex digits,
        # the data file's with the digits reserved for its number, or the
        # number the error names where that is wider; or
        # None where the error names no such file, or its name fits. The
        # remedy asked for the package, the user's cache directory, IPython's
        # or the variable at a shorter path, which move nothing of the name,
        # then offered a file system that takes the name where none in use
        # does, never naming the shorter name, then one that takes the name
        # where numba's save through the temporary name failed on it, and
        # then one where the index fit and the data file did not.
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
        # is tried: numba's IPython locator a cell, or a file in an ipykernel
        # directory, on disk or not; its .zip locator a path with .zip in it,
        # which it caches where a part ends in .zip, an archive or a directory,
        # and raises for where none does; its user-wide one a frozen
        # application's file besides any on disk; the rest any file on disk;
        # and a class of none of numba's families, with a from_function and
        # a place of its own, what its from_function takes, asked with a
        # probe as numba asks it, where it caches being its own to say: the
        # remedy took such a class for taking any file, and told a file it
        # passed over as its own, where numba went on to the next listed.
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
        # Whether the error names a path above the file, the archive a
        # member's code names or a directory on the way, both read absolute:
        # a relative file against a name given absolute matched nothing, and
        # a member's moved archive was told as a location refusing a file.
        # The working directory is none: a path object built from an empty
        # string names it, which the remedy tells as a location that is none
        # of numba's, and it stood above a file under it as a moved archive.
        above = os.path.abspath(filename) if filename else ""
        return bool(above) and above != os.getcwd() and os.path.abspath(py_file).startswith(above + os.sep)

    def names_another_location():
        # Whether the error names a location of one of numba's locators for a
        # file on disk other than IPython's, or a file in it: NUMBA_CACHE_DIR's,
        # the __pycache__ beside the source, or the user's cache directory;
        # one that is none of them, a temporary directory IPython gave, is
        # IPython's locator's.
        if not filename:
            return False
        named = os.path.abspath(filename)
        if named == os.path.abspath(py_file):
            # The source itself, which numba reads: refused, it is told to be
            # made readable, whatever IPython gives for its directory.
            return True
        others = [os.path.abspath(config.CACHE_DIR)] if config.CACHE_DIR else []
        others.append(os.path.abspath(os.path.join(os.path.dirname(py_file), "__pycache__")))
        others.append(os.path.abspath(AppDirs(appname="numba", appauthor=False).user_cache_dir))
        return any(named == other or named.startswith(other + os.sep) for other in others)

    asked = []

    def ipython_numba_cache():
        # IPython's cache directory, asked for once: under a home it cannot
        # write IPython makes a temporary directory per call and warns.
        if not asked:
            asked.append(_ipython_numba_cache())
        return asked[0]

    def names_a_numba_cache():
        # Whether the error names IPython's location for numba, a numba_cache
        # directory, or a file in one, as not there: the error numba's first
        # save raises under an IPython giving a temporary directory per call
        # is FileNotFoundError, naming the numba_cache it could not save in,
        # a directory made in another, and so is the check's; a caller's
        # naming another location shows nothing of that locator, and was told
        # IPYTHONDIR for a cell on disk the in-tree locator took, and for a
        # member under such an IPython on any errno, a refused directory of
        # its own among them.
        return (
            bool(filename) and failure.errno == errno.ENOENT
            and "numba_cache" in os.path.abspath(filename).split(os.sep))

    if isinstance(failure, OSError) and names_a_numba_cache() and not names_a_path_above() and (
            _ipython_locator_reads(py_file)) and (
            not a_file_on_disk or not names_another_location()
    ) and _ipython_cache_is_temporary(ipython_numba_cache()):
        # numba's IPython locator, reached first for a member in an ipykernel
        # directory and for a cell file on disk there once the locators before
        # it passed the file over, asks IPython for its cache directory at
        # every save, and IPython, able to write neither that directory nor
        # its own, gives a new temporary directory on every call: numba makes
        # numba_cache in one and saves in another, dying there. The remedy
        # told the one the error names, a temporary directory, as refusing a
        # file, and for a file on disk offered NUMBA_CACHE_DIR beside it; an
        # error naming no file for a file on disk shows nothing of which
        # locator took it, and is told the locations numba could have taken.
        # inspect's error, with no errno and no name, came before the locator
        # tried any location, and is told as inspect's below, whatever
        # IPython gives: the remedy blamed the directory for a cell whose
        # source linecache does not hold, and for a member shipped as .pyc.
        # An error naming no file is told the locations numba could have
        # taken, with its reason, as the docstring says, whatever IPython
        # gives: the one numba's first save raises here names the directory
        # it could not save in, and a caller's with a reason and no name, a
        # full disk's, was told IPYTHONDIR with the reason dropped. An error
        # naming a path above the file, the archive a member's code names
        # where that has since moved, is the stamp's, and is told below; the
        # temporary directory's error names a numba_cache under it.
        # IPython reads XDG_CACHE_HOME for its cache directory on every POSIX
        # system from IPython 8, macOS included, which IPython 7 left out;
        # the remedy said Linux, and steered a macOS reader off the variable.
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
        # its location, and a class of none of numba's families in the order
        # took the file, asked, so none listed after either is reached: the
        # remedy read the locators after such a class as reached, and told an
        # error naming IPython's location as numba caching the file there.
        zip_at = next((at for at, cls in enumerate(order) if one_of(cls, for_a_zip)), None)
        stops_at = next((at for at, cls in enumerate(order) if one_of(cls, for_a_zip) or not of_numbas(cls)), None)
        reached = order if stops_at is None else order[:stops_at + 1]
        # The .zip locator listed after such a class is not reached, and the
        # errors it raises are a caller's under the list: the remedy told them
        # as the .zip locator's, the locators before it having passed the
        # file over, where one of them took it.
        zip_at = zip_at if zip_at == stops_at else None
        variable_at = next((at for at, cls in enumerate(order) if reads_the_variable(cls)), None)
        cache_dir_read = variable_at is not None

        def instead(taken, asks="a short path", shared=None, made="a writable directory at a short path"):
            # NUMBA_CACHE_DIR as the alternative to the location numba took,
            # offered where numba tries the variable before that location's
            # locator: set, it was passed over, unwritable or too deep, and
            # ``made`` is what it is asked to be made; unset,
            # it would be taken, and ``asks`` is what it is asked to be. None
            # is a location that is none of numba's, taken for the first in
            # the order other than the variable's own; it shows nothing of
            # what numba did with the variable, so a set one is named without
            # the claim that numba passed it over. ``shared`` is the last
            # locator numba reaches whose place the error names too, where
            # the user-wide and .zip locators give the file one path under
            # the user's cache directory: which of them took it the error
            # cannot show, so the variable is offered where numba tries it
            # before the last, without the claim; the remedy read the first
            # as the one, and left the variable out where it sat between.
            if subpath is None:
                # numba names its directory under the variable by hashing
                # the source's directory as UTF-8, and raises for one that is
                # not: the remedy offered the variable, and the import died
                # on it where the warning had stood.
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
            # under no directory named after the source's: the one of numba's
            # locators for a file on disk that a name too long leaves, made
            # writable where numba reaches it and it passed the file over, or
            # listed where NUMBA_CACHE_LOCATOR_CLASSES leaves it out, or
            # before the .zip locator where that one, listed first, took a
            # path with .zip in it and left it unreached; the offer to make
            # the __pycache__ writable went to a list numba never reached it
            # under.
            if any(caches_beside_the_source(cls) for cls in reached):
                beside = os.path.abspath(os.path.join(os.path.dirname(py_file), "__pycache__"))
                return (
                    f", or make the __pycache__ beside the source, {beside}, writable, where InTreeCacheLocator "
                    "caches it under no such name"
                )
            if listed:
                # Listed after the locator that took the file, the .zip one
                # or a class of none of numba's families, the in-tree locator
                # goes before it; the remedy named the .zip locator for any.
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
            # Where the class places the file: the .zip locator under the user's
            # cache directory like the user-wide one. None for a class that is
            # not numba's, whose place is its own, for the user-provided one
            # without NUMBA_CACHE_DIR, which takes nothing without it, and for
            # the IPython one without IPython, or where IPython raises OSError
            # for its directory, which numba passes the locator over on.
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
            # inspect's error, which no call of the file system's raised, so
            # with no errno and no name: numba's IPython locator reads the
            # function's source, from the file itself for one on disk, and
            # inspect raises where it reads none there, a file it cannot read
            # or, before Python 3.13, an empty one, before that locator tries
            # its location, the
            # ones numba tries before it having passed the file over; the
            # remedy told the location numba took, which it had not, as
            # refusing a file.
            return (
                f"numba's IPython locator reads the function's source from its file before trying its location, and "
                f"read none in {py_file} ({failure}): make the file readable, with the source in it; or {silence}"
            )
        if isinstance(failure, OSError) and filename and os.path.abspath(filename) == os.path.abspath(py_file):
            # numba reads the source itself: its IPython locator for the
            # function's source, and from 0.68 every locator for the stamp,
            # which hashes the file; a file that refuses to be read was told
            # as the location numba took refusing a file, to make writable.
            reason = failure.strerror or failure
            return (
                f"numba reads the function's source from its file, {py_file}, which cannot be read ({reason}): make "
                f"it readable; or {silence}"
            )
        # Under a list with no locator that takes the file, or none with a
        # place for it, the user-provided one alone with NUMBA_CACHE_DIR unset,
        # numba took no location, and the list is the remedy, below, for an
        # error a caller passes on as for numba's no-locator one; a class of
        # none of numba's families has a place of its own, which an error
        # can name.
        if isinstance(failure, OSError) and (any(locations.values()) or not all(of_numbas(cls) for cls in reached)):
            # The error names the location numba took, or a file numba writes
            # in it: numba's own check passed the location, its temporary file
            # fitting where its cache files would not, or the package's named
            # file was refused since, a full disk or permissions changed. The
            # locator that took it decides the remedy, the first in numba's
            # order whose place the error names: whole paths, since the places
            # can nest, an install under either cache directory or
            # NUMBA_CACHE_DIR above the user's; and in order, since
            # NUMBA_CACHE_DIR set to the user's cache directory makes one path
            # of two.
            if filename:
                named = os.path.abspath(filename)
                named = {named, os.path.dirname(named)}
            else:
                # A caller can pass on an error that names nothing, and numba's
                # first save raises one where a write fails, as on a full disk.
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
                # The variable, where numba tries it before the locator that
                # took the file and a file system in use takes the name,
                # pointed at a directory on one, as it is offered at a short
                # path for a path too long; the variable's own location took
                # the file only set, so it is named.
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
                    # and the first save overflows: numba keeps taking the
                    # variable as it is, so the user's cache directory at a
                    # shorter path, offered where the variable is set to it,
                    # moved nothing numba reads.
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
                # numba's .zip locator takes a file on disk for the ".zip" in
                # its path once the locators before it passed the file over,
                # takes the first part of the path ending in .zip for the
                # archive, a directory so named included, and from 0.68 opens
                # it to read the source, which a directory refuses; the remedy
                # told the reader to make room in that directory, no cache
                # location at all, with the remedy for a member alone knowing
                # the case; under a list with no .zip locator the error is a
                # caller's, and is told as the location it names.
                if zip_at:
                    reached_how = "reached once the locators before it passed this file over"
                    cure = f"make one of those locators' locations writable{instead(None, 'a writable directory')}, or "
                else:
                    # Listed first, the .zip locator takes the file before any
                    # locator for a file on disk; the remedy spoke of locators
                    # before it that were not there.
                    reached_how = "which NUMBA_CACHE_LOCATOR_CLASSES puts before every locator for a source file on disk"
                    cure = "list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or " \
                           "UserWideCacheLocator before it, or "
                return (
                    f"numba's .zip locator, {reached_how}, takes the first part of the path ending in .zip, {filename}, "
                    "for an archive, and it is a directory, which numba 0.68 and later open for the source: "
                    f"{cure}put the file under a path with no such directory; or {silence}"
                )
            # numba took the location, so the variable was not passed over for
            # it; set elsewhere it moves the cache off a disk that is full. A
            # location that is none of numba's is named as the error names it,
            # and an error that names none is told the ones numba could have
            # taken.
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
        # numba itself passes a location it cannot make or write over, for a
        # source on disk, so the error here is the no-locator one, every
        # locator tried and passed over, or the .zip locator's for a path with
        # .zip in it and no part ending in it, which numba raises where it
        # reaches that locator: the locators before it passed the file over,
        # and none after it was tried.
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
            # Every locator numba tried, of those it reaches, a class listed
            # after the .zip locator being unreached for a path that locator
            # takes, that names its directory after the source's found the
            # name too long for its file system, the
            # variable's by that of the directory it is set to or, unset, by
            # the limit nearly every file system has, and passed the file
            # over for that, which no writable directory at a short path
            # cures; the remedy asked for one, and, where the .zip locator
            # then raised for a path with .zip in it and no part ending in
            # it, asked for the variable without a word of the name.
            too_long = too_long_a_name(locations[named_after_the_source[0]])
            # Under the user-provided locator alone, with the variable unset,
            # numba took no location at all and never measured the name: the
            # variable is what it takes nothing without, and the name is what
            # would stop it once set; the remedy said the name alone.
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
            # and the locator that names it so raises: the variable's, set,
            # before the in-tree locator is reached, which caches beside the
            # source and hashes no path, or the user-wide one once the in-tree
            # locator passed an unwritable __pycache__ over. The remedy said
            # the variable cures nothing and asked for a UTF-8 path alone,
            # where the variable unset, the same install imports cached;
            # under a list without the variable's locator it told the list,
            # and its locations to make writable, which hashing refuses
            # whatever they are. numba hashed where the error is its
            # UnicodeEncodeError, or where a locator that names its
            # directory so was tried: the variable's, set, or the user-wide
            # one; the in-tree locator alone hashes nothing, and its
            # no-locator error is the __pycache__ refused.
            hashed = (
                "numba hashes the source's directory as UTF-8 for the directory it caches this file in under "
                f"NUMBA_CACHE_DIR or the user's cache directory, and {os.path.dirname(py_file)} is not"
            )

            # The locators that hash: the variable's, set, the user-wide one,
            # and the .zip one for a path with .zip in it. The in-tree one
            # caches beside the source, in its __pycache__, hashing no path:
            # listed after one that hashes, it is never reached, and the
            # remedy asked for its __pycache__ to be made writable, which it
            # was, or for it to be listed, after the one that hashes, where
            # numba still raises; the cure is the in-tree locator before the
            # first that hashes, the variable unset where its locator alone
            # is that one, or the __pycache__ made writable where the
            # in-tree locator came first and passed it over.
            def hashes(cls):
                return ((reads_the_variable(cls) and bool(config.CACHE_DIR)) or caches_under_the_user_cache_dir(cls)
                        or (one_of(cls, for_a_zip) and ".zip" in py_file))

            beside = os.path.abspath(os.path.join(os.path.dirname(py_file), "__pycache__"))
            in_tree_at = next((at for at, cls in enumerate(order) if caches_beside_the_source(cls)), None)
            before_in_tree = [cls for cls in (order if in_tree_at is None else order[:in_tree_at]) if hashes(cls)]
            # Whether the in-tree locator takes the file, asked as numba asks
            # it: the __pycache__ it can write; the remedy hedged "where that
            # directory can be written" and never asked for it, so the
            # variable unset as told left numba raising the same error.
            if in_tree is not None and _takes(in_tree, py_file):
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
            # Set and passed over: unwritable, or too deep for numba to make
            # its directory there, which a writable one as deep would not cure.
            return (
                f"NUMBA_CACHE_DIR is set to {config.CACHE_DIR}, which numba could not use: set it to a writable "
                f"directory at a short path, or {silence}"
            )
        return f"Set NUMBA_CACHE_DIR to a writable directory, or {silence}"

    def the_source_files():
        # The source files, on disk or in a .zip, each with the locator it
        # needs listed where NUMBA_CACHE_LOCATOR_CLASSES leaves it out.
        on_disk = f"install {package} with its source files on disk, unpacked from any archive"
        if listed:
            # The install caches through a locator for a file on disk the list
            # names: none, and it caches no more than the archive does; the
            # user-provided one alone, and that takes nothing without
            # NUMBA_CACHE_DIR.
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
            "import it from a .zip holding its source files, which numba 0.61 and later cache in the user's cache "
            "directory"
        )
        found = _archive_above(py_file)
        if found is not None and for_a_zip is not None and not found[0].endswith(".zip"):
            # The file is in an archive already, APP.ZIP or an .egg, which
            # numba's .zip locator never takes, looking for a part of the path
            # ending in ".zip", lower-case: the remedy offered a .zip holding
            # the source files to a reader importing from one, and never named
            # the extension numba does not read; and it held the rename back
            # where a directory above the archive had ".zip" in its name, the
            # locator taking the file for that and finding no archive.
            # numba takes the first part of the path ending in .zip for the
            # archive, so a directory so named above it is in the way, and the
            # rename alone left numba opening the directory: the remedy
            # offered the rename under one.
            archive_parts = pathlib.Path(found[0]).parts
            in_the_way = next(
                (str(pathlib.Path(*archive_parts[:at + 1])) for at, part in enumerate(archive_parts[:-1])
                 if part.endswith(".zip")), None)
            moved = (
                f", and put it under a path with no part ending in .zip above it, {in_the_way} being one, which that "
                "locator takes for the archive first" if in_the_way else "")
            a_zip = (
                f"rename the archive, {found[0]}, to end in .zip, lower-case, which numba's .zip locator looks for in "
                f"the path, with its source files in it{moved}, so that numba 0.61 and later cache it in the user's "
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
        # any member, the name built with the platform's separator (numba
        # issue 10889, fixed in 0.68.1). The remedy told the archive's user
        # to add source files that were there; and one that read a backslash
        # as the separator's doing alone told a module shipped as .pyc alone
        # that numba 0.68.1 reads the archive as it is, which holds no
        # source for it under the name that release reads either.
        member = _missing_member(failure)
        held = member.replace("\\", "/")
        on_disk, a_zip = the_source_files()
        if os.name == "nt" and "\\" in member and _numbas_archive_holds(py_file, held):
            # The install on disk is a cure now; a numba release with the fix,
            # none out as of 0.68.0, reads the member as the archive names it,
            # and is named by the fix, not as a release to install.
            return (
                f"numba 0.68.0 reads a .zip member's source by a name built with Windows's separator, {member}, which "
                "no archive holds (numba issue https://github.com/numba/numba/issues/10889, fixed for numba 0.68.1): "
                f"to cache, {on_disk}, which no separator touches, or a numba release that carries the fix, which "
                f"reads the member as the archive names it; or {silence}"
            )
        naming = _pyc_naming(py_file)
        if naming is not None and naming[1]:
            # The archive holds the module's source, beside a .pyc compiled
            # to name another path in the archive, which zipimport runs and
            # numba reads the source by: the remedy told the module as
            # shipped as .pyc alone, and asked for source files that were
            # there.
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
    # The locators numba reaches for a file not on disk, in its order: its
    # .zip locator takes a file without trying its location, and a class of
    # none of numba's families in the order took the file, asked, so none
    # listed after either is reached; its IPython and user-wide ones pass a
    # location they cannot make or write over, and the next is tried.
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
        # Where the class places the file: the IPython locator in
        # numba_cache under IPython's cache directory, None without
        # IPython, or where IPython raises OSError for the directory, which
        # numba passes the locator over on; IPython is asked for the
        # directory where a locator of that family is reached and for no
        # other file, since asked for any .zip member it warned, with its
        # own directory unwritable, on every import of the package, and
        # left a temporary directory behind; the .zip and user-wide ones in
        # a directory of their own under the user's cache directory; a class
        # of none of numba's families in a place of its own, None here.
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
        # numba's user-wide locator alone of numba's takes a frozen
        # application, so a list without it leaves numba no locator of its
        # own for one: its no-locator error, whatever classes of the reader's
        # own the list holds, those having passed the file over, and any
        # error a caller passes on where the list has no place for the file;
        # the remedy asked for a writable directory numba never tried.
        return (
            "numba caches a frozen application through UserWideCacheLocator alone, and "
            f"NUMBA_CACHE_LOCATOR_CLASSES, {listed}, leaves it out: list it, or {silence}"
        )
    # Under a list with no locator that takes the file, or none with a place
    # for it, a class of none of numba's families having one of its own,
    # numba took no location, and the list is the remedy, below, for an error
    # a caller passes on as for numba's no-locator one, as for a source on
    # disk.
    parts = pathlib.Path(py_file).parts
    in_a_zip = any(part.endswith(".zip") for part in parts)
    # A directory named .zip above the file, which numba's .zip locator takes
    # for an archive all the same, and caches nothing under; a cell under one,
    # whose source linecache does not hold, was told the archive remedy, the
    # source files to ship.
    a_directory_named_zip = any(
        os.path.isdir(os.path.join(*parts[:at + 1])) for at, part in enumerate(parts) if part.endswith(".zip"))
    # A member of any archive, an .egg or .whl as a .zip, which zipimport runs
    # and whose source inspect reads through its loader; the .zip is the one
    # numba's .zip locator takes, and the only one it caches.
    in_an_archive = (in_a_zip and not a_directory_named_zip) or _archive_above(py_file) is not None
    # A caller's error with no errno and no name looked like inspect's and
    # was told to make the file readable; inspect's are known by their words.
    inspects = _is_inspects(failure) and any(one_of(cls, for_ipython) for cls in reached)
    if inspects and not in_an_archive:
        # inspect's error, with no errno and no name, for a cell whose source
        # linecache does not hold: numba's IPython locator reads a cell's
        # source from linecache, where IPython holds it while it runs the
        # cell, and inspect raises before the locator tries its location;
        # the remedy told the location numba took as refusing a file. For a
        # member of an archive that holds no source for it, a .pyc alone,
        # inspect's error gets the archive remedy below, the source files to
        # ship; a member of an .egg got the cell's, the archive unknown by
        # its name.
        return (
            f"numba's IPython locator reads a cell's source from linecache, which holds none for {py_file} "
            f"({failure}), and numba caches a cell only while IPython holds its source; or {silence}"
        )
    not_utf8 = _member_not_utf8(py_file) if inspects else None
    if not_utf8:
        # inspect's error for a member of an archive that holds its source,
        # not UTF-8: zipimport decodes a member's source as UTF-8 and raises
        # for one that is not, and the probe's loader read it as zipimport
        # does; the error came out of the check as its own, no cache error,
        # and the import died where numba would die at decoration.
        return (
            f"numba's IPython locator reads the member's source as UTF-8, which {py_file} is not ({not_utf8}): ship "
            f"it as UTF-8; or {silence}"
        )
    hashed_by = next((cls for cls in reached if (one_of(cls, user_wide) and getattr(sys, "frozen", False))
                      or (one_of(cls, for_a_zip) and in_a_zip)), None)
    if isinstance(failure, UnicodeEncodeError) and hashed_by is not None:
        # numba's .zip locator names its directory under the user's cache
        # directory after the source's directory, the archive's path in it,
        # hashed as UTF-8, and raises for bytes that are not, at decoration
        # and in the check alike, as the user-wide locator does for a frozen
        # application's file; the remedy, taking the error for a caller's
        # with no place for the file, offered the .zip the reader was
        # importing from, or the source files on disk, and said nothing of
        # the path. The first of the two numba reaches is the one that
        # raised: the user-wide locator, before the .zip one in numba's
        # order, for a frozen application under a directory named .zip,
        # which the remedy told as an archive.
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
            # ".zip" in its path, without trying its location, and raises
            # where no part of the path ends in .zip, an archive or a
            # directory: listed before the user-wide locator, which alone
            # takes the file, it leaves that one unreached, and the remedy is
            # to list it before the .zip one; after it, the user-wide one
            # passed its location over, unwritable or too deep, and that
            # location is the remedy, as for a source on disk.
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
        # it, as the first save's does, and the locator that took it decides
        # the remedy, the first reached whose place the error names; the
        # frozen application's no-locator error names nothing.
        taken = next((cls for cls in reached if locations[cls] and locations[cls] in named), None)
        if taken is not None and one_of(taken, for_ipython):
            # numba caches the member in numba_cache under IPython's cache
            # directory: the remedy a cell file on disk gets there, with no
            # NUMBA_CACHE_DIR to offer for a source not on disk.
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
        # cache directory, is what a reader can act on, named by the error or
        # the directory of the file it names; an error that names no file is
        # told the locations numba could have taken, IPython's numba_cache
        # among them where a locator of that family is reached; the frozen
        # application's no-locator error names nothing, and the user's cache
        # directory is told.
        if taken is not None:
            location = filename if os.path.abspath(filename) == locations[taken] else os.path.dirname(filename)
        else:
            location = filename or user_cache_dir
        # Where numba caches the file: a .zip's or a frozen application's
        # location is under the user's cache directory; a location that is
        # none of numba's is not, and an error that names no file for a
        # member a locator of IPython's family, or a class of none of numba's
        # families, is reached for may be that one's, with IPython's
        # directory given or not, so nothing is said of that directory for
        # any of them.
        in_the_user_cache_alone = taken is not None or (
            not filename and all(of_numbas(cls) and not one_of(cls, for_ipython) for cls in reached))
        where = (
            "numba caches a .zip, or a frozen application, in the user's cache directory, and "
            if in_the_user_cache_alone else "")
        opening = f"{where}NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: "
        if isinstance(failure, FileNotFoundError) and names_a_path_above():
            # The stamp numba reads at decoration, of the archive the code
            # names: .pyc members compiled to name an archive since moved. A
            # location that cannot be made, under a dangling link, is ENOENT
            # too, and is not under the code's file.
            suggestion = "" if numba_version >= 68 else "compile the archive's .pyc members to name its path now, or "
            return (
                f"the module's code names {filename}, which is not there: its .pyc was compiled to name "
                f"that path, so {suggestion}ship its source files; "
                f"or {silence}"
            )
        if (isinstance(failure, OSError) and failure.errno == errno.EISDIR and filename and os.path.isdir(filename)
                and filename.endswith(".zip") and names_a_path_above()):
            # numba takes the first part of the path ending in .zip for the
            # archive, a directory so named above it included, and from 0.68
            # opens it to read the member's source, which a directory
            # refuses; the remedy told the reader to make room in it.
            return (
                f"{opening}numba takes the first part of the path ending in .zip, {filename}, for the archive, and it "
                "is a directory, which numba 0.68 and later open to read the member's source: put the archive under "
                f"a path with no such directory above it, or {silence}"
            )
        if isinstance(failure, OSError) and names_a_path_above():
            # The archive the member's code names, which numba reads for the
            # stamp, and from 0.68 opens for the member's source, and cannot:
            # the remedy told it as a location refusing a file, to be made
            # writable, which reads nothing.
            reason = failure.strerror or failure
            return (
                f"{opening}numba reads the archive, {filename}, for the member's stamp, and from 0.68 for the "
                f"member's source, and cannot ({reason}): make it readable, or {silence}"
            )
        if isinstance(failure, OSError) and filename and taken is None:
            # A location that is none of numba's, under the list and in its
            # order, which an error a caller passes on can name: told as the
            # error names it, as for a source on disk, with nothing of what
            # moves the user's cache directory, which moves nothing of it, and
            # nothing of where numba caches the file, which the location is
            # none of: a member numba's IPython locator takes, with IPython
            # not importable, numba caches nowhere.
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
            # after the source's, which that directory at a shorter path
            # cures nothing of, refused whatever the errno, EINVAL on
            # Windows; the remedy asked for one.
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
                # all where no locator of IPython's family is reached.
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
            # permissions, which the error gives, as the remedy for a source on
            # disk gives it; the frozen application's no-locator error gives
            # none, numba having passed the location over on it.
            reason = failure.strerror or failure
            if taken is None:
                return (
                    f"{opening}no file can be written in {could_have_taken(locations)} ({reason}), so make room "
                    f"there, or make it writable, or {silence}"
                )
            # numba's .zip locator takes the file without making its
            # location, and the first save makes it, or the check does, under
            # the first directory above it that is there, which refused: the
            # remedy asked for room in a directory that was not there, or for
            # it to be made writable, and never named the one refusing. An
            # error naming a file in the location found the location there.
            made_under = ""
            if os.path.abspath(filename) == locations[taken] and not os.path.isdir(location):
                above = _existing_above(location)
                if not os.path.isdir(above):
                    # A file in the way, which the remedy passed over for the
                    # directory above it, writable, and asked for room in.
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
        # numba having passed the user-wide location over on its error, an
        # unwritable directory or one too deep to make alike, or one whose
        # name, after the source's directory, is too long for the file
        # system, which is told where it is, no shorter path curing it.
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
        # numba's no-locator error for a file its IPython locator alone of
        # numba's takes, a cell or a .zip member in an ipykernel directory
        # under a list with no .zip locator after it: the locator passed its
        # location over, as numba's user-wide one does a frozen application's,
        # unwritable or too deep to make, or IPython raised for its directory
        # and gave none; a class of none of numba's families listed with it
        # passed the file over too, numba having found no locator, and is
        # named, where the remedy, taking it for one of numba's, answered
        # with the archive remedy; asked, it is in the order no longer.
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
        # locator for one; the archive remedy offered source files on disk and
        # locators that take none of these.
        return (
            f"numba caches a notebook cell through IPythonCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, {listed}, "
            f"leaves it out: list it; or {silence}"
        )
    if in_a_zip and not a_directory_named_zip and listed_without(for_a_zip):
        # numba's .zip locator alone takes a source in a .zip, so a list
        # without it leaves numba no locator for one, and the archive remedy
        # offered the .zip the reader was importing from; a source that is
        # nowhere under a directory named .zip was told to list it too, where
        # that locator takes the directory for an archive and caches nothing.
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
    """``options`` as given, or with ``cache`` off after one warning where numba can write no cache for numbox.

    numba sets a cached function up when it is decorated and raises ``RuntimeError`` there when no cache location
    can be written: a read-only install whose user cache directory cannot be written either, or an import from an
    ``.egg``, ``.whl`` or ``.pyz`` archive, which Spark's ``--py-files`` ships. For a ``.zip`` it takes the user's
    cache directory without checking at decoration that it can be written, and the first write raises ``OSError``
    instead. Either way the import died at the first decorated function, and nothing named the way out.

    Every function numbox caches decorates under the one ``jit_options``, so the question is put here, once, for
    a source file in each directory of the package that holds a module, and answered for the package. Where
    numba can cache no function of one of those files the options come back with ``cache`` off and one warning
    names the remedy: ``NUMBA_CACHE_DIR`` for a source file on disk, and where the location is too long for the
    file system what shortens the one it is, as ``cache_remedy`` sets out; for a ``.zip``, the location numba took
    under the user's cache directory made writable or given room, with the reason, or that directory at a shorter
    path where the path is too long, and the ``.zip`` locator to list where ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves it
    out, numba reading ``NUMBA_CACHE_DIR`` only for a source file on disk; for any other archive,
    or a ``.pyc``-only install, or a ``.pyc`` in a ``.zip``, the source files on disk or a ``.zip`` holding them,
    which numba 0.61 and later cache in the user's cache directory, and so for a ``.zip`` member the archive holds
    no source for, a ``.pyc`` alone, from numba 0.68, which reads the member's source from the archive for the
    stamp, the member named; on Windows under numba 0.68.0 every member is such, its name built with the
    platform's separator (numba issue 10889), and the remedy is the install on disk, or a numba release carrying the
    fix. ``NUMBOX_JIT_OPTIONS='{"cache": false}'``
    turns caching off and silences the warning. An error that is not the cache's, as ``is_a_cache_error`` draws
    the line, is raised as it was.

    A ``.zip`` whose cache directory holds every entry but can no longer be written takes the fallback too, where
    numba alone would have loaded the entries: the writability check is the rule numba applies to every other
    placement, and the one the ``.zip`` locator is missing, reported as https://github.com/numba/numba/issues/10888.

    Options without a ``cache`` key, as ``NUMBOX_JIT_OPTIONS`` can give, leave ``njit`` to its default, off, but
    the sqlite virtual-table and table-valued-function callbacks cache under them, so they are asked about too and
    come back with ``cache`` off, said, where the answer is no.
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

#: Number of slots in the `FunctionModel` struct: (c_addr, py_addr, jit_addr) on numba 0.61
#: and later, (addr, pyaddr) before it. Read off the data model rather than inferred from the
#: numba version, so the layout is discovered rather than assumed.
function_struct_size = _function_model.field_count


def function_struct_has(field: str) -> bool:
    """Whether numba's `FunctionModel` carries a slot named `field`.

    `get_field_position` is numba's own accessor for the question and raises `KeyError` when
    the slot is absent, which is how a numba predating it reads.
    """
    try:
        _function_model.get_field_position(field)
    except KeyError:
        return False
    return True
