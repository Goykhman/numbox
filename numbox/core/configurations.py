import errno
import importlib.machinery
import inspect
import os
import json
import sys
import tempfile
import warnings
import zipfile
import zipimport

from importlib.metadata import version

import numba.experimental.function_type  # noqa: F401  registers `FunctionModel` against `FunctionType`
from numba.core.caching import CompileResultCacheImpl
from numba.core.datamodel import default_manager
from numba.misc.appdirs import AppDirs
from numba.core.types import FunctionType, void


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


def check_cache_location(py_file, longest_file_name=0):
    """Raise as numba would where a function whose source is ``py_file`` cannot be cached; else return.

    The question is put the way numba puts it: the cache set-up that decoration runs, which picks the location
    for the file or raises ``RuntimeError`` with no locator, then the source's stamp, which decoration reads
    and which stats the archive for a ``.zip``, then the writability check, which decoration runs for every
    location but a ``.zip``'s and the first save runs for all, raising ``OSError``. The probe is compiled with
    ``py_file`` as its file, which is all a locator reads of it. Nothing is compiled, and nothing is written
    but the cache directory itself. An ``OSError`` from the check names the location numba picked.

    numba's writability check makes a temporary file whose name is short, or none at all on Linux, and the
    files it saves have names of up to a hundred bytes and more, so a location within their length of the path
    limit passes the check and the first save dies. Given ``longest_file_name``, a file with a name that long
    is made and removed in the location too, and the ``OSError`` is the location's: the package asks with
    ``LONGEST_CACHE_FILE_NAME``, the bound on numba's names for its own files.
    """
    namespace = {}
    exec(compile("def _cache_probe():\n    pass\n", os.fspath(py_file), "exec"), namespace)  # nosec B102 - fixed source
    probe = namespace["_cache_probe"]
    locator = CompileResultCacheImpl(probe).locator
    # numba reads the source's stamp at decoration too, the archive's for a
    # .zip, which is not there where the code names an archive since moved.
    locator.get_source_stamp()
    try:
        locator.ensure_cache_path()
        if longest_file_name:
            # tempfile's name is the prefix and eight characters of its own.
            with tempfile.NamedTemporaryFile(dir=locator.get_cache_path(), prefix="x" * (longest_file_name - 8)):
                pass
    except OSError as error:
        # The error names what failed, a temporary file's name among the
        # possibilities; the location it is in is what a reader can act on.
        raise OSError(error.errno, error.strerror, locator.get_cache_path()) from error


def is_a_cache_error(error):
    """Whether ``error`` is numba's for a cache it cannot set up: no locator, or a location it cannot use.

    numba itself passes over a location on any ``OSError`` from making its directory or writing a file there,
    permission denied, a read-only file system, a path into a file, a component too long, a full disk, so any
    ``OSError`` counts. So does the ``ValueError`` numba raises for an archive under a directory whose name
    holds ``.zip``: its ``.zip`` locator takes the file by that substring and then finds no ``.zip`` in it.
    """
    if isinstance(error, OSError):
        return True
    if isinstance(error, ValueError):
        return "No zip file found" in str(error)
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
    """One module per directory of numbox inside the ``.zip`` that holds ``own``, this module as imported, and the
    file each ``.pyc`` in it that zipimport would run was compiled from; nothing for any other archive.

    numba caches a ``.zip`` per directory of it, each in a location of its own under the user's cache directory,
    so the directories answer separately there too. A ``.pyc`` run from the archive keeps the file it was compiled
    from as its code's file, and that is what numba looks up for its functions, the archive's own path for the
    module is not; so such a member asks by that file, there or gone. The archive is the first part of the path
    named ``.zip`` that is one, a directory so named above it being no archive. Any other archive has no location
    at all, and the probe's own file has already asked.
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
    compiled to name the archive, and stands for the directory as a ``.py`` member does; else it is the file
    elsewhere the ``.pyc`` was compiled from, there or gone, and asks for itself. No code at all, for whatever
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


def cache_remedy(py_file, failure, silence):
    """The remedy for ``failure``, numba's for a function whose file is ``py_file``, ending in ``silence``.

    ``NUMBA_CACHE_DIR`` for a source file on disk, which is the only kind numba reads the variable for; where its
    location is too long for the file system, a shorter ``NUMBA_CACHE_DIR`` or none, since each location numba
    picks for a source on disk but the one beside it appends the source's directory path, else the package at a
    shorter path. For a ``.zip`` or a frozen application, both cached under the user's cache directory, the
    location made writable:
    the ``.zip``'s error names it, a directory of numba's under the user's cache directory; the frozen
    application's is the no-locator one, numba having passed the location over on its error, so the user's
    cache directory is named. For any other archive, or a module without its source, the source files on disk
    or a ``.zip`` holding them. A ``.zip``'s location too long for the file system is the user's cache
    directory's doing, with the names numba makes bounded, and the remedy is that directory at a shorter
    path, through ``XDG_CACHE_HOME`` or ``HOME``.
    """
    if os.path.exists(py_file):
        if isinstance(failure, OSError) and failure.errno == errno.ENAMETOOLONG:
            # numba's own check passes the location, its temporary file
            # fitting where its cache files would not; the error names it.
            return (
                "the path is too long for the file system: a shorter NUMBA_CACHE_DIR, or none, where it is set, "
                f"else numbox installed at a shorter path; or {silence}"
            )
        # numba itself passes a location it cannot make or write over, for a
        # source on disk, so the error here is the no-locator one.
        return f"Set NUMBA_CACHE_DIR to a writable directory, or {silence}"
    if isinstance(failure, OSError) or getattr(sys, "frozen", False):
        # The .zip's error names the location numba picked, which is under
        # the user's cache directory; the frozen application's names nothing.
        location = getattr(failure, "filename", None) or AppDirs(appname="numba", appauthor=False).user_cache_dir
        if isinstance(failure, FileNotFoundError) and py_file.startswith(failure.filename + os.sep):
            # The stamp numba reads at decoration, of the archive the code
            # names: .pyc members compiled to name an archive since moved. A
            # location that cannot be made, under a dangling link, is ENOENT
            # too, and is not under the code's file.
            return (
                f"the module's code names {failure.filename}, which is not there: its .pyc was compiled to name "
                "that path, so compile the archive's .pyc members to name its path now, or ship its source files; "
                f"or {silence}"
            )
        if isinstance(failure, OSError) and failure.errno == errno.ENAMETOOLONG:
            return (
                "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR "
                f"has no effect here, because the source is not a file on disk: the path is too long for the file "
                f"system, so put that directory, {location}, at a shorter path, through XDG_CACHE_HOME or HOME, or "
                f"{silence}"
            )
        return (
            "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR "
            f"has no effect here, because the source is not a file on disk: make that directory, {location}, "
            f"writable, or {silence}"
        )
    return (
        "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, "
        "install numbox with its source files on disk, unpacked from any archive, or import it from a "
        f".zip holding its source files, which numba 0.61 and later cache in the user's cache directory; or "
        f"{silence}"
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
    names the remedy: ``NUMBA_CACHE_DIR`` for a source file on disk, or a shorter one, or none, where the
    location is too long for the file system; for a ``.zip``, the user's cache directory made writable, since
    numba reads ``NUMBA_CACHE_DIR`` only for a source file on disk; for any other archive,
    or a ``.pyc``-only install, or a ``.pyc`` in a ``.zip``, the source files on disk or a ``.zip`` holding them,
    which numba 0.61 and later cache in the user's cache directory. ``NUMBOX_JIT_OPTIONS='{"cache": false}'``
    turns caching off and silences the warning. An error that is not the cache's, as ``is_a_cache_error`` draws
    the line, is raised as it was.

    A ``.zip`` whose cache directory holds every entry but can no longer be written takes the fallback too, where
    numba alone would have loaded the entries: the writability check is the rule numba applies to every other
    placement, and the one the ``.zip`` locator is missing.

    Options without a ``cache`` key, as ``NUMBOX_JIT_OPTIONS`` can give, leave ``njit`` to its default, off, but
    the sqlite virtual-table and table-valued-function callbacks cache under them, so they are asked about too and
    come back with ``cache`` off, said, where the answer is no.
    """
    if not options.get("cache", True):
        return options
    for py_file in _module_files():
        try:
            check_cache_location(py_file, LONGEST_CACHE_FILE_NAME)
        except (RuntimeError, OSError, ValueError) as error:
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


numba_version = int(version("numba").split(".")[1])
assert numba_version >= 60, numba_version

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
