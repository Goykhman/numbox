numbox.core.configurations
==========================

Overview
++++++++

Every function numbox caches is decorated under one set of numba options, ``jit_options``, read once from
the ``NUMBOX_JIT_OPTIONS`` environment variable when this module is first imported; a bare ``@njit`` in
numbox, as in ``lowlevel.py``, ``meminfo.py`` and ``make_vector``, is not cached and takes none. The value is a JSON
object passed to ``@njit`` as keyword arguments, and any other shape is refused by name, as is a ``cache``
that is not ``true`` or ``false`` (the string ``"false"`` is true to numba, which reads the option's truth);
unset means ``{"cache": true}``, so numbox compiles into numba's on-disk cache by default, and
``export NUMBOX_JIT_OPTIONS='{"cache": false}'`` turns that off.

.. _where_the_cache_lands:

Where the cache lands
+++++++++++++++++++++

numba writes a cached function's entries under ``NUMBA_CACHE_DIR`` when that is set, else into the
``__pycache__`` directory beside the function's source file, else into the user's cache directory, taking the
first of those it can write. It sets the cache up when the function is decorated, so with caching on the question
of where it lands is settled at import. The user's cache directory is ``~/Library/Caches/numba`` on macOS,
``%LOCALAPPDATA%\numba\Cache`` on Windows, and ``$XDG_CACHE_HOME/numba`` elsewhere, ``~/.cache/numba`` when
``XDG_CACHE_HOME`` is unset.

Where no cache can be written, numba raises at decoration, and numbox's import used to die on its first module
with ``RuntimeError: cannot cache function ...: no locator available for file ...``. The two placements that do
that are a read-only install whose user cache directory cannot be written either, and an import from an
``.egg``, ``.whl`` or ``.pyz`` archive, which Spark's ``--py-files`` ships. Since every function numbox caches
decorates under the one ``jit_options``, numbox puts the question once, when this module is imported, and
answers it for the package: for a probe in each directory of the package that holds a module, since numba's
in-tree cache is a ``__pycache__`` beside each source, it runs the cache set-up numba runs at decoration and
the writability check numba runs at the first save, compiling nothing, and where either fails for any
directory numbox compiles without a cache and one ``RuntimeWarning`` names the remedy. Every directory
counts, whether or not its modules cache anything, so the answer errs toward uncached, which is never wrong.
A module that survives as ``.pyc`` alone is asked by the file it was compiled from, which its code keeps and
numba looks up: the ``.py`` that is gone where it was compiled in place, or a tree elsewhere, on disk or not.
For a ``.zip``, which numba caches per directory of the archive, each in a location of its own under the
user's cache directory, the archive's directories are listed and the question put for one module of each; a
``.pyc`` in the archive that zipimport would run, which it takes before the ``.py`` beside it unless it is
stale against it or of another interpreter, and whose code keeps the file it was compiled from, is asked by
that file, since that is what numba looks up for it, on disk or gone, and one zipimport would pass over is
passed over, and a ``.pyc`` whose code names a member the archive does not hold, one alone with no ``.py`` beside
it or one compiled to name another path in the archive, ``compileall -d``, beside its ``.py`` or not, is asked for
itself whichever member stands for its directory, since numba 0.68 and later read the member's source from the
archive by the name its code gives and find none for it; any other archive has no location at all. A directory of
the package reached through a symlink
is walked like the rest, wherever the link points. The check makes the cache directories it asks about, as
numba would at the first decoration in each; with caching beside the sources that is an empty ``__pycache__``
per directory of the package, a linked one included. numba's own writability check makes a temporary file,
one without a name on Linux, and the files it saves have names of a hundred bytes and more, so a location
within their length of the path limit, 4096 on Linux, passes numba's check and the first save overflows;
the check here makes a file named as long as the longest numba writes for the package's files (128 bytes,
which a test holds every function of the package under), so that location turns caching off with the
warning instead.

- For a source file on disk that numba cannot read, which its IPython locator reads for the function's source and
  every locator reads for the stamp from numba 0.68, the remedy is to make it readable. For one it can, the remedy
  is ``NUMBA_CACHE_DIR`` pointed at a writable directory; where it is
  set and numba passed it over, the warning names it and asks for a writable directory at a short path, since
  numba passes over one too deep to make its directory in as it does an unwritable one; and where
  ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves the user-provided locator out, it is the locations
  the list names made writable, numba never reading the variable. Where the
  location is too long for the file system, the remedy is for the location the error names: numba takes a
  directory under ``NUMBA_CACHE_DIR`` where that is set, else the ``__pycache__`` beside the source, else a
  directory under the user's cache directory, each of the two under a cache directory named for the source's
  directory, by its name and a hash of its path; the path the error names is matched whole against each location
  a locator numba reaches gives it, in the order numba tries them, since the three can nest and
  ``NUMBA_CACHE_DIR`` set to the user's cache directory makes one path of two, and the error is read as that
  locator's, the ones numba tries before it having passed the file over, as numba's own errors mean and an error a
  caller passes on naming a location numba did not take need not; numba reaches no locator listed
  after its ``.zip`` one for a path with ``.zip`` in it, which that locator takes without trying its location, so
  a location one of those would give is none of numba's. So a shorter ``NUMBA_CACHE_DIR`` for the first; the
  package installed at a shorter path for the second; the user's cache directory at a shorter path, through
  ``XDG_CACHE_HOME`` or ``HOME``, ``HOME`` alone on macOS and nothing on Windows, where numba asks the system,
  for the third; and for either of the last two ``NUMBA_CACHE_DIR`` set to a short path, where numba tries it
  before the locator that took the location, or, where it is set and numba passed it over, unwritable or too
  deep, the warning names it and the remedy is a writable directory at a short path. The directory numba names after
  the source's, under ``NUMBA_CACHE_DIR`` or the user's cache directory, is that directory's name and forty-one
  bytes, and one longer than a name can be on the file system holding the location, which the system is asked for,
  255 where it cannot say, in bytes, or in the UTF-16 units Windows and macOS count a name in, can be made at no
  path, and is refused whatever the errno, EINVAL on Windows: an error naming such a location, numba's no-locator
  error
  for a file on disk where every locator it tried that names its directory so found the name too long for its file
  system, the variable's unset by that limit, and a frozen application's are told the name and its length, and the
  file in a directory of a shorter name, with, for a file on disk, the ``__pycache__`` beside the source made
  writable where the in-tree locator is listed, or that locator listed where it is not, which caches under no such
  name. A cache file numba could not make or open in the location it took, whose name, given after the source's name
  and the function's qualified name, is longer than a name can be on that file system, is told the name and its
  length, which no shorter path cures, and a shorter name for the function or its file, the package's to give,
  and the check's error for a name as long as the package's functions get, which that file system takes no name
  as long as, is told the length, with
  ``NUMBA_CACHE_DIR`` pointed at a directory on a file system that takes a name that long offered where a file
  system in use takes the name, 255 being their limit, less the 29 of the longest name numba writes for the
  function, a data file's with three digits for its number, numba numbering them from one without bound, through
  the temporary name it writes each file through,
  and numba tries the variable before the locator that took
  the file, as it is offered at a short path for a path too long, and nothing offered for a ``.zip`` member, which
  the variable does nothing for.
  numba hashes the source's directory path as UTF-8 for that name and raises for bytes that are not, at
  decoration, where a locator that names its directory so reaches the file, so for such a path the remedy names no
  location under ``NUMBA_CACHE_DIR`` or the user's cache directory, and a ``.zip`` under such a path, whose
  archive's path numba's ``.zip`` locator hashes the same way for its directory under the user's cache directory,
  is told the archive under a UTF-8 path, a frozen application under one, whose path numba's user-wide locator
  hashes the same way, the application under a UTF-8 path, and a file on disk under one is told the
  ``__pycache__`` beside it too,
  which numba's in-tree locator hashes no path for: that locator to list before the first that hashes, where the
  list leaves it out or puts it after one, ``NUMBA_CACHE_DIR`` to unset where its locator alone hashes before the
  in-tree one, or that directory to make writable where the in-tree locator came first and passed it over, and
  with any of these where it cannot be written.
  A location that is none of
  numba's is named as the error names it, with the variable, where numba tries it before any other locator, named
  as set or asked for at a short path, and nothing said of numba passing it over, which that location cannot show.
  The name an error carries is read as a string, bytes or a path object alike, and a name of another kind, an integer
  file descriptor's, as none. An error can name no file, or carry an
  empty string or empty bytes as its name. A caller can build such an error, and numba's first save raises one where a
  write fails, as on a full disk. So does inspect where, for numba's IPython locator, it reads no source in a cell file
  on disk, unreadable or, before Python 3.13, empty, with no errno either, no call of the file system's having raised
  it, and that error,
  for a file on disk that a locator of IPython's family numba reaches takes, is told as inspect's, that locator having
  tried no location: the file to make readable, with the source in it; for a cell, whose source numba reads from
  linecache, where IPython holds it while it runs the cell, as that, numba caching a cell only while IPython holds
  its source; and for a ``.zip`` member whose archive holds no source, a ``.pyc`` alone, as the archive's, its source
  files to ship. Every other OSError that ``check_cache_location``
  raises names a file. An error with no name, or an empty
  one, is told the locations numba could have taken, in numba's order and each once. For a path too long, the remedy is
  to put whichever of them is too long at a shorter path. For a file refused for another reason, such as a full disk,
  the remedy gives the reason and asks for room or a writable directory there. A path object is never empty: pathlib
  reads an empty string as the working directory, which the remedy names as a location that is none of numba's.
  ``NUMBA_CACHE_LOCATOR_CLASSES`` decides that order, each entry a class of numba's caching module or, by its
  dotted path, a subclass of one, which takes what its parent takes, caches where it does and is told as it is, or a
  class of none of numba's families, with a ``from_function`` and a place of its own, which takes what its
  ``from_function``, asked with a probe as numba asks it, takes, where it caches being its own to say: an error
  naming a location that no locator of
  numba's gives the file is told that location as the error names it, which is then that class's, and one naming no
  file is told the location numba took as the class's own, with no path to name, among numba's locators' places where
  those are reached; an
  entry numba cannot resolve, which it refuses before any location is tried, or raises importlib's error for, a
  relative module path's TypeError, and one that names no class, or a class with no ``from_function``, which it
  fails on at a function it decorates where it reaches the entry, are no locators here, and numba's own error for
  either is raised as it was.
  numba's IPython locator caches a cell file it takes in ``numba_cache`` under IPython's cache directory, with no
  directory per file, and reads the function's source when it takes the file, as inspect reads it, a file on disk
  itself, a cell from linecache and a ``.zip`` member through the module's loader, which the check's probe has, a
  module of its own whose loader reads the member from the archive as zipimport's does, the member's lines then in
  linecache as any reader leaves them; that location too long is told as
  IPython's cache directory at a shorter path, for a cell file on disk and for a ``.zip`` member in an ipykernel
  directory alike, where that locator is listed before the ``.zip`` one, which takes a file without trying its
  location, so that none listed after it is reached, with no ``NUMBA_CACHE_DIR`` offered for a source not on
  disk; a cell, which no other locator of numba's takes, gets numba's no-locator error where that locator passes its
  location over, and is told that location, to make writable or put at a shorter path, or IPython's cache directory
  to make a home for where IPython raised for it, with a class of none of numba's families listed with it named as
  having passed the cell over too; the remedy asks IPython for that directory where a locator of that family is
  listed
  and takes the file, and leaves IPython alone for every other file, a plain ``.zip`` member among them, since
  IPython warns when asked under a home it cannot write and leaves a temporary directory behind, a new one on
  every call where it can write neither its cache directory nor its own, in which case numba makes ``numba_cache``
  in one and saves in another, the check raises the error numba's first save would, ``FileNotFoundError`` naming
  the directory, an error of another errno naming a ``numba_cache`` being a caller's, told as the location it
  names, and the warning asks for
  ``IPYTHONDIR``, or ``XDG_CACHE_HOME`` on a POSIX system, where IPython 8 and later read it, at a writable path;
  numba
  imports IPython for the location when it makes it and catches only OSError there, so where IPython is not
  importable it raises ImportError at decoration for a file that locator takes, no cache error, and an error a
  caller passes on naming that location is told it as none of numba's where another locator numba reaches has a
  place for the file, and the list where none has, as numba's no-locator error is told it; and where IPython raises
  OSError for its directory numba passes the locator over, as it does a location it cannot make, and the warning,
  asked the same, has no location of IPython's to match the error against. numba's IPython locator takes a file on
  disk
  only in an ipykernel directory and its ``.zip`` locator only a path with ``.zip`` in it, so either ahead of
  the rest changes nothing for any other file; the ``.zip`` locator ahead of them all takes such a path first and
  caches it under the user's cache directory where a part of the path ends in ``.zip``, an archive or a
  directory, or, where none does, finds no archive
  in it, and the warning then asks for a locator for a file on disk listed before it, the user-provided one with
  ``NUMBA_CACHE_DIR`` set, which alone it takes nothing without; after some of them, it finds no archive once
  those have passed the file over, numba trying none after it, and the warning asks for one of their locations
  made writable or such a locator listed before it. A list with no locator that takes the file is told so, for an
  error a caller passes on under it as for numba's no-locator one, and so is one whose only locator for the file is
  the user-provided one with ``NUMBA_CACHE_DIR`` unset, which takes nothing without it: the variable to set, as
  numba's no-locator error is told there. Where the location numba took refuses a file for another reason, a
  full disk or permissions changed since numba's own check, the warning names the location and the reason and
  asks for room or a writable directory there, with ``NUMBA_CACHE_DIR`` as the alternative where numba tries it
  before that location, or set to another directory where the location is the variable's own; numba passes over
  a location it cannot make or write in, so only its no-locator error means the variable was passed over. An
  error naming a location that is none of numba's is told that location as the error names it, and one naming
  no file is told the locations numba could have taken.
- For a ``.zip``, or a frozen application, it is the user's cache directory made writable, or room made there,
  with the reason the error gives, a full disk or permissions: a ``.zip`` is the one archive numba caches
  and it caches it there, taking the directory without checking that it can be written; the error names
  that location, or a file numba's first save writes in it, and the location is told with the reason; an error
  naming a location that is none of numba's, under the list and in its order, is told that location as the error
  names it, as for a source on disk, with nothing of what moves the user's cache directory and nothing of where
  numba caches the file, which that location is none of; a frozen
  application (``sys.frozen``) is cached there too, its sources not being on disk, and its error is numba's
  no-locator one, which gives no reason, numba having passed the location over on its error, unwritable or too deep
  alike, so that directory is named as one numba could not use, to be made writable or put at a shorter path, or,
  where the name numba gives the directory after the source's is too long for the file system, told so, as is a
  cache file's name in that location, or in IPython's ``numba_cache`` for a member in an ipykernel directory, too
  long for the file system, which no shorter path cures.
  A frozen application's path with ``.zip`` in it and no part ending in it gets the ``.zip`` locator's error instead,
  where the list puts that locator before the user-wide one or once the user-wide one passed its location over, and
  the warning asks for the user-wide locator listed before it, or for that location made writable or put at a
  shorter path.
  numba reads ``NUMBA_CACHE_DIR`` only for a source file on disk, so the variable changes nothing for either.
  An error that names no file, a caller's, or numba's first save's on a full disk, is told the locations numba
  could have taken, as for a source on disk, in its order and each once, with the reason, or, for a path too long,
  to put whichever is too long at a shorter path; for a member in an ipykernel directory where a locator of
  IPython's family is reached, the warning says nothing of the user's cache directory as where numba caches the
  file, and IPython's ``numba_cache`` is among them where IPython gives its directory.
  A ``.zip`` whose cache directory holds every entry but can no longer be written falls back too, where numba
  alone would have loaded the entries: the writability check is the rule numba applies to every other
  placement. numba caches a ``.zip`` through its ``.zip`` locator alone, so a ``NUMBA_CACHE_LOCATOR_CLASSES``
  that leaves that locator out gives numba no locator for a source in a ``.zip``, and the warning asks for it to
  be listed; so for a notebook cell and numba's IPython locator, and for a frozen application and numba's
  user-wide locator, which alone takes one; any of them is told
  for an error a caller passes on under such a list as for numba's no-locator one, since numba took no location
  under it, and a source neither in a ``.zip`` nor a frozen application's, which no locator takes, is told the
  source files on disk or a ``.zip`` holding them for any error.
- For an ``.egg``, ``.whl`` or ``.pyz``, a ``.pyc``-only install or a ``.pyc`` in a ``.zip``, it is the source
  files on disk or a ``.zip`` holding them, each with the locator it needs listed where
  ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves it out: one of the three locators for a file on disk for the first, the
  ``.zip`` locator for the second, and ``NUMBA_CACHE_DIR`` set where the user-provided locator is the one listed for
  a file on disk, which takes nothing without it.
- From numba 0.68 the archive's stamp hashes each member's source, read from the archive by the name the
  module's code gives under the archive's path: a member shipped as ``.pyc`` alone is told the source files, on
  disk or in a ``.zip``, naming the member; on Windows under numba 0.68.0 every member is refused, its name
  built with the platform's separator (`numba issue 10889 <https://github.com/numba/numba/issues/10889>`_,
  fixed for 0.68.1), and the warning asks for the install on disk, or a release carrying the fix; and a directory
  whose name
  ends in ``.zip`` above the
  archive, which numba takes for the archive by the first such part of the path and from 0.68 opens for the
  source, is told as the directory it is, to put the archive under a path with none such above it; a source
  file on disk under such a directory, which that locator takes once the locators before it passed the file
  over, is told to make one of their locations writable or to put the file elsewhere.
- ``NUMBOX_JIT_OPTIONS='{"cache": false}'`` turns caching off and silences the warning in every case, the
  package's options being what it sets; the anchors' warning under a caller's own options, below, is
  silenced by those.

A function numba cannot cache is compiled in every process that uses it, never wrong; that is the cost the
warning reports. An error at decoration that is not the cache's is raised as it was.

A package built on numbox can put the same question for its own files: ``check_cache_location`` asks it for
one file, ``is_a_cache_error`` tells numba's cache errors from the rest, and ``cache_remedy`` words the
remedy, given the package's name for the two remedies that tell the reader to install it again, at a shorter
path or with its source files on disk. The bound on file names is the package's own:
``LONGEST_CACHE_FILE_NAME`` holds numbox's functions only, so a package passes ``check_cache_location`` the
longest name numba writes for its own, as ``longest_file_name``, or a location that fits numbox's names and not
the package's passes the check and the first save overflows it.

The code numbox generates at run time, ``make_structref``'s, ``compile_kernel``'s, the work builder's derives
and the sqlite registrations', is anchored to a file under ``NUMBA_CACHE_DIR`` or the user's cache directory
so that numba can cache it. That directory can be unwritable where the package's own functions cache, since
those cache beside their sources, so each anchor puts the same question for its own file when it is written,
and the code it names compiles without a cache where the answer is no, after a warning of the same shape (the
builder's derive falls back without one, as it did). ``make_structref``, ``compile_kernel`` and the builder
take jit options of the caller's, which the variable does not reach, and ``compile_kernel``'s ``cache``
argument overrides those too, so that warning's silence is ``cache`` off in the options the code was given,
the argument where it takes one, or the variable where the options are the package's. ``make_graph``'s
kernel is anchored to the builder's own file and cached beside it, so it puts the question for that file
under the options it was given, and falls back the same way with the package's remedy for the placement.
See the cache-anchor section of :doc:`numbox.utils`.

Modules
++++++++

numbox.core.configurations
--------------------------

.. automodule:: numbox.core.configurations
   :members:
   :show-inheritance:
   :undoc-members:
