numbox.core.configurations
==========================

Overview
++++++++

numba `caches <https://numba.readthedocs.io/en/stable/user/jit.html#cache>`_ a compiled function only in a directory
where it can write the cache files, and it picks that directory when the function is decorated. If it finds none, the
decoration raises a ``RuntimeError``, and so does the import of the module that holds the function. numbox decorates
many functions with caching enabled as its modules are imported. Its first import would then fail wherever numba has
no writable location for numbox's files: in a read-only install whose user cache directory is read-only too, for
instance, or in an import from an ``.egg`` archive.

This module prevents that failure. It holds ``jit_options``, the options that numbox passes to numba's ``njit``
when it compiles its own functions. It also holds the check, the function
:func:`~numbox.core.configurations.uncached_where_no_cache_can_be_written`, which runs when this module is first
imported. The check asks numba whether it could cache a function in each directory of numbox. If the answer is no for
any one directory, numbox disables caching in ``jit_options`` and warns once, naming what to change. numbox's functions
then compile without a cache in every process, with the same results.

A package built on numbox can run the same check on its own files, as `Checking a package built on numbox`_ shows.

Setting the options
-------------------

numbox reads ``jit_options`` from the ``NUMBOX_JIT_OPTIONS`` environment variable once, when
``numbox.core.configurations`` is first imported. The variable holds a JSON object, which numbox passes to ``njit`` as
keyword arguments. If the variable is unset, the object is ``{"cache": true}``, and ``jit_options`` is the Python
dictionary ``{"cache": True}``. To disable caching, set the variable before Python starts, or at least before numbox
is first imported:

.. code-block:: sh

    export NUMBOX_JIT_OPTIONS='{"cache": false}'

numbox passes other keys, as in ``{"cache": true, "boundscheck": true}``, unchanged to ``njit`` for the functions that
it compiles with these options. Keep ``"cache"`` in any value you set. Without it, most of numbox's functions compile
without a cache, because ``njit`` does not cache by default. numba still caches two kinds of function in that case.
:func:`~numbox.core.variable.compile_kernel.compile_kernel` enables caching for its kernels when no option names
``"cache"``. numbox's SQLite bindings also compile C callbacks, the functions that SQLite calls for numbox's virtual
tables and table-valued functions. numbox compiles those callbacks with no option other than ``"cache"``. It reads that
option from ``jit_options`` and treats caching as enabled when the key is absent.

numba tests the cache option for truth, so the string ``"false"`` in ``{"cache": "false"}`` would leave caching enabled.
numbox therefore refuses a ``"cache"`` value that is not ``true`` or ``false``, with a ``ValueError`` that names the
variable. It refuses a value that is not a JSON object in the same way.

Functions of your own can compile under the same options, so that one variable controls caching for numbox's functions
and for yours. The file ``totals.py`` below defines one such function:

.. code-block:: python

    import numpy as np
    from numba import njit

    from numbox.core.configurations import jit_options


    @njit(**jit_options)
    def total(xs):
        acc = 0.0
        for x in xs:
            acc += x
        return acc


    print(jit_options, total(np.arange(4.0)))

``python totals.py`` prints ``{'cache': True} 6.0`` and caches ``total``; a second run loads ``total`` from the cache.
When the check disables caching for numbox, ``total`` is not cached either, even where numba could cache ``totals.py``.
`Checking a package built on numbox`_ shows how to run the check on ``totals.py`` instead, so that numba caches
``total`` wherever it can cache that file.

.. _where_the_cache_lands:

Where numba writes the cache
++++++++++++++++++++++++++++

numba stores a cached function in an index file, whose name ends in ``.nbi``, and in one data file for each compiled
signature, ending in ``.nbc``. When the function's source is a file on disk, numba writes them in the first of three
locations that it can write in. numba calls the class that chooses a location a
`cache locator <https://numba.readthedocs.io/en/stable/developer/caching.html#cache-locators>`_, and each location has
its own locator:

- a subdirectory of ``NUMBA_CACHE_DIR``, if that
  `environment variable <https://numba.readthedocs.io/en/stable/reference/envvars.html#envvar-NUMBA_CACHE_DIR>`_ is
  set, chosen by ``UserProvidedCacheLocator``;
- the ``__pycache__`` directory beside the source file, chosen by ``InTreeCacheLocator``;
- a subdirectory of the user's cache directory, chosen by ``UserWideCacheLocator``. The user's cache directory is
  ``~/Library/Caches/numba`` on macOS, ``%LOCALAPPDATA%\numba\Cache`` on Windows, and ``$XDG_CACHE_HOME/numba``
  elsewhere, or ``~/.cache/numba`` if ``XDG_CACHE_HOME`` is unset.

numba names each subdirectory after the source file's directory: the directory's name, an underscore and a SHA-1 hash
of the directory's path, such as ``core_df4387379757d214fdc85570af6767de3b7850dc``.

numba picks the location when the function is decorated. It skips a location that it cannot create or write in, and it
issues no warning. If ``NUMBA_CACHE_DIR`` cannot be written in, numba therefore uses the next location. If no
location is left, the decoration raises ``RuntimeError: cannot cache function 'f': no locator available for file ...``.
The variable `NUMBA_CACHE_LOCATOR_CLASSES
<https://numba.readthedocs.io/en/stable/reference/envvars.html#envvar-NUMBA_CACHE_LOCATOR_CLASSES>`_ can reorder the
locators or leave some out.

Three other kinds of source have locations of their own:

- a function defined in an IPython or Jupyter cell: ``numba_cache`` under IPython's cache directory, chosen by
  ``IPythonCacheLocator``;
- a module imported from a ``.zip`` archive: a subdirectory of the user's cache directory, chosen by
  ``ZipCacheLocator``;
- a module of a frozen application, such as one built with
  `PyInstaller <https://pyinstaller.org/en/stable/runtime-information.html>`_: a subdirectory of the user's cache
  directory, chosen by ``UserWideCacheLocator``.

``NUMBA_CACHE_DIR`` does not change the location of any of these, because numba reads it only for a source file on disk.
A source of any other kind has no location at all: a module in an ``.egg`` archive, for instance, or one present only as
a ``.pyc`` file.

Reading the warning
-------------------

When ``numbox.core.configurations`` is first imported, the check asks numba where it would cache a function defined in
one file from each directory of numbox. If numba could not cache a function defined in one of those files, numbox
compiles all of its functions without a cache and warns once, as here:

.. code-block:: text

    RuntimeWarning: numba cannot cache numbox here (cannot cache function '_cache_probe': no locator available for
    file '/opt/python/lib/python3.12/site-packages/numbox/core/configurations.py'); it compiles without a cache. Set
    NUMBA_CACHE_DIR to a writable directory, or NUMBOX_JIT_OPTIONS='{"cache": false}' to turn caching off and silence
    this warning

The warning has three parts. The parentheses contain the error that stopped the check, here numba's own: no locator
accepted the file. In that error, ``_cache_probe`` is the probe, a stand-in function that the check builds to ask numba
about a file; `How numbox checks`_ describes it. The last sentence is the remedy, which says what to change so that
numba can cache. Every remedy ends with a clause that suggests disabling caching instead. Following that advice silences
the warning, because the check does not run when ``"cache"`` is ``false``.

Each case below says first what to do, and then quotes its remedy so that you can match it to your warning.

If numbox is installed read-only
--------------------------------

Set ``NUMBA_CACHE_DIR`` to a writable directory in the environment, before Python starts:

.. code-block:: sh

    export NUMBA_CACHE_DIR=/tmp/numba-cache

numba then caches numbox's functions, and yours, under that directory. On a machine that other users share, choose a
directory that only you can write in, because numba loads and runs the compiled code that it finds there.

The warning under `Reading the warning`_ is this case, and its remedy is the fragment
``Set NUMBA_CACHE_DIR to a writable directory``. The case arises where the directories that hold numbox are read-only
to you, as in a system Python or a shared environment. numba cannot write in the ``__pycache__`` beside numbox's
sources. It uses the user's cache directory instead, and it issues no warning. numbox warns only if that directory
cannot be written in either, as with a read-only home directory on some cluster nodes and in some containers.

If ``NUMBA_CACHE_DIR`` is set already, change it to a writable directory with a short path. numba skips a directory that
it cannot write in. It also skips a directory whose path is so long that the subdirectory it would create there exceeds
the path limit. The remedy then names the directory, and it begins:

.. code-block:: text

    NUMBA_CACHE_DIR is set to /scratch/numba-cache, which numba could not use: set it to a writable directory at a
    short path

If ``NUMBA_CACHE_LOCATOR_CLASSES`` lists locators without ``UserProvidedCacheLocator``, numba never reads
``NUMBA_CACHE_DIR``. Make one of the listed locations writable, or add ``UserProvidedCacheLocator`` to the list. The
remedy repeats the value of ``NUMBA_CACHE_LOCATOR_CLASSES``, here ``InTreeCacheLocator,UserWideCacheLocator``, and
begins:

.. code-block:: text

    numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, InTreeCacheLocator,UserWideCacheLocator, says: make one of
    those locations writable

If numbox is imported from an archive
-------------------------------------

If numbox comes from an archive whose name does not end in ``.zip``, such as a ``.whl``, an ``.egg`` or a ``.pyz``,
install numbox with its source files on disk. Or rename the archive so that its name ends in the lower-case ``.zip``,
with numbox's source files in it. Python can `import <https://docs.python.org/3/library/zipimport.html>`_ modules from a
zip archive of any name, but numba caches a module from an archive only if the archive's name ends in ``.zip``. The
``.egg`` case arises with Spark, which ships ``.zip`` and ``.egg`` files to its executors through the ``--py-files``
option of `spark-submit
<https://spark.apache.org/docs/latest/submitting-applications.html#bundling-your-applications-dependencies>`_. The
remedy for this case begins:

.. code-block:: text

    NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, install numbox with its
    source files on disk, unpacked from any archive, or rename the archive, /jobs/numbox-0.0.0-py3.12.egg, to end in
    .zip, lower-case, which numba's .zip locator looks for in the path, with its source files in it, so that numba
    caches it in the user's cache directory

For a ``.zip``, numba caches each directory of the archive in a subdirectory of the user's cache directory, named as
above, and ``NUMBA_CACHE_DIR`` has no effect. If your home directory is read-only, make the user's cache directory
writable, or set its location to a writable directory. ``XDG_CACHE_HOME`` or ``HOME`` sets its location on Linux, and
``HOME`` sets it on macOS. On Windows, numba queries the system for the location, and no environment variable changes
it. The remedy names both actions regardless of the error. It therefore asks for free space even when, as here, the
error is ``Permission denied``. It begins:

.. code-block:: text

    numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no effect
    here, because the source is not a file on disk: no file can be written in that directory,
    /home/user/.cache/numba/core_1544a931bcaa3f414cb218856f1bf59ef75bcc9f, which cannot be made under /home/user
    (Permission denied), so make room there, or make it writable

numba uses a ``.zip``'s location without testing that it can be written in. Without the check, numba's first save
would raise an ``OSError``. numbox compiles some of its functions as its modules are imported, so the import would
fail; a function that numba compiles lazily fails at its first call instead. The check runs numba's test on that
location, as numba runs it for every other kind of location.

The check is stricter than numba in one case. numbox warns for a ``.zip`` whose location still holds every cache entry
but can no longer be written in, although numba by itself would load those entries. numba's ``.zip``
locator lacks the test, as `numba issue 10888 <https://github.com/numba/numba/issues/10888>`_ reports.

If numbox is bundled with PyInstaller
-------------------------------------

A program bundled with PyInstaller imports numbox from an archive inside the bundle. numba recognizes such a frozen
application by ``sys.frozen``, which PyInstaller sets, and caches its functions in the user's cache directory. If
numba cannot use that directory, make it writable, or set ``XDG_CACHE_HOME`` or ``HOME`` so that the directory has a
shorter path. Up to its first colon, the remedy is the same as the remedy for a ``.zip``. After that colon, it reads:

.. code-block:: text

    numba could not use that directory, /home/user/.cache/numba: make it writable, or put it at a shorter path,
    through XDG_CACHE_HOME or HOME

If numba 0.68 cannot read an archive member
-------------------------------------------

A file inside a ``.zip`` is called a member. From version 0.68, numba reads the source of each member when it sets up
the member's cache. It hashes that source into the stamp, the value that numba stores with a cache entry to
tell whether the source has changed. Three cases fail at that step:

- A directory whose name ends in ``.zip`` precedes the archive in its path, as ``/jobs.zip`` does in
  ``/jobs.zip/numbox.zip``. Put the archive elsewhere. numba treats the first part of the path that ends in ``.zip``
  as the archive, here the directory ``/jobs.zip``, and opening a directory fails. On Linux, the remedy contains the
  words ``put the archive under a path with no such directory above it``.
- On Windows, numba 0.68.0 builds each member's name with backslashes and finds no member of that name, so it can cache
  no module in a ``.zip``. Upgrade numba, or install numbox with its source files on disk. The bug is
  `numba issue 10889 <https://github.com/numba/numba/issues/10889>`_, fixed in numba 0.68.1.
- A member shipped as a ``.pyc`` file alone has no source in the archive. The section
  `If numbox is installed without its source files`_ describes that case.

In IPython and Jupyter
----------------------

numbox's own files are not cells, so numba caches numbox's functions in a notebook as in any
other process. For functions you define in a cell, numba writes the cache files in ``numba_cache`` under IPython's
cache directory, one directory for all cells. On Linux, and on macOS from IPython 8, IPython's cache directory is
``ipython`` under ``XDG_CACHE_HOME``, or under ``~/.cache``, if that base directory exists and can be written in.
Otherwise it is IPython's own directory: ``~/.ipython``, or the directory that
`IPYTHONDIR <https://ipython.readthedocs.io/en/stable/development/config.html#configuration-file-location>`_ names.
numba reads the function's source when it sets up the cache, so it caches a cell's function only while IPython holds
the cell's source.

If IPython can write in neither its cache directory nor its own directory, set ``IPYTHONDIR``, or ``XDG_CACHE_HOME`` on
Linux or macOS, to a writable directory before IPython starts. Otherwise the first call of a cached function in a cell
fails with a ``FileNotFoundError``. IPython warns first:

.. code-block:: text

    UserWarning: IPython parent '/home/user' is not a writable location, using a temp directory.

IPython then gives numba a new temporary directory each time numba asks for the cache directory. numba asks more than
once, so it creates ``numba_cache`` in one temporary directory and then saves in another, where no ``numba_cache``
exists, and the save fails.

numbox itself never runs the check on a cell. A package that runs it on a function that its user defined in a cell,
as `Checking a package built on numbox`_ shows, gets a remedy for this case, which begins:

.. code-block:: text

    numba caches this file in numba_cache under IPython's cache directory, and IPython, able to write neither that
    directory nor its own, gives a new temporary directory on every call, so numba makes numba_cache in one and saves
    in another: set IPYTHONDIR, or XDG_CACHE_HOME on a POSIX system, to a writable directory

Two other remedies concern cells only. If IPython no longer holds a cell's source, the remedy says that numba caches a
cell only while IPython holds its source. If ``NUMBA_CACHE_LOCATOR_CLASSES`` does not list ``IPythonCacheLocator``, the
remedy asks for it to be listed, because no other locator accepts a cell.

If numbox is installed without its source files
-----------------------------------------------

If numbox is installed from ``.pyc`` files alone, install it with its ``.py`` files, or import it from a ``.zip`` that
holds them; ``NUMBA_CACHE_DIR`` does not help. A ``.pyc`` file records the path of the source it was compiled from, and
numba places the cache by that recorded path. When no file exists at that path, no locator accepts it. A single module
without its source makes the check disable caching for all of numbox. numba's error in the warning names that module's
missing ``.py`` file. The remedy begins:

.. code-block:: text

    NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, install numbox with its
    source files on disk, unpacked from any archive, or import it from a .zip holding its source files, which numba
    caches in the user's cache directory

A ``.pyc`` file compiled from a tree that is still on disk elsewhere records a path in that tree, and numba caches the
module as though it ran from there.

In a ``.zip``, put the ``.py`` files in the archive, or install numbox with its source files on disk. A ``.pyc`` member
can record its own path in the archive, as the
`-d <https://docs.python.org/3/library/compileall.html#cmdoption-compileall-d>`_ option of ``compileall`` sets it.
numba 0.68 and later read the member's source from the archive at that path, and a member shipped as a ``.pyc`` file
alone has no source there. The error in the warning is then zipfile's
``There is no item named 'numbox/core/bindings/libm.py' in the archive``, and the remedy begins:

.. code-block:: text

    numba 0.68 and later read a .zip member's source from the archive, and this one holds none under
    numbox/core/bindings/libm.py, the module shipped as .pyc alone: to cache, install numbox with its source files on
    disk, unpacked from any archive, or import it from a .zip holding its source files, which numba caches in the
    user's cache directory

If a path or a name is too long
-------------------------------

If a location lies so close to the path limit, 4096 bytes on Linux, that numba's cache file names no longer fit, shorten
the location that numba chose. The remedy contains the words ``path is too long for the file system`` and names what to
shorten, according to the location that numba chose:

- under ``NUMBA_CACHE_DIR``: a shorter ``NUMBA_CACHE_DIR``;
- the ``__pycache__`` beside numbox's sources: numbox installed at a shorter path, or ``NUMBA_CACHE_DIR`` set to a
  short path;
- under the user's cache directory: that directory at a shorter path, set through ``XDG_CACHE_HOME`` or ``HOME``, or
  ``NUMBA_CACHE_DIR`` set to a short path;
- under the user's cache directory for a ``.zip``: that directory at a shorter path, set the same way, while
  ``NUMBA_CACHE_DIR`` has no effect.

numba's own test of a location passes such a location, so without the check numba would fail later with
``File name too long``, as `How numbox checks`_ explains.

A name can be too long regardless of its path. If the file system that holds the location cannot store a file name of
128 bytes, no shorter path helps: set ``NUMBA_CACHE_DIR`` to a directory on a file system that allows such names. The
remedy then contains the words ``so no shorter path cures it``.

The 128 bytes bound the cache file names that numba writes for numbox's own functions, numba's temporary suffix
included, and a numbox test checks that they stay under it. The names of generated code fit as well.
:func:`~numbox.utils.highlevel.make_structref` generates a numba ``StructRef`` type, called a struct below, from a name
and a list of fields. It shortens long names in the code that it writes. `If generated code cannot be cached`_ describes
that code.

If a path is not UTF-8
----------------------

On Linux, a file name can hold bytes that are not UTF-8. numba names its subdirectory under ``NUMBA_CACHE_DIR`` or the
user's cache directory with a hash of the source directory's path, encoded as UTF-8. For a path that is not UTF-8,
that encoding raises a ``UnicodeEncodeError``. The ``__pycache__`` beside the source has no hash in its name, so it is
the one location left.

For numbox installed under such a path, unset ``NUMBA_CACHE_DIR`` and make sure that numba can write in the
``__pycache__`` beside numbox's sources; or install numbox under a UTF-8 path. With the variable set, the remedy says
that ``NUMBA_CACHE_DIR`` "is hashed for first". These words mean that numba tries the variable first and
fails to hash the path for it. That remedy begins:

.. code-block:: text

    numba hashes the source's directory as UTF-8 for the directory it caches this file in under NUMBA_CACHE_DIR or the
    user's cache directory, and /srv/caf\udce9/numbox/core is not, and NUMBA_CACHE_DIR, which is set to
    /scratch/numba-cache, is hashed for first: unset it, and numba caches beside the source, in its __pycache__,
    /srv/caf\udce9/numbox/core/__pycache__, hashing no path; or put the file under a UTF-8 path

If the variable is unset and the install is read-only, the remedy begins:

.. code-block:: text

    numba hashes the source's directory as UTF-8 for the directory it caches this file in under NUMBA_CACHE_DIR or the
    user's cache directory, and /srv/caf\udce9/numbox/core is not: put the file under a UTF-8 path, or make the
    __pycache__ beside the source, /srv/caf\udce9/numbox/core/__pycache__, writable, where InTreeCacheLocator caches
    it hashing no path

If generated code cannot be cached
----------------------------------

numbox also generates Python source at run time and compiles it. A warning that names a generated file, such as
``Struct_ce9e24b31bee6866.py``, comes from that code:

.. code-block:: text

    RuntimeWarning: numba cannot cache Struct_ce9e24b31bee6866.py here ([Errno 20] Not a directory:
    '/scratch/numba-cache/numbox-structref'); it compiles without a cache. Set NUMBA_CACHE_DIR to a writable
    directory, or compile without a cache to silence this warning: "cache" off in the jit options this code was given,
    or in its cache argument where it takes one, or NUMBOX_JIT_OPTIONS='{"cache": false}' where the options are the
    package's

numbox could not write the file that holds the generated code, or numba could not cache a function of that file. The
code still works, but it compiles without a cache in every process. To cache it, follow the remedy, here
``NUMBA_CACHE_DIR`` set to a writable directory.

Four kinds of numbox function generate code. :func:`~numbox.utils.highlevel.make_structref` generates a struct.
:func:`~numbox.core.variable.compile_kernel.compile_kernel` generates a kernel for a graph of formulas.
:func:`~numbox.core.work.builder.make_graph` builds a graph of ``Work`` nodes, and it compiles the derive function of
each :class:`~numbox.core.work.builder.Derived` spec: the Python function that computes that node's value from the
values of the nodes it depends on. The SQLite registrations
:func:`~numbox.core.bindings.sqlite.udf_helpers.register_aggregate`,
:func:`~numbox.core.bindings.sqlite.udf_helpers.register_window` and
:func:`~numbox.core.bindings.sqlite.tvf.register_tvf` generate the callbacks that SQLite calls.

numba caches a function only if it can find the function's source, and a function compiled from a string has none.
``@njit(cache=True)`` on such a function ``g`` raises
``RuntimeError: cannot cache function 'g': no locator available for file '<string>'``. numbox therefore writes each
piece of generated code to a file, whose name carries a digest, so that each version of the code has a file of its
own. numbox writes the files under ``NUMBA_CACHE_DIR``, or under the user's cache directory if the variable is unset, in
directories such as ``numbox-structref`` (:ref:`where the generated code is written <utils_generated_code_location>`).
That directory can be unwritable even where numba caches numbox's own functions, because numba caches those beside their
sources.

The generating functions determine their njit options in different ways. ``make_structref`` uses the options you pass
without merging them into numbox's options. ``compile_kernel`` and ``make_graph`` merge your options into numbox's, and
your values take precedence for the keys that you set. ``NUMBOX_JIT_OPTIONS`` therefore still controls caching unless
your options name ``"cache"``. The ``cache`` argument of ``compile_kernel`` overrides both your options and
``NUMBOX_JIT_OPTIONS``. The SQLite registrations always use numbox's options. The last clause of the warning quoted
above therefore gives three ways to compile without a cache, and what it calls "the package's" options are numbox's.

When the file of a derive function cannot be cached, the derive function compiles without a cache too, but numbox
gives no warning for it.

For a path that is too long, the remedy contains the words ``path is too long for the file system`` and asks for
``NUMBA_CACHE_DIR`` to be set to a shorter path. numbox shortens the names of the generated files, so only the directory
can make the path too long. On Windows, a path component that is too long fails with ``[WinError 123]`` instead, and the
remedy is the general one, ``Set NUMBA_CACHE_DIR to a writable directory``. A shorter ``NUMBA_CACHE_DIR`` fixes that
case too.

``make_graph`` also generates the graph maker, the function that builds the graph's nodes, which
:ref:`the builder overview <builder>` describes. It compiles the graph maker as though it were defined in numbox's
``builder`` module, so numba caches it beside that module. Options you pass to ``make_graph`` can enable caching for the
graph maker where the check disabled it for numbox. ``make_graph`` therefore runs the check on the builder
module's file under those options. Its warning begins ``numba cannot cache the code generated at builder.py here`` and
gives the remedy for numbox's own files.

The utils page explains why each version of the code needs a file of its own, under
:ref:`Cache-anchor mechanism <utils_cache_anchor>`.

How numbox checks
-----------------

The check runs once per process, when the module's first import sets ``jit_options``. It is skipped only if the
options set ``"cache"`` to ``false``. The check still runs for a value of ``NUMBOX_JIT_OPTIONS`` without ``"cache"``,
because numba caches the kernels of ``compile_kernel`` and the SQLite callbacks under such options. If the check then
fails, numbox warns once, and ``jit_options`` contains ``"cache": False``.

numba picks a location for a function, not for a file, so the check uses the probe to ask numba about a file.
:func:`~numbox.core.configurations.check_cache_location` has Python compile a two-line function named
``_cache_probe`` as though it were defined in the file in question. It passes the probe to numba's cache set-up, as
numba does for a function decorated with ``cache=True``. numba then picks a locator by its own rules, in its own order
or in the order that ``NUMBA_CACHE_LOCATOR_CLASSES`` gives, and raises its own error if no locator accepts the file.
numba never compiles the probe.

The check then repeats the other steps that numba takes at decoration and at the first save:

- It reads the source's stamp, which from numba 0.68 is a hash of the source, so the source must be readable.
- It runs numba's test that the location can be written in. numba's decoration runs that test for every location
  except that of a ``.zip``.
- It asks numba for the location once more, as numba does at each save.
- It creates and removes a file with a name of 128 bytes.

The last step matters for a location near the path limit. numba's own test creates a temporary file. On Linux, that file
is usually unnamed. It has no entry in the directory, so it has no name that could be too long. Elsewhere its name is
short. The files that numba saves later carry the source file's name, the function's qualified name and line, and the
Python version. numba writes each of them first under a temporary name 21 bytes longer. The 128 bytes cover the longest
such name for numbox's functions, temporary suffix included. Without this step, a location close to the path limit would
pass numba's test. numba would then fail with ``File name too long`` the first time it opened a function's index file,
which for numbox happens at import.

The check asks first about this module's own file, and then about one source file in each directory of numbox that
contains a module, because numba's ``__pycache__`` locations are per directory. It follows symbolic links, so it also
asks about a directory of numbox that is a link into another tree, which may be read-only. When a module is present
only as a ``.pyc`` file, the check asks about the file that the ``.pyc`` file records. For a ``.zip``, the check lists
the archive's directories, because an archive has no directories on disk.

The check stops at the first file that numba cannot cache, and it then disables caching for all of numbox. It does so
whether or not the modules in that file's directory define a cached function. Compiling without a cache costs compile
time in each process, but it never gives a wrong result.

Empty cache directories are an expected side effect of the check. For each file it asks about, the check creates the
cache directory that numba would use, one for each directory of numbox. numba itself creates such a directory only
when it caches a function there, and for a ``.zip`` only at the first save. Some of these directories therefore hold
none of numba's files until a function in them is compiled. The cache directory for numbox's top-level directory never
holds any of numba's files, because the top-level directory holds no compiled code.

The error in a warning's parentheses is usually numba's own, as numba would raise it at decoration. In three cases, the
check raises its own error instead. That error names the location that numba chose. The three are a failed test of
the location, a 128-byte file that cannot be created, and a temporary directory from IPython that does not exist.

Only numba's cache errors lead to the warning. They are the errors for which
:func:`~numbox.core.configurations.is_a_cache_error` returns ``True``:

- any ``OSError``;
- numba's ``RuntimeError`` when no locator accepts the file;
- numba's ``ValueError`` for a path that contains ``.zip`` but has no part that ends in ``.zip``;
- the ``KeyError`` that numba 0.68 and later raise for a ``.zip`` member without its source;
- the ``UnicodeEncodeError`` for a path that is not UTF-8.

The check raises any other error unchanged, so a fault elsewhere in numba appears as an error, not as a warning.

:func:`~numbox.core.configurations.cache_remedy` composes the remedy from the error and the file. It compares the path
that the error names with the location that each locator would give the file. It considers the locators in the order
that numba tries them, and it composes the remedy for the first locator whose location matches. For an error that names
no file, such as one raised for a full disk, it lists the locations that numba could have chosen. The docstring of
``cache_remedy``, under `Modules`_, describes the cases that it distinguishes.

Checking a package built on numbox
++++++++++++++++++++++++++++++++++

If numba caches the functions of your package, the cases above apply to your package as they do to numbox. Three
functions of this module let you run the same check on your own files.
:func:`~numbox.core.configurations.check_cache_location` asks numba about one file and raises numba's error if numba
could not cache a function of it. :func:`~numbox.core.configurations.is_a_cache_error` distinguishes those errors from
all other errors. :func:`~numbox.core.configurations.cache_remedy` composes the remedy.

The module below, ``kernels.py``, enables caching for its functions only where numba can cache them. It passes
``longest_file_name=128``, a bound on the length of the cache file names that numba writes for it; the rule for your own
bound follows the example.

.. code-block:: python

    """Kernels that numba caches only where it can write a cache for this file."""
    import inspect
    import os
    import warnings

    from numba import njit
    from numbox.core.configurations import cache_remedy, check_cache_location, is_a_cache_error


    def _cache_probe():
        """Never compiled: the check asks numba about the file that holds this function's code."""


    def _jit_options():
        if os.environ.get("KERNELS_CACHE") == "0":
            return {"cache": False}
        py_file = inspect.getfile(_cache_probe)
        try:
            check_cache_location(py_file, longest_file_name=128)
        except (RuntimeError, OSError, KeyError, ValueError) as error:
            if not is_a_cache_error(error):
                raise
            silence = "KERNELS_CACHE=0 to turn caching off and silence this warning"
            remedy = cache_remedy(py_file, error, silence, package="kernels")
            warnings.warn(f"numba cannot cache kernels here ({error}); it compiles without a cache. {remedy}",
                          RuntimeWarning, stacklevel=2)
            return {"cache": False}
        return {"cache": True}


    jit_options = _jit_options()


    @njit(**jit_options)
    def scale(x, factor):
        return x * factor

A second file in the same directory, ``use_kernels.py``, calls ``scale``:

.. code-block:: python

    from kernels import scale

    print(scale(2.0, 3.0))

``python use_kernels.py`` prints ``6.0`` and caches ``scale``; a second run loads ``scale`` from the cache. Where
numba cannot cache ``kernels.py``, the import of ``kernels`` warns once, and ``scale`` compiles without a cache:

.. code-block:: text

    RuntimeWarning: numba cannot cache kernels here (cannot cache function '_cache_probe': no locator available for
    file '/srv/kernels/kernels.py'); it compiles without a cache. Set NUMBA_CACHE_DIR to a writable directory, or
    KERNELS_CACHE=0 to turn caching off and silence this warning

The module asks about ``inspect.getfile(_cache_probe)``, the file that the function's code records, because that is
the file numba looks up. A module's ``__file__`` can name another file: for a module present only as a ``.pyc`` file,
it names the ``.pyc``, while numba looks for the ``.py`` file the ``.pyc`` was compiled from. The ``except`` clause
names the four classes that numba's cache errors belong to; a ``UnicodeEncodeError`` is a ``ValueError``, so the
clause catches it too. The ``package`` argument names your package in the remedies that ask to install it again. The
``silence`` argument supplies the clause that ends every remedy.

Choose ``longest_file_name`` from the names that numba writes. For a function compiled with ``njit``, numba builds
each cache file name from four parts:

- ``stem``, the module's file name without ``.py``;
- ``qualname``, the function's qualified name, with any angle brackets removed;
- ``line``, the line of the function's first decorator;
- the Python version, written as ``py`` and the ``major`` and ``minor`` version numbers, followed by any ``abiflags``,
  as in ``py312``.

The index file adds ``.nbi``, and each data file adds a dot, a counter that starts at 1, and ``.nbc``. numba writes
every file first under a temporary name 21 bytes longer. For up to 999 compiled signatures, the longest name is
therefore ``len(f"{stem}.{qualname}-{line}.py{major}{minor}{abiflags}.999.nbc".encode()) + 21`` bytes. Allow one
byte more for each further digit of the counter.

For ``kernels.py`` the formula gives 51 bytes, well under the bound of 128. ``LONGEST_CACHE_FILE_NAME``, numbox's own
bound of 128 bytes, covers numbox's functions only. A location that fits numbox's names but not yours passes the
check, and numba fails later with ``File name too long``.

numba's cache locations are per directory, so if your package has modules in several directories, check one file in
each. Define a ``_cache_probe`` in one module of each directory, run the check on the file of each, and stop at the
first error, as numbox does. The same calls can check a function that a user passes to your package, through
``check_cache_location(inspect.getfile(func), ...)``. That file is a cell when the user works in a notebook, and the
remedies under `In IPython and Jupyter`_ then apply.

A name in your package can be too long even where numbox's names fit. numba names its subdirectory under
``NUMBA_CACHE_DIR`` or the user's cache directory with the source directory's name, an underscore and the 40 hexadecimal
digits of a SHA-1 hash. The subdirectory's name is 41 bytes longer than the source directory's name. A directory whose
name is longer than 214 bytes therefore gets a subdirectory name longer than 255 bytes, the limit of nearly every file
system. The remedy then names the subdirectory and its length, with the words
``261 bytes, which is longer than a name can be on that file system (255)``. It asks for the file to be put in a
directory with a shorter name. As an alternative, it offers the ``__pycache__`` beside the source, whose name contains
no hash. The limit in parentheses is the one that the file system holding the location reports, or 255 where the file
system does not report a limit. When ``NUMBA_CACHE_DIR`` is unset, the remedy has no directory whose file system it
could query, so it measures the name against 255.

A function's own name can also be too long. numba's cache file names for it then exceed the file system's limit, and
the function's first call fails with ``File name too long``, when numba opens the function's index file. Rename the
function or its module. Given that error, ``cache_remedy`` asks for a shorter name for the function or its file. Its
last clause says that the shorter name "is kernels's to give". These words mean that only your package can choose the
new name. If ``kernels.py`` held a function whose name had 260 letters, the remedy would read as follows, with the
name shortened here:

.. code-block:: text

    numba names the cache files for a function after the source's name and the function's qualified name,
    kernels.fff...fff-4.py312.nbi, 280 bytes, which is longer than a name can be on that file system (255), and no
    shorter path cures it: a shorter name for the function or its file, which is kernels's to give

The remedy asks for a shorter name only for numba's own error, which names the long file. If the
``longest_file_name`` you pass is itself longer than the file system allows, the remedy is different. It offers
``NUMBA_CACHE_DIR`` set to a directory on a file system that allows names of that length. It does not ask for a shorter
name.

Modules
++++++++

numbox.core.configurations
--------------------------

.. automodule:: numbox.core.configurations
   :members:
   :show-inheritance:
   :undoc-members:
