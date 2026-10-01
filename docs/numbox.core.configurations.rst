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

Where the cache lands
+++++++++++++++++++++

numba writes a cached function's entries under ``NUMBA_CACHE_DIR`` when that is set, else into the
``__pycache__`` directory beside the function's source file, else into the user's cache directory
(``~/.cache/numba`` on Linux), taking the first of those it can write. It sets the cache up when the function is
decorated, so with caching on the question of where it lands is settled at import.

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
passed over; any other archive has no location at all. A directory of the package reached through a symlink
is walked like the rest, wherever the link points. The check makes the cache directories it asks about, as
numba would at the first decoration in each; with caching beside the sources that is an empty ``__pycache__``
per directory of the package, a linked one included. The check writes a name shorter than numba's, so a
cache directory within a few dozen bytes of the path limit, 4096 on Linux, passes it and overflows at
numba's first save instead, where numba's own error names the length; a shorter path is the remedy.

- For a source file on disk the remedy is ``NUMBA_CACHE_DIR`` pointed at a writable directory.
- For a ``.zip``, or a frozen application, it is the user's cache directory made writable: a ``.zip`` is the
  one archive numba caches, from 0.61 on, and it caches it there, taking the directory without checking that
  it can be written; a frozen application (``sys.frozen``) is cached there too, its sources not being on disk.
  numba reads ``NUMBA_CACHE_DIR`` only for a source file on disk, so the variable changes nothing for either.
  A ``.zip`` whose cache directory holds every entry but can no longer be written falls back too, where numba
  alone would have loaded the entries: the writability check is the rule numba applies to every other
  placement.
- For an ``.egg``, ``.whl`` or ``.pyz``, a ``.pyc``-only install or a ``.pyc`` in a ``.zip``, it is the source
  files on disk or a ``.zip`` holding them.
- ``NUMBOX_JIT_OPTIONS='{"cache": false}'`` turns caching off and silences the warning in every case, the
  package's options being what it sets.

A function numba cannot cache is compiled in every process that uses it, never wrong; that is the cost the
warning reports. An error at decoration that is not the cache's is raised as it was.

Modules
++++++++

numbox.core.configurations
--------------------------

.. automodule:: numbox.core.configurations
   :members:
   :show-inheritance:
   :undoc-members:
