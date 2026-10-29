numbox.core.cache_key
=====================

Overview
++++++++

numba's on-disk cache, which `cache=True <https://numba.readthedocs.io/en/stable/user/jit.html#cache>`_ enables, stores
each compiled function under a key. The key holds the signature, the target machine with its CPU features, and
hashes of the function's bytecode and of the values its closure captures. It does not hold the jit options, the
keyword arguments other than ``cache`` that you pass to the decorator, such as ``fastmath``, ``error_model`` and
``boundscheck``. Nor does it hold most of the ``NUMBA_*`` `environment variables
<https://numba.readthedocs.io/en/stable/reference/envvars.html>`_ that change the code numba emits, such as
``NUMBA_BOUNDSCHECK`` and ``NUMBA_OPT``. This page calls every ``NUMBA_*`` environment variable a setting.

If a jit option or a code-changing setting differs between two runs while the function's source file stays the same, the
second run loads the code that the first run compiled. numba gives no warning. Suppose you set ``NUMBA_BOUNDSCHECK=1``
to find an indexing bug: a function cached in an earlier run still runs without bounds checks, and an index past the end
of an array reads whatever memory follows the array.

Jit options have the same problem when the decorator takes them from outside the function's file, as in
``@njit(cache=True, **OPTIONS)`` with ``OPTIONS`` imported from a configuration module or read from an environment
variable. A change there does not modify the file, so numba loads the code compiled under the old options. A change
to the options written in the decorator line does not have this problem: it changes the file, and numba compiles the
function again.

numbox adds a part of its own to the key for the functions it compiles itself, such as its bindings to C libraries,
and for the functions you decorate with :func:`~numbox.core.cache_key.numbox_njit` or
:func:`~numbox.core.cache_key.numbox_cfunc` and ``cache=True``. All of these are called keyed functions below, and the
section `Which functions are keyed`_ lists them. numbox's part holds the jit options, the settings that change the
emitted code, and numbox's version. The version is in the key because numba compiles the numbox code that a function
calls into the function itself, and numba's key does not cover that code.

A keyed function compiles again when a jit option or a setting in its key changes. Its cache keeps one entry for each
combination it has run under, so a run that returns to an earlier combination loads that combination's entry. When a
keyed function compiles more often than you expect, :func:`~numbox.core.cache_key.explain_cache_key` shows what numbox
added to its key; the section `Reading the key`_ shows how to compare two runs.

A function that you decorate with ``numba.njit(cache=True)`` is not keyed. numba caches it under its own key, so
it can load code compiled under old jit options or settings. After a numbox upgrade, it can also load code that numba
compiled from an older numbox's ``@overload`` and ``@intrinsic`` implementations. Decorate it with ``numbox_njit``
instead, or delete its cache files after such a change, as described in `Deleting a function's cache files`_. numbox
handles one case itself: it repairs a cached caller of a function decorated with ``@proxy``, numbox's decorator that
keeps one compiled copy for all callers, after that function changes. The section "After you edit a proxied function" of
the :doc:`proxy page <numbox.core.proxy>` describes the repair.

Decorating your own functions
+++++++++++++++++++++++++++++

Replace ``numba.njit`` with ``numbox_njit``, or ``numba.cfunc`` with ``numbox_cfunc``, and keep ``cache=True``. Both
take numba's arguments unchanged, including signatures and options. Without ``cache=True``, they behave exactly as
numba's decorators. The following example is saved as ``bounds.py``:

.. code-block:: python

    import numpy as np
    from numba import njit

    from numbox.core.cache_key import numbox_njit


    @njit(cache=True)
    def index_numba(a, i):
        return a[i]


    @numbox_njit(cache=True)
    def index_numbox(a, i):
        return a[i]


    view = np.arange(10.0)[:3]
    print(index_numba(view, 3))
    print(index_numbox(view, 3))

``view`` holds the first three elements of a longer array, so the index 3 is past the end of ``view`` but inside
memory that numpy allocated. If you run ``python bounds.py``, both functions compile and print ``3.0``, because numba
compiles without bounds checks by default. Then run ``NUMBA_BOUNDSCHECK=1 python bounds.py``. ``index_numba`` loads
the code from the first run and prints ``3.0`` again. ``index_numbox`` misses the cache, because its key holds the
setting. It compiles again with bounds checks, and its call raises ``IndexError: index is out of bounds``.

Both of ``index_numbox``'s entries stay in the cache. To see that, run the script a third time without the setting and
with ``NUMBA_DEBUG_CACHE=1``, which makes numba report its cache use. numba prints a ``[cache] data loaded from`` line
for each function and no ``data saved to`` line, because each function loads the entry from the first run. Inside a
program, a dispatcher's ``stats.cache_hits`` and ``stats.cache_misses`` count the same events.

The key covers an eager compile, for which the decorator is given a signature, as well as a lazy compile at the first
call, because numbox attaches its part when numba sets up the function's cache at decoration.

Under numba's own key, a ``numba.cfunc`` and a ``numba.njit`` of the same Python function share their entries if both
are given the signature as a numba ``Signature`` object, such as ``float64(float64)`` built from ``numba.float64``.
The two entries are not interchangeable. numba compiles a C callback without the wrapper through which a dispatcher
calls compiled code from Python. An njit that loads the callback's entry therefore calls a wrapper that the entry does
not contain. The process then crashes with a segmentation fault. The njit's entry holds both wrappers, so a C callback
that loads it works. numbox's part holds the decorator kind, so numba caches a ``numbox_cfunc`` and a ``numbox_njit`` of
the same function in separate entries. numba's own key also differs between the two when the signature is a string, as
in ``@numbox_cfunc("float64(float64)", cache=True)``, because numba keys the njit by the string and the C callback by
the parsed signature.

.. _cache_key_which_functions:

Which functions are keyed
+++++++++++++++++++++++++

Every function that numbox compiles with caching enabled is keyed, whichever decorator compiles it:

- its own functions, among them its :doc:`bindings <numbox.core.bindings>`, the jitted wrappers of functions in
  libc, libm and SQLite;
- the jitted constructor, accessors and methods of each struct type that
  :func:`~numbox.utils.highlevel.make_structref` generates, a numba structref that jitted code and Python share;
- the fused kernel that :func:`~numbox.core.variable.compile_kernel.compile_kernel` compiles from a graph of
  formulas;
- the functions that :func:`~numbox.core.work.builder.make_graph` compiles to build a graph of work nodes, among them,
  for each :class:`~numbox.core.work.builder.Derived` node, the derive function that computes the node's value from
  other nodes;
- the functions you pass to :func:`~numbox.core.proxy.proxy.proxy`, which keeps one compiled copy of a function for
  all its callers;
- the functions you pass to :func:`~numbox.utils.highlevel.cres`, which compiles a function for one signature and
  returns it as a first-class function value.

numbox's own functions take their jit options from the ``NUMBOX_JIT_OPTIONS`` environment variable
(:doc:`numbox.core.configurations`), not from the files that define them. A change of that variable makes each of them
compile again; with numba's key only, they would load the old code.

Of your functions, only the ones that ``numbox_njit`` and ``numbox_cfunc`` return are keyed. A ``numba.njit`` of the
same Python function is cached under numba's key alone. numbox has no counterpart of ``@vectorize``, ``@guvectorize`` or
numba's CUDA decorators, so the functions they compile are cached under numba's key as well.

What the key holds
++++++++++++++++++

numbox appends its part to numba's key and changes nothing else in the key. The part holds three items.

The first is numbox's version. numba compiles numbox's ``@overload`` and ``@intrinsic`` code into each function that
calls that code, while numba's key covers only the caller's own bytecode. With the version in the key, a keyed caller
compiles again after an upgrade instead of loading the code compiled from an older numbox. The section `After you
upgrade numbox`_ says what the version is in a release and in a source checkout.

The second is the jit options. numba parses them into what it calls flags, filling in its defaults, and the key holds
the flags together with the decorator kind, ``njit`` or ``cfunc``, and the types given in ``locals``. The key records
the options that numba compiles under, because they can differ from the ones in the decorator. When numba compiles a
function for a caller that it is compiling, the callee takes the caller's ``fastmath``, ``error_model``, ``forceinline``
and ``nrt`` (which enables numba's runtime for memory management) wherever the callee's own options leave them unset.
The callee's key holds the values that the callee used, so numba saves the code from that compile in a separate entry.

The third is the settings that change the emitted code and that numba's own key leaves out. The key holds each with
its value in ``numba.config``, the module where numba keeps the current value of every setting. numbox reads the
values when numba looks the function up in its cache, and again when numba saves the compiled code.

Which settings the key holds
----------------------------

:data:`~numbox.core.cache_key.KEYED_SETTINGS` lists the settings in the key, among them ``NUMBA_BOUNDSCHECK``,
``NUMBA_OPT`` and ``NUMBA_DEBUGINFO``. A setting is on that list if numba's own key leaves it out and changing it
changes the LLVM IR or the machine code that numba emits, even if only speed or debugging information changes. A
survey of numba 0.63 to 0.68, each minor version at its latest patch release, applied that rule to every setting. The
description of ``KEYED_SETTINGS`` explains how the survey applied the rule.

:data:`~numbox.core.cache_key.UNKEYED_SETTINGS` lists the other settings, each with the reason it is left out.
``NUMBA_CPU_NAME`` and ``NUMBA_CPU_FEATURES`` are in numba's own key already, through the target machine.
``NUMBA_DISABLE_JIT`` disables compilation: the decorated functions run as plain Python, so there is nothing to cache.
The list gives ``NUMBA_USE_LEGACY_TYPE_SYSTEM`` the same reason, that it produces no binary; only numba 0.63 and 0.64
read it, to choose which scalar types numba defines. ``NUMBA_CACHE_DIR`` and ``NUMBA_CACHE_LOCATOR_CLASSES`` choose
where numba stores the cache, not what the cache contains. The rest change no emitted code, ``NUMBA_NUM_THREADS`` and
``NUMBA_THREADING_LAYER`` among them: a parallel function cached under one thread count loads under another and returns
the same result. Between them, the two lists name every ``NUMBA_*`` setting that numba 0.63 to 0.68 reads.

Reading the key
+++++++++++++++

:func:`~numbox.core.cache_key.explain_cache_key` returns numbox's part of a keyed function's key. Pass it the dispatcher
or the ``@cfunc`` of a keyed function, or the object that ``cres`` returns. A proxied function's ``.as_func`` works as
well. ``.as_func`` is the proxied function as a first-class function value (:ref:`passing the function as a value
<proxy_passing_as_value>`). ``explain_cache_key`` reports the values that ``numba.config`` holds when you call it, which
are the values the next cache lookup in this process uses. The jit options it reports are the ones the function uses
when it is called from Python. A caller in jitted code can pass the function other options, as described in `What the
key holds`_, and those are not shown.

The result is a dict with three entries: ``"numbox"`` holds numbox's version, ``"flags"`` holds the jit options as one
string of ``name=value`` items, and ``"settings"`` maps each setting in the key to its value as text. The following
example is saved as ``show_key.py``:

.. code-block:: python

    from numbox.core.cache_key import explain_cache_key, numbox_njit


    @numbox_njit(cache=True)
    def total(a):
        return a.sum()


    key = explain_cache_key(total)
    print(repr(key["numbox"]))
    print([item for item in key["flags"].split(";") if item.startswith(("kind", "error_model", "nrt", "locals"))])
    print({name: key["settings"][name] for name in ("NUMBA_BOUNDSCHECK", "NUMBA_OPT")})

In a source checkout, with no ``NUMBA_*`` setting in the environment, ``python show_key.py`` prints the version, four of
the flags and two of the settings:

.. code-block:: text

    ''
    ['kind=njit', "error_model='python'", 'nrt=True', 'locals=[]']
    {'NUMBA_BOUNDSCHECK': 'None', 'NUMBA_OPT': '3'}

The version is empty in a source checkout, and a release shows its version number there. An unset setting appears as the
text ``'None'``. A setting can appear in the flags as well: ``NUMBA_DEBUGINFO=1`` also sets the ``debuginfo`` flag.
Two results that differ belong to two different keys, so you can save the result in each run and compare the runs to
find out why a function compiled again. The following example is saved as ``compare_keys.py``:

.. code-block:: python

    import json
    import pathlib

    from numbox.core.cache_key import explain_cache_key, numbox_njit


    @numbox_njit(cache=True)
    def total(a):
        return a.sum()


    saved = pathlib.Path("total_key.json")
    now = explain_cache_key(total)
    if saved.exists():
        old = json.loads(saved.read_text())
        print("numbox changed:", now["numbox"] != old["numbox"])
        print("flags changed:", now["flags"] != old["flags"])
        print("settings changed:", sorted(k for k, v in now["settings"].items() if v != old["settings"].get(k)))
    saved.write_text(json.dumps(now))

Run ``python compare_keys.py``, then ``NUMBA_BOUNDSCHECK=1 python compare_keys.py``. The second run prints:

.. code-block:: text

    numbox changed: False
    flags changed: False
    settings changed: ['NUMBA_BOUNDSCHECK']

If the two results match but the function compiled again, something that numba checks itself changed: the
source file, the argument types, the values the closure captures, the numba or Python version, the CPU, or the cache
directory.

``explain_cache_key`` raises a ``ValueError`` for a function that is not keyed, such as a ``numba.njit`` function or a
``numbox_njit`` function without ``cache=True``. It also raises one for a function that numbox keeps out of the on-disk
cache, with the reason that the ``UncachedWarning`` gives, as described in
`If numbox warns that a function is not cached`_.

Deleting a function's cache files
+++++++++++++++++++++++++++++++++

numba names a function's cache files after the module, the function, the line of its decorator and the Python version,
such as ``bounds.index_numbox-12.py312.nbi`` and ``bounds.index_numbox-12.py312.1.nbc`` for ``index_numbox`` in
``bounds.py`` under Python 3.12. The ``.nbi`` file is the index, and each ``.nbc`` file holds the code of one entry.
numba keeps them in the first of these places that it can write to:

- under ``NUMBA_CACHE_DIR``, if that variable is set, in a subdirectory named after the source's directory: the
  directory's base name, an underscore and the SHA-1 hash of its full path;
- otherwise in the ``__pycache__`` directory beside the source;
- otherwise in numba's user cache directory, which the configurations page describes in
  :ref:`where numba writes the cache <where_the_cache_lands>`.

Delete the function's ``.nbi`` file and all its ``.nbc`` files. numba then compiles the function again and saves new
files.

.. _cache_key_leaves_out:

What the key leaves out
+++++++++++++++++++++++

numba's cache key for a function does not include the code of the functions it calls, and numbox's part does not either.
numba compiles each callee into the caller and checks a cached caller only against the source file that defines the
caller. An edit to a callee in another file therefore does not invalidate the cached caller, which still runs the
callee's old code. `numba's documentation <https://numba.readthedocs.io/en/stable/user/jit.html#cache>`_ names this
limit. The callee may be an ``@njit`` function, an ``@overload`` or an ``@intrinsic``, in your own code or in a package
that you edit or upgrade.

Delete the caller's cache files after such an edit, as described in `Deleting a function's cache files`_. numba then
compiles the caller again at its next call.

numbox handles two such cases itself. A keyed caller compiles again after an upgrade of numbox from one release to
another, because the key holds numbox's version. numbox also repairs a cached caller of a function decorated with its
``@proxy`` after an edit to that function, whether or not the caller is keyed. The section "After you edit a proxied
function" of the :doc:`proxy page <numbox.core.proxy>` describes the repair. The repair does not cover
:ref:`a caller that holds the function as a module-level constant <proxy_module_level_constant>`.

The key also leaves out the values of the globals that a function reads. numba compiles a global's value into the
code as a constant, so a cached function still uses the value from the run that compiled it. A global whose value comes
from outside the function's file, read from an environment variable at import or imported from another module, can
change while the file stays the same. The following example is saved as ``scaling.py``:

.. code-block:: python

    import os

    from numbox.core.cache_key import numbox_njit

    SCALE = float(os.environ.get("SCALE", "2.0"))


    @numbox_njit(cache=True)
    def scaled_by_global(x):
        return x * SCALE


    @numbox_njit(cache=True)
    def scaled(x, scale):
        return x * scale


    print(scaled_by_global(1.0), scaled(1.0, SCALE))

Run ``python scaling.py``, then ``SCALE=3 python scaling.py``. The first run prints ``2.0 2.0``, and the second
prints ``2.0 3.0``. ``scaled_by_global`` loads its cached code, which holds the old value. ``scaled`` takes the value
as an argument, so it uses the new value while its code still loads from the cache.

The same applies to a value that your module derives from ``NUMBA_NUM_THREADS`` at import. The thread count itself
changes no emitted code, so it is not in the key. numba compiles the derived value into the code as a constant,
though, so a run under another thread count loads code that holds the old value. Pass such a value as an argument, as
``scaled`` does, or delete the function's cache files after it changes.

If numbox warns that a function is not cached
+++++++++++++++++++++++++++++++++++++++++++++

The key cannot hold every jit option. If a function's options include one that the key cannot hold, numbox keeps the
function out of the on-disk cache, so numba compiles it again in every process. An
:class:`~numbox.core.cache_key.UncachedWarning` names the function once per process:

.. code-block:: text

    UncachedWarning: numbox: total is compiled without an on-disk cache: the jit flag parallel is a dict, which numba empties as it parses it. numbox's cache key cannot hold it reliably, so the function is not cached.

The warning names the function, here ``total``, and the option that the key cannot hold, here a ``parallel`` dict.
Four kinds of option cause it, and each item below says what to pass or do instead.

- A ``parallel`` dict, such as ``parallel={"fusion": False}``. numba empties the dict as it reads it, so a later
  compile given the same dict would use numba's defaults. Pass ``parallel=True``, which the key holds, if numba's
  default parallel options suit the function. Otherwise pass ``parallel=ParallelOptions({"fusion": False})``, with
  ``ParallelOptions`` imported from ``numba.core.cpu_options``; the key holds that as well.
- A ``pipeline_class`` other than numba's default. The pipeline is code, and the key does not hold code. If numba's
  key is enough for this function, with the limits that the `Overview`_ describes, decorate it with ``numba.njit``
  instead.
- A function passed as the ``inline`` option, which numba calls to decide whether to inline each call. Pass
  ``inline="always"`` or ``inline="never"`` instead, which the key holds.
- A type in ``locals`` whose name numba builds from an object's memory address, as it does for a jitclass instance
  type and any type that contains one. The address differs in every process, so the key cannot hold the name. If you
  leave that variable out of ``locals``, numba infers its type.

In a test run, you can turn the warning into an error with pytest's ``-W error::numbox.core.cache_key.UncachedWarning``.

If you change a setting inside a running process
++++++++++++++++++++++++++++++++++++++++++++++++

Set each ``NUMBA_*`` variable in the environment before Python starts. A value changed inside a running process,
through ``os.environ`` or ``numba.config``, can be saved under a key that does not describe the compiled code. A later
run whose settings match that key then loads the mismatched code.

numba copies every ``NUMBA_*`` variable into ``numba.config`` when it is imported, and its compiler reads the values
there. It copies the environment again when a compile starts, if a ``NUMBA_*`` variable has changed. numbox computes its
part from ``numba.config`` twice for each compile: when numba looks the function up before the compile, and when numba
saves the code after it. numba therefore copies a ``NUMBA_*`` variable changed through ``os.environ`` into
``numba.config`` at the next compile. The saved key then holds the new value. Until that compile, ``numba.config`` keeps
the old value, so a lookup can still load the entry for the old value.

The code and the key do not match when numba has already built something from a setting before that refresh: the parsed
jit options of the compile, or the target machine of the process. numba parses the jit options, taking some of their
defaults from the settings, before it copies the environment again. So ``NUMBA_DEBUGINFO`` set through ``os.environ``
gives a key with debugging information enabled and code without it. numba builds its target machine once, at the
first jit decoration in the process, from ``NUMBA_OPT`` and ``NUMBA_ENABLE_AVX``, and keeps it until the process
exits. Importing a numbox module that defines jitted functions counts as such a decoration.

For example, a program imports a module of jitted functions and then sets ``os.environ["NUMBA_OPT"] = "0"``. Its next
compile runs numba's optimization passes at level 0, while the target machine stays at level 3, the default. numba saves
the entry under a key that holds the value 0 for ``NUMBA_OPT``. That entry's ``.nbc`` file held 25,375 bytes for a
function that loops over an array, on numba 0.68.0; a run started with ``NUMBA_OPT=0`` saves 28,415 bytes for the same
function. A later run started with ``NUMBA_OPT=0`` loads the 25,375-byte entry. The same happens if the program sets the
variable after ``import numba`` but before its first decoration, because numba copies the environment again only when a
compile starts.

numba documents its settings as environment variables, not as values to assign in ``numba.config``. numba does build its
target machine from a value assigned to ``numba.config`` before the first jit decoration. Such an assignment is too
late, however, for a setting that numba uses only when it is imported, such as ``NUMBA_DISABLE_INTEL_SVML``. A plain
``numba.config.OPT = 0`` before the first decoration causes that decoration to fail with
``AttributeError: 'int' object has no attribute 'is_opt_max'``; set ``NUMBA_OPT`` in the environment instead.

A running process keeps the compiled code of every function it has compiled until it exits. A setting changed later
does not make it compile the function again.

If several processes compile the same function at once
++++++++++++++++++++++++++++++++++++++++++++++++++++++

numba's cache save takes no lock (`numba issue 10862 <https://github.com/numba/numba/issues/10862>`_). If two
processes save new entries for the same function at the same time, two keys can share one ``.nbc`` file. A lookup under
one of the two keys then loads code compiled for the other. Under numbox's key, processes that run under different
settings or jit options save different entries for one signature. Two such processes can therefore save at the same
time even when the function already has an entry for that signature from another combination.

To prevent it, fill the cache before the processes start. Run the function once for each signature under each
combination of settings and jit options that the processes use, one process at a time.

Installing the key
++++++++++++++++++

Do not remove numbox's modules from ``sys.modules`` to import them again. After such a fresh import, the functions that
``numbox_njit``, ``numbox_cfunc``, ``cres`` and ``proxy`` decorate are cached under numba's key alone. numbox's own
functions remain keyed. After ``importlib.reload`` of a numbox module, every function remains keyed. The difference
comes from how numbox installs its part: it wraps numba's cache functions once per process, when ``numbox.core`` is
first imported, and those wrappers never see the functions that a freshly imported second copy of the decorators marks.

If numbox cannot wrap numba's cache functions, it leaves numba unchanged. numbox then warns once, with the error in
parentheses:

.. code-block:: text

    RuntimeWarning: numbox: could not add its part to numba's cache key (...); what numbox compiles is cached under numba's key alone

The warning means that the installed numba lacks one of the functions that numbox wraps. Install a numba version that
numbox supports, from 0.63.1 up to but not including 0.69. Until then, numba caches every function under its own key,
including the functions that you decorate with ``numbox_njit`` and ``numbox_cfunc``.

.. _cache_key_after_upgrade:

After you upgrade numbox
++++++++++++++++++++++++

numbox's release builds write the release's version into ``numbox.__version__``, and numbox's part of every key holds
it. The first run after an upgrade from one release to another therefore compiles each keyed function again, once for
each signature and each combination of jit options and settings that it runs under. Later runs load the new entries.

The entries that your keyed functions saved under the old version remain on disk. They match a lookup again only if you
reinstall that version. To reclaim the space, delete the functions' cache files, as described in
`Deleting a function's cache files`_. numbox's own functions have no old entries in their index. An upgrade replaces
their source files, so numba starts a new index for them at the first save. If you reinstall the old version, numba
compiles them again.

In a source checkout, ``numbox.__version__`` is an empty string, and it stays empty when the checkout changes. An
update of the checkout, or an edit to numbox's code, does not change the version in the key. A cached function that
calls numbox's ``@overload`` or ``@intrinsic`` code in another file then still runs the old code, whether the function
is yours or numbox's own. Delete its cache files after such a change.

Modules
++++++++

numbox.core.cache_key
---------------------

.. automodule:: numbox.core.cache_key
   :members:
   :show-inheritance:
   :undoc-members:
