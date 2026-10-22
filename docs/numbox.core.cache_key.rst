numbox.core.cache_key
=====================

Overview
++++++++

numba keys a cached function on its signature, the target machine, and hashes of its bytecode and closure. Two
things that change the binary numba builds are not in that key: the jit flags (``fastmath``, ``error_model``,
``boundscheck`` and the rest), and the ``NUMBA_*`` settings that change the code numba emits. Change one of them
between two runs and the second run is served the first run's binary, with nothing said.

numbox adds both, with its own version, to the key of every function it compiles, its bindings, structrefs,
kernels, derives and the functions passed to ``@proxy`` and ``@cres`` among them, and of any function decorated with
:func:`~numbox.core.cache_key.keyed_njit` or :func:`~numbox.core.cache_key.keyed_cfunc`. Everything else keeps
numba's key.

.. code-block:: python

    from numbox.core.cache_key import keyed_njit


    @keyed_njit(cache=True)
    def index(a, i):
        return a[i]

Run that once, then again with ``NUMBA_BOUNDSCHECK=1``. Under ``numba.njit`` the second run loads the first run's
binary, which has no bounds checks, so an index past the end reads past the array. Under ``keyed_njit`` the key holds
the setting, so the second run misses and compiles anew with the checks. Both entries stay in the cache, so a third
run without the setting loads the first again.

``keyed_njit`` and ``keyed_cfunc`` take ``numba.njit``'s and ``numba.cfunc``'s arguments unchanged, signatures and
options included, and without ``cache=True`` they are those decorators. The key is set up when numba sets the cache
up, at decoration, so it applies to an eager compile, a signature given, as to a lazy one. It belongs to the cache
numba sets up for that decoration, not to the function: a plain ``numba.njit`` of the same function elsewhere keeps
numba's key. ``@vectorize``, ``@guvectorize`` and CUDA kernels keep numba's key.

What the key holds
++++++++++++++++++

Past numba's own key, the key of a function numbox keys holds three parts:

- **numbox's version.** numbox's ``@overload`` and ``@intrinsic`` code generators shape the code of whatever calls
  them, and their version stands in for them.
- **The flags.** The decorator, ``njit`` or ``cfunc``, which numba keys alike; the flags numba compiles under; and the
  ``locals`` given at decoration. The flags are the decorator's options as numba parses them when a compile starts,
  with its defaults filled in. A function compiled inside the compile of another, as a callee
  is, takes the ``nrt``, ``fastmath``, ``error_model`` and ``forceinline`` its options leave unset from that compile,
  and the key holds them as numba took them.
- **The settings.** Each keyed ``NUMBA_*`` setting (see `The settings`_), as ``numba.config`` holds it when numba
  looks the function up, which is when the function is first called with a signature, or decorated with one.

What the key leaves out
+++++++++++++++++++++++

The key leaves out what numba's leaves out of a function's code: the values of the globals it reads, and the code of
the functions it calls. numba checks a cached function against the file that defines it, so an edit to that file
misses. A changed global or an edited callee in another file does not, and the old binary is served. Clear the cache
after such a change, or pass the value as an argument, which keeps the function cached:

.. code-block:: python

    import settings
    from numbox.core.cache_key import keyed_njit


    @keyed_njit(cache=True)
    def scaled(x, scale):
        return x * scale


    scaled(1.0, settings.SCALE)

The same holds for numbox's own code. A release changes numbox's version, and with it every key numbox adds. An edit
in a development checkout, or an edit in place to a package's ``@intrinsic`` or ``@overload``, changes nothing the
key holds: clear the cache after one.

The settings
++++++++++++

A setting is keyed if flipping it changes the LLVM IR or assembly numba emits: a survey of numba 0.63 to 0.68, each
minor at its latest patch, compiled a fixed set of functions under each setting flipped and compared the code.
Settings that change only speed or debugging information count, as do ``NUMBA_DUMP_ASSEMBLY``, ``NUMBA_DEBUG`` and
``NUMBA_DIFF_IR``, which change the code as a side effect of the dumps they ask for. Three change it only beside
another setting, which a flip of one setting at a time cannot show, and were found in numba's source:
``NUMBA_DEBUG_NRT_STACK_LIMIT`` under ``NUMBA_DEBUG_NRT``, and ``NUMBA_FULL_TRACEBACKS`` and ``NUMBA_DEVELOPER_MODE``
in a bounds check. ``NUMBA_DISABLE_INTEL_SVML`` changes the code only where Intel's SVML library is installed,
which the survey's machines lacked. The keyed settings are :data:`~numbox.core.cache_key.KEYED_SETTINGS`:

=================================== ===================================================================
Setting                             What flipping it changes
=================================== ===================================================================
``NUMBA_BOUNDSCHECK``               bounds checks on indexing
``NUMBA_OPT``                       every function, ``max`` included
``NUMBA_LOOP_VECTORIZE``            vectorized loops
``NUMBA_SLP_VECTORIZE``             vectorized straight-line code
``NUMBA_ENABLE_AVX``                instruction selection
``NUMBA_DEBUGINFO``                 debug information in every function
``NUMBA_ENABLE_PROFILING``          the same, as ``NUMBA_DEBUGINFO``'s default
``NUMBA_EXTEND_VARIABLE_LIFETIMES`` every function
``NUMBA_LLVM_REFPRUNE_PASS``        reference count pruning
``NUMBA_LLVM_REFPRUNE_FLAGS``       which pruning runs
``NUMBA_DEBUG_NRT``                 runtime debug code in every allocating function
``NUMBA_DEBUG_NRT_STACK_LIMIT``     the reference counting calls, under ``NUMBA_DEBUG_NRT``
``NUMBA_FULL_TRACEBACKS``           a runtime message in each bounds check
``NUMBA_DEVELOPER_MODE``            the same, as ``NUMBA_FULL_TRACEBACKS``'s default
``NUMBA_DEBUG_JIT``                 a runtime ``printf`` per lowered instruction
``NUMBA_DEBUG_ARRAY_OPT``           runtime ``printf`` calls in parallel code
``NUMBA_DEBUG_ARRAY_OPT_RUNTIME``   runtime ``printf`` calls in parallel code
``NUMBA_PARFOR_MAX_TUPLE_SIZE``     whether a parallel loop over a tuple compiles
``NUMBA_GDB_BINARY``                the path compiled into a call of ``numba.gdb()``
``NUMBA_DUMP_ASSEMBLY``             register allocation and instruction order
``NUMBA_DEBUG``                     the same, through ``NUMBA_DUMP_ASSEMBLY``
``NUMBA_DIFF_IR``                   code layout
``NUMBA_DISABLE_INTEL_SVML``        vectorized math through SVML, where it is installed
=================================== ===================================================================

Every other setting numba 0.63 to 0.68 reads is in :data:`~numbox.core.cache_key.UNKEYED_SETTINGS` with the reason
it stays out: ``NUMBA_CPU_NAME`` and ``NUMBA_CPU_FEATURES`` are in numba's own key already; ``NUMBA_DISABLE_JIT``
compiles nothing; the thread count, the threading layer and the rest change no code on any version surveyed.
numba's parallel code reads the thread count at run time, and a binary built for one count runs correctly under
another. A count frozen into a global at import is a global like any other, which the key leaves out: pass it as an
argument. A test fails on any numba release that reads a ``NUMBA_*`` setting in neither table, in ``config.py`` or
through ``os.environ`` elsewhere.

Limits
++++++

- **Set the settings early.** Set a ``NUMBA_*`` variable before Python starts. numba reads
  ``NUMBA_DISABLE_INTEL_SVML`` as it is imported, and from 0.66 on ``NUMBA_DEBUG_NRT`` and
  ``NUMBA_DEBUG_NRT_STACK_LIMIT`` as well. It reads some other settings once, when the first jit decoration in a
  process builds its code generator. It reads ``os.environ`` again only as a compile starts, after the key is read.
  Any setting but those three can also be changed in code through ``numba.config`` before the first jit decoration.
  A change made later can leave the key and the binary apart.
- **Flags the key cannot hold.** A function given one of these compiles without an on-disk cache, in every process,
  and an :class:`~numbox.core.cache_key.UncachedWarning` names it, once per process:

  - a ``parallel`` dict, which numba empties as it parses it, so that a later compile given the same dict takes
    numba's defaults; ``parallel=True`` is keyed;
  - a pipeline class other than numba's default;
  - a function given as the ``inline`` cost model; ``inline='always'`` and ``inline='never'`` are keyed;
  - a numba type in ``locals`` that numba names after an object's address, as it names any type that holds a jitclass.
- **Import numbox once.** numbox installs its key once per process. Remove numbox's modules from ``sys.modules`` and
  import them again, and what ``keyed_njit``, ``keyed_cfunc``, ``@cres`` and ``@proxy`` then decorate is cached under
  numba's key alone. ``importlib.reload`` keeps the key.
- **Compile once before running in parallel.** numba's cache save takes no lock (`numba issue 10862
  <https://github.com/numba/numba/issues/10862>`_). Two processes saving a function's first entries at once can
  leave one entry pointing at the other's binary, and under numbox's key the two can be two settings.
- **What numba has compiled stays.** numba keeps the binary of a function it has compiled for the life of the
  process, whatever changes afterwards.

Reading the key
+++++++++++++++

:func:`~numbox.core.cache_key.explain_cache_key` lists the parts as they stand for a direct call's next compile in
this process. It takes a dispatcher, a ``@cfunc``, or what ``@cres`` returns:

.. code-block:: python

    from numbox.core.cache_key import explain_cache_key, keyed_njit


    @keyed_njit(cache=True, fastmath=True)
    def scaled(x):
        return x * 2.0


    parts = explain_cache_key(scaled)
    print(parts["numbox"])  # numbox's version
    print(parts["flags"])  # kind=njit;flags=[...;fastmath=fastmath[fast];...];locals=[]
    print(parts["settings"]["NUMBA_OPT"])  # 3

Two calls that disagree on a part are two different keys, which is how to find what made a function miss.

Upgrading
+++++++++

numbox's key holds parts numba's does not, so a function numba cached under its own key compiles once more, cold,
in each environment, the first time numbox keys it; the entries numba wrote stay in its index and are never loaded
again. numbox's version is in the key, so each upgrade of numbox does the same.

Modules
++++++++

numbox.core.cache_key
---------------------

.. automodule:: numbox.core.cache_key
   :members:
   :show-inheritance:
   :undoc-members:
