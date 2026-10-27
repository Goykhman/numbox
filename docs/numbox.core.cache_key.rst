numbox.core.cache_key
=====================

Overview
++++++++

numba keys a cached function on its signature, the target machine, and hashes of its bytecode and closure. Two
things that do change the binary numba builds, but are not in that key, are the jit flags (such as ``fastmath``,
``error_model`` and ``boundscheck``) and the ``NUMBA_*`` settings that change the code numba emits. If one of them
changes between two runs, the second run loads the binary cached by the first, and numba issues no warning.

numbox adds both to the key of every function it compiles, and of any function decorated with
:func:`~numbox.core.cache_key.numbox_njit` or :func:`~numbox.core.cache_key.numbox_cfunc`. The functions numbox
compiles include its :doc:`bindings <numbox.core.bindings>`, the structrefs of
:func:`~numbox.utils.highlevel.make_structref`, the kernels of
:func:`~numbox.core.variable.compile_kernel.compile_kernel`, the derives of :class:`~numbox.core.work.builder.Derived`
nodes, and the functions passed to :func:`~numbox.core.proxy.proxy.proxy` and :func:`~numbox.utils.highlevel.cres`.
Every other function is cached under numba's key alone.

.. code-block:: python

    from numbox.core.cache_key import numbox_njit


    @numbox_njit(cache=True)
    def index(a, i):
        return a[i]

Suppose you run that once, and then again with ``NUMBA_BOUNDSCHECK=1``. Under ``numba.njit``, the second run loads
the binary cached by the first, which has no bounds checks, so an index past the end reads past the array. Under
``numbox_njit``, the key holds the setting, so the second run misses the cache and compiles the function again with
bounds checks. Both entries remain in the cache, so a third run without the setting loads the first entry.

``numbox_njit`` and ``numbox_cfunc`` take the arguments of ``numba.njit`` and ``numba.cfunc`` unchanged, including
signatures and options. Without ``cache=True``, they behave exactly as those decorators. numbox sets up the key when
numba creates the cache, at decoration, so the key applies to an eager compile, for which a signature is given, as
well as to a lazy one. The key belongs to the cache that numba creates for that decoration, not to the function: a
plain ``numba.njit`` of the same function elsewhere is cached under numba's key alone. ``@vectorize``,
``@guvectorize`` and CUDA kernels are also cached under numba's key alone.

What the key holds
++++++++++++++++++

In addition to numba's own key, the key of a function that numbox keys holds three parts:

- **numbox's version.** numbox's ``@overload`` and ``@intrinsic`` implementations produce part of the code of the
  functions that call them, and numbox's version represents them in the key.
- **The flags.** The decorator kind, ``njit`` or ``cfunc``, which numba's key does not distinguish; the flags numba
  compiles under; and the ``locals`` function body variables' numba types given at decoration. The flags are the
  decorator's options as numba parses them when a compile starts, with numba's defaults filled in. When a callee is
  compiled during the compilation of its caller, the callee inherits from the caller's compilation any of ``nrt``,
  ``fastmath``, ``error_model`` and ``forceinline`` that its options leave unset, and the key holds the inherited
  values.
- **The settings.** Each keyed ``NUMBA_*`` setting (see `The settings`_), with the value that ``numba.config`` holds
  when numba looks the function up. numba looks a function up when it is first called for a signature, when it is
  decorated with one, or when it is compiled during the compilation of a caller.

What the key leaves out
+++++++++++++++++++++++

The key leaves out what numba's key leaves out of a function's code: the values of the globals the function reads,
and the code of the functions it calls. numba checks a cached function against the file that defines it, so an edit
to that file causes a cache miss. A changed global, or an edited callee in another file, does not cause a miss, and
numba loads the old binary. Clear the cache after such a change, or pass the value as an argument, which keeps the
function cached:

.. code-block:: python

    import settings
    from numbox.core.cache_key import numbox_njit


    @numbox_njit(cache=True)
    def scaled(x, scale):
        return x * scale


    scaled(1.0, settings.SCALE)

The same holds for numbox's own code. Each release changes numbox's version, and with it every key that numbox adds.
An edit in a development checkout, or an edit in place to a package's ``@intrinsic`` or ``@overload``, changes
nothing that the key holds, so clear the cache after such an edit.

The settings
++++++++++++

Of the settings that numba's own key leaves out, numbox keys one if and only if changing it changes the LLVM IR or
assembly that numba emits. A survey of numba
0.63 to 0.68, each minor version at its latest patch release, compiled a fixed set of functions with each setting
changed in turn and compared the code. Settings that change only speed or debugging information count as well, and
so do ``NUMBA_DUMP_ASSEMBLY``, ``NUMBA_DEBUG`` and ``NUMBA_DIFF_IR``, which change the code as a side effect of the
dumps they request. Three settings change the code only in combination with another setting or flag, so a survey
that changes one setting at a time cannot reveal them. They were found in numba's source:
``NUMBA_DEBUG_NRT_STACK_LIMIT`` changes the code only under ``NUMBA_DEBUG_NRT``, and ``NUMBA_FULL_TRACEBACKS`` and
``NUMBA_DEVELOPER_MODE`` change it only where bounds checking is enabled.
``NUMBA_DISABLE_INTEL_SVML`` changes the code only where Intel's SVML library is installed, which the survey's
machines lacked. The keyed settings are :data:`~numbox.core.cache_key.KEYED_SETTINGS`. The table gives numba's
default for each setting, which is the same on every version surveyed that reads the setting:

=================================== ============================== ===================================================
Setting                             Default                        Effect on the emitted code
=================================== ============================== ===================================================
``NUMBA_BOUNDSCHECK``               unset                          bounds checks on indexing
``NUMBA_OPT``                       ``3``                          every function, ``max`` included
``NUMBA_LOOP_VECTORIZE``            ``1``                          vectorized loops
``NUMBA_SLP_VECTORIZE``             ``0``                          vectorized straight-line code
``NUMBA_ENABLE_AVX``                depends on the CPU             instruction selection
``NUMBA_DEBUGINFO``                 as ``NUMBA_ENABLE_PROFILING``  debug information in every function
``NUMBA_ENABLE_PROFILING``          ``1`` under VTune, else ``0``  the same, through ``NUMBA_DEBUGINFO``
``NUMBA_EXTEND_VARIABLE_LIFETIMES`` ``0``                          every function
``NUMBA_LLVM_REFPRUNE_PASS``        ``1``                          reference count pruning
``NUMBA_LLVM_REFPRUNE_FLAGS``       ``all`` while the pass is on   which pruning runs
``NUMBA_DEBUG_NRT``                 ``0``                          runtime debug code in every allocating function
``NUMBA_DEBUG_NRT_STACK_LIMIT``     ``0``, from numba 0.66         reference counting calls, under ``NUMBA_DEBUG_NRT``
``NUMBA_FULL_TRACEBACKS``           as ``NUMBA_DEVELOPER_MODE``    a runtime message in each bounds check
``NUMBA_DEVELOPER_MODE``            ``0``                          the same, through ``NUMBA_FULL_TRACEBACKS``
``NUMBA_DEBUG_JIT``                 ``0``                          a runtime ``printf`` per lowered instruction
``NUMBA_DEBUG_ARRAY_OPT``           ``0``                          runtime ``printf`` calls in parallel code
``NUMBA_DEBUG_ARRAY_OPT_RUNTIME``   ``0``                          runtime ``printf`` calls in parallel code
``NUMBA_PARFOR_MAX_TUPLE_SIZE``     ``100``                        whether a parallel loop over a tuple compiles
``NUMBA_GDB_BINARY``                ``gdb``                        the path compiled into a call of ``numba.gdb()``
``NUMBA_DUMP_ASSEMBLY``             as ``NUMBA_DEBUG``             register allocation and instruction order
``NUMBA_DEBUG``                     ``0``                          the same, through ``NUMBA_DUMP_ASSEMBLY``
``NUMBA_DIFF_IR``                   ``0``                          code layout
``NUMBA_DISABLE_INTEL_SVML``        ``0`` (``1`` on 32-bit Python) vectorized math through SVML, where it is installed
=================================== ============================== ===================================================

Every other setting that numba 0.63 to 0.68 reads is in :data:`~numbox.core.cache_key.UNKEYED_SETTINGS`, with the
reason it is not keyed: ``NUMBA_CPU_NAME`` and ``NUMBA_CPU_FEATURES`` are already in numba's own key;
``NUMBA_DISABLE_JIT`` compiles nothing; the thread count, the threading layer and the other settings change no code
on any version surveyed. numba's parallel code reads the thread count at run time, and a binary built for one count
runs correctly under another. A thread count that is stored in a global variable at import is treated like any other
global, which the key leaves out, so pass it as an argument instead.

Limits
++++++

- **When numba reads the settings.** Set a ``NUMBA_*`` variable before Python starts. numba reads
  ``NUMBA_DISABLE_INTEL_SVML`` when it is imported, and from 0.66 on ``NUMBA_DEBUG_NRT`` and
  ``NUMBA_DEBUG_NRT_STACK_LIMIT`` as well. It reads some other settings once, when the first jit decoration in a
  process builds its code generator. It reads ``os.environ`` again only when a compile starts, after the key is read.
  Any setting other than those three can also be changed in code through ``numba.config`` before the first jit
  decoration. A change made later can leave the key inconsistent with the binary.
- **Flags the key cannot hold.** A function given one of these compiles without an on-disk cache, in every process,
  and an :class:`~numbox.core.cache_key.UncachedWarning` names it, once per process:

  - a ``parallel`` dict, which numba empties as it parses it, so that a later compile given the same dict uses
    numba's defaults; ``parallel=True`` is keyed;
  - a pipeline class other than numba's default;
  - a function given as the ``inline`` cost model; ``inline='always'`` and ``inline='never'`` are keyed;
  - a numba type in ``locals`` that numba names after an object's address, as it names any type that holds a
    jitclass.
- **numbox installs its key once per process.** If you remove numbox's modules from ``sys.modules`` and import them
  again, then numba's key will be used in ``numbox_njit``, ``numbox_cfunc``, ``@cres`` and ``@proxy``. numbox's key will
  not be used. However, ``importlib.reload`` keeps numbox's key.
- **Saves from several processes at once.** numba's cache save does not acquire a lock (`numba issue 10862
  <https://github.com/numba/numba/issues/10862>`_). Two processes that save new entries for the same function at the
  same time can cause one entry to reference the binary that the other process saved. Under numbox's key, processes
  that run with different settings or flags save different entries, so this can happen even after the function has
  been compiled once. Compiling the function before the processes run, for each signature and each combination of
  settings and flags that they use, prevents this.
- **Compiled code within a process.** numba keeps the binary of a function it has compiled for the life of the
  process, regardless of any later change to the settings.

Reading the key
+++++++++++++++

:func:`~numbox.core.cache_key.explain_cache_key` lists the parts with their current values, for the next compile of a
direct call in this process. It takes a dispatcher, a ``@cfunc``, or the object that ``@cres`` returns:

.. code-block:: python

    from numbox.core.cache_key import explain_cache_key, numbox_njit


    @numbox_njit(cache=True, fastmath=True)
    def scaled(x):
        return x * 2.0


    parts = explain_cache_key(scaled)
    print(parts["numbox"])  # numbox's version
    print(parts["flags"])  # kind=njit;flags=[...;fastmath=fastmath[fast];...];locals=[]
    print(parts["settings"]["NUMBA_OPT"])  # 3

Two results of ``explain_cache_key`` that differ belong to two different keys, so comparing them shows which of
numbox's parts caused a function to miss the cache.

Upgrading
+++++++++

numbox's key holds parts that numba's key does not, so a function that numba cached under its own key is compiled
once more in each environment, the first time numbox keys it. The entries numba wrote remain in its index and are
never loaded again. numbox's version is in the key, so each upgrade of numbox has the same effect.

Modules
++++++++

numbox.core.cache_key
---------------------

.. automodule:: numbox.core.cache_key
   :members:
   :show-inheritance:
   :undoc-members:
