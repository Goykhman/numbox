numbox.utils
============

The modules of ``numbox.utils`` are the tools that the rest of numbox is built on, and several of them are useful on
their own. :func:`~numbox.utils.highlevel.make_structref` defines a numba StructRef in one call.
:func:`~numbox.utils.highlevel.cres` compiles a function into a value that jitted code can take as an argument, an
instance of :class:`~numbox.utils.derive_wap.DeriveWAP`. ``lowlevel``, ``meminfo``, ``cstrings`` and ``pysqlite_bridge``
pass pointers between Python, jitted code and C libraries. ``preprocessing``, ``digest`` and ``fingerprint`` let numbox
cache the code that it generates at run time and tell one version of that code from another. ``timer`` times calls
made from Python.

numbox.utils.highlevel
----------------------

Dynamically defining StructRef
''''''''''''''''''''''''''''''

Defining a numba
`StructRef <https://numba.readthedocs.io/en/stable/extending/high-level.html#implementing-mutable-structures>`_, a
mutable structure that Python code and jitted code share by reference, requires writing a lot of boilerplate code.
:func:`~numbox.utils.highlevel.make_structref` defines a StructRef type in one call, and the code it generates supports
caching. It needs a type class: an empty subclass of numba's ``StructRef``, registered with numba's ``@register``. numba
uses the type class as the class of the struct's numba type. A type class defined in the script that you run makes every
run after the first fail, as `shown below <If the second run fails with "No conversion"_>`_. Define the type class in a
separate module that your program imports instead, such as ``type_classes.py``::

    from numba.experimental.structref import register
    from numba.core.types import StructRef


    @register
    class DataStructTypeClass(StructRef):
        pass

Then, in a different module, ``main.py``::

    from numba.core.types import float32, unicode_type
    from numpy import isclose
    from numbox.utils.highlevel import make_structref
    from type_classes import DataStructTypeClass


    def derive_output(struct_):
        if struct_.control == "double":
            return struct_.value * 2
        else:
            return struct_.value


    data_struct = make_structref(
        "DataStruct",
        {"value": float32, "control": unicode_type},
        DataStructTypeClass,
        struct_methods={
            "derive_output": derive_output
        }
    )


    if __name__ == "__main__":
        data_1 = data_struct(3.14, "double")
        data_2 = data_struct(2.17, "something else")
        assert isclose(data_1.derive_output(), 6.28)
        assert isclose(data_2.derive_output(), 2.17)

``make_structref`` returns the struct's class, here ``data_struct``. Calling the class builds a struct, in Python or in
jitted code. Each function in ``struct_methods`` becomes a method that Python and jitted code can call, and the method
receives the struct as its first argument. In a third file, ``run.py``, a cached jitted function builds a struct and
calls its method:

.. code-block:: python

    from numba import njit

    from main import data_struct


    @njit(cache=True)
    def use(value):
        return data_struct(value, "double").derive_output()


    print(use(3.14))   # 6.28000020980835

The fields may also be given as a sequence of names, such as ``("value", "control")``. numba then infers the field types
from the arguments of each call to the class, so two calls can build structs whose fields have different types. Each
struct needs a type class of its own: a second ``make_structref`` call with the same class, in the same process, raises
``KeyError: "duplicate registration for <class 'type_classes.DataStructTypeClass'>"``.

If the second run fails with "No conversion"
''''''''''''''''''''''''''''''''''''''''''''

The type class must live in a module that the script imports, not in the script itself. numba pickles the struct's numba
type into its cache, along with the compiled constructor. The pickler stores a class from the running script by value.
The next run therefore reads back a copy of the type class, not the class that the script has just defined. The two
classes have the same name, but numba cannot convert one to the other, and every run after the first fails at the
``make_structref`` call:

.. code-block:: text

    numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
    No conversion from numba.DataStructTypeClass(('value', float32), ('control', unicode_type)) to
    numba.DataStructTypeClass(('value', float32), ('control', unicode_type)) for '$26return_value.5', defined at None

    File ".../numbox-structref/DataStruct_b62da08f4520823e.py", line 45:
    def make_datastruct(value, control):
        return DataStruct(value, control)
        ^

The ``No conversion`` error means that the cached constructor returns a struct of the copied class. numbox writes each
struct's generated code to a file of its own, the struct's cache anchor, which the message quotes (:ref:`where the
generated code is written <utils_generated_code_location>`). Move the type class to an imported module, as
``type_classes.py`` above, and then delete the struct's cache files, which still hold the copy. Their names start with
the anchor's name, here ``DataStruct_b62da08f4520823e``. They are in the ``__pycache__`` directory beside the anchor or,
when ``NUMBA_CACHE_DIR`` is set, in a directory under it whose name starts with ``numbox-structref_``.

If a method cannot find a global name
'''''''''''''''''''''''''''''''''''''

A method's source is copied into the generated code, which runs in a namespace of its own, not in the method's module. A
module or a function that the method reads as a global, such as ``np`` or a jitted helper, is therefore missing there.
The first call of the method fails with a ``TypingError`` whose text includes ``NameError: name 'np' is not defined``.
Pass each such global to ``make_structref`` in its ``ns`` argument, as in ``ns={"np": np}``. The values in ``ns`` that a
method reads become part of the method's fingerprint, which the section on `numbox.utils.fingerprint`_ describes. A new
value therefore changes the generated code, which is compiled again.

After you edit a struct's method
''''''''''''''''''''''''''''''''

``make_structref`` notices the edit: the next run generates new code and compiles it. A cached jitted function in
another file that calls the method, such as ``use`` in ``run.py``, does not notice. Its cache entry holds the compiled
code of the method, and numba checks the entry only against the caller's own file (`numba's caching limitations
<https://numba.readthedocs.io/en/stable/user/jit.html#cache>`_). The caller then returns the old method's result,
with no warning. After you edit a method, delete the cache files of each cached caller in another file. numba names
them after the caller's module, the function's name, the line of its first decorator and the Python version: for
``use``, ``run.use-6.py312.nbi`` and ``run.use-6.py312.1.nbc`` under Python 3.12. The configurations page says
:ref:`where they are <where_the_cache_lands>`.

Compiling a function into a value
'''''''''''''''''''''''''''''''''

A cached jitted function that takes a dispatcher as an argument compiles again in every run, as the section on
`numbox.utils.derive_wap`_ below explains. :func:`~numbox.utils.highlevel.cres` avoids that. It compiles a function for
one signature when it is decorated, as ``njit`` does when it is given a signature. It returns a first-class function
value instead of a dispatcher, and a cached function that takes the value as an argument loads from the cache in the
next run. The same section shows how such a value is called and what becomes of an exception raised in it. A derive is
the function that computes the value of a node in a numbox calculation graph, and a ``cres`` value can serve as one
(:doc:`work page <numbox.core.work>`).

A binding is a jitted function that calls one function of a C library. :func:`~numbox.utils.highlevel.cres_if_available`
builds one for a C function that older versions of the library lack. It takes the library, as the ``ctypes.CDLL`` handle
that ``numbox.core.bindings.utils.load_lib`` returns, and the signature of the decorated function. If the library has a
symbol named after the decorated function, it returns what ``cres`` returns; if not, it returns a stub that raises
``NotImplementedError`` when called. numbox's own bindings use the decorator :func:`~numbox.core.proxy.proxy.proxy`
instead. It compiles a function once, at import, so that every jitted caller calls the one compiled copy, and its
variant ``proxy_if_available`` makes the same check (:ref:`optional C symbols <proxy_optional_c_symbols>`).


.. automodule:: numbox.utils.highlevel
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.derive_wap
-----------------------

A `first-class function value <https://numba.readthedocs.io/en/stable/reference/types.html#functions>`_ is numba's form
of a compiled function as data: jitted code can take one as an argument, read one from a global variable, and call it.

:func:`~numbox.utils.highlevel.cres` returns such a value, and so does the ``.as_func`` attribute of a function
decorated with :func:`~numbox.core.proxy.proxy.proxy` (:ref:`passing the function as a value <proxy_passing_as_value>`).
Both are instances of :class:`~numbox.utils.derive_wap.DeriveWAP`. WAP stands for numba's wrapper address protocol:
``WrapperAddressProtocol`` is numba's base class for the Python objects that jitted code can call as first-class
functions. numbox gives a ``DeriveWAP`` the numba type :class:`~numbox.utils.derive_wap.DeriveFunctionType`, a subclass
of numba's own ``FunctionType``. This page calls numba's type the plain ``FunctionType``.

An exception raised in the body of a ``DeriveWAP`` propagates out of a call in jitted code, as it does out of a call
to a dispatcher. Saved as ``ops.py``:

.. code-block:: python

    from numba import float64

    from numbox.utils.highlevel import cres


    @cres(float64(float64), cache=True)
    def checked_sqrt(x):
        if x < 0.0:
            raise ValueError("negative input")
        return x ** 0.5

In a second file, ``callers.py``:

.. code-block:: python

    from numba import njit

    from ops import checked_sqrt


    @njit(cache=True)
    def apply(f, x):
        return f(x)


    print(apply(checked_sqrt, 4.0))   # 2.0
    try:
        apply(checked_sqrt, -1.0)
    except ValueError as error:
        print(error)                  # negative input

``apply`` loads from the cache in the next run, because the value's numba type,
``DeriveFunctionType[float64(float64)]``, is the same in every process. The type of a dispatcher includes the memory
address of the Python function it wraps, and that address differs from one process to the next. A cached ``apply``
that is passed a dispatcher therefore misses the cache in every run, compiles again and adds a cache entry each time.

How the value carries its entry point
'''''''''''''''''''''''''''''''''''''

numba stores a first-class function value as three addresses. The first is the function's numba entry point, the start
of its code in numba's own calling convention. The second is its C entry point, a wrapper of the kind that ``@cfunc``
builds, and the third is the address of the Python object. When the value holds a numba entry point, a call from jitted
code goes through it, and an exception raised in the body propagates to the caller. Without one, the call goes through
the C entry point, which cannot pass an exception on: numba prints the exception to stderr as ignored, and the call
returns zero (`numba issue 8246 <https://github.com/numba/numba/issues/8246>`_).

numba records the numba entry point only for a dispatcher. A ``DeriveWAP`` records the numba entry point of the function
that ``cres`` or ``@proxy`` compiled. When the value enters jitted code as an argument, numbox fills in the numba entry
point from the recorded address. When jitted code reads the value as a constant, numbox declares the numba entry point
by its symbol name instead, as `described below <Holding the value as a module-level constant_>`_. For a ``cres`` value,
:func:`~numbox.utils.lowlevel.get_func_tuple` shows that the numba entry point is set. numba's own ``_get_jit_address``
returns 0 for the same value, because it resolves an address only for a dispatcher.

Where an exception is still dropped
'''''''''''''''''''''''''''''''''''

A call drops the exception whenever the value reaches jitted code without a numba entry point. That happens in three
cases:

- a ``@cfunc``, passed as a function value;
- a ``CompileResultWAP``, numba's own function value for a compile result, as in
  ``CompileResultWAP(njit(sig)(f).get_compile_result(sig))``, with ``CompileResultWAP`` from
  ``numba.core.types.function_type``;
- a ``DeriveWAP`` passed from Python to a parameter declared with the plain ``FunctionType``, as in
  ``@njit(float64(FunctionType(float64(float64)), float64))`` with ``FunctionType`` from ``numba.types``. numba
  converts the value to the plain ``FunctionType`` on the way in and keeps only its C entry point.

For the last case, let numba infer the parameter's type, or declare it as ``DeriveFunctionType(float64(float64))``, and
the exception propagates. Inside jitted code, by contrast, a ``DeriveWAP`` passed on to a parameter of the plain
``FunctionType`` keeps its numba entry point, because numbox converts the value there without loss. For a ``@cfunc``,
compile the function with ``cres`` instead. For a ``CompileResultWAP``, call
:func:`~numbox.utils.derive_wap.rewrap_derive` from Python, which turns it into a ``DeriveWAP``.
:func:`~numbox.core.work.work.make_work` calls ``rewrap_derive`` for every derive that it receives from Python. A tuple
that mixes ``DeriveWAP`` values with other function values is subject to further limits, which the
:ref:`work page <derive_mixed_function_values>` lists.

Holding the value as a module-level constant
''''''''''''''''''''''''''''''''''''''''''''

numba reads a module-level variable as a compile-time constant, so jitted code can also call the value as a global.
In a third file, ``constant.py``:

.. code-block:: python

    from numba import njit

    from ops import checked_sqrt


    @njit(cache=True)
    def twice(x):
        return checked_sqrt(x) + checked_sqrt(x)


    print(twice(4.0))   # 4.0

This form gives a direct call, which LLVM can inline, and an exception raised in the body propagates as it does through
an argument. numba caches ``twice`` too. numbox declares the body's numba entry point by its symbol name and links the
body's compiled code into ``twice``, so ``twice`` holds no address that changes between processes. numba itself refuses
to cache a caller that holds one of numba's own function values as a constant, and it warns that the caller uses dynamic
globals.

The cost is that the cache entry of ``twice`` holds its own copy of the body. The copy can go stale, as the cached
caller of a struct's method does (`after you edit a struct's method <After you edit a struct's method_>`_). After you
edit ``checked_sqrt`` in ``ops.py``, ``twice`` may keep running the old body. Delete the cache files of ``twice`` after
every edit of the body, here ``constant.twice-6.py312.nbi`` and ``constant.twice-6.py312.1.nbc``. Unless the inlining
matters, pass the value as an argument instead: ``apply`` calls whichever value it is passed, so it always runs the
current body. The proxy page covers the same form for ``.as_func`` (:ref:`holding the function as a module-level
constant <proxy_module_level_constant>`).

.. automodule:: numbox.utils.derive_wap
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.preprocessing
--------------------------

Several numbox functions generate Python source at run time and compile it:
:func:`~numbox.utils.highlevel.make_structref`, :func:`~numbox.core.variable.compile_kernel.compile_kernel`, the work
builder for each :class:`~numbox.core.work.builder.Derived` node, and the SQLite registrations
:func:`~numbox.core.bindings.sqlite.udf_helpers.register_aggregate`,
:func:`~numbox.core.bindings.sqlite.udf_helpers.register_window` and
:func:`~numbox.core.bindings.sqlite.tvf.register_tvf`. numba caches a function only if it can find the function's source
file. ``@njit(cache=True)`` on a function ``g`` compiled from a string raises
``RuntimeError: cannot cache function 'g': no locator available for file '<string>'``.

numbox therefore writes each piece of generated code to an anchor of its own. It compiles the code in memory under the
anchor's file name, so numba finds the file and caches the code against it. ``make_structref`` and the SQLite
registrations write the anchor even with caching off, because numba quotes the code from it in its messages.
``compile_kernel`` and the work builder write one only when they cache. ``@proxy`` works differently. numbox compiles
the code that it generates for a proxied function as though that code stood in your own file. That is why the function's
first decorator must be on line 8 of its file or later (:ref:`where the function may start
<proxy_where_the_function_may_start>`).

.. _utils_generated_code_location:

Where the generated code is written
'''''''''''''''''''''''''''''''''''

The anchors are under ``NUMBA_CACHE_DIR`` when that variable is set, and otherwise under numba's user cache directory,
which the configurations page gives for each platform (:ref:`where the cache lands <where_the_cache_lands>`). Each kind
of code has a subdirectory of its own there: ``numbox-structref``, ``numbox-compile-kernel``, ``numbox-derive``,
``numbox-sqlite-udaf`` and ``numbox-sqlite-tvf``. An anchor's name ends in a digest that covers the generated code and
the functions of yours that the code calls. A new version of the generated code, or of a function of yours that it
calls, therefore gets an anchor of its own. For the struct of the example above, the anchor is a file such as
``numbox-structref/DataStruct_b62da08f4520823e.py``.

numba names the cache files of the functions in an anchor after the anchor, as in
``DataStruct_b62da08f4520823e.make_datastruct-43.py312.nbi``. They are in the ``__pycache__`` directory beside the
anchor. When ``NUMBA_CACHE_DIR`` is set, they are in a directory under it instead, whose name starts with the name of
the anchor's subdirectory, such as ``numbox-structref_38dbc719b23e...``.

numba quotes generated code from its anchor in its messages. A typing error in a struct's method therefore names a file
such as ``numbox-structref/DataStruct_6ed7f92aca43a51f.py`` and a line in it. Open the file to read the code around
that line; the source of your methods is near its end.

:func:`~numbox.core.work.builder.make_graph` writes no anchor for the kernel it generates. It compiles the kernel under
a name that carries a digest, as though the kernel stood in numbox's own module ``numbox.core.work.builder``, so numba
caches it beside that module. Its cache files have names such as ``builder._make_<digest>-2.py312.nbi``, in the
``__pycache__`` directory beside ``builder.py`` or, when ``NUMBA_CACHE_DIR`` is set, in a directory under it whose name
starts with ``work_``. If numba cannot write there, as in a read-only installation, the kernel compiles without a cache
after a warning whose remedy is the one for numbox's own modules (:ref:`where the cache lands <where_the_cache_lands>`).

numbox never deletes an anchor, so every version of the generated code leaves its anchor and its cache files behind.
You can delete them between runs (`numba's notes on clearing the cache
<https://numba.readthedocs.io/en/stable/developer/caching.html#cache-clearing>`_). numbox writes an anchor again when it
needs one, and numba compiles again whatever it can no longer load from the cache.

If numbox warns that numba cannot cache generated code
''''''''''''''''''''''''''''''''''''''''''''''''''''''

With caching on, numbox writes the anchor and asks numba whether it can cache a function of that file before the code
is compiled. If either step fails, the code compiles without a cache, after one warning such as this one:

.. code-block:: text

    RuntimeWarning: numba cannot cache DataStruct_b62da08f4520823e.py here ([Errno 13] Permission denied:
    '.../numbox-structref/DataStruct_b62da08f4520823e.py.tmp-c483cpcq'); it compiles without a cache. Set
    NUMBA_CACHE_DIR to a writable directory, or compile without a cache to silence this warning: "cache" off in the jit
    options this code was given, or in its cache argument where it takes one, or NUMBOX_JIT_OPTIONS='{"cache": false}'
    where the options are the package's

It means that numbox could not write the anchor, so the struct works but compiles again in every process. The
parenthesized text is the underlying error; the list below gives the right ``NUMBA_CACHE_DIR`` for each:

- If the path is too long for the file system, set ``NUMBA_CACHE_DIR`` to a shorter path.
- If the directory's path is not valid UTF-8, set ``NUMBA_CACHE_DIR`` to a UTF-8 path. numba hashes the path as
  UTF-8 to name a directory of its own.
- If the file system does not accept names as long as the ones numba writes for this code, set
  ``NUMBA_CACHE_DIR`` to a directory on a file system that does.
- For any other error, such as the permission error above, set ``NUMBA_CACHE_DIR`` to a directory that you can
  write.

The derives of the work builder fall back to compiling without a cache in the same cases, but without a warning. To
silence the warning, compile the code without a cache: pass ``jit_options={"cache": False}`` to ``make_structref`` or to
``make_graph``, or ``cache=False`` to ``compile_kernel``. ``NUMBOX_JIT_OPTIONS='{"cache": false}'`` turns caching off in
numbox's default jit options, the numba options that numbox compiles with. That silences the warning for every numbox
function that compiles under those options, the SQLite registrations included, since they take no options of their own.

Names longer than 40 bytes
''''''''''''''''''''''''''

The names of an anchor and of its cache files contain the names of the struct and of its fields and methods. Most file
systems allow a name of at most 255 bytes, and a long struct name would overflow that. numbox therefore shortens a name
longer than 40 bytes in UTF-8 before it puts the name into these file names. It keeps as many whole characters as fit in
31 bytes, and it adds an underscore and eight hexadecimal digits of a digest of the whole name, as
:func:`~numbox.utils.preprocessing.bounded_stem` computes. A struct named
``MeasurementsFromTheSecondCalibrationRunOfTheDetector`` has an anchor such as
``MeasurementsFromTheSecondCalibr_67d21543_d1964660f30366df.py``. Its first suffix, ``_67d21543``, comes from shortening
the name; the second, ``_d1964660f30366df``, is the code digest that ends every anchor's name. The struct's
``__name__``, ``__qualname__`` and ``repr`` keep the full name, and the struct caches like any other.

.. _utils_cache_anchor:

Cache-anchor mechanism
''''''''''''''''''''''

Each version of the generated code needs a file of its own, because numba's cache cannot tell some versions apart. numba
checks each cache entry against a stamp of its source file, so an edit to a file of yours invalidates the entries of the
functions in it. It names the cache files after the source file, the function's qualified name and the line of its first
decorator. It keys each entry in those files on the function's signature, the target machine, and hashes of the
function's bytecode and closure values.

That key hashes the function's bytecode but not the tables of constants and names kept beside it, nor the function's
default values. An edit that changes only a literal, a default value or a name can therefore leave the key as it was.
Python 3.14 writes the integers 0 to 255 into the bytecode itself, so on 3.14 an edit that changes such an integer does
change the key.

If one shared file held every version of a struct's code, that file would never change, so its stamp would never
invalidate an entry. Two versions that differ only by such an edit would then have the same cache files and the same
key. The second would load the binary compiled for the first. The digest in the anchor's name rules this out: each
version of the code has a file of its own, and so cache files of its own.

This is also how a new value in ``ns`` changes the generated code. The generated struct code gives each method a
compiled helper, a jitted function that the method calls. numbox names the helper after the method and the method's
fingerprint. A change that the method's source does not show, such as a new value in ``ns``, therefore changes that
name, and with it the generated code and its anchor.


.. automodule:: numbox.utils.preprocessing
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.digest
-------------------

:func:`~numbox.utils.digest.digest` returns a hash of 16 hexadecimal digits for a subject and a list of functions. The
subject is an object whose ``repr`` identifies the code that you generate, such as a type or a tuple of types. Use the
hash to name a cache entry or a generated file for code that you build around functions passed in by a user. numbox does
so for SQLite's aggregate, window and table-valued functions. The hash changes when the subject's ``repr`` changes, when
the fingerprint of one of the functions changes, when numbox's default jit options change, or when numba's version
changes. The section on `numbox.utils.fingerprint`_ below says what a fingerprint covers. A dispatcher in the list is
hashed together with the options it compiles under:

.. code-block:: python

    from numba import njit

    from numbox.utils.digest import digest


    def plus_one(x):
        return x + 1


    print(len(digest("state", [plus_one])))   # 16
    fast = digest("state", [njit(fastmath=True)(plus_one)])
    print(fast == digest("state", [njit(plus_one)]))   # False: the options differ

For a Python function or a dispatcher, the hash is the same in every process. A function that reads
`a value without a fingerprint <When a function reads a value without a fingerprint_>`_ gets a weaker fingerprint, and
the SQLite registrations then miss some edits, as explained
`below <If a SQLite function keeps returning old results_>`_.

If the digest differs in every process
''''''''''''''''''''''''''''''''''''''

A ``functools.partial``, a builtin and a callable object have no code for the fingerprint to read, so ``digest`` hashes
a pickle of the object instead. The pickle is the same in every process when the object's class is defined in an
imported module. When the class is defined in the script that you run, the pickle holds the class itself. It also holds
an identifier that cloudpickle, the pickler that numba uses, draws at random in each process. The pickle, and with it
the digest, then differs in every process, and a cache named after the digest misses on every run. Define such a class
in a module that the script imports.

.. automodule:: numbox.utils.digest
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.fingerprint
------------------------

numbox fingerprints the Python functions that you give it, to tell when code it generated and cached earlier no longer
fits them. It does so for struct methods, ``compile_kernel`` formulas, the work builder's derives and the body of a
``@proxy`` function, and ``digest`` does so for the callbacks of the SQLite registrations. A fingerprint is a text
that describes a function. numbox hashes it into the name of the generated code, so a function with a new fingerprint
gets new code, which is compiled again. The fingerprint of a function is built from:

- its module and qualified name;
- its bytecode, its literals and the names it uses, and the same for each function or comprehension defined
  inside it;
- its default values and the values in its closure;
- the value of each global that it reads by name;
- the value of each module attribute that it reads, such as ``cfg.SCALE``, when the value is a number, a string,
  an array or another kind of data, or a dispatcher;
- for each function or dispatcher among these values, that function's own fingerprint, and the options that a
  dispatcher compiles under.

An edit to any of these changes the fingerprint. That includes the edits that numba's own key misses: a new literal or
default value, a renamed attribute or global, and a new value of a global. Renaming a local variable or moving the
function to another line leaves the fingerprint as it was. Most other edits change it, even an edit to the docstring,
which changes nothing that the function computes. ``digest`` shows the fingerprint at work:

.. code-block:: python

    from numbox.utils.digest import digest


    def scale(x):
        factor = 2.0
        return factor * x


    before = digest("example", [scale])


    def scale(x):
        k = 2.0  # a local variable renamed, on other lines: the same fingerprint
        return k * x


    print(digest("example", [scale]) == before)   # True


    def scale(x):
        k = 3.0  # a new literal: a new fingerprint
        return k * x


    print(digest("example", [scale]) == before)   # False

When a function reads a value without a fingerprint
'''''''''''''''''''''''''''''''''''''''''''''''''''

Some values have no form that means the same thing in another process, so numbox cannot fingerprint them. The ones that
jitted code reads most often are a numba type used as a global, such as ``float64`` or ``intp``, numba's ``prange``, a
class and a NumPy dtype. A builtin function imported by name, such as ``sqrt`` after ``from math import sqrt``, has
none, and neither has an ``@intrinsic``. No jitted helper of ``numbox.utils.lowlevel``, such as ``array_data_p``,
``load_at``, ``store_at`` or ``get_str_from_p_as_int``, has a fingerprint. numbox's bindings, such as ``strlen``, do, so
a function that calls a binding keeps its fingerprint. A jitted function that reads a value without a fingerprint has
none itself.

When a struct method, a ``compile_kernel`` formula or a derive reads such a value, numbox cannot tell one version of the
function from another. The code that depends on the function then compiles without an on-disk cache: it compiles again
in every process and never runs a stale version, and numbox gives no warning. To tell, look for the struct's cache
files: a struct that compiles without a cache has none named after its anchor. To keep such a struct cached, do not
import a value that never changes, such as a numba type or a builtin function, by name. Reach it through its module
instead, as in ``math.sqrt`` after ``import math`` or ``numba.float64`` after ``import numba``. The fingerprint leaves
out a module attribute whose value is not data, so the method then reads no value without a fingerprint, and it has a
fingerprint of its own.

A value that a module provides only through a module-level ``__getattr__`` is not read at all, so a change to it leaves
the fingerprint as it was. Delete the cache files of the code that depends on such a value after it changes. For a
``@proxy`` function, numbox computes from such a fingerprint the name under which callers reach the compiled body. The
proxy page explains what happens when the body reads a value without a fingerprint (:ref:`if numbox warns of an alias
collision <proxy_alias_collision>`).

If a SQLite function keeps returning old results
''''''''''''''''''''''''''''''''''''''''''''''''

A SQLite aggregate, window or table-valued function can keep running code compiled from an earlier version of its
callbacks, with no warning. This happens when a callback reads a value without a fingerprint: ``digest`` then gives the
callback a weaker fingerprint. The weaker fingerprint still covers the callback's code and the values of the globals it
reads by name. It puts the name of each value's type in place of a value without a fingerprint, though, and it leaves
out every module attribute. A later change to such a value, or to a module attribute that the callback reads, keeps
the old anchor, and the registration loads the code compiled before the change.

This is the common case, not a rare one. A step callback receives the number of SQL arguments as ``argc`` and the
address of SQLite's array of pointers to the argument values as ``argv_pp``, as the docstring of
:func:`~numbox.core.bindings.sqlite.udf_helpers.register_aggregate` lists. The usual way to read the arguments,
``carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)``, calls numbox's ``_cast_int_to_void_p``, an
``@intrinsic``, before numba's ``carray`` makes an array of the address. A callback that reads its arguments this way
therefore always gets the weaker fingerprint.

Read the values that a callback depends on as names in the callback's own module, as in ``from config import SCALE``,
not as module attributes such as ``config.SCALE``, which the weaker fingerprint leaves out. This is the reverse of the
advice for a full fingerprint, where reaching a value through its module keeps a struct cached. Delete the
registration's anchor and its cache files after a change that the weaker fingerprint cannot see, such as an edit to a
helper without a fingerprint. The anchor is ``numbox-sqlite-udaf/udaf_<name>_<digest>.py`` for an aggregate,
``numbox-sqlite-udaf/wudaf_<name>_<digest>.py`` for a window function and ``numbox-sqlite-tvf/tvf_<name>_<digest>.py``
for a table-valued function. Its cache files start with the same name (:ref:`where the generated code is written
<utils_generated_code_location>`).

.. automodule:: numbox.utils.fingerprint
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.lowlevel
---------------------

This module holds the small helpers that numbox uses to work with raw memory, most of them jitted functions that Python
code can call as well. :func:`~numbox.utils.lowlevel.load_at` and :func:`~numbox.utils.lowlevel.store_at` read and write
a value of a numba type at an address held as an ``intp``. :func:`~numbox.utils.lowlevel.load_unaligned` and
:func:`~numbox.utils.lowlevel.store_unaligned` do the same at an address that may be misaligned.
:func:`~numbox.utils.lowlevel.array_data_p` returns the address of an array's data as an ``intp``, the type that
numbox's bindings take for a pointer:

.. code-block:: python

    import numpy as np
    from numba import int32

    from numbox.utils.lowlevel import array_data_p, load_at, store_at

    a = np.zeros(2, dtype=np.int32)
    p = array_data_p(a)
    store_at(p + 4, int32(7))
    print(a, load_at(p + 4, int32))   # [0 7] 7

Called on a value, a numba type converts it, so ``int32(7)`` is a 32-bit integer and ``store_at`` writes four bytes.

:func:`~numbox.utils.lowlevel.get_str_from_p_as_int` copies a NUL-terminated C string into a ``str``, one character per
byte. For the UTF-8 bytes of ``"caf\u00e9"``, it returns the five characters ``'caf\xc3\xa9'``. Call the returned string
``s``: ``s.encode("latin-1").decode("utf-8")`` recovers the text. :func:`~numbox.utils.lowlevel.cast` reinterprets a
structref as another structref type over the same memory. :func:`~numbox.utils.lowlevel.get_func_tuple` returns the
three addresses that a first-class function value holds, which the section on `numbox.utils.derive_wap`_ describes. The
functions that take a context and a builder, such as :func:`~numbox.utils.lowlevel.extract_struct_member`, are for code
that you write inside an ``@intrinsic``.

.. automodule:: numbox.utils.lowlevel
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.meminfo
--------------------

Handing a reference through a pointer
'''''''''''''''''''''''''''''''''''''

Some code can hold a structref only as an integer. A C library that keeps a ``void *`` for its caller is one example,
and the slot that holds the state of numbox's SQLite aggregates is another. :func:`~numbox.utils.meminfo.export_meminfo`
returns the address of the structref's MemInfo, numba's reference-counted header for its memory, as an ``intp``.
``export_meminfo`` also takes a reference, which keeps the structref alive.
:func:`~numbox.utils.meminfo.borrow_structref`, called from jitted code or from Python, turns the address back into a
structref of the type you pass it. :func:`~numbox.utils.meminfo.release_meminfo` gives the reference back. With
``main.py`` from the highlevel section:

.. code-block:: python

    from numba import njit, typeof

    from main import data_struct
    from numbox.utils.meminfo import borrow_structref, export_meminfo, get_nrt_refcount, release_meminfo

    d = data_struct(3.14, "double")
    DataStructType = typeof(d)


    @njit
    def read_value(p):
        return borrow_structref(DataStructType, p).value


    p = export_meminfo(d)
    print(get_nrt_refcount(d))   # 2: one reference for d, one for p
    print(read_value(p))         # 3.140000104904175
    release_meminfo(p)
    print(get_nrt_refcount(d))   # 1

The structref stays alive while the exported reference is held, even after Python drops its own. ``borrow_structref``
adds a reference for the structref it returns, and that reference goes away when the structref goes out of scope, so a
borrow leaves the count as it was. Release each exported address once, and do not use it afterward, because the
structref's memory may then be freed. ``export_meminfo`` also accepts a NumPy array, and the address then keeps the
array alive until it is released. ``borrow_structref`` can rebuild a structref from an address, but it cannot rebuild an
array.

After upgrading from numbox 0.7.5 or earlier on numba 0.65 or earlier
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Delete, once, the cache files of each ``cache=True`` function that calls ``export_meminfo`` with an array and was
compiled before the upgrade. numba ignores cache files that another version of numba wrote, so the step is needed only
while numba stays at the same version. Such a function keeps loading its old code from the cache, because numba's key
for it leaves out the code of the functions it calls. In that old code, ``export_meminfo`` takes no reference. An
optimization pass of numba 0.65 and earlier, ``removerefctpass``, removed the reference count change from
``export_meminfo``. The pass removes every reference count change from a function that seems not to need one. The
address that ``export_meminfo`` gives that function then no longer keeps the array alive, and the array's memory can be
freed while the address is in use. numbox 0.7.6 and later mark ``export_meminfo`` so that the pass leaves it alone, and
numba 0.66 removed the pass.

.. automodule:: numbox.utils.meminfo
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.cstrings
---------------------

Allocating C strings for the bindings layer
'''''''''''''''''''''''''''''''''''''''''''

numbox's bindings, such as those in ``numbox.core.bindings.libc`` and ``numbox.core.bindings.sqlite``, take every text
argument as an ``intp`` pointer to a NUL-terminated C string. From Python, :func:`~numbox.utils.cstrings.c_string` makes
one. It copies the UTF-8 encoding of a ``str`` into a new C buffer, yields the buffer's address, and frees the buffer
when the ``with`` block ends:

.. code-block:: python

    from numbox.core.bindings.libc import strlen
    from numbox.utils.cstrings import c_string

    with c_string("caf\u00e9") as p:
        print(strlen(p))   # 5: U+00E9 takes two bytes in UTF-8

The address is valid only inside the block. For several strings at once, open them in one ``with`` statement, as in
``with c_string("pear") as a, c_string("apple") as b:``. When their number is known only at run time, enter each one
into a `contextlib.ExitStack <https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack>`_. ``c_string``
raises ``ValueError`` for a string that contains a NUL character, because the C string would end at that character.

:func:`~numbox.utils.lowlevel.get_unicode_data_p` also returns an address for a ``str``, without making a copy. That
address points into the string's own memory, where CPython stores one, two or four bytes per character, as the widest
character of the string needs. The bytes there are the text's UTF-8 only for ASCII text. CPython stores ``"caf\u00e9"``
with one byte per character, so its last character is the single Latin-1 byte ``0xE9``, not the two UTF-8 bytes
``0xC3 0xA9``.

From jitted code
''''''''''''''''

``c_string`` works only in Python. numba does not compile a ``with`` statement over it and raises
``UnsupportedBytecodeError``, with the message ``The 'with (context manager) as (variable):' construct is not
supported``. For jitted code, encode the text in Python, add a NUL byte, and pass the bytes as a ``uint8`` array.
:func:`~numbox.utils.lowlevel.array_data_p` gives the array's address inside the jitted function:

.. code-block:: python

    import numpy as np
    from numba import njit

    from numbox.core.bindings.libc import strlen
    from numbox.utils.lowlevel import array_data_p


    @njit
    def c_length(buf):
        return strlen(array_data_p(buf))


    buf = np.frombuffer("caf\u00e9".encode("utf-8") + b"\0", dtype=np.uint8)
    print(c_length(buf))   # 5

.. automodule:: numbox.utils.cstrings
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.pysqlite_bridge
----------------------------

Bridging Python ``sqlite3`` connections to the bindings layer
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

numbox's SQLite bindings take a connection as the ``sqlite3 *`` handle of SQLite's C API. A connection that Python's
`sqlite3 <https://docs.python.org/3/library/sqlite3.html>`_ module opens keeps that handle in a private field of its C
structure. :func:`~numbox.utils.pysqlite_bridge.extract_connection_ptr` reads the field with ``ctypes`` and returns the
handle as an ``int``, so Python code and the bindings can use one connection:

.. code-block:: python

    import sqlite3

    from numbox.core.bindings.sqlite.conn import sqlite3_changes
    from numbox.utils.pysqlite_bridge import extract_connection_ptr

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t(x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1), (2), (3)")
    db_p = extract_connection_ptr(conn)
    print(sqlite3_changes(db_p))   # 3, the rows that the last INSERT changed

The handle belongs to ``conn``, and an ``int`` cannot keep ``conn`` alive. Keep ``conn`` open and referenced for as
long as any code uses the handle: after ``conn.close()``, or once ``conn`` is garbage-collected, the handle points to
freed memory.

numbduck is a sister package that brings DuckDB's C API to numba. Its
`pybridge <https://github.com/Goykhman/numbduck/blob/main/numbduck/pybridge.py>`_ module reads a DuckDB connection's
handle the same way, and ``extract_connection_ptr`` follows its approach.

If extract_connection_ptr raises RuntimeError
'''''''''''''''''''''''''''''''''''''''''''''

Before it reads the field, the function checks with :func:`~numbox.utils.pysqlite_bridge.libraries_coordinated` that
numbox's bindings and Python's ``sqlite3`` module report the same SQLite version. The two can use different copies of
SQLite on macOS, where the system's copy is mapped into every process and can take precedence over the copy that
Python uses. A handle from one copy passed to the other can crash the process, so the function raises ``RuntimeError``
instead, and the message names both versions. The module docstring below gives the remedy, which is to load Python's
copy first with ``DYLD_INSERT_LIBRARIES``.

After reading the handle, the function checks it by calling ``sqlite3_errmsg``, and it raises ``RuntimeError`` unless
SQLite answers ``not an error``. A closed connection raises ``extracted null sqlite3* from sqlite3.Connection`` before
that call. A live connection whose last statement failed raises, too, with the text
``extracted connection pointer failed validation: b'near "SELEC": syntax error'`` after a failed ``SELEC 1``; the
function works again once a statement on the connection succeeds. An argument that is not a ``sqlite3.Connection``
raises ``TypeError``.

.. automodule:: numbox.utils.pysqlite_bridge
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.timer
------------------

The ``timer`` decorator, an instance of :class:`~numbox.utils.timer.Timer`, times calls made from Python, as the module
docstring below describes. It works on a jitted function too. The first call of a jitted function includes numba's
compilation, and ``Timer.times`` keeps only the last call's time, so read it after a later call. The wrapper that
``timer`` returns is a plain Python function, not a dispatcher, so jitted code cannot call it. A call from jitted code
to a timed function named ``total`` fails with numba's
``Untyped global name 'total': Cannot determine Numba type of <class 'function'>``.

.. automodule:: numbox.utils.timer
   :members:
   :show-inheritance:
   :undoc-members:
