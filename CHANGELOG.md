# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `numbox.core.read_only_cache.ReadOnlyCacheLocator`, a cache locator for a `NUMBA_CACHE_DIR` that is read-only at run
  time, for example in a container image. It is listed in numba's `NUMBA_CACHE_LOCATOR_CLASSES` in place of numba's
  `UserProvidedCacheLocator`. `UserProvidedCacheLocator` skips a directory that cannot be written in.
  `ReadOnlyCacheLocator` also accepts a cache subdirectory that exists, that the process can search and that cannot be
  written in, and numba then loads the entries that the subdirectory holds. Where the subdirectory can be written in,
  `ReadOnlyCacheLocator` and `UserProvidedCacheLocator` behave alike, so the cache can be built under the same value of
  `NUMBA_CACHE_LOCATOR_CLASSES`. ([#53])
- `numbox.core.read_only_cache.cannot_be_written(error, location)`, which reports whether `error` is the error of a
  write that the operating system denied in the directory `location`. It returns True for an `OSError` whose error
  number is `EROFS`, `EACCES` or `EPERM`, if `location` is a directory that the process can search. ([#53])
- A section of the `numbox.core.configurations` documentation page on a cache directory that is read-only at run time.
  ([#53])

### Changed

- `check_cache_location` in `numbox.core.configurations` accepts a location that exists, that the process can search and
  that cannot be written in. numbox therefore keeps caching enabled for its own functions in such a location, and numba
  loads the entries that the location holds. Before, numbox emitted a `RuntimeWarning` for such a location and compiled
  its own functions without a cache. ([#53])
- When a cache save fails with `EROFS`, `EACCES` or `EPERM` in a cache directory that exists and that the process can
  search, numbox skips the failed save. numbox does so for every function that numba caches in the process. It emits one
  `RuntimeWarning` per process, for the first skipped save. A function whose save is skipped compiles again in every
  process. Before, the call that compiled the function raised the error of the failed save. On Windows, numba itself
  ignores a save that fails with `EACCES`, and numbox emits no warning for that save. ([#53])
- numba's `Dispatcher.recompile()` empties the index file of a function before compiling the function again. When the
  write that empties the index file fails in a cache directory that cannot be written in, numbox disables the cache of
  the function for the rest of the process. `explain_cache_key` in `numbox.core.cache_key` then raises `ValueError` for
  the function. ([#53])
- The warning for generated code that numbox cannot cache names a new remedy when the cache directory cannot be written
  in and holds no file for that code. The remedy is to build the cache again with a run that generates the code. numbox
  gives this remedy only while `NUMBA_CACHE_DIR` is set and `NUMBA_CACHE_LOCATOR_CLASSES` lists a class that is not one
  of numba's own locators. ([#53])

## [0.7.9] - 2026-11-05

### Changed

- The documentation pages for the proxy, cache key, work, variable, bindings, configurations and utils modules were
  rewritten for a reader who knows numba and is new to numbox. The examples on these pages can be copied and run without
  changes. Every public function, class and method has a docstring. The docstring of each binding links to the reference
  for its C function. ([#50], [#51])
- The docstrings added in this release change each binding's symbol name. A cached caller of a numbox binding therefore
  recompiles once after the upgrade, and numbox emits a `StaleProxyCacheWarning` for each cache entry that it discards.
  numbox does not recompile a caller that holds a binding's `.as_func` as a module-level constant, so the user must
  delete the cache files of that caller. ([#51])

### Fixed

- A function decorated with `@proxy` keeps its docstring, so `help()` and Sphinx autodoc show it. Before, the `__doc__`
  of the returned dispatcher was `None`. ([#51])
- numbox wraps five of numba's caching methods when numbox is imported. If wrapping one of the methods fails, numbox now
  restores the methods that it has already wrapped, and the next import of numbox tries to wrap the methods again.
  Before, numbox left those methods wrapped, and no later import tried to wrap the methods again. No supported numba
  version causes the wrapping to fail. ([#49])

## [0.7.8] - 2026-10-27

### Added

- `numbox.core.cache_key`, a module that extends numba's on-disk cache key with the jit flags, the `NUMBA_*` settings
  that change emitted code, and the numbox version. Before, a run with `NUMBA_BOUNDSCHECK=1`, for example, could load a
  binary that was cached without bounds checks. ([#48])
- `numbox_njit` and `numbox_cfunc` in `numbox.core.cache_key`. `numbox_njit` takes the arguments of `numba.njit`, and
  `numbox_cfunc` takes the arguments of `numba.cfunc`. A function that is compiled with either of them and `cache=True`
  is cached under the extended key. ([#48])
- `explain_cache_key` in `numbox.core.cache_key`. It returns the parts that numbox adds to a function's cache key. These
  parts show why a function missed the cache. ([#48])
- `UncachedWarning`, a subclass of `RuntimeWarning`, and the tables `KEYED_SETTINGS` and `UNKEYED_SETTINGS` in
  `numbox.core.cache_key`. ([#48])
- A documentation page for `numbox.core.cache_key`. ([#48])
- A `package` argument for `cache_remedy`, with the default `"numbox"`. A package built on numbox can pass its own name,
  so that the remedy tells users to reinstall that package. ([#44])

### Changed

- The supported numba range is now `>=0.63.1,<0.69.0`. It was `>=0.60.0,<0.68.0`. This adds numba 0.68 and drops numba
  0.60.0 through 0.63.0. ([#45])
- Everything numbox compiles is cached under the extended key: its bindings, `make_structref` structs, `compile_kernel`
  kernels, the derive functions of `Derived` nodes, and functions passed to `@proxy` and `cres`. Each cached function
  recompiles once after this upgrade and after every later numbox upgrade. ([#48])
- Importing any module under `numbox.core` wraps five of numba's caching methods for the whole process. Functions whose
  cache key numbox does not extend keep numba's own key. ([#48])
- numbox compiles a function without an on-disk cache when one of the function's jit options has a value that numbox's
  extended key cannot represent. numbox then emits `UncachedWarning`. The extended key cannot represent a `parallel`
  dict, a pipeline class other than numba's default, a function given as the `inline` cost model, or a `locals` type
  whose name contains a memory address. Before, numba used its own key to cache a function that had one of these values.
  numba's own key ignores these jit options. ([#48])
- The remedy in the import-time warning "numba cannot cache numbox here" matches the cache location that numba chose,
  and takes `NUMBA_CACHE_LOCATOR_CLASSES` into account. Before, the remedy could suggest a step that could not help.
  ([#46])
- `is_a_cache_error` also returns true for the `UnicodeEncodeError` that numba raises for a path that is not valid
  UTF-8. On numba 0.68 and later, it also returns true for the `KeyError` that numba raises for a `.zip` member whose
  source is not in the archive. ([#45], [#46])

### Removed

- `numbox.core.configurations.function_struct_has`, with no replacement. numbox used it to test for the `jit_addr` slot,
  which is always present on the supported numba versions.
- `numbox.utils.derive_wap.jit_addr_supported`, with no replacement. On the supported numba versions it would always
  return `True`, so callers can remove the check. `cres` and the `.as_func` of a `@proxy` binding always return a
  `DeriveWAP`.

### Fixed

- Importing numbox from a directory whose name is not valid UTF-8 no longer fails with `UnicodeEncodeError` when
  `NUMBA_CACHE_DIR` is set or `__pycache__` cannot be written. numbox compiles without a cache and emits its import-time
  warning. ([#46])
- Code generated by `make_structref`, `compile_kernel`, the work builder and the SQLite registration helpers compiles
  without a cache, with a `RuntimeWarning`, in three more cases. In the first, the cache directory is too close to the
  path length limit. In the second, the file system's limit on file names is shorter than numba's cache file names. In
  the third, the path of the directory that holds the generated code is not valid UTF-8. Before, these cases failed with
  `OSError` or `UnicodeEncodeError`. ([#46])
- `cache_remedy` accepts a source file given as a `Path` or as bytes. `cache_remedy` also accepts a `FileNotFoundError`
  that carries no file name, and a `FileNotFoundError` whose file name is given as bytes, as a `Path` or as an integer.
  Before, `cache_remedy` raised an exception in these cases. ([#46])

## [0.7.7] - 2026-10-04

### Added

- `check_cache_location`, `is_a_cache_error`, `cache_remedy`, `uncached_where_no_cache_can_be_written` and the constant
  `LONGEST_CACHE_FILE_NAME` in `numbox.core.configurations`. A package built on numbox can use them to test whether
  numba can cache a function that is defined in a given file. When numba cannot cache the function, the package can use
  them to produce the text of the remedy. ([#42])
- `numbox.utils.preprocessing.bounded_stem`, which shortens a name to at most 40 UTF-8 bytes for use in a file name. A
  longer name becomes a prefix plus a digest. ([#42])
- A documentation page for `numbox.core.configurations`. It covers `NUMBOX_JIT_OPTIONS`, where numba puts its cache on
  each platform, and the fallback to compiling without a cache. ([#42], [#43])

### Changed

- `NUMBOX_JIT_OPTIONS` is validated when `numbox.core.configurations` is imported. A value that is not a JSON object
  raises `ValueError`, and so does a `"cache"` entry that is not `true` or `false`. Before, `{"cache": "false"}` left
  caching on without a message. ([#42])
- `compile_kernel` emits a `RuntimeWarning` that names the remedy when it cannot cache a kernel. The warning was a
  `UserWarning`. ([#42])

### Fixed

- Importing numbox works in an environment where numba cannot write a cache for numbox. Examples are an import from an
  `.egg`, `.whl` or `.pyz` archive, and a read-only install whose user cache directory is also unwritable. Before, the
  import failed with `RuntimeError: cannot cache function`. numbox now emits one `RuntimeWarning` that names the remedy,
  and compiles without a cache for the rest of the process. Setting `NUMBOX_JIT_OPTIONS='{"cache": false}'` turns
  caching off and silences the warning. ([#42])
- `make_structref`, `compile_kernel`, `make_graph`, `register_aggregate`, `register_window` and `register_tvf` compile
  their generated code without a cache, with a `RuntimeWarning`, when numba cannot cache it. Before, they failed with
  `OSError` or `RuntimeError`. ([#42])
- `make_structref` accepts long struct names, field names and method names. Before, a struct name of about 93 characters
  raised `OSError: File name too long`. A struct gets new cache file names if its struct name, one of its field names or
  one of its method names is longer than 40 bytes. That struct recompiles once after the upgrade. ([#42])

## [0.7.6] - 2026-09-29

### Added

- Documentation for `numbox.utils.meminfo`. It explains how `export_meminfo`, `borrow_structref` and `release_meminfo`
  pass a reference through an integer pointer. ([#41])

### Changed

- Importing `numbox.utils.timer` no longer configures the root logger. `Timer` and `timer` log each duration at `INFO`,
  not `WARNING`, so timings are silent until the application enables `INFO` for the `numbox.utils.timer` logger. ([#41])

### Fixed

- On numba 0.65 and earlier, `export_meminfo` keeps the reference that it takes to a NumPy array. Before, numba's
  reference count pruning removed that reference, so the returned pointer did not keep the array alive. A `cache=True`
  caller compiled with an earlier numbox release keeps the old behavior until its cache files are deleted. ([#41])
- A query that names a hidden argument column of a table-valued function registered with `register_tvf` returns the
  value of that argument, as in `SELECT arg0 FROM f(1, 4)`. Before, the query read past the column buffers and returned
  arbitrary values. ([#41])
- A query on a table-valued function can filter a hidden argument column in `WHERE`, as in `SELECT n FROM f(2, 5) WHERE
  arg0 = 2`. Before, the statement failed to prepare. ([#41])
- A table-valued function with 64 or more arguments answers queries. Before, it registered without an error and every
  query on it failed. ([#41])

## [0.7.5] - 2026-09-15

No user-facing changes. The release contains a private helper and a change to the documentation workflow.

## [0.7.4] - 2026-09-07

### Added

- `numbox.core.proxy.proxy.AliasCollisionWarning`, a subclass of `RuntimeWarning`. numbox emits it when `@proxy` would
  register two different bodies under the same symbol name. In that case nothing that is stable across processes
  distinguishes the two bodies. The second body then gets a name that is local to the process, and the callers of both
  bodies recompile in every process. numbox also emits the warning when a `proxy_if_available` binding is absent and a
  body in the process already has the binding's symbol name. The symbol name then continues to refer to the body that
  already has the name. ([#39])

### Changed

- The supported numba range is now `>=0.60.0,<0.68.0`, which adds numba 0.67. It was `>=0.60.0,<0.67.0`.
- On numba 0.61 and later, the `.as_func` of a `@proxy` binding is a `DeriveWAP`. An exception raised in the proxied
  body propagates out of a call made through `.as_func`, for example when `.as_func` is the derive of a `Work` node. A
  direct call of the binding from jitted code still discards the exception. ([#39])
- On numba 0.61 and later, numba caches a `cache=True` function that references the `.as_func` of a binding as a global.
  Before, numba refused to cache such a function. After an edit to the proxied body, the function can keep running the
  old code until its cache files are deleted. ([#39])
- On numba 0.61 and later, a tuple that holds both the `.as_func` of a binding and a plain `CompileResultWAP` fails to
  compile. ([#39])
- `compile_kernel` and `numbox.utils.digest.digest` can fingerprint four more kinds of value that a formula captures: an
  `ExternalFunction`, a ctypes function taken from a loaded library, a `cres` function and the `.as_func` of a binding.
  A kernel whose formula captures one of these values is cached on disk. Before, such a kernel recompiled in every
  process. ([#39])
- The symbol name of a user's `@proxy` body changes once if the body captures an `ExternalFunction`, a ctypes function
  taken from a loaded library, a `cres` function or the `.as_func` of a binding. The cached callers of that body
  recompile once after the upgrade, and numbox emits a `StaleProxyCacheWarning` for each cache entry that it discards.
  The symbol names of the bindings that numbox ships do not change. ([#39])

### Fixed

- Two `@proxy` bindings made by one factory over different C functions, such as `ExternalFunction("floor", ...)` and
  `ExternalFunction("ceil", ...)`, no longer share one symbol name. Before, a jitted caller could get the other
  function's result without an error. ([#39])
- `proxy_if_available` no longer overwrites a working binding with the raising placeholder that it registers for a
  missing C symbol. Before, `proxy_if_available` overwrote the working binding when one function was bound first to a
  library that has the symbol and then to a library that lacks the symbol. A jitted caller that was compiled after the
  second binding then got a wrong result without an error. ([#39])

## [0.7.3] - 2026-08-16

### Added

- `numbox.utils.derive_wap`, first-class function values whose exceptions propagate. `DeriveWAP` is a subclass of
  numba's `CompileResultWAP`, and `DeriveFunctionType` is the numba type of a `DeriveWAP`. `jit_addr_supported()`
  reports whether the installed numba supports `DeriveWAP`, and `rewrap_derive(derive)` converts a plain
  `CompileResultWAP` into a `DeriveWAP`. ([#38])
- `numbox.core.configurations.function_struct_has(field)`, which reports whether numba's data model for first-class
  functions has a slot of that name. `function_struct_size` and `numba_version` moved to `numbox.core.configurations`
  and stay importable from `numbox.utils.lowlevel`. ([#38])

### Changed

- On numba 0.61 and later, an exception raised inside a `Work` derive propagates out of `calculate` with its type and
  message. The node's `data` is left as it was and the node can be calculated again. Before, the exception was printed
  to standard error, `data` was zero-filled and the node was marked as derived. The exception is still discarded in
  three cases. It is discarded when the derive is a `cfunc`. It is discarded when the derive is a plain
  `CompileResultWAP` and the node is made inside jitted code. It is discarded when Python passes the derive to a jitted
  function whose parameter is declared as a plain `FunctionType`. ([#38])
- A `calculate` call that is made from inside a `prange` body does not raise the exception that the called node's derive
  raised. On Linux, the call raises numba's `SystemError`, with the derive's exception as its `__cause__`. On macOS, the
  call raises nothing. On both operating systems, the node keeps its `data` and can be calculated again. ([#38])
- On numba 0.61 and later, `numbox.utils.highlevel.cres` returns a `DeriveWAP`. When jitted code calls a `cres` function
  and the function raises, the exception now propagates to the caller. The exception also propagates when a `cres`
  function is used as a `compile_kernel` formula. Before, the `cres` function returned zero in both cases. ([#38])
- `numbox.core.work.work.make_work` is a plain Python function, not a numba dispatcher. Jitted callers reach it through
  an overload, and `make_work.py_func` is kept. ([#38])
- On numba 0.61 and later, a tuple that holds a `cres` function together with a plain `CompileResultWAP`, a
  signature-declared `njit` dispatcher or a `cfunc` fails to compile, because `cres` functions have their own numba
  type. Tuples whose elements are all `cres` functions are not affected. ([#38])
- On numba 0.61 and later, a `cache=True` function that calls a `cres` function referenced as a global is written to
  numba's cache. Before, numba refused to cache the `cache=True` function. If the `cres` function is defined in another
  module and is edited, the `cache=True` function runs the stale code until the cache files of the `cache=True` function
  are deleted. ([#38])
- The upgrade does not refresh existing numba cache entries. A cached user function that takes a plain `FunctionType`
  derive keeps discarding the exceptions of the derive until its cache files are deleted. ([#38])

### Fixed

- Reading the `data` of a `unicode_type` `Work` node from Python no longer crashes the interpreter after the node's
  derive has raised an exception that propagated out of `calculate`. In the cases where the exception is still
  discarded, reading the `data` still crashes the interpreter. ([#38])

## [0.7.2] - 2026-08-15

### Changed

- `Any.get_as(ty)` in `numbox.core.any.any_type` no longer checks `ty` against the type the value was stored with.
  Before, a mismatch raised `NumbaError`. The stored bytes are now read as `ty` without an error, so the caller must
  keep track of the stored type.
- `numbox.core.work.combine_utils.load_dict_into_array` no longer raises when the stored type of a value differs from
  the type of the record field. It reads each value with `Any.get_as`, which no longer checks the stored type.

### Removed

- The `type_info` property of `Any`, and the `t` field of `AnyType` that held the name of the stored type.
- The module `numbox.core.any.any_lite`, which was added in 0.7.1. `numbox.core.any.any_type` now behaves as `any_lite`
  did.

## [0.7.1] - 2026-08-11

### Added

- `numbox.core.any.any_lite`, a lighter variant of `numbox.core.any.any_type`. Its `Any` stores no type information, and
  its `get_as` does not check the type.
- An "Exception handling" section in the `numbox.core.work` documentation. It states that an exception that escapes a
  derive is discarded, which leaves the node's `data` zero-filled. ([#36])

## [0.7.0] - 2026-07-31

### Added

- `StaleProxyCacheWarning` and `UnvalidatedProxyCacheWarning` in `numbox.core.proxy.proxy`, both subclasses of
  `RuntimeWarning`. numbox emits `StaleProxyCacheWarning` when a cached function references a `@proxy` symbol that this
  process never registered. numbox then discards the cache entry, and numba recompiles the function. numbox emits
  `UnvalidatedProxyCacheWarning` once per process when it cannot read a cache entry to check the entry's `@proxy`
  symbols. numba then loads the entry unchecked. ([#35])
- `StaleProxyCacheError` and `UnvalidatedProxyCacheError` in `numbox.core.proxy.proxy`, both subclasses of
  `RuntimeError`. In strict mode, which the `NUMBOX_PROXY_CACHE_STRICT` environment variable turns on, numbox raises
  them in place of the two warnings. ([#35])
- The `NUMBOX_PROXY_CACHE_STRICT` environment variable. Any value other than empty, `0`, `false`, `no` or `off` turns
  strict mode on. ([#35])

### Changed

- Importing `numbox.core.proxy.proxy` installs a check on every load from numba's cache in the process. If a cached
  function references a `@proxy` symbol that the process cannot call, numba recompiles the function instead of loading
  it from the cache. ([#35])
- The symbol name that `@proxy` registers for a binding now also depends on the body of the decorated function and on
  the `jit_options`. Before, the name depended only on the module, qualified name and signature of the decorated
  function. Every cached caller of a numbox binding recompiles once after the upgrade, and numbox emits a
  `StaleProxyCacheWarning` for each cache entry that it discards. ([#35])
- `compile_kernel` emits a `UserWarning` and compiles without an on-disk cache when a formula is an `@njit(cache=True)`
  dispatcher or references one. `compile_kernel` emits the same warning when its own jit flags cannot be fingerprinted.
  In that case `compile_kernel` already compiled without a cache before this release, and only the warning is new.
  ([#35])
- `make_structref_code_txt` returns a 3-tuple `(code_txt, fields_types, cacheable)`. Before, it returned `(code_txt,
  fields_types)`. ([#35])
- `make_structref` compiles a struct without an on-disk cache when one of its methods, or an `ns` value that a method
  reads, cannot be fingerprinted. ([#35])
- `make_graph` compiles a derive without an on-disk cache when the derive reads a module global or cannot be
  fingerprinted. It compiles the whole graph without one when a derive or a jit flag cannot be fingerprinted, or when
  the type of a node holds a dispatcher. ([#35])

### Fixed

- A `cache=True` function that calls a `@proxy` binding defined in another file no longer terminates the process after
  the binding's signature changes. Before, every run ended with a segmentation fault or an `LLVM ERROR: Symbol not
  found` abort until the user deleted the numba cache files. ([#35])
- A cached caller that was compiled while a `proxy_if_available` binding was present no longer crashes when numba loads
  the caller in a process where the binding's C symbol is absent. ([#35])
- Two `@proxy` bindings with the same module, qualified name and signature no longer share one symbol name when their
  bodies or `jit_options` differ. Before, a caller could run the other binding's code. ([#35])
- `make_graph` no longer reuses a kernel that was cached under different jit flags. Before, a graph built with
  `error_model="numpy"` and one built with the default error model shared a cache entry. ([#35])
- `make_graph` accepts a derive function defined at the interactive prompt or with `exec`. Before, it raised `OSError`.
  ([#35])
- `make_graph` no longer misses its cache on every run for a graph with an array-of-record node. ([#35])
- `make_graph` no longer adds a cache entry on every run for a graph with a dispatcher-typed node. Such a graph now
  compiles without an on-disk cache. ([#35])
- `make_structref` no longer loads a stale cached struct after a change to something that one of its methods reads: a
  global, a closure value or an `ns` entry. ([#35])
- `hash_type` returns the same value in every process for record and dispatcher types. ([#35])
- `compile_kernel` no longer loads a stale cached kernel after one of four changes. The changes are a new value of a
  module attribute that a formula reads, a new layout of a structured dtype constant, new options of a `@vectorize`
  formula, and a new value of a numba code generation setting such as `NUMBA_BOUNDSCHECK`. ([#35])
- `numbox.utils.digest.digest` gives different results for jitted functions that differ only in jit options such as
  `fastmath`. ([#35])
- `register_aggregate` and `register_window` no longer share a cached implementation between two state types that have
  the same class name and are defined in different modules. ([#35])
- `register_tvf` no longer adds an entry to numba's on-disk cache index on every registration. ([#35])

## [0.6.3] - 2026-07-16

### Fixed

- `sqlite3_value_frombind` in `numbox.core.bindings.sqlite.value` is a stub that raises `NotImplementedError` when the
  loaded SQLite library is older than 3.28, because such a library does not have this function. Before, importing
  `numbox.core.bindings.sqlite.value` failed with such a library. Importing the `vtable` and `tvf` modules, which import
  `value`, failed as well.

## [0.6.2] - 2026-07-08

### Changed

- A call from jitted code to a `proxy_if_available` binding whose C symbol is missing from the loaded library raises a
  `TypingError` that names the binding and the symbol. Before, numba reported an untyped global without the reason. A
  call of such a binding from Python still raises `NotImplementedError`. ([#34])

## [0.6.1] - 2026-07-07

### Added

- The `numbox.core.bindings.sqlite` subpackage, which holds the SQLite binding modules, with its own documentation page.
  ([#33])

### Changed

- Six binding modules under `numbox.core.bindings` were renamed, and importing the old module paths fails. `_c` became
  `libc`, `_math` became `libm`, `_errno` became `errno`, `_fmtio` became `fmtio`, `_stdio` became `stdio` and
  `_strerror` became `strerror`. Each module exports the same names as before. ([#33])
- The SQLite binding modules moved into `numbox.core.bindings.sqlite` and lost the `_sqlite_` prefix. For example,
  `numbox.core.bindings._sqlite_conn` became `numbox.core.bindings.sqlite.conn`. The modules `bind`, `blob`, `column`,
  `constants`, `exec`, `hooks`, `query`, `result`, `stmt`, `tvf`, `udf`, `udf_helpers`, `value` and `vtable` moved in
  the same way. `_sqlite_typemap` became `numbox.core.bindings.sqlite._typemap`. ([#33])
- Importing one binding module no longer imports and compiles every binding. ([#33])

### Removed

- The re-exports from `numbox.core.bindings`. An import such as `from numbox.core.bindings import cos, sqlite3_open,
  SQLITE_OK` fails. Import each name from its module, for example `from numbox.core.bindings.libm import cos`. ([#33])

## [0.6.0] - 2026-07-03

### Changed

- The supported numba range is now `>=0.60.0,<0.67.0`, which adds numba 0.66. It was `>=0.60.0,<0.66.0`.

## [0.5.23] - 2026-06-30

### Fixed

- A cached jitted function that calls a `@proxy` function no longer aborts with `LLVM ERROR: Symbol not found` when
  numba loads it from the cache. Before, the abort occurred when separate processes built their caches at the same time.
  Most of the C bindings in numbox are `@proxy` functions. ([#32])

## [0.5.22] - 2026-06-29

### Changed

- `printf`, `fprintf` and `snprintf` check the literal format string of a jitted call against its arguments at compile
  time. A wrong number or kind of arguments raises `TypingError`. Before, such a call printed a wrong value or crashed.
  A `%s` argument may be a string or an `intp` or `uintp` pointer. ([#31])
- `printf`, `fprintf` and `snprintf` reject the glibc `'` flag for thousands grouping, with `TypingError` in jitted code
  and `ValueError` in Python. ([#31])
- `Variable` raises `ValueError` when its `name` contains a dot, because the dot separates the parts of a qualified
  name. ([#31])
- `@proxy` raises `ValueError` when the decorated function is defined on line 12 or earlier of its source file. ([#31])

### Fixed

- `get_str_from_p_as_int` in `numbox.utils.lowlevel` reads C strings that contain bytes from 0x80 to 0xFF. Before, such
  a byte raised `ValueError` in jitted code. Each byte maps to one code point, and UTF-8 is not decoded. ([#31])
- A query on a table-valued function registered with `register_tvf` fails with an SQLite error when the callback raises.
  Before, the query succeeded and returned no rows. ([#31])
- `get_input` on `Node` and `Work` raises `NumbaError` for a negative index. Before, `get_input(-1)` returned the last
  input. ([#31])
- `explain` in `numbox.core.work.explain` works on graphs built with `make_work` or `ll_make_work`. For each derive
  function of such a graph, the output shows a placeholder in place of the function's name and source code. Before,
  `explain` raised `AttributeError`. ([#31])
- With caching on, `make_graph` no longer reuses a cached kernel for two graphs that differ only in the declared type of
  a node. ([#31])
- `make_structref` accepts a method whose signature is not reproduced by `repr()`, such as one with a default written
  `1_000` or with an annotation. Before, it raised `AssertionError`. ([#31])
- `_call_lib_func`, the internal helper that the bindings are built on, lays out tuple arguments and return values with
  alignment padding. Before, `_call_lib_func` summed the field widths. That total was too small for a tuple such as
  `(int32, int64)`. The calling conventions of SysV x86-64 and AAPCS64 classify each 8-byte unit of a struct, called an
  eightbyte. An eightbyte that holds both integer and floating-point fields is now classified as INTEGER. Before, one
  floating-point field was enough to classify the eightbyte as SSE. A tuple field with no fixed bit width raises
  `TypingError`. No binding shipped with numbox was affected. ([#31])

## [0.5.21] - 2026-06-21

### Added

- `register_table` accepts a mapping of column names to 1-D NumPy arrays of equal length. It exposes them as a
  read-only, zero-copy columnar SQLite virtual table. The keys of the mapping name the columns, and passing `columns` as
  well raises `ValueError`. ([#28])
- `numbox.utils.pysqlite_bridge.libraries_coordinated()`, which reports whether the SQLite version that numbox loaded
  equals `sqlite3.sqlite_version`. ([#28])

### Changed

- The third parameter of `register_table` was renamed from `arr` to `data`. A call that passes `arr=` raises
  `TypeError`. ([#28])

### Fixed

- A `register_table` virtual table returns every matching row when an integer column is compared with a REAL value.
  Before, the virtual table truncated the REAL value to an integer, so `WHERE c < 4.5` dropped the row with `c = 4`.
  ([#28])

## [0.5.20] - 2026-06-18

### Added

- `Params` in `numbox.core.variable.variable`. A `Variable` holds a `Params` value in its `params` attribute. The
  `jitable` field of `Params` declares whether the formula of the variable can be jitted, and the `type` field gives the
  numba type of the variable. ([#27])
- `External.declare(name, params)`, which declares a typed external input before compilation. ([#27])
- `compile_kernel` builds a fully declared graph during the `compile_kernel` call and raises type errors then. Before,
  both happened at the first call of the kernel. The new `CompiledKernel.is_declared` attribute is `True` for a fully
  declared graph. Graphs that are not fully declared behave as before. ([#27])
- On a declared kernel, `CompiledKernel.recompute` raises `ValueError` for a changed value that cannot be converted to
  the declared type. ([#27])

### Changed

- `compile_kernel` no longer raises `ValueError` for an external variable that has a formula. The kernel takes such a
  variable as an input, whereas `CompiledGraph` computes it from the formula. ([#27])

### Fixed

- `CompiledKernel.recompute` raises `RuntimeError` when the first call of `kernel` raised. The error says that a full
  call of the kernel is required first. Before, `recompute` reused the inputs of the failed call. ([#27])
- The API section of the `numbox.core.vector` documentation page shows the vector module. Before, the page showed the
  variable module. ([#27])

## [0.5.19] - 2026-06-17

### Added

- `make_structref` accepts `ns`, a dict of extra globals that the method bodies in `struct_methods` can reference, such
  as a jitted helper function.

### Fixed

- `make_structref` names each method after its key in `struct_methods`. Before, a method whose function name differed
  from its key failed with a typing error when it was called.

## [0.5.18] - 2026-06-15

### Added

- `compile_kernel` in `numbox.core.variable.compile_kernel`. It compiles a `Variable` graph into one fused `@njit`
  kernel for the requested variables and returns a `CompiledKernel`. ([#24])
- `CompiledKernel`, with `kernel` (positional inputs in, a tuple out), `execute` (a dict in, a dict out) and
  `recompute`, which refreshes only the outputs that depend on changed inputs. Its attributes are `params`, `outputs`,
  `source`, `identifiers` and `partition`. ([#24])
- Formulas that numba cannot compile run in plain Python between the fused `@njit` segments. `CompiledKernel.partition`
  reports the split and the reason for each formula that is not compiled. ([#24])
- The fused kernel is cached on disk under a fingerprint. The fingerprint covers the code, constants, defaults, closure
  values and referenced globals of each formula. The fingerprint also covers the jit flags of the kernel. The `cache`
  keyword of `compile_kernel` controls whether the kernel is cached. ([#24])

### Fixed

- The SQLite registration helpers for aggregate, window and table-valued functions cache a binary for a callback. The
  helpers no longer reuse that binary for a second callback whose closure values or globals differ from those of the
  first callback. Before, the helpers reused the binary whenever the two callbacks had the same code. ([#24])

## [0.5.17] - 2026-06-14

### Added

- `query_to_array(db, sql_p, dtype)` in `numbox.core.bindings._sqlite_query`. It runs an SQL query and returns its rows
  as a 1-D NumPy structured array with one field per column. ([#22])
- `register_tvf(db, name, arg_types, out_dtype, fn)` in `numbox.core.bindings._sqlite_tvf`. It registers an SQLite
  table-valued function, so that `SELECT * FROM name(args)` returns the rows of the structured array that `fn(*args)`
  produces. ([#22])
- `numbox.core.bindings._sqlite_typemap`, with the jitted converters `utf8_to_utf32` and `utf32_to_utf8`.
  `utf32_to_utf8` moved there from `numbox.core.bindings._sqlite_vtable` and remains importable from `_sqlite_vtable`.
  ([#22])
- `numbox.utils.lowlevel.store_unaligned(p, v)`, which works like `store_at` and is safe on a misaligned address.
  ([#22])
- Eight `SQLITE_INDEX_CONSTRAINT_*` constants in `numbox.core.bindings._sqlite_constants`. ([#22])
- The binding `sqlite3_create_module_v2` in `numbox.core.bindings._sqlite_vtable`, which registers a virtual table
  module together with a destructor. ([#22])

### Changed

- `register_table` returns `None`, and numbox keeps the array alive until SQLite drops the table. Before,
  `register_table` returned a handle that the caller had to keep. ([#22])
- A `register_table` table applies `=`, `<`, `<=`, `>` and `>=` conditions on numeric and boolean columns inside the
  virtual table, so a filtered query skips the rows that do not match. ([#22])

### Removed

- The binding `sqlite3_create_module` in `numbox.core.bindings._sqlite_vtable`. Use `sqlite3_create_module_v2`, which
  takes one more argument, a destructor. ([#22])
- The `cacheable` option of variables in `numbox.core.variable`. Building a `Variables` namespace from a spec that has a
  `"cacheable"` key raises `TypeError`. The `cache` attribute of `Values` and of the `Storage` protocol was removed as
  well. ([#26])

## [0.5.16] - 2026-06-03

### Added

- `register_aggregate` and `register_window` in `numbox.core.bindings._sqlite_udf_helpers`. They register an SQL
  aggregate or window function from `init`, `step` and `finalize` callbacks over a structref state, plus `inverse` and
  `value` for a window function. A callback can be a plain Python function or a jitted one. An exception raised in a
  callback is reported to SQLite as an SQL error. ([#19])
- `register_table(db, name, arr, columns=None, *, text_as_blob=False)` in `numbox.core.bindings._sqlite_vtable`. It
  exposes a 2-D or structured NumPy array as a read-only, zero-copy SQLite virtual table. The caller must keep the
  returned handle while the table is in use. ([#21])
- The bindings `sqlite3_malloc` in `numbox.core.bindings._sqlite_exec`, and `sqlite3_create_module` and
  `sqlite3_declare_vtab` in `numbox.core.bindings._sqlite_vtable`. ([#21])
- `numbox.utils.lowlevel.load_unaligned(p, ty)`, which works like `load_at` and is safe on a misaligned address, such as
  a field of a packed structured dtype. ([#21])
- `utf32_to_utf8` in `numbox.core.bindings._sqlite_vtable`, a jitted function that converts a UTF-32 buffer to UTF-8.
  ([#21])
- `numbox.utils.digest.digest(subject, fns)`, which returns a short content hash for use in cache keys. ([#20])

### Changed

- All bindings compile with the JIT options that `NUMBOX_JIT_OPTIONS` sets, not with a fixed `cache=True`. For example,
  `NUMBOX_JIT_OPTIONS='{"cache": false}'` now turns off caching of the bindings as well. The default is still
  `cache=True`.

### Fixed

- `make_graph` no longer recompiles and writes a new cache entry when only the initial values of the graph's nodes
  change. ([#19])

## [0.5.15] - 2026-05-30

### Added

- 34 SQLite bindings for writing scalar, aggregate and window SQL functions in jitted code, in three new modules under
  `numbox.core.bindings`. `_sqlite_value` reads the arguments of a function, `_sqlite_result` sets its return value, and
  `_sqlite_udf` registers the function. ([#17])
- Six flags for function registration in `numbox.core.bindings._sqlite_constants`: `SQLITE_UTF8`,
  `SQLITE_DETERMINISTIC`, `SQLITE_DIRECTONLY`, `SQLITE_INNOCUOUS`, `SQLITE_SUBTYPE` and `SQLITE_RESULT_SUBTYPE`. ([#17])

## [0.5.14] - 2026-05-28

### Added

- 60 SQLite C API bindings that can be called from jitted code, three of which moved from the removed `_sqlite` module.
  They cover connections, prepared statements, parameter binding, column access, `sqlite3_exec`, incremental BLOB I/O
  and callback hooks. They are in seven modules under `numbox.core.bindings`: `_sqlite_conn`, `_sqlite_stmt`,
  `_sqlite_bind`, `_sqlite_column`, `_sqlite_exec`, `_sqlite_blob` and `_sqlite_hooks`. ([#15])
- Five of the SQLite bindings raise `NotImplementedError` when they are called and the loaded SQLite library lacks their
  symbol: `sqlite3_changes64`, `sqlite3_total_changes64`, `sqlite3_column_database_name`, `sqlite3_column_table_name`
  and `sqlite3_column_origin_name`. ([#15])
- `numbox.core.bindings._sqlite_constants`, with 53 `SQLITE_*` constants: result codes, column type codes, open flags,
  BLOB flags, trace flags and the destructor values `SQLITE_STATIC` and `SQLITE_TRANSIENT`. ([#15])
- `numbox.utils.cstrings.c_string`, a context manager that yields a pointer to a NUL-terminated UTF-8 copy of a Python
  string. The pointer can be passed to the bindings. ([#15])
- `numbox.utils.pysqlite_bridge.extract_connection_ptr`, which returns the `sqlite3*` handle behind a Python
  `sqlite3.Connection`. It raises `RuntimeError` when the SQLite version that numbox loaded differs from
  `sqlite3.sqlite_version`. ([#15])

### Changed

- `sqlite3_libversion_number()` returns SQLite's version number as an integer. Before, it returned a pointer to the
  version string. The new `sqlite3_libversion()` returns the pointer to the version string. ([#15])
- `numbox.core.bindings.utils.load_lib(name)` returns the loaded `ctypes.CDLL` handle and caches it per name. Before, it
  returned `None`. ([#15])
- On Windows, `load_lib("sqlite3")` loads only the `sqlite3.dll` that ships with Python, and never a copy found on
  `PATH`. This makes the SQLite bindings usable on Windows. For libraries other than `"c"`, `"m"` and `"sqlite3"`,
  `load_lib` on Windows tries the DLLs that ship with Python before it searches `PATH`. ([#15])

### Removed

- The module `numbox.core.bindings._sqlite`. Its functions `sqlite3_open`, `sqlite3_close` and
  `sqlite3_libversion_number` moved to `numbox.core.bindings._sqlite_conn` and stay importable from
  `numbox.core.bindings`. ([#15])

## [0.5.13] - 2026-05-23

### Added

- `numbox.core.proxy.proxy.proxy_if_available(lib, sig, jit_options=None)`. It works like `@proxy`. When `lib` lacks a C
  symbol with the name of the decorated function, `proxy_if_available` returns a stub that raises `NotImplementedError`
  when it is called. ([#13])

### Fixed

- `@proxy`, and the methods passed to `make_structref`, accept a string as the default value of a parameter. Before, the
  wrapper code that numbox generates for a function with a string default value failed with `NameError`. ([#13])
- `load_at` and `store_at` raise `TypingError` when the pointer is not an `intp` or a `uintp`. Before, both functions
  accepted a narrower integer without an error. A narrower integer cannot hold a full address. ([#13])
- `printf`, `fprintf` and `snprintf` called from plain Python accept the length modifiers `j`, `z`, `t`, `q`, `I32` and
  `I64`, as in `%zd`. Before, they raised `ValueError`. ([#13])
- The formatted output bindings reject the positional form of `%n`, such as `%1$n`. Before, numbox accepted a format
  string with the positional form of `%n`, and a call that used that format string could crash the process. ([#13])

## [0.5.12] - 2026-05-23

### Added

- 25 libc bindings in `numbox.core.bindings._c` that can be called from jitted code: stdio stream functions, string
  functions, memory functions and `getenv`. Pointers are passed as `intp`. ([#12])
- `numbox.core.bindings._stdio`, with `stdout`, `stderr` and `stdin`. Each returns the C `FILE*` of that standard stream
  as an integer. ([#12])
- `numbox.core.bindings._errno`, with `errno_get` and `errno_set`, which read and write the C `errno` of the calling
  thread from jitted code. ([#12])
- `numbox.core.bindings._strerror.strerror_safe(errnum, buf, buflen)`. It writes the message for an error number into a
  buffer that the caller supplies, and is safe to call from several threads. ([#12])
- `printf`, `fprintf`, `snprintf` and `sscanf` in `numbox.core.bindings._fmtio`, which take C-style variadic arguments.
  In jitted code the format string must be a literal. `sscanf` can be called only from jitted code. ([#12])
- `load_at` and `store_at` in `numbox.utils.lowlevel`, which read and write a typed value at a pointer held as an
  integer. ([#12])
- `extract_literal_str` and `intp_ll_type` in `numbox.core.bindings.utils`, two helpers for writing intrinsics. ([#12])
- An `.as_func` attribute on a function decorated with `@proxy`. The attribute holds the first-class function object for
  the first signature given to `@proxy`. That object can be passed where an argument of a function type is expected.
  ([#12])

### Changed

- The bindings in `numbox.core.bindings._c`, `_math` and `_sqlite` are built with `@proxy`, not with `cres`. An
  `@njit(cache=True)` function that calls them can be cached. Before, numba did not cache such a function. ([#12])
- Each binding is a numba dispatcher, not a first-class function object. Code that passed a binding as an argument of a
  function type must pass `binding.as_func`. ([#12])
- `numbox.core.configurations.get_default_jit_options` was renamed `get_jit_options`, and the module attribute
  `default_jit_options` was renamed `jit_options`. Importing the old names fails. The `NUMBOX_JIT_OPTIONS` environment
  variable is unchanged.
- `@proxy`, and the methods passed to `make_structref`, raise `ValueError` for a function with `*args` or `**kwargs`.
  Before, the wrapper code that numbox generates treated `*args` and `**kwargs` as plain positional parameters. ([#12])

### Fixed

- `make_structref` with caching no longer loads stale machine code when the only change to a method is in a constant,
  such as a numeric literal. ([#12])

## [0.5.11] - 2026-05-01

### Added

- `_call_lib_func`, the internal helper that the bindings are built on, can return a C struct larger than 16 bytes as a
  tuple. ([#11])

### Fixed

- `_call_lib_func` passes and returns a 16-byte struct of integers only, such as `{int32, int32, int64}`, with every
  field intact on SysV x86-64 and AAPCS64. Before, fields were lost. No binding shipped with numbox was affected.
  ([#11])

## [0.5.10] - 2026-04-27

### Added

- The `NUMBOX_JIT_OPTIONS` environment variable. It holds a JSON object that replaces the default JIT options,
  `{"cache": true}`, for the functions that numbox itself compiles. Invalid JSON raises `ValueError` on import.
- `numbox.core.configurations.get_default_jit_options()`, which returns the options parsed from `NUMBOX_JIT_OPTIONS`, or
  `{"cache": True}` when the variable is unset.
- A documentation page for the vector container.
- A section of the bindings page on how C structs are passed and returned on each platform. ([#10])

### Changed

- The vector API moved from the module `numbox.core.vector` to `numbox.core.vector.vector`. `make_vector`,
  `vector_push`, `vector_extend` and `Vector` keep their names. Importing them from `numbox.core.vector` fails.
- The vector type class `VectorType` was renamed `VectorTypeClass`.
- `vector_push`, `vector_extend`, `len()`, indexing and item assignment on a vector now compile with numbox's default
  JIT options, so they are cached on disk by default. Before, they were compiled with numba's own default options, which
  do not enable the cache.
- `_call_lib_func`, the internal helper that the bindings are built on, passes and returns C structs by value according
  to the calling convention of the platform. The three struct helpers in `numbox.core.bindings.abi`, added in 0.5.9,
  were removed. The module now holds the platform and struct classification helpers that `_call_lib_func` uses. ([#10])

## [0.5.9] - 2026-04-24

### Added

- `numbox.core.vector`, a growable vector of scalars backed by a NumPy array. Its numba type is the same in every
  process, so jitted code that uses it and is cached on disk keeps working. ([#9])
- `make_vector(elem_type)` in `numbox.core.vector`. It returns a JIT factory `create(capacity)` and the vector type for
  that element type. ([#9])
- `vector_push(v, val)` and `vector_extend(dst, src)` in `numbox.core.vector`, which append to a vector and grow its
  storage as needed. A vector supports `len()`, indexing and item assignment in jitted code. ([#9])
- `borrow_structref(struct_type, p)`, `export_meminfo(s)` and `release_meminfo(p)` in `numbox.utils.meminfo`. The three
  functions pass a structref across a C boundary as an integer MemInfo pointer and keep the reference count of the
  structref balanced. ([#9])
- `numbox.core.bindings.utils.load_lib_path(path)`, which loads a shared library by path or soname and returns its
  `ctypes.CDLL` handle. ([#9])
- `numbox.utils.highlevel.cres_if_available(lib, sig, **kwargs)`, a variant of `cres`. When `lib` lacks a C symbol with
  the name of the decorated function, `cres_if_available` returns a stub that raises `NotImplementedError`. ([#9])
- `numbox.utils.lowlevel.array_data_p(arr)`, which returns the data pointer of a NumPy array as an integer, from Python
  or from jitted code. ([#9])
- `numbox.core.bindings.abi`, internal helpers that pass and return C structs of up to 16 bytes by value from jitted
  code. ([#9])

### Changed

- The parameter of `numbox.utils.meminfo.get_nrt_refcount` was renamed from `meminfo_p` to `s`. A call that passes
  `meminfo_p=` fails.

## [0.5.8] - 2026-04-13

### Added

- `numbox.utils.clock.monotonic_ns()`, a monotonic clock for jitted code. It returns nanoseconds since an unspecified
  start as an `int64`. ([#8])

### Changed

- The supported numba range is now `>=0.60.0,<0.66.0`, which adds the 0.64 patch releases and numba 0.65. It was
  `>=0.60.0,<=0.64.0`.

## [0.5.7] - 2026-03-30

### Added

- `Graph.compile` accepts `debug=True`. The compiled graph then writes one message to standard error before calling the
  formula of a variable. The message names the variable and gives the metadata of the variable. Calling `compile` again
  with the same required names returns the graph compiled earlier and ignores the new `debug` value.

### Changed

- A variable may have a formula that takes no inputs. `Graph.compile` no longer raises `RuntimeError` for such a
  variable.

## [0.5.6] - 2026-03-15

### Added

- 39 libm bindings in `numbox.core.bindings._math`: 30 functions of one argument, such as `exp`, `log`, `sqrt` and
  `erf`, and 9 functions of two arguments, such as `atan2`, `pow` and `hypot`. They take and return `float64` and can be
  called from Python and from jitted code. `_math` already had `cos`, `sin` and `tan`. ([#2], [#4])

### Changed

- `from numbox.core.bindings import *` binds the new names. The bindings `round` and `pow` then shadow the Python
  built-ins of the same names in the importing module. ([#2], [#4])

### Fixed

- `numbox.utils.lowlevel.get_unicode_data_p` returns the correct data pointer for a string that was created inside
  jitted code.

## [0.5.5] - 2026-03-01

### Changed

- `CompiledGraph.execute` and `CompiledGraph.recompute` raise `RuntimeError` when an input of a formula has no value.
  Before, they raised `AssertionError`.
- `VarSpec` marks `name` as a required key for static type checkers.

### Fixed

- `CompiledGraph.execute` accepts `None` as the value of an external variable again. From 0.5.0 to 0.5.4 it raised
  `KeyError` for such a value.
- `Graph.explain` accepts a namespace name that contains a dot. Before, such a name raised `ValueError`.

## [0.5.4] - 2026-02-21

### Changed

- The supported numba range is now `>=0.60.0,<=0.64.0`, which adds numba 0.64.0. It was `>=0.60.0,<0.64.0`.

## [0.5.3] - 2026-02-16

### Added

- `numbox.core.variable.node.make_node(name, source, registry)`, which builds the `Node` tree for a graph variable. It
  reuses one `Node` for a variable that appears more than once.

### Changed

- `Node` in `numbox.core.variable.node` takes a `Variable` and a list of input nodes, as `Node(variable, inputs)`.
  Before, the signature was `Node(name, source, registry)`. Use `make_node` to build a tree from the registry of a
  graph.

### Removed

- The `jit_options` argument of `numbox.utils.highlevel.make_structref_code_txt`, which the function had ignored.
  `make_structref` still accepts `jit_options`.

## [0.5.2] - 2026-02-15

### Added

- `numbox.core.variable.node.Node`, which wraps a graph variable and its inputs, so that
  `numbox.core.work.print_tree.make_image` can draw the dependency tree of a variable as text.

### Changed

- `Graph` raises `ValueError` when the name of an external source is also the name of a namespace. Before, it raised
  `AssertionError`.

## [0.5.1] - 2026-02-05

### Changed

- `CompiledGraph.required_external_variables` and `CompiledGraph.dependents` are plain dicts, not `defaultdict` objects.
  A lookup of an absent key raises `KeyError`. Before, such a lookup returned an empty dict from
  `required_external_variables` and an empty list from `dependents`.

## [0.5.0] - 2026-02-04

### Added

- `Namespace.keys()`, which returns the variable names in a `Variables` or `External` namespace, and
  `Namespace.update(key, var)`, which adds a `Variable` to a namespace after the namespace is built.
- `Values` is iterable and yields the `Variable` instances it holds values for.

### Changed

- Iterating a `Variables` or `External` namespace yields variable names, not `Variable` instances. Use `namespace[name]`
  to get the `Variable`.
- `CompiledGraph.execute` raises `KeyError` when the value supplied for an external variable is `None`. Before, `None`
  was accepted as a value. Version 0.5.5 restores the earlier behavior.
- The `Storage` protocol names its value map `_values`, not `values`, and requires an `__iter__` method.

### Removed

- The `variables` attribute of `Variables` and `External`, and the `values` attribute of `Values`. For a namespace, use
  `keys()`, iteration and `namespace[name]`. For `Values`, use `Values.get(variable)`.

## [0.4.0] - 2026-02-02

### Added

- `Storage` in `numbox.core.variable.variable`, a protocol for the value store that `CompiledGraph.execute` and
  `CompiledGraph.recompute` accept. `Values` satisfies it.
- `Variables` and `External` namespaces support `in` as a test for a variable name. `External` has a `variables`
  attribute, as `Variables` already had.

### Changed

- The `direct` parameter of `Graph.explain` was renamed `right_to_left`. A call that passes `direct=` raises
  `TypeError`.
- `Namespace` is an abstract base class, and `Variables` and `External` are its subclasses. `VarSpec` is a `TypedDict`.
  Before, `Namespace` and `VarSpec` were type aliases.

### Fixed

- `Graph.compile` accepts a namespace name that contains a dot. Before, such a name raised `ValueError`.

## [0.3.4] - 2026-01-31

### Added

- `Graph.dependents_of(qual_names)`, which returns the qualified names of every variable that depends on the given
  names, directly or indirectly, together with the given names.
- `make_qual_name(namespace_name, var_name)` in `numbox.core.variable.variable`, which builds a qualified variable name
  such as `"basket.y"`.
- `Variables` and `External` namespaces are iterable and yield their `Variable` instances.

### Changed

- `Graph.compile` raises `RuntimeError` for a variable that has a formula and no inputs. It no longer raises for a
  variable that has inputs and no formula.

## [0.3.3] - 2026-01-30

### Added

- `Graph.explain(qual_name, direct=True)`, which returns a text account of how a variable is derived, back to its
  external sources.
- A `metadata` field on `Variable`. It holds an optional string, such as the source text of a formula. `Graph.explain`
  prints the string for each derived variable. Among the fields of `Variable`, `metadata` comes before `cacheable`, so
  pass `cacheable` by keyword.

## [0.3.2] - 2026-01-29

### Added

- A documentation page for `numbox.core.variable`, with a worked example that builds, compiles, executes and recomputes
  a graph.

### Changed

- `CompiledGraph.execute` and `CompiledGraph.recompute` raise `AssertionError` when an input of a formula has no value.
  Before, the formula was called with an internal placeholder object as that argument.

## [0.3.1] - 2026-01-27

### Added

- `numbox.core.variable.variable`, a pure Python framework for a graph calculation over named variables. A formula can
  be a plain function or a JIT-compiled dispatcher.
- `Graph`, which is built from namespaces of `Variable` specs and from external sources. `Graph.compile(required)`
  returns a `CompiledGraph` of the variables that are needed, in topological order.
- `CompiledGraph.execute(external_values, values)`, which computes every required variable, and
  `CompiledGraph.recompute(changed, values)`, which recomputes only the variables that depend on changed values.
- The supporting classes `Variable`, `Variables`, `External`, `Value` and `Values`. A `Variable` with `cacheable=True`
  reuses its result when its inputs repeat.

## [0.3.0] - 2026-01-18

### Fixed

- The builder example in the `numbox.core.work` documentation imports `AnyType`, the name that the example uses. Before,
  the example imported `Any`. The package code is the same as in 0.2.17.

## [0.2.17] - 2026-01-18

### Changed

- The supported numba range is now `>=0.60.0,<0.64.0` on every Python version. It was `>=0.60.0,<0.63.0` below Python
  3.14 and `==0.63.1` on Python 3.14.

### Fixed

- `numbox.utils.lowlevel.get_func_tuple`, `Work.derive` and `explain` work on numba 0.60. Before, they read three fields
  of numba's function structure, which has two fields in numba 0.60. `get_func_tuple` and `Work.derive` now return a
  2-tuple on numba 0.60 and a 3-tuple on later versions.

## [0.2.16] - 2025-12-28

### Changed

- llvmlite and numpy are no longer direct dependencies of numbox, so the installed numba determines their versions.
  Before, numbox required llvmlite `>=0.43.0,<0.46.0`, or `==0.46.0` on Python 3.14, and numpy `>=2.1.0,<2.4.0`.

## [0.2.15] - 2025-12-16

No user-facing changes. The release removes a demonstration block that ran only when a module was executed as a script.

## [0.2.14] - 2025-12-11

### Changed

- The minimum Python version is now 3.10. It was 3.9.
- On Python 3.14, numbox requires the stable releases numba `==0.63.1` and llvmlite `==0.46.0`. It required the
  pre-releases `0.63.0b1` and `0.46.0b1`.
- Below Python 3.14, the supported llvmlite range is now `>=0.43.0,<0.46.0`. It was `>=0.44.0,<0.46.0`.

## [0.2.13] - 2025-10-14

### Added

- Windows support for the libc and libm bindings in `numbox.core.bindings`. `load_lib` maps the `"c"` and `"m"`
  libraries to the Microsoft C runtime and finds other libraries with `ctypes.util.find_library`. Before, importing
  `numbox.core.bindings` failed on Windows. The SQLite bindings are not yet tested on Windows.

### Changed

- `load_lib` raises `RuntimeError` on Windows when `ctypes.util.find_library` cannot find a library and the library is
  not `"c"` or `"m"`. It also raises `RuntimeError` on any platform other than Linux, macOS and Windows.

## [0.2.12] - 2025-10-11

### Added

- Support for Python 3.14. On Python 3.14, numbox requires the pre-releases numba `0.63.0b1` and llvmlite `0.46.0b1`.
  The ranges for other Python versions are unchanged.

## [0.2.11] - 2025-09-19

### Changed

- The supported dependency ranges were widened. numba is now `>=0.60.0,<0.63.0` and was `~=0.61.0`. llvmlite is now
  `>=0.44.0,<0.46.0` and was `==0.44.0`. numpy is now `>=2.1.0,<2.4.0` and was `~=2.1.3`.

## [0.2.10] - 2025-08-04

### Added

- `numbox.core.work.builder_utils.infer_sources_dependencies(access_nodes)`. It returns a dict with one key for each
  node that is reachable from the given nodes and that has a source or a dependent. The key is the name of the node, and
  the value is the set of names of the nodes that depend on that node, directly or indirectly.

### Changed

- A `Derived` node may have no sources. `sources` defaults to an empty tuple, and `derive` is then called with no
  arguments. Before, an empty `sources` made `make_graph` generate invalid code.

## [0.2.9] - 2025-07-23

### Added

- A `registry` keyword for `End`, `Derived` and `make_graph`. It is a dict in which node specifications are registered
  by name. Two nodes in one registry cannot have the same name. Graphs with separate registries can reuse a node name.

### Changed

- `infer_end_and_derived_nodes` requires a second argument, `registry`.

## [0.2.8] - 2025-07-23

### Added

- A `precision` argument for `Timer` in `numbox.utils.timer`, which sets the number of decimal places in the logged
  duration. The default stays at 3.

### Changed

- The supported numba range is now `~=0.61.0` and was `==0.61.0`. numpy is now `~=2.1.3` and was `==2.1.3`. llvmlite
  stays at `==0.44.0`. The dependencies are declared in `pyproject.toml`, not in `requirements.txt`.

## [0.2.7] - 2025-07-20

### Added

- `numbox.utils.highlevel.make_structref(struct_name, struct_fields, struct_type_class, *, struct_methods=None,
  jit_options=None)`, which defines a numba StructRef type and its Python class in one call. The caller defines one
  `StructRef` type class for each structref and passes it as `struct_type_class`. `make_structref` generates the
  constructor, the field properties, `__repr__` and optional jitted methods. A jitted function that uses the generated
  type can be cached.
- `make_structref` accepts `struct_fields` as a list of names or as a dict of names to numba types. If `struct_fields`
  is a list of names, the field types are inferred when an instance of the struct is constructed. `make_structref` takes
  methods as `struct_methods`, a dict of names to plain Python functions.
- `numbox.utils.standard.make_params_strings`. It moved there from `numbox.core.proxy.proxy`, and the old import path
  keeps working.

## [0.2.6] - 2025-07-15

### Added

- `infer_end_and_derived_nodes(access_nodes)` in `numbox.core.work.builder`. It returns the `End` and `Derived`
  specifications that the given access nodes depend on.

### Changed

- `make_graph` takes only the access nodes, as `make_graph(*access_nodes, jit_options=None)`. It follows the sources of
  each access node to find the `End` and `Derived` nodes that the access nodes depend on. The call `make_graph(inputs,
  derived, access_nodes)` no longer works.
- `make_graph` returns an `Access` named tuple with one field per access node, such as `make_graph(w3_).w3`. Before,
  `make_graph` returned the bare node for one access node and a plain tuple for several.
- The output of `explain` starts with a line that lists every end node that the explained node requires.

## [0.2.5] - 2025-07-10

### Added

- `numbox.core.work.explain.explain(work)`. It returns a text trace of how a node built with `make_graph` is derived,
  down to its end nodes, with the source of each `derive` function.
- `Work.derive` can be read from Python. It returns the low-level pointers of the derive function as a tuple of
  integers.

### Changed

- `End` and `Derived` raise `ValueError` when a node name is already in use. The names are tracked for the whole
  process, not per graph, so a name cannot be reused for a second graph in the same process.
- `End` and `Derived` require `name` as a keyword argument. A positional call such as `End("w1", 137)` raises
  `AssertionError`.

## [0.2.4] - 2025-07-06

### Changed

- `End` and `Derived` in `numbox.core.work.builder` accept NumPy arrays and StructRef instances as `init_value`. Before,
  `make_graph` failed for anything other than simple scalars.
- `make_graph` recompiles its cached graph maker when an `init_value` or the source of a `derive` function changes.

## [0.2.3] - 2025-07-05

### Fixed

- `Work.combine` fills every requested node. Before, when the graph reached a requested node through more than one path,
  the search in `Work.combine` counted that node on each visit. The search could then stop before it reached the
  remaining requested nodes. `Work.combine` now returns `None`, not an internal count.

## [0.2.2] - 2025-07-04

### Added

- `numbox.core.work.builder`, which builds a JIT-compiled graph of `Work` nodes from pure Python specifications. Declare
  leaves with `End` and computed nodes with `Derived`, then call `make_graph(inputs, derived, access_nodes)` to get the
  requested nodes.
- `Node.all_end_nodes()` and `Work.all_end_nodes()`, which return the names of the end nodes, the nodes with no inputs,
  in the subtree. They can be called from Python and from jitted code.

## [0.2.1] - 2025-07-01

### Added

- A documentation site built with Sphinx, with pages for `numbox.core.any`, `numbox.core.bindings`, `numbox.core.proxy`,
  `numbox.core.work` and `numbox.utils`. A new `docs` extra installs Sphinx.
- `Work` and `make_work(name, data, sources=(), derive=None)` in `numbox.core.work.work`. A `Work` is a named node of a
  calculation graph. It can be used from Python and from jitted code. `Work.calculate()` computes the sources of the
  node and then applies `derive` to their data. A node is calculated only once.
- `Work.load(data)` and `Work.combine(data)`, where `data` is a typed dict keyed by node name. `load` sets the data of
  the named nodes and marks those nodes and their dependents for recalculation. `combine` copies the current data of the
  named nodes into the dict.
- `Node` and `make_node(name, inputs=())` in `numbox.core.work.node`. A `Node` is a lightweight jitted graph node that
  holds a name and a list of input nodes. `Node` and `Work` both provide `get_input`, `get_inputs_names`,
  `all_inputs_names` and `depends_on`.
- `numbox.core.work.lowlevel_work_utils.ll_make_work`, a low-level `Work` constructor for building large graphs inside
  jitted code with less compile time and memory.
- `make_work_helper` and `make_init_data` in `numbox.core.work.work_utils`. `make_work_helper` builds a `Work` from
  Python and compiles a plain Python `derive` function with a signature inferred from the data.
- `numbox.core.work.print_tree.make_image(node)`, which draws the dependency tree of a `Node` or `Work` as text.
- Helpers that move graph data to and from NumPy structured arrays: `load_array_row_into_dict` in
  `numbox.core.work.loader_utils`, and `make_sheaf_dict`, `make_requested_dtype` and `load_dict_into_array` in
  `numbox.core.work.combine_utils`.
- `numbox.core.bindings`, with nine wrappers of C library functions that can be called from Python and from jitted code:
  `rand`, `srand` and `strlen` from libc, `cos`, `sin` and `tan` from libm, and `sqlite3_open`, `sqlite3_close` and
  `sqlite3_libversion_number` from SQLite. Pointers are passed as integers.
- `numbox.core.bindings.utils.load_lib(name)`, which loads a shared library so that its functions can be bound in the
  same way as the libc, libm and SQLite functions are bound.
- `numbox.utils.highlevel.cres(sig, **kwargs)`. It compiles a function for one signature and returns a first-class
  function object, not a dispatcher. A jitted function can take this object as an argument and still be cached.
- `hash_type(ty)` and `determine_field_index(struct_ty, field_name)` in `numbox.utils.highlevel`.
- 11 low-level helpers in `numbox.utils.lowlevel`. They cast tuples of structrefs to one type, read C strings and
  function pointers, test a structref for null, and help with writing intrinsics.
- `numbox.utils.void_type.VoidType`, an empty structref type used as a type-erased reference.
- The `numbox.utils.timer.timer` decorator, which logs the duration of each call and records it in `Timer.times`.
- The `Any.type_info` property, which returns the name of the stored type.

### Changed

- `Any` moved to the package `numbox.core.any`. Import `make_any`, `Any`, `AnyType` and `AnyTypeClass` from
  `numbox.core.any.any_type`, not from `numbox.core.any_type`.
- `ContentTypeClass` moved to `numbox.core.any.content_wrap`, and `ErasedTypeClass` and `ErasedType` moved to
  `numbox.core.any.erased_type`.
- `structref_meminfo` and `get_nrt_refcount` moved from `numbox.core.meminfo` to `numbox.utils.meminfo`.
- `numbox.core.proxy` is a package. Import the decorator with `from numbox.core.proxy.proxy import proxy`.
- `numbox.utils.lowlevel.deref` was renamed `deref_payload`.
- `default_jit_options` moved from `numbox.core.any_type` to `numbox.core.configurations`.
- `Any.get_as(ty)` raises `NumbaError` when `ty` differs from the stored type. Before, it reinterpreted the stored value
  as `ty` without a check.
- `make_any` no longer converts a function compiled with `njit` into a first-class function type. To store a jitted
  function in an `Any`, compile it with `cres`.

### Removed

- `make_any_maker` and `make_any_prototype`, with no replacement. `make_any` accepts any value.

## [0.1.4] - 2025-03-03

### Added

- `make_any_maker(arg_type=None, jit_options=None)` in `numbox.core.any_type`. It builds a `make_any` that is compiled
  for one numba type. With this `make_any`, numba can cache the code that stores a jitted function in an `Any`.
  `make_any_maker` takes a numba type. The `make_any_maker` of 0.1.0 took a `Signature`.
- `make_any_prototype(x)` in `numbox.core.any_type`, the uncompiled body of `make_any`.

### Changed

- `make_any`, `Any.get_as` and `Any.reset` are compiled with `cache=True`, so their machine code is cached on disk and
  reused across runs.

## [0.1.3] - 2025-02-23

### Added

- The `numbox.core.proxy.proxy(sig, jit_options=None)` decorator. It compiles the function for the given signature and
  binds the function's name to a small jitted proxy. A caller then links only a declaration of the compiled function,
  not the whole code of that function.
- `proxy` passes `jit_options`, such as `{'cache': True}`, to the compilation of the decorated function.
- `proxy` accepts a list of signatures. The first signature is the main signature, which gives the concrete type of
  every parameter. The other signatures can use numba's `Omitted` type for parameters that have default values, so
  callers can leave those parameters out.

## [0.1.2] - 2025-02-15

### Removed

- The module-level functions `get_as` and `reset` in `numbox.core.any_type`. Call the methods `Any.get_as(ty)` and
  `Any.reset(val)`.

## [0.1.1] - 2025-02-11

### Removed

- `make_any_maker`, which built a `make_any` compiled for a given signature, with no replacement.
- `any_constructor`, the constructor that `make_any` called. Call `make_any`, from Python or from jitted code.

## [0.1.0] - 2025-02-11

### Added

- First release. It pins `numba==0.61.0`, `llvmlite==0.44.0` and `numpy==2.1.3`.
- `numbox.core.any_type`, a type-erased container. `make_any(x)` wraps a value of any type in an `Any` object that can
  be used in Python and in jitted code. The value can be, for example, a structref, an array, a string or a jitted
  function.
- `Any.get_as(ty)`, which returns the held value as type `ty`, and `Any.reset(val)`, which replaces the held value and
  releases the old one. Both are also available as the functions `get_as` and `reset` in the same module.
- `make_any_maker(sig=None)`, which builds a `make_any` that is compiled for a given numba `Signature`.
- `numbox.core.meminfo`. `structref_meminfo(s)` returns the addresses of the MemInfo and of the data payload of an
  object that numba's runtime manages. `get_nrt_refcount(s)` returns the reference count of that object.
- `numbox.utils.lowlevel`. `cast(source, dest_ty)` reinterprets a value as another type, and `deref(p, ty)` reads the
  payload of a structref as type `ty`.
- `numbox.utils.highlevel.prune_type`, which turns the dispatcher type of a jitted function with one signature into a
  first-class function type.

[unreleased]: https://github.com/Goykhman/numbox/compare/0.7.9...HEAD
[0.7.9]: https://github.com/Goykhman/numbox/compare/0.7.8...0.7.9
[0.7.8]: https://github.com/Goykhman/numbox/compare/0.7.7...0.7.8
[0.7.7]: https://github.com/Goykhman/numbox/compare/0.7.6...0.7.7
[0.7.6]: https://github.com/Goykhman/numbox/compare/0.7.5...0.7.6
[0.7.5]: https://github.com/Goykhman/numbox/compare/0.7.4...0.7.5
[0.7.4]: https://github.com/Goykhman/numbox/compare/0.7.3...0.7.4
[0.7.3]: https://github.com/Goykhman/numbox/compare/0.7.2...0.7.3
[0.7.2]: https://github.com/Goykhman/numbox/compare/0.7.1...0.7.2
[0.7.1]: https://github.com/Goykhman/numbox/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/Goykhman/numbox/compare/0.6.3...0.7.0
[0.6.3]: https://github.com/Goykhman/numbox/compare/0.6.2...0.6.3
[0.6.2]: https://github.com/Goykhman/numbox/compare/0.6.1...0.6.2
[0.6.1]: https://github.com/Goykhman/numbox/compare/0.6.0...0.6.1
[0.6.0]: https://github.com/Goykhman/numbox/compare/0.5.23...0.6.0
[0.5.23]: https://github.com/Goykhman/numbox/compare/0.5.22...0.5.23
[0.5.22]: https://github.com/Goykhman/numbox/compare/0.5.21...0.5.22
[0.5.21]: https://github.com/Goykhman/numbox/compare/0.5.20...0.5.21
[0.5.20]: https://github.com/Goykhman/numbox/compare/0.5.19...0.5.20
[0.5.19]: https://github.com/Goykhman/numbox/compare/0.5.18...0.5.19
[0.5.18]: https://github.com/Goykhman/numbox/compare/0.5.17...0.5.18
[0.5.17]: https://github.com/Goykhman/numbox/compare/0.5.16...0.5.17
[0.5.16]: https://github.com/Goykhman/numbox/compare/0.5.15...0.5.16
[0.5.15]: https://github.com/Goykhman/numbox/compare/0.5.14...0.5.15
[0.5.14]: https://github.com/Goykhman/numbox/compare/0.5.13...0.5.14
[0.5.13]: https://github.com/Goykhman/numbox/compare/0.5.12...0.5.13
[0.5.12]: https://github.com/Goykhman/numbox/compare/0.5.11...0.5.12
[0.5.11]: https://github.com/Goykhman/numbox/compare/0.5.10...0.5.11
[0.5.10]: https://github.com/Goykhman/numbox/compare/0.5.9...0.5.10
[0.5.9]: https://github.com/Goykhman/numbox/compare/0.5.8...0.5.9
[0.5.8]: https://github.com/Goykhman/numbox/compare/0.5.7...0.5.8
[0.5.7]: https://github.com/Goykhman/numbox/compare/0.5.6...0.5.7
[0.5.6]: https://github.com/Goykhman/numbox/compare/0.5.5...0.5.6
[0.5.5]: https://github.com/Goykhman/numbox/compare/0.5.4...0.5.5
[0.5.4]: https://github.com/Goykhman/numbox/compare/0.5.3...0.5.4
[0.5.3]: https://github.com/Goykhman/numbox/compare/0.5.2...0.5.3
[0.5.2]: https://github.com/Goykhman/numbox/compare/0.5.1...0.5.2
[0.5.1]: https://github.com/Goykhman/numbox/compare/0.5.0...0.5.1
[0.5.0]: https://github.com/Goykhman/numbox/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/Goykhman/numbox/compare/0.3.4...0.4.0
[0.3.4]: https://github.com/Goykhman/numbox/compare/0.3.3...0.3.4
[0.3.3]: https://github.com/Goykhman/numbox/compare/0.3.2...0.3.3
[0.3.2]: https://github.com/Goykhman/numbox/compare/0.3.1...0.3.2
[0.3.1]: https://github.com/Goykhman/numbox/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/Goykhman/numbox/compare/0.2.17...0.3.0
[0.2.17]: https://github.com/Goykhman/numbox/compare/0.2.16...0.2.17
[0.2.16]: https://github.com/Goykhman/numbox/compare/0.2.15...0.2.16
[0.2.15]: https://github.com/Goykhman/numbox/compare/0.2.14...0.2.15
[0.2.14]: https://github.com/Goykhman/numbox/compare/0.2.13...0.2.14
[0.2.13]: https://github.com/Goykhman/numbox/compare/0.2.12...0.2.13
[0.2.12]: https://github.com/Goykhman/numbox/compare/0.2.11...0.2.12
[0.2.11]: https://github.com/Goykhman/numbox/compare/0.2.10...0.2.11
[0.2.10]: https://github.com/Goykhman/numbox/compare/0.2.9...0.2.10
[0.2.9]: https://github.com/Goykhman/numbox/compare/0.2.8...0.2.9
[0.2.8]: https://github.com/Goykhman/numbox/compare/0.2.7...0.2.8
[0.2.7]: https://github.com/Goykhman/numbox/compare/0.2.6...0.2.7
[0.2.6]: https://github.com/Goykhman/numbox/compare/0.2.5...0.2.6
[0.2.5]: https://github.com/Goykhman/numbox/compare/0.2.4...0.2.5
[0.2.4]: https://github.com/Goykhman/numbox/compare/0.2.3...0.2.4
[0.2.3]: https://github.com/Goykhman/numbox/compare/0.2.2...0.2.3
[0.2.2]: https://github.com/Goykhman/numbox/compare/0.2.1...0.2.2
[0.2.1]: https://github.com/Goykhman/numbox/compare/0.1.4...0.2.1
[0.1.4]: https://github.com/Goykhman/numbox/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/Goykhman/numbox/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/Goykhman/numbox/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/Goykhman/numbox/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/Goykhman/numbox/releases/tag/0.1.0

[#2]: https://github.com/Goykhman/numbox/pull/2
[#4]: https://github.com/Goykhman/numbox/pull/4
[#8]: https://github.com/Goykhman/numbox/pull/8
[#9]: https://github.com/Goykhman/numbox/pull/9
[#10]: https://github.com/Goykhman/numbox/pull/10
[#11]: https://github.com/Goykhman/numbox/pull/11
[#12]: https://github.com/Goykhman/numbox/pull/12
[#13]: https://github.com/Goykhman/numbox/pull/13
[#15]: https://github.com/Goykhman/numbox/pull/15
[#17]: https://github.com/Goykhman/numbox/pull/17
[#19]: https://github.com/Goykhman/numbox/pull/19
[#20]: https://github.com/Goykhman/numbox/pull/20
[#21]: https://github.com/Goykhman/numbox/pull/21
[#22]: https://github.com/Goykhman/numbox/pull/22
[#24]: https://github.com/Goykhman/numbox/pull/24
[#26]: https://github.com/Goykhman/numbox/pull/26
[#27]: https://github.com/Goykhman/numbox/pull/27
[#28]: https://github.com/Goykhman/numbox/pull/28
[#31]: https://github.com/Goykhman/numbox/pull/31
[#32]: https://github.com/Goykhman/numbox/pull/32
[#33]: https://github.com/Goykhman/numbox/pull/33
[#34]: https://github.com/Goykhman/numbox/pull/34
[#35]: https://github.com/Goykhman/numbox/pull/35
[#36]: https://github.com/Goykhman/numbox/pull/36
[#38]: https://github.com/Goykhman/numbox/pull/38
[#39]: https://github.com/Goykhman/numbox/pull/39
[#41]: https://github.com/Goykhman/numbox/pull/41
[#42]: https://github.com/Goykhman/numbox/pull/42
[#43]: https://github.com/Goykhman/numbox/pull/43
[#44]: https://github.com/Goykhman/numbox/pull/44
[#45]: https://github.com/Goykhman/numbox/pull/45
[#46]: https://github.com/Goykhman/numbox/pull/46
[#48]: https://github.com/Goykhman/numbox/pull/48
[#49]: https://github.com/Goykhman/numbox/pull/49
[#50]: https://github.com/Goykhman/numbox/pull/50
[#51]: https://github.com/Goykhman/numbox/pull/51
[#53]: https://github.com/Goykhman/numbox/pull/53
