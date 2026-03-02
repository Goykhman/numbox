# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

numbox — a toolbox of low-level utilities for working with numba. Provides type erasure (`Any`), native library bindings (`Bindings`), graph nodes (`Node`), function proxies (`Proxy`), graph calculation (`Variable`), and units of work (`Work`).

## Build & Dev

- Venv: `python3.10 -m venv venv && venv/bin/pip install -e . flake8 pytest`
- Install: `pip install -e .`
- Test: `pytest`
- Lint: `flake8` (max-line-length=127, max-complexity=10)
- Docs: `cd docs && make html` (Sphinx)
- Python: >=3.10 (CI tests 3.10–3.14)
- Key dependency: `numba>=0.60.0,<=0.64.0` (use `numba==0.60.0` locally)

## Architecture

### Bindings System (core/bindings/)

The bindings subsystem wraps C library functions for use inside numba `@njit` code. Four layers:

1. **`utils.py`** — loads shared libraries via `ctypes.CDLL` with `RTLD_GLOBAL` so symbols are visible to LLVM
2. **`signatures.py`** — flat dict mapping C function names to numba type signatures (e.g., `"cos": float64(float64)`). Organized by library: `signatures_c`, `signatures_m`, `signatures_sqlite`
3. **`call.py`** — `@numba.extending.intrinsic` that generates LLVM IR to call native functions directly via `llvmlite`
4. **`_math.py`, `_c.py`, `_sqlite.py`** — thin Python wrappers using `@cres(signatures.get("func"), cache=True)`

### Adding a New Binding

1. Add signature to `signatures.py` in the appropriate sub-dict
2. Add wrapper to the corresponding `_*.py` file following this pattern:
```python
@cres(signatures.get("func_name"), cache=True)
def func_name(x):
    return _call_lib_func("func_name", (x,))
```
3. Function names must match the C library names exactly
4. Args passed as tuple literal to `_call_lib_func`

### Core Modules

- **`core/any/`** — type erasure: wraps any value into uniform type
- **`core/bindings/`** — JIT-compatible wrappers for native C libraries
- **`core/proxy/`** — function proxies with specified signatures for JIT caching
- **`core/variable/`** — graph calculation framework with JIT dispatcher
- **`core/work/`** — JIT-compatible units of calculation with dependencies

### Utilities (utils/)

- `highlevel.py` — `cres` decorator (compiles to `CompileResultWAP` with explicit signature)
- `lowlevel.py` — low-level numba helpers
- `meminfo.py` — memory info utilities
- `standard.py` — standard utilities
- `timer.py` — timing utilities
- `void_type.py` — void type support

## Key Paths

- `numbox/core/bindings/signatures.py` — all native function type signatures
- `numbox/core/bindings/_math.py` — libm wrappers (34 float64→float64 functions)
- `numbox/core/bindings/_c.py` — libc wrappers
- `numbox/core/bindings/_sqlite.py` — libsqlite3 wrappers
- `test/core/` — tests for all core modules

## Preferences

- Never include "Co-Authored-By" in git commit messages
- Avoid shell variable substitution in bash — inline actual values directly into commands
- Prefer simpler approaches
- Always git pull before making edits
- Commit messages must not mention AI, Claude, Anthropic, or any AI tooling — only attribute to the user
- Keep all memories in both MEMORY.md and the project CLAUDE.md (CLAUDE.md is in git and survives OS reinstalls)
- Environment details go in MEMORY.md only (may change between OS installs)

## CI

- **numbox_ci.yml** — lint + test + build on push/PR (matrix: Python 3.10–3.14, ubuntu + ubuntu-arm + windows)
- **docs.yml** — Sphinx docs → GitHub Pages on push to main
- **release.yml** — build + publish to PyPI on release
