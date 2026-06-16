"""Formula and identifier helpers for the fused-kernel compiler.

Shared utilities used by `numbox.core.variable.compile_kernel`: identifier
assignment for generated kernel source, formula njit-wrapping, and formula
arity validation. Kept here so they have a stable home as the kernel machinery
grows and so a second consumer can reuse them without importing compile_kernel.
"""
import hashlib
import inspect
import keyword
import re

from typing import Callable

from numba import njit
from numba.core.ccallback import CFunc
from numba.core.dispatcher import Dispatcher
from numba.core.types import Array
from numba.core.types.function_type import CompileResultWAP
from numba.np.ufunc.dufunc import DUFunc

from numbox.core.variable.variable import Variable

# Names injected into the kernel exec namespace; identifiers must avoid them.
# Kept in lockstep with the exec namespace assembled in compile_kernel._compile.
_RESERVED = frozenset({"njit", "_kernel_jit_options"})


def _sanitize(qual_name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_]", "_", qual_name)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    if not s or s[0].isdigit():
        s = "v_" + s
    return s


def _assign_identifiers(variables: list[Variable]) -> dict[Variable, str]:
    """Map each Variable to a unique, valid, readable Python identifier.

    Readable (from the qual_name) with a minimal deterministic sha256 suffix
    only where names would otherwise collide. Reserves both the node temp `t`
    and its formula global `f_<t>` so those namespaces never clash, and avoids
    the injected reserved names.
    """
    used = set(_RESERVED)
    idents = {}
    for var in variables:
        base = _sanitize(var.qual_name())
        digest = hashlib.sha256(var.qual_name().encode("utf-8")).hexdigest()
        cand = base
        i = 0
        while cand in used or ("f_" + cand) in used or keyword.iskeyword(cand):
            i += 1
            if i > len(digest):
                raise RuntimeError(
                    f"Cannot assign a unique identifier for {var.qual_name()!r}; "
                    f"all sha256 prefixes exhausted"
                )
            cand = f"{base}_{digest[:i]}"
        used.add(cand)
        used.add("f_" + cand)
        idents[var] = cand
    return idents


def _wrap_formula(formula: Callable, flags: dict | None = None) -> Dispatcher | CompileResultWAP | DUFunc | CFunc:
    """Return an njit-callable for `formula`; plain-Python callables are njit-wrapped
    with the kernel's effective jit `flags` so their semantics match the fused kernel."""
    if isinstance(formula, (Dispatcher, CompileResultWAP, DUFunc, CFunc)):
        return formula
    if not callable(formula):
        raise TypeError(f"formula {formula!r} is not callable")
    return njit(**(flags or {}))(formula)


def _strip_cache(flags: dict | None) -> dict:
    """Inner formulas must never be cached (a cached inner stale-hits on a
    numeric-literal edit and inlines a stale body into the fused kernel)."""
    return {k: v for k, v in (flags or {}).items() if k != "cache"}


def _validate_declared_return(formula, input_types: tuple, declared, flags: dict | None = None) -> None:
    """Raise if the formula's NATURAL return type at `input_types` differs from
    `declared`.

    njit(sig) silently coerces every convertible scalar mismatch (int<->float,
    narrowing, sign), so the declared return cannot be trusted to fail fast. The
    natural type is read by:
    - CFunc: `_sig.return_type` (data already present).
    - cres (CompileResultWAP): `signature().return_type` (no `_sig` attribute).
    - DUFunc (@vectorize): no single return type; probe via an unconstrained
      @njit shim applying the DUFunc at the declared input types.
    - plain Python: an unconstrained @njit probe over the declared input types.

    The comparison is equality (no coercion allowed) so the guard truly fires,
    except that array layout is ignored: an elementwise formula naturally yields a
    C-contiguous array, while a declared `float64[:]` is layout 'A', and numba freely
    assigns one to the other -- a layout difference is not a declared-type violation.
    """
    opts = _strip_cache(flags)
    if isinstance(formula, CFunc):
        natural = formula._sig.return_type
    elif isinstance(formula, CompileResultWAP):
        natural = formula.signature().return_type
    elif isinstance(formula, DUFunc):
        names = ", ".join(f"a{i}" for i in range(len(input_types)))
        ns = {"_f": formula}
        exec(f"def _shim({names}):\n    return _f({names})\n", ns)  # nosec B102
        probe = njit(**opts)(ns["_shim"])
        probe.compile(input_types)
        natural = probe.nopython_signatures[-1].return_type
    elif isinstance(formula, Dispatcher):
        formula.compile(input_types)
        natural = formula.overloads[input_types].signature.return_type
    else:
        probe = njit(**opts)(formula)
        probe.compile(input_types)
        natural = probe.nopython_signatures[-1].return_type
    if isinstance(natural, Array) and isinstance(declared, Array):
        # Normalize both to layout 'A' so only dtype/ndim/readonly mismatches count
        # (numba assigns C<->A freely).
        mismatch = natural.copy(layout="A") != declared.copy(layout="A")
    else:
        mismatch = natural != declared
    if mismatch:
        raise ValueError(
            f"declared type {declared} but formula yields {natural} at "
            f"input types {input_types}"
        )


def _check_formula_arity(formula, n_inputs: int, qual_name: str) -> None:
    target = getattr(formula, "py_func", None) or getattr(formula, "__wrapped__", None) or formula
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return
    try:
        sig.bind(*range(n_inputs))
    except TypeError as e:
        raise ValueError(
            f"{qual_name!r}: formula signature {sig} cannot accept its "
            f"{n_inputs} declared input(s) passed positionally ({e})"
        ) from None
