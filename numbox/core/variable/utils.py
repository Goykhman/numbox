"""Helpers that compile_kernel uses to name the variables in a kernel's source, and to check and wrap formulas.

:mod:`numbox.core.variable.compile_kernel` uses them to assign each variable a unique Python identifier and to
decorate a plain Python formula with ``@njit``. The other helpers check that a formula can take its inputs, and compare
a declared return type with the type that numba infers. The helpers are defined in this module, not in compile_kernel,
so that another module can use them without importing compile_kernel.
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

# The names that `_compile` in compile_kernel adds, besides the formulas, to the namespace in which a kernel's source
# runs. An identifier must avoid them, so this set must change with that namespace. `_compile` adds `__name__` as
# well, which needs no entry, because an identifier never starts with an underscore.
_RESERVED = frozenset({"njit", "_kernel_jit_options"})


def _sanitize(qual_name: str) -> str:
    # The function replaces each run of characters other than letters and digits with one underscore, removes leading
    # and trailing underscores, and lowercases the name. It adds the prefix v_ to a name that is then empty or starts
    # with a digit, because a Python identifier cannot be empty or start with a digit.
    s = re.sub(r"[^0-9A-Za-z_]", "_", qual_name)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    if not s or s[0].isdigit():
        s = "v_" + s
    return s


def _assign_identifiers(variables: list[Variable]) -> dict[Variable, str]:
    """Return a unique, valid and readable Python identifier for each variable, made from its qualified name.

    Each identifier `t` reserves two names: `t` itself and ``f_<t>``, the global through which the kernel's source
    calls the variable's formula. If either name is already reserved, or `t` is a Python keyword, the function appends
    a suffix to `t`: an underscore and the shortest prefix of the sha256 digest of the qualified name with which
    neither name is reserved. The names in `_RESERVED` are reserved from the start. The result therefore depends only
    on the variables and their order. Raises ``RuntimeError`` if every prefix of the digest leaves a name reserved.
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
    """Return `formula` in a form that compiled code can call, decorating a plain Python formula with ``@njit``.

    The function returns a numba dispatcher, a cres function, a @vectorize function or a @cfunc unchanged. It
    decorates any other callable with ``@njit`` under `flags`, the kernel's jit options, so that the formula compiles
    with the same options as the fused kernel. Raises ``TypeError`` if `formula` is not callable.
    """
    if isinstance(formula, (Dispatcher, CompileResultWAP, DUFunc, CFunc)):
        return formula
    if not callable(formula):
        raise TypeError(f"formula {formula!r} is not callable")
    return njit(**(flags or {}))(formula)


def _strip_cache(flags: dict | None) -> dict:
    """Return `flags` without ``cache``, for the return-type probe of `_validate_declared_return`.

    The probe compiles a function only to read its return type, so it must not read or write numba's on-disk cache.
    The freshness of the fused kernel does not depend on this function, because the kernel's name holds the
    fingerprint of each formula.
    """
    return {k: v for k, v in (flags or {}).items() if k != "cache"}


def _validate_declared_return(formula, input_types: tuple, declared, flags: dict | None = None) -> None:
    """Raise ``ValueError`` if `formula` does not return the type `declared` for the argument types `input_types`.

    numba converts a result to the return type of a declared signature whenever such a conversion exists: between
    integers and floats, from a wider type to a narrower one, and between signed and unsigned types. Compiling the
    formula for the declared signature would therefore not detect a wrong declaration. This function reads the type
    that the formula returns when no return type is declared, and compares it with `declared` for equality. The
    comparison ignores only the layout of arrays: an elementwise formula returns a C-contiguous array, a declared
    ``float64[:]`` has the layout 'A', and numba assigns one to the other. numba's ``TypingError`` propagates if the
    formula does not compile for `input_types`.
    """
    opts = _strip_cache(flags)
    if isinstance(formula, CFunc):
        # A @cfunc stores its signature, so its return type can be read without a compilation.
        natural = formula._sig.return_type
    elif isinstance(formula, CompileResultWAP):
        # A cres function has no `_sig`; its signature() method returns the signature of its compile result.
        natural = formula.signature().return_type
    elif isinstance(formula, DUFunc):
        # A @vectorize function has no single return type. The return type of a one-line @njit function that applies
        # it to the declared input types is the type that numba infers.
        names = ", ".join(f"a{i}" for i in range(len(input_types)))
        ns = {"_f": formula}
        exec(f"def _shim({names}):\n    return _f({names})\n", ns)  # nosec B102
        probe = njit(**opts)(ns["_shim"])
        probe.compile(input_types)
        natural = probe.nopython_signatures[-1].return_type
    elif isinstance(formula, Dispatcher):
        # A dispatcher compiles for the input types under its own options. This compilation adds an overload to the
        # dispatcher, and a dispatcher compiled with cache=True reads and writes its own cache here.
        formula.compile(input_types)
        natural = formula.overloads[input_types].signature.return_type
    else:
        # numba compiles a plain Python formula as a new @njit function, without a declared return type.
        probe = njit(**opts)(formula)
        probe.compile(input_types)
        natural = probe.nopython_signatures[-1].return_type
    if isinstance(natural, Array) and isinstance(declared, Array):
        # The comparison converts both types to the layout 'A', so that only a difference in dtype, dimensions,
        # mutability or alignment causes a mismatch.
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
