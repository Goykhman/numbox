"""Tests for the content-fingerprint walker (numbox.utils.fingerprint).

Two concerns run through these: canonicalizing a *value* stably (so that two artifacts
differing only in captured data get different digests), and telling a genuine global read
(``LOAD_GLOBAL``) apart from an attribute access (``LOAD_ATTR``) whose name merely collides
with a module global. ``co_names`` pools the latter two, so folding globals off it either
over-invalidates the fingerprint (a colliding global that canonicalizes) or makes the whole
function un-fingerprintable (a colliding global that does not) -- which, on the builder's
derive path, forces an address-bearing fallback and unbounded cache growth.
``_loaded_global_names`` reads the opcodes instead.
"""
import textwrap

import numpy as np
import pytest

from numbox.utils.fingerprint import (
    _Unfingerprintable, _canon_value, _fingerprint_function, _loaded_global_names,
    _opaque_values_match, _referenced_global_names,
)
from test.auxiliary_utils import open_libm


def _make(src, **globs):
    """Compile ``src`` (defining ``f``) with ``globs`` as its module globals."""
    ns = dict(globs)
    exec(textwrap.dedent(src), ns)  # nosec B102 - test helper compiling a fixed snippet
    return ns["f"]


def test_canon_value_structured_dtype_preserves_layout():
    # dtype.str is |V<n> for structured dtypes (field layout erased), so same-byte
    # different-layout arrays would collide. The full descriptor discriminates them.
    raw = (1).to_bytes(4, "little") + (2).to_bytes(4, "little")
    two_i4 = np.frombuffer(raw, dtype=np.dtype([("a", "<i4"), ("b", "<i4")])).copy()
    two_f4 = np.frombuffer(raw, dtype=np.dtype([("a", "<f4"), ("b", "<f4")])).copy()
    one_i8 = np.frombuffer(raw, dtype=np.dtype([("x", "<i8")])).copy()

    assert two_i4.tobytes() == two_f4.tobytes() == one_i8.tobytes()
    canons = {_canon_value(a, set()) for a in (two_i4, two_f4, one_i8)}
    assert len(canons) == 3, "structured-dtype arrays with identical bytes collided"


def test_canon_value_plain_dtype_unchanged():
    canon = _canon_value(np.arange(3, dtype=np.float64), set())
    assert canon.startswith("ndarray(<f8;")


def test_proxy_binding_callback_fingerprints_cleanly_and_distinctly():
    # A callback that calls a @proxy'd binding used to abort the walker (the
    # wrapper recurses into its @intrinsic, which has no canonical form), so the
    # cache digest degraded to a coarse fallback. The dispatcher is now identified
    # by its stable alias, so the callback fingerprints cleanly and two callbacks
    # calling different bindings differ.
    from numbox.core.bindings.sqlite.value import sqlite3_value_int64, sqlite3_value_double

    def cb_int(v):
        return sqlite3_value_int64(v)

    def cb_dbl(v):
        return sqlite3_value_double(v)

    fp_int = _fingerprint_function(cb_int, set())   # must not raise
    fp_dbl = _fingerprint_function(cb_dbl, set())
    assert "proxy(numbox_pxy_" in fp_int
    assert fp_int != fp_dbl


def test_proxy_binding_canon_does_not_recurse_into_intrinsic():
    from numbox.core.bindings.sqlite.value import sqlite3_value_int64

    canon = _canon_value(sqlite3_value_int64, set())
    assert canon.startswith("proxy(numbox_pxy_sqlite3_value_int64_")


def test_unfingerprintable_value_still_raises_for_strict_walker():
    # The strict walker must keep raising on a genuinely un-canonicalizable value
    # (so degrade-to-uncached consumers still trigger); only the best-effort walker
    # swallows it.
    class Opaque:
        pass

    def cb():
        return Opaque()

    # Opaque() is a const-less local; reference it via a closure to force the walk.
    marker = Opaque()

    def cb2():
        return marker

    try:
        _fingerprint_function(cb2, set())
    except _Unfingerprintable:
        pass
    else:
        raise AssertionError("expected _Unfingerprintable for an opaque closure value")


def test_loaded_global_names_distinguishes_globals_from_attributes():
    """An attribute access is a ``LOAD_ATTR``; only ``LOAD_GLOBAL`` / ``LOAD_NAME`` name globals."""
    attr = _make("def f(a):\n    return a.c1\n")
    assert _referenced_global_names(attr.__code__) == {"c1"}, "co_names should pool the attribute name"
    assert _loaded_global_names(attr.__code__) == set(), "an attribute access names no global"

    glob = _make("def f(x):\n    return x + K\n", K=5)
    assert "K" in _loaded_global_names(glob.__code__)

    nested = _make("def f(xs):\n    return [y + K for y in xs]\n", K=5)
    assert "K" in _loaded_global_names(nested.__code__), "a global read inside a comprehension counts"


def test_fingerprint_ignores_a_global_shadowed_by_an_attribute_name():
    """A record-field / chained attribute whose name matches a module global must not fold that global.

    With a canonicalizable colliding global the old behaviour over-invalidated (folded a value the body
    never reads); with an un-canonicalizable one it raised, discarding the whole fingerprint. Neither may
    happen now: ``a.c1`` reads an attribute, not the global ``c1``.
    """
    canon = _make("def f(a):\n    return a.c1\n", c1=3)
    assert "c1=" not in _fingerprint_function(canon, set()), "folded a global the body only accesses as an attribute"

    uncanon = _make("def f(a):\n    return a.c1\n", c1=np.array([object()], dtype=object))
    # Must not raise despite the colliding global having no canonical form.
    fp = _fingerprint_function(uncanon, set())
    assert "c1=" not in fp

    # The fingerprint depends on the attribute name itself (it is still in co_names), so a different
    # field is a different body -- the fix drops only the spurious global-value fold, not the attribute.
    other = _make("def f(a):\n    return a.c2\n", c1=3)
    assert _fingerprint_function(canon, set()) != _fingerprint_function(other, set())


def test_fingerprint_still_folds_a_genuine_global_read():
    """A real ``LOAD_GLOBAL`` read is folded, and changing its value re-keys the fingerprint."""
    f = _make("def f(x):\n    return x * K\n", K=2)
    assert "K=" in _fingerprint_function(f, set())
    g = _make("def f(x):\n    return x * K\n", K=9)
    assert _fingerprint_function(f, set()) != _fingerprint_function(g, set())

    # An un-canonicalizable *genuine* global read still raises -- that degrade path is intended and unchanged.
    bad = _make("def f(x):\n    return x * K\n", K=np.array([object()], dtype=object))
    try:
        _fingerprint_function(bad, set())
    except _Unfingerprintable:
        pass
    else:
        raise AssertionError("an un-canonicalizable genuine global read should still be _Unfingerprintable")


def test_module_attribute_read_still_rekeys(tmp_path):
    """Non-regression: a value read through a module (``cfg.SCALE``) is still folded, so changing it
    re-keys; an attribute present on the module but never referenced (``cfg.OFFSET``) does not. The fix keeps
    passing the full ``co_names`` set to the module-attribute walk, so this path is unaffected."""
    import types

    cfg = types.ModuleType("cfgfake")
    cfg.SCALE = 2.0
    cfg.OFFSET = 9.0
    f = _make("def f(x):\n    return x * cfg.SCALE\n", cfg=cfg)

    before = _fingerprint_function(f, set())
    cfg.OFFSET = 99.0
    assert _fingerprint_function(f, set()) == before, "an unread module attribute must not re-key"
    cfg.SCALE = 3.0
    assert _fingerprint_function(f, set()) != before, "a read module attribute must re-key"


def test_an_external_function_is_canonicalized_by_symbol_and_signature():
    """Two bindings a factory made over different C symbols must not share a fingerprint.

    ``ExternalFunction`` has no ``_canon_value`` branch of its own until one is written for it,
    so it used to reach the terminal ``raise`` and, on the best-effort path that the ``@proxy``
    alias runs on, collapse to ``<opaque:ExternalFunction>``: the same placeholder for every
    symbol. numba's own key for the type is ``(symbol, signature)``, both of which mean the same
    thing in the next process, so folding those separates the two without costing stability.
    """
    from numba.core.types import ExternalFunction, float64

    sin = _canon_value(ExternalFunction("sin", float64(float64)), set())
    cos = _canon_value(ExternalFunction("cos", float64(float64)), set())
    assert sin != cos, "two C symbols collapsed onto one canonical form"
    assert "sin" in sin and "cos" in cos
    assert _canon_value(ExternalFunction("sin", float64(float64)), set()) == sin, (
        "the canonical form has to be the same for an equal value, or it is not a fingerprint"
    )


def test_a_library_bound_ctypes_pointer_is_canonicalized_by_library_and_symbol():
    """The same for a pointer reached by attribute access on a loaded library.

    What is folded is the string the library was opened with, not the library the loader
    resolved it to, so two handles on one library opened under different strings -- a soname
    and an absolute path -- do canonicalize differently. That is over-discrimination: it
    costs a recompile the load-time guard heals, never a wrong answer.
    """
    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take two symbols from")
    floor = _canon_value(libm.floor, set())
    ceil = _canon_value(libm.ceil, set())
    assert floor != ceil, "two C functions from one library collapsed onto one canonical form"
    assert _canon_value(open_libm().floor, set()) == floor, (
        "the canonical form followed the handle object rather than what it names"
    )


def test_a_ctypes_pointer_built_from_an_address_stays_unfingerprintable():
    """The residual class, and why it is deliberate.

    A pointer built from a raw address carries no library and no symbol name. The only thing
    separating two of them is the address, which ASLR moves, so folding it would buy
    discrimination at the cost of the process-stability the fingerprint exists for. Refusing is
    the honest answer, and it leaves the caller to detect the collision instead of hiding it.

    The refusal has to come from the ctypes branch deciding it, not from the walker never
    reaching one. A walker with no ctypes handling at all refuses this pointer too, and would
    satisfy a bare ``pytest.raises``; what separates the two is the message, which the terminal
    raise fills with the type name and the ctypes branch fills with its own reason.
    """
    import ctypes

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take a symbol address from")
    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
    fp = proto(ctypes.cast(libm.floor, ctypes.c_void_p).value)
    with pytest.raises(_Unfingerprintable) as refusal:
        _canon_value(fp, set())
    assert str(refusal.value) != type(fp).__name__, (
        "the refusal is the walker's terminal type-name raise, so the ctypes branch was never "
        "reached and this passes against a walker that cannot fingerprint any pointer at all"
    )


def test_a_proxied_binding_and_a_cres_are_not_canonicalized_alike_as_function_values():
    """Two first-class function values must not fold onto one canonical form.

    ``CompileResultWAP`` had no branch here, so a body capturing one reached the terminal
    ``raise`` and collapsed to ``<opaque:{type name}>`` on the best-effort path the alias
    runs on. That placeholder holds a Python type name and nothing else, so what separated
    a proxied binding's ``.as_func`` from a :func:`~numbox.utils.highlevel.cres` value was
    the accident that one was a ``CompileResultWAP`` and the other a ``DeriveWAP``. Two
    otherwise-identical bodies over those two handles then shared an alias the moment both
    became the same type, which retires the alias and costs both cross-process caching.

    What is folded instead means the same thing next run: the binding's own content-addressed
    alias for the first, and the compiled body's fingerprint with its signature for the
    second. The wrapper itself offers neither -- it carries a compiled address and a mangled
    name holding a per-process counter -- so numbox tags the two it mints.
    """
    from numba import float64

    from numbox.core.proxy.proxy import proxy
    from numbox.utils.highlevel import cres

    @proxy(float64(float64))
    def binding(x):
        return x + 1.0

    @cres(float64(float64))
    def compiled(x):
        return x + 1.0

    as_func = _canon_value(binding.as_func, set())
    derive = _canon_value(compiled, set())
    assert as_func != derive, "a proxied binding and a cres folded onto one canonical form"
    assert binding._numbox_proxy_alias in as_func, (
        "the binding's function value is not keyed by the alias that identifies its body"
    )
    assert _canon_value(binding.as_func, set()) == as_func, (
        "the canonical form has to be the same for an equal value, or it is not a fingerprint"
    )
    assert _canon_value(binding, set()) != as_func, (
        "the dispatcher and its function value fold alike, though they call differently"
    )


def test_a_function_value_numbox_did_not_mint_stays_unfingerprintable():
    """The residual class here, for the same reason as the raw-address pointer.

    A ``CompileResultWAP`` built directly against numba carries no tag, and the compile
    result identifies its body only by a compiled address and a mangled name whose abi-tag
    is a per-process counter. Neither means the same thing next run, so refusing is the
    honest answer and leaves the collision to be detected rather than hidden.

    Refusing this one only means something while numbox's own function value of the same shape
    is accepted. The untagged wrapper falls through to the very terminal raise a walker with no
    ``CompileResultWAP`` branch at all would give it, so the ``cres`` below is what says the tag
    is doing the separating rather than the type being unreachable.
    """
    from numba import float64, njit
    from numba.core.types.function_type import CompileResultWAP

    from numbox.utils.highlevel import cres

    compiled = njit(float64(float64))(lambda x: x + 1.0)
    foreign = CompileResultWAP(compiled.get_compile_result(compiled.nopython_signatures[0]))

    @cres(float64(float64))
    def minted(x):
        return x + 1.0

    assert _canon_value(minted, set()), (
        "numbox's own function value is unfingerprintable too, so the refusal below is not "
        "about the missing tag and rules nothing out"
    )
    with pytest.raises(_Unfingerprintable):
        _canon_value(foreign, set())


def test_two_python_callbacks_sharing_a_name_are_not_canonicalized_alike():
    """A Python callback is not a library symbol, and must not be keyed as if it were.

    ``_objects["0"]`` holds the loaded library for a pointer taken off one, but for a
    callback built from a Python function it holds the CThunkObject instead, which has no
    ``_name``. Reading that as a library folded the ``None`` into the key, so two
    unrelated callbacks that happened to share a ``__name__`` -- which any decorator
    copying ``__name__`` arranges -- canonicalized alike, and the alias they mint stopped
    telling them apart.
    """
    import ctypes

    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
    one = proto(lambda x: x + 1.0)
    two = proto(lambda x: x + 2.0)
    one.__name__ = two.__name__ = "cb"

    assert one(1.0) != two(1.0), "the two callbacks are not distinguishable to begin with"
    for cb in (one, two):
        with pytest.raises(_Unfingerprintable):
            _canon_value(cb, set())


def test_unset_and_empty_argtypes_are_canonicalized_apart():
    """An unset ``argtypes`` and an empty one must not fold onto one canonical form.

    numba refuses a call through a pointer whose ``argtypes`` is unset and accepts one
    declared to take nothing, so a key folding the two together pairs a caller with a body
    it cannot call. ``getattr(value, "argtypes", None) or ()`` folds them together, and that
    is the implementation this rules out: the two pointers below differ in nothing else, so
    their canonical forms have to differ with that attribute.

    They come off two handles because ``CDLL.__getattr__`` caches the pointer it builds on
    the instance: one handle asked twice hands back one object, and setting ``argtypes``
    would set it on both. ``open_libm`` opening a fresh handle per call is what keeps this a
    two-object comparison, and is pinned by ``test_open_libm_hands_back_a_usable_handle``.
    """
    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take a symbol from")
    unset = open_libm().floor
    assert unset.argtypes is None, "ctypes stopped leaving argtypes unset by default"
    empty = open_libm().floor
    assert unset is not empty, "both pointers came off one handle, so setting argtypes sets both"
    empty.argtypes = ()
    assert _canon_value(unset, set()) != _canon_value(empty, set()), (
        "an unset argtypes and an empty one collapsed onto one canonical form"
    )


def test_two_prototypes_over_one_address_are_not_read_as_one_value():
    """An address is not a call, so it cannot decide on its own that two values are one body.

    ``_opaque_values_match`` answers whether two equal fingerprints stood on the same values,
    and a raw-address ctypes pointer is what reaches it, because :func:`_ctypes_func_key`
    refuses one for want of a process-stable identity. Comparing two of them by the address
    alone is the implementation this rules out. numba types a call from ``restype``,
    ``argtypes`` and the convention and lowers it against them, so one symbol reached through
    two prototypes is two calls that compile to two different bodies; reading them as one
    value hands the second the alias the first was published under, with no collision to
    report and no warning.

    The three mismatching pairs differ from the first pointer in the return type, in the
    arguments, and in both, so a fix folding only one half of the prototype still fails one
    of them. The matching pair is what stops the check reading every module reload and every
    twice-called factory as a collision, which retires their alias and costs both bodies
    cross-process caching for nothing. It has to be two objects over one address, or the
    identity test at the top of the comparison answers before the pointer branch is reached.
    """
    import ctypes

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take a symbol address from")
    addr = ctypes.cast(libm.floor, ctypes.c_void_p).value
    as_double = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(addr)
    other_restype = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_double)(addr)
    other_argtypes = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(addr)
    as_float = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float)(addr)
    again = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(addr)

    assert as_double is not again, (
        "one object compared with itself is answered by the identity test, not by the pointer branch"
    )
    for other in (other_restype, other_argtypes, as_float, again):
        assert ctypes.cast(other, ctypes.c_void_p).value == addr, (
            "the pointers do not share an address, so nothing here reaches the comparison it is about"
        )
    for differing, part in ((other_restype, "return type"), (other_argtypes, "arguments"),
                            (as_float, "return type and arguments")):
        assert not _opaque_values_match([as_double], [differing]), (
            f"two prototypes differing in the {part} over one address read as one body"
        )
    assert _opaque_values_match([as_double], [again]), (
        "one prototype over one address read as two bodies, so a module reload and a factory "
        "called twice both turn into collisions"
    )


def test_a_retired_alias_is_not_folded_into_a_body_that_captured_it():
    """A name that stopped identifying a body must not go on identifying it to its captors.

    An alias is retired when two bodies reach it, because which of them ends up under the
    shared name and which under the minted one is decided by the order this process built
    them in. That reasoning does not stop at the two bodies. A third body capturing one of
    them and folding ``proxy(<alias>)`` gets a fingerprint that means the other body in a
    process that built the pair the other way round, and it is published under a fresh name
    that resolves, so unlike the colliding pair it keeps a cross-process cache that is now
    wrong.

    Falling through to the ordinary walk is what prevents it. The dispatcher has no canonical
    form there, so a derived body reaches ``_opaque_values_match`` like any other pair the
    fingerprint cannot separate, and its own alias is retired in turn.
    """
    import ctypes

    from numba import float64

    from numbox.core.proxy.proxy import AliasCollisionWarning, proxy, _ABSENT_ALIASES

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take two distinct symbol addresses from")
    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

    def make(symbol):
        fp = proto(ctypes.cast(getattr(libm, symbol), ctypes.c_void_p).value)

        @proxy(float64(float64))
        def body(x):
            return fp(x)
        return body

    winner = make("floor")
    with pytest.warns(AliasCollisionWarning):
        loser = make("ceil")

    assert winner._numbox_proxy_alias != loser._numbox_proxy_alias, (
        "the two bodies did not collide, so nothing here is about a retired alias"
    )
    for binding, which in ((winner, "shared"), (loser, "minted")):
        alias = binding._numbox_proxy_alias
        assert alias in _ABSENT_ALIASES, (
            f"the {which} name was not retired, so this asserts nothing about folding one"
        )
        with pytest.raises(_Unfingerprintable):
            _canon_value(binding, set())
        with pytest.raises(_Unfingerprintable):
            _canon_value(binding.as_func, set())


def test_a_live_alias_is_still_folded():
    """The refusal above is keyed on retirement, not on being a proxied binding at all.

    A binding whose alias still identifies it is the ordinary case and has to keep folding by
    that alias, or every body capturing any proxied binding becomes un-fingerprintable and the
    cache digests that rest on it degrade.
    """
    from numba import float64

    from numbox.core.proxy.proxy import proxy

    @proxy(float64(float64))
    def uncollided(x):
        return x * 3.0

    assert _canon_value(uncollided, set()) == f"proxy({uncollided._numbox_proxy_alias})", (
        "a binding holding a name that identifies it stopped folding by that name"
    )
    assert _canon_value(uncollided.as_func, set()) == f"proxyfunc({uncollided._numbox_proxy_alias})", (
        "the function value of an uncollided binding stopped folding by its alias"
    )
