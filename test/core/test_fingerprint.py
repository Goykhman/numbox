"""Unit tests for the function/global fingerprint walker in ``numbox.utils.fingerprint``.

Focus: telling a genuine global read (``LOAD_GLOBAL``) apart from an attribute access
(``LOAD_ATTR``) whose name merely collides with a module global. ``co_names`` pools the
two, so folding globals off it either over-invalidates the fingerprint (a colliding
global that canonicalizes) or makes the whole function un-fingerprintable (a colliding
global that does not) -- which, on the builder's derive path, forces an address-bearing
fallback and unbounded cache growth (issue #73). ``_loaded_global_names`` reads the
opcodes instead.
"""
import textwrap

import numpy as np

from numbox.utils.fingerprint import (
    _Unfingerprintable, _fingerprint_function, _loaded_global_names, _referenced_global_names,
)


def _make(src, **globs):
    """Compile ``src`` (defining ``f``) with ``globs`` as its module globals."""
    ns = dict(globs)
    exec(textwrap.dedent(src), ns)  # nosec B102 - test helper compiling a fixed snippet
    return ns["f"]


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

    # An un-canonicalizable *genuine* global still raises -- that degrade path is intended and unchanged.
    bad = _make("def f(x):\n    return x * K\n", K=np.array([object()], dtype=object))
    try:
        _fingerprint_function(bad, set())
    except _Unfingerprintable:
        pass
    else:
        raise AssertionError("an un-canonicalizable genuine global read should still be _Unfingerprintable")


def test_module_attribute_read_still_rekeys(tmp_path):
    """T4/H6 non-regression: a value read through a module (``cfg.SCALE``) is still folded, so changing it
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
