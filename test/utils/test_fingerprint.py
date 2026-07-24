"""Tests for the content-fingerprint walker (numbox.utils.fingerprint)."""
import numpy as np

from numbox.utils.fingerprint import _canon_value, _fingerprint_function, _Unfingerprintable


def test_canon_value_structured_dtype_preserves_layout():
    # dtype.str is |V<n> for structured dtypes (field layout erased), so same-byte
    # different-layout arrays would collide. The full descriptor discriminates them
    # (issue #73 L21).
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
    # cache digest degraded to a coarse fallback (issue #73 H8). The dispatcher is
    # now identified by its stable alias, so the callback fingerprints cleanly and
    # two callbacks calling different bindings differ.
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
