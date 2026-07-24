import os
import re
import time
from typing import Callable

import pytest
from numba import njit, typeof
from numba.core.types import float64, int8, unicode_type
from numba.core.types.function_type import CompileResultWAP, FunctionType
from numba.core.types.functions import Dispatcher
from numpy import isclose

from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib_path
from numbox.utils.highlevel import (
    _method_identity,
    _signature_identity,
    _type_identity,
    cres,
    cres_if_available,
    determine_field_index,
    hash_type,
    make_structref,
    make_structref_code_txt,
)
from numbox.utils.preprocessing import (
    _ORPHAN_AGE_SECONDS,
    _anchor_root,
    _materialize_anchor,
    _orphan_anchor_sweep,
)
from test.auxiliary_utils import collect_and_run_tests
from test.common_structrefs import S1Type
from test.utils.auxiliaries import aux_1
from test.utils.common_struct_type_classes import S1TypeClass, S2TypeClass, S3TypeClass, S4TypeClass, S5TypeClass


def test_cres_njit():
    aux_sig = float64(float64)

    @cres(aux_sig, cache=True)
    def aux_1(x):
        return 2 * x

    @njit(aux_sig, cache=True)
    def aux_2(x):
        return 2 * x

    assert abs(aux_1(3.14) - 2 * 3.14) < 1e-15
    assert abs(aux_2(3.14) - 2 * 3.14) < 1e-15

    assert isinstance(typeof(aux_1), FunctionType)
    assert isinstance(typeof(aux_2), Dispatcher)

    @njit
    def run(func):
        return func(3.14)

    assert abs(run(aux_1) - 2 * 3.14) < 1e-15
    assert abs(run(aux_2) - 2 * 3.14) < 1e-15

    assert isinstance(run.nopython_signatures[0].args[0], FunctionType)
    assert isinstance(run.nopython_signatures[1].args[0], Dispatcher)


def test_determine_field_index():
    assert determine_field_index(S1Type, "x1") == 0
    assert determine_field_index(S1Type, "x2") == 1
    assert determine_field_index(S1Type, "x3") == 2


@njit(cache=True)
def aux_test_make_structref(s):
    return s.y


def test_make_structref_1():
    make_s1 = make_structref("S1", ("x", "y"), S1TypeClass)
    s1_1 = make_s1(3.141, 2)
    s1_2 = make_s1(2.17, 3)
    assert isclose(s1_1.x, 3.141)
    assert s1_2.y == 3
    assert aux_test_make_structref(s1_1) == 2


def test_make_structref_2():
    make_s2 = make_structref("S2", ("x", "y", "z"), S2TypeClass)
    s2_1_z = "hello"
    s2_1 = make_s2(1.41, 45, s2_1_z)
    assert isclose(s2_1.x, 1.41)
    assert s2_1.y == 45
    assert s2_1.z == s2_1_z
    assert aux_test_make_structref(s2_1) == 45


_method_digest_re = re.compile(r"(?<=_)[0-9a-f]{64}\b")

ref_s3_code_txt = """
class S3(StructRefProxy):
    def __new__(cls, x, y):
        return make_s3(x, y)

    def __repr__(self):
        return f'S3(x={self.x}, y={self.y})'

    @property
    @njit(**jit_options)
    def x(self):
        return self.x

    @property
    @njit(**jit_options)
    def y(self):
        return self.y

    def calculate_1(self, z, w=1):
        return self.calculate_1_<digest>(z, w)

    @njit(**jit_options)
    def calculate_1_<digest>(self, z, w=1):
        return self.calculate_1(z, w)

    def calculate_2(self):
        return self.calculate_2_<digest>()

    @njit(**jit_options)
    def calculate_2_<digest>(self):
        return self.calculate_2()

define_boxing(S3TypeClass, S3)

@overload(S3, strict=False, jit_options=jit_options)
def ol_s3(x_ty, y_ty):
    fields_types = [x_ty, y_ty]
    fields_and_their_types = list(zip(fields, fields_types))
    S3Type = S3TypeClass(fields_and_their_types)

    def ctor(x, y):
        struct_ = new(S3Type)
        struct_.x = x
        struct_.y = y
        return struct_
    return ctor

make_s3_sig = None

@njit(make_s3_sig, **jit_options)
def make_s3(x, y):
    return S3(x, y)

@overload_method(S3TypeClass, "calculate_1", jit_options=jit_options)
def ol_calculate_1(self, z, w=1):
    def _(self, z, w=1):
        return self.x + z * w

    return _

@overload_method(S3TypeClass, "calculate_2", jit_options=jit_options)
def ol_calculate_2(self):
    def _(self):
        return self.y * 3 + aux()

    return _
"""


def test_make_structref_3():
    def calculate_1(self, z, w=1):
        return self.x + z * w

    @njit
    def aux():
        return 0.0

    def calculate_impl(self):
        return self.y * 3 + aux()
    m1 = {"calculate_1": calculate_1, "calculate_2": calculate_impl}
    code_txt = make_structref_code_txt("S3", ("x", "y"), S3TypeClass, struct_methods=m1)[0]
    # The method-name suffix is a fingerprint that folds code.co_code, so it moves
    # with every CPython bytecode change and can only ever match one interpreter.
    # Pin what is portable -- the generated structure, that each call site names
    # the method it defines, and that distinct methods get distinct suffixes --
    # and normalise the suffix itself out of the comparison.
    digests = _method_digest_re.findall(code_txt)
    assert len(digests) == 4, digests
    assert digests[0] == digests[1], digests
    assert digests[2] == digests[3], digests
    assert digests[0] != digests[2], digests
    assert _method_digest_re.sub("<digest>", code_txt) == ref_s3_code_txt
    make_s3 = make_structref("S3", ("x", "y"), S3TypeClass, struct_methods=m1, ns={"aux": aux})
    s1 = make_s3(2.17, 3.14)

    ref = 2.17 + 5 * 6
    assert isclose(s1.calculate_1(5, 6), ref)
    assert isclose(aux_1(s1), ref), (aux_1(s1), ref)

    assert isclose(s1.calculate_2(), 9.42)


def test_make_structref_method_param_edge_cases():
    """COR-4 / task 8: make_structref must accept method signatures whose source
    spelling differs from repr() of the default (1_000, 0x10, quote style), drops
    annotations, or contains a nested ')'. The old regex cross-check assert in
    make_structref_code_txt rejected these. Codegen-only (no numba needed)."""
    def m_underscore(self, n=1_000):
        return self.x + n

    def m_hex(self, n=0x10):
        return self.x + n

    def m_str_comma(self, s="a, b"):
        return self.x

    def factory():
        return 0

    def m_factory(self, x=factory()):
        return self.x + x

    def m_callable(self, f: Callable[[int], int] = None):
        return self.x

    for name, method in [
        ("m_underscore", m_underscore),
        ("m_hex", m_hex),
        ("m_str_comma", m_str_comma),
        ("m_factory", m_factory),
        ("m_callable", m_callable),
    ]:
        code = make_structref_code_txt("S3", ("x", "y"), S3TypeClass, struct_methods={name: method})[0]
        assert isinstance(code, str) and f"def {name}(" in code, name


ref_s4_code_txt = """
class S4(StructRefProxy):
    def __new__(cls, x, y):
        return make_s4(x, y)

    def __repr__(self):
        return f'S4(x={self.x}, y={self.y})'

    @property
    @njit(**jit_options)
    def x(self):
        return self.x

    @property
    @njit(**jit_options)
    def y(self):
        return self.y

define_boxing(S4TypeClass, S4)

fields_and_their_types = list(zip(fields, fields_types))
S4Type = S4TypeClass(fields_and_their_types)

@overload(S4, strict=False, jit_options=jit_options)
def ol_s4(x_ty, y_ty):

    def ctor(x, y):
        struct_ = new(S4Type)
        struct_.x = x
        struct_.y = y
        return struct_
    return ctor

make_s4_sig = S4Type(*fields_types)

@njit(make_s4_sig, **jit_options)
def make_s4(x, y):
    return S4(x, y)
"""


def test_make_structref_4():
    assert make_structref_code_txt("S4", {"x": int8, "y": unicode_type}, S4TypeClass)[0] == ref_s4_code_txt
    make_s4 = make_structref("S4", {"x": int8, "y": unicode_type}, S4TypeClass)
    s4_1 = make_s4(50, "hello")
    assert str(typeof(s4_1)) == "numba.S4TypeClass(('x', int8), ('y', unicode_type))"
    assert str(s4_1) == "S4(x=50, y=hello)"


ref_s5_code_txt = """
class S5(StructRefProxy):
    def __new__(cls, ):
        return make_s5()

    def __repr__(self):
        return f'S5()'

define_boxing(S5TypeClass, S5)

fields_and_their_types = list(zip(fields, fields_types))
S5Type = S5TypeClass(fields_and_their_types)

@overload(S5, strict=False, jit_options=jit_options)
def ol_s5():

    def ctor():
        struct_ = new(S5Type)

        return struct_
    return ctor

make_s5_sig = S5Type(*fields_types)

@njit(make_s5_sig, **jit_options)
def make_s5():
    return S5()
"""


def test_make_structref_5():
    assert make_structref_code_txt("S5", {}, S5TypeClass)[0] == ref_s5_code_txt
    make_s5 = make_structref("S5", (), S5TypeClass)
    s5_1 = make_s5()
    assert str(s5_1) == "S5()"


def _locate_cos_lib():
    from numbox.core.bindings.utils import platform_
    if platform_ == "Windows":
        from ctypes.util import find_msvcrt
        return find_msvcrt()
    from ctypes.util import find_library
    return find_library("m")


def test_cres_if_available_present_symbol_returns_real_wrapper():
    lib_path = _locate_cos_lib()
    if lib_path is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    prior = signatures.get("cos")
    try:
        signatures["cos"] = float64(float64)
        lib = load_lib_path(lib_path)

        @cres_if_available(lib, signatures["cos"])
        def cos(x):
            return _call_lib_func("cos", (x,))

        assert isinstance(cos, CompileResultWAP)
    finally:
        if prior is None:
            signatures.pop("cos", None)
        else:
            signatures["cos"] = prior


def test_cres_if_available_missing_symbol_returns_stub():
    lib_path = _locate_cos_lib()
    if lib_path is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    prior = signatures.get("nonexistent_fn")
    try:
        signatures["nonexistent_fn"] = float64(float64)
        lib = load_lib_path(lib_path)

        @cres_if_available(lib, signatures["nonexistent_fn"])
        def nonexistent_fn(x):
            return x

        assert nonexistent_fn.__name__ == "nonexistent_fn"
        with pytest.raises(NotImplementedError, match="nonexistent_fn is not available"):
            nonexistent_fn(1.0)
    finally:
        if prior is None:
            signatures.pop("nonexistent_fn", None)
        else:
            signatures["nonexistent_fn"] = prior


def test_orphan_anchor_sweep_spares_fresh_tmp_files():
    """Concurrent-import race protection: ``_orphan_anchor_sweep`` must not
    unlink ``.tmp-*`` files that another process just created via
    ``_materialize_anchor``'s ``mkstemp`` and is about to ``replace`` into
    place. The age filter (``_ORPHAN_AGE_SECONDS``) ensures only stale
    files get unlinked.
    """
    root = _anchor_root("numbox-structref")
    root.mkdir(parents=True, exist_ok=True)
    fresh = root / "test_sweep_race.py.tmp-FRESH"
    aged = root / "test_sweep_race.py.tmp-AGED"
    fresh.write_text("fresh content")
    aged.write_text("aged content")
    aged_mtime = time.time() - _ORPHAN_AGE_SECONDS - 5
    os.utime(aged, (aged_mtime, aged_mtime))
    try:
        _orphan_anchor_sweep("numbox-structref")
        assert fresh.exists(), "fresh .tmp-* file was unlinked — race with concurrent writer"
        assert not aged.exists(), "aged .tmp-* file was NOT unlinked — sweep is broken"
    finally:
        fresh.unlink(missing_ok=True)
        aged.unlink(missing_ok=True)


def test_materialize_anchor_writes_utf8_bytes(tmp_path):
    """Anchor file must be written as utf-8 to match ``_anchor_path``'s
    sha256(content.encode('utf-8')) addressing. On Windows the default
    encoding from ``os.fdopen`` can be cp1252; explicit utf-8 ensures
    round-trip integrity for non-ASCII source AND keeps the bytes-on-disk
    consistent with the path hash.
    """
    content = "# café — non-ASCII sentinel\nx = 1\n"
    anchor = tmp_path / "anchor.py"
    _materialize_anchor(anchor, content)
    assert anchor.read_bytes() == content.encode("utf-8")


def test_type_identity_plain_type_is_cacheable():
    h, ok = _type_identity(float64)
    assert ok is True
    assert isinstance(h, str) and len(h) == 64
    assert hash_type(float64) == h        # hash_type delegates to _type_identity


def test_type_identity_dispatcher_is_uncacheable_and_content_sensitive():
    """A Dispatcher type is un-cacheable regardless of whether its wrapped body
    fingerprints -- numba cannot cross-process-cache a Dispatcher signature -- yet
    the hash stays content-sensitive so a changed body re-keys the in-process name.
    """
    @njit
    def plain(x):
        return x + 1.0

    @njit
    def intrinsic_ref(x):
        return _call_lib_func("cos", (x,))     # references an @intrinsic -> unfingerprintable body

    h_plain, ok_plain = _type_identity(typeof(plain))
    h_intr, ok_intr = _type_identity(typeof(intrinsic_ref))
    assert ok_plain is False and ok_intr is False
    assert len(h_plain) == 64 and len(h_intr) == 64
    assert h_plain != h_intr                   # distinct bodies -> distinct hashes


def test_type_identity_nested_dispatcher_is_uncacheable():
    """A Dispatcher nested in a heterogeneous container escapes the top-level
    ``isinstance`` check but must still be un-cacheable -- numba cannot
    cross-process-cache a Dispatcher-carrying signature at any depth."""
    @njit
    def helper(x):
        return x + 1.0

    het = typeof((1.0, helper))                # Tuple(float64, Dispatcher)
    assert "CPUDispatcher(" in str(het)        # the substring the guard keys on
    _h, ok = _type_identity(het)
    assert ok is False


def test_type_identity_record_is_cacheable():
    import numpy as np
    from numba import from_dtype
    rec = from_dtype(np.dtype([("a", np.float64), ("b", np.int64)]))
    h, ok = _type_identity(rec)
    assert ok is True and len(h) == 64
    assert "Record(" in str(rec)               # exercises the structural branch


def test_signature_identity_folds_component_cacheability():
    @njit
    def helper(x):
        return x + 1.0

    text_plain, ok_plain = _signature_identity(float64(float64, float64))
    assert ok_plain is True and isinstance(text_plain, str)

    text_disp, ok_disp = _signature_identity(float64(float64, typeof(helper)))
    assert ok_disp is False                    # a Dispatcher component makes the sig un-cacheable
    assert isinstance(text_disp, str)


def test_method_identity_folds_ns_values_read_as_globals_only():
    """An `ns` value is folded when the method reads it as a global, and not
    when the method merely names it as an attribute.

    Deciding this off `co_names` cannot tell `LOAD_GLOBAL knob` from
    `LOAD_ATTR knob`, so a method doing `self.knob` against an `ns` carrying an
    unrelated `knob` folded a value it never reads -- and, when that value has no
    canonical form, dropped the struct's on-disk cache for nothing.
    """
    class Opaque:
        pass

    def reads_attribute(self):
        return self.knob

    def reads_global(self):
        return knob                            # noqa: F821 - resolved from the exec'd ns

    ident_attr, cacheable_attr = _method_identity(reads_attribute, {"knob": Opaque()})
    assert cacheable_attr is True, (
        "an attribute name colliding with an ns key dropped the cache; only a "
        "global read can reach an ns value"
    )
    assert "ns:knob" not in ident_attr

    ident_global, cacheable_global = _method_identity(reads_global, {"knob": Opaque()})
    assert cacheable_global is False           # genuinely read, and opaque -> uncacheable
    assert "ns:knob" in ident_global


if __name__ == '__main__':
    collect_and_run_tests(__name__)
