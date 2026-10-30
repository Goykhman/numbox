"""Tests for ``numbox.utils.highlevel`` and for two anchor helpers of ``numbox.utils.preprocessing``.

The tests compare the code that ``make_structref_code_txt`` generates with expected text, and they build structs whose
fields are given by name or by type, with methods and without fields, which Python and cached jitted functions then
call. They check ``cres`` against ``njit``, ``cres_if_available`` for a symbol that the C library has and for one that
it lacks, and the process-stable identities of numba types, signatures and struct methods. Two tests check how
``_materialize_anchor`` writes an anchor and which temporary files ``_orphan_anchor_sweep`` deletes.
"""
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
from test.auxiliary_utils import collect_and_run_tests, open_libm
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
    # A helper's name ends in the hash of its method's identity, which folds the method's code.co_code, so the hash
    # differs between CPython versions. The test checks that each method calls its helper under the name that the
    # helper is defined with, that the two methods have different hashes, and that the code matches the expected text
    # with the hashes replaced.
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
    """``make_structref_code_txt`` generates code for methods whose parameter list differs from the text numbox writes.

    numbox writes each default as its ``repr`` and drops annotations. The methods' defaults are written as ``1_000``, as
    ``0x10``, as a string in double quotes that holds a comma, and as a call. One parameter has an annotation. The test
    generates the code and compiles nothing.
    """
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


def test_cres_if_available_present_symbol_returns_real_wrapper():
    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    # _call_lib_func takes the result type from the signatures table, so the test sets the entry for cos and restores
    # the earlier entry afterwards.
    prior = signatures.get("cos")
    try:
        signatures["cos"] = float64(float64)

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
    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    prior = signatures.get("nonexistent_fn")
    try:
        signatures["nonexistent_fn"] = float64(float64)

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
    """``_orphan_anchor_sweep`` deletes a temporary anchor file older than ``_ORPHAN_AGE_SECONDS``, not a newer one.

    A newer file can belong to a ``_materialize_anchor`` call in another process, between its ``mkstemp`` and its
    ``replace``, and deleting the file would make the ``replace`` raise.
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
    """``_materialize_anchor`` writes the code as UTF-8, the encoding in which ``_anchor_path`` hashes it for a name.

    On Windows, ``os.fdopen`` defaults to the locale's encoding, such as cp1252, which writes other bytes for non-ASCII
    source.
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
    """A dispatcher type is not cacheable, and two dispatchers with different bodies have different digests.

    ``intrinsic_ref`` calls ``_call_lib_func``, an ``@intrinsic`` without a fingerprint, so its digest comes from the
    best-effort fingerprint, which must still distinguish the two bodies.
    """
    @njit
    def plain(x):
        return x + 1.0

    @njit
    def intrinsic_ref(x):
        return _call_lib_func("cos", (x,))

    h_plain, ok_plain = _type_identity(typeof(plain))
    h_intr, ok_intr = _type_identity(typeof(intrinsic_ref))
    assert ok_plain is False and ok_intr is False   # a dispatcher type is never cacheable
    assert len(h_plain) == 64 and len(h_intr) == 64
    assert h_plain != h_intr                   # the bodies differ, so the digests differ


def test_type_identity_cres_is_cacheable():
    """The type of a ``cres`` value is cacheable, because its ``str`` holds the signature and no address or counter.

    numba can cache code typed with it across processes, unlike code typed with a dispatcher type.
    """
    @cres(float64(float64))
    def compiled(x):
        return x + 1.0

    ty = typeof(compiled)
    # test_cres_mints_a_derive_wap in test/utils/test_derive_wap.py checks that cres returns a DeriveWAP. This test
    # checks that the str of the value's type holds only the signature.
    assert str(ty) == "DeriveFunctionType[float64(float64)]"
    h, ok = _type_identity(ty)
    assert ok is True and len(h) == 64


def test_type_identity_nested_dispatcher_is_uncacheable():
    """A tuple type that holds a dispatcher type is not cacheable, although the tuple type is not a dispatcher type.

    A dispatcher type holds an address that differs between processes, so a signature that holds one at any depth
    misses numba's cache in every process.
    """
    @njit
    def helper(x):
        return x + 1.0

    het = typeof((1.0, helper))                # Tuple(float64, Dispatcher)
    assert "CPUDispatcher(" in str(het)        # the substring that _type_identity looks for
    _h, ok = _type_identity(het)
    assert ok is False


def test_type_identity_record_is_hashed_structurally_not_mangled():
    """A Record type is hashed by its ``str``, which describes its fields, not by numba's mangled name for it.

    numba's mangled name for a Record holds a counter of the order in which the process creates types, so it differs
    between processes. The digest must differ from the hash of the mangled name, and two records with swapped field
    types must have different digests.
    """
    import hashlib
    import numpy as np
    from numba import from_dtype
    from numba.core.itanium_mangler import mangle_type_or_value

    rec = from_dtype(np.dtype([("a", np.float64), ("b", np.int64)]))
    swapped = from_dtype(np.dtype([("a", np.int64), ("b", np.float64)]))
    h, ok = _type_identity(rec)
    assert ok is True and len(h) == 64
    assert h != hashlib.sha256(mangle_type_or_value(rec).encode("utf-8")).hexdigest()
    assert _type_identity(swapped)[0] != h     # the swapped field types give another digest


def test_signature_identity_folds_component_cacheability():
    @njit
    def helper(x):
        return x + 1.0

    @cres(float64(float64))
    def compiled(x):
        return x + 1.0

    text_plain, ok_plain = _signature_identity(float64(float64, float64))
    assert ok_plain is True and isinstance(text_plain, str)

    # The type of a cres value in a signature leaves the signature cacheable.
    text_cres, ok_cres = _signature_identity(float64(float64, typeof(compiled)))
    assert ok_cres is True and isinstance(text_cres, str)

    text_disp, ok_disp = _signature_identity(float64(float64, typeof(helper)))
    assert ok_disp is False                    # a dispatcher type makes the signature uncacheable
    assert isinstance(text_disp, str)


def test_method_identity_folds_ns_values_read_as_globals_only():
    """An ``ns`` value enters a method's identity if the method reads it as a global, not if an attribute has its name.

    ``co_names`` holds global and attribute names alike, so ``_method_identity`` reads the opcodes instead. Folding a
    value that the method never reads would change the identity of a method whose behavior is the same, and a value
    without a fingerprint would turn off the struct's cache.
    """
    class Opaque:
        pass

    def reads_attribute(self):
        return self.knob

    def reads_global(self):
        return knob                            # noqa: F821 - in a struct, ns provides knob

    ident_attr, cacheable_attr = _method_identity(reads_attribute, {"knob": Opaque()})
    assert cacheable_attr is True, (
        "an attribute name colliding with an ns key dropped the cache; only a "
        "global read can reach an ns value"
    )
    assert "ns:knob" not in ident_attr

    ident_global, cacheable_global = _method_identity(reads_global, {"knob": Opaque()})
    assert cacheable_global is False           # read as a global, and without a fingerprint
    assert "ns:knob" in ident_global

    # A global that the method reads and that has a fingerprint keeps the struct cacheable. Its value is folded, not
    # only its name, so a new value changes the identity.
    ident_five, cacheable_five = _method_identity(reads_global, {"knob": 5})
    assert cacheable_five is True
    assert "ns:knob" in ident_five
    assert _method_identity(reads_global, {"knob": 6})[0] != ident_five


if __name__ == '__main__':
    collect_and_run_tests(__name__)
