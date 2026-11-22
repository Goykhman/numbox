"""Tests for ``numbox.utils.fingerprint``, the texts that describe functions and values the same way in every process.

The tests check that ``_canon_value`` gives different texts to values that numba compiles into different code, such
as structured arrays with the same bytes, two C symbols, ctypes pointers whose ``argtypes`` differ and two kinds of
function value. They check that it raises ``_Unfingerprintable`` for a value with no text that means the same thing in
another process.
Other tests check that a fingerprint folds the globals and the module attributes that a function reads, and not a
module global that only shares an attribute's name. ``co_names`` holds both kinds of name, so ``_loaded_global_names``
reads the opcodes. Folding such a global would change the fingerprint when only the global changes, and a global
without a fingerprint would make the fingerprint fail, so the work builder would compile the derive without a cache.
The last tests cover ``_opaque_values_match`` and the ``@proxy`` aliases that a collision retires. The tests that need
a C math library skip if ``open_libm`` finds none.
"""
import importlib.util
import math
import sys
import textwrap
import types

import numpy as np
import pytest
from numba import float64, njit, vectorize

from numbox.utils.fingerprint import (
    _MAX_REPR_INT_BITS, _Unfingerprintable, _canon_value, _fingerprint_function, _fingerprint_function_best_effort,
    _fingerprint_module_attrs, _loaded_global_names, _named_function_text, _opaque_values_match,
    _referenced_global_names,
)
from test.auxiliary_utils import open_libm


def _make(src, **globs):
    """Return the function ``f`` that the source ``src`` defines, compiled with ``globs`` as its module globals."""
    ns = dict(globs)
    exec(textwrap.dedent(src), ns)  # nosec B102 - test helper compiling a fixed snippet
    return ns["f"]


def test_canon_value_structured_dtype_preserves_layout():
    # dtype.str is |V<itemsize> for every structured dtype and drops the fields, so arrays with the same bytes and
    # different layouts would have one text. _dtype_key folds the dtype's str, descr and itemsize.
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


def test_canon_value_separates_the_field_orders_of_a_reordered_structured_array():
    # NumPy defines no descr for the dtype of base[["b", "a"]], whose fields are not in the order of their offsets.
    # The dtype's str lists the offsets, so the text tells that view from the views with other fields or another order.
    base = np.zeros(2, dtype=[("a", "<f8"), ("b", "<f8"), ("c", "<f8")])

    canons = {_canon_value(base[names], set()) for names in (["b", "a"], ["a", "b"], ["c", "a"])}
    assert len(canons) == 3
    assert _canon_value(base[["b", "a"]][0], set()) != _canon_value(base[["c", "a"]][0], set())


def test_canon_value_writes_an_int_the_same_way_under_every_digit_limit():
    # repr raises for an int of more decimal digits than the interpreter's limit, and a process can change the limit.
    # An int above the bound is written with hex, which has no limit. An int at the bound has no more digits than the
    # lowest limit that a process can set, so repr accepts it in every process.
    huge = 7 ** 6000

    assert _canon_value(huge, set()) == hex(huge)
    assert len({_canon_value(value, set()) for value in (huge, huge + 1, -huge)}) == 3
    assert _canon_value([huge], set()) == f"list[{hex(huge)}]"

    bound = 2 ** _MAX_REPR_INT_BITS - 1
    assert _canon_value(bound, set()) == repr(bound)
    assert len(repr(bound)) <= sys.int_info.str_digits_check_threshold


def test_named_function_text_names_a_ufunc_or_a_builtin_function_that_its_module_holds():
    # numpy holds sin under the ufunc's own name, and math holds sqrt, so the module and the name identify each of them
    # in every process. np.abs is a second name for the ufunc absolute, and the text holds the ufunc's own name.
    assert _named_function_text(np.sin) == "ufunc(numpy.sin)"
    assert _named_function_text(np.abs) == "ufunc(numpy.absolute)"
    assert _named_function_text(math.sqrt) == "builtin(math.sqrt)"
    assert _named_function_text(len) == "builtin(builtins.len)"
    # The strict walk does not use the name: a value that _canon_value has no text for makes the code that reads it
    # compile without a cache, and that stays so for these functions.
    for value in (np.sin, math.sqrt, len):
        with pytest.raises(_Unfingerprintable):
            _canon_value(value, set())


def test_named_function_text_is_none_for_a_function_that_no_module_holds_by_name(monkeypatch):
    # The text names a function by its module and its name, so it must not be given to another function of that name.
    # A ufunc that the user built and named sin is not the one that numpy holds. The parallel target of vectorize
    # returns a NumPy ufunc. A method of an object records no module. A function that was replaced in its module is
    # no longer found there, and the function that replaced it is named as its own module holds it.
    @vectorize([float64(float64)], target="parallel")
    def sin(x):
        return x + 1.0

    assert isinstance(sin, np.ufunc) and sin.__name__ == "sin"
    replaced = math.sqrt
    monkeypatch.setattr(math, "sqrt", math.sin)
    # A builtin function accepts any object as its __module__. One whose __module__ is not a string has no name.
    monkeypatch.setattr(math.fsum, "__module__", ["math"])

    for value in (sin, [].append, replaced, math.fsum, float64, 2.0, test_canon_value_structured_dtype_preserves_layout):
        assert _named_function_text(value) is None
    assert _named_function_text(math.sqrt) == "builtin(math.sin)"


def test_named_function_text_reads_only_a_loaded_module(tmp_path, monkeypatch):
    # A builtin function accepts any string as its __module__, and sys.modules can hold anything under that string.
    # Reading the __dict__ of a module that importlib.util.LazyLoader has not loaded yet runs the module, so the lookup
    # must leave such an entry alone, and an entry that is no module as well.
    source = tmp_path / "lazy_home.py"
    source.write_text("raise RuntimeError('the lookup ran a lazy module')\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("lazy_home", source)
    spec.loader = importlib.util.LazyLoader(spec.loader)
    lazy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lazy)
    monkeypatch.setitem(sys.modules, "lazy_home", lazy)
    monkeypatch.setitem(sys.modules, "no_module", types.SimpleNamespace(fsum=math.fsum))

    for home in ("lazy_home", "no_module"):
        monkeypatch.setattr(math.fsum, "__module__", home)
        assert _named_function_text(math.fsum) is None
    assert type(lazy) is not types.ModuleType, "the lookup loaded the lazy module"


def test_named_function_text_does_not_ask_an_impostor_for_its_name():
    # An object can report any class as its __class__, and isinstance believes it. Such an object is not a ufunc or a
    # builtin function, and reading its __name__ can run arbitrary code.
    class Impostor:
        def __init__(self, reported):
            self._reported = reported

        @property
        def __class__(self):
            return self._reported

        @property
        def __name__(self):
            raise RuntimeError("the lookup read the name of an impostor")

    helpers = types.ModuleType("helpers")
    for reported in (np.ufunc, type(len)):
        impostor = Impostor(reported)
        assert isinstance(impostor, reported)
        assert _named_function_text(impostor) is None
        helpers.op = impostor
        assert _fingerprint_module_attrs({"op"}, {"helpers": helpers}, set(), True) == ""


def test_the_fold_runs_no_code_of_a_str_subclass_or_of_an_odd_module(monkeypatch):
    # A module's __name__ and a builtin function's __module__ can be instances of a subclass of str, whose methods
    # are user code. A subclass of the module type can provide a __dict__ that is not a dict. The fold reads such a
    # __dict__ only by membership and by item, as the walk over data attributes does.
    class Loud(str):
        def partition(self, sep):
            raise RuntimeError("the fold called a method of a str subclass")

        def __hash__(self):
            raise RuntimeError("the fold hashed a str subclass")

        def __eq__(self, other):
            raise RuntimeError("the fold compared a str subclass")

    helpers = types.ModuleType("helpers")
    helpers.__dict__["__name__"] = Loud("numpy.lookalike")
    helpers.op = np.sin
    assert _fingerprint_module_attrs({"op"}, {"helpers": helpers}, set(), True) == (
        ";module_attrs=[helpers.op=ufunc(numpy.sin)]"
    )

    class NoGet:
        def __init__(self, items):
            self._items = items

        def __contains__(self, key):
            return key in self._items

        def __getitem__(self, key):
            return self._items[key]

    class OddModule(types.ModuleType):
        @property
        def __dict__(self):
            return NoGet({"__name__": "odd", "op": np.cos})

    assert _fingerprint_module_attrs({"op"}, {"odd": OddModule("odd")}, set(), True) == (
        ";module_attrs=[odd.op=ufunc(numpy.cos)]"
    )

    monkeypatch.setattr(math.fsum, "__module__", Loud("math"))
    assert _named_function_text(math.fsum) is None

    # A value that is data must keep the entry that it has as data. The fold must not take it for a ufunc because
    # its class has a metaclass whose __eq__ agrees with any operand, or because it reports another __class__.
    class AgreesWithAll(type):
        def __eq__(cls, other):
            return True

        def __hash__(cls):
            return 0

    class Scale(float, metaclass=AgreesWithAll):
        pass

    class Disguised(float):
        @property
        def __class__(self):
            return np.ufunc

    plain = types.ModuleType("plain")
    for value in (Scale(2.0), Disguised(2.0)):
        plain.SCALE = value
        folded = _fingerprint_module_attrs({"SCALE"}, {"plain": plain}, set(), True)
        assert folded == _fingerprint_module_attrs({"SCALE"}, {"plain": plain}, set()) == ";module_attrs=[plain.SCALE=2.0]"


def test_a_function_that_reads_a_named_function_by_a_bare_name_has_no_fingerprint():
    # After "from math import sqrt", a function reads sqrt as a global. Such a function has no fingerprint, so code
    # that is cached by the fingerprint compiles without a cache for it. The best-effort text, which names a @proxy
    # function, holds a placeholder for sqrt and hands the function to the caller. @proxy compares the values handed
    # over, and that comparison is what tells two bodies with equal texts apart.
    src = """
        def f(x):
            return op(x)
    """
    for op in (math.sqrt, np.sin):
        f = _make(src, op=op)
        with pytest.raises(_Unfingerprintable):
            _fingerprint_function(f, set())
        opaque = []
        assert f"op=<opaque:{type(op).__name__}>" in _fingerprint_function_best_effort(f, opaque)
        assert opaque == [op]


def test_a_module_attribute_that_holds_a_ufunc_or_a_builtin_function_is_folded_by_its_name():
    # helpers.op can hold numpy.sin in one process and numpy.cos in the next. The code of f names only op, so the
    # fingerprint folds the name of the function that the attribute holds.
    helpers = types.ModuleType("helpers")
    f = _make("""
        def f(x):
            return helpers.op(x)
    """, helpers=helpers)

    texts, plain = [], []
    for op in (np.sin, np.cos, math.sqrt):
        helpers.op = op
        texts.append(_fingerprint_function(f, set(), fold_named=True))
        plain.append(_fingerprint_function(f, set()))
    assert len(set(texts)) == 3
    assert "helpers.op=ufunc(numpy.sin)" in texts[0]
    assert "helpers.op=builtin(math.sqrt)" in texts[2]
    # Without fold_named the text is the one that @proxy and cres hash into an alias. It leaves the attribute out, so
    # that an alias does not change with it.
    assert len(set(plain)) == 1 and "module_attrs" not in plain[0]
    # Another module that holds a function under the function's own name can hold another function in the next
    # process too, so that attribute is folded as well.
    helpers.sin = np.sin
    assert _fingerprint_module_attrs({"sin"}, {"helpers": helpers}, set(), True) == (
        ";module_attrs=[helpers.sin=ufunc(numpy.sin)]"
    )
    assert _fingerprint_module_attrs({"sin"}, {"helpers": helpers}, set()) == ""


@pytest.mark.parametrize("kind", [
    "global", "closure", "default", "keyword default", "tuple", "set", "dict key", "dict value", "dispatcher",
    "vectorize", "cres", "module attribute",
])
def test_fold_named_reaches_a_function_that_the_fingerprinted_function_calls(kind):
    # The attribute helpers.op is read by a helper. The fingerprinted function reaches the helper in one of the ways
    # that the walk follows, and the walk must fold the attribute on each of them.
    from numbox.utils.highlevel import cres

    helpers = types.ModuleType("helpers")
    helpers.op = np.sin
    helper = _make("""
        def f(x):
            return helpers.op(x)
    """, helpers=helpers)
    call_global = """
        def f(x):
            return helper(x)
    """
    if kind == "global":
        caller = _make(call_global, helper=helper)
    elif kind == "closure":
        def caller(x):
            return helper(x)
    elif kind == "default":
        caller = _make("""
            def f(x, g=helper):
                return g(x)
        """, helper=helper)
    elif kind == "keyword default":
        caller = _make("""
            def f(x, *, g=helper):
                return g(x)
        """, helper=helper)
    elif kind == "module attribute":
        outer = types.ModuleType("outer")
        outer.inner = njit(helper)
        caller = _make("""
            def f(x):
                return outer.inner(x)
        """, outer=outer)
    else:
        held = {
            "tuple": lambda: (helper,), "set": lambda: frozenset({helper}), "dict key": lambda: {helper: 1.0},
            "dict value": lambda: {"k": helper}, "dispatcher": lambda: njit(helper),
            "vectorize": lambda: vectorize(helper), "cres": lambda: cres(float64(float64))(helper),
        }[kind]()
        caller = _make(call_global, helper=held)

    texts, plain = [], []
    for op in (np.sin, np.cos):
        helpers.op = op
        texts.append(_fingerprint_function(caller, set(), fold_named=True))
        plain.append(_fingerprint_function(caller, set()))
    assert texts[0] != texts[1]
    assert plain[0] == plain[1]


def test_a_named_function_reached_through_its_own_package_is_not_folded():
    # numpy.add, numpy.abs and math.sqrt are the same functions in every process, so they add nothing to the
    # fingerprint of a function that calls them through their modules. A submodule of numpy that holds add as well,
    # under that name or another, is loaded in some processes and not in others. numpy.strings.add and
    # numpy.ma.core.narray are such attributes. They are not folded either, so the fingerprint does not depend on which
    # submodules are loaded.
    f = _make("""
        def f(x):
            return math.sqrt(np.abs(np.add(x, 1.0)))
    """, np=np, math=math)
    assert "module_attrs" not in _fingerprint_function(f, set(), fold_named=True)

    submodule = types.ModuleType("numpy.a_submodule")
    submodule.add = np.add
    submodule.narray = np.array
    assert _fingerprint_module_attrs({"add", "narray"}, {"sub": submodule}, set(), True) == ""
    # A module of another package whose name starts with "numpy" is not a module of numpy.
    lookalike = types.ModuleType("numpy_extras.ops")
    lookalike.add = np.add
    assert _fingerprint_module_attrs({"add"}, {"sub": lookalike}, set(), True) == (
        ";module_attrs=[sub.add=ufunc(numpy.add)]"
    )


def test_proxy_binding_callback_fingerprints_cleanly_and_distinctly():
    # A @proxy function is folded as proxy(<alias>), so the walk does not reach the @intrinsic that its wrapper calls,
    # which has no fingerprint. A callback that calls a binding therefore has a fingerprint, and callbacks that call
    # different bindings have different fingerprints.
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
    # _fingerprint_function raises for a value without a fingerprint, so the callers that compile such code without a
    # cache detect it. Only _fingerprint_function_best_effort puts a placeholder in place of the value.
    class Opaque:
        pass

    def cb():
        return Opaque()

    # cb2 captures an instance of Opaque in its closure, and the fingerprint folds closure values.
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
    """``_loaded_global_names`` returns the names that a function reads as globals, also in a comprehension.

    ``co_names``, which ``_referenced_global_names`` reads, also holds the name of an attribute that the function reads,
    and ``_loaded_global_names`` leaves that name out.
    """
    attr = _make("def f(a):\n    return a.c1\n")
    assert _referenced_global_names(attr.__code__) == {"c1"}, "co_names should pool the attribute name"
    assert _loaded_global_names(attr.__code__) == set(), "an attribute access names no global"

    glob = _make("def f(x):\n    return x + K\n", K=5)
    assert "K" in _loaded_global_names(glob.__code__)

    nested = _make("def f(xs):\n    return [y + K for y in xs]\n", K=5)
    assert "K" in _loaded_global_names(nested.__code__), "a global read inside a comprehension counts"


def test_fingerprint_ignores_a_global_shadowed_by_an_attribute_name():
    """The fingerprint of a function that reads the attribute ``a.c1`` does not fold a module global named ``c1``.

    Folding it would change the fingerprint when only the global changes, and a global without a fingerprint would make
    the fingerprint fail.
    """
    canon = _make("def f(a):\n    return a.c1\n", c1=3)
    assert "c1=" not in _fingerprint_function(canon, set()), "folded a global the body only accesses as an attribute"

    uncanon = _make("def f(a):\n    return a.c1\n", c1=np.array([object()], dtype=object))
    # The global c1 has no fingerprint, and the fingerprint of f must not raise.
    fp = _fingerprint_function(uncanon, set())
    assert "c1=" not in fp

    # The attribute's name is still in co_names, which the fingerprint folds, so a function that reads another
    # attribute has another fingerprint.
    other = _make("def f(a):\n    return a.c2\n", c1=3)
    assert _fingerprint_function(canon, set()) != _fingerprint_function(other, set())


def test_fingerprint_still_folds_a_genuine_global_read():
    """The fingerprint folds the value of a global that the function reads, so a new value gives a new fingerprint."""
    f = _make("def f(x):\n    return x * K\n", K=2)
    assert "K=" in _fingerprint_function(f, set())
    g = _make("def f(x):\n    return x * K\n", K=9)
    assert _fingerprint_function(f, set()) != _fingerprint_function(g, set())

    # A global that the function reads and that has no fingerprint makes _fingerprint_function raise.
    bad = _make("def f(x):\n    return x * K\n", K=np.array([object()], dtype=object))
    try:
        _fingerprint_function(bad, set())
    except _Unfingerprintable:
        pass
    else:
        raise AssertionError("an un-canonicalizable genuine global read should still be _Unfingerprintable")


def test_module_attribute_read_still_rekeys(tmp_path):
    """The fingerprint folds a module attribute that the function reads, such as ``cfg.SCALE``, not one it never reads.

    The walk of module attributes takes every name in ``co_names``, so it receives the attribute names that
    ``_loaded_global_names`` leaves out.
    """
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
    """``_canon_value`` gives an ``ExternalFunction`` a text of its symbol and signature, so two C symbols differ.

    The text is numba's own key for the type, and both parts mean the same thing in the next process. Without that
    case, the best-effort fingerprint that names a ``@proxy`` body would give every ``ExternalFunction`` one
    placeholder.
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
    """A ctypes pointer taken from a loaded library has a text of the library's name and the symbol.

    Two symbols of one library have different texts, and two handles opened with one string give one text. The text
    holds the string that opened the library, so one library opened under a soname and under a path gives two texts.
    The difference can cause a recompile but never a wrong result.
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
    """``_canon_value`` raises ``_Unfingerprintable`` from its ctypes case for a pointer built from a raw address.

    Only the address separates two such pointers, and ASLR changes it in every process, so the caller must decide how
    to treat the pointer. The final ``raise`` of ``_canon_value`` gives the type name as its message, so the test
    requires another message, the one from the ctypes case.
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
    """The ``.as_func`` of a ``@proxy`` function and a ``cres`` value with the same body have different texts.

    A function value holds only an address and a mangled name with a per-process counter, so numbox tags the two kinds
    of value that it creates. ``_canon_value`` folds the alias of the ``@proxy`` function for the first, and the body's
    fingerprint with the signature for the second. If the two had one text, two bodies that differ only in which of the
    values they capture would compute one alias. The collision would retire it, and neither body would be cached across
    processes.
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
    """``_canon_value`` raises ``_Unfingerprintable`` for an untagged ``CompileResultWAP`` built with numba directly.

    Its compile result identifies the body only by an address and by a mangled name whose abi tag is a per-process
    counter. The test also checks that a ``cres`` value has a text, so the refusal comes from the missing tag and not
    from the type.
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
    """``_canon_value`` raises ``_Unfingerprintable`` for a ctypes callback built from a Python function.

    For such a callback, ``_objects["0"]`` holds a ``CThunkObject``, which has no ``_name``, instead of a library. A
    text that folded the missing name would be the same for two callbacks with one ``__name__``, such as two callbacks
    to which a decorator copies the same name.
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
    """A ctypes pointer whose ``argtypes`` is unset and one whose ``argtypes`` is empty have different texts.

    numba refuses a call through the first and accepts a call through the second, so one text for both would pair a
    caller with a body that it cannot call. A text that read ``argtypes or ()`` would give both pointers one text,
    because they differ in nothing else.
    """
    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take a symbol from")
    # The pointers come from two handles, because CDLL.__getattr__ stores the pointer that it builds on the handle.
    # One handle would return one object twice, and setting argtypes on it would set it for both. open_libm opens a new
    # handle on each call, which test_open_libm_hands_back_a_usable_handle in test/core/test_proxy.py checks.
    unset = open_libm().floor
    assert unset.argtypes is None, "ctypes stopped leaving argtypes unset by default"
    empty = open_libm().floor
    assert unset is not empty, "both pointers came off one handle, so setting argtypes sets both"
    empty.argtypes = ()
    assert _canon_value(unset, set()) != _canon_value(empty, set()), (
        "an unset argtypes and an empty one collapsed onto one canonical form"
    )


def test_two_prototypes_over_one_address_are_not_read_as_one_value():
    """``_opaque_values_match`` treats two ctypes pointers over one address as one value only if their prototypes match.

    numba types and lowers a call from the pointer's ``restype``, ``argtypes`` and calling convention, so two prototypes
    over one symbol describe two calls. If the two pointers counted as one value, the second body would receive the
    alias of the first, and no collision would be reported.
    """
    import ctypes

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take a symbol address from")
    # A raw-address pointer reaches _opaque_values_match because _ctypes_func_key finds no library and symbol for it.
    # The three pointers that must not match as_double differ from it in the return type, in the arguments and in both,
    # so a comparison of only one part of the prototype fails the test. again must match as_double, or every module
    # reload and every factory called twice would count as a collision and retire the alias. again is a second object,
    # because the comparison accepts an object compared with itself before it reaches the ctypes case.
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
    """A ``@proxy`` function and its ``.as_func`` have no text once a collision has retired their alias.

    A process gives the shared alias to the colliding body that it builds first. A body that captured one of them and
    folded the alias could therefore mean the other body in another process. The comment above ``_ABSENT_ALIASES`` in
    ``numbox/utils/fingerprint.py`` describes how the walk treats such a body.
    """
    import ctypes

    from numba import float64

    from numbox.core.proxy.proxy import AliasCollisionWarning, proxy, _ABSENT_ALIASES

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take two distinct symbol addresses from")
    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

    # Each body captures a ctypes pointer built from a raw address, which has no fingerprint, so the two bodies compute
    # one alias and collide.
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
    """A ``@proxy`` function with an alias that is not retired, and its ``.as_func``, are folded by that alias.

    The refusal in the test above depends on the retirement alone. If every proxied function had no text, no body that
    captures one would have a fingerprint, so code generated for such a body would compile without a cache or be named
    by the weaker fingerprint.
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
