"""Tests of Params, the declaration of a variable for compile_kernel, and of how compile_kernel uses the declarations.

compile_kernel uses the declarations only when every calculated variable in the requested part of the graph, and every
external variable that a formula reads, has a declared type. _classify then assigns case A, one fused kernel, or case
B, compiled segments around the formulas declared ``jitable=False``. Any other graph is case C, whose kernel
determines its mode at the first call. The tests cover the classification and the return-type check, for each kind of
formula: a plain function, a dispatcher, a @vectorize function, a @cfunc and a cres function. They check that
compile_kernel compiles a declared kernel or its segments before the first call, and that recompute refuses a value
that numba cannot convert to the declared type. They also check that a kernel built from declared types raises an error
instead of searching again. test_compile_kernel.py tests kernels without declarations.
"""
import numpy as np
import pytest
from dataclasses import FrozenInstanceError
from numba import cfunc, float32, float64, int32, int64, vectorize
from numba import njit as _njit, int64 as _int64, float64 as _float64
from numba import njit as _njit_t3
from numba.core.errors import NumbaError
from numbox.core.variable.variable import (
    Graph, Params, Variable, Variables, External,
)
from numbox.core.variable.compile_kernel import _classify, compile_kernel
from numbox.core.variable._kernel_partition import _evaluate as _evaluate_fn
from numbox.core.variable.utils import _validate_declared_return
from numbox.utils.highlevel import cres


def test_params_frozen_defaults():
    """A Params instance declares a jittable formula without a type by default, and it is frozen."""
    p = Params()
    assert p.jitable is True and p.type is None
    assert Params(type=float64).type is float64
    with pytest.raises(FrozenInstanceError):
        p.jitable = False


def test_variable_params_roundtrip_and_identity_unchanged():
    """A variable keeps its declaration, which the variable's equality and hash ignore.

    A variable with a declaration therefore equals the same variable without one, and a set of the two has one element.
    """
    a = Variable(name="a", source="m", params=Params(type=float64))
    assert a.params.type is float64
    bare = Variable(name="a", source="m")
    assert a == bare and hash(a) == hash(bare)
    assert {a, bare} == {a}


def test_varspec_params_passthrough():
    """A Variables namespace passes the ``params`` entry of a specification to the variable that it creates."""
    vs = Variables("m", [{"name": "a", "formula": lambda: 1.0, "params": Params(type=float64)}])
    assert vs["a"].params.type is float64


def test_external_declare_attaches_params():
    """External.declare stores a variable whose declaration a later lookup of the name returns."""
    e = External("ext")
    e.declare("x", Params(type=int64))
    assert e["x"].params.type is int64


def _graph_all_jittable():
    """Return a graph in case A, in which every variable is declared float64.

    c.a computes e.x + 1.0, and c.b computes c.a * 2.0.
    """
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0, "params": Params(type=float64)},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0, "params": Params(type=float64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    return g


def test_classify_case_a_all_jittable():
    """_classify assigns case A, one fused kernel, to a graph whose formulas are jittable and declared with a type.

    The graph's external variable has a declared type as well.
    """
    g = _graph_all_jittable()
    compiled = g.compile(["c.b"])
    case, dispositions, consumed = _classify(compiled)
    assert case == "A"
    assert all(d == "STATIC_JIT" for d in dispositions.values())


def test_classify_case_b_declared_python_mix():
    """_classify assigns case B, compiled segments around the Python formulas, when c.b is declared ``jitable=False``.

    Every variable of the graph, including c.b, is declared with a type.
    """
    g = _graph_all_jittable()
    g.registry["c"].update("b", Variable(
        name="b", source="c", inputs={"a": "c"},
        formula=lambda a: a * 2.0, params=Params(jitable=False, type=float64)))
    g.registry["c"].update("d", Variable(
        name="d", source="c", inputs={"b": "c"},
        formula=lambda b: b + 1.0, params=Params(type=float64)))
    compiled = g.compile(["c.d"])
    case, dispositions, _ = _classify(compiled)
    assert case == "B"
    assert dispositions[g.registry["c"]["b"]] == "STATIC_PY"


def test_classify_case_c_untyped_python_boundary():
    """_classify assigns case C when c.b is declared ``jitable=False`` without a type.

    c.d reads c.b, so c.d has an input without a declared type. _classify then marks c.d as UNKNOWN, and a graph with
    an UNKNOWN variable is case C.
    """
    g = _graph_all_jittable()
    g.registry["c"].update("b", Variable(
        name="b", source="c", inputs={"a": "c"},
        formula=lambda a: a * 2.0, params=Params(jitable=False)))
    g.registry["c"].update("d", Variable(
        name="d", source="c", inputs={"b": "c"},
        formula=lambda b: b + 1.0, params=Params(type=float64)))
    compiled = g.compile(["c.d"])
    case, _, _ = _classify(compiled)
    assert case == "C"


def test_njit_probe_reads_natural_return_type():
    """numba infers float64 for x * 1.5 with an int64 argument when no return type is declared.

    _validate_declared_return reads this inferred type, because numba converts a result to the return type of a
    declared signature without an error.
    """
    f = _njit(lambda x: x * 1.5)
    f.compile((_int64,))
    rt = f.nopython_signatures[-1].return_type
    assert rt == _float64


def test_validate_rejects_coercible_wrong_scalar_type():
    """The return-type check refuses an int64 declaration for x * 1.5, which numba infers as float64 for int64 input.

    If numba compiled the formula for the declared signature, the formula would return its result converted to int64.
    """
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        _validate_declared_return(lambda x: x * 1.5, (int64,), int64, flags={})


def test_validate_accepts_correct_declaration():
    """The return-type check accepts a float64 declaration for x * 1.5 with int64 input."""
    _validate_declared_return(lambda x: x * 1.5, (int64,), float64, flags={})


def test_validate_rejects_nonconvertible_return():
    """The return-type check refuses an int64 declaration for a formula that returns a string."""
    with pytest.raises(ValueError):
        _validate_declared_return(lambda x: "s", (int64,), int64, flags={})


def test_validate_rejects_dufunc_wrong_output():
    """The return-type check refuses a float64 declaration for a @vectorize function whose int64 loop returns int64.

    The check reads the type from a one-line @njit function that applies the function to the declared input types.
    """
    vf = vectorize(["int64(int64)", "float64(float64)"])(lambda a: a + a)
    with pytest.raises(ValueError):
        _validate_declared_return(vf, (int64,), float64, flags={})


def test_validate_accepts_cfunc_correct_declaration():
    """The return-type check accepts the int64 declaration of a @cfunc that returns int64."""
    cf = cfunc(int64(int64))(lambda x: x + 1)
    _validate_declared_return(cf, (int64,), int64, flags={})


def test_validate_rejects_cfunc_wrong_declaration():
    """The return-type check refuses a float64 declaration for a @cfunc that returns int64.

    The check reads the return type from the signature that the @cfunc stores, without a compilation.
    """
    cf = cfunc(int64(int64))(lambda x: x + 1)
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        _validate_declared_return(cf, (int64,), float64, flags={})


def test_validate_accepts_cres_correct_declaration():
    """The return-type check accepts the float64 declaration of a cres function that returns float64."""
    cf = cres(float64(float64))(lambda x: x + 1.0)
    _validate_declared_return(cf, (float64,), float64, flags={})


def test_validate_rejects_cres_wrong_declaration():
    """The return-type check refuses an int64 declaration for a cres function that returns float64.

    The check reads the return type from the signature of the function's compile result.
    """
    cf = cres(float64(float64))(lambda x: x + 1.0)
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        _validate_declared_return(cf, (float64,), int64, flags={})


def test_validate_accepts_dispatcher_correct_declaration():
    """The return-type check accepts a float64 declaration for a dispatcher that computes x * 1.5 for int64 input."""
    d = _njit(lambda x: x * 1.5)
    _validate_declared_return(d, (int64,), float64, flags={})


def test_validate_rejects_dispatcher_wrong_declaration():
    """The return-type check refuses an int64 declaration for a dispatcher that returns float64 for int64 input.

    The check compiles the dispatcher for the input types and reads the return type of that overload.
    """
    d = _njit(lambda x: x * 1.5)
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        _validate_declared_return(d, (int64,), int64, flags={})


def test_evaluate_honors_fixed_demotion_set():
    """_evaluate computes every value of a graph without a search, and runs the formulas in ``demoted`` in Python.

    CompiledKernel._ensure_store calls it to fill the store of a kernel built from declared types. The test puts c.b
    in ``demoted``.
    """
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0},
    ]}, ["e"])
    compiled = g.compile(["c.b"])
    a = g.registry["c"]["a"]
    b = g.registry["c"]["b"]
    ext = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    x = next(iter(ext))
    bindings = {a: _njit_t3(a.formula), b: _njit_t3(b.formula)}
    values = {x: 3.0}
    demoted = {b}  # c.b runs its Python function, not its dispatcher.
    _evaluate_fn(compiled.ordered_nodes, ext, values, bindings, {}, demoted)
    assert values[a] == 4.0 and values[b] == 8.0


def test_case_a_partition_fused_at_build():
    """A kernel in case A has is_declared True and a fused partition before its first call, and computes 8.0 for 3.0."""
    ck = compile_kernel(_graph_all_jittable(), "c.b")
    assert ck.partition is not None and ck.partition.mode == "fused"
    assert ck.is_declared is True
    assert ck.kernel(3.0) == (8.0,)


def test_case_a_recompute_after_fused_call():
    """recompute on a fused kernel built from declared types computes 10.0 for 4.0 after a first call with 3.0."""
    ck = compile_kernel(_graph_all_jittable(), "c.b")
    assert ck.kernel(3.0) == (8.0,)
    assert ck.recompute({"e": {"x": 4.0}}) == (10.0,)


def test_case_a_coercible_wrong_type_raises_at_build():
    """compile_kernel refuses an int64 declaration for x * 1.5 before any call, with the return-type check's error."""
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x * 1.5, "params": Params(type=int64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=int64))
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        compile_kernel(g, "c.a")


def test_case_a_passthrough_external_output_compiles():
    """A kernel in case A can return e.y, a requested external variable that no formula reads.

    The declared signature leaves e.y out, so numba compiles the kernel at its first call instead of at compile_kernel.
    """
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0, "params": Params(type=float64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    ck = compile_kernel(g, ["c.a", "e.y"])
    assert ck.is_declared is True
    assert ck.partition is not None and ck.partition.mode == "fused"
    assert ck.execute({"e": {"x": 3.0, "y": 9.0}}) == {"c.a": 4.0, "e.y": 9.0}


def test_undeclared_graph_stays_virgin_with_no_partition():
    """A kernel without declarations has is_declared False and no partition until its first call."""
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0},
    ]}, ["e"])
    ck = compile_kernel(g, "c.b")
    assert ck.is_declared is False
    assert ck.partition is None
    assert ck.kernel(3.0) == (8.0,)


def test_undeclared_external_only_graph_stays_undiscovered():
    """A request that names only an external variable is case C, the case of a graph without declarations.

    _classify finds no calculated variable to classify, so compile_kernel compiles nothing before the first call and
    partition stays None.
    """
    g = Graph({"c": []}, ["e"])
    ck = compile_kernel(g, "e.x")
    assert ck.is_declared is False
    assert ck.partition is None
    assert ck.execute({"e": {"x": 5.5}}) == {"e.x": 5.5}


def _declared_mix():
    """Return a graph in case B, in which c.b runs in Python between the compiled c.a and c.d.

    Every variable is declared float64. c.a computes e.x + 1.0, c.b computes c.a * 2.0, and c.d computes c.b + 1.0.
    """
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0, "params": Params(type=float64)},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0,
         "params": Params(jitable=False, type=float64)},
        {"name": "d", "inputs": {"b": "c"}, "formula": lambda b: b + 1.0, "params": Params(type=float64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    return g


def test_case_b_segmented_partition_and_result():
    """A kernel in case B has a segmented partition before its first call, with c.b among its Python formulas.

    It computes (3 + 1) * 2 + 1 = 9 for 3.0.
    """
    ck = compile_kernel(_declared_mix(), "c.d")
    assert ck.partition is not None and ck.partition.mode == "segmented"
    assert ck.is_declared is True
    assert "c.b" in ck.partition.python_nodes
    assert ck.kernel(3.0) == (9.0,)


def test_case_b_python_run_inputs_exclude_values_made_inside_the_run():
    """The report of a declared Python run lists as inputs only the values that the run does not compute itself.

    c.b and c.p form one Python run: c.b reads c.a, and c.p reads c.b and e.x. The run computes c.b itself, so its
    inputs are c.a and e.x. The kernel computes a = 4, b = 8, p = 11 and d = 12 for 3.0.
    """
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0, "params": Params(type=float64)},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0,
         "params": Params(jitable=False, type=float64)},
        {"name": "p", "inputs": {"b": "c", "x": "e"}, "formula": lambda b, x: b + x,
         "params": Params(jitable=False, type=float64)},
        {"name": "d", "inputs": {"p": "c"}, "formula": lambda p: p + 1.0, "params": Params(type=float64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    ck = compile_kernel(g, "c.d")
    py = [s for s in ck.partition.segments if s.kind == "python"]
    assert [(s.nodes, s.inputs, s.outputs) for s in py] == [
        (("c.b", "c.p"), ("c.a", "e.x"), ("c.b", "c.p")),
    ]
    assert ck.kernel(3.0) == (12.0,)


def test_case_b_no_probing_declared_python_honored():
    """A formula declared ``jitable=False`` runs in Python although numba could compile it.

    A kernel in case B takes its Python formulas from the declarations and never searches.
    """
    ck = compile_kernel(_declared_mix(), "c.d")
    assert "c.b" in ck.partition.python_nodes
    assert ck.kernel(3.0) == (9.0,)
    # The search, in _discover_and_run, and the first recompute, in _ensure_store, fill _store. After a kernel call
    # without a recompute, _store is still None, so no search ran. _run_segmented records the arguments of the first
    # call in _last_args, from which a later recompute fills the store.
    assert ck._store is None
    assert ck._last_args == (3.0,)


def test_case_b_recompute():
    """recompute on a kernel in case B computes (4 + 1) * 2 + 1 = 11 after a first call with 3.0."""
    ck = compile_kernel(_declared_mix(), "c.d")
    assert ck.kernel(3.0) == (9.0,)
    out = ck.recompute({"e": {"x": 4.0}})
    assert out == (11.0,)


def test_declared_return_validation_is_memoized(monkeypatch):
    """compile_kernel records each return-type check that passes, and does not run it again for the same key.

    The key holds the formula's fingerprint, the input types, the declared type and the jit options, so a second graph
    built by the same function needs no check.
    """
    import numbox.core.variable.compile_kernel as ck_mod
    ck_mod._validated_returns.clear()
    calls = []
    orig = ck_mod._validate_declared_return

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(ck_mod, "_validate_declared_return", spy)
    compile_kernel(_graph_all_jittable(), "c.b")
    first = len(calls)
    assert first > 0
    compile_kernel(_graph_all_jittable(), "c.b")
    assert len(calls) == first


def test_declared_type_variants_get_distinct_anchors():
    """A graph declared int64 and the same graph declared float64 compile from one source under two kernel names.

    The source holds no types, and the digest holds the declared types, so numba compiles and caches each version under
    a name of its own.
    """
    def make(t):
        g = Graph({"c": [{"name": "a", "inputs": {"x": "e"},
                          "formula": lambda x: x + 1, "params": Params(type=t)}]}, ["e"])
        g.external["e"].declare("x", Params(type=t))
        return compile_kernel(g, "c.a", cache=True)
    ck_i = make(int64)
    ck_f = make(float64)
    assert ck_i.kernel(3) == (4,)
    assert ck_f.kernel(3.0) == (4.0,)
    assert ck_i.source == ck_f.source
    assert ck_i._fused.__name__ != ck_f._fused.__name__


def test_inner_formula_bindings_stay_uncached():
    """numbox decorates a plain formula with @njit without the cache option.

    The formula therefore has no cache of its own. A formula with a cache of its own would make compile_kernel compile
    the kernel without a cache, with a warning.
    """
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0, "params": Params(type=float64)},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0, "params": Params(type=float64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    ck = compile_kernel(g, "c.b")
    bindings_by_var = ck._ctx[2]
    for binding in bindings_by_var.values():
        assert binding.targetoptions.get("cache") in (None, False)


def test_declared_array_recompute_accepts_c_contiguous():
    """recompute accepts a C-contiguous array for a variable declared ``float64[:]``, whose layout is 'A'.

    numba can convert the one layout to the other, so the check of the declared type passes.
    """
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0,
         "params": Params(type=float64[:])},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64[:]))
    ck = compile_kernel(g, "c.a")
    base = np.zeros(4)
    assert ck.kernel(base)[0].tolist() == [1, 1, 1, 1]
    out = ck.recompute({"e": {"x": np.ones(4)}})
    assert out[0].tolist() == [2, 2, 2, 2]


def test_declared_recompute_off_contract_type_raises_crisp():
    """recompute raises ValueError for a complex value of a variable declared float64.

    numba cannot convert complex128 to float64, and a kernel built from declared types checks each changed value.
    """
    ck = compile_kernel(_graph_all_jittable(), "c.b")
    ck.kernel(3.0)
    with pytest.raises(ValueError, match="declared type"):
        ck.recompute({"e": {"x": 3j}})


def test_declared_recompute_non_typeable_value_raises_crisp():
    """recompute raises the declared-type ValueError for a value that numba cannot type, such as object().

    The check catches the error of numba's typeof, so the caller sees one kind of error for every refused value.
    """
    ck = compile_kernel(_graph_all_jittable(), "c.b")
    ck.kernel(3.0)
    with pytest.raises(ValueError, match="declared type"):
        ck.recompute({"e": {"x": object()}})


def test_failed_first_fused_call_leaves_kernel_unseeded():
    """A first call that raises leaves a fused kernel built from declared types without starting values for recompute.

    _last_args stays None, so recompute raises RuntimeError instead of filling the store from a call that failed.
    """
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: 100 // x, "params": Params(type=int64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=int64))
    ck = compile_kernel(g, "c.a")
    with pytest.raises(ZeroDivisionError):
        ck.kernel(0)
    assert ck._last_args is None
    with pytest.raises(RuntimeError, match="requires a prior full call"):
        ck.recompute({"e": {"x": 5}})


def test_failed_first_segmented_call_leaves_kernel_unseeded():
    """A first call that raises leaves a segmented kernel built from declared types without starting values as well.

    _last_args stays None, and recompute raises RuntimeError.
    """
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: 100 // x, "params": Params(type=int64)},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a + 1,
         "params": Params(jitable=False, type=int64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=int64))
    ck = compile_kernel(g, "c.b")
    with pytest.raises(ZeroDivisionError):
        ck.kernel(0)
    assert ck._last_args is None
    with pytest.raises(RuntimeError, match="requires a prior full call"):
        ck.recompute({"e": {"x": 5}})


def test_case_a_fused_recompute_still_works():
    """recompute on a fused kernel built from declared types computes 10.0 for 4.0 after a first call with 3.0.

    _ensure_store fills the store with _evaluate, which uses the declared set of Python formulas instead of a search.
    """
    ck = compile_kernel(_graph_all_jittable(), "c.b")
    assert ck.kernel(3.0) == (8.0,)
    assert ck.recompute({"e": {"x": 4.0}}) == (10.0,)


def _case_c_with_declared_node():
    """Return a graph in case C: c.a is declared jittable with a type, and c.b has no declaration.

    compile_kernel then ignores every declaration, and the kernel's is_declared is False.
    """
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0, "params": Params(type=float64)},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    return g


def test_case_c_declared_node_skips_contract_check():
    """recompute on a kernel in case C does not check a changed value against its declaration.

    An int for e.x, which is declared float64, raises no ValueError: numba compiles the recompute plan for the int.
    """
    ck = compile_kernel(_case_c_with_declared_node(), "c.b")
    assert ck.is_declared is False
    assert ck.kernel(3.0) == (8.0,)
    out = ck.recompute({"e": {"x": 5}})
    assert out == (12.0,)


def test_eager_kernel_no_silent_rediscover_off_contract():
    """recompute raises ValueError for a complex value and does not change the kernel's set of Python formulas.

    A kernel built from declared types never searches, so the declarations alone define its Python formulas.
    """
    ck = compile_kernel(_graph_all_jittable(), "c.b")
    ck.kernel(3.0)
    demoted_before = dict(ck._demoted)
    with pytest.raises(ValueError, match="declared type"):
        ck.recompute({"e": {"x": 3j}})
    assert ck._demoted == demoted_before


def test_shared_dispatcher_multi_overload_validation():
    """compile_kernel accepts three declared variables that share one dispatcher with an int64 and a float64 overload.

    compile_kernel skips the check of c.d, because c.d has the fingerprint, types and options of c.a, whose check has
    passed. The test therefore does not detect a check that reads the overload compiled last instead of the overload
    for the variable's own input types.
    """
    shared = _njit(lambda x: x + 1)
    g = Graph({"c": [
        {"name": "a", "inputs": {"p": "e"}, "formula": shared, "params": Params(type=int64)},
        {"name": "b", "inputs": {"q": "e"}, "formula": shared, "params": Params(type=float64)},
        {"name": "d", "inputs": {"r": "e"}, "formula": shared, "params": Params(type=int64)},
    ]}, ["e"])
    g.external["e"].declare("p", Params(type=int64))
    g.external["e"].declare("q", Params(type=float64))
    g.external["e"].declare("r", Params(type=int64))
    ck = compile_kernel(g, ["c.a", "c.b", "c.d"])
    assert ck.is_declared is True
    assert ck.execute({"e": {"p": 3, "q": 2.0, "r": 5}}) == {"c.a": 4, "c.b": 3.0, "c.d": 6}


def test_declared_exotic_node_end_to_end():
    """A declared cres formula compiles into a fused kernel at compile_kernel and computes 4.0 for 3.0.

    The return-type check reads the type from the cres function's own signature.
    """
    cf = cres(float64(float64))(lambda x: x + 1.0)
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": cf, "params": Params(type=float64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    ck = compile_kernel(g, "c.a")
    assert ck.is_declared is True
    assert ck.partition is not None and ck.partition.mode == "fused"
    assert ck.kernel(3.0) == (4.0,)


def test_declared_exotic_node_wrong_declaration_raises_at_build():
    """compile_kernel refuses an int64 declaration for a cres formula that returns float64, before any call."""
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"},
         "formula": cres(float64(float64))(lambda x: x + 1.0), "params": Params(type=int64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        compile_kernel(g, "c.a")


def test_declared_segmented_throughput_off_contract_reraises():
    """A segmented kernel built from declared types raises numba's error for a call that a segment cannot compile for.

    Here the call passes a string to the first segment, which computes x + 1.0. The kernel does not search again, so
    the declarations still define its set of Python formulas.
    """
    ck = compile_kernel(_declared_mix(), "c.d")
    assert ck.partition.mode == "segmented"
    assert ck.kernel(3.0) == (9.0,)
    demoted_before = dict(ck._demoted)
    with pytest.raises(NumbaError):
        ck.kernel("not a number")
    assert ck._demoted == demoted_before


def test_declared_float32_graph_end_to_end():
    """A graph declared float32 compiles into a fused kernel, because its formula x + float32(1.0) returns float32."""
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + float32(1.0),
         "params": Params(type=float32)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float32))
    ck = compile_kernel(g, "c.a")
    assert ck.is_declared is True
    assert ck.kernel(np.float32(2.0)) == (3.0,)


def test_declared_int32_over_widening_body_rejected_at_build():
    """compile_kernel refuses an int32 declaration for x + 1, because numba types int32 + 1 as int64."""
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1, "params": Params(type=int32)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=int32))
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        compile_kernel(g, "c.a")
