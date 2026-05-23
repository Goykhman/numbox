"""Tests for ``numbox.utils.standard`` source-string helpers.

``make_params_strings`` is consumed by ``make_structref``'s method-binding
codegen and by the ``@proxy`` decorator. Both produce Python source that
gets ``exec()``'d; the helper must refuse parameter kinds it cannot
faithfully reproduce, otherwise the generated wrapper silently binds
arguments differently from the user's intent.
"""
import pytest

from numbox.utils.standard import make_params_strings
from test.auxiliary_utils import collect_and_run_tests


def test_make_params_strings_positional_or_keyword():
    def f(a, b, c):
        pass
    assert make_params_strings(f) == ("a, b, c", "a, b, c")


def test_make_params_strings_with_default():
    def f(a, b, c=1):
        pass
    assert make_params_strings(f) == ("a, b, c=1", "a, b, c")


def test_make_params_strings_rejects_var_positional():
    def f(a, *args):
        pass
    with pytest.raises(ValueError, match=r"\*args"):
        make_params_strings(f)


def test_make_params_strings_rejects_var_keyword():
    def f(a, **kwargs):
        pass
    with pytest.raises(ValueError, match=r"\*\*kwargs"):
        make_params_strings(f)


def test_make_params_strings_flattens_keyword_only():
    """Keyword-only `*,` separator is silently dropped — this is the
    behavior `@proxy` relies on for `Omitted`-style overloads written
    as `def f(x, *, y=default)`. See test_proxy.py::test_2."""
    def f(a, *, b):
        pass
    assert make_params_strings(f) == ("a, b", "a, b")


def test_make_params_strings_flattens_positional_only():
    """Positional-only `/` separator is silently dropped to match the
    kw-only handling — both are non-Critical for the codegen since they
    only widen the calling-convention surface area, they don't
    miscompile."""
    def f(a, b, /, c):
        pass
    assert make_params_strings(f) == ("a, b, c", "a, b, c")


def test_make_params_strings_rejects_combined_var_args_and_kwargs():
    def f(a, b=10, *args, **kwargs):
        pass
    with pytest.raises(ValueError):
        make_params_strings(f)


def test_make_params_strings_string_default_uses_repr():
    """String defaults must round-trip through the generated source.

    ``str("hello") == "hello"`` (no quotes) which is invalid as a
    default-value expression — it would parse as a bare identifier
    and NameError at exec time. ``repr("hello") == "'hello'"`` round-
    trips correctly.
    """
    def f(x="hello"):
        pass
    sig, call = make_params_strings(f)
    assert sig == "x='hello'"
    assert call == "x"
    namespace = {}
    exec(f"def wrapper({sig}): return x", namespace)
    assert namespace["wrapper"]() == "hello"


def test_make_params_strings_none_default_round_trips():
    """None default — both str and repr produce ``"None"``; pin the
    repr fix didn't break the case that ``str`` handled correctly."""
    def f(x=None):
        pass
    sig, _ = make_params_strings(f)
    assert sig == "x=None"
    namespace = {}
    exec(f"def wrapper({sig}): return x", namespace)
    assert namespace["wrapper"]() is None


@pytest.mark.parametrize("bad_float", [float('nan'), float('inf'), float('-inf')])
def test_make_params_strings_nan_inf_floats_documented_no_roundtrip(bad_float):
    """nan/inf floats render via repr() to bare identifiers (``nan`` /
    ``inf`` / ``-inf``) which are not valid Python expressions and raise
    NameError at exec time. The docstring documents this as a known
    limitation. Pin the behavior so the gotcha stays visible if the
    codepath ever changes."""
    def f(x=bad_float):
        pass
    sig, _ = make_params_strings(f)
    namespace = {}
    with pytest.raises(NameError):
        exec(f"def wrapper({sig}): return x", namespace)


if __name__ == '__main__':
    collect_and_run_tests(__name__)
