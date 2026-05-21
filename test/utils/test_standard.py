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


if __name__ == '__main__':
    collect_and_run_tests(__name__)
