"""Tests for ``numbox.utils.standard``, which formats a function's parameters for the source that numbox generates.

``make_structref`` and ``@proxy`` write the strings of ``make_params_strings`` into source that they run with
``exec``. The tests check the two strings for plain and defaulted parameters, and the ``ValueError`` for ``*args`` and
``**kwargs``, which the generated code would bind as plain parameters. They check that the function drops the markers
``*`` and ``/``. Other tests run the generated ``def`` for string and ``None`` defaults and for ``nan`` and ``inf``,
whose ``repr`` makes the ``def`` raise ``NameError``.
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
    """``make_params_strings`` drops the keyword-only marker ``*``, so every parameter becomes positional-or-keyword.

    ``@proxy`` relies on this for a function such as ``def f(x, *, y=default)`` with an ``Omitted`` signature for
    ``y``. ``test_2`` in ``test/core/test_proxy.py`` calls such a function with ``y`` passed by position.
    """
    def f(a, *, b):
        pass
    assert make_params_strings(f) == ("a, b", "a, b")


def test_make_params_strings_flattens_positional_only():
    """``make_params_strings`` drops the positional-only marker ``/``, as it drops ``*``.

    Dropping a marker lets a caller pass a parameter in one more way, and it does not change what the generated code
    computes.
    """
    def f(a, b, /, c):
        pass
    assert make_params_strings(f) == ("a, b, c", "a, b, c")


def test_make_params_strings_rejects_combined_var_args_and_kwargs():
    def f(a, b=10, *args, **kwargs):
        pass
    with pytest.raises(ValueError):
        make_params_strings(f)


def test_make_params_strings_string_default_uses_repr():
    """A string default is written as its ``repr``, so the generated ``def`` returns the same string.

    The string's ``str`` has no quotes, so the generated code would read it as a name and raise ``NameError``.
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
    """A ``None`` default, whose ``str`` and ``repr`` are both ``None``, is written correctly as well."""
    def f(x=None):
        pass
    sig, _ = make_params_strings(f)
    assert sig == "x=None"
    namespace = {}
    exec(f"def wrapper({sig}): return x", namespace)
    assert namespace["wrapper"]() is None


@pytest.mark.parametrize("bad_float", [float('nan'), float('inf'), float('-inf')])
def test_make_params_strings_nan_inf_floats_documented_no_roundtrip(bad_float):
    """The defaults ``nan``, ``inf`` and ``-inf`` give a ``def`` that raises ``NameError`` when it runs.

    Their ``repr`` refers to the name ``nan`` or ``inf``, which the generated code does not define, as the docstring of
    ``make_params_strings`` states. The test fails if the function writes these defaults in another way, so that the
    docstring changes with the function.
    """
    def f(x=bad_float):
        pass
    sig, _ = make_params_strings(f)
    namespace = {}
    with pytest.raises(NameError):
        exec(f"def wrapper({sig}): return x", namespace)


if __name__ == '__main__':
    collect_and_run_tests(__name__)
