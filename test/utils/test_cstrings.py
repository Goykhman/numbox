from contextlib import ExitStack
from ctypes import c_char

from numba import njit
import pytest

from numbox.utils.cstrings import c_string
from test.auxiliary_utils import collect_and_run_tests, str_from_p_as_int


def test_ascii_round_trip():
    with c_string("hello") as p:
        assert p != 0
        assert str_from_p_as_int(p) == "hello"


def test_utf8_round_trip():
    s = "café — 你好 — 🚀"
    with c_string(s) as p:
        assert p != 0
        # Read back the UTF-8 bytes through to NUL
        roundtrip = str_from_p_as_int(p)
        assert roundtrip == s


def test_empty_string_returns_pointer_to_nul():
    with c_string("") as p:
        assert p != 0
        first_byte = c_char.from_address(p).value
        assert first_byte == b"\x00"


def test_nul_terminator_present():
    s = "abc"
    with c_string(s) as p:
        # Index 3 is the position after "abc" — must be NUL
        assert c_char.from_address(p + len(s.encode("utf-8"))).value == b"\x00"


def test_distinct_strings_get_distinct_pointers():
    with ExitStack() as stack:
        a_p = stack.enter_context(c_string("first"))
        b_p = stack.enter_context(c_string("second"))
        assert a_p != b_p
        assert str_from_p_as_int(a_p) == "first"
        assert str_from_p_as_int(b_p) == "second"


def test_context_manager_in_njit_raises():
    """Numba doesn't support arbitrary context managers — guard against
    a future numba version silently accepting and miscompiling. If this
    test breaks because numba started supporting it, revisit the
    cstrings module docstring's JIT caveat.

    Catches the broad ``Exception`` deliberately. The exception class
    numba raises for this construct varies across the support matrix:

    - numba 0.60.x raises ``UnsupportedError`` (inherits from ``NumbaError``)
    - numba 0.61.0+ raises ``UnsupportedBytecodeError`` (inherits from
      ``Exception`` directly, NOT ``NumbaError`` — see numba commit
      ``57a53878ca``)

    Since they share no common numba-specific base, ``Exception`` is
    the only ancestor that catches both. The test's intent is to
    assert "compilation fails" — any exception means numba rejected
    the construct; the regression we'd catch (numba silently accepting
    and miscompiling) wouldn't raise at all.
    """
    @njit
    def kernel():
        with c_string("x") as p:
            return p

    with pytest.raises(Exception):
        kernel()


if __name__ == "__main__":
    collect_and_run_tests(__name__)
