"""Tests for ``numbox.utils.cstrings``, the NUL-terminated UTF-8 C strings that ``c_string`` creates.

The tests read each string back through its address, for ASCII text, for non-ASCII text and for an empty string. They
check the NUL terminator, that two strings open at once have different addresses, and that ``c_string`` raises
``ValueError`` for a string that contains a NUL character. One test checks that numba refuses ``c_string`` as a context
manager in jitted code.
"""
from contextlib import ExitStack
from ctypes import c_char

from numba import njit
from numba.core.errors import UnsupportedBytecodeError
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
        # str_from_p_as_int reads the bytes up to the NUL and decodes them as UTF-8.
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
        # The byte after the three bytes of "abc" must be the NUL terminator.
        assert c_char.from_address(p + len(s.encode("utf-8"))).value == b"\x00"


def test_embedded_nul_rejected():
    """``c_string`` raises ``ValueError`` for a string that contains a NUL character.

    The C string would end at that character, so a C function would receive a shorter string without any error.
    """
    with pytest.raises(ValueError, match="embedded NUL"):
        with c_string("foo\x00bar"):
            pass


def test_distinct_strings_get_distinct_pointers():
    with ExitStack() as stack:
        a_p = stack.enter_context(c_string("first"))
        b_p = stack.enter_context(c_string("second"))
        assert a_p != b_p
        assert str_from_p_as_int(a_p) == "first"
        assert str_from_p_as_int(b_p) == "second"


def test_context_manager_in_njit_raises():
    """numba raises ``UnsupportedBytecodeError`` for a jitted function that uses ``c_string`` as its context manager.

    The docstrings of ``numbox.utils.cstrings`` say that ``c_string`` works only in Python. If a numba release compiles
    the function, the test fails, and those docstrings need a review.
    """
    @njit
    def kernel():
        with c_string("x") as p:
            return p

    with pytest.raises(UnsupportedBytecodeError):
        kernel()


if __name__ == "__main__":
    collect_and_run_tests(__name__)
