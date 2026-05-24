"""Python-side helpers for passing C strings into the bindings layer.

The bindings layer (``numbox.core.bindings._sqlite_*``, ``_c``, etc.)
takes ``intp`` pointers for every text argument. Producing a valid
NUL-terminated UTF-8 C string from a Python ``str`` is non-trivial:
``get_unicode_data_p`` in ``lowlevel.py`` returns a pointer to the
Python string's internal data payload, which CPython stores as
UCS-1/2/4 depending on contents -- only safe for ASCII inputs.

This module's :func:`c_string` allocates a real C buffer with the
UTF-8 encoding of the input and yields the pointer with safe
lifetime management via the ``with`` statement.

**Python-only.** ``c_string`` is a context manager; numba does not
support arbitrary context managers inside ``@njit`` (raises
``UnsupportedBytecodeError``), and ``ctypes`` objects can't be
manipulated under JIT anyway. For ``@njit`` callers that need a C
string, pre-allocate a numpy ``uint8`` buffer containing the UTF-8
bytes + a trailing NUL outside the JIT scope, then pass
``array_data_p(buf)`` from ``numbox.utils.lowlevel`` into the JIT
kernel.
"""
from contextlib import contextmanager
from ctypes import c_char_p, c_void_p


@contextmanager
def c_string(s):
    """Yield an ``intp`` pointer to a freshly-allocated NUL-terminated
    UTF-8 C string for ``s``.

    Usage::

        from numbox.utils.cstrings import c_string
        from numbox.core.bindings import sqlite3_exec

        with c_string("CREATE TABLE t(x INTEGER)") as sql_p:
            sqlite3_exec(db_p, sql_p, 0, 0, 0)
        # buffer freed here

    The underlying ``ctypes`` buffer lives for the duration of the
    ``with`` block. Once the block exits, the Python reference is
    dropped and ctypes frees the memory; the pointer is then dangling
    and must not be used.

    For concurrent multi-string needs, nest ``with`` statements or use
    ``contextlib.ExitStack``::

        from contextlib import ExitStack
        with ExitStack() as stack:
            a_p = stack.enter_context(c_string("a"))
            b_p = stack.enter_context(c_string("b"))
            # both pointers valid here

    Python-only -- not callable inside ``@njit``. See module docstring
    for the JIT alternative.
    """
    buf = c_char_p(s.encode("utf-8"))
    yield c_void_p.from_buffer(buf).value
