"""NUL-terminated UTF-8 C strings, created from Python strings, for numbox's bindings.

numbox's bindings, such as those in ``numbox.core.bindings.libc`` and ``numbox.core.bindings.sqlite``, take every
text argument as an ``intp`` address of a NUL-terminated C string. :func:`c_string` copies the UTF-8 encoding of a
``str`` into a new C buffer and yields the buffer's address for the duration of a ``with`` block.
:func:`numbox.utils.lowlevel.get_unicode_data_p` returns the address of a string's own data without a copy. CPython
stores one, two or four bytes per character there, so those bytes are the text's UTF-8 only for ASCII text.

``c_string`` works only in Python: numba raises ``UnsupportedBytecodeError`` for a ``with`` statement that uses it as
the context manager, and numba cannot use the ctypes buffer either. For jitted code, encode the text in Python, add a
NUL byte, store the bytes in a NumPy ``uint8`` array, and take the array's address in the jitted code with
:func:`numbox.utils.lowlevel.array_data_p`.
"""
from contextlib import contextmanager
from ctypes import addressof, create_string_buffer


@contextmanager
def c_string(s):
    """Yield the ``intp`` address of a new NUL-terminated C string that holds the UTF-8 encoding of ``s``.

    The C buffer exists until the ``with`` block ends. The generator then drops its reference to the buffer and ctypes
    frees it, so the address must not be used after the block. ``ValueError`` is raised if ``s`` contains a NUL
    character, because the C string would end there; CPython's ``sqlite3.connect`` refuses such a file name in the same
    way. Several strings can be open at once, in one ``with`` statement or, when their number is known only at run
    time, through a ``contextlib.ExitStack``::

        from contextlib import ExitStack

        from numbox.core.bindings.libc import strlen
        from numbox.utils.cstrings import c_string

        with c_string("pear") as p:
            print(strlen(p))   # 4

        with ExitStack() as stack:
            pointers = [stack.enter_context(c_string(word)) for word in ["pear", "apple"]]
            print([strlen(p) for p in pointers])   # [4, 5]

    ``c_string`` works only in Python; the module docstring describes how to pass a string to jitted code.
    """
    encoded = s.encode("utf-8")
    if b"\x00" in encoded:
        raise ValueError(
            "c_string: input contains an embedded NUL byte; would truncate "
            "the C string at the first NUL"
        )
    buf = create_string_buffer(encoded)
    yield addressof(buf)
