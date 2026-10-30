"""Tests of the libc bindings in numbox.core.bindings.libc and of library loading in numbox.core.bindings.utils.

The tests call the bindings for random numbers, strings, memory, files and the environment, from Python and from
cached jitted functions, and check the results against known values. The loading tests open a library by its path
and by its short name. Two tests check the Windows rules of ``_resolve_lib_path`` on any host, with the platform
patched, and the last test checks in a subprocess that importing libm does not import the SQLite, stdio or fmtio
modules. The other binding modules have test files of their own, such as ``test_math.py``, ``test_fmtio.py`` and
``test_abi.py``.
"""
import errno
import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from ctypes import c_char_p, c_void_p
from numba import njit
from numbox.core.bindings.libc import (
    fclose, fopen, fread, fwrite, getenv, memchr, memcmp, memcpy, memmove, memset,
    rand, srand, strchr, strcmp, strerror, strlen, strncmp, strncpy, strrchr, strstr,
)
from numbox.core.bindings.utils import platform_
from numbox.utils.lowlevel import array_data_p, get_unicode_data_p, get_str_from_p_as_int
from test.auxiliary_utils import collect_and_run_tests


@njit(cache=True)
def _write_and_read(path_str, mode_w, mode_r, payload_arr, read_back):
    """Write payload_arr to the file path_str, read the file back into read_back, and return the two counts.

    A count of -1 marks the step whose ``fopen`` failed.
    """
    wpath = get_unicode_data_p(path_str)
    wmode = get_unicode_data_p(mode_w)
    rmode = get_unicode_data_p(mode_r)
    wfp = fopen(wpath, wmode)
    if wfp == 0:
        return -1, 0
    wbuf = array_data_p(payload_arr)
    nw = fwrite(wbuf, 1, payload_arr.size, wfp)
    fclose(wfp)
    rfp = fopen(wpath, rmode)
    if rfp == 0:
        return nw, -1
    rbuf = array_data_p(read_back)
    nr = fread(rbuf, 1, read_back.size, rfp)
    fclose(rfp)
    return nw, nr


def test_c():
    """Check srand, rand and strlen called from Python, not from jitted code.

    rand returns positive values after srand(2), and strlen counts the bytes of a C string.
    """
    srand(2)
    r1 = rand()
    r2 = rand()
    assert r1 > 0
    assert r2 > 0

    s_ = "another random string"
    s = c_char_p(s_.encode())
    s_p = c_void_p.from_buffer(s).value
    assert strlen(s_p) == len(s_)


def test_load_lib_path_returns_handle_with_known_symbol():
    """Check that ``load_lib_path`` opens a library from its path and returns a handle that exports a known symbol.

    On Linux and macOS, the test passes the result of ``find_library("m")``, a soname such as ``libm.so.6`` on Linux,
    and looks for ``cos``. On Windows, ``find_msvcrt()`` returns None on every CPython built with Visual Studio 2015
    or later, so the test has no path of the C runtime to pass. It passes the ``sqlite3.dll`` bundled with Python
    instead and looks for ``sqlite3_libversion``. Every Windows Python whose ``sqlite3`` module works has that DLL,
    because ``_sqlite3.pyd`` links to it, so the test runs the ``winmode=0`` branch of ``load_lib_path`` there.
    """
    from numbox.core.bindings.utils import _windows_bundled_dll_path, load_lib_path

    if platform_ == "Windows":
        lib_path, known_symbol = _windows_bundled_dll_path("sqlite3"), "sqlite3_libversion"
    else:
        from ctypes.util import find_library
        lib_path, known_symbol = find_library("m"), "cos"
    if lib_path is None:
        pytest.skip("No library discoverable by path on this platform")
    lib = load_lib_path(lib_path)
    assert hasattr(lib, known_symbol)


def test_load_lib_returns_queryable_handle():
    from numbox.core.bindings.utils import load_lib
    handle = load_lib("c")
    assert handle is not None
    # Every supported C library exports strlen.
    assert hasattr(handle, "strlen")
    # hasattr is False for a name that the library does not export.
    assert not hasattr(handle, "definitely_not_a_real_symbol_xyzzy")


def test_c_stdio(tmp_path):
    """Check that fopen, fwrite, fread and fclose write 18 bytes, a NUL among them, and read the same bytes back."""
    path = tmp_path / "rt.bin"
    payload = b"hello-from-njit\x00\x01\x02"
    payload_arr = np.frombuffer(payload, dtype=np.uint8).copy()
    read_back = np.zeros(len(payload), dtype=np.uint8)
    nw, nr = _write_and_read(str(path), "wb", "rb", payload_arr, read_back)
    assert nw == len(payload)
    assert nr == len(payload)
    assert bytes(read_back) == payload


@njit(cache=True)
def _strings_compare_search():
    a = get_unicode_data_p("hello")
    b = get_unicode_data_p("hello")
    c = get_unicode_data_p("world")
    eq = strcmp(a, b)
    ne = strcmp(a, c)
    # strncmp compares at most n bytes: "hello" and "help!" share their first 3 bytes and differ in the fourth.
    d_hello = get_unicode_data_p("hello")
    d_help = get_unicode_data_p("help!")
    n_match = strncmp(d_hello, d_help, 3)
    n_differ = strncmp(d_hello, d_help, 4)
    h = get_unicode_data_p("hello world")
    ord_l = np.int32(108)
    first_l = strchr(h, ord_l)
    last_l = strrchr(h, ord_l)
    substr = strstr(h, get_unicode_data_p("world"))
    return eq, ne, n_match, n_differ, first_l - h, last_l - h, substr - h


@njit(cache=True)
def _strings_copy(dst):
    src = get_unicode_data_p("abcdef")
    dst_p = array_data_p(dst)
    strncpy(dst_p, src, 6)
    return dst_p


def test_c_strings():
    eq, ne, n_match, n_differ, off_first, off_last, off_sub = _strings_compare_search()
    assert eq == 0
    assert ne != 0
    # strncmp("hello", "help!", 3) is 0, because the first 3 bytes, "hel", are equal.
    assert n_match == 0, n_match
    # strncmp("hello", "help!", 4) is not 0, because byte 3 is 'l' in one string and 'p' in the other.
    assert n_differ != 0, n_differ
    assert off_first == 2
    assert off_last == 9
    assert off_sub == 6

    # strncpy writes no NUL when the source is n bytes long, so the zeroed buffer supplies the NUL at which
    # get_str_from_p_as_int stops.
    dst = np.zeros(8, dtype=np.uint8)
    dst_p = _strings_copy(dst)
    assert bytes(dst[:6]) == b"abcdef"
    assert get_str_from_p_as_int(dst_p) == "abcdef"


@njit(cache=True)
def _strerror_lookup_enoent():
    return strerror(np.int32(errno.ENOENT))


def test_c_strerror():
    p = _strerror_lookup_enoent()
    assert p != 0
    assert len(get_str_from_p_as_int(p)) > 0


@njit(cache=True)
def _mem_do_copy(dst, src):
    return memcpy(array_data_p(dst), array_data_p(src), src.nbytes)


@njit(cache=True)
def _mem_do_move(arr):
    p = array_data_p(arr)
    return memmove(p + 2, p, 5)


@njit(cache=True)
def _mem_do_set(arr):
    return memset(array_data_p(arr), np.int32(0x7F), arr.nbytes)


@njit(cache=True)
def _mem_do_cmp(a, b):
    return memcmp(array_data_p(a), array_data_p(b), a.nbytes)


@njit(cache=True)
def _mem_do_chr(h):
    p = array_data_p(h)
    return memchr(p, np.int32(3), h.nbytes) - p


def test_c_memory():
    src = np.arange(10, dtype=np.uint8)
    dst = np.zeros(10, dtype=np.uint8)
    _mem_do_copy(dst, src)
    assert (dst == src).all()

    overlap = np.arange(10, dtype=np.uint8).copy()
    _mem_do_move(overlap)
    assert (overlap[2:7] == np.arange(5, dtype=np.uint8)).all()

    fill = np.zeros(8, dtype=np.uint8)
    _mem_do_set(fill)
    assert (fill == 0x7F).all()

    a = np.array([1, 2, 3, 4], dtype=np.uint8)
    b = np.array([1, 2, 3, 5], dtype=np.uint8)
    assert _mem_do_cmp(a, b) < 0
    assert _mem_do_cmp(b, a) > 0
    assert _mem_do_cmp(a, a) == 0

    haystack = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
    assert _mem_do_chr(haystack) == 3


@njit(cache=True)
def _env_lookup(name):
    return getenv(get_unicode_data_p(name))


@pytest.fixture
def _getenv_test_var(monkeypatch):
    """Set an environment variable that the ``getenv`` binding can read, and remove it at teardown.

    On Linux and macOS, ``monkeypatch.setenv`` is enough. On Windows, ``load_lib("c")`` loads ``msvcrt.dll``, the
    legacy C runtime that ``ctypes.cdll.msvcrt`` returns, while CPython links the Universal C Runtime (UCRT) and sets
    ``os.environ`` through it. Each C runtime has its own environment table, so the binding does not see a variable
    that Python sets. The fixture therefore sets the variable with ``_putenv_s`` of the same ``msvcrt.dll``
    that the binding calls.
    """
    var_name = "NUMBOX_TEST_GETENV_VAR_7d4f1c"
    var_value = "numbox-getenv-roundtrip-sentinel"
    if platform_ == "Windows":
        import ctypes
        msvcrt = ctypes.cdll.msvcrt
        msvcrt._putenv_s(var_name.encode(), var_value.encode())
        try:
            yield var_name, var_value
        finally:
            # _putenv_s removes the variable when the value is empty, as Microsoft documents.
            msvcrt._putenv_s(var_name.encode(), b"")
    else:
        monkeypatch.setenv(var_name, var_value)
        yield var_name, var_value


def test_c_env(_getenv_test_var):
    """Check that ``getenv`` returns the address of a set variable's value, and 0 for a variable that is not set.

    The test reads the string at the returned address and compares it with the value, because a wrong address that
    is not 0 would pass a check of the address alone. It sets a variable of its own through the ``_getenv_test_var``
    fixture, instead of reading ``PATH``, which a CI runner can clear.
    """
    var_name, var_value = _getenv_test_var
    found_p = _env_lookup(var_name)
    assert found_p != 0, (
        f"getenv returned 0 for {var_name} (which was just set to {var_value!r})"
    )
    assert _env_lookup("NUMBOX_NONEXISTENT_XYZZY") == 0, (
        "getenv returned non-zero for a deliberately-unset variable"
    )
    got = get_str_from_p_as_int(found_p)
    assert got == var_value, (
        f"getenv({var_name!r}) returned a pointer to {got!r}; "
        f"expected {var_value!r}"
    )


def test_resolve_lib_path_windows_sqlite3_never_falls_back(monkeypatch):
    """Check that ``_resolve_lib_path("sqlite3")`` on Windows returns only the DLL bundled with Python.

    The function never calls ``find_library`` for ``"sqlite3"``, so it returns None when no bundled DLL exists. A
    ``sqlite3.dll`` on the ``PATH`` can be a copy that another tool installs for its own use: AWS CLI v2 installs one
    that writes through a null pointer inside ``sqlite3_open`` when a process other than ``aws.exe`` calls it. The
    test patches the platform, ``find_library`` and the lookup of the bundled DLL, so it runs on any host.
    """
    import ctypes.util as _ctypes_util
    from numbox.core.bindings import utils

    monkeypatch.setattr(utils, "platform_", "Windows")
    # ctypes.util defines find_msvcrt only on Windows. The test adds it, so that the Windows branch of
    # _resolve_lib_path can import it on any host.
    monkeypatch.setattr(_ctypes_util, "find_msvcrt", lambda: "C:\\msvcrt.dll", raising=False)
    find_library_calls = []
    monkeypatch.setattr(
        utils, "find_library",
        lambda n: find_library_calls.append(n) or f"C:\\PATH\\{n}.dll")

    monkeypatch.setattr(utils, "_windows_bundled_dll_path", lambda n: "C:\\Py\\DLLs\\sqlite3.dll")
    assert utils._resolve_lib_path("sqlite3") == "C:\\Py\\DLLs\\sqlite3.dll"
    assert find_library_calls == []

    monkeypatch.setattr(utils, "_windows_bundled_dll_path", lambda n: None)
    assert utils._resolve_lib_path("sqlite3") is None
    assert find_library_calls == []


def test_resolve_lib_path_windows_nonsqlite_prefers_bundled(monkeypatch):
    """Check that ``_resolve_lib_path`` on Windows prefers the DLL bundled with Python for a name other than sqlite3.

    The function returns the bundled DLL when one exists, and the result of ``find_library`` only when none does. The
    test patches the platform, ``find_library`` and the lookup of the bundled DLL, so it runs on any host.
    """
    import ctypes.util as _ctypes_util
    from numbox.core.bindings import utils

    monkeypatch.setattr(utils, "platform_", "Windows")
    monkeypatch.setattr(_ctypes_util, "find_msvcrt", lambda: "C:\\msvcrt.dll", raising=False)
    monkeypatch.setattr(utils, "find_library", lambda n: f"C:\\PATH\\{n}.dll")

    monkeypatch.setattr(utils, "_windows_bundled_dll_path", lambda n: "C:\\Py\\DLLs\\foo.dll")
    assert utils._resolve_lib_path("foo") == "C:\\Py\\DLLs\\foo.dll"

    monkeypatch.setattr(utils, "_windows_bundled_dll_path", lambda n: None)
    assert utils._resolve_lib_path("foo") == "C:\\PATH\\foo.dll"


def test_import_one_binding_stays_lazy(tmp_path):
    """Check that importing ``numbox.core.bindings.libm`` imports neither the SQLite subpackage nor stdio or fmtio.

    ``numbox/core/bindings/__init__.py`` imports none of its modules, so numbox never compiles a binding module that a
    program does not import. If ``__init__.py`` re-exported the bindings, or ``call.py`` or ``utils.py`` imported a
    binding module at module level, importing libm would compile the SQLite bindings as well, and no other test
    would fail. The test imports libm in a subprocess, because pytest has already imported every binding in its own
    process.
    """
    probe = tmp_path / "lazy_probe.py"
    probe.write_text(textwrap.dedent('''
        import json
        import sys
        import numbox.core.bindings.libm  # noqa: F401
        loaded = sorted(m for m in sys.modules if m.startswith("numbox.core.bindings"))
        print(json.dumps(loaded))
    '''), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    loaded = json.loads(r.stdout.splitlines()[-1])
    assert "numbox.core.bindings.libm" in loaded, loaded
    forbidden = [
        m for m in loaded
        if m == "numbox.core.bindings.sqlite"
        or m.startswith("numbox.core.bindings.sqlite.")
        or m in ("numbox.core.bindings.stdio", "numbox.core.bindings.fmtio")
    ]
    assert not forbidden, f"importing libm eagerly loaded heavy modules: {forbidden}"


if __name__ == "__main__":
    collect_and_run_tests(__name__)
