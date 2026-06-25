import errno

import numpy as np
import pytest
from ctypes import c_char_p, c_void_p
from numba import njit
from numbox.core.bindings import *
from numbox.core.bindings.utils import platform_
from numbox.utils.lowlevel import array_data_p, get_unicode_data_p, get_str_from_p_as_int
from test.auxiliary_utils import collect_and_run_tests


@njit(cache=True)
def _write_and_read(path_str, mode_w, mode_r, payload_arr, read_back):
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
    from numbox.core.bindings.utils import load_lib_path

    if platform_ == "Windows":
        from ctypes.util import find_msvcrt
        lib_path = find_msvcrt()
    else:
        from ctypes.util import find_library
        lib_path = find_library("m")
    if lib_path is None:
        pytest.skip("No suitable math/C runtime library discoverable")
    lib = load_lib_path(lib_path)
    assert hasattr(lib, "cos")


def test_load_lib_returns_queryable_handle():
    from numbox.core.bindings.utils import load_lib
    handle = load_lib("c")
    assert handle is not None
    # strlen is in libc on every supported platform
    assert hasattr(handle, "strlen")
    # A symbol that doesn't exist returns False
    assert not hasattr(handle, "definitely_not_a_real_symbol_xyzzy")


def test_c_stdio(tmp_path):
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
    # Bounded comparison: "hello" vs "help!" first matches 3 chars, differs at 4
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
    # strncmp("hello", "help!", 3) == 0: first 3 bytes "hel" match
    assert n_match == 0, n_match
    # strncmp("hello", "help!", 4) != 0: differs at byte 3 ('l' vs 'p')
    assert n_differ != 0, n_differ
    assert off_first == 2
    assert off_last == 9
    assert off_sub == 6

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
    """Set a unique env var that numbox's getenv binding can read, and
    clean it up on teardown.

    Plain ``monkeypatch.setenv`` is sufficient on POSIX. On Windows it
    isn't: ``numbox.core.bindings.utils.load_lib("c")`` resolves to
    ``msvcrt.dll`` (the legacy CRT compatibility shim — what
    ``ctypes.cdll.msvcrt`` returns), while Python 3.10+ uses UCRT for
    ``os.environ`` / ``monkeypatch.setenv``. MSVCRT and UCRT have
    separate ``environ`` tables, so a var set via the Python-side path
    is invisible to the binding's getenv call. To make the test
    deterministic on Windows, set the var via the same MSVCRT instance
    the binding uses (``msvcrt._putenv_s``); the env-table the binding
    reads is then the one that contains the var.
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
            # _putenv_s with empty value removes the var (MSDN docs).
            msvcrt._putenv_s(var_name.encode(), b"")
    else:
        monkeypatch.setenv(var_name, var_value)
        yield var_name, var_value


def test_c_env(_getenv_test_var):
    """getenv contract: returns a pointer to the environ-table string for set
    variables and 0 for unset variables. Verify both the presence/absence
    distinction AND that the returned pointer dereferences to the correct
    value — a stale or wrong-variable pointer would slip past an address-only
    check.

    Uses a dedicated test-controlled env var rather than the ambient
    ``PATH`` (hermetic CI runners can clear ``PATH``); see the
    ``_getenv_test_var`` fixture for the platform-aware setup.
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
    # On Windows sqlite3 MUST resolve to the CPython-bundled DLL and NEVER fall
    # back to find_library: a PATH-resident third-party sqlite3.dll (AWS CLI v2)
    # writes to NULL inside sqlite3_open from a foreign process. Pure logic, so
    # it runs on any host by monkeypatching the platform + candidate check.
    import ctypes.util as _ctypes_util
    from numbox.core.bindings import utils

    monkeypatch.setattr(utils, "platform_", "Windows")
    # find_msvcrt is Windows-only; inject it so the Windows branch imports on any host
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
    # For a non-sqlite3 name the bundled DLL is preferred over find_library when
    # both exist; find_library is the last resort only when no bundled DLL.
    import ctypes.util as _ctypes_util
    from numbox.core.bindings import utils

    monkeypatch.setattr(utils, "platform_", "Windows")
    monkeypatch.setattr(_ctypes_util, "find_msvcrt", lambda: "C:\\msvcrt.dll", raising=False)
    monkeypatch.setattr(utils, "find_library", lambda n: f"C:\\PATH\\{n}.dll")

    monkeypatch.setattr(utils, "_windows_bundled_dll_path", lambda n: "C:\\Py\\DLLs\\foo.dll")
    assert utils._resolve_lib_path("foo") == "C:\\Py\\DLLs\\foo.dll"

    monkeypatch.setattr(utils, "_windows_bundled_dll_path", lambda n: None)
    assert utils._resolve_lib_path("foo") == "C:\\PATH\\foo.dll"


if __name__ == "__main__":
    collect_and_run_tests(__name__)
