import pytest
from ctypes import addressof, c_char_p, c_int64, c_void_p
from numbox.core.bindings import *
from numbox.core.bindings.utils import platform_
from test.auxiliary_utils import collect_and_run_tests, str_from_p_as_int


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


@pytest.mark.skipif(platform_ == "Windows", reason="Need to add windows support")
def test_sqlite():
    version_ = sqlite3_libversion_number()
    version_ = str_from_p_as_int(version_)
    assert "." in version_

    db_name_ = ":memory:"
    db_name = c_char_p(db_name_.encode())
    db_name_p = c_void_p.from_buffer(db_name).value

    assert str_from_p_as_int(db_name_p) == db_name_
    db_p = c_int64(0)
    assert db_p.value == 0
    db_pp = addressof(db_p)
    rc = sqlite3_open(db_name_p, db_pp)
    assert rc == 0, "could not open db connection"
    assert db_p.value != 0
    db_p = db_p.value
    rc = sqlite3_close(db_p)
    assert rc == 0, "could not close db connection"


def test_load_lib_path_returns_handle_with_known_symbol():
    from ctypes.util import find_library
    from numbox.core.bindings.utils import load_lib_path

    libm_path = find_library("m")
    assert libm_path is not None, "libm not discoverable on this platform"
    lib = load_lib_path(libm_path)
    assert hasattr(lib, "cos")


if __name__ == "__main__":
    collect_and_run_tests(__name__)
