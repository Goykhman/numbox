import logging
import re
import sys
from ctypes import addressof, c_char, c_char_p, c_int64, c_void_p
from io import BytesIO
from numba import njit
from numba.core import types
from numba.extending import intrinsic


# https://stackoverflow.com/a/14693789
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def collect_and_run_tests(module_name):
    module = sys.modules[module_name]
    for name, item in module.__dict__.items():
        if name.startswith("test_") and callable(item):
            logger.info(f" Running {name}")
            item()


@intrinsic
def _deref_int64_intp(typingctx, p_int_ty):
    sig = types.int64(types.intp)

    def codegen(context, builder, signature, args):
        p_ty_ll = context.get_value_type(p_int_ty).as_pointer()
        ptr = builder.inttoptr(args[0], p_ty_ll)
        return builder.load(ptr)
    return sig, codegen


@njit
def deref_int64_intp(p_int):
    return _deref_int64_intp(p_int)


def test_deref_int64_intp():
    v1 = c_int64(137)
    v1_p = addressof(v1)
    assert deref_int64_intp(v1_p) == 137


def str_from_p_as_int(p_as_int):
    b = BytesIO()
    while True:
        c = c_char.from_address(p_as_int).value
        if c == b"\x00":
            break
        b.write(c)
        p_as_int += 1
    return b.getvalue().decode("utf-8")


def test_str_from_p_as_int():
    s1_ = "a random string"
    s1_b = s1_.encode("utf-8")
    s1 = c_char_p(s1_b)
    s1_p = c_void_p.from_buffer(s1).value
    assert str_from_p_as_int(s1_p) == s1_
