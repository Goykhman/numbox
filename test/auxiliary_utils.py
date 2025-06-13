import logging
import sys
from numba import njit
from numba.core import types
from numba.extending import intrinsic


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
