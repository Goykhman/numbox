import numpy
from numba import typeof
from types import FunctionType as PyFunctionType, NoneType

from numbox.core.work.work import make_work
from numbox.utils.highlevel import cres


def make_work_helper(name, init_data, sources=(), derive_py=None, jit_options=None):
    """ Utility for creating instances of `Work` from Python scope.
     Python abstraction for the jitted scope functionality. """
    assert isinstance(derive_py, (NoneType, PyFunctionType))
    if sources or derive_py:
        jit_options = jit_options if jit_options is not None else {}
        sources_ty = []
        for source in sources:
            sources_ty.append(typeof(source.data))
        derive_sig = typeof(init_data)(*sources_ty)
        derive = cres(derive_sig, **jit_options)(derive_py)
        return make_work(name, data=init_data, sources=sources, derive=derive)
    else:
        return make_work(name, data=init_data)


def make_init_data(shape=(), val=0.0, ty=None):
    if not shape:
        return val
    return numpy.full(shape, val, ty)
