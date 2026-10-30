"""Helpers for building ``Work`` nodes from Python. :func:`make_work_helper` compiles a plain Python function as a
node's derive, and :func:`make_init_data` makes a scalar or array initial value.
"""
import numpy
from numba import typeof
from types import FunctionType as PyFunctionType, NoneType

from numbox.core.work.work import make_work
from numbox.utils.highlevel import cres


def make_work_helper(name, init_data, sources=(), derive_py=None, jit_options=None):
    """Create a ``Work`` node from Python, compiling the plain Python function ``derive_py`` as its derive.

    The derive's signature is the numba type of ``init_data`` over the numba types of the sources' current ``data``, as
    numba's ``typeof`` gives them. ``derive_py`` is compiled for that signature with
    :func:`~numbox.utils.highlevel.cres`, which receives ``jit_options``; without them, the derive compiles without a
    cache. A node with neither ``sources`` nor ``derive_py`` is an end node. ``sources`` without ``derive_py`` raises
    ``TypeError``, and a ``derive_py`` that is not a plain function fails an assertion.
    """
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
    """Return a node's initial value: ``val`` for an empty ``shape``, else an array of ``shape`` filled with ``val``.

    ``ty`` is the array's dtype; by default NumPy infers it from ``val``.
    """
    if not shape:
        return val
    return numpy.full(shape, val, ty)
