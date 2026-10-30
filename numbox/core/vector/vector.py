"""A growable vector of one numba scalar type for jitted code, stored in a NumPy array.

:func:`make_vector` returns a jitted factory of vectors of one element type, and their numba type. In jitted code,
``len(v)``, ``v[i]`` and ``v[i] = x`` work on a vector, :func:`vector_push` appends an element and
:func:`vector_extend` appends the elements of another vector. If the elements to append do not fit, the buffer is
replaced by one whose capacity is the old capacity doubled until they fit.
"""
import numpy
import operator

from numba import njit, types as nb_types
from numba.core.errors import NumbaError
from numba.experimental import structref
from numba.experimental.structref import StructRefProxy, define_boxing, new
from numba.extending import overload

from numbox.core.configurations import jit_options


@structref.register
class VectorTypeClass(nb_types.StructRef):
    """The numba type class of ``Vector``.

    Each element type has an instance of its own, a structref type with the fields ``buf`` and ``size``.
    """
    def preprocess_fields(self, fields):
        """Return ``fields`` with each literal field type replaced by its non-literal type."""
        return tuple((n, nb_types.unliteral(t)) for n, t in fields)


deleted_vector_ctor_error = "Use the factory returned by make_vector(elem_type)"


class Vector(StructRefProxy):
    """The Python object for a vector of one numba scalar type, whose buffer grows as elements are appended.

    Create one with the ``create`` function that :func:`make_vector` returns. Calling the class raises
    ``NotImplementedError`` in Python, and jitted code that calls it fails to compile.
    """
    def __new__(cls, *args, **kwargs):
        raise NotImplementedError(deleted_vector_ctor_error)

    @property
    @njit(**jit_options)
    def size(self):
        """The number of elements in the vector."""
        return self.size

    @property
    @njit(**jit_options)
    def buf(self):
        """The NumPy array that holds the elements.

        Its length is the capacity of the vector, and only its first ``size`` elements are set.
        """
        return self.buf


def _vector_deleted_ctor(*args):
    raise NumbaError(deleted_vector_ctor_error)


overload(Vector)(_vector_deleted_ctor)
define_boxing(VectorTypeClass, Vector)


_vector_cache = {}


def make_vector(elem_type):
    """Return ``(create, vector_type)``: a jitted factory of vectors of ``elem_type``, and their numba type.

    ``create(capacity)`` returns an empty vector whose buffer holds ``capacity`` elements. It takes no buffer, so the
    buffer's dtype always follows from ``elem_type``. A capacity below 1 fails an ``assert`` in ``create``. A buffer of
    capacity 0 would never grow, because each growth doubles it: ``vector_extend`` would loop forever, and
    ``vector_push`` would write past the end of the buffer. A vector grows and never shrinks, so a positive capacity
    stays positive.

    The buffer's dtype is ``numpy.dtype(str(elem_type))``, so ``elem_type`` must be a scalar type whose name NumPy
    accepts, such as ``float64`` or ``int64``. For another type, such as ``unicode_type``, NumPy raises ``TypeError``.
    The pair is stored in ``_vector_cache`` under ``elem_type.key``, and a later call with the same key returns it.
    """
    key = elem_type.key
    if key in _vector_cache:
        return _vector_cache[key]

    type_inst = VectorTypeClass([
        ("buf", nb_types.Array(elem_type, 1, 'C')),
        ("size", nb_types.int64),
    ])

    np_dtype = numpy.dtype(str(elem_type))

    @njit
    def create(capacity):
        """Return an empty vector whose buffer holds ``capacity`` elements; a capacity below 1 fails the ``assert``."""
        assert capacity >= 1
        v = new(type_inst)
        v.buf = numpy.empty(capacity, dtype=np_dtype)
        v.size = 0
        return v

    result = (create, type_inst)
    _vector_cache[key] = result
    return result


@overload(len, jit_options=jit_options)
def _vector_len(v):
    if isinstance(v, VectorTypeClass):
        def impl(v):
            """Return the number of elements in the vector ``v``, for ``len(v)`` in jitted code."""
            return v.size
        return impl


@overload(operator.getitem, jit_options=jit_options)
def _vector_getitem(v, i):
    if isinstance(v, VectorTypeClass):
        def impl(v, i):
            """Return ``v.buf[i]``, for ``v[i]`` in jitted code; ``i`` is not checked against ``len(v)``.

            A negative ``i`` counts from the end of the buffer, not from the last element.
            """
            return v.buf[i]
        return impl


@overload(operator.setitem, jit_options=jit_options)
def _vector_setitem(v, i, val):
    if isinstance(v, VectorTypeClass):
        def impl(v, i, val):
            """Set ``v.buf[i]`` to ``val``, for ``v[i] = val`` in jitted code.

            ``i`` is not checked against ``len(v)``, and ``len(v)`` does not change.
            """
            v.buf[i] = val
        return impl


@njit(**jit_options)
def vector_push(v, val):
    """Append ``val`` to the vector ``v``, doubling the buffer's capacity first if the buffer is full.

    numba converts ``val`` to the element type as it does for an array assignment, so a float appended to an integer
    vector is truncated.
    """
    if v.size == v.buf.shape[0]:
        new_buf = numpy.empty(v.buf.shape[0] * 2, v.buf.dtype)
        new_buf[:v.size] = v.buf[:v.size]
        v.buf = new_buf
    v.buf[v.size] = val
    v.size += 1


@njit(**jit_options)
def vector_extend(dst, src):
    """Append the elements of the vector ``src`` to the vector ``dst``, doubling the capacity of ``dst`` until they fit.

    numba converts the elements to the element type of ``dst`` as it does for an array assignment.
    """
    needed = dst.size + src.size
    cap = dst.buf.shape[0]
    if needed > cap:
        while cap < needed:
            cap *= 2
        new_buf = numpy.empty(cap, dst.buf.dtype)
        new_buf[:dst.size] = dst.buf[:dst.size]
        dst.buf = new_buf
    dst.buf[dst.size:dst.size + src.size] = src.buf[:src.size]
    dst.size += src.size
