numbox.core.vector
====================

Overview
++++++++

Generic growable numba vector backed by a numpy array.

Compared to ``numba.typed.List``:

- ``List`` supports arbitrary element types (including other structrefs)
  and exposes a richer API (``append``, ``pop``, ``insert``, ``remove``,
  slicing).
- ``Vector`` is restricted to scalar element types where ``str(elem_type)``
  matches a numpy dtype (``float64``, ``int64``, etc.).
  ``make_vector`` memoises instances by ``elem_type.key``, so cached code
  keeps the same type identity across processes. Storage is a single
  ``numpy.ndarray``, so per-element overhead is the scalar itself plus
  amortised geometric growth.

Modules
++++++++

numbox.core.vector.vector
-----------------------------

.. automodule:: numbox.core.variable.variable
   :members:
   :show-inheritance:
   :undoc-members:
