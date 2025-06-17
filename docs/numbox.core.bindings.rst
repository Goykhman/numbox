numbox.core.bindings
====================

Overview
++++++++

Loads dynamic libraries available in the Python environment, such as, `libc`, `libm`, and `libsqlite3`
in global symbol mode (`RTLD_GLOBAL`) via `ctypes`.
This adds global symbols (including native API) exported from those libraries to the LLVM symbol table.
These functions can then be invoked from the numba jitted code [#f1]_, complementing the suite
of numba-supported functionality.

Analogous technique can be expanded as needed for the user custom code.

.. rubric:: References

.. [#f1] See `numbsql <https://github.com/cpcloud/numbsql>`_ for previous work on jit-wrapping FFI imported functions.

Modules
++++++++

numbox.core.bindings._c
-----------------------

.. automodule:: numbox.core.bindings._c
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._math
--------------------------

.. automodule:: numbox.core.bindings._math
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._sqlite
----------------------------

.. automodule:: numbox.core.bindings._sqlite
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.call
-------------------------

.. automodule:: numbox.core.bindings.call
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.signatures
-------------------------------

.. automodule:: numbox.core.bindings.signatures
   :members:
   :show-inheritance:
   :undoc-members:


numbox.core.bindings.utils
--------------------------

.. automodule:: numbox.core.bindings.utils
   :members:
   :show-inheritance:
   :undoc-members:
