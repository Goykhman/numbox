numbox.core.proxy
=================

Overview
++++++++

Implementation of :func:`numbox.core.proxy.proxy` decorator that swaps definition of a jit-compiled
function in-place for a declaration (while delegating the actual implementation
to a different function that is only accessible indirectly). As a result, statically linking in libraries
corresponding to proxy-jitted functions called from other jitted functions will
only paste a declaration rather than the entire LLVM IR code.

Modules
++++++++

numbox.core.proxy.proxy
-----------------------

.. automodule:: numbox.core.proxy.proxy
   :members:
   :show-inheritance:
   :undoc-members:
