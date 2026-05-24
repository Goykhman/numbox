"""Project-level pytest conftest.

On macOS, ``numbox.utils.pysqlite_bridge`` patches numba's symbol
resolution so ``@njit`` sqlite3 bindings link to Python's bundled
libsqlite3 (statically linked into ``_sqlite3.so``) rather than the
system ``/usr/lib/libsqlite3.dylib`` (older, struct-layout-incompatible).
The patch must run *before* ``numbox.core.bindings._sqlite_conn``
imports trigger eager ``@proxy`` compilation. Pytest collects test files
in alphabetical order and ``test/core/test_sqlite_*.py`` come before
``test/utils/test_pysqlite_bridge.py``, so without this hook the
sqlite bindings would already be eagerly compiled (and linked to
``/usr/lib``) by the time the bridge module loads.

No-op on Linux and Windows -- the issue is macOS-specific.
"""
from platform import system

if system() == "Darwin":
    import numbox.utils.pysqlite_bridge  # noqa: F401
