"""ABI primitives for numba bindings: platform / struct classification."""
import platform
import sys

from numba.core import types as nb_types
from numba.core.errors import TypingError


_PLATFORM_WIN_X64 = "win_x64"
_PLATFORM_SYSV_X86_64 = "sysv_x86_64"
_PLATFORM_AAPCS64 = "aapcs64"


def _current_platform():
    """Identify the C calling convention for the current host.

    Returns one of ``_PLATFORM_WIN_X64``, ``_PLATFORM_SYSV_X86_64``,
    ``_PLATFORM_AAPCS64``. Used by the ABI-aware codegen in
    ``numbox.core.bindings.call._call_lib_func`` to pick the right
    struct-passing convention. Raises ``RuntimeError`` on unsupported
    ``(sys.platform, platform.machine())`` combinations rather than
    silently misclassifying — Windows ARM64 (`platform.machine() ==
    "ARM64"`) is unsupported and would otherwise default to the wrong
    ABI dispatch.
    """
    machine = platform.machine().lower()
    if sys.platform == "win32":
        if machine in ("x86_64", "amd64"):
            return _PLATFORM_WIN_X64
        raise RuntimeError(
            f"Unsupported Windows architecture for ABI dispatch: "
            f"{platform.machine()} (only x86_64 / AMD64 supported)"
        )
    if machine in ("x86_64", "amd64"):
        return _PLATFORM_SYSV_X86_64
    if machine in ("arm64", "aarch64"):
        return _PLATFORM_AAPCS64
    raise RuntimeError(
        f"Unsupported platform for ABI dispatch: "
        f"{sys.platform}/{platform.machine()}"
    )


_CLASS_SCALAR = "scalar"
_CLASS_STRUCT_SMALL = "struct_small"
_CLASS_STRUCT_LARGE = "struct_large"


_WIN_REGISTER_PASSABLE_SIZES = (1, 2, 4, 8)


def _is_windows_register_passable(struct_bytes):
    """Whether a struct of size ``struct_bytes`` is passed/returned in
    registers on the Windows x64 ABI.

    Windows x64 passes aggregates of size 1, 2, 4, or 8 bytes directly
    (in integer registers) and returns them in RAX. Other sizes (3, 5,
    6, 7, or anything > 8) go via caller-allocated pointer for args and
    via ``sret`` for returns.
    """
    return struct_bytes in _WIN_REGISTER_PASSABLE_SIZES


def _classify(ty):
    """Classify a numba type for ABI dispatch.

    Returns one of:

    - ``_CLASS_SCALAR`` — any non-struct numba type (e.g. ``int32``,
      ``float64``, pointers represented as ``intp``).
    - ``_CLASS_STRUCT_SMALL`` — ``Record`` / ``BaseTuple`` of size
      <= 16 bytes; passed by value on register-passing ABIs.
    - ``_CLASS_STRUCT_LARGE`` — ``Record`` / ``BaseTuple`` of size
      > 16 bytes; passed by pointer with ``byval`` on SysV x86-64,
      by pointer (no special attribute) on other ABIs.
    """
    if not isinstance(ty, (nb_types.Record, nb_types.BaseTuple)):
        return _CLASS_SCALAR
    size = _struct_bytes(ty, "_classify")
    return _CLASS_STRUCT_SMALL if size <= 16 else _CLASS_STRUCT_LARGE


def _struct_bytes(ty, fn_name):
    """Compute struct size in bytes for a numba struct-shaped type.

    Supports ``types.Record`` (via ``.size``) and ``types.BaseTuple``
    subclasses — ``types.Tuple``, ``types.UniTuple``, and
    ``types.NamedTuple`` — via ``.types`` (a tuple of scalar field
    types with ``.bitwidth``). Anything else raises a clean
    ``TypingError`` naming the caller so a misuse doesn't surface as a
    cryptic ``AttributeError`` / ``KeyError``.
    """
    if isinstance(ty, nb_types.Record):
        return ty.size
    if isinstance(ty, nb_types.BaseTuple):
        return sum(t.bitwidth for t in ty.types) // 8
    raise TypingError(
        f"{fn_name}: expected a struct-shaped type (Record, Tuple, "
        f"UniTuple, or NamedTuple), got {ty!r}."
    )
