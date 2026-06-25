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


def _align_up(n, alignment):
    return (n + alignment - 1) // alignment * alignment


def _basetuple_layout(ty, fn_name):
    """Natural-alignment C layout for a ``BaseTuple`` of scalar fields.

    numba lowers a tuple to a *non-packed* LLVM struct, so each field sits at
    the next offset aligned up to its own size and the total is padded up to
    the largest field's alignment — exactly the layout of the equivalent
    ``Record`` (verified against ``get_data_type`` / ``get_abi_sizeof``). A
    plain ``sum(bitwidth)`` bit-packed model under-counts the size and mislocates
    fields, which corrupts both the SMALL/LARGE classification and the eightbyte
    split. Returns ``(total_size, [(offset, size, is_float), ...])``.

    A field with no fixed ``bitwidth`` (a pointer / function / other non-scalar
    type) raises a clean ``TypingError`` naming ``fn_name`` rather than a cryptic
    ``AttributeError`` — raw pointers are expected to be passed as ``intp``.
    """
    offset = 0
    max_align = 1
    fields = []
    for ft in ty.types:
        if not hasattr(ft, "bitwidth"):
            raise TypingError(
                f"{fn_name}: tuple field {ft!r} has no fixed bitwidth; only "
                f"fixed-width scalar fields (Integer, Float) are supported "
                f"(pointers must be passed as intp)."
            )
        size = ft.bitwidth // 8
        align = size  # primitive scalar fields: alignment == size
        offset = _align_up(offset, align)
        fields.append((offset, size, isinstance(ft, nb_types.Float)))
        offset += size
        max_align = max(max_align, align)
    return _align_up(offset, max_align), fields


def _struct_bytes(ty, fn_name):
    """Compute struct size in bytes for a numba struct-shaped type.

    Supports ``types.Record`` (via ``.size``) and ``types.BaseTuple``
    subclasses — ``types.Tuple``, ``types.UniTuple``, and
    ``types.NamedTuple`` — via natural-alignment layout (see
    ``_basetuple_layout``). Anything else raises a clean ``TypingError``
    naming the caller so a misuse doesn't surface as a cryptic
    ``AttributeError`` / ``KeyError``.
    """
    if isinstance(ty, nb_types.Record):
        return ty.size
    if isinstance(ty, nb_types.BaseTuple):
        return _basetuple_layout(ty, fn_name)[0]
    raise TypingError(
        f"{fn_name}: expected a struct-shaped type (Record, Tuple, "
        f"UniTuple, or NamedTuple), got {ty!r}."
    )


_EIGHTBYTE_CLASS_INTEGER = "integer"
_EIGHTBYTE_CLASS_SSE = "sse"


def _iter_struct_fields(ty, fn_name):
    """Yield ``(offset, size, is_float)`` for each scalar field of a
    struct-shaped numba type. ``is_float`` is true for ``Float`` types
    (float / double); the SysV ABI's "field is float ⇒ eightbyte is
    SSE" mapping is the consumer's responsibility (see
    ``_classify_eightbytes``). Size is needed to detect fields that
    span the 8-byte eightbyte boundary.

    For ``BaseTuple`` the fields follow the natural-alignment layout (see
    ``_basetuple_layout``, mirroring ``_struct_bytes``), so padded gaps match
    the non-packed LLVM struct numba lowers tuples to. For ``Record`` each
    ``_RecordField`` carries an explicit ``offset`` that already accounts for
    any C-layout padding gaps.
    """
    if isinstance(ty, nb_types.BaseTuple):
        yield from _basetuple_layout(ty, fn_name)[1]
        return
    if isinstance(ty, nb_types.Record):
        for fld in ty.fields.values():
            yield (
                fld.offset,
                fld.type.bitwidth // 8,
                isinstance(fld.type, nb_types.Float),
            )
        return
    raise TypingError(
        f"{fn_name}: expected a struct-shaped type (Record, Tuple, "
        f"UniTuple, or NamedTuple), got {ty!r}."
    )


def _classify_eightbytes(ty):
    """Classify the two eightbytes of a 16-byte struct.

    Returns ``(class_lo, class_hi)`` where each is
    ``_EIGHTBYTE_CLASS_INTEGER`` (ints / pointers) or
    ``_EIGHTBYTE_CLASS_SSE`` (float / double). The SysV merge rule
    applies per eightbyte: an eightbyte is SSE only if every field that
    touches it is float / double; if it holds *any* integer field it is
    INTEGER (SSE + INTEGER -> INTEGER, i.e. passed in a GP register).
    The class names are SysV-flavoured but the helper drives the
    repack path on both SysV x86-64 and AAPCS64. (The full SysV ABI
    has X87 / NO_CLASS / etc.; our scope is fixed-size 16-byte
    aggregates of scalar integer-or-float fields, which is what
    numbox bindings consume.)

    Used by ``_call_lib_func`` together with
    ``_is_canonical_int64_pair_layout`` to decide when a 16-byte by-
    value arg needs repacking via memory bitcast to ``{i64, i64}`` to
    work around llvmlite not modelling the eightbyte-packing rule (it
    drops fields like the second ``i32`` in ``{i32, i32, i64}`` —
    duckdb_interval).

    Raises ``TypingError`` if ``ty`` is not a 16-byte struct-shaped
    type — the caller should already have classified the size.
    """
    size = _struct_bytes(ty, "_classify_eightbytes")
    if size != 16:
        raise TypingError(
            f"_classify_eightbytes: expected a 16-byte struct, got "
            f"{size}-byte ({ty!r})."
        )
    lo_has_int = hi_has_int = False
    lo_has_float = hi_has_float = False
    for offset, size, is_float in _iter_struct_fields(
            ty, "_classify_eightbytes"):
        # A field spanning the 8-byte boundary touches both eightbytes.
        touches_lo = offset < 8
        touches_hi = offset + size > 8
        if is_float:
            lo_has_float |= touches_lo
            hi_has_float |= touches_hi
        else:
            lo_has_int |= touches_lo
            hi_has_int |= touches_hi
    # SysV merge rule: an eightbyte holding any integer field is INTEGER
    # (passed in a GP register), even if it also holds a float -- SSE +
    # INTEGER -> INTEGER. Only an all-float eightbyte is SSE. This is the
    # rule llvmlite does not model (it would otherwise lower a mixed
    # int/float eightbyte through SSE and drop the integer field), which is
    # exactly why the int/int repack in call.py exists.
    cls_lo = (_EIGHTBYTE_CLASS_SSE if (lo_has_float and not lo_has_int)
              else _EIGHTBYTE_CLASS_INTEGER)
    cls_hi = (_EIGHTBYTE_CLASS_SSE if (hi_has_float and not hi_has_int)
              else _EIGHTBYTE_CLASS_INTEGER)
    return cls_lo, cls_hi


def _is_canonical_int64_pair_layout(ty):
    """Whether ``ty`` already lowers to LLVM ``{i64, i64}`` — i.e. two
    64-bit integer fields at offsets 0 and 8 with no gaps. When True,
    the SysV-x86-64 small-struct INT/INT path needs no repack: llvmlite
    already emits the canonical eightbyte-pair shape.
    """
    if isinstance(ty, nb_types.BaseTuple):
        return len(ty.types) == 2 and all(
            isinstance(f, nb_types.Integer) and f.bitwidth == 64
            for f in ty.types
        )
    if isinstance(ty, nb_types.Record):
        if ty.size != 16:
            return False
        # Sort by offset so a Record built with fields declared in
        # non-offset order (e.g. via numpy structured dtype with
        # explicit `offsets`) is still recognized as canonical when
        # the offsets sort to [0, 8].
        flds = sorted(ty.fields.values(), key=lambda f: f.offset)
        if len(flds) != 2:
            return False
        if [f.offset for f in flds] != [0, 8]:
            return False
        return all(
            isinstance(f.type, nb_types.Integer) and f.type.bitwidth == 64
            for f in flds
        )
    return False
