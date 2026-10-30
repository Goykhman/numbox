"""The C calling-convention rules that the bindings' internal helper applies to each call.

``numbox.core.bindings.call._call_lib_func`` uses this module to determine the calling convention that the host follows
and the class of each value in a call: a number or a pointer, a struct of 16 bytes or less, or a larger struct. The
module computes the size of a struct written as a numba tuple or record, and the offset of each field, as a C compiler
lays them out. For a 16-byte struct, the module computes the class of each 8-byte half, and those classes determine
whether the helper repacks the struct before the call. Every name in the module is private to the bindings.
"""
import platform
import sys

from numba.core import types as nb_types
from numba.core.errors import TypingError


_PLATFORM_WIN_X64 = "win_x64"
_PLATFORM_SYSV_X86_64 = "sysv_x86_64"
_PLATFORM_AAPCS64 = "aapcs64"


def _current_platform():
    """Return the C calling convention that the host follows, by which ``_call_lib_func`` passes structs.

    The result is ``_PLATFORM_WIN_X64`` on Windows on x86-64, ``_PLATFORM_SYSV_X86_64`` on any other system on x86-64,
    such as Linux or macOS, and ``_PLATFORM_AAPCS64`` on a system other than Windows on 64-bit Arm. The function raises
    ``RuntimeError`` for any other pair of ``sys.platform`` and ``platform.machine()``, so that no host is classified
    under another host's convention. It raises the error on Windows on Arm64 as well, although that machine's name,
    ``ARM64``, would otherwise select AAPCS64.
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
    """Return True if Windows x64 passes and returns a struct of ``struct_bytes`` bytes in a register.

    The Windows x64 convention passes a struct of 1, 2, 4 or 8 bytes in an integer register and returns it in RAX. It
    passes a struct of any other size as a pointer to a copy that the caller makes, and returns one through a hidden
    pointer argument, ``sret``.
    """
    return struct_bytes in _WIN_REGISTER_PASSABLE_SIZES


# This comment lists what C receives when _call_lib_func passes a numba tuple by value. The rule holds on x86-64 Linux,
# where it was measured against C functions that take and return structs by value; it has not been measured on
# AAPCS64 or Windows x64.
# - Up to 16 bytes on SysV x86-64 and AAPCS64: the helper passes and returns the struct as an LLVM aggregate, which
#   LLVM lowers one field at a time. call.py repacks a 16-byte value to {i64, i64} when both 8-byte halves are
#   INTEGER class (_classify_eightbytes, _needs_int_int_eightbyte_repack). A struct is passed and returned intact if
#   it has a single field or if it is exactly 16 bytes with one field in each half. A struct of exactly 16 bytes with
#   two or more fields in a half is also intact if neither half is all floating point. A struct under 16 bytes with two
#   or more fields is corrupted: C's div, for example, returns an 8-byte div_t whose remainder the helper reads as 0. A
#   struct of exactly 16 bytes is also corrupted if a half holds two or more fields and either half is all floating
#   point, such as (float32, float32, int64) or (int32, int32, float64).
# - Windows x64: the helper passes a struct of 1, 2, 4 or 8 bytes by value and a struct of any other size as a pointer
#   to a copy. A result of any other size is returned through sret.
# - Over 16 bytes: on SysV x86-64, the argument has the byval attribute, so LLVM copies the struct onto the stack and
#   C receives the copy rather than a pointer. The function that makes the call is then compiled with optnone and
#   noinline. AAPCS64 and Windows x64 pass a pointer to a copy. A result is returned through sret. Structs over 16
#   bytes are passed and returned intact on x86-64 Linux.
# - Record: numba's value for a record is a pointer to its data, so C receives an address instead of the fields. A
#   Record result of 16 bytes or less crashes the process with a segmentation fault, and _call_lib_func refuses a
#   larger one.
# LLVM does not implement the C convention for passing structs; the compiler front end applies it:
# https://github.com/numba/llvmlite/issues/300#issuecomment-327235846
def _classify(ty):
    """Return the class by which ``_call_lib_func`` passes a value of the numba type ``ty``.

    The class is ``_CLASS_SCALAR`` for a type other than a ``Record`` or a tuple, such as ``int32``, ``float64`` or a
    pointer held as ``intp``. A ``Record`` or a tuple of 16 bytes or less is ``_CLASS_STRUCT_SMALL``, and a larger one
    is ``_CLASS_STRUCT_LARGE``. ``_call_lib_func`` passes a small struct as a value on SysV x86-64 and AAPCS64, and on
    Windows x64 only at 1, 2, 4 or 8 bytes; it passes a large struct through memory. A tuple with a field of no fixed
    bit width raises ``TypingError``.
    """
    if not isinstance(ty, (nb_types.Record, nb_types.BaseTuple)):
        return _CLASS_SCALAR
    size = _struct_bytes(ty, "_classify")
    return _CLASS_STRUCT_SMALL if size <= 16 else _CLASS_STRUCT_LARGE


def _align_up(n, alignment):
    """Return ``n`` rounded up to a multiple of ``alignment``."""
    return (n + alignment - 1) // alignment * alignment


def _basetuple_layout(ty, fn_name):
    """Return the C layout of a tuple of scalar fields, as ``(total_size, [(offset, size, is_float), ...])``.

    Each field is placed at the next offset that is a multiple of its size, and the total is padded to a multiple of the
    largest field's alignment, as in the C struct with the same fields. ``is_float`` is True for a ``Float`` field. A
    field with no fixed bit width, such as a ``voidptr`` or a function, raises ``TypingError`` naming ``fn_name``
    instead of an ``AttributeError``; a caller passes a pointer as ``intp``.
    """
    # numba lowers a heterogeneous tuple to an unpacked LLVM struct and a UniTuple to an LLVM array. In both cases, each
    # field is placed at its natural alignment, as in the C struct and in the equivalent Record. The total equals the
    # size that numba's get_abi_sizeof reports for the type. A sum of the fields' bit widths would undercount the size
    # and misplace fields. Both the small or large class and the split into 8-byte halves would then be wrong.
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
        align = size  # A scalar field's alignment equals its size.
        offset = _align_up(offset, align)
        fields.append((offset, size, isinstance(ft, nb_types.Float)))
        offset += size
        max_align = max(max_align, align)
    return _align_up(offset, max_align), fields


def _struct_bytes(ty, fn_name):
    """Return the size in bytes of a ``Record`` or a tuple type.

    A ``Record`` gives its own ``size``. The size of a tuple, any numba ``BaseTuple`` such as ``Tuple``, ``UniTuple`` or
    ``NamedTuple``, is the size of its C layout from ``_basetuple_layout``. Any other type raises ``TypingError`` naming
    ``fn_name``, so that a misuse fails with that message rather than with an ``AttributeError`` or a ``KeyError``.
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
    """Yield ``(offset, size, is_float)`` for each scalar field of a ``Record`` or a tuple type.

    ``is_float`` is True for a ``Float`` field, ``float32`` or ``float64``; ``_classify_eightbytes`` turns these flags
    into the SysV class of each 8-byte half. ``_classify_eightbytes`` uses the size to find a field that spans the
    boundary between the halves. A tuple's fields and the padding between them follow the layout of
    ``_basetuple_layout``, which matches the way numba lays out the tuple in memory. Each field of a ``Record`` has its
    own offset, which already accounts for C's padding. Any other type raises ``TypingError`` naming ``fn_name``.
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
    """Return the SysV class of each 8-byte half of a 16-byte struct, as ``(class_lo, class_hi)``.

    A half is ``_EIGHTBYTE_CLASS_SSE`` if every field that overlaps it is a float, and ``_EIGHTBYTE_CLASS_INTEGER`` if
    any field that overlaps it is an integer, including a pointer held as ``intp``. ``_call_lib_func`` uses the classes
    on SysV x86-64 and on AAPCS64 alike: with ``_is_canonical_int64_pair_layout``, they determine whether a 16-byte
    struct passed by value is repacked to ``{i64, i64}``. The SysV convention defines further classes, such as X87 and
    NO_CLASS. This function does not model them, because the bindings pass structs of integer and float fields only. A
    ``ty`` that is not a 16-byte ``Record`` or tuple raises ``TypingError``.
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
        # A field that spans the 8-byte boundary overlaps both halves.
        touches_lo = offset < 8
        touches_hi = offset + size > 8
        if is_float:
            lo_has_float |= touches_lo
            hi_has_float |= touches_hi
        else:
            lo_has_int |= touches_lo
            hi_has_int |= touches_hi
    # Under the SysV merge rule, a half that holds any integer field is INTEGER class, even if it also holds a float,
    # and is passed in a general-purpose register. Only a half that holds floats alone is SSE class. LLVM does not
    # apply this rule to an aggregate. It passes each field in a register of its own, while C expects the fields of each
    # half packed into one register. For that reason, the repack in call.py passes a struct with two INTEGER halves as
    # {i64, i64}. LLVM then passes each half in the register where C reads it.
    cls_lo = (_EIGHTBYTE_CLASS_SSE if (lo_has_float and not lo_has_int)
              else _EIGHTBYTE_CLASS_INTEGER)
    cls_hi = (_EIGHTBYTE_CLASS_SSE if (hi_has_float and not hi_has_int)
              else _EIGHTBYTE_CLASS_INTEGER)
    return cls_lo, cls_hi


def _is_canonical_int64_pair_layout(ty):
    """Return True if ``ty`` is two 64-bit integer fields at offsets 0 and 8, with nothing else.

    A tuple qualifies if it has exactly two 64-bit ``Integer`` fields, and a ``Record`` qualifies if its size is 16 and
    it has two such fields at offsets 0 and 8. ``_call_lib_func`` does not repack such a type to ``{i64, i64}``. LLVM
    already passes its two 64-bit integers in two general-purpose registers, as C passes a struct with two INTEGER
    halves.
    """
    if isinstance(ty, nb_types.BaseTuple):
        return len(ty.types) == 2 and all(
            isinstance(f, nb_types.Integer) and f.bitwidth == 64
            for f in ty.types
        )
    if isinstance(ty, nb_types.Record):
        if ty.size != 16:
            return False
        # A Record built from a NumPy dtype with explicit offsets can declare its fields out of offset order, so the
        # fields are sorted by offset, and such a Record qualifies when the sorted offsets are 0 and 8.
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
