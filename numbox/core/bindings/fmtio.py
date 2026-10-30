"""C's ``printf``, ``fprintf``, ``snprintf`` and ``sscanf``, for Python and for jitted code.

The four public functions are ordinary Python functions that take C's arguments in C's order, with the values after the
format as separate arguments, not in a tuple. In jitted code, numba compiles each call, through an ``@overload``, into a
call of the C function of the same name, with the format embedded as a constant. :func:`printf`, :func:`fprintf` and
:func:`snprintf` run from the same source under Python and in jitted code: Python's ``%`` operator formats the text
under Python, and C formats it in jitted code. :func:`sscanf` runs in jitted code only. numba compiles a jitted call
into its caller's code. A caller cached with ``cache=True`` therefore runs the fmtio code of the numbox version that
compiled it, as the section "Caching callers of the formatted I/O functions" of the bindings page explains. The section
"Formatted output and input" there describes the conversions, where Python's output and C's differ, ``snprintf``'s
truncation and ``sscanf``'s storage. The rest of the module is private: the intrinsics that emit the C calls, and the
checks that numbox runs on a format when numba compiles a call.
"""
import ctypes
import re
import sys

from llvmlite import ir as llir
from numba.core import cgutils
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import (
    BaseTuple, Boolean, Float, Integer, UnicodeType, int32, intp, uintp,
    unliteral,
)
from numba.extending import intrinsic, overload

from numbox.core.bindings.utils import (
    extract_literal_str, intp_ll_type, load_lib, platform_,
)


__all__ = ["printf", "fprintf", "snprintf", "sscanf"]


load_lib("c")


# On Windows, jitted code calls _snprintf, which msvcrt.dll exports with a truncation rule older than C99's: it returns
# -1 when the text does not fit, and it writes no NUL when the text fills the buffer exactly. The UCRT defines C99's
# snprintf as an inline function in its stdio.h, and a call declared as i32 @snprintf(...) crashes with an access
# violation when the JIT linker resolves it. snprintf's docstring states the Windows result. Under Python, snprintf
# follows the C99 rule on every platform.
_SNPRINTF_SYMBOL = "_snprintf" if platform_ == "Windows" else "snprintf"


# The pattern matches C's length modifiers: hh, h, l, ll and L, the C99 j, z and t, the BSD q, and Microsoft's I32 and
# I64. Python's % operator ignores a single h, l or L and rejects every other modifier with "ValueError: unsupported
# format character". The Python path therefore removes them all before it formats. Python then prints each value at
# its full width. C prints the same text whenever the value fits the type that its conversion names, such as signed
# char for %hhd. Unlike the checks below, the pattern does not mask a %% escape first. It also removes a modifier after
# a literal %%, so "%%lld" becomes "%%d", which Python prints as %d while C prints %lld.
_LENGTH_MODIFIER_RE = re.compile(
    r'%([-+0# ]*[0-9*]*\.?[0-9*]*)(hh|ll|h|l|L|j|z|t|q|I32|I64)([diouxXfFeEgGaAcsp])'
)


def _python_fmt_compat(fmt):
    """Return ``fmt`` without C's length modifiers, so that Python's ``%`` operator accepts it.

    A modifier after a literal ``%%`` is removed as well, so ``"%%lld"`` becomes ``"%%d"``.
    """
    return _LENGTH_MODIFIER_RE.sub(r'%\1\3', fmt)


# ---------------------------------------------------------------------------
# Private helpers for the intrinsics' code generation and checks
# ---------------------------------------------------------------------------


def _promote_for_varargs(builder, arg_ty, arg_val):
    """Return ``arg_val`` as the C variadic call takes it: an integer or ``bool`` as 64 bits, a float as a double.

    A ``float32`` is extended to ``float64``, as C's default argument promotion does. A ``bool`` and an unsigned
    integer narrower than 64 bits are zero-extended to 64 bits, and a signed one is sign-extended. A 64-bit integer, a
    pointer held as ``intp`` and a ``float64`` pass unchanged. Any other type raises ``TypingError``.

    C promotes a narrow integer only to ``int``, so the widening to 64 bits goes beyond C's rule. It lets ``%lld`` read
    an ``int32`` correctly; without it, C would read 8 bytes from a 4-byte argument, and the high 4 bytes would hold
    indeterminate data. It also matches the Python path, which prints each value at its full width. The cost is one LLVM
    ``sext``, ``zext`` or ``fpext`` instruction per argument, which costs nothing at run time.
    """
    i64_ll = llir.IntType(64)
    if isinstance(arg_ty, Float):
        if arg_ty.bitwidth == 32:
            return builder.fpext(arg_val, llir.DoubleType())
        return arg_val
    if isinstance(arg_ty, Boolean):
        return builder.zext(arg_val, i64_ll)
    if isinstance(arg_ty, Integer):
        if arg_ty.bitwidth < 64:
            if arg_ty.signed:
                return builder.sext(arg_val, i64_ll)
            return builder.zext(arg_val, i64_ll)
        return arg_val
    # The overloads reject any other argument type when numba types the call (_validate_writer_arg_type), so this
    # raise statement is a second guard. It stops the code generation, instead of passing such a value to C, if a
    # change in numba, such as in the way numba flattens tuples, admits another type.
    raise TypingError(
        f"variadic arg of type {arg_ty!r} is not supported by printf-family "
        f"bindings; allowed: Float, Integer, Boolean, or intp pointer"
    )


def _validate_writer_arg_type(name, idx, ty):
    """Raise ``TypingError`` unless ``ty`` is an argument type that ``printf``, ``fprintf`` and ``snprintf`` take.

    The types are ``Float``, ``Integer``, ``Boolean`` and a unicode type, and ``Integer`` includes a pointer held as
    ``intp``. The overload passes a unicode argument to C as the address of its characters, through
    ``get_unicode_data_p``. The check runs when numba types the call, so an array, a complex number or a tuple fails
    there, with a message that lists the accepted types.
    """
    if isinstance(unliteral(ty), (Float, Integer, Boolean, UnicodeType)):
        return
    raise TypingError(
        f"{name}: arg {idx} has unsupported type {ty!r}; allowed: Float, "
        f"Integer, Boolean, unicode (auto-converted to char* via "
        f"get_unicode_data_p), or intp (raw pointer / preconverted string)"
    )


_PERCENT_N_RE = re.compile(
    r'%(?:[0-9]+\$)?[-+0# ]*[*0-9]*(?:[0-9]+\$)?'
    r'(?:\.[*0-9]*(?:[0-9]+\$)?)?'
    r'(?:hh|ll|h|l|L|j|z|t|q|I32|I64)?n'
)


def _reject_percent_n_or_raise(name, fmt_str):
    """Raise ``TypingError`` if ``fmt_str`` holds a ``%n`` conversion, with any flags, width, precision or modifier.

    ``%n`` makes C write the number of bytes printed so far through a pointer that the caller passes. Python's ``%``
    operator rejects it with ``ValueError``, so allowing it in jitted code would make the two modes differ. A write
    through a pointer argument is also a memory-safety hazard: static analyzers flag it, and glibc's
    ``_FORTIFY_SOURCE`` makes a program abort on ``%n`` in a format held in writable memory. A literal ``%%`` followed
    by ``n``, ``%%n``, is allowed.
    """
    # Each %% escape becomes two NULs, on which the pattern cannot match, so the pattern does not read the n of %%n as a
    # conversion. _reject_grouping_flag_or_raise and _validate_format_vs_args mask the format the same way.
    stripped = fmt_str.replace('%%', '\x00\x00')
    if _PERCENT_N_RE.search(stripped):
        raise TypingError(
            f"{name}: %n directive in format string {fmt_str!r} is not "
            f"allowed (writes byte-count-written through caller pointer; "
            f"memory-safety hazard, also diverges from pure-Python behavior). "
            f"Use sscanf if you need %n's read-position semantics."
        )


# A conversion whose flags include ', the digit-grouping flag, matched in the format after the %% escapes are masked.
_GROUPING_FLAG_RE = re.compile(r"%[-+0# ']*'")


def _reject_grouping_flag_or_raise(name, fmt_str):
    """Raise ``TypingError`` if ``fmt_str`` uses the ``'`` flag, which groups the digits of a number by thousands.

    POSIX defines the flag, and glibc implements it, but C99 and Python's ``%`` operator lack it, so honoring it would
    make jitted code and Python print differently. ``_FMT_SPEC_RE`` does not recognize the flag either: a ``%'d``
    conversion would not be counted, and the conversion after it would read an argument slot that the call never filled.
    The result is indeterminate output, a leak of other memory, or a crash, which is the hazard that
    ``_validate_format_vs_args`` exists to prevent. The flag is therefore refused in both modes, as ``%n`` is.
    """
    stripped = fmt_str.replace('%%', '\x00\x00')
    if _GROUPING_FLAG_RE.search(stripped):
        raise TypingError(
            f"{name}: the ' (thousands-grouping) flag in format string "
            f"{fmt_str!r} is not supported (glibc-only, absent from C99 and "
            f"Python's % operator). Remove it or group the digits yourself."
        )


_FMT_SPEC_RE = re.compile(
    r'%[-+0# ]*'                          # flags
    r'(?:\*|[0-9]*)'                       # field width (number or '*')
    r'(?:\.(?:\*|[0-9]*))?'               # .precision (number or '*')
    r'(?:hh|ll|h|l|L|j|z|t|q|I32|I64)?'   # length modifier
    r'([diouxXeEfFgGaAcsp])'              # conversion (group 1); 'n' is rejected
)                                          # separately, '%%' is masked first

_CONV_INT = set("diouxXc")
_CONV_FLOAT = set("eEfFgGaA")


def _validate_format_vs_args(name, fmt_str, args_ty):
    """Raise ``TypingError`` unless the conversions of the literal format match the argument types.

    The check is the one that GCC's ``-Wformat`` performs. The number of arguments must equal the number of conversions,
    plus one integer for each ``*`` width or precision, which comes before the conversion's own argument. Each
    conversion must get the kind of argument it takes. ``%d``, ``%i``, ``%u``, ``%o``, ``%x``, ``%X`` and ``%c`` take an
    integer or a ``bool``, and ``%e``, ``%f``, ``%g``, ``%a`` and their capitals take a float. ``%s`` takes a unicode
    string or a pointer, and ``%p`` a pointer, which is an ``intp`` or a ``uintp``.

    Without the check, ``printf('%d %d', x)`` would read an argument slot that the call never filled, and the result
    would be indeterminate output, a leak of other memory or a crash. ``printf('%s', 5)`` would make C read the integer
    as a ``char *``, and ``printf('%d', 3.5)`` would print a wrong value, because C would read an integer where the call
    passed a double. numba's ``intp`` is ``int64``, so the check cannot distinguish a pointer from an ``int64`` given
    for ``%s`` or ``%p``; it refuses narrower integers, floats and booleans there. The intrinsics run the check on the
    types after the overload has replaced each ``str`` with its address, so a ``str`` also passes for an integer
    conversion or ``%p``.
    """
    stripped = fmt_str.replace('%%', '\x00\x00')
    expected = []
    for m in _FMT_SPEC_RE.finditer(stripped):
        expected.extend(['int'] * m.group(0).count('*'))
        conv = m.group(1)
        if conv in _CONV_INT:
            expected.append('int')
        elif conv in _CONV_FLOAT:
            expected.append('float')
        elif conv == 's':
            expected.append('str')
        else:  # 'p'
            expected.append('ptr')
    arg_types = tuple(args_ty)
    if len(expected) != len(arg_types):
        raise TypingError(
            f"{name}: format {fmt_str!r} has {len(expected)} conversion "
            f"specifier(s) but {len(arg_types)} argument(s) were passed"
        )
    for i, (cls, ty) in enumerate(zip(expected, arg_types)):
        uty = unliteral(ty)
        if cls == 'int':
            ok = isinstance(uty, (Integer, Boolean))
        elif cls == 'float':
            ok = isinstance(uty, Float)
        elif cls == 'str':
            ok = isinstance(uty, UnicodeType) or uty in (intp, uintp)
        else:  # 'ptr'
            ok = uty in (intp, uintp)
        if not ok:
            raise TypingError(
                f"{name}: format specifier #{i + 1} in {fmt_str!r} expects "
                f"{cls}, but arg {i} has type {ty!r}"
            )


def _unpack_args_tuple(builder, args_ty, args_pack):
    """Return a ``(numba type, LLVM value)`` pair for each argument in the LLVM tuple ``args_pack``."""
    arg_types = tuple(args_ty)
    return [
        (arg_types[i], builder.extract_value(args_pack, i))
        for i in range(len(arg_types))
    ]


def _emit_variadic_call(builder, symbol, fmt_str, leading_vals, args_ty, args_pack, *, leading_tys):
    """Emit a call of the variadic C function ``symbol`` and return its ``int32`` result.

    The call passes ``leading_vals``, then the format, then the arguments in ``args_pack``, each promoted by
    ``_promote_for_varargs``. ``leading_tys`` gives the LLVM types of the leading arguments, those before the format in
    the C signature, such as a ``FILE *``, a ``char *`` or a ``size_t``. The caller states them, rather than this
    function reading the types of ``leading_vals``, so that each call site declares the C signature itself; for
    ``snprintf``'s ``size_t``, it derives the type with ``intp_ll_type(context)``, as numba lowers ``intp``.
    """
    i8p = llir.IntType(8).as_pointer()
    i32_ll = llir.IntType(32)
    mod = builder.module
    # The format becomes a constant in the IR, which is why numba must know it when it compiles the call. It is
    # encoded as UTF-8, and C copies every byte outside a conversion unchanged, so non-ASCII text reaches the stream
    # or the buffer as UTF-8.
    fmt_bytes = cgutils.make_bytearray((fmt_str + '\x00').encode('utf-8'))
    # cgutils.global_constant gives the global internal linkage and a name that module.get_unique_name makes unique
    # in the module, such as printf_format and printf_format.1. An internal global is private to its module, and
    # LLVM's linker renames such globals when it merges modules into the JIT. Each call site, with the same format or
    # another, therefore gets a global of its own.
    global_fmt = cgutils.global_constant(mod, f"{symbol}_format", fmt_bytes)
    fmt_p = builder.bitcast(global_fmt, i8p)
    unpacked = _unpack_args_tuple(builder, args_ty, args_pack)
    promoted = [_promote_for_varargs(builder, t, v) for t, v in unpacked]
    fn_ty = llir.FunctionType(i32_ll, list(leading_tys) + [i8p], var_arg=True)
    fn = get_or_insert_function(mod, fn_ty, symbol)
    return builder.call(fn, list(leading_vals) + [fmt_p] + promoted)


# ---------------------------------------------------------------------------
# Private intrinsics, which emit the C calls in jitted code
# ---------------------------------------------------------------------------


@intrinsic(prefer_literal=True)
def _printf_intrinsic(typingctx, fmt_ty, args_ty):
    """Emit C's ``printf`` call for :func:`printf`, with the literal format and the arguments in one tuple.

    The intrinsic repeats the overload's checks and adds the check of the conversions against the arguments.
    """
    fmt_str = extract_literal_str("printf", fmt_ty, field="format string")
    _reject_percent_n_or_raise("printf", fmt_str)
    _reject_grouping_flag_or_raise("printf", fmt_str)
    if not isinstance(args_ty, BaseTuple):
        raise TypingError(f"printf: args must be a tuple, got {args_ty!r}")
    for i, ty in enumerate(tuple(args_ty)):
        _validate_writer_arg_type("printf", i, ty)
    _validate_format_vs_args("printf", fmt_str, args_ty)

    def codegen(context, builder, sig, llvm_args):
        _, args_pack = llvm_args
        return _emit_variadic_call(
            builder, "printf", fmt_str, [], args_ty, args_pack,
            leading_tys=[])

    return int32(fmt_ty, args_ty), codegen


@intrinsic(prefer_literal=True)
def _fprintf_intrinsic(typingctx, fp_ty, fmt_ty, args_ty):
    """Emit C's ``fprintf`` call for :func:`fprintf`, with the stream, the literal format and the arguments in a tuple.

    The intrinsic repeats the overload's checks and adds the check of the conversions against the arguments.
    """
    fmt_str = extract_literal_str("fprintf", fmt_ty, field="format string")
    _reject_percent_n_or_raise("fprintf", fmt_str)
    _reject_grouping_flag_or_raise("fprintf", fmt_str)
    if not isinstance(args_ty, BaseTuple):
        raise TypingError(f"fprintf: args must be a tuple, got {args_ty!r}")
    if fp_ty != intp:
        raise TypingError(
            f"fprintf: fp must be intp (FILE* as pointer-as-int), got {fp_ty!r}"
        )
    for i, ty in enumerate(tuple(args_ty)):
        _validate_writer_arg_type("fprintf", i, ty)
    _validate_format_vs_args("fprintf", fmt_str, args_ty)

    def codegen(context, builder, sig, llvm_args):
        i8p = llir.IntType(8).as_pointer()
        fp_int, _, args_pack = llvm_args
        fp_ptr = builder.inttoptr(fp_int, i8p)
        return _emit_variadic_call(
            builder, "fprintf", fmt_str, [fp_ptr], args_ty, args_pack,
            leading_tys=[i8p])

    return int32(fp_ty, fmt_ty, args_ty), codegen


@intrinsic(prefer_literal=True)
def _snprintf_intrinsic(typingctx, buf_ty, size_ty, fmt_ty, args_ty):
    """Emit C's ``snprintf`` call, ``_snprintf`` on Windows, for :func:`snprintf`, with the arguments in a tuple.

    The intrinsic repeats the overload's checks and adds the check of the conversions against the arguments.
    """
    fmt_str = extract_literal_str("snprintf", fmt_ty, field="format string")
    _reject_percent_n_or_raise("snprintf", fmt_str)
    _reject_grouping_flag_or_raise("snprintf", fmt_str)
    if not isinstance(args_ty, BaseTuple):
        raise TypingError(f"snprintf: args must be a tuple, got {args_ty!r}")
    if buf_ty != intp:
        raise TypingError(
            f"snprintf: buf must be intp (pointer-as-int), got {buf_ty!r}"
        )
    if size_ty != intp:
        raise TypingError(
            f"snprintf: size must be intp (size_t-as-int), got {size_ty!r}"
        )
    for i, ty in enumerate(tuple(args_ty)):
        _validate_writer_arg_type("snprintf", i, ty)
    _validate_format_vs_args("snprintf", fmt_str, args_ty)

    def codegen(context, builder, sig, llvm_args):
        i8p = llir.IntType(8).as_pointer()
        buf_int, size_val, _, args_pack = llvm_args
        buf_ptr = builder.inttoptr(buf_int, i8p)
        # size_t is as wide as a pointer on every 64-bit platform that numbox supports. The helper derives its LLVM
        # type from numba's intp, so the declared C signature follows the platform and matches the intp check above.
        return _emit_variadic_call(
            builder, _SNPRINTF_SYMBOL, fmt_str,
            [buf_ptr, size_val], args_ty, args_pack,
            leading_tys=[i8p, intp_ll_type(context)])

    return int32(buf_ty, size_ty, fmt_ty, args_ty), codegen


@intrinsic(prefer_literal=True)
def _sscanf_intrinsic(typingctx, buf_ty, fmt_ty, args_ty):
    """Emit C's ``sscanf`` call for :func:`sscanf`, with the text's address, the literal format and the addresses.

    Each argument is the ``intp`` address of the storage for one conversion. C receives the addresses unconverted,
    because C's default argument promotion leaves a pointer unchanged.
    """
    fmt_str = extract_literal_str("sscanf", fmt_ty, field="format string")
    if not isinstance(args_ty, BaseTuple):
        raise TypingError(f"sscanf: args must be a tuple, got {args_ty!r}")
    if buf_ty != intp:
        raise TypingError(
            f"sscanf: buf must be intp (pointer-as-int), got {buf_ty!r}"
        )
    for i, ty in enumerate(tuple(args_ty)):
        if ty != intp:
            raise TypingError(
                f"sscanf: args[{i}] must be intp (output pointer), got {ty!r}"
            )

    def codegen(context, builder, sig, llvm_args):
        i8p = llir.IntType(8).as_pointer()
        buf_int, _, args_pack = llvm_args
        buf_ptr = builder.inttoptr(buf_int, i8p)
        return _emit_variadic_call(
            builder, "sscanf", fmt_str, [buf_ptr], args_ty, args_pack,
            leading_tys=[i8p])

    return int32(buf_ty, fmt_ty, args_ty), codegen


# ---------------------------------------------------------------------------
# Helpers that build each overload's implementation, with str arguments passed as addresses
# ---------------------------------------------------------------------------


def _build_args_tuple_expr_from_starargs(arg_tys):
    """Return the Python source of the tuple of arguments that an overload's implementation passes to its intrinsic.

    The source indexes the implementation's ``args``, as in ``(args[0], args[1])``, and wraps a unicode argument in
    ``get_unicode_data_p``, so that C receives the address of a NUL-terminated string for ``%s``. numba requires the
    implementation to take ``*args`` when the overload's typing function takes ``*args``, so the source cannot name one
    parameter per argument.
    """
    n = len(arg_tys)
    if n == 0:
        return "()"
    parts = []
    for i, ty in enumerate(arg_tys):
        # A string literal, numba's StringLiteral, needs the conversion as much as a unicode_type value does.
        # StringLiteral derives from Literal, Dummy and Type, not from UnicodeType, so unliteral(ty) removes the literal
        # before the check.
        if isinstance(unliteral(ty), UnicodeType):
            parts.append(f"get_unicode_data_p(args[{i}])")
        else:
            parts.append(f"args[{i}]")
    inner = ", ".join(parts)
    return f"({inner},)" if n == 1 else f"({inner})"


def _build_overload_impl(name, fixed_params, args_tys, intrinsic_callable,
                         get_unicode_data_p):
    """Return an overload's implementation, compiled from generated source, that calls ``intrinsic_callable``.

    The implementation takes ``fixed_params`` and then ``*args``, the shape of the overload's typing function. It
    passes the fixed parameters and one tuple of the arguments to the intrinsic, with each unicode argument replaced by
    its address through ``get_unicode_data_p``. The intrinsics take the arguments as one tuple because numba's
    intrinsic typing function does not accept ``*args``.
    """
    args_tuple_expr = _build_args_tuple_expr_from_starargs(args_tys)
    sig_params = ", ".join(list(fixed_params) + ["*args"])
    intrinsic_args = ", ".join(list(fixed_params) + [args_tuple_expr])
    src = f"def impl({sig_params}):\n    return _intr({intrinsic_args})\n"
    ns = {"_intr": intrinsic_callable, "get_unicode_data_p": get_unicode_data_p}
    exec(compile(src, f"<{name}-overload-impl>", "exec"), ns)  # nosec B102 - JIT codegen of internal source
    return ns["impl"]


def _get_unicode_data_p_lazy():
    """Return :func:`numbox.utils.lowlevel.get_unicode_data_p`, which it imports when called."""
    # The import is deferred until numba types the first jitted call of one of the four functions, so importing this
    # module does not import numbox.utils.lowlevel.
    from numbox.utils.lowlevel import get_unicode_data_p
    return get_unicode_data_p


# ---------------------------------------------------------------------------
# The map from C stream addresses to Python streams, built at fprintf's first call under Python
# ---------------------------------------------------------------------------


_PY_STREAM_BY_FP = None


def _ensure_py_stream_cache():
    """Build ``_PY_STREAM_BY_FP``, the map from the three C streams' addresses to Python's streams, once.

    The map holds the ``sys.stdout``, ``sys.stderr`` and ``sys.stdin`` objects that are current at the first call, so a
    redirection installed after that call does not change where :func:`fprintf` writes under Python.
    """
    global _PY_STREAM_BY_FP
    if _PY_STREAM_BY_FP is not None:
        return
    # The import is deferred until the first call, so importing this module does not import and compile the stdio
    # bindings; under NUMBA_DISABLE_JIT=1 the import fails with AttributeError, because proxy needs numba's compiler.
    from numbox.core.bindings.stdio import stdout, stderr, stdin
    _PY_STREAM_BY_FP = {
        int(stdout()): sys.stdout,
        int(stderr()): sys.stderr,
        int(stdin()):  sys.stdin,
    }


# ---------------------------------------------------------------------------
# The public functions and their overloads
# ---------------------------------------------------------------------------


def _reject_percent_n_in_python(name, fmt):
    """Raise ``ValueError`` if ``fmt`` holds a ``%n`` conversion, as ``_reject_percent_n_or_raise`` does in jitted code.

    Python's ``%`` operator would also raise ``ValueError`` for ``%n``, an unsupported format character, but with a
    generic message. This check raises the error with the jitted path's message.
    """
    stripped = fmt.replace('%%', '\x00\x00')
    if _PERCENT_N_RE.search(stripped):
        raise ValueError(
            f"{name}: %n directive in format string {fmt!r} is not allowed "
            f"(writes byte-count-written through caller pointer; memory-"
            f"safety hazard). Use sscanf if you need %n's read-position "
            f"semantics."
        )


def _reject_grouping_flag_in_python(name, fmt):
    """Raise ``ValueError`` if ``fmt`` uses the ``'`` flag, as ``_reject_grouping_flag_or_raise`` does in jitted code.

    Python's ``%`` operator would also raise ``ValueError`` for ``%'d``, an unsupported format character, but with a
    generic message. This check raises the error with the jitted path's message.
    """
    stripped = fmt.replace('%%', '\x00\x00')
    if _GROUPING_FLAG_RE.search(stripped):
        raise ValueError(
            f"{name}: the ' (thousands-grouping) flag in format string "
            f"{fmt!r} is not supported (glibc-only, absent from C99 and "
            f"Python's % operator). Remove it or group the digits yourself."
        )


def printf(fmt, *args):
    """Format ``args`` by the C format string ``fmt``, write the text to stdout, and return the number of bytes written.

    C reference: `printf(3) <https://man7.org/linux/man-pages/man3/printf.3.html>`_.

    Under Python, ``fmt`` can be any ``str``. The function formats with Python's ``%`` operator, writes the text to the
    current ``sys.stdout`` and flushes it, and returns the length in bytes of the text in UTF-8. Python's ``%`` ignores
    the length modifiers ``h``, ``l`` and ``L`` and rejects the others, such as the ``ll`` of ``%lld``, so the function
    removes every length modifier before it formats. It also removes one that follows a literal ``%%``, so ``"%%lld"``
    prints ``%d`` under Python. ``%p`` works only in jitted code, because Python's ``%`` rejects it. ``%s`` given an
    address prints the number under Python, where C prints the string at that address.

    In jitted code, numba compiles the call into a call of C's ``printf``, which writes to C's stdout and returns an
    ``int32``. ``fmt`` must be a string that numba reads as a constant, such as a literal, and C receives it in UTF-8.
    numbox checks the conversions against the arguments when numba compiles the call, as GCC's ``-Wformat`` does, and
    raises ``TypingError`` for a mismatch. numbox passes a ``str`` argument to C as the address of its characters, which
    C reads as text only for an ASCII string. The check is applied to that address, so a ``str`` given to ``%d``
    compiles and prints the address. numbox passes every integer and ``bool`` to C as 64 bits and every float as a
    ``float64``, and C converts each value to the type that its conversion names, such as ``int`` for ``%d``. On
    Windows, C's ``long`` has 32 bits, so use ``%lld`` rather than ``%ld`` for an ``int64``. A width such as ``%-10s``
    counts bytes, not characters; to pad by characters, pad the string first, for example with ``str.ljust``.

    In both modes, ``%n`` and the ``'`` flag raise an error: ``TypingError`` in jitted code and ``ValueError`` under
    Python.
    """
    _reject_percent_n_in_python("printf", fmt)
    _reject_grouping_flag_in_python("printf", fmt)
    text = _python_fmt_compat(fmt) % args
    sys.stdout.write(text)
    sys.stdout.flush()
    return len(text.encode('utf-8'))


@overload(printf)
def _overload_printf(fmt, *args):
    """Return the jitted implementation of :func:`printf` for these argument types.

    The overload checks that the format is a literal without ``%n`` or the ``'`` flag, and that a conversion takes each
    argument's type. ``_printf_intrinsic`` then checks the conversions against the arguments.
    """
    fmt_str = extract_literal_str("printf", fmt, field="format string")
    _reject_percent_n_or_raise("printf", fmt_str)
    _reject_grouping_flag_or_raise("printf", fmt_str)
    for i, ty in enumerate(args):
        _validate_writer_arg_type("printf", i, ty)
    impl = _build_overload_impl(
        "printf", ["fmt"], args, _printf_intrinsic,
        _get_unicode_data_p_lazy(),
    )
    return impl


def fprintf(fp, fmt, *args):
    """Format ``args`` by ``fmt``, write the text to the stream ``fp``, and return the number of bytes written.

    C reference: `fprintf(3) <https://man7.org/linux/man-pages/man3/fprintf.3.html>`_.

    ``fp`` is a ``FILE *`` as an ``intp``, such as one that :func:`~numbox.core.bindings.stdio.stdout` or ``fopen``
    returns. The format and the arguments follow the rules of :func:`printf`, in both modes. In jitted code, numba
    compiles the call into a call of C's ``fprintf``, which accepts any ``FILE *``.

    Under Python, the function writes to ``sys.stdout``, ``sys.stderr`` or ``sys.stdin`` for the address of the matching
    C stream, and flushes it. It raises ``RuntimeError`` for any other ``FILE *``, such as one from ``fopen``; use
    Python's ``open`` and ``write`` for a file there. The function maps the three addresses to the ``sys`` objects that
    are current at its first call, so a redirection installed after that call does not change where it writes. With
    ``NUMBA_DISABLE_JIT=1``, the first call raises ``AttributeError``, because it imports the stdio bindings.
    """
    _reject_percent_n_in_python("fprintf", fmt)
    _reject_grouping_flag_in_python("fprintf", fmt)
    _ensure_py_stream_cache()
    py_stream = _PY_STREAM_BY_FP.get(int(fp))
    if py_stream is None:
        raise RuntimeError(
            f"fprintf in pure-Python mode only supports stdout / stderr / "
            f"stdin handles; got fp={int(fp):#x}. For arbitrary FILE* "
            f"(e.g. fopen) use @njit, or use Python's open() + write()."
        )
    text = _python_fmt_compat(fmt) % args
    py_stream.write(text)
    py_stream.flush()
    return len(text.encode('utf-8'))


@overload(fprintf)
def _overload_fprintf(fp, fmt, *args):
    """Return the jitted implementation of :func:`fprintf` for these argument types.

    The overload checks that ``fp`` is an ``intp``, that the format is a literal without ``%n`` or the ``'`` flag, and
    that a conversion takes each argument's type. ``_fprintf_intrinsic`` then checks the conversions against the
    arguments.
    """
    fmt_str = extract_literal_str("fprintf", fmt, field="format string")
    _reject_percent_n_or_raise("fprintf", fmt_str)
    _reject_grouping_flag_or_raise("fprintf", fmt_str)
    if fp != intp:
        raise TypingError(
            f"fprintf: fp must be intp (FILE* as pointer-as-int), got {fp!r}"
        )
    for i, ty in enumerate(args):
        _validate_writer_arg_type("fprintf", i, ty)
    impl = _build_overload_impl(
        "fprintf", ["fp", "fmt"], args, _fprintf_intrinsic,
        _get_unicode_data_p_lazy(),
    )
    return impl


def snprintf(buf_p, size, fmt, *args):
    """Format ``args`` by ``fmt`` into the ``size`` bytes at address ``buf_p``, and return the length of the whole text.

    C reference: `snprintf(3) <https://man7.org/linux/man-pages/man3/snprintf.3.html>`_.

    ``buf_p`` is the ``intp`` address of at least ``size`` bytes, such as ``array_data_p(buf)``, which the caller keeps
    alive during the call. The function writes at most ``size`` bytes, the terminating NUL included, and returns the
    length in bytes of the text without the NUL, so a result of ``size`` or more means that it truncated the text. The
    format and the arguments follow the rules of :func:`printf`.

    In jitted code, ``buf_p`` and ``size`` must be ``intp`` values: ``a.size``, ``np.intp(16)`` and an ``int64``
    argument of the jitted function work, but a literal such as ``16`` and an ``np.int32`` fail to compile with
    ``TypingError``. On Windows, jitted code calls C's `_snprintf
    <https://learn.microsoft.com/cpp/c-runtime-library/reference/snprintf-snprintf-snprintf-l-snwprintf-snwprintf-l>`__
    instead. It returns -1 when the text is longer than ``size``, and when the text is exactly ``size`` bytes long, it
    returns ``size`` and writes no NUL. The test ``n < 0 or n >= size`` covers both cases on every platform.

    Under Python, the function follows the C99 rule on every platform. ``ctypes.memmove`` writes the text, and it
    rejects a NumPy integer as ``buf_p``. ``array_data_p`` returns one under ``NUMBA_DISABLE_JIT=1``, so pass
    ``int(array_data_p(buf))`` there.
    """
    _reject_percent_n_in_python("snprintf", fmt)
    _reject_grouping_flag_in_python("snprintf", fmt)
    text_bytes = (_python_fmt_compat(fmt) % args).encode('utf-8')
    n_would_have = len(text_bytes)
    if size > 0:
        n_write = min(n_would_have, size - 1)
        # The NUL is appended after the slice, so a truncated copy still ends with it. Appending the NUL to the whole
        # text and copying n_write + 1 bytes would leave a truncated copy with no terminator.
        src = text_bytes[:n_write] + b'\x00'
        ctypes.memmove(buf_p, src, len(src))
    return n_would_have


@overload(snprintf)
def _overload_snprintf(buf_p, size, fmt, *args):
    """Return the jitted implementation of :func:`snprintf` for these argument types.

    The overload checks that ``buf_p`` and ``size`` are ``intp``, that the format is a literal without ``%n`` or the
    ``'`` flag, and that a conversion takes each argument's type. ``_snprintf_intrinsic`` then checks the conversions
    against the arguments.
    """
    fmt_str = extract_literal_str("snprintf", fmt, field="format string")
    _reject_percent_n_or_raise("snprintf", fmt_str)
    _reject_grouping_flag_or_raise("snprintf", fmt_str)
    if buf_p != intp:
        raise TypingError(
            f"snprintf: buf must be intp (pointer-as-int), got {buf_p!r}"
        )
    # numba types an overload's call first with the arguments' non-literal types and then with their literal types.
    # The first typing fails, because the format must be a literal, so this check receives a literal size such as 16
    # as Literal[int](16), not as intp, and refuses it. It refuses an int32 as well, and it accepts a.size, np.intp(16)
    # and an int64 argument.
    if size != intp:
        raise TypingError(
            f"snprintf: size must be intp (size_t-as-int), got {size!r}"
        )
    for i, ty in enumerate(args):
        _validate_writer_arg_type("snprintf", i, ty)
    impl = _build_overload_impl(
        "snprintf", ["buf_p", "size", "fmt"], args, _snprintf_intrinsic,
        _get_unicode_data_p_lazy(),
    )
    return impl


def sscanf(buf, fmt, *args):
    """Read values by ``fmt`` from the NUL-terminated text at address ``buf`` into storage, in jitted code only.

    C reference: `sscanf(3) <https://man7.org/linux/man-pages/man3/sscanf.3.html>`_.

    The result is the number of values stored, or -1, C's ``EOF``, if the text ends before the first conversion. ``buf``
    is the ``intp`` address of the text, such as ``get_unicode_data_p(text)`` for an ASCII ``str``. Each argument after
    ``fmt`` is the ``intp`` address of the storage for one conversion, such as ``array_data_p`` of a one-element array
    of the matching dtype. numbox checks only that each of them is an ``intp``. It does not compare the format with the
    arguments, so the caller must make them match in number and in size. ``fmt`` must be a string that numba reads as a
    constant, as for :func:`printf`. Under Python, the function raises ``NotImplementedError``; use ``int()``,
    ``float()`` or ``re`` to parse the text there.
    """
    raise NotImplementedError(
        "sscanf is @njit-only; wrap the call in @njit, or use Python's "
        "int() / float() / re for pure-Python parsing"
    )


@overload(sscanf)
def _overload_sscanf(buf, fmt, *args):
    """Return the jitted implementation of :func:`sscanf` for these argument types.

    The overload checks that the format is a literal and that ``buf`` and every argument are ``intp``. It does not
    check the format against the arguments.
    """
    extract_literal_str("sscanf", fmt, field="format string")  # raises unless the format is a literal str
    if buf != intp:
        raise TypingError(
            f"sscanf: buf must be intp (pointer-as-int), got {buf!r}"
        )
    # Each argument is the address of storage for one conversion, so it must be an intp. A str is not accepted here,
    # so the shared builder below wraps no argument in get_unicode_data_p, and C receives each address unchanged.
    for i, ty in enumerate(args):
        if ty != intp:
            raise TypingError(
                f"sscanf: args[{i}] must be intp (output pointer), got {ty!r}"
            )
    impl = _build_overload_impl(
        "sscanf", ["buf", "fmt"], args, _sscanf_intrinsic,
        _get_unicode_data_p_lazy(),  # unused, because no argument is a unicode string
    )
    return impl
