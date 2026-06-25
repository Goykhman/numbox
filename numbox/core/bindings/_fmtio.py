"""Variadic formatted I/O — ``printf``, ``fprintf``, ``snprintf``, ``sscanf``
callable from BOTH plain Python and ``@njit`` code.

The public bindings are regular Python functions; numba dispatches to a
private ``@intrinsic`` codegen path via ``@overload`` when called inside
``@njit``. Same source runs unchanged in either mode, matching numba's
own convention for builtins like ``print`` and ``range``::

    def debug_kernel(x, label):
        printf("step %d: %s\\n", x, label)
        fflush(stdout())
        return x * 2

    debug_kernel(7, "before")        # pure Python: writes via sys.stdout
    njit(debug_kernel)(7, "before")  # jitted: writes via libc printf

Call convention
---------------

User-facing API is C-like — positional ``*args`` after the format string;
no tuple wrapper at the call site. Internally the @overload path bundles
the args into a tuple before calling the private ``_xxx_intrinsic`` (the
intrinsic itself still uses the tuple-as-args shape because numba's
``@intrinsic`` typing function doesn't accept Python-level ``*args``)::

    printf("x = %d, ratio = %.3f\\n", n, ratio)
    fprintf(stderr(), "warning: %s\\n", msg)
    snprintf(array_data_p(buf), buf.size, "[%d:%d]", lo, hi)
    sscanf(buf_p, "%d %lf", array_data_p(n_out), array_data_p(x_out))
    printf("no args here\\n")

Format string must be a literal in @njit
----------------------------------------

Required so the format string can be embedded as an IR global constant —
the same constraint a C compiler operates under when emitting a
format-checked printf call. A runtime-built ``unicode`` raises a clean
``TypingError`` at call typing time. In pure-Python mode the format
string can of course be any str.

Because the format is a literal, the ``@njit`` writers also cross-check it
against the arguments at typing time (like a C compiler's ``-Wformat``): the
conversion-specifier count must equal the argument count, and each specifier's
class must match its arg (``%d``/``%x``/… → Integer/Boolean, ``%f``/``%g``/… →
Float, ``%s`` → unicode or an ``intp``/``uintp`` pointer, ``%p`` → ``intp``/``uintp``). A mismatch
raises ``TypingError`` instead of reading an unpushed varargs slot, dereferencing
an int as ``char*``, or reading a GP register for an SSE-passed float. Caveat:
numba's ``intp`` is ``int64``, so a pointer-width int passed for ``%s``/``%p``
cannot be told apart from a real pointer (narrower ints, floats and booleans
are still rejected).

Format string encoding: UTF-8
-----------------------------

Non-ASCII codepoints in the literal are encoded as UTF-8 byte sequences
and embedded into the IR global. printf treats every non-``%`` byte as
opaque pass-through, so the bytes flow through libc to stdout / FILE\\* /
the snprintf buffer unmodified. Modern terminals, files, and Windows
10+ consoles all expect UTF-8.

.. note::
   ``%-Ns`` width is byte-counted by printf in every libc, so non-ASCII
   output won't right-pad to a codepoint count. That's printf's contract.
   Pad in numba-side string formatting (``f"{s:<10}"``) before passing
   through ``%s`` if codepoint-counted widths matter.

Cross-mode caveats
------------------

1. **Length modifiers in format strings.** ``%lld``, ``%ld``, ``%lf``,
   ``%hd``, etc. are valid in C printf but rejected by Python's ``%``
   operator. The pure-Python impls strip length modifiers via a regex
   before formatting (``%lld`` → ``%d``, ``%.3lf`` → ``%.3f``). The
   stripped form produces identical output for typical values because
   Python ints / floats carry the same width independent of the spec.

2. **C-ABI auto-promotion of integer args to 64-bit.** The @njit impl
   widens every integer variadic arg to 64-bit before the libc call
   (sext / zext as appropriate). Without this, a user writing
   ``printf("%lld", np.int32(7))`` in @njit would have libc read 8 bytes
   from a 4-byte source — register garbage in the high bits. With the
   widening, ``%lld`` against int32 / int16 / int8 / bool works
   correctly. Diverges from C ABI (C doesn't promote int to long long)
   but matches user expectations and the pure-Python ``%`` behavior.

3. **String args + ``%s``.** Pure-Python's ``%`` handles strings
   natively. The @njit @overload auto-converts ``unicode_type`` args
   via ``get_unicode_data_p`` so libc sees a NUL-terminated C string.
   Users no longer need to call ``get_unicode_data_p`` themselves at
   the call site.

4. **``%ld`` on Win64 (LLP64).** ``long`` is 4 bytes on Win64 but 8 on
   LP64; ``%ld`` against int64 truncates the high 32 bits on Win64 in
   @njit. Pure-Python mode hides this because Python's ``%`` ignores
   length modifiers. Prefer ``%lld`` + int64 for portable 8-byte width.

5. **``snprintf`` truncation rc on Windows.** Pure-Python and Linux/macOS
   @njit follow C99 semantics (return would-have-written count).
   Windows @njit targets MSVCRT ``_snprintf`` (returns ``-1`` on
   truncation, no NUL-term guarantee). Portable check that works on
   every platform: ``(rc < 0) or (rc >= size)``.

6. **``fprintf`` to non-stdio FILE\\* in pure-Python.** The Python impl
   routes ``stdout()`` / ``stderr()`` / ``stdin()`` handles to the
   corresponding ``sys.*`` streams via an address cache. ``fopen``-
   returned FILE\\* values aren't dereferenceable from Python without
   a ctypes call, so they raise a clear error in pure-Python mode
   (use ``open()`` + ``f.write()`` for Python-side file I/O).

7. **``sscanf`` is @njit-only.** Pure-Python implementations of
   sscanf-style parsing are usually better served by ``int()`` /
   ``float()`` / ``re``; calling from pure-Python raises
   ``NotImplementedError``.

References
----------

- `printf(3) <https://man7.org/linux/man-pages/man3/printf.3.html>`_
- `fprintf(3) <https://man7.org/linux/man-pages/man3/fprintf.3.html>`_
- `snprintf(3) <https://man7.org/linux/man-pages/man3/snprintf.3.html>`_
- `sscanf(3) <https://man7.org/linux/man-pages/man3/sscanf.3.html>`_
- `Microsoft _snprintf
  <https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/snprintf-snprintf-snprintf-l-snwprintf-snwprintf-l>`_
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


# Windows MSVCRT exports "_snprintf" with non-C99 truncation semantics
# (returns -1 on truncation, no NUL guarantee). UCRT's C99-compliant
# "snprintf" is a header-only inline over __stdio_common_vsnprintf that
# isn't directly linkable in the simple C99 calling shape — declaring
# `i32 @snprintf(...)` in LLVM IR and letting the JIT linker resolve it
# crashes with an access violation. So on Windows the @njit binding
# deliberately targets MSVCRT (see snprintf module-docstring section on
# cross-mode caveats); pure-Python uses POSIX/C99 semantics universally.
_SNPRINTF_SYMBOL = "_snprintf" if platform_ == "Windows" else "snprintf"


# C-printf length modifiers: 'hh', 'h', 'l', 'll', 'L'. Python's % operator
# rejects them with `ValueError: unsupported format character`. Strip them
# before pure-Python formatting; the stripped %d / %f produces equivalent
# output for typical values (Python's % uses the value's natural width).
_LENGTH_MODIFIER_RE = re.compile(
    r'%([-+0# ]*[0-9*]*\.?[0-9*]*)(hh|ll|h|l|L|j|z|t|q|I32|I64)([diouxXfFeEgGaAcsp])'
)


def _python_fmt_compat(fmt):
    """Strip C printf length modifiers so Python's % accepts the format."""
    return _LENGTH_MODIFIER_RE.sub(r'%\1\3', fmt)


# ---------------------------------------------------------------------------
# Helpers for the @intrinsic codegen layer (private)
# ---------------------------------------------------------------------------


def _promote_for_varargs(builder, arg_ty, arg_val):
    """C-ABI promotion + opportunistic widening of all integer args to 64-bit.

    - ``float32`` → ``float64`` (fpext)
    - ``bool`` → ``int64`` (zext)
    - any signed ``Integer`` of width < 64 → ``int64`` (sext)
    - any unsigned ``Integer`` of width < 64 → ``int64`` (zext)
    - 64-bit ints, doubles, pointers — pass through

    Widening to 64-bit (rather than just int32 per strict C ABI) is a
    deliberate choice: it makes ``%lld + int32`` work in @njit, which
    matches pure-Python's behavior (Python ignores length modifiers and
    uses the value's natural width). The cost is a single LLVM sext /
    zext / fpext per arg — free at runtime.

    Arg-type validation is done at typing time via
    ``_validate_writer_arg_type``; the ``else: raise`` here is
    defense-in-depth — should typing-layer validation fail to filter
    everything (e.g. a future numba changes how tuples flatten), the
    codegen path stops rather than dropping garbage into libc's
    variadic call.
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
    raise TypingError(
        f"variadic arg of type {arg_ty!r} is not supported by printf-family "
        f"bindings; allowed: Float, Integer, Boolean, or intp pointer"
    )


def _validate_writer_arg_type(name, idx, ty):
    """Raise ``TypingError`` unless ``ty`` is a scalar type supported by
    ``printf`` / ``fprintf`` / ``snprintf`` — Float, Integer (incl. intp
    for raw pointers), Boolean, or a unicode type (which the @overload
    layer auto-converts via ``get_unicode_data_p``).

    Without this guard, ``_promote_for_varargs`` would silently
    ``return arg_val`` for unsupported types, dropping numpy arrays,
    complex numbers, tuples, etc. directly into libc's variadic call as
    LLVM aggregates — silent miscompilation, not a clean error.
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
    """Raise ``TypingError`` if ``fmt_str`` contains a ``%n`` directive
    (with or without flags / width / precision / length modifier).

    ``%n`` causes printf to write the byte-count-written-so-far through a
    caller-supplied ``int*`` pointer. Allowing it would (a) be a memory
    write through an arg that pure-Python's ``%`` operator rejects with
    ``ValueError`` — breaking dual-mode equivalence — and (b) be a memory-
    safety hazard widely flagged by static analyzers and disabled by
    glibc's ``_FORTIFY_SOURCE`` for writable format strings.

    ``%%n`` (a literal ``%`` followed by ``n``) is allowed: the regex
    operates on the format string after stripping ``%%`` pairs to a
    sentinel that the directive matcher cannot land on.
    """
    stripped = fmt_str.replace('%%', '\x00\x00')
    if _PERCENT_N_RE.search(stripped):
        raise TypingError(
            f"{name}: %n directive in format string {fmt_str!r} is not "
            f"allowed (writes byte-count-written through caller pointer; "
            f"memory-safety hazard, also diverges from pure-Python behavior). "
            f"Use sscanf if you need %n's read-position semantics."
        )


# A directive whose flag run contains the glibc ``'`` (thousands-grouping)
# flag. Operates on the ``%%``-masked format string.
_GROUPING_FLAG_RE = re.compile(r"%[-+0# ']*'")


def _reject_grouping_flag_or_raise(name, fmt_str):
    """Raise ``TypingError`` if ``fmt_str`` uses the glibc ``'`` grouping flag.

    ``'`` (thousands grouping) is a legal glibc printf flag but is absent from
    C99 and unsupported by Python's ``%`` operator, so honoring it would
    diverge between the @njit and pure-Python paths. It is also not recognized
    by ``_FMT_SPEC_RE``: a ``%'d`` directive would not match at all, so the
    specifier count would be silently under-counted and a following directive
    would read an unpushed varargs slot (garbage / info-disclosure / crash) —
    the exact hazard ``_validate_format_vs_args`` exists to prevent. Reject it
    cleanly in both modes, mirroring ``%n`` handling.
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
    """Cross-check the literal format's conversion specifiers against the arg
    types at typing time (what a C compiler's ``-Wformat`` does):

    - the specifier count must equal the argument count, and
    - each specifier's class must match its arg
      (``%d/%i/%u/%o/%x/%X/%c`` → Integer/Boolean; ``%e/%f/%g/...`` → Float;
      ``%s`` → unicode or an ``intp``/``uintp`` pointer; ``%p`` → ``intp``/``uintp``).

    Each ``*`` dynamic width/precision consumes a preceding Integer arg.

    Without this, ``printf('%d %d', x)`` reads an unpushed varargs slot
    (garbage / info-disclosure / crash), ``printf('%s', 5)`` makes libc
    dereference an int as ``char*``, and ``printf('%d', 3.5)`` reads a GP
    register for an SSE-passed float (silent wrong result). Note ``intp`` is
    ``int64`` in numba, so a pointer-width int passed for ``%s``/``%p`` cannot
    be distinguished from a real pointer; narrower ints, floats and booleans
    for those specifiers ARE rejected.
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
    """Extract individual LLVM values from a tuple-of-args LLVM aggregate."""
    arg_types = tuple(args_ty)
    return [
        (arg_types[i], builder.extract_value(args_pack, i))
        for i in range(len(arg_types))
    ]


def _emit_variadic_call(builder, symbol, fmt_str, leading_vals, args_ty, args_pack, *, leading_tys):
    """Emit IR for ``symbol(*leading_vals, fmt_p, *promoted_args) -> i32``.

    ``leading_tys`` is the explicit list of LLVM types for the leading
    positional args (size_t, FILE*, char*, etc. — anything that precedes
    the format string in the libc signature). Callers pass it so the
    function-type declaration documents the libc ABI contract at the
    codegen call site, rather than implicitly reading
    ``[v.type for v in leading_vals]`` and relying on numba's lowering to
    produce the right LLVM types. The explicit form is what's needed for
    size_t in snprintf — derive via ``intp_ll_type(context)`` to keep
    consistency with numba's intp lowering API.
    """
    i8p = llir.IntType(8).as_pointer()
    i32_ll = llir.IntType(32)
    mod = builder.module
    fmt_bytes = cgutils.make_bytearray((fmt_str + '\x00').encode('utf-8'))
    # cgutils.global_constant uses linkage='internal' and routes through
    # module.get_unique_name → scope.deduplicate, which auto-suffixes when
    # the same name is re-used within a module (printf_format,
    # printf_format.1, ...). Across modules, internal-linkage globals are
    # module-private; LLVM's linker further renames on merge into the
    # shared MCJIT engine. So multiple call sites — same or distinct
    # format strings — each get their own deduplicated global.
    global_fmt = cgutils.global_constant(mod, f"{symbol}_format", fmt_bytes)
    fmt_p = builder.bitcast(global_fmt, i8p)
    unpacked = _unpack_args_tuple(builder, args_ty, args_pack)
    promoted = [_promote_for_varargs(builder, t, v) for t, v in unpacked]
    fn_ty = llir.FunctionType(i32_ll, list(leading_tys) + [i8p], var_arg=True)
    fn = get_or_insert_function(mod, fn_ty, symbol)
    return builder.call(fn, list(leading_vals) + [fmt_p] + promoted)


# ---------------------------------------------------------------------------
# Private @intrinsics — the @njit codegen path
# ---------------------------------------------------------------------------


@intrinsic(prefer_literal=True)
def _printf_intrinsic(typingctx, fmt_ty, args_ty):
    """libc printf via a tuple-of-args. Internal; user code calls printf()."""
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
    """libc fprintf via a tuple-of-args. Internal; user code calls fprintf()."""
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
    """libc snprintf via a tuple-of-args. Internal; user code calls snprintf()."""
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
        # size_t is pointer-width on all current 64-bit platforms; derive
        # the LLVM type from numba's intp via the shared helper so the
        # libc snprintf signature stays correct under platform changes and
        # matches the typing-time `size_ty != intp` guard above.
        return _emit_variadic_call(
            builder, _SNPRINTF_SYMBOL, fmt_str,
            [buf_ptr, size_val], args_ty, args_pack,
            leading_tys=[i8p, intp_ll_type(context)])

    return int32(buf_ty, size_ty, fmt_ty, args_ty), codegen


@intrinsic(prefer_literal=True)
def _sscanf_intrinsic(typingctx, buf_ty, fmt_ty, args_ty):
    """libc sscanf via a tuple-of-args. Internal; user code calls sscanf().

    Args are intp output pointers; no default-arg promotion applies
    (pointers don't promote). See sscanf() for the caller-facing contract.
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
# @overload helper: build an impl source string with str args auto-converted
# ---------------------------------------------------------------------------


def _build_args_tuple_expr_from_starargs(arg_tys):
    """Build a Python source fragment ``(...)`` that constructs the args
    tuple for the intrinsic call, indexing into the impl's ``*args`` and
    auto-converting any ``UnicodeType`` arg via ``get_unicode_data_p``
    so libc sees a NUL-terminated C string for ``%s``.

    Numba requires the @overload's impl signature to match the typing
    signature shape exactly — ``*args`` in typing must be ``*args`` in
    impl. So we cannot expand per-arity explicit parameters; we have
    to index into the ``args`` tuple from inside the impl.
    """
    n = len(arg_tys)
    if n == 0:
        return "()"
    parts = []
    for i, ty in enumerate(arg_tys):
        # Both numba's runtime ``unicode_type`` and the compile-time
        # ``Literal[str]`` (StringLiteral) need conversion. StringLiteral
        # is NOT a UnicodeType subclass directly — its MRO is
        # ``StringLiteral → Literal → Dummy → Type`` — so we use
        # ``unliteral(ty)`` to strip any Literal wrapping before the check.
        if isinstance(unliteral(ty), UnicodeType):
            parts.append(f"get_unicode_data_p(args[{i}])")
        else:
            parts.append(f"args[{i}]")
    inner = ", ".join(parts)
    return f"({inner},)" if n == 1 else f"({inner})"


def _build_overload_impl(name, fixed_params, args_tys, intrinsic_callable,
                         get_unicode_data_p):
    """Build an impl function via exec'd source that:

    - Takes ``(fixed_params..., *args)`` — matching the typing-function shape
    - Bundles ``*args`` into a tuple via index expressions, auto-converting
      ``UnicodeType`` args via ``get_unicode_data_p``
    - Calls the underlying intrinsic with the bundled tuple
    """
    args_tuple_expr = _build_args_tuple_expr_from_starargs(args_tys)
    sig_params = ", ".join(list(fixed_params) + ["*args"])
    intrinsic_args = ", ".join(list(fixed_params) + [args_tuple_expr])
    src = f"def impl({sig_params}):\n    return _intr({intrinsic_args})\n"
    ns = {"_intr": intrinsic_callable, "get_unicode_data_p": get_unicode_data_p}
    exec(compile(src, f"<{name}-overload-impl>", "exec"), ns)  # nosec B102 - JIT codegen of internal source
    return ns["impl"]


# Lazy import to avoid circular dependency at module load
def _get_unicode_data_p_lazy():
    from numbox.utils.lowlevel import get_unicode_data_p
    return get_unicode_data_p


# ---------------------------------------------------------------------------
# Pure-Python stdio-handle address cache (lazy init for fprintf routing)
# ---------------------------------------------------------------------------


_PY_STREAM_BY_FP = None


def _ensure_py_stream_cache():
    global _PY_STREAM_BY_FP
    if _PY_STREAM_BY_FP is not None:
        return
    # Defer the imports to avoid a circular dep at module load — _fmtio is
    # imported AFTER _stdio by numbox.core.bindings.__init__, but doing
    # the calls here at first use guarantees the bindings are ready.
    from numbox.core.bindings import stdout, stderr, stdin
    _PY_STREAM_BY_FP = {
        int(stdout()): sys.stdout,
        int(stderr()): sys.stderr,
        int(stdin()):  sys.stdin,
    }


# ---------------------------------------------------------------------------
# Public Python-callable wrappers + @overload registrations
# ---------------------------------------------------------------------------


def _reject_percent_n_in_python(name, fmt):
    """Pure-Python equivalent of ``_reject_percent_n_or_raise``: raise
    ``ValueError`` if ``%n`` appears in ``fmt``. Python's ``%`` operator
    would itself raise on ``%n`` (it's an unsupported format character),
    but the message is generic; this gives users the same clear error
    message in both Python and @njit modes.
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
    """Pure-Python equivalent of ``_reject_grouping_flag_or_raise``: raise
    ``ValueError`` if the glibc ``'`` grouping flag appears in ``fmt``.
    Python's ``%`` operator would itself raise on ``%'d`` (unsupported
    format character), but the message is generic; this gives users the
    same clear error in both Python and @njit modes.
    """
    stripped = fmt.replace('%%', '\x00\x00')
    if _GROUPING_FLAG_RE.search(stripped):
        raise ValueError(
            f"{name}: the ' (thousands-grouping) flag in format string "
            f"{fmt!r} is not supported (glibc-only, absent from C99 and "
            f"Python's % operator). Remove it or group the digits yourself."
        )


def printf(fmt, *args):
    """C-style ``printf(fmt, *args)`` — dual-mode (plain Python AND @njit).

    From plain Python: writes to ``sys.stdout`` via ``str.__mod__`` after
    stripping C length modifiers (``%lld`` → ``%d``, etc.).

    From @njit: ``@overload`` below routes to the private ``_printf_intrinsic``
    after auto-converting any ``unicode_type`` args via ``get_unicode_data_p``
    so libc ``%s`` receives a NUL-terminated C string. Format string must be
    a literal in @njit (see module docstring for caveats).

    ``%n`` is rejected in both modes (see ``_reject_percent_n_or_raise``).

    Returns the number of bytes written (or written-equivalent), as int32.
    """
    _reject_percent_n_in_python("printf", fmt)
    _reject_grouping_flag_in_python("printf", fmt)
    text = _python_fmt_compat(fmt) % args
    sys.stdout.write(text)
    sys.stdout.flush()
    return len(text.encode('utf-8'))


@overload(printf)
def _overload_printf(fmt, *args):
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
    """C-style ``fprintf(fp, fmt, *args)`` — dual-mode.

    ``fp`` is a FILE\\* as ``intp`` (from ``stdout()`` / ``stderr()`` /
    ``stdin()`` or ``fopen()``).

    From plain Python: routes ``stdout()`` / ``stderr()`` / ``stdin()``
    handles to the corresponding ``sys.*`` streams via an address cache.
    Arbitrary FILE\\* values (e.g. ``fopen``-returned) raise
    ``RuntimeError`` — use ``open()`` + ``f.write()`` for Python-side
    file I/O.

    From @njit: routes to ``_fprintf_intrinsic`` with str auto-conversion.
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
    """C-style ``snprintf(buf_p, size, fmt, *args)`` — dual-mode.

    ``buf_p`` is an ``intp`` pointer to the destination buffer (caller-
    owned). Typically ``array_data_p(numpy_array)`` — that helper works
    in both modes.

    Returns the number of characters that WOULD have been written if
    ``size`` were unlimited (excluding the trailing NUL), as int32. See
    the module docstring for the Windows @njit truncation-rc divergence
    (Python and Linux/macOS @njit follow C99; Windows @njit uses
    MSVCRT ``_snprintf`` which returns ``-1`` on truncation).
    """
    _reject_percent_n_in_python("snprintf", fmt)
    _reject_grouping_flag_in_python("snprintf", fmt)
    text_bytes = (_python_fmt_compat(fmt) % args).encode('utf-8')
    n_would_have = len(text_bytes)
    if size > 0:
        n_write = min(n_would_have, size - 1)
        # Slice content BEFORE appending NUL so the NUL is always at the
        # correct position even when truncating. The previous form
        # ``memmove(buf_p, text_bytes + b'\x00', n_write + 1)`` left the
        # NUL out of the copied prefix when truncating — buf got
        # n_write content bytes with no terminator.
        src = text_bytes[:n_write] + b'\x00'
        ctypes.memmove(buf_p, src, len(src))
    return n_would_have


@overload(snprintf)
def _overload_snprintf(buf_p, size, fmt, *args):
    fmt_str = extract_literal_str("snprintf", fmt, field="format string")
    _reject_percent_n_or_raise("snprintf", fmt_str)
    _reject_grouping_flag_or_raise("snprintf", fmt_str)
    if buf_p != intp:
        raise TypingError(
            f"snprintf: buf must be intp (pointer-as-int), got {buf_p!r}"
        )
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
    """C-style ``sscanf(buf, fmt, *args)`` — @njit-only.

    Args are intp output pointers (typically ``array_data_p`` of a
    1-element numpy array of the right dtype). See the @njit-only
    docstring on ``_sscanf_intrinsic`` for the pointer-vs-spec contract.

    Pure-Python users: this binding raises ``NotImplementedError``. For
    parsing in pure Python use ``int()``, ``float()``, or ``re``.
    """
    raise NotImplementedError(
        "sscanf is @njit-only; wrap the call in @njit, or use Python's "
        "int() / float() / re for pure-Python parsing"
    )


@overload(sscanf)
def _overload_sscanf(buf, fmt, *args):
    extract_literal_str("sscanf", fmt, field="format string")  # validates Literal[str]
    if buf != intp:
        raise TypingError(
            f"sscanf: buf must be intp (pointer-as-int), got {buf!r}"
        )
    # sscanf args are output POINTERS — must all be intp. No auto-conversion;
    # pointers don't promote. Build the impl by reusing the helper but
    # without get_unicode_data_p (intp args pass through unchanged).
    for i, ty in enumerate(args):
        if ty != intp:
            raise TypingError(
                f"sscanf: args[{i}] must be intp (output pointer), got {ty!r}"
            )
    impl = _build_overload_impl(
        "sscanf", ["buf", "fmt"], args, _sscanf_intrinsic,
        _get_unicode_data_p_lazy(),  # unused (no UnicodeType in args)
    )
    return impl
