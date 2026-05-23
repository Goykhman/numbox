import inspect


_REJECTED_KINDS = {
    inspect.Parameter.VAR_POSITIONAL: "*args",
    inspect.Parameter.VAR_KEYWORD: "**kwargs",
}


def make_params_strings(func):
    """Render a function's parameters as two source-string fragments suitable
    for inlining into generated ``def`` headers and call sites.

    ``*args`` and ``**kwargs`` are rejected with ``ValueError`` — their
    ``*`` / ``**`` markers cannot be reproduced by the existing source-
    string codegen and the generated wrapper would silently bind those
    parameters as plain positional args (a different calling convention
    from what the user wrote).

    Keyword-only (``*,``) and positional-only (``/``) markers are
    silently flattened to positional-or-keyword — by design. The
    ``@proxy`` decorator relies on this loosening to support ``Omitted``-
    style overloads written as ``def f(x, *, y=default)``: the proxy
    wrapper exposes ``f(x, y=default)`` so callers can pass ``y``
    positionally as well. Documented at :func:`numbox.core.proxy.proxy`.

    Default-argument values are formatted via ``repr(default)`` —
    primitives whose ``repr()`` is a valid Python expression (ints,
    finite floats, strings, ``None``, booleans, tuples thereof) round-
    trip. Non-round-tripping exceptions: ``float('nan')`` /
    ``float('inf')`` render as bare identifiers ``nan`` / ``inf`` and
    ``NameError`` at exec time; complex objects whose ``repr()`` is e.g.
    ``"<MyClass object at 0x...>"`` likewise render to invalid source and
    raise ``SyntaxError`` / ``NameError``. These are visible failures,
    not silent miscompiles.
    """
    func_params = inspect.signature(func).parameters
    for name, p in func_params.items():
        if p.kind in _REJECTED_KINDS:
            raise ValueError(
                f"{func.__qualname__}: parameter {name!r} of kind "
                f"{_REJECTED_KINDS[p.kind]} is not supported by source-string codegen; "
                f"use explicit positional-or-keyword parameters"
            )
    func_params_str = ', '.join(
        [k if v.default == inspect._empty else f'{k}={v.default!r}' for k, v in func_params.items()]
    )
    func_names_params_str = ', '.join(func_params.keys())
    return func_params_str, func_names_params_str
