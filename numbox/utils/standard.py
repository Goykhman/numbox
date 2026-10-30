"""A helper that formats a function's parameters for the source code that numbox generates.

:func:`make_params_strings` returns the text of a function's parameters for a generated ``def`` line and for a call.
``make_structref`` and ``@proxy`` use it to write the methods and wrappers that they generate.
"""
import inspect


_REJECTED_KINDS = {
    inspect.Parameter.VAR_POSITIONAL: "*args",
    inspect.Parameter.VAR_KEYWORD: "**kwargs",
}


def make_params_strings(func):
    """Return two strings for generated source: the parameters of ``func`` with their defaults, and their names.

    For ``def f(x, y=2)`` the strings are ``"x, y=2"``, for a ``def`` line, and ``"x, y"``, for a call. A ``*args`` or
    ``**kwargs`` parameter raises ``ValueError``. The generated code cannot reproduce the ``*`` or ``**`` marker, so it
    would bind such a parameter as a plain positional one and change how the function is called.

    A keyword-only marker (``*,``) or a positional-only marker (``/``) is dropped deliberately, and every parameter
    becomes positional-or-keyword. ``@proxy`` relies on this for a function with a defaulted keyword-only parameter,
    such as ``def f(x, *, y=default)``, whose further signatures type an omitted ``y`` as ``Omitted``. Its wrapper
    takes ``f(x, y=default)``, so a caller can also pass ``y`` by position (:func:`numbox.core.proxy.proxy.proxy`).

    Each default is written as its ``repr``. A default whose ``repr`` is a valid expression, such as an int, a finite
    float, a string, ``None``, a bool or a tuple of these, is written correctly. ``float('nan')`` and ``float('inf')``
    become the names ``nan`` and ``inf``, which raise ``NameError`` when the generated code runs. An object whose
    ``repr`` has the form ``<MyClass object at 0x...>`` produces invalid source, which raises ``SyntaxError`` or
    ``NameError``. Each of these cases fails with an error instead of producing wrongly compiled code.
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
