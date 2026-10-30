"""A smoke test that the ``numbox`` package imports.

The package's ``__init__.py`` defines only ``__version__``, so the test imports no submodule.
"""


def test_0():
    import numbox  # noqa: F401
