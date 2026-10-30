"""The type classes of the structs that ``test/utils/test_highlevel.py`` defines with ``make_structref``.

Each struct needs a type class of its own. The classes are defined in an imported module because numba pickles a
struct's type into its cache, and a type class from a script that runs does not match the cached copy in the next run.
``test_highlevel.py`` can also run as a script, through its ``collect_and_run_tests`` block.
"""
from numba.experimental.structref import register
from numba.core.types import StructRef


@register
class S1TypeClass(StructRef):
    pass


@register
class S2TypeClass(StructRef):
    pass


@register
class S3TypeClass(StructRef):
    pass


@register
class S4TypeClass(StructRef):
    pass


@register
class S5TypeClass(StructRef):
    pass
