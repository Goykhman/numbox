"""A structref ``Entity`` with one numba type, and one compiled ``calculate``, for each type of its attribute ``x1``.

``test/test_multiple_overloads.py`` uses it. ``ol_entity`` is the ``@overload`` of ``Entity``: when numba types a call
to ``Entity``, it creates an ``EntityTypeClass`` instance for the type of ``x1``. ``ol_calculate`` generates the source
of ``_calculate_``, compiles it under this file's name, and returns it as ``calculate`` for every entity type.
"""
from io import StringIO
from inspect import getfile, getmodule
from numba import njit
from numba.core import types
from numba.experimental import structref
from numba.extending import overload, overload_method

from numbox.core.configurations import jit_options


def _file_anchor():
    """Give ``getmodule`` and ``getfile`` an object defined in this file. The function is never called."""
    raise NotImplementedError


@structref.register
class EntityTypeClass(types.StructRef):
    pass


class Entity(structref.StructRefProxy):
    def __new__(cls, x1):
        return make_entity(x1)

    @njit(**jit_options)
    def calculate(self):
        return self.calculate()


structref.define_boxing(EntityTypeClass, Entity)


@overload(Entity, strict=False, prefer_literal=False)
def ol_entity(x1_ty):
    entity_type = EntityTypeClass([("x1", x1_ty)])

    def _(x1):
        entity_ = structref.new(entity_type)
        entity_.x1 = x1
        return entity_
    return _


@njit(**jit_options)
def make_entity(x1):
    return Entity(x1)


# ol_calculate stores the generated _calculate_ here, so the source is compiled once per process, and every entity
# type receives the same function.
_calculate_registry = {}


def _make_calculate_code():
    code_txt = StringIO()
    code_txt.write("""def _calculate_(entity):
    return 2.17""")
    return code_txt.getvalue()


@overload_method(EntityTypeClass, "calculate", strict=False, jit_options=jit_options)
def ol_calculate(entity_ty):
    _calculate = _calculate_registry.get(0, None)
    if _calculate is None:
        code_txt = _make_calculate_code()
        # The generated source is compiled under this file's name and runs in this module's namespace. numba caches a
        # function only if the file that its code names exists and its globals hold __name__. numba imports the module
        # that __name__ names to rebuild the environment of the cached code in a later process.
        ns = getmodule(_file_anchor).__dict__
        code = compile(code_txt, getfile(_file_anchor), mode="exec")
        exec(code, ns)
        _calculate = ns["_calculate_"]
        _calculate_registry[0] = _calculate
    return _calculate
