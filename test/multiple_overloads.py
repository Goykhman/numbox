from io import StringIO
from inspect import getfile, getmodule
from numba import njit
from numba.core import types
from numba.experimental import structref
from numba.extending import overload, overload_method

from numbox.core.configurations import default_jit_options


def _file_anchor():
    raise NotImplementedError


@structref.register
class EntityTypeClass(types.StructRef):
    pass


class Entity(structref.StructRefProxy):
    def __new__(cls, x1):
        return make_entity(x1)

    @njit(**default_jit_options)
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


@njit(**default_jit_options)
def make_entity(x1):
    return Entity(x1)


_calculate_registry = {}


def _make_calculate_code():
    code_txt = StringIO()
    code_txt.write("""def _calculate_(entity):
    return 2.17""")
    return code_txt.getvalue()


@overload_method(EntityTypeClass, "calculate", strict=False, jit_options=default_jit_options)
def ol_calculate(entity_ty):
    _calculate = _calculate_registry.get(0, None)
    if _calculate is None:
        code_txt = _make_calculate_code()
        ns = getmodule(_file_anchor).__dict__
        code = compile(code_txt, getfile(_file_anchor), mode="exec")
        exec(code, ns)
        _calculate = ns["_calculate_"]
        _calculate_registry[0] = _calculate
    return _calculate
