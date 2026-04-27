"""
These functions included in this module::

    get_or_make_global
    _get_global
    _set_global

were mainly based on the

    `CognitiveRuleEngine <https://github.com/DannyWeitekamp/Cognitive-Rule-Engine/blob/main/cre/utils.py>`_

open-source project, distributed under

MIT License

Copyright (c) 2023 Daniel Weitekamp

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  # noqa: E501
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:  # noqa: E501

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.  # noqa: E501

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.  # noqa: E501
"""

from llvmlite import ir
from numba import njit
from numba.extending import intrinsic
from numba.core import cgutils
from numba.core.types import DictType, unicode_type, void
from numba.typed.typeddict import Dict

from numbox.core.configurations import default_jit_options
from numbox.core.work.node import NodeType


def get_or_make_global(context, builder, fe_type, name):
    mod = builder.module
    try:
        gv = mod.get_global(name)
    except KeyError:
        ll_ty = context.get_value_type(fe_type)
        gv = ir.GlobalVariable(mod, ll_ty, name=name)
        gv.linkage = "common"
        gv.initializer = cgutils.get_null_value(gv.type.pointee)
    return gv


@intrinsic(prefer_literal=True)
def _get_global(typingctx, type_ref, name_ty):
    ty = type_ref.instance_type
    name = name_ty.literal_value

    def codegen(context, builder, signature, arguments):
        gv = get_or_make_global(context, builder, ty, name)
        v = builder.load(gv)
        context.nrt.incref(builder, ty, v)
        return v
    sig = ty(type_ref, name_ty)
    return sig, codegen


@intrinsic(prefer_literal=True)
def _set_global(typingctx, type_ref, name_ty, v_ty):
    ty = type_ref.instance_type
    name = name_ty.literal_value

    def codegen(context, builder, signature, arguments):
        _, __, v = arguments
        gv = get_or_make_global(context, builder, ty, name)
        builder.store(v, gv)
    sig = void(type_ref, name_ty, v_ty)
    return sig, codegen


registry_type = DictType(unicode_type, NodeType)


@njit(**default_jit_options)
def set_global(registry_):
    _set_global(registry_type, "_work_registry", registry_)


registry = Dict.empty(unicode_type, NodeType)
set_global(registry)
