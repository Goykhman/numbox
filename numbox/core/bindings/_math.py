from numbox.utils.highlevel import cres
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib


load_lib("m")


@cres(signatures.get("cos"), cache=True)
def cos(x):
    return _call_lib_func("cos", (x,))


@cres(signatures.get("sin"), cache=True)
def sin(x):
    return _call_lib_func("sin", (x,))


@cres(signatures.get("tan"), cache=True)
def tan(x):
    return _call_lib_func("tan", (x,))
