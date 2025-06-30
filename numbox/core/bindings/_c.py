from numbox.utils.highlevel import cres
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib


load_lib("c")


@cres(signatures.get("rand"), cache=True)
def rand():
    return _call_lib_func("rand", ())


@cres(signatures.get("srand"), cache=True)
def srand(s):
    return _call_lib_func("srand", (s,))


@cres(signatures.get("strlen"), cache=True)
def strlen(s):
    return _call_lib_func("strlen", (s,))
