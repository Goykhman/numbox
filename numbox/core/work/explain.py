from inspect import getsource
from textwrap import indent

from numbox.core.work.builder import _derive_funcs
from numbox.core.work.work import Work


def get_func_code(derive_func_p_):
    derive_func_ = _derive_funcs.get(derive_func_p_)
    return derive_func_.__name__, indent(getsource(derive_func_), "    ")


def _explain(work: Work, derivation_: list, derived_: set):
    if work.name in derived_:
        return
    if len(work.sources) == 0:
        derivation_.append(f"{work.name}: end node\n")
        derived_.add(work.name)
        return
    inputs_names = []
    for source in work.sources:
        _explain(source, derivation_, derived_)
        inputs_names.append(source.name)
    derive_func_ptrs = work.derive
    derive_func_name, derive_func_code = get_func_code(derive_func_ptrs[1])
    derivation_.append(f"""{work.name}: {derive_func_name}({", ".join(inputs_names)})

{derive_func_code}""")
    derived_.add(work.name)


def explain(work: Work):
    all_end_nodes = work.all_end_nodes()
    derivation = []
    _explain(work, derivation, set())
    explain_txt = f"All required end nodes: {all_end_nodes}\n\n"
    explain_txt += "\n".join(derivation)
    return explain_txt
