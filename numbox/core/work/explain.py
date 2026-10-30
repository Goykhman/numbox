"""Explain how the value of a node is computed. :func:`explain` lists the end nodes that a node needs and, for each node
of its graph, the derive and the sources that compute it, with the derive's source code where
:func:`~numbox.core.work.builder.make_graph` compiled the derive.
"""
from inspect import getsource
from textwrap import indent

from numbox.core.work.builder import _derive_funcs
from numbox.core.work.work import Work


def get_func_code(derive_func_p_):
    """Return the name and the indented source code of a derive's Python function, or ``(None, None)``.

    ``derive_func_p_`` is the address of the derive's Python object, the second element of the tuple that
    ``Work.derive`` returns. :func:`~numbox.core.work.builder.make_graph` records the Python function of each derive it
    compiles under that address. A derive passed to its node by other means, such as ``make_work`` or
    ``ll_make_work``, has no record, and the result is ``(None, None)``.
    """
    derive_func_ = _derive_funcs.get(derive_func_p_)
    if derive_func_ is None:
        return None, None
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
    if derive_func_name is None:
        derivation_.append(
            f"{work.name}: <derive built outside builder.make_graph>"
            f"({', '.join(inputs_names)})\n")
    else:
        derivation_.append(f"""{work.name}: {derive_func_name}({", ".join(inputs_names)})

{derive_func_code}""")
    derived_.add(work.name)


def explain(work: Work):
    """Return a text that explains how the value of ``work`` is computed.

    The text names the end nodes that ``work`` needs, the nodes without sources, and then lists each node of its graph
    once, after its sources. A derived node is listed with the name of its derive and the names of its sources,
    followed by the derive's source code when :func:`~numbox.core.work.builder.make_graph` compiled the derive. A derive
    that ``make_graph`` did not compile is shown as a placeholder. A node with sources but no derive raises
    ``TypeError``.
    """
    all_end_nodes = work.all_end_nodes()
    derivation = []
    _explain(work, derivation, set())
    explain_txt = f"All required end nodes: {all_end_nodes}\n\n"
    explain_txt += "\n".join(derivation)
    return explain_txt
