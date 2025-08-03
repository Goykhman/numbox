from typing import NamedTuple, Tuple

from numbox.core.work.builder import SpecTy, End


def _infer_sources_dependencies(spec: SpecTy, sources_dependencies_):
    if isinstance(spec, End):
        return
    for source in spec.sources:
        spec_dependencies = sources_dependencies_.setdefault(spec.name, set())
        sources_dependencies_.setdefault(source.name, set()).add(spec.name)
        sources_dependencies_[source.name] |= spec_dependencies
        _infer_sources_dependencies(source, sources_dependencies_)


def infer_sources_dependencies(access_nodes: NamedTuple | Tuple):
    """
    For all nodes names accessible from the given `access_nodes`,
    return dictionary of all nodes names that depend on each of the
    nodes in the accessible graph. For instance::

        m1 -- m2 -- m3 -- m5
              |
              m4

    will return::

        {
            "m1": set(),
            "m2": {"m1"},
            "m3": {"m1", "m2"},
            "m4": {"m1"},
            "m5": {"m1", "m2", "m3"}
        }

    """
    sources_dependencies_ = {}
    for access_node in access_nodes:
        _infer_sources_dependencies(access_node, sources_dependencies_)
    return sources_dependencies_
