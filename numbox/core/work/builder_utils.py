"""Queries on a graph of node specs, or of the ``Work`` nodes built from them. :func:`infer_sources_dependencies` finds,
for each node, the nodes that depend on it.
"""
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
    Return a dictionary from the name of each node of the graph of ``access_nodes`` to the names of its dependents.

    A node's dependents are the nodes that read it as a source, directly or through other nodes. ``access_nodes`` holds
    specs, such as ``End`` and ``Derived``, or the ``Work`` nodes that :func:`~numbox.core.work.builder.make_graph`
    returns. An access node without sources has no entry. In the graph below, ``m1`` has the sources ``m2`` and
    ``m4``, ``m2`` has the source ``m3``, and ``m3`` has the source ``m5``::

        m1 -- m2 -- m3 -- m5
              |
              m4

    For the access node ``m1``, the function returns::

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
