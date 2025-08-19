from __future__ import annotations
from typing import Dict, List
from .types import GraphSpec

def topo_sort(graph: GraphSpec) -> List[str]:
    """Deterministic Kahn topo sort; raises on cycle; stable sibling order."""
    indeg: Dict[str, int] = {n.id: 0 for n in graph.nodes}
    adj: Dict[str, List[str]] = {n.id: [] for n in graph.nodes}
    for u, v in graph.edges:
        adj[u].append(v)
        indeg[v] += 1

    ready = sorted([nid for nid, d in indeg.items() if d == 0])
    order: List[str] = []

    while ready:
        u = ready.pop(0)
        order.append(u)
        for v in sorted(adj[u]):  # stable ordering
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
                ready.sort()

    if len(order) != len(indeg):
        raise ValueError("Graph has cycles")

    return order