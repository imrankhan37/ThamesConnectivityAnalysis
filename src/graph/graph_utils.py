"""Small, shared graph utilities (pure functions only)."""

from __future__ import annotations

import networkx as nx


def canon_edge(u: str, v: str) -> tuple[str, str]:
    """Canonical undirected edge key."""
    return (u, v) if u < v else (v, u)


def gcc_subgraph(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component subgraph (copy)."""
    if G.number_of_nodes() == 0:
        return G.copy()
    comps = list(nx.connected_components(G))
    if not comps:
        return G.copy()
    gcc = max(comps, key=len)
    return G.subgraph(gcc).copy()
