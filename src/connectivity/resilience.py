"""Resilience simulations for targeted edge removals"""

from __future__ import annotations

import networkx as nx

from src.connectivity.crp import compute_crp_values
from src.graph.graph_utils import gcc_subgraph


def mean_crossbank_distance_unweighted(
    G: nx.Graph,
    *,
    north_nodes: set[str],
    south_nodes: set[str],
) -> float:
    """Mean shortest-path length across connected north-south pairs (unweighted).

    Only counts pairs that are connected (i.e., within the same component).
    """
    north = sorted({str(n) for n in north_nodes if str(n) in G})
    south_set = {str(n) for n in south_nodes if str(n) in G}
    if not north or not south_set:
        return float("nan")

    total = 0
    count = 0
    # BFS per north node; accumulate to south nodes reachable in its component.
    for s in north:
        dist = nx.single_source_shortest_path_length(G, s)
        for t in south_set:
            d = dist.get(t)
            if d is None:
                continue
            total += int(d)
            count += 1
    return (total / count) if count else float("nan")


def compute_crp_on_gcc(
    G: nx.Graph,
    *,
    node_to_bank: dict[str, str],
) -> dict[str, float]:
    """Compute CRP on the graph's GCC (returns station_id -> crp)."""
    gcc = gcc_subgraph(G)
    if not nx.is_connected(gcc):
        raise ValueError("GCC must be connected.")

    # Strict: require a bank label for every GCC node.
    nodes = [str(n) for n in gcc.nodes]
    missing = [n for n in nodes if n not in node_to_bank]
    if missing:
        raise ValueError(f"Missing bank labels for {len(missing)} GCC nodes (e.g. {missing[:10]}).")
    bank = {n: str(node_to_bank[n]) for n in nodes}
    return compute_crp_values(gcc, node_to_bank=bank)
