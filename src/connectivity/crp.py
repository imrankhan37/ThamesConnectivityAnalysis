"""Cross-river penalty (CRP) and related node-level dependence metrics."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class CRPSummary:
    n_nodes: int
    n_pairs_same: int
    n_pairs_opp: int
    k_values: tuple[int, ...]


def all_pairs_shortest_path_lengths_unweighted(G: nx.Graph) -> dict[str, dict[str, int]]:
    """All-pairs shortest path lengths for a connected unweighted graph.

    Deterministic: node ordering is sorted by string value.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty.")
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected for all-pairs distances.")

    nodes = sorted((str(n) for n in G.nodes), key=str)

    # BFS to get all-pairs shortest path lengths
    raw = nx.all_pairs_shortest_path_length(G)
    raw_map: dict[str, dict[str, int]] = {
        str(s): {str(t): int(d) for t, d in dist.items()} for s, dist in raw
    }

    out: dict[str, dict[str, int]] = {}
    for s in nodes:
        dist = raw_map.get(s)
        if dist is None:
            raise RuntimeError(f"Missing distances for source {s!r}.")
        missing = [t for t in nodes if t not in dist]
        if missing:
            raise RuntimeError(
                f"Distances for {s!r} missing {len(missing)} targets (e.g. {missing[:5]})."
            )
        out[s] = {t: int(dist[t]) for t in nodes}
    return out


def compute_crp(
    *,
    dists: dict[str, dict[str, int]],
    node_to_bank: dict[str, str],
    k_values: tuple[int, ...] = (8, 10),
) -> tuple[list[dict[str, object]], CRPSummary]:
    """Compute CRP and reachability shares for each node.

    CRP(v) = mean_d(v -> opposite bank) - mean_d(v -> same bank)
    Also computes reachability share within k hops to opposite vs same bank.

    Returns:
      (rows, summary) where rows is a list of dicts with keys:
        - station_id, bank, mean_d_same, mean_d_opp, crp
        - reach_same_k{K}, reach_opp_k{K} for each K in k_values
    """
    if not dists:
        raise ValueError("dists is empty.")
    if not node_to_bank:
        raise ValueError("node_to_bank is empty.")

    nodes = sorted(dists.keys(), key=str)
    missing_bank = [n for n in nodes if n not in node_to_bank]
    if missing_bank:
        raise ValueError(
            f"Missing bank labels for {len(missing_bank)} nodes (e.g. {missing_bank[:5]})."
        )

    banks = {n: str(node_to_bank[n]) for n in nodes}
    bad_bank = sorted({b for b in banks.values() if b not in {"north", "south"}})
    if bad_bank:
        raise ValueError(f"Invalid bank labels (expected 'north'/'south'): {bad_bank}")
    if len(set(banks.values())) < 2:
        raise ValueError("Need at least two banks present to compute CRP.")

    # Precompute node lists per bank
    by_bank: dict[str, list[str]] = {}
    for n, b in banks.items():
        by_bank.setdefault(b, []).append(n)

    rows: list[dict[str, object]] = []
    n_pairs_same = 0
    n_pairs_opp = 0

    for v in nodes:
        b = banks[v]
        same_nodes = [u for u in by_bank.get(b, []) if u != v]
        opp_nodes = [u for bb, lst in by_bank.items() if bb != b for u in lst]

        dv = dists[v]
        missing = [u for u in nodes if u not in dv]
        if missing:
            raise ValueError(f"dists[{v!r}] missing {len(missing)} targets (e.g. {missing[:5]}).")
        same_ds = [dv[u] for u in same_nodes if u in dv]
        opp_ds = [dv[u] for u in opp_nodes if u in dv]

        n_pairs_same += len(same_ds)
        n_pairs_opp += len(opp_ds)

        mean_same = float(np.mean(same_ds)) if same_ds else float("nan")
        mean_opp = float(np.mean(opp_ds)) if opp_ds else float("nan")
        crp = (
            mean_opp - mean_same
            if (np.isfinite(mean_opp) and np.isfinite(mean_same))
            else float("nan")
        )

        row: dict[str, object] = {
            "station_id": v,
            "bank": b,
            "mean_d_same": mean_same,
            "mean_d_opp": mean_opp,
            "crp": crp,
        }
        for k in k_values:
            # Shares are relative to the available same/opp sets (exclude self)
            reach_same = float(np.mean([d <= k for d in same_ds])) if same_ds else float("nan")
            reach_opp = float(np.mean([d <= k for d in opp_ds])) if opp_ds else float("nan")
            row[f"reach_same_k{k}"] = reach_same
            row[f"reach_opp_k{k}"] = reach_opp
        rows.append(row)

    summary = CRPSummary(
        n_nodes=len(nodes),
        n_pairs_same=int(n_pairs_same),
        n_pairs_opp=int(n_pairs_opp),
        k_values=tuple(int(k) for k in k_values),
    )
    return rows, summary


def compute_crp_on_graph(
    G: nx.Graph,
    *,
    node_to_bank: dict[str, str],
    k_values: tuple[int, ...] = (8, 10),
) -> tuple[list[dict[str, object]], CRPSummary]:
    """Convenience wrapper: compute CRP from a connected unweighted graph."""
    dists = all_pairs_shortest_path_lengths_unweighted(G)
    return compute_crp(dists=dists, node_to_bank=node_to_bank, k_values=k_values)


def compute_crp_values(
    G: nx.Graph,
    *,
    node_to_bank: dict[str, str],
) -> dict[str, float]:
    """Return station_id -> CRP value for a connected unweighted graph."""
    rows, _ = compute_crp_on_graph(G, node_to_bank=node_to_bank, k_values=())
    out: dict[str, float] = {}
    for r in rows:
        v = str(r["station_id"])
        crp = float(r["crp"])
        if np.isfinite(crp):
            out[v] = crp
    return out
