from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphBuildResult:
    G: nx.Graph
    gcc_nodes: set[str]
    excluded_nodes: set[str]
    excluded_edges: int


def build_graph_from_edges(
    *,
    stations: pd.DataFrame,
    edges: pd.DataFrame,
    use_gcc: bool = True,
    weight_col: str | None = None,
) -> GraphBuildResult:
    """Build an undirected NetworkX graph from station/edge tables.

    If `use_gcc` is True, the returned `G` is the GCC subgraph.
    """
    if "station_id" not in stations.columns:
        raise ValueError("stations must have 'station_id' column")
    if not {"u", "v"}.issubset(edges.columns):
        raise ValueError("edges must have 'u' and 'v' columns")

    st_ids = set(stations["station_id"].astype(str))
    if not st_ids:
        raise ValueError("stations has 0 station_id values")

    e = edges.copy()
    e["u"] = e["u"].astype(str)
    e["v"] = e["v"].astype(str)

    bad_u = sorted(set(e.loc[~e["u"].isin(st_ids), "u"].tolist()))
    bad_v = sorted(set(e.loc[~e["v"].isin(st_ids), "v"].tolist()))
    if bad_u or bad_v:
        sample = (bad_u + bad_v)[:10]
        raise ValueError(
            f"edges reference unknown station_id(s): {len(set(bad_u) | set(bad_v))} (e.g. {sample})"
        )

    if weight_col is not None:
        if weight_col not in e.columns:
            raise ValueError(f"weight_col={weight_col!r} not found in edges")
        w = e[weight_col].astype(float)
        if w.isna().any() or (~np.isfinite(w.to_numpy())).any():
            bad = e.loc[w.isna() | (~np.isfinite(w.to_numpy())), ["u", "v", weight_col]].head(5)
            raise ValueError(
                f"edges has invalid weights in {weight_col!r} (e.g. {bad.to_dict(orient='records')})"
            )
        e2 = e[["u", "v"]].copy()
        e2["weight"] = w.astype(float)
        G = nx.from_pandas_edgelist(
            e2,
            source="u",
            target="v",
            edge_attr="weight",
            create_using=nx.Graph(),
        )
    else:
        e2 = e[["u", "v"]]
        G = nx.from_pandas_edgelist(e2, source="u", target="v", create_using=nx.Graph())

    # Ensure all stations exist as nodes (including isolates not present in any edge)
    G.add_nodes_from(sorted(st_ids))

    excluded_edges = 0

    comps = list(nx.connected_components(G))
    if not comps:
        return GraphBuildResult(
            G=G, gcc_nodes=set(), excluded_nodes=set(st_ids), excluded_edges=excluded_edges
        )
    gcc = max(comps, key=len)
    excluded_nodes = set(G.nodes) - set(gcc)

    if use_gcc:
        G = G.subgraph(gcc).copy()
    return GraphBuildResult(
        G=G, gcc_nodes=set(gcc), excluded_nodes=excluded_nodes, excluded_edges=excluded_edges
    )
