from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import pandas as pd

from src.core.config import get_paths
from src.graph.metrics import build_graph_from_edges
from src.io import read_csv_validated
from src.models.schemas import (
    EDGE_CROSSING,
    EDGES,
    STATION_BANK,
    STATIONS,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisDataset:
    """Complete dataset for Thames connectivity analysis."""

    # Core data (always loaded)
    stations: pd.DataFrame
    edges: pd.DataFrame

    # Spatial enrichment data (conditionally loaded)
    station_bank: pd.DataFrame | None
    edge_crossing: pd.DataFrame | None

    # Computed graph properties
    graph: nx.Graph
    graph_weighted: nx.Graph | None
    gcc_nodes: set[str]
    excluded_nodes: set[str]

    # Summary metadata
    summary: dict[str, object]


def load_analysis_dataset(
    paths=None,
    *,
    use_gcc: bool = True,
    validate_schemas: bool = True,
    load_spatial: bool = True,
    weight_col: str | None = None,
) -> AnalysisDataset:
    """Load and validate core analysis dataset."""
    if paths is None:
        paths = get_paths()

    LOGGER.info("Loading core transit network data...")

    stations = read_csv_validated(
        paths.processed_transit / "stations_london.csv",
        dtype={"station_id": "string"},
        schema=STATIONS,
    )
    edges = read_csv_validated(
        paths.processed_transit / "edges_london.csv",
        dtype={"u": "string", "v": "string"},
        schema=EDGES,
    )
    LOGGER.info("Core data validation passed")

    station_bank = edge_crossing = None
    if load_spatial:
        LOGGER.info("Loading spatial enrichment data...")
        station_bank = read_csv_validated(
            paths.processed_spatial / "station_bank.csv",
            dtype={"station_id": "string"},
            schema=STATION_BANK,
        )
        edge_crossing = read_csv_validated(
            paths.processed_spatial / "edge_is_thames_crossing.csv",
            dtype={"u": "string", "v": "string"},
            schema=EDGE_CROSSING,
        )
        LOGGER.info("Spatial data validation passed")

    # Build graphs (unweighted always; weighted optional)
    LOGGER.info("Building network graph (use_gcc=%s)...", use_gcc)
    graph_result = build_graph_from_edges(
        stations=stations, edges=edges, use_gcc=use_gcc, weight_col=None
    )

    graph_weighted: nx.Graph | None = None
    if weight_col is not None:
        graph_weighted = build_graph_from_edges(
            stations=stations, edges=edges, use_gcc=use_gcc, weight_col=weight_col
        ).G

    n_stations = len(stations)
    n_gcc = len(graph_result.gcc_nodes)

    summary: dict[str, object] = {
        "n_stations_input": int(n_stations),
        "n_edges_input": int(len(edges)),
        "n_stations_graph": int(graph_result.G.number_of_nodes()),
        "n_edges_graph": int(graph_result.G.number_of_edges()),
        "n_gcc_nodes": int(n_gcc),
        "n_excluded_nodes": int(len(graph_result.excluded_nodes)),
        "gcc_share": float(n_gcc / n_stations) if n_stations else 0.0,
        "use_gcc": bool(use_gcc),
        "has_spatial": bool(load_spatial),
        "has_weights": bool(weight_col is not None),
        "weight_col": str(weight_col) if weight_col is not None else None,
    }

    if load_spatial and station_bank is not None:
        bank_counts = station_bank["bank"].value_counts()
        summary.update(
            {
                "n_north_stations": int(bank_counts.get("north", 0)),
                "n_south_stations": int(bank_counts.get("south", 0)),
            }
        )

        if edge_crossing is not None:
            crossing_count = edge_crossing["is_thames_crossing"].fillna(False).sum()
            summary.update(
                {
                    "n_crossing_edges": int(crossing_count),
                    "crossing_share": float(crossing_count / len(edges)) if len(edges) else 0.0,
                }
            )

    LOGGER.info(
        "Dataset loaded successfully: %d stations, %d edges, GCC=%d (%.1f%%)",
        n_stations,
        len(edges),
        n_gcc,
        summary["gcc_share"] * 100,
    )

    return AnalysisDataset(
        stations=stations,
        edges=edges,
        station_bank=station_bank,
        edge_crossing=edge_crossing,
        graph=graph_result.G,
        graph_weighted=graph_weighted,
        gcc_nodes=graph_result.gcc_nodes,
        excluded_nodes=graph_result.excluded_nodes,
        summary=summary,
    )
