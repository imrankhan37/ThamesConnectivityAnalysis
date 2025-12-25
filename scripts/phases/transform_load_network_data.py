"""Network Builder: Build clean station and edge tables from TfL ingestion outputs.

Run from repo root:
  uv run python scripts/phases/transform_load_network_data.py

Outputs:
- data/processed/transit/stations.csv
- data/processed/transit/edges.csv
- data/processed/transit/stations_london.csv
- data/processed/transit/edges_london.csv
- data/processed/_meta/network_sanity.csv
- artifacts/graph.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.collections import LineCollection
from shapely.geometry import Point

# Ensure repo root is on sys.path so `import src...` works when executing this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import configure_logging, get_paths
from src.data_processing.network_build import (
    ALLOWED_MODES_DEFAULT,
    compute_edge_distances_m,
    compute_network_metrics,
    edges_from_route_sequences,
    filter_stations_by_geography,
    stations_from_stop_points_and_sequences,
)
from src.io import read_json
from src.models.schemas import EDGES, STATIONS
from src.models.validate import validate_df

# Constants
CRS_PLOT = "EPSG:27700"  # British National Grid for correct geometry in meters
CRS_INPUT = "EPSG:4326"  # WGS84 for input coordinates
FIGURE_SIZE = (10, 10)
PLOT_DPI = 300
BOUNDARY_COLOR = "#222222"
BOUNDARY_ALPHA = 0.6
BOUNDARY_LINEWIDTH = 1.0
NODE_COLOR = "#2563eb"  # blue-600
NODE_SIZE = 10
NODE_ALPHA = 0.85
EDGE_COLOR = (0, 0, 0, 0.2)
EDGE_LINEWIDTH = 0.9
PADDING_FACTOR = 0.1

# File paths
TFL_STOP_POINTS_FILE = "tfl_stop_points_stations.json"
TFL_ROUTE_SEQUENCES_DIR = "tfl_route_sequences"
ONS_BOUNDARY_FILE = "ons_regions_2021_en_bgc.gpkg"
STATIONS_FILE = "stations.csv"
EDGES_FILE = "edges.csv"
STATIONS_LONDON_FILE = "stations_london.csv"
EDGES_LONDON_FILE = "edges_london.csv"
NETWORK_SANITY_FILE = "network_sanity.csv"
NETWORK_OVERVIEW_FIG = "fig01_network_overview.png"
GRAPH_PICKLE_FILE = "graph.pkl"

STATION_ID_DTYPE = "string"

LOGGER = logging.getLogger("network_builder")

MIN_LONDON_GCC_SHARE = 0.95


def _load_route_sequences(dir_path: Path) -> list[dict[str, Any]]:
    files = sorted(dir_path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No route sequence JSON files found in {dir_path}")
    out: list[dict[str, Any]] = []
    for fp in files:
        out.append(read_json(fp))
    return out


def _plot_network_overview(stations: pd.DataFrame, edges: pd.DataFrame, out_path: Path) -> None:
    st = stations.set_index("station_id")

    # Filter to edges where both endpoints have coordinates.
    ok = edges["u"].isin(st.index) & edges["v"].isin(st.index)
    e = edges.loc[ok, ["u", "v"]].copy()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_facecolor("white")

    # Optional: overlay official London boundary if present
    boundary_path = out_path.parents[1] / "data" / "raw" / "ons_regions_2021_en_bgc.gpkg"
    if boundary_path.exists():
        try:
            london = gpd.read_file(boundary_path)
            if not london.empty and london.geometry.notna().any():
                london = london.to_crs(CRS_PLOT)
                london.boundary.plot(
                    ax=ax,
                    color=BOUNDARY_COLOR,
                    linewidth=BOUNDARY_LINEWIDTH,
                    alpha=BOUNDARY_ALPHA,
                    zorder=1,
                )
                minx, miny, maxx, maxy = london.total_bounds
                pad_x = (maxx - minx) * PADDING_FACTOR
                pad_y = (maxy - miny) * PADDING_FACTOR
                ax.set_xlim(minx - pad_x, maxx + pad_x)
                ax.set_ylim(miny - pad_y, maxy + pad_y)
        except Exception as exc:
            LOGGER.warning("Could not overlay London boundary (%s): %s", boundary_path, exc)

    # Project stations to plotting CRS
    st_valid = st.dropna(subset=["lon", "lat"]).copy()
    st_gdf = gpd.GeoDataFrame(
        st_valid,
        geometry=[Point(xy) for xy in zip(st_valid["lon"], st_valid["lat"], strict=True)],
        crs=CRS_INPUT,
    ).to_crs(CRS_PLOT)
    st_x = st_gdf.geometry.x.to_numpy()
    st_y = st_gdf.geometry.y.to_numpy()

    # Build segments in projected coordinates (LineCollection is much cleaner + faster than per-edge ax.plot)
    # Note: edges are in station_id space; map to projected coordinates via index lookup.
    st_xy = pd.DataFrame({"x": st_gdf.geometry.x, "y": st_gdf.geometry.y}, index=st_gdf.index)
    ok2 = e["u"].isin(st_xy.index) & e["v"].isin(st_xy.index)
    e2 = e.loc[ok2, ["u", "v"]]
    x0 = st_xy.loc[e2["u"], "x"].to_numpy(dtype=float)
    y0 = st_xy.loc[e2["u"], "y"].to_numpy(dtype=float)
    x1 = st_xy.loc[e2["v"], "x"].to_numpy(dtype=float)
    y1 = st_xy.loc[e2["v"], "y"].to_numpy(dtype=float)
    segs = list(
        zip(
            zip(x0, y0, strict=True),
            zip(x1, y1, strict=True),
            strict=True,
        )
    )

    # edges
    lc = LineCollection(segs, colors=EDGE_COLOR, linewidths=EDGE_LINEWIDTH, zorder=2)
    ax.add_collection(lc)

    # nodes
    ax.scatter(
        st_x,
        st_y,
        s=NODE_SIZE,
        color=NODE_COLOR,
        alpha=NODE_ALPHA,
        linewidths=0,
        zorder=3,
    )

    ax.set_title(
        "TfL rapid-transit network (stations + adjacent-stop edges)\nONS London boundary (projected: EPSG:27700)"
    )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_alpha(0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI)
    # Also write a JPEG copy (some tooling struggles with PNGs in this environment).
    fig.savefig(out_path.with_suffix(".jpg"), dpi=PLOT_DPI)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build station/edge tables from TfL ingestion outputs.")
    return p.parse_args()


def main() -> None:
    configure_logging()
    paths = get_paths()

    allowed_modes = ALLOWED_MODES_DEFAULT
    LOGGER.info("Allowed modes: %s", list(allowed_modes))

    stop_points_path = paths.data_raw / TFL_STOP_POINTS_FILE
    route_dir = paths.data_raw / TFL_ROUTE_SEQUENCES_DIR

    LOGGER.info("Loading stop points: %s", stop_points_path)
    stop_points = read_json(stop_points_path)
    if not isinstance(stop_points, list):
        raise TypeError("Expected stop points JSON to be a list.")
    LOGGER.info("Stop points loaded: %d", len(stop_points))

    LOGGER.info("Loading route sequences from: %s", route_dir)
    route_sequences = _load_route_sequences(route_dir)
    LOGGER.info("Route sequence files loaded: %d", len(route_sequences))

    # Note: we do not build a NaPTAN->ICS mapping here. The current edge builder uses
    # RouteSequence stopPointSequences[*].stopPoint[*].icsId directly as station ids.

    stations = stations_from_stop_points_and_sequences(
        stop_points, route_sequences, allowed_modes=allowed_modes
    )

    edges = edges_from_route_sequences(route_sequences, allowed_modes=allowed_modes)

    # Validate contracts early (coerces nullable dtypes) so subsequent operations work with clean data
    stations = validate_df(stations, STATIONS)
    edges = validate_df(edges, EDGES)

    # Add edge distance in meters (BNG) - now works with properly typed data
    edges = edges.copy()
    edges["distance_m"] = compute_edge_distances_m(edges, stations)

    LOGGER.info("Stations: %d", len(stations))
    LOGGER.info("Edges (deduped undirected): %d", len(edges))

    # Build graph for sanity checks
    G = nx.Graph()
    for _, r in stations.iterrows():
        G.add_node(r["station_id"])
    for _, r in edges.iterrows():
        G.add_edge(r["u"], r["v"])

    sanity = compute_network_metrics(G)
    LOGGER.info("Components: %s", int(sanity.loc[sanity.metric == "n_components", "value"].iloc[0]))
    LOGGER.info("GCC nodes: %s", int(sanity.loc[sanity.metric == "gcc_nodes", "value"].iloc[0]))

    # Write outputs
    stations_out = paths.processed_transit / STATIONS_FILE
    edges_out = paths.processed_transit / EDGES_FILE
    sanity_out = paths.processed_meta / NETWORK_SANITY_FILE
    fig_out = paths.figures / NETWORK_OVERVIEW_FIG
    graph_out = paths.artifacts / GRAPH_PICKLE_FILE

    paths.processed_transit.mkdir(parents=True, exist_ok=True)
    paths.processed_meta.mkdir(parents=True, exist_ok=True)
    paths.artifacts.mkdir(parents=True, exist_ok=True)

    stations.to_csv(stations_out, index=False)
    edges.to_csv(edges_out, index=False)
    sanity.to_csv(sanity_out, index=False)
    _plot_network_overview(stations, edges, fig_out)

    # London-only subset (always; analysis default)
    stations_london_out = paths.processed_transit / STATIONS_LONDON_FILE
    edges_london_out = paths.processed_transit / EDGES_LONDON_FILE
    boundary_gpkg = paths.data_raw / ONS_BOUNDARY_FILE

    st_l, e_l = filter_stations_by_geography(stations, edges, boundary_gpkg)
    st_l = validate_df(st_l, STATIONS)
    e_l = validate_df(e_l, EDGES)
    st_l.to_csv(stations_london_out, index=False)
    e_l.to_csv(edges_london_out, index=False)
    LOGGER.info("Wrote %s", stations_london_out)
    LOGGER.info("Wrote %s", edges_london_out)

    G_l = nx.Graph()
    for sid in st_l["station_id"].astype(str):
        G_l.add_node(sid)
    G_l.add_edges_from(
        [(u, v) for u, v in e_l[["u", "v"]].astype(str).itertuples(index=False, name=None)]
    )
    comps = list(nx.connected_components(G_l))
    n_components = len(comps)
    gcc_n = len(max(comps, key=len)) if comps else 0
    gcc_share = (gcc_n / G_l.number_of_nodes()) if G_l.number_of_nodes() else 0.0
    LOGGER.info(
        "London-only connectivity QA: nodes=%s edges=%s components=%s gcc_nodes=%s gcc_share=%.3f",
        G_l.number_of_nodes(),
        G_l.number_of_edges(),
        n_components,
        gcc_n,
        gcc_share,
    )
    assert gcc_share >= MIN_LONDON_GCC_SHARE, (
        f"London-only GCC share too low: {gcc_share:.3f} < {MIN_LONDON_GCC_SHARE:.2f}"
    )

    # Graph artifact

    G_out = nx.Graph()
    for sid in st_l["station_id"]:
        G_out.add_node(sid)
    for _, r in e_l.iterrows():
        G_out.add_edge(r["u"], r["v"])
    with open(graph_out, "wb") as f:
        pickle.dump(G_out, f)
    LOGGER.info("Wrote %s", graph_out)

    LOGGER.info("Wrote %s", stations_out)
    LOGGER.info("Wrote %s", edges_out)
    LOGGER.info("Wrote %s", sanity_out)
    LOGGER.info("Wrote %s", fig_out)
    LOGGER.info("Graph scope: London-only")


if __name__ == "__main__":
    _parse_args()
    main()
