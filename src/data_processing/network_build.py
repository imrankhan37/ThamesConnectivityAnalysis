"""Build a station/edge representation of the TfL rapid-transit network.

Design goals:
- Deterministic outputs (stable sorting, no timestamps).
- Keep logic in `src/` and I/O orchestration in `scripts/`.
- Use TfL RouteSequence `stopPointSequences[*].stopPoint[*].icsId` as the source of stop order.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point

ALLOWED_MODES_DEFAULT: tuple[str, ...] = ("tube", "dlr", "overground", "elizabeth-line")

# Geographic constants
CRS_INPUT = "EPSG:4326"  # WGS84 for input coordinates
CRS_PROJECTED = "EPSG:27700"  # British National Grid for metric calculations


@dataclass(frozen=True)
class EdgeAgg:
    line_ids: set[str]
    modes: set[str]
    directions: set[str]
    route_names: set[str]


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _first_non_empty(*vals: Any) -> Any:
    """Return the first value that is neither None nor empty-string."""
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v == "":
            continue
        return v
    return None


def _as_str_or_none(v: Any) -> str | None:
    return None if v is None else str(v)


def _as_str_or_empty(v: Any) -> str:
    return "" if v is None else str(v)


def _as_float_or_none(v: Any) -> float | None:
    return None if v is None else float(v)


def _is_transport_interchange(stop_type: Any) -> bool:
    return str(stop_type) == "TransportInterchange"


def _require_any(rows: list[dict[str, Any]], *, name: str) -> None:
    """Fail-fast guard: if an expected block yields no usable rows, schema likely changed."""
    if not rows:
        raise ValueError(
            f"{name}: produced 0 usable rows. This likely means the TfL schema changed or inputs are empty."
        )


def _iter_stop_points_recursive(stop_points: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Yield stop point dicts, including nested children, in a stable order."""
    for sp in stop_points:
        if not isinstance(sp, dict):
            continue
        yield sp
        children = [c for c in _as_list(sp.get("children")) if isinstance(c, dict)]
        if children:
            yield from _iter_stop_points_recursive(children)


def _edge_key(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def _agg_mode_union(series: pd.Series) -> str:
    s: set[str] = set()
    for x in series.dropna().astype(str):
        if x:
            s |= set([p for p in x.split(";") if p])
    return ";".join(sorted(s))


def _agg_choose_name(series: pd.Series) -> str:
    s = series.dropna().astype(str)
    if s.empty:
        return ""
    return s.value_counts().index[0]


def _agg_zone_min(series: pd.Series) -> str:
    vals: list[float] = []
    for x in series.dropna().astype(str):
        try:
            vals.append(float(x))
        except ValueError:
            continue
    if not vals:
        return ""
    z = min(vals)
    return str(int(z)) if float(int(z)) == z else str(z)


def stations_from_stop_points_and_sequences(
    stop_points: list[dict[str, Any]],
    route_sequences: Iterable[dict[str, Any]],
    *,
    allowed_modes: tuple[str, ...] = ALLOWED_MODES_DEFAULT,
) -> pd.DataFrame:
    """Create a station table collapsed to one node per physical station (ICS id).

    We aggregate attributes across all NaPTAN stops that share the same `ics_id`.
    """

    allow = set(allowed_modes)

    # Validate once
    if not isinstance(stop_points, list) or any(not isinstance(x, dict) for x in stop_points):
        raise TypeError("stop_points must be a list[dict] from TfL StopPoint endpoint.")
    if any(not isinstance(rs, dict) for rs in route_sequences):
        raise TypeError(
            "route_sequences must be an iterable of dict payloads from TfL RouteSequence endpoint."
        )

    # Collect per-NaPTAN attributes from both sources (StopPoint + RouteSequence stations list)
    rows: list[dict[str, Any]] = []

    for sp2 in _iter_stop_points_recursive(stop_points):
        # StopPoint shape
        naptan = _first_non_empty(sp2.get("naptanId"), sp2.get("stationNaptan"), sp2.get("id"))
        if not naptan:
            continue
        ics = sp2.get("icsCode")
        if not ics:
            continue
        modes = [m for m in _as_list(sp2.get("modes")) if isinstance(m, str)]
        modes = sorted(set(modes) & allow)
        if not modes:
            continue
        name = _first_non_empty(sp2.get("commonName"), sp2.get("name"))
        lat = sp2.get("lat")
        lon = sp2.get("lon")
        stop_type = sp2.get("stopType")
        # Exclude interchange "hub" objects from being treated as stations.
        # These appear as stopType=TransportInterchange with their own icsCode but no track adjacency,
        # which otherwise creates isolated singleton nodes (e.g., HUBWWA / Woolwich Arsenal).
        if _is_transport_interchange(stop_type):
            continue
        hub = sp2.get("hubNaptanCode")
        rows.append(
            {
                "ics_id": str(ics),
                "naptan_id": str(naptan),
                "name": _as_str_or_none(name),
                "lat": _as_float_or_none(lat),
                "lon": _as_float_or_none(lon),
                "modes": ";".join(modes),
                "stop_type": _as_str_or_none(stop_type),
                "zone": "",
                "hub_naptan_code": _as_str_or_empty(hub),
                "ics_code": str(ics),
            }
        )

    for rs in route_sequences:
        if str(rs.get("mode") or "") not in allow:
            continue
        # Fail-fast on schema expectation: we need stopPointSequences for edges and stations coverage.
        # If a RouteSequence for an allowed mode lacks this key entirely, that's a likely schema change.
        if "stopPointSequences" not in rs:
            raise KeyError("RouteSequence payload missing stopPointSequences for an allowed mode.")
        for st in _as_list(rs.get("stations")):
            if not isinstance(st, dict):
                continue
            naptan = _first_non_empty(st.get("id"), st.get("stationId"))
            ics = st.get("icsId")
            if not naptan:
                continue
            if not ics:
                continue
            modes = [m for m in _as_list(st.get("modes")) if isinstance(m, str)]
            modes = sorted(set(modes) & allow)
            if not modes:
                continue
            name = st.get("name")
            lat = st.get("lat")
            lon = st.get("lon")
            stop_type = st.get("stopType")
            if _is_transport_interchange(stop_type):
                continue
            rows.append(
                {
                    "ics_id": str(ics),
                    "naptan_id": str(naptan),
                    "name": _as_str_or_none(name),
                    "lat": _as_float_or_none(lat),
                    "lon": _as_float_or_none(lon),
                    "modes": ";".join(modes),
                    "stop_type": _as_str_or_none(stop_type),
                    "zone": _as_str_or_empty(st.get("zone")),
                    "hub_naptan_code": "",
                    "ics_code": "" if ics is None else str(ics),
                }
            )

        # Also ingest stopPointSequences[*].stopPoint[*] for better coverage of rail/overground station ids.
        for sps in _as_list(rs.get("stopPointSequences")):
            if not isinstance(sps, dict):
                continue
            for sp in _as_list(sps.get("stopPoint")):
                if not isinstance(sp, dict):
                    continue
                naptan = _first_non_empty(sp.get("id"), sp.get("stationId"), sp.get("naptanId"))
                if not naptan:
                    continue
                ics = sp.get("icsId")
                if not ics:
                    continue
                modes = [m for m in _as_list(sp.get("modes")) if isinstance(m, str)]
                modes = sorted(set(modes) & allow)
                if not modes:
                    continue
                name = _first_non_empty(sp.get("name"), sp.get("commonName"))
                lat = sp.get("lat")
                lon = sp.get("lon")
                stop_type = sp.get("stopType")
                if _is_transport_interchange(stop_type):
                    continue
                hub = sp.get("topMostParentId") or sp.get("parentId")
                ics_code = sp.get("icsId")
                rows.append(
                    {
                        "ics_id": str(ics),
                        "naptan_id": str(naptan),
                        "name": _as_str_or_none(name),
                        "lat": _as_float_or_none(lat),
                        "lon": _as_float_or_none(lon),
                        "modes": ";".join(modes),
                        "stop_type": _as_str_or_none(stop_type),
                        "zone": _as_str_or_empty(sp.get("zone")),
                        "hub_naptan_code": _as_str_or_empty(hub),
                        "ics_code": _as_str_or_empty(ics_code),
                    }
                )

    _require_any(rows, name="stations_from_stop_points_and_sequences")
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw

    # Aggregate to one row per ICS id
    st = (
        raw.groupby("ics_id", as_index=False)
        .agg(
            station_id=("ics_id", "first"),
            name=("name", _agg_choose_name),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            modes=("modes", _agg_mode_union),
            stop_type=("stop_type", _agg_choose_name),
            zone=("zone", _agg_zone_min),
            naptan_ids=("naptan_id", lambda s: ";".join(sorted(set(s.astype(str))))),
            hub_naptan_code=("hub_naptan_code", _agg_choose_name),
            ics_ids=("ics_code", lambda s: ";".join(sorted(set([x for x in s.astype(str) if x])))),
        )
        .drop(columns=["ics_id"], errors="ignore")
    )

    st = st.sort_values(["station_id"], kind="mergesort").reset_index(drop=True)
    return st


def edges_from_route_sequences(
    route_sequences: Iterable[dict[str, Any]],
    *,
    allowed_modes: tuple[str, ...] = ALLOWED_MODES_DEFAULT,
) -> pd.DataFrame:
    """Build an undirected, deduplicated edge list from RouteSequence payloads.

    Uses RouteSequence `stopPointSequences[*].stopPoint[*].icsId` ordering as station ids.
    """

    allow = set(allowed_modes)
    agg: dict[tuple[str, str], EdgeAgg] = {}

    for rs in route_sequences:
        line_id = str(rs.get("lineId") or "")
        direction = str(rs.get("direction") or "")
        mode = str(rs.get("mode") or "")

        if mode and mode not in allow:
            continue

        # Prefer stopPointSequences ordering (station-level ids and parent hubs are consistent).
        sps_list = _as_list(rs.get("stopPointSequences"))
        for sps in sps_list:
            branch_id = sps.get("branchId")
            route_name = f"branch_{branch_id}" if branch_id is not None else ""
            stops = [x for x in _as_list(sps.get("stopPoint")) if isinstance(x, dict)]
            # Use ICS ids directly as node ids when available (preferred).
            node_ids = [str(sp.get("icsId") or "") for sp in stops]
            node_ids = [x for x in node_ids if x]
            if len(node_ids) < 2:
                continue

            for u, v in zip(node_ids[:-1], node_ids[1:]):
                if u == v:
                    continue
                k = _edge_key(u, v)
                if k not in agg:
                    agg[k] = EdgeAgg(set(), set(), set(), set())
                agg[k].line_ids.add(line_id)
                if mode:
                    agg[k].modes.add(mode)
                if direction:
                    agg[k].directions.add(direction)
                if route_name:
                    agg[k].route_names.add(route_name)

    rows: list[dict[str, Any]] = []
    for (u, v), a in agg.items():
        rows.append(
            {
                "u": u,
                "v": v,
                "line_ids": ";".join(sorted(x for x in a.line_ids if x)),
                "modes": ";".join(sorted(x for x in a.modes if x)),
                "directions": ";".join(sorted(x for x in a.directions if x)),
                "route_names": ";".join(sorted(x for x in a.route_names if x)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["u", "v"], kind="mergesort").reset_index(drop=True)
    return df


def compute_edge_distances_m(edges: pd.DataFrame, stations: pd.DataFrame) -> pd.Series:
    """Compute straight-line distances (meters) between edge endpoints using EPSG:27700."""
    st = stations.dropna(subset=["lon", "lat"]).copy()
    st_gdf = gpd.GeoDataFrame(
        st,
        geometry=[Point(xy) for xy in zip(st["lon"], st["lat"])],
        crs=CRS_INPUT,
    ).to_crs(CRS_PROJECTED)
    # IMPORTANT: avoid pandas index alignment; use raw arrays when setting a new index.
    xy = pd.DataFrame(
        {"x": st_gdf.geometry.x.to_numpy(), "y": st_gdf.geometry.y.to_numpy()},
        index=st_gdf["station_id"].to_numpy(),
    )

    u = edges["u"]
    v = edges["v"]
    ok = u.isin(xy.index) & v.isin(xy.index)
    dist = pd.Series(pd.NA, index=edges.index, dtype="Float64")
    if ok.any():
        dx = xy.loc[u[ok], "x"].to_numpy() - xy.loc[v[ok], "x"].to_numpy()
        dy = xy.loc[u[ok], "y"].to_numpy() - xy.loc[v[ok], "y"].to_numpy()
        dist.loc[ok.values] = (dx * dx + dy * dy) ** 0.5
    return dist


def filter_stations_by_geography(
    stations: pd.DataFrame, edges: pd.DataFrame, boundary_gpkg: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter stations to those within a geographic boundary polygon; keep edges with both endpoints."""
    if not boundary_gpkg.exists():
        raise FileNotFoundError(f"Boundary file not found at {boundary_gpkg}")

    st = stations.dropna(subset=["lon", "lat"]).copy()
    st_gdf = gpd.GeoDataFrame(
        st,
        geometry=[Point(xy) for xy in zip(st["lon"], st["lat"])],
        crs=CRS_INPUT,
    ).to_crs(CRS_PROJECTED)
    boundary = gpd.read_file(boundary_gpkg).to_crs(CRS_PROJECTED)
    boundary_poly = boundary.geometry.union_all()

    inside = st_gdf.geometry.within(boundary_poly)
    keep_ids = set(st_gdf.loc[inside, "station_id"])

    stations_filtered = stations[stations["station_id"].isin(keep_ids)].copy()
    edges_filtered = edges[edges["u"].isin(keep_ids) & edges["v"].isin(keep_ids)].copy()
    stations_filtered = stations_filtered.sort_values(["station_id"], kind="mergesort").reset_index(
        drop=True
    )
    edges_filtered = edges_filtered.sort_values(["u", "v"], kind="mergesort").reset_index(drop=True)
    return stations_filtered, edges_filtered


def compute_network_metrics(G: nx.Graph) -> pd.DataFrame:
    """Compute basic network connectivity metrics."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    comps = list(nx.connected_components(G))
    n_components = len(comps)
    gcc = max(comps, key=len) if comps else set()
    gcc_n = len(gcc)
    degs = [d for _, d in G.degree()]
    return pd.DataFrame(
        [
            {"metric": "N_nodes", "value": n},
            {"metric": "E_edges", "value": m},
            {"metric": "n_components", "value": n_components},
            {"metric": "gcc_nodes", "value": gcc_n},
            {"metric": "gcc_share", "value": (gcc_n / n) if n else 0.0},
            {"metric": "mean_degree", "value": float(sum(degs) / len(degs)) if degs else 0.0},
            {"metric": "median_degree", "value": float(pd.Series(degs).median()) if degs else 0.0},
            {"metric": "max_degree", "value": int(max(degs)) if degs else 0},
        ]
    )
