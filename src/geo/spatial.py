"""Spatial helpers: joins, bank classification, and Thames crossing detection."""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import nearest_points, unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree

from src.core.config import CRS_BNG, CRS_WGS84

LOGGER = logging.getLogger(__name__)


def stations_to_gdf(stations: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert `data/processed/stations.csv` to a GeoDataFrame in WGS84."""
    required = {"station_id", "lon", "lat"}
    missing = required - set(stations.columns)
    if missing:
        raise ValueError(f"stations missing required columns: {sorted(missing)}")
    if stations[["lon", "lat"]].isna().any(axis=None):
        bad = stations[stations[["lon", "lat"]].isna().any(axis=1)][
            ["station_id", "lon", "lat"]
        ].head(5)
        raise ValueError(
            f"stations has missing lon/lat values (e.g. {bad.to_dict(orient='records')})"
        )

    df = stations.copy()
    df["station_id"] = df["station_id"].astype(str)
    geom = gpd.points_from_xy(df["lon"].astype(float), df["lat"].astype(float))
    return gpd.GeoDataFrame(df, geometry=geom, crs=CRS_WGS84)


def spatial_join_stations_to_lsoa(
    *,
    stations_gdf_wgs84: gpd.GeoDataFrame,
    lsoa_gdf_wgs84: gpd.GeoDataFrame,
    lsoa_code_col: str = "LSOA21CD",
    nearest_max_m: float = 750.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Assign stations to LSOA polygons using point-in-polygon, with nearest fallback."""
    if stations_gdf_wgs84.crs is None:
        raise ValueError("stations_gdf_wgs84.crs is None; expected WGS84")
    if lsoa_gdf_wgs84.crs is None:
        raise ValueError("lsoa_gdf_wgs84.crs is None; expected WGS84")
    if str(stations_gdf_wgs84.crs) != CRS_WGS84:
        raise ValueError(
            f"stations_gdf_wgs84 CRS must be {CRS_WGS84}, got {stations_gdf_wgs84.crs}"
        )
    if str(lsoa_gdf_wgs84.crs) != CRS_WGS84:
        raise ValueError(f"lsoa_gdf_wgs84 CRS must be {CRS_WGS84}, got {lsoa_gdf_wgs84.crs}")

    stations = stations_gdf_wgs84.to_crs(CRS_BNG)
    lsoa = lsoa_gdf_wgs84.to_crs(CRS_BNG)

    pip = gpd.sjoin(
        stations[["station_id", "geometry"]],
        lsoa[[lsoa_code_col, "geometry"]],
        how="left",
        predicate="within",
    )
    pip = pip.drop(columns=[c for c in pip.columns if c.startswith("index_")], errors="ignore")

    out = pd.DataFrame(
        {
            "station_id": pip["station_id"].astype(str),
            "lsoa_code": pip[lsoa_code_col].astype("string"),
            "method": np.where(pip[lsoa_code_col].notna(), "pip", pd.NA),
            "distance_m": np.where(pip[lsoa_code_col].notna(), 0.0, np.nan),
        }
    )

    missing = out["lsoa_code"].isna()
    if missing.any():
        st_miss = stations.loc[missing.values, ["station_id", "geometry"]].copy()
        nearest = gpd.sjoin_nearest(
            st_miss,
            lsoa[[lsoa_code_col, "geometry"]],
            how="left",
            distance_col="distance_m",
        )
        nearest = nearest.drop(
            columns=[c for c in nearest.columns if c.startswith("index_")], errors="ignore"
        )
        within_thresh = nearest["distance_m"].astype(float) <= float(nearest_max_m)

        fill_ids = nearest.loc[within_thresh, "station_id"].astype(str).tolist()
        fill_mask = out["station_id"].isin(fill_ids)
        fill_map = dict(
            zip(
                nearest.loc[within_thresh, "station_id"].astype(str),
                nearest.loc[within_thresh, lsoa_code_col].astype(str),
                strict=True,
            )
        )
        fill_dist = dict(
            zip(
                nearest.loc[within_thresh, "station_id"].astype(str),
                nearest.loc[within_thresh, "distance_m"].astype(float),
                strict=True,
            )
        )

        out.loc[fill_mask, "lsoa_code"] = out.loc[fill_mask, "station_id"].map(fill_map)
        out.loc[fill_mask, "method"] = "nearest"
        out.loc[fill_mask, "distance_m"] = out.loc[fill_mask, "station_id"].map(fill_dist)

    out["lsoa_code"] = out["lsoa_code"].astype("string")
    out["method"] = out["method"].astype("string")

    n_total = len(out)
    n_pip = int((out["method"] == "pip").sum())
    n_near = int((out["method"] == "nearest").sum())
    n_unmatched = int(out["lsoa_code"].isna().sum())

    qa = {
        "n_stations_total": float(n_total),
        "n_pip": float(n_pip),
        "n_nearest": float(n_near),
        "n_unmatched": float(n_unmatched),
        "share_matched": float((n_total - n_unmatched) / n_total) if n_total else 0.0,
        "nearest_max_m": float(nearest_max_m),
    }
    return out.sort_values(["station_id"], kind="mergesort").reset_index(drop=True), qa


def _to_multiline(geom) -> MultiLineString:
    if isinstance(geom, MultiLineString):
        return geom
    if isinstance(geom, LineString):
        return MultiLineString([geom])
    raise TypeError(f"Expected LineString/MultiLineString, got {type(geom)}")


def classify_station_bank_by_local_orientation(
    *,
    stations_wgs84: gpd.GeoDataFrame,
    river_centerline_wgs84: gpd.GeoDataFrame,
    overrides: dict[str, str] | None = None,
    bank_col: str = "bank",
) -> pd.DataFrame:
    """Classify stations as north/south bank using local river orientation."""
    overrides = {} if overrides is None else overrides

    st = stations_wgs84.to_crs(CRS_BNG).copy()
    st["station_id"] = st["station_id"].astype(str)
    river_bng = river_centerline_wgs84.to_crs(CRS_BNG)
    if river_bng.empty:
        raise ValueError("river_centerline is empty")

    river_geom = unary_union(list(river_bng.geometry))
    river_mls = _to_multiline(river_geom)

    seg_lines: list[LineString] = []
    for ls in river_mls.geoms:
        coords = list(ls.coords)
        for a, b in zip(coords[:-1], coords[1:], strict=True):
            # Skip degenerate segments
            if a != b:
                seg_lines.append(LineString([a, b]))
    if not seg_lines:
        raise RuntimeError("River geometry produced 0 usable segments.")

    tree = STRtree(seg_lines)

    bank: list[str] = []
    nearest_y: list[float] = []
    for sid, pt in zip(st["station_id"].tolist(), st.geometry, strict=True):
        if sid in overrides:
            bank.append(overrides[sid])
            nearest_y.append(np.nan)
            continue

        # Find nearest river segment via spatial index
        seg = seg_lines[int(tree.nearest(pt))]

        # Recover endpoints for orientation from the segment coordinates.
        (x0, y0), (x1, y1) = list(seg.coords)[:2]
        p_r = nearest_points(pt, seg)[1]
        nearest_y.append(float(p_r.y))

        tx = float(x1 - x0)
        ty = float(y1 - y0)
        if tx < 0:
            tx, ty = -tx, -ty
        px, py = -ty, tx
        vx, vy = float(pt.x - p_r.x), float(pt.y - p_r.y)
        s = px * vx + py * vy
        bank.append("north" if s >= 0 else "south")

    out = pd.DataFrame(
        {
            "station_id": st["station_id"].tolist(),
            bank_col: pd.Series(bank, dtype="string"),
            "nearest_river_northing_m": nearest_y,
        }
    )
    return out.sort_values(["station_id"], kind="mergesort").reset_index(drop=True)


def label_thames_crossing_edges(
    *,
    edges: pd.DataFrame,
    stations_bng: gpd.GeoDataFrame,
    station_bank: pd.DataFrame,
    river_centerline_wgs84: gpd.GeoDataFrame,
    buffer_m: float = 75.0,
) -> pd.DataFrame:
    """Label edges as Thames crossings."""
    if not {"u", "v"}.issubset(edges.columns):
        raise ValueError("edges must have 'u' and 'v' columns")
    if "station_id" not in stations_bng.columns or "geometry" not in stations_bng.columns:
        raise ValueError("stations_bng must include 'station_id' and 'geometry'")
    if not {"station_id", "bank"}.issubset(station_bank.columns):
        raise ValueError("station_bank must include 'station_id' and 'bank'")

    st = stations_bng.copy()
    st["station_id"] = st["station_id"].astype(str)
    st = st.set_index("station_id")

    e = edges.copy()
    e["u"] = e["u"].astype(str)
    e["v"] = e["v"].astype(str)

    missing_st = sorted(set(e["u"]).union(set(e["v"])) - set(st.index))
    if missing_st:
        raise ValueError(f"edges reference unknown station_id(s) (e.g. {missing_st[:10]})")

    bank_df = station_bank.copy()
    bank_df["station_id"] = bank_df["station_id"].astype(str)
    bank_map = bank_df.set_index("station_id")["bank"].to_dict()
    missing_bank = sorted(set(e["u"]).union(set(e["v"])) - set(bank_map.keys()))
    if missing_bank:
        raise ValueError(f"Missing bank labels for station(s) (e.g. {missing_bank[:10]})")
    bad_bank = sorted(
        {
            str(bank_map[s])
            for s in set(e["u"]).union(set(e["v"]))
            if str(bank_map[s]) not in {"north", "south"}
        }
    )
    if bad_bank:
        raise ValueError(f"Invalid bank labels (expected 'north'/'south'): {bad_bank}")

    river_bng = river_centerline_wgs84.to_crs(CRS_BNG)
    river_geom = unary_union(list(river_bng.geometry))
    river_mls = _to_multiline(river_geom)
    river_buf = river_mls.buffer(float(buffer_m))
    river_buf_prepared = prep(river_buf)

    rows: list[dict[str, object]] = []
    for u, v in e[["u", "v"]].itertuples(index=False, name=None):
        bu = str(bank_map[u])
        bv = str(bank_map[v])
        opp = bu != bv

        p0 = st.loc[u].geometry
        p1 = st.loc[v].geometry
        seg = LineString([(float(p0.x), float(p0.y)), (float(p1.x), float(p1.y))])
        intersects = bool(river_buf_prepared.intersects(seg))
        is_crossing = bool(opp and intersects)

        rows.append(
            {
                "u": u,
                "v": v,
                "bank_u": bu,
                "bank_v": bv,
                "opposite_banks": opp,
                "intersects_river_buffer": intersects,
                "buffer_m": float(buffer_m),
                "is_thames_crossing": is_crossing,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["u", "v"], kind="mergesort").reset_index(drop=True)
    return out
