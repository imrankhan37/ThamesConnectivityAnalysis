"""Equity analysis helpers

Builds an LSOA-level analysis frame by assigning each London LSOA centroid to its
nearest station, then joining station-level dependence/vulnerability metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from src.core.config import CRS_BNG, CRS_WGS84


def project_points_to_bng(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    crs_from: str = CRS_WGS84,
    crs_to: str = CRS_BNG,
) -> tuple[np.ndarray, np.ndarray]:
    """Project arrays of lon/lat to BNG x/y (meters)."""
    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def build_station_kdtree_bng(
    stations: pd.DataFrame,
    *,
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> tuple[cKDTree, pd.DataFrame]:
    """Return (KDTree, stations_xy) where stations_xy includes bng_x/bng_y."""
    required = {"station_id", lon_col, lat_col}
    missing = required - set(stations.columns)
    if missing:
        raise ValueError(f"stations missing required columns: {sorted(missing)}")

    st = stations.copy()

    st["station_id"] = st["station_id"].astype(str)
    x, y = project_points_to_bng(
        st[lon_col].astype(float).to_numpy(), st[lat_col].astype(float).to_numpy()
    )
    st["bng_x"] = x
    st["bng_y"] = y
    tree = cKDTree(st[["bng_x", "bng_y"]].to_numpy())
    return tree, st


def assign_nearest_station(
    lsoa: pd.DataFrame,
    stations_xy: pd.DataFrame,
    tree: cKDTree,
    *,
    lsoa_x_col: str = "bng_e",
    lsoa_y_col: str = "bng_n",
) -> pd.DataFrame:
    """Assign nearest station_id to each LSOA row using BNG coordinates."""
    required_lsoa = {lsoa_x_col, lsoa_y_col}
    missing = required_lsoa - set(lsoa.columns)
    if missing:
        raise ValueError(f"lsoa missing required columns: {sorted(missing)}")

    df = lsoa.copy()
    if df[[lsoa_x_col, lsoa_y_col]].isna().any(axis=None):
        bad = df[df[[lsoa_x_col, lsoa_y_col]].isna().any(axis=1)][[lsoa_x_col, lsoa_y_col]].head(5)
        raise ValueError(f"lsoa has missing BNG coordinates (e.g. {bad.to_dict(orient='records')})")

    q = df[[lsoa_x_col, lsoa_y_col]].astype(float).to_numpy()
    dist, idx = tree.query(q, k=1)
    idx = idx.astype(int)
    nearest = stations_xy.iloc[idx][["station_id"]].reset_index(drop=True)

    df["nearest_station_id"] = nearest["station_id"].astype(str).to_numpy()
    df["dist_to_nearest_station_m"] = dist.astype(float)
    return df


def distance_to_centre_m(
    lsoa: pd.DataFrame,
    *,
    centre_lon: float,
    centre_lat: float,
    lsoa_x_col: str = "bng_e",
    lsoa_y_col: str = "bng_n",
) -> pd.Series:
    """Euclidean distance from LSOA centroid to a fixed centre point (meters)."""
    cx, cy = project_points_to_bng(np.array([centre_lon]), np.array([centre_lat]))
    cx0, cy0 = float(cx[0]), float(cy[0])

    if lsoa[[lsoa_x_col, lsoa_y_col]].isna().any(axis=None):
        raise ValueError("lsoa has missing BNG coordinates")
    x = lsoa[lsoa_x_col].astype(float)
    y = lsoa[lsoa_y_col].astype(float)
    return np.sqrt((x - cx0) ** 2 + (y - cy0) ** 2)
