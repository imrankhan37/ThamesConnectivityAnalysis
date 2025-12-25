"""Step 3: Ingest LSOA 2021 boundaries (London bbox subset) + IMD 2025 (London subset).

Run:
  uv run python scripts/data_ingestion/03_ingest_lsoa_imd.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Ensure repo root is on sys.path so `import src...` works when executing this file directly.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from src.core.config import LONDON_BBOX_WGS84, configure_logging, get_paths
from src.io import (
    IngestRecord,
    cached,
    download_file,
    get_json,
    infer_lsoa_code_column,
    load_ingest_config,
    sha256_file,
    upsert_ingest_summary,
)

LOGGER = logging.getLogger("ingest_lsoa_imd")

_PATHS = get_paths()
_INGEST_CFG = load_ingest_config(_PATHS.root / "config" / "ingest_config.yaml")

# Sources
LSOA_FEATURESERVER = str(_INGEST_CFG["lsoa_imd"]["sources"]["lsoa_featureserver"])
REGIONS_FEATURESERVER_0 = str(_INGEST_CFG["lsoa_imd"]["sources"]["regions_featureserver_0"])
LONDON_REGION_CODE = str(_INGEST_CFG["lsoa_imd"]["sources"]["london_region_code"])
IMD_FILE7_URL = str(_INGEST_CFG["lsoa_imd"]["sources"]["imd_file7_url"])

# ArcGIS fetch parameters
ARCGIS_CHUNK = int(_INGEST_CFG["lsoa_imd"]["arcgis"]["chunk"])
ARCGIS_TIMEOUT_COUNT_S = int(_INGEST_CFG["lsoa_imd"]["arcgis"]["timeouts_seconds"]["count"])
ARCGIS_TIMEOUT_PAGE_S = int(_INGEST_CFG["lsoa_imd"]["arcgis"]["timeouts_seconds"]["page"])
ARCGIS_TIMEOUT_WHERE_S = int(_INGEST_CFG["lsoa_imd"]["arcgis"]["timeouts_seconds"]["where"])

# Output filenames
REGIONS_OUTFILE = str(_INGEST_CFG["lsoa_imd"]["outputs"]["regions_gpkg"])
REGIONS_LAYER = str(_INGEST_CFG["lsoa_imd"]["outputs"]["regions_layer"])
LSOA_OUTFILE = str(_INGEST_CFG["lsoa_imd"]["outputs"]["lsoa_gpkg"])
LSOA_LAYER = str(_INGEST_CFG["lsoa_imd"]["outputs"]["lsoa_layer"])
IMD_RAW_OUTFILE = str(_INGEST_CFG["lsoa_imd"]["outputs"]["imd_raw_csv"])
IMD_LONDON_OUTFILE = str(_INGEST_CFG["lsoa_imd"]["outputs"]["imd_london_csv"])

# CLI arguments
ARG_FORCE = "--force"
ARG_CHECKPOINT = "--checkpoint"


def arcgis_query_geojson_bbox_paged(
    layer_url: str,
    bbox_wgs84: tuple[float, float, float, float],
    *,
    out_fields: str = "*",
    chunk: int = 1000,
) -> dict:
    """Query an ArcGIS FeatureServer layer to GeoJSON within bbox, paging via resultOffset."""
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    geometry = {
        "xmin": min_lon,
        "ymin": min_lat,
        "xmax": max_lon,
        "ymax": max_lat,
        "spatialReference": {"wkid": 4326},
    }
    geom_str = json.dumps(geometry)

    count_params = {
        "f": "json",
        "where": "1=1",
        "geometry": geom_str,
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "4326",
        "returnCountOnly": "true",
    }
    count_json = get_json(f"{layer_url}/query", params=count_params, timeout=ARCGIS_TIMEOUT_COUNT_S)
    count = int(count_json.get("count", 0))
    if count <= 0:
        raise RuntimeError("ArcGIS bbox count returned 0 features.")
    LOGGER.info("ArcGIS bbox feature count: %d", count)

    features: list[dict] = []
    for offset in range(0, count, chunk):
        params = {
            "f": "geojson",
            "where": "1=1",
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": "4326",
            "geometry": geom_str,
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": "4326",
            "resultOffset": offset,
            "resultRecordCount": chunk,
        }
        gj = get_json(f"{layer_url}/query", params=params, timeout=ARCGIS_TIMEOUT_PAGE_S)
        batch = gj.get("features") or []
        features.extend(batch)
        LOGGER.info("Fetched %d/%d features", min(offset + chunk, count), count)

        # Early exit if the service returns fewer than requested and doesn't signal transfer limit.
        if len(batch) < chunk and not gj.get("properties", {}).get("exceededTransferLimit", False):
            break

    return {"type": "FeatureCollection", "features": features}


def arcgis_query_geojson_where(layer_url: str, where: str, *, out_fields: str = "*") -> dict:
    """Small GeoJSON query helper (no paging) for small layers (e.g., 9 regions)."""
    params = {
        "f": "geojson",
        "where": where,
        "outFields": out_fields,
        "returnGeometry": "true",
        "outSR": "4326",
    }
    return get_json(f"{layer_url}/query", params=params, timeout=ARCGIS_TIMEOUT_WHERE_S)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest LSOA 2021 boundaries (London) + IMD 2025 (London subset)."
    )
    p.add_argument(
        ARG_FORCE, action="store_true", help="Re-fetch/rebuild outputs even if cached files exist."
    )
    p.add_argument(ARG_CHECKPOINT, action="store_true", help="Print checkpoint hashes/counts.")
    return p.parse_args()


def run(*, force: bool = False, checkpoint: bool = False) -> dict[str, int | str]:
    """Run the LSOA+IMD ingest. Returns a small dict of key counts."""
    paths = get_paths()

    # --- London region boundary (standard definition) ---
    regions_out = paths.data_raw / REGIONS_OUTFILE

    def _read_regions(p: Path) -> gpd.GeoDataFrame:
        return gpd.read_file(p, layer=REGIONS_LAYER)

    def _build_regions() -> gpd.GeoDataFrame:
        LOGGER.info("Fetching London region boundary (RGN21CD=%s)...", LONDON_REGION_CODE)
        regions_gj = arcgis_query_geojson_where(
            REGIONS_FEATURESERVER_0, f"RGN21CD='{LONDON_REGION_CODE}'"
        )
        return gpd.GeoDataFrame.from_features(regions_gj, crs="EPSG:4326")

    def _write_regions(gdf: gpd.GeoDataFrame, p: Path) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(p, layer=REGIONS_LAYER, driver="GPKG")

    regions_gdf, regions_used_cache = cached(
        regions_out,
        force=force,
        read=_read_regions,
        build=_build_regions,
        write=_write_regions,
        validate=lambda x: len(x) > 0,
    )
    if regions_used_cache:
        LOGGER.info("Loaded cached London region boundary: %s", regions_out)

    if len(regions_gdf) != 1:
        raise RuntimeError(f"Expected exactly 1 London region feature, got {len(regions_gdf)}")
    london_geom = regions_gdf.geometry.iloc[0]

    # --- LSOA boundaries (bbox subset) ---
    # Use bbox to reduce transfer, then filter strictly by the London region polygon.
    lsoa_out = paths.data_raw / LSOA_OUTFILE

    def _read_lsoa(p: Path) -> gpd.GeoDataFrame:
        return gpd.read_file(p, layer=LSOA_LAYER)

    def _build_lsoa() -> gpd.GeoDataFrame:
        LOGGER.info(
            "Querying LSOA boundaries via ArcGIS FeatureServer (bbox prefilter, then London region mask)..."
        )
        lsoa_geojson = arcgis_query_geojson_bbox_paged(
            LSOA_FEATURESERVER, LONDON_BBOX_WGS84, chunk=ARCGIS_CHUNK
        )
        lsoa_gdf = gpd.GeoDataFrame.from_features(lsoa_geojson, crs="EPSG:4326")
        lsoa_gdf = lsoa_gdf[lsoa_gdf.intersects(london_geom)].copy()
        LOGGER.info("LSOA polygons fetched (bbox subset): %d", len(lsoa_gdf))
        return lsoa_gdf

    def _write_lsoa(gdf: gpd.GeoDataFrame, p: Path) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(p, layer=LSOA_LAYER, driver="GPKG")

    lsoa_gdf, lsoa_used_cache = cached(
        lsoa_out,
        force=force,
        read=_read_lsoa,
        build=_build_lsoa,
        write=_write_lsoa,
        validate=lambda x: len(x) > 0,
    )
    if lsoa_used_cache:
        LOGGER.info("Loaded cached LSOA polygons: %s", lsoa_out)

    # Identify expected code column for later joins (best-effort)
    lsoa_code_cols = [
        c for c in lsoa_gdf.columns if c.lower().endswith("cd") and "lsoa" in c.lower()
    ]
    lsoa_code_col = lsoa_code_cols[0] if lsoa_code_cols else None
    if lsoa_code_col is None:
        LOGGER.warning("Could not find a clear LSOA code column in LSOA polygons.")

    # --- IMD 2025 ---
    imd_raw = paths.data_raw / IMD_RAW_OUTFILE
    download_file(IMD_FILE7_URL, imd_raw, timeout=120)
    imd_df = pd.read_csv(imd_raw)
    LOGGER.info("IMD file7 rows: %d, cols: %d", len(imd_df), len(imd_df.columns))

    imd_lsoa_col = infer_lsoa_code_column(imd_df)
    LOGGER.info("IMD LSOA code column inferred: %s", imd_lsoa_col)

    if lsoa_code_col:
        london_codes = set(lsoa_gdf[lsoa_code_col].astype(str))
        imd_london = imd_df[imd_df[imd_lsoa_col].astype(str).isin(london_codes)].copy()
    else:
        imd_london = imd_df.copy()
        LOGGER.warning("No LSOA code col found in polygons; not subsetting IMD.")

    imd_out = paths.processed_boundaries / IMD_LONDON_OUTFILE
    imd_out.parent.mkdir(parents=True, exist_ok=True)
    imd_london.to_csv(imd_out, index=False)
    LOGGER.info("IMD London subset rows: %d", len(imd_london))

    records = [
        IngestRecord(
            dataset="ons_regions_2021_en_bgc_london",
            stage="raw",
            path=str(regions_out.relative_to(paths.root)),
            rows=len(regions_gdf),
            cols=len(regions_gdf.columns),
            crs=str(regions_gdf.crs),
            bytes=regions_out.stat().st_size if regions_out.exists() else None,
            source=REGIONS_FEATURESERVER_0,
            notes=f"London region polygon (RGN21CD={LONDON_REGION_CODE}) used as standardized study area mask.",
        ),
        IngestRecord(
            dataset="lsoa_2021_ew_bfe_v10_london_bbox",
            stage="raw",
            path=str(lsoa_out.relative_to(paths.root)),
            rows=len(lsoa_gdf),
            cols=len(lsoa_gdf.columns),
            crs=str(lsoa_gdf.crs),
            bytes=lsoa_out.stat().st_size if lsoa_out.exists() else None,
            source="ONS Open Geography Portal FeatureServer (bbox query)",
            notes=f"Subset fetched by bbox={LONDON_BBOX_WGS84}.",
        ),
        IngestRecord(
            dataset="imd_2025_file7",
            stage="raw",
            path=str(imd_raw.relative_to(paths.root)),
            rows=len(imd_df),
            cols=len(imd_df.columns),
            crs=None,
            bytes=imd_raw.stat().st_size if imd_raw.exists() else None,
            source=IMD_FILE7_URL,
            notes="IMD 2025 File 7 (all ranks/scores/deciles) downloaded as CSV.",
        ),
        IngestRecord(
            dataset="imd_2025_london",
            stage="processed",
            path=str(imd_out.relative_to(paths.root)),
            rows=len(imd_london),
            cols=len(imd_london.columns),
            crs=None,
            bytes=imd_out.stat().st_size if imd_out.exists() else None,
            source=IMD_FILE7_URL,
            notes="Filtered IMD rows to LSOAs intersecting London bbox subset polygons.",
        ),
    ]
    upsert_ingest_summary(records, paths.processed_meta / "ingest_summary.csv")
    LOGGER.info("Wrote/updated %s", paths.processed_meta / "ingest_summary.csv")

    out: dict[str, int | str] = {"lsoa_polygons": len(lsoa_gdf), "imd_london_rows": len(imd_london)}
    if checkpoint:
        out["sha_regions_gpkg"] = sha256_file(regions_out)
        out["sha_lsoa_gpkg"] = sha256_file(lsoa_out)
        out["sha_imd_london_csv"] = sha256_file(imd_out)
        LOGGER.info("CHECKPOINT: %s", out)
    return out


def main() -> None:
    configure_logging()
    args = _parse_args()
    run(force=args.force, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
