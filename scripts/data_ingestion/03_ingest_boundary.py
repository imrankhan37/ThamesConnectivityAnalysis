"""Step 3: Ingest London boundary polygon (ONS Regions 2021) from ArcGIS FeatureServer.

Run:
  uv run python scripts/data_ingestion/03_ingest_boundary.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd

# Ensure repo root is on sys.path so `import src...` works when executing this file directly.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from src.core.config import configure_logging, get_paths
from src.io import (
    IngestRecord,
    cached,
    get_json,
    load_ingest_config,
    sha256_file,
    upsert_ingest_summary,
)

LOGGER = logging.getLogger("ingest_boundary")

_PATHS = get_paths()
_INGEST_CFG = load_ingest_config(_PATHS.root / "config" / "ingest_config.yaml")

# Sources
REGIONS_FEATURESERVER_0 = str(_INGEST_CFG["boundary"]["sources"]["regions_featureserver_0"])
LONDON_REGION_CODE = str(_INGEST_CFG["boundary"]["sources"]["london_region_code"])

# ArcGIS timeouts
ARCGIS_TIMEOUT_WHERE_S = int(_INGEST_CFG["boundary"]["arcgis"]["timeouts_seconds"]["where"])

# Outputs
REGIONS_OUTFILE = str(_INGEST_CFG["boundary"]["outputs"]["regions_gpkg"])
REGIONS_LAYER = str(_INGEST_CFG["boundary"]["outputs"]["regions_layer"])


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
    p = argparse.ArgumentParser(description="Ingest London boundary polygon (ONS Regions 2021).")
    p.add_argument(
        "--force", action="store_true", help="Re-fetch/rebuild output even if cached exists."
    )
    p.add_argument("--checkpoint", action="store_true", help="Print checkpoint hashes/counts.")
    return p.parse_args()


def run(*, force: bool = False, checkpoint: bool = False) -> dict[str, int | str]:
    """Run the boundary ingest. Returns a small dict of key counts."""
    paths = get_paths()
    out_path = paths.data_raw / REGIONS_OUTFILE

    def _read(p: Path) -> gpd.GeoDataFrame:
        return gpd.read_file(p, layer=REGIONS_LAYER)

    def _build() -> gpd.GeoDataFrame:
        LOGGER.info("Fetching London region boundary (RGN21CD=%s)...", LONDON_REGION_CODE)
        regions_gj = arcgis_query_geojson_where(
            REGIONS_FEATURESERVER_0, f"RGN21CD='{LONDON_REGION_CODE}'"
        )
        return gpd.GeoDataFrame.from_features(regions_gj, crs="EPSG:4326")

    def _write(gdf: gpd.GeoDataFrame, p: Path) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(p, layer=REGIONS_LAYER, driver="GPKG")

    regions_gdf, used_cache = cached(
        out_path,
        force=force,
        read=_read,
        build=_build,
        write=_write,
        validate=lambda x: len(x) > 0,
    )
    if used_cache:
        LOGGER.info("Loaded cached London boundary: %s", out_path)

    if len(regions_gdf) != 1:
        raise RuntimeError(f"Expected exactly 1 London region feature, got {len(regions_gdf)}")

    record = IngestRecord(
        dataset="ons_regions_2021_en_bgc_london",
        stage="raw",
        path=str(out_path.relative_to(paths.root)),
        rows=len(regions_gdf),
        cols=len(regions_gdf.columns),
        crs=str(regions_gdf.crs),
        bytes=out_path.stat().st_size if out_path.exists() else None,
        source=REGIONS_FEATURESERVER_0,
        notes=f"London region polygon (RGN21CD={LONDON_REGION_CODE}) used as standardized study area mask.",
    )
    upsert_ingest_summary([record], paths.processed_meta / "ingest_summary.csv")
    LOGGER.info("Wrote/updated %s", paths.processed_meta / "ingest_summary.csv")

    out: dict[str, int | str] = {"regions_rows": len(regions_gdf)}
    if checkpoint:
        out["sha_regions_gpkg"] = sha256_file(out_path)
        LOGGER.info("CHECKPOINT: %s", out)
    return out


def main() -> None:
    configure_logging()
    args = _parse_args()
    run(force=args.force, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
