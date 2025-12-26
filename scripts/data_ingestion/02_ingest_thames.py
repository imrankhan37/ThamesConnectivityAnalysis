"""Step 1: Ingest Thames geometry (river centerline).

This project uses OS Open Rivers (GeoPackage)
You must manually download the GeoPackage and place it at:
  `data/raw/oprvrs_gb.gpkg`

Run:
  uv run python scripts/data_ingestion/02_ingest_thames.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

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
    ensure_unzipped,
    load_ingest_config,
    sha256_file,
    upsert_ingest_summary,
)

LOGGER = logging.getLogger("ingest_thames")

_PATHS = get_paths()
_INGEST_CFG = load_ingest_config(_PATHS.root / "config" / "ingest_config.yaml")

OS_OPEN_RIVERS_FILE = str(_INGEST_CFG["thames"]["os_open_rivers_file"])
OS_OPEN_RIVERS_LAYER = str(_INGEST_CFG["thames"]["os_open_rivers_layer"])
THAMES_OUTFILE = str(_INGEST_CFG["thames"]["outputs"]["centerline_geojson"])

# CLI arguments
ARG_FORCE = "--force"
ARG_CHECKPOINT = "--checkpoint"


def load_thames_from_os_open_rivers(paths) -> gpd.GeoDataFrame:
    """Load River Thames segments from OS Open Rivers GeoPackage."""
    src = ensure_unzipped(paths.data_raw / OS_OPEN_RIVERS_FILE)

    gdf = gpd.read_file(src, layer=OS_OPEN_RIVERS_LAYER)
    if gdf.empty:
        raise RuntimeError(
            f"{OS_OPEN_RIVERS_FILE} layer {OS_OPEN_RIVERS_LAYER} contains 0 features."
        )

    # OS Open Rivers provides `watercourse_name` and `watercourse_name_alternative`.
    mask = gdf["watercourse_name"].astype(str).str.contains("Thames", case=False, na=False) | gdf[
        "watercourse_name_alternative"
    ].astype(str).str.contains("Thames", case=False, na=False)
    thames = gdf[mask].copy()
    if thames.empty:
        raise RuntimeError("No Thames features found in OS Open Rivers (watercourse_link).")

    return thames.to_crs("EPSG:4326")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Thames geometry (merged centerline).")
    p.add_argument(
        ARG_FORCE, action="store_true", help="Rebuild output even if cached file exists."
    )
    p.add_argument(ARG_CHECKPOINT, action="store_true", help="Print checkpoint hashes/counts.")
    return p.parse_args()


def run(*, force: bool = False, checkpoint: bool = False) -> dict[str, int | str]:
    """Run the Thames ingest. Returns a small dict of key counts."""
    paths = get_paths()

    thames_source = f"OS Open Rivers ({OS_OPEN_RIVERS_FILE}:{OS_OPEN_RIVERS_LAYER})"
    thames_license = "OGL v3.0 (Ordnance Survey OpenData)"

    out_path = paths.data_raw / THAMES_OUTFILE

    def _read(p: Path) -> gpd.GeoDataFrame:
        return gpd.read_file(p)

    def _build() -> gpd.GeoDataFrame:
        os_gdf = load_thames_from_os_open_rivers(paths)
        geom = os_gdf.geometry.union_all()
        if isinstance(geom, LineString):
            mls = MultiLineString([geom])
        elif isinstance(geom, MultiLineString):
            mls = geom
        else:
            mls = MultiLineString([linemerge(geom)])

        return gpd.GeoDataFrame(
            pd.DataFrame(
                [
                    {
                        "name": "River Thames",
                        "source": thames_source,
                        "license": thames_license,
                        "bbox_wgs84": ",".join(map(str, LONDON_BBOX_WGS84)),
                    }
                ]
            ),
            geometry=[mls],
            crs="EPSG:4326",
        )

    def _write(gdf: gpd.GeoDataFrame, p: Path) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(gdf.to_json(drop_id=True), encoding="utf-8")

    gdf, used_cache = cached(
        out_path,
        force=force,
        read=_read,
        build=_build,
        write=_write,
        validate=lambda x: hasattr(x, "geometry") and len(x) >= 1,
    )

    geom0 = gdf.geometry.iloc[0]
    segs = len(list(geom0.geoms)) if hasattr(geom0, "geoms") else 1

    records = [
        IngestRecord(
            dataset="thames_centerline",
            stage="raw",
            path=str(out_path.relative_to(paths.root)),
            rows=len(gdf),
            cols=len(gdf.columns),
            crs=str(gdf.crs),
            bytes=out_path.stat().st_size if out_path.exists() else None,
            source=thames_source,
            notes=(
                "Thames geometry stored as merged MultiLineString."
                f" License: {thames_license}. used_cache={int(used_cache)}"
            ),
        )
    ]
    upsert_ingest_summary(records, paths.processed_meta / "ingest_summary.csv")
    LOGGER.info("Wrote/updated %s", paths.processed_meta / "ingest_summary.csv")

    out2: dict[str, int | str] = {
        "thames_segments": int(segs),
        "thames_used_cache": int(used_cache),
    }
    if checkpoint:
        out2["sha_thames_centerline_geojson"] = sha256_file(out_path)
        LOGGER.info("CHECKPOINT: %s", out2)
    return out2


def main() -> None:
    configure_logging()
    args = _parse_args()
    run(force=args.force, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
