"""Step 1: Ingest TfL transit topology (stations + route sequences).

This script fetches Transport for London (TfL) data including:
- Station-like stop points filtered to relevant transport modes
- Line metadata for included transport modes
- Route sequence data for each line and direction

Run:
  uv run python scripts/data_ingestion/01_ingest_transit.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import src...` works when executing this file directly.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

import pandas as pd
from src.core.cli_utils import create_base_parser
from src.core.config import configure_logging, get_paths
from src.io import (
    IngestRecord,
    cached,
    get_json,
    load_ingest_config,
    read_json,
    upsert_ingest_summary,
    write_json,
)

LOGGER = logging.getLogger("ingest_transit")

# Central ingest config (YAML)
_PATHS = get_paths()
_INGEST_CFG = load_ingest_config(_PATHS.root / "config" / "ingest_config.yaml")

# TfL Unified API
TFL_BASE = str(_INGEST_CFG["tfl"]["base_url"])
STOPPOINT_URL_TEMPLATE = TFL_BASE + "/StopPoint/Type/{stop_type}/page/1"
LINE_MODE_URL_TEMPLATE = TFL_BASE + "/Line/Mode/{mode}"
ROUTE_SEQUENCE_URL_TEMPLATE = TFL_BASE + "/Line/{line_id}/Route/Sequence/{direction}"
ROUTE_SEQUENCE_PARAMS: dict[str, str] = dict(_INGEST_CFG["tfl"]["route_sequence"]["params"])

# Project scope
INCLUDE_MODES = set(_INGEST_CFG["tfl"]["include_modes"])

# StopPoint types (station-like)
STOPPOINT_STATION_TYPES = list(_INGEST_CFG["tfl"]["stoppoint_station_types"])

# Output locations
STOPPOINTS_OUTFILE = str(_INGEST_CFG["tfl"]["outputs"]["stoppoints_json"])
LINES_OUTFILE = str(_INGEST_CFG["tfl"]["outputs"]["lines_by_mode_json"])
ROUTE_SEQUENCE_DIR = str(_INGEST_CFG["tfl"]["outputs"]["route_sequences_dir"])


def fetch_stop_points(*, force: bool = False) -> tuple[list[dict], Path, bool]:
    """Fetch TfL *StopPoints* that represent stations (then filter to our project modes).

    What is a TfL "StopPoint"?
    - In TfL's Unified API, a `StopPoint` is a generic location where passengers can access transport.
      It can represent many things (stations, piers, bus stops, etc.). Each StopPoint has:
      - an `id` (NaPTAN-style identifier),
      - a `stopType` (what kind of stop it is),
      - coordinates (`lat`, `lon`),
      - and a list of served `modes` (e.g., `tube`, `dlr`, `overground`, `elizabeth-line`).

    This function:
    - Queries only 'station-like' StopPoint types:
      - `NaptanMetroStation` and `NaptanRailStation`
    - Filters the returned StopPoints to those serving at least one of `INCLUDE_MODES`.
    - Saves the result to `data/raw/tfl_stop_points_stations.json`

    Returns:
        - stop_points: list[dict] - The filtered list of StopPoints.
        - out_path: Path - The path to the saved JSON file.
        - used_cache: bool - Whether the cache was used.
    """
    paths = get_paths()
    out_path = paths.data_raw / STOPPOINTS_OUTFILE

    def _build() -> list[dict]:
        stop_points: list[dict] = []
        for stop_type in STOPPOINT_STATION_TYPES:
            url = STOPPOINT_URL_TEMPLATE.format(stop_type=stop_type)
            data = get_json(url)
            if not isinstance(data, list):
                raise RuntimeError(
                    f"Unexpected StopPoint response type for {stop_type}: {type(data)}"
                )
            stop_points.extend(data)

        # filter by modes intersection
        return [sp for sp in stop_points if set(sp.get("modes") or []).intersection(INCLUDE_MODES)]

    stop_points, used_cache = cached(
        out_path,
        force=force,
        read=read_json,
        build=_build,
        write=lambda obj, p: write_json(obj, p),
        validate=lambda obj: isinstance(obj, list),
    )
    return stop_points, out_path, used_cache


def fetch_lines_by_modes(modes: set[str]) -> list[dict]:
    """Fetch line metadata for each mode and de-dupe by line id."""
    paths = get_paths()
    out_path = paths.data_raw / LINES_OUTFILE

    def _build() -> list[dict]:
        lines_by_id: dict[str, dict] = {}
        for mode in sorted(modes):
            url = LINE_MODE_URL_TEMPLATE.format(mode=mode)
            data = get_json(url)
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected Line/Mode response for {mode}: {type(data)}")
            for ln in data:
                lid = ln.get("id")
                if lid:
                    lines_by_id[lid] = ln
        return [lines_by_id[k] for k in sorted(lines_by_id.keys())]

    lines, _ = cached(
        out_path,
        force=False,
        read=read_json,
        build=_build,
        write=lambda obj, p: write_json(obj, p),
        validate=lambda obj: isinstance(obj, list),
    )
    return lines


def fetch_route_sequences(
    line_ids: list[str], *, force: bool = False
) -> tuple[list[Path], int, int]:
    """Fetch route sequences for each line id and both directions, saving raw JSON per request."""
    paths = get_paths()
    out_dir = paths.data_raw / ROUTE_SEQUENCE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    skipped_existing = 0
    attempted = 0
    for lid in line_ids:
        for direction in ["inbound", "outbound"]:
            attempted += 1
            url = ROUTE_SEQUENCE_URL_TEMPLATE.format(line_id=lid, direction=direction)
            params = ROUTE_SEQUENCE_PARAMS
            out_path = out_dir / f"tfl_line_{lid}_route_sequence_{direction}.json"

            if not force and out_path.exists() and out_path.stat().st_size > 0:
                skipped_existing += 1
                saved.append(out_path)
                continue
            try:
                data = get_json(url, params=params)
            except Exception as e:
                LOGGER.warning("Failed route sequence for %s %s: %s", lid, direction, e)
                continue
            write_json(data, out_path)
            saved.append(out_path)
    return saved, attempted, skipped_existing


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the transit ingestion script."""
    return create_base_parser(
        "Ingest TfL transit topology (stations + route sequences)."
    ).parse_args()


def run(*, force: bool = False, checkpoint: bool = False) -> dict[str, int | str]:
    """Run the transit ingest. Returns a small dict of key counts."""
    paths = get_paths()

    LOGGER.info("Ingesting TfL stop points for modes: %s", sorted(INCLUDE_MODES))
    stop_points, sp_path, stop_points_cached = fetch_stop_points(force=force)
    sp_df = pd.DataFrame(
        [
            {
                "id": sp.get("id"),
                "commonName": sp.get("commonName"),
                "lat": sp.get("lat"),
                "lon": sp.get("lon"),
                "modes": ",".join(sp.get("modes") or []),
                "stopType": sp.get("stopType"),
            }
            for sp in stop_points
        ]
    )
    LOGGER.info("Stations kept: %d", len(sp_df))

    LOGGER.info("Fetching TfL line ids for included modes...")
    lines = fetch_lines_by_modes(INCLUDE_MODES)
    line_ids = [ln["id"] for ln in lines if "id" in ln]
    LOGGER.info("Lines found: %d", len(line_ids))

    LOGGER.info("Fetching route sequences (inbound/outbound) for each line...")
    seq_files, attempted, skipped_existing = fetch_route_sequences(line_ids, force=force)
    LOGGER.info(
        "Route sequence JSON files available: %d (attempted=%d, skipped_existing=%d)",
        len(seq_files),
        attempted,
        skipped_existing,
    )
    # Update global ingest summary
    records = [
        IngestRecord(
            dataset="tfl_stop_points_stations",
            stage="raw",
            path=str(sp_path.relative_to(paths.root)),
            rows=len(sp_df),
            cols=len(sp_df.columns),
            crs="EPSG:4326",
            bytes=sp_path.stat().st_size if sp_path.exists() else None,
            source="https://api.tfl.gov.uk/StopPoint/Type/NaptanMetroStation/page/1 and NaptanRailStation/page/1",
            notes="Filtered to modes tube/dlr/overground/elizabeth-line; saved JSON list of stop points.",
        ),
        IngestRecord(
            dataset="tfl_route_sequences",
            stage="raw",
            path=str((paths.data_raw / "tfl_route_sequences").relative_to(paths.root)),
            rows=len(seq_files),
            cols=None,
            crs="EPSG:4326",
            bytes=sum(p.stat().st_size for p in seq_files) if seq_files else 0,
            source="https://api.tfl.gov.uk/Line/{id}/Route/Sequence/{direction}?serviceTypes=Regular",
            notes="One JSON per line-direction request (some may fail; see logs).",
        ),
    ]
    upsert_ingest_summary(records, paths.processed_meta / "ingest_summary.csv")
    LOGGER.info("Wrote/updated %s", paths.processed_meta / "ingest_summary.csv")

    out: dict[str, int | str] = {
        "stations_kept": len(sp_df),
        "line_ids": len(line_ids),
        "route_sequence_files": len(seq_files),
        "stop_points_cached": int(stop_points_cached),
        "route_sequences_skipped_existing": skipped_existing,
    }

    return out


def main() -> None:
    configure_logging()
    args = _parse_args()
    run(force=args.force, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
