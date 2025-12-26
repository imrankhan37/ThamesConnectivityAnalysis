"""Project configuration (paths, constants, deterministic seed)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

# Reproducibility
SEED: int = 20250101

# CRS defaults
CRS_WGS84: str = "EPSG:4326"
CRS_BNG: str = "EPSG:27700"  # British National Grid

# Study area (WGS84) - used for initial London subsetting during ingestion.
# (min_lon, min_lat, max_lon, max_lat)
LONDON_BBOX_WGS84: tuple[float, float, float, float] = (-0.55, 51.25, 0.35, 51.75)


def project_root() -> Path:
    """Return repository root assuming this file lives in `<root>/src/core/config.py`."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path
    data_raw: Path
    data_processed: Path
    data_analysis: Path  # data produced by a notebook or script that never gets used downstream

    # Processed data is data produced by a notebook or script that gets used downstream
    processed_transit: Path
    processed_boundaries: Path
    processed_spatial: Path
    processed_metrics: Path
    processed_meta: Path

    artifacts: Path
    figures: Path
    docs: Path
    docs_qa: Path
    src: Path
    scripts: Path
    tests: Path
    notebooks: Path


def get_paths(root: Path | None = None) -> Paths:
    r = project_root() if root is None else Path(root).resolve()
    data_processed = r / "data" / "processed"
    data_analysis = r / "data" / "analysis"
    return Paths(
        root=r,
        data_raw=r / "data" / "raw",
        data_processed=data_processed,
        data_analysis=data_analysis,
        processed_transit=data_processed / "transit",
        processed_boundaries=data_processed / "boundaries",
        processed_spatial=data_processed / "spatial",
        processed_metrics=data_processed / "metrics",
        processed_meta=data_processed / "_meta",
        artifacts=r / "artifacts",
        figures=r / "figures",
        docs=r / "docs",
        docs_qa=r / "docs" / "qa",
        src=r / "src",
        scripts=r / "scripts",
        tests=r / "tests",
        notebooks=r / "notebooks",
    )


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
