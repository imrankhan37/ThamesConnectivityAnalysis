"""Ingest all core datasets (transit + Thames + LSOA/IMD) in one run.

Run:
  uv run python scripts/phases/ingest_data.py
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import src...` works when executing this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.cli_utils import IngestStats, add_skip_flags, create_base_parser
from src.core.config import configure_logging

LOGGER = logging.getLogger("ingest_all")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the main ingestion script."""
    parser = create_base_parser("Ingest all core datasets for the Thames connectivity project.")
    add_skip_flags(parser)
    return parser.parse_args()


def _run_ingestion_step(module_name: str, step_name: str, stats: IngestStats) -> bool:
    try:
        LOGGER.info("Running %s ingest...", step_name)
        mod = importlib.import_module(module_name)
        step_stats = mod.run()
        stats.update(step_stats)
        stats.add_step(step_name)
        LOGGER.info("Completed %s ingest", step_name)
        return True
    except Exception as e:
        LOGGER.error("Failed to run %s ingest: %s", step_name, e)
        raise


def main() -> None:
    configure_logging()
    args = _parse_args()
    stats = IngestStats()

    # Execute each ingestion step based on CLI flags
    ingestion_steps = [
        ("scripts.data_ingestion.01_ingest_transit", "transit", not args.skip_transit),
        ("scripts.data_ingestion.02_ingest_thames", "thames", not args.skip_thames),
        ("scripts.data_ingestion.03_ingest_lsoa_imd", "lsoa_imd", not args.skip_lsoa_imd),
    ]

    for module_name, step_name, should_run in ingestion_steps:
        if should_run:
            _run_ingestion_step(module_name, step_name, stats)
        else:
            LOGGER.info("Skipping %s ingest", step_name)

    summary = stats.get_summary()
    LOGGER.info(
        "Ingestion complete. Steps: %d, Total stats: %s", summary.get("step_count", 0), len(summary)
    )


if __name__ == "__main__":
    main()
