"""Common CLI utilities for ingestion scripts."""

from __future__ import annotations

import argparse
from typing import Any


def create_base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download/rebuild outputs even if cached files exist.",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Print checkpoint summary including output hashes.",
    )
    return parser


def add_skip_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skip-transit", action="store_true", help="Skip TfL transit ingest.")
    parser.add_argument("--skip-thames", action="store_true", help="Skip Thames geometry ingest.")
    parser.add_argument(
        "--skip-boundary",
        action="store_true",
        help="Skip London boundary ingest (ONS regions polygon used for London-only filtering/plots).",
    )
    parser.add_argument(
        "--write-qa",
        action="store_true",
        help="Write a minimal combined QA/provenance markdown note (recommended).",
    )


class IngestStats:
    """Simple container for collecting statistics across ingestion steps."""

    def __init__(self) -> None:
        self.stats: dict[str, Any] = {}
        self.completed_steps: list[str] = []

    def update(self, step_stats: dict[str, Any]) -> None:
        self.stats.update(step_stats)

    def add_step(self, step_name: str) -> None:
        self.completed_steps.append(step_name)

    def get_summary(self) -> dict[str, Any]:
        return {
            "completed_steps": self.completed_steps,
            "step_count": len(self.completed_steps),
            **self.stats,
        }
