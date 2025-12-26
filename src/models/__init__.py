"""Pydantic models and dataframe schema validators.

These are contracts to keep the pipeline deterministic:
- Each script validates its inputs/outputs at boundaries.
- Analysis logic remains in `src/` pure functions; scripts orchestrate I/O.
"""

from __future__ import annotations

from src.models.schemas import (
    EDGE_CROSSING,
    EDGES,
    STATION_BANK,
    STATIONS,
    TableSchema,
)
from src.models.validate import validate_df

__all__ = [
    "TableSchema",
    "validate_df",
    "STATIONS",
    "EDGES",
    "STATION_BANK",
    "EDGE_CROSSING",
]
