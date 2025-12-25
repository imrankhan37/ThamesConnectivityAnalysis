"""Schema definitions for pipeline dataframe contracts.

This module contains only:
- `TableSchema` (schema metadata container)
- concrete table schemas (e.g., `STATIONS`, `EDGES`, ...)
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field


class TableSchema(BaseModel):
    """A simple schema for a pandas DataFrame (column-level contract)."""

    name: str
    required_columns: tuple[str, ...] = Field(default_factory=tuple)
    optional_columns: tuple[str, ...] = Field(default_factory=tuple)
    # pandas dtype strings, e.g. "string", "Float64", "boolean"
    dtypes: Mapping[str, str] = Field(default_factory=dict)
    non_null: tuple[str, ...] = Field(default_factory=tuple)

    def allowed_columns(self) -> set[str]:
        return set(self.required_columns) | set(self.optional_columns)


STATIONS = TableSchema(
    name="stations",
    required_columns=("station_id", "name", "lat", "lon", "modes"),
    optional_columns=("stop_type", "zone", "naptan_ids"),
    dtypes={
        "station_id": "string",
        "name": "string",
        "lat": "Float64",
        "lon": "Float64",
        "modes": "string",
        "stop_type": "string",
        "zone": "string",
        "naptan_ids": "string",
    },
    non_null=("station_id",),
)

EDGES = TableSchema(
    name="edges",
    required_columns=("u", "v"),
    optional_columns=("line_ids", "modes", "directions", "route_names", "distance_m"),
    dtypes={
        "u": "string",
        "v": "string",
        "line_ids": "string",
        "modes": "string",
        "directions": "string",
        "route_names": "string",
        "distance_m": "Float64",
    },
    non_null=("u", "v"),
)

STATION_LSOA = TableSchema(
    name="station_lsoa",
    required_columns=("station_id", "lsoa_code"),
    optional_columns=("method", "distance_m"),
    dtypes={
        "station_id": "string",
        "lsoa_code": "string",
        "method": "string",
        "distance_m": "Float64",
    },
    non_null=("station_id",),
)

STATION_BANK = TableSchema(
    name="station_bank",
    required_columns=("station_id", "bank"),
    optional_columns=("nearest_river_northing_m",),
    dtypes={
        "station_id": "string",
        "bank": "string",
        "nearest_river_northing_m": "Float64",
    },
    non_null=("station_id", "bank"),
)

EDGE_CROSSING = TableSchema(
    name="edge_is_thames_crossing",
    required_columns=("u", "v", "is_thames_crossing"),
    optional_columns=(
        "bank_u",
        "bank_v",
        "opposite_banks",
        "intersects_river_buffer",
        "buffer_m",
    ),
    dtypes={
        "u": "string",
        "v": "string",
        "bank_u": "string",
        "bank_v": "string",
        "opposite_banks": "boolean",
        "intersects_river_buffer": "boolean",
        "buffer_m": "Float64",
        "is_thames_crossing": "boolean",
    },
    non_null=("u", "v"),
)
