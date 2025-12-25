"""Validation utilities for pipeline dataframe contracts."""

from __future__ import annotations

import pandas as pd

from src.models.schemas import TableSchema


def validate_df(
    df: pd.DataFrame,
    schema: TableSchema,
    *,
    coerce_dtypes: bool = True,
    allow_extra_columns: bool = True,
) -> pd.DataFrame:
    """Validate a dataframe against a schema. Returns a (possibly coerced) copy."""
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{schema.name}: missing required columns: {missing}")

    if not allow_extra_columns:
        extra = [c for c in df.columns if c not in schema.allowed_columns()]
        if extra:
            raise ValueError(f"{schema.name}: unexpected columns: {extra}")

    out = df.copy()

    if coerce_dtypes and schema.dtypes:
        for col, dtype in schema.dtypes.items():
            if col not in out.columns:
                continue
            try:
                # pandas nullable dtypes work well here ("string", "Float64", "boolean", "Int64")
                out[col] = out[col].astype(dtype)
            except Exception as exc:  # noqa: BLE001 - surface as actionable schema error
                raise TypeError(
                    f"{schema.name}: failed to coerce column '{col}' to dtype '{dtype}': {exc}"
                ) from exc

    if schema.non_null:
        bad = [c for c in schema.non_null if c in out.columns and out[c].isna().any()]
        if bad:
            counts = {c: int(out[c].isna().sum()) for c in bad}
            raise ValueError(f"{schema.name}: non-null columns contain NA values: {counts}")

    return out
