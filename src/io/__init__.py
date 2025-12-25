"""Lightweight I/O helpers.

This module centralises:
- validated CSV reads (`read_csv_validated`) at pipeline boundaries
- simple JSON/text helpers used by scripts
- ingestion metadata helpers used in `scripts/data_ingestion/*`
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.models.validate import validate_df


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_parent_dir(path)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Any, path: Path) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def cached(
    path: Path,
    *,
    force: bool,
    read,
    build,
    write=None,
    validate=None,
) -> tuple[Any, bool]:
    """Cache-to-disk helper used by ingestion scripts.

    Returns (obj, used_cache).
    """
    if not force and path.exists() and path.stat().st_size > 0:
        obj = read(path)
        if validate is None or validate(obj):
            return obj, True
    obj = build()
    if write is not None:
        write(obj, path)
    return obj, False


def get_json(url: str, *, params: dict[str, str] | None = None, timeout: float = 60.0) -> Any:
    """HTTP GET JSON helper for ingestion scripts.

    Kept intentionally small (scripts already implement caching at the file level).
    """
    import requests

    global _SESSION  # noqa: PLW0603
    try:
        session = _SESSION
    except NameError:
        session = None

    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "ThamesConnectivityAnalysis/0.1 (uv; ingestion)"})
        _SESSION = session

    from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

    def _retryable(exc: BaseException) -> bool:
        if isinstance(exc, requests.exceptions.HTTPError):
            resp = getattr(exc, "response", None)
            code = getattr(resp, "status_code", None)
            return code in {429, 500, 502, 503, 504}
        return isinstance(exc, requests.exceptions.RequestException)

    @retry(
        retry=retry_if_exception(_retryable),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=10.0),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _do_get() -> Any:
        r = session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()

    return _do_get()


def read_csv_validated(
    path: Path,
    *,
    dtype: dict[str, str],
    schema: Any,
) -> pd.DataFrame:
    """Read a CSV and validate it against a schema-like object.

    We validate by *shape* (required attributes) rather than strict class identity,
    so notebook kernels don't break across refactors.
    """
    required_attrs = ("name", "required_columns", "optional_columns", "dtypes", "non_null")
    missing = [a for a in required_attrs if not hasattr(schema, a)]
    if missing:
        raise TypeError(f"schema missing required attributes {missing}; got {type(schema)}")

    df = pd.read_csv(path, dtype=dtype)
    return validate_df(df, schema)


def sha256_file(path: Path) -> str:
    """Return SHA256 hex digest for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, out_path: Path, *, timeout_s: float = 60.0) -> None:
    """Download a URL to disk (used by ingestion scripts)."""
    import requests

    ensure_parent_dir(out_path)
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    out_path.write_bytes(r.content)


@dataclass(frozen=True)
class IngestRecord:
    """Row-level ingest metadata for the ingest inventory CSV (default: `data/processed/_meta/ingest_summary.csv`)."""

    dataset: str
    stage: str  # e.g. "raw" / "processed"
    path: str
    rows: int | None
    cols: int | None
    crs: str | None
    bytes: int | None
    source: str
    notes: str = ""


def load_ingest_config(path: Path) -> dict[str, Any]:
    """Load a YAML ingest config file."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def upsert_ingest_summary(records: list[IngestRecord], summary_csv: Path) -> None:
    """Upsert ingest records into a summary CSV keyed by `dataset`."""
    ensure_parent_dir(summary_csv)
    rows = [
        {
            "dataset": r.dataset,
            "stage": r.stage,
            "path": r.path,
            "rows": r.rows,
            "cols": r.cols,
            "crs": r.crs,
            "bytes": r.bytes,
            "source": r.source,
            "notes": r.notes,
        }
        for r in records
    ]

    new_df = pd.DataFrame(rows)
    new_df["dataset"] = new_df["dataset"].astype(str)

    if summary_csv.exists():
        old = pd.read_csv(summary_csv, dtype={"dataset": "string"})
        old["dataset"] = old["dataset"].astype(str)
        old = old[~old["dataset"].isin(set(new_df["dataset"].tolist()))]
        df = pd.concat([old, new_df], ignore_index=True)
    else:
        df = new_df

    df = df.sort_values(["dataset"], kind="mergesort").reset_index(drop=True)
    df.to_csv(summary_csv, index=False)


def infer_lsoa_code_column(df: pd.DataFrame) -> str:
    """Infer the LSOA code column name from a dataframe (IMD releases vary slightly)."""
    candidates = [c for c in df.columns if "lsoa" in c.lower() and "code" in c.lower()]
    if candidates:
        return candidates[0]
    # common fallbacks
    for c in ("LSOA21CD", "lsoa21cd", "LSOA code (2011)", "LSOA Code"):
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer LSOA code column from columns: {list(df.columns)[:20]}")
