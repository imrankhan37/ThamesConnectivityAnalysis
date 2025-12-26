from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile


def ensure_unzipped(path: Path) -> Path:
    """Ensure `path` exists, extracting `path + ".zip"` into place if needed.

    - if `path` exists, return it
    - else require `path + ".zip"`
    - zip must contain exactly one file whose basename matches `path.name`
    - extract that member to `path.parent` and move it to `path`
    """
    path = Path(path)
    if path.exists():
        return path

    zip_path = path.with_suffix(path.suffix + ".zip")
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing required file: {path} (or zipped: {zip_path})")

    path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path) as zf:
        files = [n for n in zf.namelist() if n and not n.endswith("/")]
        if not files:
            raise ValueError(f"Zip contains no files: {zip_path}")

        matches = [n for n in files if Path(n).name == path.name]
        if len(matches) != 1:
            raise ValueError(f"Zip {zip_path} must contain exactly one file named {path.name!r}.")
        member = matches[0]

        # Extract then move into the expected target path.
        extracted = Path(zf.extract(member, path.parent))
        extracted.replace(path)

    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"Extraction failed: {zip_path} -> {path}")
    return path
