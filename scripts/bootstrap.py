from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path(script_file: str | Path, *, parents: int = 2) -> Path:
    """Ensure repo root is on sys.path (so `import src...` works)."""
    p = Path(script_file).resolve()
    repo_root = p.parents[parents]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root
