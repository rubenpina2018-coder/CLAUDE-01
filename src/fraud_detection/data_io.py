"""I/O helpers shared by the pipeline steps (pandas is imported lazily)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def resolve_csv(path: Path, preferred: str | None = None) -> Path:
    """Accept a CSV file or a folder (Azure ML ``uri_folder``) that contains one."""
    path = Path(path)
    if path.is_file():
        return path
    if preferred and (path / preferred).is_file():
        return path / preferred
    candidates = sorted(path.glob("*.csv")) if path.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"cannot find a unique CSV file in {path} (looked for {preferred!r})")


def read_transactions(path: Path, preferred: str | None = None) -> pd.DataFrame:
    import pandas as pd

    return pd.read_csv(resolve_csv(path, preferred))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
