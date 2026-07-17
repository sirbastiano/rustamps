from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MtPrepSummary:
    dataset_root: Path
    patch_count: int
    candidate_count: int
    patch_rows: list[dict[str, Any]]
