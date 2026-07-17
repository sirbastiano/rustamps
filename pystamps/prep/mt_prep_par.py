from __future__ import annotations

from pathlib import Path


def par_int(path: Path, *keys: str) -> int | None:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        values[key.strip()] = value.strip().split()[0] if value.strip() else ""
    for key in keys:
        raw = values.get(key)
        if raw:
            return int(round(float(raw)))
    return None
