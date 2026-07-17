from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .mt_prep_types import MtPrepSummary

_NATIVE_MODULE: Any | None = None
_NATIVE_IMPORT_ATTEMPTED = False


def native_export() -> Any | None:
    global _NATIVE_IMPORT_ATTEMPTED, _NATIVE_MODULE
    if _NATIVE_MODULE is not None:
        fn = getattr(_NATIVE_MODULE, "mt_prep_prepare_snap_inputs", None)
        return fn if callable(fn) else None
    if _NATIVE_IMPORT_ATTEMPTED:
        importlib.invalidate_caches()
    _NATIVE_IMPORT_ATTEMPTED = True
    try:
        _NATIVE_MODULE = importlib.import_module("pystamps.kernels._stage2_native")
    except Exception:
        _NATIVE_MODULE = None
        return None
    fn = getattr(_NATIVE_MODULE, "mt_prep_prepare_snap_inputs", None)
    return fn if callable(fn) else None


def summary_from_payload(root: Path, payload: dict[str, Any]) -> MtPrepSummary:
    patch_rows = [
        {
            "patch": str(row["patch"]),
            "candidates": int(row["candidates"]),
            "bounds": tuple(int(v) for v in row["bounds"]),
            "noover": tuple(int(v) for v in row["noover"]),
        }
        for row in payload["patch_rows"]
    ]
    return MtPrepSummary(
        root,
        int(payload["patch_count"]),
        int(payload["candidate_count"]),
        patch_rows,
    )
