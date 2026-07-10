#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pystamps.io.mat import read_mat, write_mat


def crop_bounds(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    margin: int = 0,
) -> tuple[int, int, int, int]:
    nrow, ncol = shape
    row_min, row_max, col_min, col_max = bbox
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if row_min > row_max or col_min > col_max:
        raise ValueError("bbox must be inclusive row_min,row_max,col_min,col_max")
    row_start = max(0, row_min - margin)
    row_stop = min(nrow, row_max + margin + 1)
    col_start = max(0, col_min - margin)
    col_stop = min(ncol, col_max + margin + 1)
    if row_start >= row_stop or col_start >= col_stop:
        raise ValueError("bbox does not overlap fixture")
    return row_start, row_stop, col_start, col_stop


def crop_arrays(
    nzix: np.ndarray,
    ifgw: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    snaphu: np.ndarray,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> dict[str, np.ndarray]:
    nzix = np.asarray(nzix, dtype=bool)
    ifgw = np.asarray(ifgw, dtype=np.complex64)
    snaphu = np.asarray(snaphu, dtype=np.float32)
    if nzix.shape != ifgw.shape or nzix.shape != snaphu.shape:
        raise ValueError("nzix, ifgw, and snaphu must have matching 2-D shapes")
    nrow, ncol = nzix.shape
    if not (0 <= row_start < row_stop <= nrow and 0 <= col_start < col_stop <= ncol):
        raise ValueError("crop bounds are outside fixture shape")
    shaped_rowcost, shaped_colcost = _shape_costs(rowcost, colcost, (nrow, ncol))
    return {
        "nzix": nzix[row_start:row_stop, col_start:col_stop],
        "ifgw": ifgw[row_start:row_stop, col_start:col_stop],
        "rowcost": shaped_rowcost[row_start : row_stop - 1, col_start:col_stop, :],
        "colcost": shaped_colcost[row_start:row_stop, col_start : col_stop - 1, :],
        "snaphu": snaphu[row_start:row_stop, col_start:col_stop],
    }


def crop_fixture(
    source: str | Path,
    dest: str | Path,
    *,
    bbox: tuple[int, int, int, int],
    margin: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(source)
    dest = Path(dest)
    nzix, ifgw, rowcost, colcost, snaphu = _load_fixture(source)
    bounds = crop_bounds(nzix.shape, bbox, margin=margin)
    crop = crop_arrays(nzix, ifgw, rowcost, colcost, snaphu, *bounds)
    _write_fixture(dest, crop, overwrite=overwrite)
    row_start, row_stop, col_start, col_stop = bounds
    return {
        "source": str(source),
        "dest": str(dest),
        "bounds": [row_start, row_stop, col_start, col_stop],
        "shape": [int(row_stop - row_start), int(col_stop - col_start)],
    }


def _shape_costs(
    rowcost: np.ndarray,
    colcost: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    nrow, ncol = shape
    rows = np.asarray(rowcost, dtype=np.int16)
    cols = np.asarray(colcost, dtype=np.int16)
    if rows.shape == (nrow - 1, ncol * 4):
        rows = rows.reshape((nrow - 1, ncol, 4))
    if cols.shape == (nrow, (ncol - 1) * 4):
        cols = cols.reshape((nrow, ncol - 1, 4))
    if rows.shape != (nrow - 1, ncol, 4):
        raise ValueError("rowcost must have shape (nrow - 1, ncol, 4) or (nrow - 1, ncol * 4)")
    if cols.shape != (nrow, ncol - 1, 4):
        raise ValueError("colcost must have shape (nrow, ncol - 1, 4) or (nrow, (ncol - 1) * 4)")
    return rows, cols


def _load_fixture(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nzix = np.asarray(read_mat(root / "uw_grid.mat")["nzix"], dtype=bool)
    nrow, ncol = nzix.shape
    row_elems = (nrow - 1) * ncol * 4
    expected = row_elems + nrow * (ncol - 1) * 4
    cost_raw = np.fromfile(root / "snaphu.costinfile", dtype=np.int16)
    if cost_raw.size != expected:
        raise RuntimeError(f"snaphu.costinfile has {cost_raw.size} int16 values, expected {expected}")
    rowcost = cost_raw[:row_elems].reshape((nrow - 1, ncol, 4))
    colcost = cost_raw[row_elems:].reshape((nrow, ncol - 1, 4))
    ifgw = np.fromfile(root / "snaphu.in", dtype=np.complex64).reshape((nrow, ncol))
    snaphu = np.fromfile(root / "snaphu.out", dtype=np.float32).reshape((nrow, ncol))
    return nzix, ifgw, rowcost, colcost, snaphu


def _write_fixture(dest: Path, crop: dict[str, np.ndarray], *, overwrite: bool) -> None:
    files = ["uw_grid.mat", "snaphu.in", "snaphu.costinfile", "snaphu.out"]
    if dest.exists() and not overwrite and any((dest / name).exists() for name in files):
        raise FileExistsError(f"{dest} already contains Stage 6 fixture files; pass --overwrite")
    dest.mkdir(parents=True, exist_ok=True)
    write_mat(dest / "uw_grid.mat", {"nzix": np.asarray(crop["nzix"], dtype=bool)})
    np.asarray(crop["ifgw"], dtype=np.complex64).tofile(dest / "snaphu.in")
    rowcost = np.asarray(crop["rowcost"], dtype=np.int16).reshape(-1)
    colcost = np.asarray(crop["colcost"], dtype=np.int16).reshape(-1)
    np.concatenate([rowcost, colcost]).tofile(dest / "snaphu.costinfile")
    np.asarray(crop["snaphu"], dtype=np.float32).tofile(dest / "snaphu.out")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop a minimal Stage 6 SNAPHU/native fixture.")
    parser.add_argument("--root", required=True, type=Path, help="Source Stage 6 fixture directory.")
    parser.add_argument("--dest", required=True, type=Path, help="Destination crop directory.")
    parser.add_argument(
        "--bbox",
        required=True,
        nargs=4,
        type=int,
        metavar=("ROW_MIN", "ROW_MAX", "COL_MIN", "COL_MAX"),
        help="Inclusive bbox, matching Stage 6 diagnostic component bbox fields.",
    )
    parser.add_argument("--margin", default=0, type=int, help="Cells to add around the inclusive bbox.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing crop files.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = crop_fixture(
        args.root,
        args.dest,
        bbox=tuple(args.bbox),
        margin=args.margin,
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
