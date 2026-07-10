#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from scripts.stage6_hf_diagnostics import (
    TWO_PI,
    _component_shift_gain,
    _desired_flows,
    _edge_cost,
    _load_fixture,
    load_native_unwrap,
)


def cut_window_candidate_summary(
    ifgw: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    native: np.ndarray,
    snaphu: np.ndarray,
    *,
    max_cells: int = 16_384,
    limit: int = 1_024,
    component_limit: int = 8,
    nshortcycle: int = 200,
) -> dict[str, Any]:
    phase = np.angle(np.asarray(ifgw, dtype=np.complex64)).astype(np.float32)
    native_labels = np.rint((np.asarray(native, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    snaphu_labels = np.rint((np.asarray(snaphu, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    diff = (native_labels - snaphu_labels).astype(np.int32)
    rows, cols = _shape_costs(rowcost, colcost, phase.shape)
    h_flow, v_flow = _label_flows(phase, native_labels)
    h_potential = _edge_potential(cols, h_flow, nshortcycle)
    v_potential = _edge_potential(rows, v_flow, nshortcycle)
    side = int(np.sqrt(max(int(max_cells), 0)))
    windows = _candidate_windows(h_potential, v_potential, phase.shape, side, limit)
    return {
        "shape": [int(phase.shape[0]), int(phase.shape[1])],
        "cut_max_cells": int(max_cells),
        "cut_side": side,
        "candidate_count": len(windows),
        "top_windows": [_window_payload(window) for window in windows[: min(12, len(windows))]],
        "components": _component_coverage(
            diff,
            windows,
            ifgw,
            rows,
            cols,
            native,
            component_limit=component_limit,
            nshortcycle=nshortcycle,
        ),
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
    if rows.shape != (nrow - 1, ncol, 4) or cols.shape != (nrow, ncol - 1, 4):
        raise ValueError("rowcost/colcost shapes are incompatible with ifgw")
    return rows, cols


def _label_flows(phase: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h_desired, v_desired = _desired_flows(phase)
    h_flow = labels[:, 1:] - labels[:, :-1] - h_desired
    v_flow = -(labels[1:, :] - labels[:-1, :] - v_desired)
    return h_flow, v_flow


def _edge_potential(cost: np.ndarray, flow: np.ndarray, nshortcycle: int) -> np.ndarray:
    old = _edge_cost(cost, flow, nshortcycle)
    down = _edge_cost(cost, flow - 1, nshortcycle)
    up = _edge_cost(cost, flow + 1, nshortcycle)
    return np.maximum(old - np.minimum(down, up), 0).astype(np.int64)


def _window_starts(limit: int, size: int) -> list[int]:
    if limit <= size:
        return [0]
    max_start = limit - size
    step = max(size // 2, 1)
    starts = list(range(0, max_start, step))
    if not starts or starts[-1] != max_start:
        starts.append(max_start)
    return starts


def _candidate_windows(
    h_potential: np.ndarray,
    v_potential: np.ndarray,
    shape: tuple[int, int],
    side: int,
    limit: int,
) -> list[dict[str, int]]:
    nrow, ncol = shape
    height = min(side, nrow)
    width = min(side, ncol)
    if height == 0 or width == 0 or limit <= 0:
        return []
    windows = []
    for row0 in _window_starts(nrow, height):
        for col0 in _window_starts(ncol, width):
            row1 = row0 + height
            col1 = col0 + width
            score = int(
                h_potential[row0:row1, max(col0 - 1, 0) : min(col1, ncol - 1)].sum()
                + v_potential[max(row0 - 1, 0) : min(row1, nrow - 1), col0:col1].sum()
            )
            if score > 0:
                windows.append({"row": row0, "col": col0, "height": height, "width": width, "score": score})
    windows.sort(key=lambda item: (-item["score"], item["row"], item["col"]))
    return windows[:limit]


def _component_coverage(
    diff: np.ndarray,
    windows: list[dict[str, int]],
    ifgw: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    native: np.ndarray,
    *,
    component_limit: int,
    nshortcycle: int,
) -> dict[str, list[dict[str, Any]]]:
    phase = np.angle(np.asarray(ifgw, dtype=np.complex64)).astype(np.float32)
    labels = np.rint((np.asarray(native, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    h_flow, v_flow = _label_flows(phase, labels)
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    out: dict[str, list[dict[str, Any]]] = {}
    for value in sorted(int(v) for v in np.unique(diff) if int(v) != 0):
        labels_cc, n_component = ndimage.label(diff == value, structure=structure)
        sizes = np.bincount(labels_cc.reshape(-1))
        if sizes.size:
            sizes[0] = 0
        components = []
        for component in np.argsort(sizes)[-component_limit:][::-1]:
            size = int(sizes[component])
            if component == 0 or size == 0:
                continue
            mask = labels_cc == component
            rows, cols = np.where(mask)
            overlap = _window_overlap_stats(mask, windows)
            components.append(
                {
                    "size": size,
                    "shift": int(-value),
                    "gain": int(
                        _component_shift_gain(mask, -value, h_flow, v_flow, rowcost, colcost, nshortcycle=nshortcycle)
                    ),
                    "bbox": [int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())],
                    "bbox_height": int(rows.max() - rows.min() + 1),
                    "bbox_width": int(cols.max() - cols.min() + 1),
                    "full_cover_candidate_count": overlap["full_cover_count"],
                    "max_overlap_pixels": overlap["best_overlap"],
                    "max_overlap_fraction": overlap["best_overlap"] / size,
                    "best_overlap_window": _window_payload(overlap["best_window"])
                    if overlap["best_window"] is not None
                    else None,
                }
            )
        out[str(value)] = components
    return out


def _window_overlap_stats(mask: np.ndarray, windows: list[dict[str, int]]) -> dict[str, Any]:
    integral = np.pad(mask.astype(np.int64).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    size = int(mask.sum())
    best_overlap = 0
    best_window = None
    full_cover_count = 0
    for window in windows:
        row0 = window["row"]
        col0 = window["col"]
        row1 = row0 + window["height"]
        col1 = col0 + window["width"]
        overlap = int(integral[row1, col1] - integral[row0, col1] - integral[row1, col0] + integral[row0, col0])
        if overlap == size:
            full_cover_count += 1
        if overlap > best_overlap:
            best_overlap = overlap
            best_window = window
    return {"best_overlap": best_overlap, "best_window": best_window, "full_cover_count": full_cover_count}


def _window_payload(window: dict[str, int]) -> dict[str, int]:
    return {key: int(window[key]) for key in ["row", "col", "height", "width", "score"] if key in window}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage 6 binary-cut candidate window coverage.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--native-file", required=True, type=Path)
    parser.add_argument("--max-cells", default=16_384, type=int)
    parser.add_argument("--limit", default=1_024, type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _nzix, ifgw, rowcost, colcost, snaphu = _load_fixture(args.root)
    native = load_native_unwrap(args.native_file, ifgw.shape)
    payload = cut_window_candidate_summary(ifgw, rowcost, colcost, native, snaphu, max_cells=args.max_cells, limit=args.limit)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
