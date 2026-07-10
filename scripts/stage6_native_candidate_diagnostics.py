#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from scripts.stage6_cut_window_diagnostics import _edge_potential, _label_flows, _shape_costs
from scripts.stage6_hf_diagnostics import TWO_PI, _load_fixture, load_native_unwrap


def potential_component_rectangles(
    ifgw: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    native: np.ndarray,
    *,
    threshold: int = 13,
    max_cells: int = 16_384,
    limit: int = 1_024,
    nshortcycle: int = 200,
) -> list[dict[str, Any]]:
    phase = np.angle(np.asarray(ifgw, dtype=np.complex64)).astype(np.float32)
    labels = np.rint((np.asarray(native, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    rows, cols = _shape_costs(rowcost, colcost, phase.shape)
    h_flow, v_flow = _label_flows(phase, labels)
    h_potential = _edge_potential(cols, h_flow, nshortcycle)
    v_potential = _edge_potential(rows, v_flow, nshortcycle)
    mask = _potential_cell_mask(h_potential, v_potential, phase.shape, threshold)
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    labels_cc, component_count = ndimage.label(mask, structure=structure)
    rectangles = []
    for component, bbox in enumerate(ndimage.find_objects(labels_cc), start=1):
        if bbox is None:
            continue
        row_slice, col_slice = bbox
        rect = [row_slice.start, row_slice.stop - 1, col_slice.start, col_slice.stop - 1]
        cells = (rect[1] - rect[0] + 1) * (rect[3] - rect[2] + 1)
        if cells > max_cells:
            continue
        score = _rectangle_score(h_potential, v_potential, rect, phase.shape)
        if score <= 0:
            continue
        rectangles.append(
            {
                "bbox": [int(value) for value in rect],
                "cells": int(cells),
                "component": int(component),
                "component_pixels": int(np.sum(labels_cc[bbox] == component)),
                "score": int(score),
            }
        )
    rectangles.sort(key=lambda item: (-item["score"], item["bbox"][0], item["bbox"][2]))
    return rectangles[:limit]


def candidate_coverage(
    rectangles: list[dict[str, Any]],
    targets: list[list[int]],
) -> list[dict[str, Any]]:
    out = []
    for target in targets:
        rank = None
        best_overlap = 0
        target_cells = (target[1] - target[0] + 1) * (target[3] - target[2] + 1)
        for idx, rectangle in enumerate(rectangles, start=1):
            bbox = rectangle["bbox"]
            overlap = _bbox_overlap_cells(bbox, target)
            best_overlap = max(best_overlap, overlap)
            if rank is None and _covers(bbox, target):
                rank = idx
        out.append(
            {
                "target": [int(value) for value in target],
                "full_cover_rank": rank,
                "best_overlap_cells": int(best_overlap),
                "best_overlap_fraction": float(best_overlap / target_cells) if target_cells else 0.0,
            }
        )
    return out


def analyze_fixture_candidates(
    root: Path,
    native_file: Path,
    *,
    thresholds: list[int],
    max_cells: int = 16_384,
    limit: int = 1_024,
    targets: list[list[int]] | None = None,
) -> dict[str, Any]:
    _nzix, ifgw, rowcost, colcost, _snaphu = _load_fixture(root)
    native = load_native_unwrap(native_file, ifgw.shape)
    cases = []
    for threshold in thresholds:
        rectangles = potential_component_rectangles(
            ifgw,
            rowcost,
            colcost,
            native,
            threshold=threshold,
            max_cells=max_cells,
            limit=limit,
        )
        record = {
            "threshold": int(threshold),
            "rectangle_count": len(rectangles),
            "top_rectangles": rectangles[:12],
        }
        if targets:
            record["target_coverage"] = candidate_coverage(rectangles, targets)
        cases.append(record)
    return {
        "fixture_root": str(root),
        "native_file": str(native_file),
        "shape": [int(ifgw.shape[0]), int(ifgw.shape[1])],
        "max_cells": int(max_cells),
        "limit": int(limit),
        "cases": cases,
    }


def _potential_cell_mask(
    h_potential: np.ndarray,
    v_potential: np.ndarray,
    shape: tuple[int, int],
    threshold: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    horizontal = h_potential >= threshold
    vertical = v_potential >= threshold
    mask[:, :-1] |= horizontal
    mask[:, 1:] |= horizontal
    mask[:-1, :] |= vertical
    mask[1:, :] |= vertical
    return mask


def _rectangle_score(
    h_potential: np.ndarray,
    v_potential: np.ndarray,
    bbox: list[int],
    shape: tuple[int, int],
) -> int:
    nrow, ncol = shape
    row0, row1, col0, col1 = bbox
    row_stop = row1 + 1
    col_stop = col1 + 1
    h_score = h_potential[row0:row_stop, max(col0 - 1, 0) : min(col_stop, ncol - 1)].sum()
    v_score = v_potential[max(row0 - 1, 0) : min(row_stop, nrow - 1), col0:col_stop].sum()
    return int(h_score + v_score)


def _bbox_overlap_cells(a: list[int], b: list[int]) -> int:
    rows = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    cols = max(0, min(a[3], b[3]) - max(a[2], b[2]) + 1)
    return rows * cols


def _covers(rectangle: list[int], target: list[int]) -> bool:
    return (
        rectangle[0] <= target[0]
        and rectangle[1] >= target[1]
        and rectangle[2] <= target[2]
        and rectangle[3] >= target[3]
    )


def _parse_targets(raw: str | None) -> list[list[int]]:
    if not raw:
        return []
    targets = json.loads(raw)
    if not isinstance(targets, list):
        raise ValueError("--targets-json must decode to a list")
    return [[int(value) for value in target] for target in targets]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize native-only Stage 6 cut-potential rectangles.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--native-file", required=True, type=Path)
    parser.add_argument("--thresholds", default="13,26,52")
    parser.add_argument("--max-cells", default=16_384, type=int)
    parser.add_argument("--limit", default=1_024, type=int)
    parser.add_argument("--targets-json", help="Optional JSON list of target bboxes for coverage reporting.")
    args = parser.parse_args()

    thresholds = [int(value) for value in args.thresholds.split(",") if value]
    payload = analyze_fixture_candidates(
        args.root,
        args.native_file,
        thresholds=thresholds,
        max_cells=args.max_cells,
        limit=args.limit,
        targets=_parse_targets(args.targets_json),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
