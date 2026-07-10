#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.stage6_hf_diagnostics import _load_fixture, load_native_unwrap
from scripts.stage6_hf_flow_diagnostics import (
    TWO_PI,
    _desired_flows,
    _edge_cost,
    _label_flows,
    _reshape_costs,
    _value_counts,
)


def changed_flow_component_summary(
    ifgw: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    native: np.ndarray,
    snaphu: np.ndarray,
    *,
    nshortcycle: int = 200,
    limit: int = 12,
) -> dict[str, Any]:
    phase = np.angle(np.asarray(ifgw, dtype=np.complex64)).astype(np.float32)
    native_labels = np.rint((np.asarray(native, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    snaphu_labels = np.rint((np.asarray(snaphu, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    h_desired, v_desired = _desired_flows(phase)
    native_h, native_v = _label_flows(native_labels, h_desired, v_desired)
    snaphu_h, snaphu_v = _label_flows(snaphu_labels, h_desired, v_desired)
    shaped_rowcost, shaped_colcost = _reshape_costs(rowcost, colcost, phase.shape)
    native_h_cost = _edge_cost(shaped_colcost, native_h, nshortcycle)
    snaphu_h_cost = _edge_cost(shaped_colcost, snaphu_h, nshortcycle)
    native_v_cost = _edge_cost(shaped_rowcost, native_v, nshortcycle)
    snaphu_v_cost = _edge_cost(shaped_rowcost, snaphu_v, nshortcycle)
    changed_h = native_h != snaphu_h
    changed_v = native_v != snaphu_v

    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def add(cell: tuple[int, int]) -> None:
        parent.setdefault(cell, cell)

    def find(cell: tuple[int, int]) -> tuple[int, int]:
        root = parent[cell]
        while root != parent[root]:
            root = parent[root]
        while cell != root:
            next_cell = parent[cell]
            parent[cell] = root
            cell = next_cell
        return root

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        add(a)
        add(b)
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for row, col in np.argwhere(changed_h):
        union((int(row), int(col)), (int(row), int(col) + 1))
    for row, col in np.argwhere(changed_v):
        union((int(row), int(col)), (int(row) + 1, int(col)))

    records = _component_cell_records(parent)
    _add_changed_edges(records, parent, changed_h, native_h, snaphu_h, native_h_cost, snaphu_h_cost, "h")
    _add_changed_edges(records, parent, changed_v, native_v, snaphu_v, native_v_cost, snaphu_v_cost, "v")

    components = sorted(
        (_final_component(record) for record in records.values()),
        key=lambda item: (
            -abs(item["native_minus_snaphu_cost"]),
            -item["changed_edges"],
            item["bbox"][0],
            item["bbox"][2],
        ),
    )
    total_native = int(sum(item["native_cost_on_changed"] for item in components))
    total_snaphu = int(sum(item["snaphu_cost_on_changed"] for item in components))
    return {
        "component_count": len(components),
        "total_changed_edges": int(np.sum(changed_h) + np.sum(changed_v)),
        "total_native_cost_on_changed": total_native,
        "total_snaphu_cost_on_changed": total_snaphu,
        "total_native_minus_snaphu_cost": total_native - total_snaphu,
        "components": components[:limit],
        "omitted_components": max(len(components) - int(limit), 0),
    }


def _component_cell_records(
    parent: dict[tuple[int, int], tuple[int, int]],
) -> dict[tuple[int, int], dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for cell in parent:
        root = _find(parent, cell)
        row, col = cell
        record = records.setdefault(
            root,
            {
                "cell_count": 0,
                "rmin": row,
                "rmax": row,
                "cmin": col,
                "cmax": col,
                "changed_h": 0,
                "changed_v": 0,
                "native_cost": 0,
                "snaphu_cost": 0,
                "deltas": [],
            },
        )
        record["cell_count"] += 1
        record["rmin"] = min(record["rmin"], row)
        record["rmax"] = max(record["rmax"], row)
        record["cmin"] = min(record["cmin"], col)
        record["cmax"] = max(record["cmax"], col)
    return records


def _find(
    parent: dict[tuple[int, int], tuple[int, int]],
    cell: tuple[int, int],
) -> tuple[int, int]:
    while cell != parent[cell]:
        cell = parent[cell]
    return cell


def _add_changed_edges(
    records: dict[tuple[int, int], dict[str, Any]],
    parent: dict[tuple[int, int], tuple[int, int]],
    changed: np.ndarray,
    native_flow: np.ndarray,
    snaphu_flow: np.ndarray,
    native_cost: np.ndarray,
    snaphu_cost: np.ndarray,
    axis: str,
) -> None:
    for row, col in np.argwhere(changed):
        root = _find(parent, (int(row), int(col)))
        record = records[root]
        record[f"changed_{axis}"] += 1
        record["native_cost"] += int(native_cost[row, col])
        record["snaphu_cost"] += int(snaphu_cost[row, col])
        record["deltas"].append(int(snaphu_flow[row, col] - native_flow[row, col]))


def _final_component(record: dict[str, Any]) -> dict[str, Any]:
    native_cost = int(record["native_cost"])
    snaphu_cost = int(record["snaphu_cost"])
    return {
        "bbox": [int(record["rmin"]), int(record["rmax"]), int(record["cmin"]), int(record["cmax"])],
        "cell_count": int(record["cell_count"]),
        "changed_edges": int(record["changed_h"] + record["changed_v"]),
        "changed_h": int(record["changed_h"]),
        "changed_v": int(record["changed_v"]),
        "delta_counts": _value_counts(np.asarray(record["deltas"], dtype=np.int64)),
        "native_cost_on_changed": native_cost,
        "snaphu_cost_on_changed": snaphu_cost,
        "native_minus_snaphu_cost": native_cost - snaphu_cost,
    }


def analyze_fixture_components(
    root: Path,
    native_file: Path,
    *,
    nshortcycle: int = 200,
    limit: int = 12,
) -> dict[str, Any]:
    _nzix, ifgw, rowcost, colcost, snaphu = _load_fixture(root)
    native = load_native_unwrap(native_file, ifgw.shape)
    payload = changed_flow_component_summary(
        ifgw,
        rowcost,
        colcost,
        native,
        snaphu,
        nshortcycle=nshortcycle,
        limit=limit,
    )
    payload["shape"] = [int(ifgw.shape[0]), int(ifgw.shape[1])]
    payload["fixture_root"] = str(root)
    payload["native_file"] = str(native_file)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize connected changed-flow components.")
    parser.add_argument("--root", type=Path, required=True, help="Stage 6 fixture root.")
    parser.add_argument("--native-file", type=Path, required=True, help="Cached native unwrap .npy file.")
    parser.add_argument("--nshortcycle", type=int, default=200)
    parser.add_argument("--limit", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = analyze_fixture_components(
        args.root,
        args.native_file,
        nshortcycle=args.nshortcycle,
        limit=args.limit,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
