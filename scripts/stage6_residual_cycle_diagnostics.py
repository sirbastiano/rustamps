#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from scripts.stage6_hf_diagnostics import TWO_PI, _desired_flows, _edge_cost, _load_fixture, load_native_unwrap


def residual_cycle_summary(
    ifgw: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    unwrapped: np.ndarray,
    *,
    nshortcycle: int = 200,
    max_nodes: int = 20_000,
) -> dict[str, Any]:
    phase = np.angle(np.asarray(ifgw, dtype=np.complex64)).astype(np.float32)
    labels = np.rint((np.asarray(unwrapped, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    rows, cols = _shape_costs(rowcost, colcost, phase.shape)
    nrow, ncol = phase.shape
    node_count = max(nrow - 1, 0) * max(ncol - 1, 0) + 1
    if node_count > max_nodes:
        return {"status": "skipped", "node_count": int(node_count), "max_nodes": int(max_nodes)}
    h_flow, v_flow = _label_flows(phase, labels)
    arcs = _residual_arcs(rows, cols, h_flow, v_flow, nshortcycle)
    has_cycle, last_cost = _has_negative_cycle(node_count, arcs)
    return {
        "status": "ok",
        "node_count": int(node_count),
        "arc_count": int(len(arcs)),
        "negative_cycle": bool(has_cycle),
        "last_relaxed_cost": int(last_cost) if last_cost is not None else None,
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


def _edge_increment(edge: np.ndarray, flow: int, delta: int, nshortcycle: int) -> int:
    base = _edge_cost(edge, np.asarray(flow, dtype=np.int64), nshortcycle)
    new = _edge_cost(edge, np.asarray(flow + delta, dtype=np.int64), nshortcycle)
    return int(new - base)


def _residual_arcs(
    rowcost: np.ndarray,
    colcost: np.ndarray,
    h_flow: np.ndarray,
    v_flow: np.ndarray,
    nshortcycle: int,
) -> list[tuple[int, int, int]]:
    nrow = colcost.shape[0]
    ncol = rowcost.shape[1] if rowcost.size else colcost.shape[1] + 1
    prn = nrow - 1
    pcn = ncol - 1
    ground = prn * pcn
    arcs: list[tuple[int, int, int]] = []
    for row in range(nrow):
        for col in range(pcn):
            edge = colcost[row, col]
            if int(edge[3]) == 0:
                continue
            if row == 0:
                start, end = ground, col
            elif row + 1 == nrow:
                start, end = (row - 1) * pcn + col, ground
            else:
                start, end = (row - 1) * pcn + col, row * pcn + col
            flow = int(h_flow[row, col])
            arcs.append((start, end, _edge_increment(edge, flow, 1, nshortcycle)))
            arcs.append((end, start, _edge_increment(edge, flow, -1, nshortcycle)))
    for row in range(prn):
        for col in range(ncol):
            edge = rowcost[row, col]
            if int(edge[3]) == 0:
                continue
            if col == 0:
                start, end = row * pcn, ground
            elif col + 1 == ncol:
                start, end = ground, row * pcn + col - 1
            else:
                start, end = row * pcn + col, row * pcn + col - 1
            flow = int(v_flow[row, col])
            arcs.append((start, end, _edge_increment(edge, flow, -1, nshortcycle)))
            arcs.append((end, start, _edge_increment(edge, flow, 1, nshortcycle)))
    return arcs


def _has_negative_cycle(node_count: int, arcs: list[tuple[int, int, int]]) -> tuple[bool, int | None]:
    adjacency = [[] for _ in range(node_count)]
    for start, end, cost in arcs:
        adjacency[start].append((end, cost))
    dist = [0] * node_count
    relax_count = [0] * node_count
    queue = deque(range(node_count))
    in_queue = [True] * node_count
    while queue:
        start = queue.popleft()
        in_queue[start] = False
        for end, cost in adjacency[start]:
            candidate = dist[start] + cost
            if candidate < dist[end]:
                dist[end] = candidate
                relax_count[end] += 1
                if relax_count[end] >= node_count:
                    return True, cost
                if not in_queue[end]:
                    queue.append(end)
                    in_queue[end] = True
    return False, None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Stage 6 unit residual-cycle availability.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--native-file", required=True, type=Path)
    parser.add_argument("--max-nodes", default=20_000, type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _nzix, ifgw, rowcost, colcost, _snaphu = _load_fixture(args.root)
    native = load_native_unwrap(args.native_file, ifgw.shape)
    payload = residual_cycle_summary(ifgw, rowcost, colcost, native, max_nodes=args.max_nodes)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
