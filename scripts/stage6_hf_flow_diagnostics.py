from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

TWO_PI = np.float32(2.0 * np.pi)


def edge_flow_diff_summary(
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
    labels_native = np.rint((np.asarray(native, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    labels_snaphu = np.rint((np.asarray(snaphu, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    h_desired, v_desired = _desired_flows(phase)
    native_h, native_v = _label_flows(labels_native, h_desired, v_desired)
    snaphu_h, snaphu_v = _label_flows(labels_snaphu, h_desired, v_desired)
    shaped_rowcost, shaped_colcost = _reshape_costs(rowcost, colcost, phase.shape)
    horizontal = _axis_summary(shaped_colcost, native_h, snaphu_h, nshortcycle, limit)
    vertical = _axis_summary(shaped_rowcost, native_v, snaphu_v, nshortcycle, limit)
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "inferred_distribution": flow_distribution_summary(
            np.concatenate([native_v.reshape(-1), native_h.reshape(-1)]),
            np.concatenate([snaphu_v.reshape(-1), snaphu_h.reshape(-1)]),
        ),
        "total_changed_edges": horizontal["changed_edges"] + vertical["changed_edges"],
        "total_native_cost_on_changed": horizontal["native_cost_on_changed"]
        + vertical["native_cost_on_changed"],
        "total_snaphu_cost_on_changed": horizontal["snaphu_cost_on_changed"]
        + vertical["snaphu_cost_on_changed"],
    }


def flow_dump_match_summary(
    ifgw: np.ndarray,
    unwrapped: np.ndarray,
    row_flow: np.ndarray,
    col_flow: np.ndarray,
) -> dict[str, Any]:
    phase = np.angle(np.asarray(ifgw, dtype=np.complex64)).astype(np.float32)
    labels = np.rint((np.asarray(unwrapped, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    h_desired, v_desired = _desired_flows(phase)
    inferred_h, inferred_v = _label_flows(labels, h_desired, v_desired)
    rows = np.asarray(row_flow, dtype=np.int64)
    cols = np.asarray(col_flow, dtype=np.int64)
    if rows.shape != inferred_v.shape:
        raise ValueError("row_flow must have shape (nrow - 1, ncol)")
    if cols.shape != inferred_h.shape:
        raise ValueError("col_flow must have shape (nrow, ncol - 1)")
    row_diff = inferred_v - rows
    col_diff = inferred_h - cols
    return {
        "exact": bool(not np.any(row_diff) and not np.any(col_diff)),
        "row_mismatch_count": int(np.sum(row_diff != 0)),
        "col_mismatch_count": int(np.sum(col_diff != 0)),
        "row_max_abs_diff": int(np.max(np.abs(row_diff))) if row_diff.size else 0,
        "col_max_abs_diff": int(np.max(np.abs(col_diff))) if col_diff.size else 0,
    }


def flow_dump_summaries(
    ifgw: np.ndarray,
    native: np.ndarray,
    snaphu: np.ndarray,
    row_flow: np.ndarray,
    col_flow: np.ndarray,
) -> dict[str, Any]:
    phase = np.angle(np.asarray(ifgw, dtype=np.complex64)).astype(np.float32)
    h_desired, v_desired = _desired_flows(phase)
    native_h, native_v = _label_flows(
        np.rint((np.asarray(native, dtype=np.float32) - phase) / TWO_PI).astype(np.int64),
        h_desired,
        v_desired,
    )
    snaphu_h, snaphu_v = _label_flows(
        np.rint((np.asarray(snaphu, dtype=np.float32) - phase) / TWO_PI).astype(np.int64),
        h_desired,
        v_desired,
    )
    return {
        "native": flow_dump_match_summary(ifgw, native, row_flow, col_flow),
        "snaphu": flow_dump_match_summary(ifgw, snaphu, row_flow, col_flow),
        "row_distribution": flow_distribution_summary(native_v, np.asarray(row_flow)),
        "col_distribution": flow_distribution_summary(native_h, np.asarray(col_flow)),
        "inferred_distribution": flow_distribution_summary(
            np.concatenate([native_v.reshape(-1), native_h.reshape(-1)]),
            np.concatenate([snaphu_v.reshape(-1), snaphu_h.reshape(-1)]),
        ),
    }


def flow_distribution_summary(native_flow: np.ndarray, snaphu_flow: np.ndarray) -> dict[str, Any]:
    native = np.asarray(native_flow, dtype=np.int64).reshape(-1)
    snaphu = np.asarray(snaphu_flow, dtype=np.int64).reshape(-1)
    if native.shape != snaphu.shape:
        raise ValueError("native_flow and snaphu_flow must have matching sizes")
    delta = snaphu - native
    return {
        "native_abs_counts": _value_counts(np.abs(native)),
        "snaphu_abs_counts": _value_counts(np.abs(snaphu)),
        "delta_counts": _value_counts(delta),
        "changed_edges": int(np.sum(delta != 0)),
    }


def load_snaphu_flow(path: str | Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    nrow, ncol = shape
    row_elems = (nrow - 1) * ncol
    expected = row_elems + nrow * (ncol - 1)
    raw = np.fromfile(path, dtype=np.int16)
    if raw.size != expected:
        raise RuntimeError(f"snaphu.flow has {raw.size} int16 values, expected {expected}")
    return raw[:row_elems].reshape((nrow - 1, ncol)), raw[row_elems:].reshape((nrow, ncol - 1))


def _value_counts(values: np.ndarray) -> list[list[int]]:
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return [[int(value), int(count)] for value, count in zip(unique, counts)]


def _label_flows(
    labels: np.ndarray,
    h_desired: np.ndarray,
    v_desired: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h_flow = labels[:, 1:] - labels[:, :-1] - h_desired
    v_flow = -(labels[1:, :] - labels[:-1, :] - v_desired)
    return h_flow, v_flow


def _reshape_costs(
    rowcost: np.ndarray,
    colcost: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    nrow, ncol = shape
    rows = np.asarray(rowcost)
    cols = np.asarray(colcost)
    if rows.shape == (nrow - 1, ncol * 4):
        rows = rows.reshape((nrow - 1, ncol, 4))
    if cols.shape == (nrow, (ncol - 1) * 4):
        cols = cols.reshape((nrow, ncol - 1, 4))
    if rows.shape != (nrow - 1, ncol, 4):
        raise ValueError("rowcost must have shape (nrow - 1, ncol, 4) or (nrow - 1, ncol * 4)")
    if cols.shape != (nrow, ncol - 1, 4):
        raise ValueError("colcost must have shape (nrow, ncol - 1, 4) or (nrow, (ncol - 1) * 4)")
    return rows, cols


def _axis_summary(
    cost: np.ndarray,
    native_flow: np.ndarray,
    snaphu_flow: np.ndarray,
    nshortcycle: int,
    limit: int,
) -> dict[str, Any]:
    changed = native_flow != snaphu_flow
    native_cost = _edge_cost(cost, native_flow, nshortcycle)
    snaphu_cost = _edge_cost(cost, snaphu_flow, nshortcycle)
    delta = snaphu_flow - native_flow
    delta_counts = []
    if np.any(changed):
        values, counts = np.unique(delta[changed], return_counts=True)
        order = np.lexsort((values, -counts))
        delta_counts = [[int(values[i]), int(counts[i])] for i in order]
    return {
        "changed_edges": int(np.sum(changed)),
        "delta_counts": delta_counts,
        "native_cost_on_changed": int(np.sum(native_cost[changed])),
        "snaphu_cost_on_changed": int(np.sum(snaphu_cost[changed])),
        "top_cost_delta_edges": _top_changed_edges(
            cost,
            native_flow,
            snaphu_flow,
            native_cost,
            snaphu_cost,
            changed,
            limit,
        ),
    }


def _top_changed_edges(
    cost: np.ndarray,
    native_flow: np.ndarray,
    snaphu_flow: np.ndarray,
    native_cost: np.ndarray,
    snaphu_cost: np.ndarray,
    changed: np.ndarray,
    limit: int,
) -> list[dict[str, Any]]:
    coords = np.argwhere(changed)
    if coords.size == 0:
        return []
    deltas = native_cost[changed] - snaphu_cost[changed]
    order = np.lexsort((coords[:, 1], coords[:, 0], -np.abs(deltas)))
    top = []
    for idx in order[:limit]:
        row, col = coords[idx]
        top.append(
            {
                "row": int(row),
                "col": int(col),
                "native_flow": int(native_flow[row, col]),
                "snaphu_flow": int(snaphu_flow[row, col]),
                "native_minus_snaphu_cost": int(native_cost[row, col] - snaphu_cost[row, col]),
                "cost": [int(v) for v in cost[row, col]],
            }
        )
    return top


def _wrap_phase(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _desired_flows(phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h_desired = np.rint(
        (phase[:, :-1] + _wrap_phase(phase[:, 1:] - phase[:, :-1]) - phase[:, 1:]) / TWO_PI
    ).astype(np.int64)
    v_desired = np.rint(
        (phase[:-1, :] + _wrap_phase(phase[1:, :] - phase[:-1, :]) - phase[1:, :]) / TWO_PI
    ).astype(np.int64)
    return h_desired, v_desired


def _edge_cost(edge: np.ndarray, flow: np.ndarray, nshortcycle: int) -> np.ndarray:
    sigsq = np.maximum(np.abs(edge[..., 1].astype(np.int64)), 1)
    offset = edge[..., 0].astype(np.int64)
    dzmax = edge[..., 2].astype(np.int64)
    laycost = edge[..., 3].astype(np.int64)
    dz = np.abs(flow.astype(np.int64) * int(nshortcycle) + offset)
    dzmax = np.where(laycost == -32000, 32000, np.maximum(dzmax, 0))
    falloff = dz - dzmax
    shelf = (dz * dz) // sigsq
    shelf = np.where((laycost != -32000) & (shelf > laycost), laycost, shelf)
    out = np.where(dz > dzmax, (falloff * falloff) // (2 * sigsq) + laycost, shelf)
    return np.where(edge[..., 1] == 32000, 0, out).astype(np.int64)
