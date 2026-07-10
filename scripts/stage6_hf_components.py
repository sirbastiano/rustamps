from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from scripts.stage6_hf_core import (
    TWO_PI,
    _component_shift_gain,
    _desired_flows,
    _edge_cost,
    _mask_flow_delta,
    initial_defo_objective,
)


def oracle_threshold_shift_summary(
    ifgw: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    native: np.ndarray,
    snaphu: np.ndarray,
    *,
    nshortcycle: int = 200,
) -> dict[str, Any]:
    phase = np.angle(np.asarray(ifgw, dtype=np.complex64)).astype(np.float32)
    native_labels = np.rint((np.asarray(native, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    snaphu_labels = np.rint((np.asarray(snaphu, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    correction = (snaphu_labels - native_labels).astype(np.int32)
    h_desired, v_desired = _desired_flows(phase)
    h_flow = native_labels[:, 1:] - native_labels[:, :-1] - h_desired
    v_flow = -(native_labels[1:, :] - native_labels[:-1, :] - v_desired)

    thresholds = []
    sequential_gain = 0
    for level in range(1, int(correction.max()) + 1):
        gain, h_flow, v_flow = _threshold_step(
            correction >= level,
            1,
            h_flow,
            v_flow,
            rowcost,
            colcost,
            nshortcycle=nshortcycle,
        )
        thresholds.append(_threshold_record(level, 1, correction >= level, gain))
        sequential_gain += gain
    for level in range(1, int(-correction.min()) + 1):
        gain, h_flow, v_flow = _threshold_step(
            correction <= -level,
            -1,
            h_flow,
            v_flow,
            rowcost,
            colcost,
            nshortcycle=nshortcycle,
        )
        thresholds.append(_threshold_record(level, -1, correction <= -level, gain))
        sequential_gain += gain

    native_objective = initial_defo_objective(ifgw, rowcost, colcost, native, nshortcycle=nshortcycle)
    snaphu_objective = initial_defo_objective(ifgw, rowcost, colcost, snaphu, nshortcycle=nshortcycle)
    return {
        "correction_min": int(correction.min()),
        "correction_max": int(correction.max()),
        "thresholds": thresholds,
        "sequential_gain": int(sequential_gain),
        "objective_delta_native_minus_snaphu": int(native_objective - snaphu_objective),
    }


def oracle_boundary_energy_summary(
    ifgw: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    native: np.ndarray,
    snaphu: np.ndarray,
    *,
    nshortcycle: int = 200,
) -> dict[str, Any]:
    phase = np.angle(np.asarray(ifgw, dtype=np.complex64)).astype(np.float32)
    native_labels = np.rint((np.asarray(native, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    snaphu_labels = np.rint((np.asarray(snaphu, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    correction = (snaphu_labels - native_labels).astype(np.int32)
    h_desired, v_desired = _desired_flows(phase)
    h_flow = native_labels[:, 1:] - native_labels[:, :-1] - h_desired
    v_flow = -(native_labels[1:, :] - native_labels[:-1, :] - v_desired)
    h_cost = _edge_cost(np.asarray(colcost), h_flow, nshortcycle)
    v_cost = _edge_cost(np.asarray(rowcost), v_flow, nshortcycle)
    total_cost = int(np.sum(h_cost) + np.sum(v_cost))

    thresholds = []
    for level in range(1, int(correction.max()) + 1):
        thresholds.append(_boundary_energy_record(level, 1, correction >= level, h_cost, v_cost))
    for level in range(1, int(-correction.min()) + 1):
        thresholds.append(_boundary_energy_record(level, -1, correction <= -level, h_cost, v_cost))
    return {
        "total_native_edge_cost": total_cost,
        "thresholds": thresholds,
    }


def _boundary_energy_record(
    level: int,
    shift: int,
    mask: np.ndarray,
    h_cost: np.ndarray,
    v_cost: np.ndarray,
) -> dict[str, Any]:
    h_boundary = mask[:, 1:] != mask[:, :-1]
    v_boundary = mask[1:, :] != mask[:-1, :]
    boundary_cost = int(np.sum(h_cost[h_boundary]) + np.sum(v_cost[v_boundary]))
    boundary_edges = int(np.sum(h_boundary) + np.sum(v_boundary))
    return {
        "level": int(level),
        "shift": int(shift),
        "pixels": int(np.sum(mask)),
        "boundary_h": int(np.sum(h_boundary)),
        "boundary_v": int(np.sum(v_boundary)),
        "boundary_edges": boundary_edges,
        "boundary_native_cost": boundary_cost,
    }


def _threshold_step(
    mask: np.ndarray,
    shift: int,
    h_flow: np.ndarray,
    v_flow: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    *,
    nshortcycle: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    h_delta, v_delta = _mask_flow_delta(mask, shift)
    gain = _component_shift_gain(mask, shift, h_flow, v_flow, rowcost, colcost, nshortcycle=nshortcycle)
    return int(gain), h_flow + h_delta, v_flow + v_delta


def _threshold_record(level: int, shift: int, mask: np.ndarray, gain: int) -> dict[str, int]:
    return {
        "level": int(level),
        "shift": int(shift),
        "pixels": int(np.sum(mask)),
        "boundary_h": int(np.sum(mask[:, 1:] != mask[:, :-1])),
        "boundary_v": int(np.sum(mask[1:, :] != mask[:-1, :])),
        "gain": int(gain),
    }


def component_shift_summary(
    ifgw: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    native: np.ndarray,
    snaphu: np.ndarray,
    *,
    limit: int = 8,
    nshortcycle: int = 200,
) -> dict[str, Any]:
    phase = np.angle(np.asarray(ifgw, dtype=np.complex64)).astype(np.float32)
    labels = np.rint((np.asarray(native, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    snaphu_labels = np.rint((np.asarray(snaphu, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    diff = (labels - snaphu_labels).astype(np.int32)
    h_desired, v_desired = _desired_flows(phase)
    h_flow = labels[:, 1:] - labels[:, :-1] - h_desired
    v_flow = -(labels[1:, :] - labels[:-1, :] - v_desired)
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    out: dict[str, Any] = {}
    for value in sorted(int(v) for v in np.unique(diff) if int(v) != 0):
        labels_cc, n_component = ndimage.label(diff == value, structure=structure)
        sizes = np.bincount(labels_cc.reshape(-1))
        if sizes.size:
            sizes[0] = 0
        order = np.argsort(sizes)[-limit:][::-1]
        top = []
        for component in order:
            size = int(sizes[component])
            if component == 0 or size == 0:
                continue
            mask = labels_cc == component
            rows, cols = np.where(mask)
            gain = _component_shift_gain(mask, -value, h_flow, v_flow, rowcost, colcost, nshortcycle=nshortcycle)
            top.append(
                {
                    "size": size,
                    "shift": int(-value),
                    "gain": int(gain),
                    "bbox": [int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())],
                    "boundary_h": int(np.sum(mask[:, 1:] != mask[:, :-1])),
                    "boundary_v": int(np.sum(mask[1:, :] != mask[:-1, :])),
                }
            )
        out[str(value)] = {
            "component_count": int(n_component),
            "top": top,
        }
    return out
