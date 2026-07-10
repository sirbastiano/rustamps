from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from scripts.stage6_hf_core import TWO_PI, _component_shift_gain, _desired_flows, _edge_cost


def component_isolation_summary(
    ifgw: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    native: np.ndarray,
    snaphu: np.ndarray,
    *,
    limit: int = 8,
    nshortcycle: int = 200,
    cut_max_cells: int = 16_384,
) -> dict[str, Any]:
    phase = np.angle(np.asarray(ifgw, dtype=np.complex64)).astype(np.float32)
    labels = np.rint((np.asarray(native, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    snaphu_labels = np.rint((np.asarray(snaphu, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    diff = (labels - snaphu_labels).astype(np.int32)
    h_desired, v_desired = _desired_flows(phase)
    h_flow = labels[:, 1:] - labels[:, :-1] - h_desired
    v_flow = -(labels[1:, :] - labels[:-1, :] - v_desired)
    h_cost = _edge_cost(np.asarray(colcost), h_flow, nshortcycle)
    v_cost = _edge_cost(np.asarray(rowcost), v_flow, nshortcycle)
    positive = np.concatenate([h_cost[h_cost > 0].reshape(-1), v_cost[v_cost > 0].reshape(-1)])
    barrier = int(positive.sum() // positive.size) if positive.size else 0
    thresholds = [max(barrier // divisor, 1) for divisor in [1, 2, 4, 8, 16, 32]] if barrier > 0 else []
    cut_side = int(np.sqrt(max(int(cut_max_cells), 0)))
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    out: dict[str, Any] = {
        "positive_edge_mean_energy": barrier,
        "barrier_thresholds": thresholds,
        "cut_max_cells": int(cut_max_cells),
        "cut_square_side": cut_side,
    }
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
            bbox_height = int(rows.max() - rows.min() + 1)
            bbox_width = int(cols.max() - cols.min() + 1)
            boundary_energy, h_boundary, v_boundary = _boundary_energy(mask, h_cost, v_cost)
            boundary_min = int(boundary_energy.min()) if boundary_energy.size else None
            top.append(
                {
                    "size": size,
                    "shift": int(-value),
                    "gain": int(
                        _component_shift_gain(
                            mask,
                            -value,
                            h_flow,
                            v_flow,
                            rowcost,
                            colcost,
                            nshortcycle=nshortcycle,
                        )
                    ),
                    "bbox": [int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())],
                    "bbox_height": bbox_height,
                    "bbox_width": bbox_width,
                    "bbox_cells": bbox_height * bbox_width,
                    "fits_cut_cell_budget": bool(bbox_height * bbox_width <= cut_max_cells),
                    "fits_square_cut_window": bool(bbox_height <= cut_side and bbox_width <= cut_side),
                    "boundary_h": int(np.sum(h_boundary)),
                    "boundary_v": int(np.sum(v_boundary)),
                    "boundary_edges": int(boundary_energy.size),
                    "boundary_min": boundary_min,
                    "boundary_p05": float(np.percentile(boundary_energy, 5)) if boundary_energy.size else None,
                    "boundary_median": float(np.median(boundary_energy)) if boundary_energy.size else None,
                    "boundary_max": int(boundary_energy.max()) if boundary_energy.size else None,
                    "isolated_thresholds": [
                        int(threshold)
                        for threshold in thresholds
                        if boundary_min is not None and boundary_min >= threshold
                    ],
                }
            )
        out[str(value)] = {
            "component_count": int(n_component),
            "top": top,
        }
    return out


def _boundary_energy(
    mask: np.ndarray,
    h_cost: np.ndarray,
    v_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_boundary = mask[:, 1:] != mask[:, :-1]
    v_boundary = mask[1:, :] != mask[:-1, :]
    boundary_energy = np.concatenate([h_cost[h_boundary].reshape(-1), v_cost[v_boundary].reshape(-1)])
    return boundary_energy, h_boundary, v_boundary
