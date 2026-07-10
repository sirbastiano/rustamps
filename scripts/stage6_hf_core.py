from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

TWO_PI = np.float32(2.0 * np.pi)


def dense_msd(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    diff_v = (arr[:-1, :] - arr[1:, :]).reshape(-1)
    diff_h = (arr[:, :-1] - arr[:, 1:]).reshape(-1)
    diff_v = diff_v[diff_v != 0]
    diff_h = diff_h[diff_h != 0]
    denom = diff_v.size + diff_h.size
    if denom == 0:
        return 0.0
    total = np.sum(diff_v.astype(np.float64) ** 2) + np.sum(diff_h.astype(np.float64) ** 2)
    return float(total / denom)


def label_diff_summary(native: np.ndarray, snaphu: np.ndarray, *, limit: int = 12) -> dict[str, Any]:
    diff = np.rint((np.asarray(native, dtype=np.float32) - np.asarray(snaphu, dtype=np.float32)) / TWO_PI)
    diff = diff.astype(np.int32)
    values, counts = np.unique(diff, return_counts=True)
    order = np.lexsort((values, -counts))
    change_h = diff[:, :-1] != diff[:, 1:]
    change_v = diff[:-1, :] != diff[1:, :]
    return {
        "diff_min": int(diff.min()),
        "diff_max": int(diff.max()),
        "diff_unique": int(values.size),
        "top_counts": [[int(values[i]), int(counts[i])] for i in order[:limit]],
        "change_edges_h": int(change_h.sum()),
        "change_edges_v": int(change_v.sum()),
        "components": label_component_summary(diff, limit=limit),
    }


def save_native_unwrap(path: Path, native: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(native, dtype=np.float32))


def load_native_unwrap(path: Path, shape: tuple[int, int]) -> np.ndarray:
    native = np.load(path)
    if native.shape != tuple(shape):
        raise ValueError(f"cached native unwrap shape {native.shape} does not match fixture shape {shape}")
    return np.asarray(native, dtype=np.float32)


def label_component_summary(diff: np.ndarray, *, limit: int = 8) -> dict[str, Any]:
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    out: dict[str, Any] = {}
    for value in sorted(int(v) for v in np.unique(diff) if int(v) != 0):
        mask = diff == value
        labels, n_component = ndimage.label(mask, structure=structure)
        sizes = np.bincount(labels.reshape(-1))
        if sizes.size:
            sizes[0] = 0
        order = np.argsort(sizes)[-limit:][::-1]
        top = []
        for component in order:
            size = int(sizes[component])
            if component == 0 or size == 0:
                continue
            rows, cols = np.where(labels == component)
            component_mask = labels == component
            top.append(
                {
                    "size": size,
                    "bbox": [int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())],
                    "boundary_h": int(np.sum(component_mask[:, 1:] != component_mask[:, :-1])),
                    "boundary_v": int(np.sum(component_mask[1:, :] != component_mask[:-1, :])),
                }
            )
        out[str(value)] = {
            "pixels": int(mask.sum()),
            "component_count": int(n_component),
            "top": top,
        }
    return out


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


def _component_shift_gain(
    mask: np.ndarray,
    shift: int,
    h_flow: np.ndarray,
    v_flow: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    *,
    nshortcycle: int,
) -> int:
    h_delta = (mask[:, 1:].astype(np.int64) - mask[:, :-1].astype(np.int64)) * int(shift)
    v_delta = (mask[:-1, :].astype(np.int64) - mask[1:, :].astype(np.int64)) * int(shift)
    h_boundary = h_delta != 0
    v_boundary = v_delta != 0
    old_h = _edge_cost(np.asarray(colcost)[h_boundary], h_flow[h_boundary], nshortcycle).sum()
    new_h = _edge_cost(
        np.asarray(colcost)[h_boundary],
        h_flow[h_boundary] + h_delta[h_boundary],
        nshortcycle,
    ).sum()
    old_v = _edge_cost(np.asarray(rowcost)[v_boundary], v_flow[v_boundary], nshortcycle).sum()
    new_v = _edge_cost(
        np.asarray(rowcost)[v_boundary],
        v_flow[v_boundary] + v_delta[v_boundary],
        nshortcycle,
    ).sum()
    return int(old_h + old_v - new_h - new_v)


def _mask_flow_delta(mask: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    h_delta = (mask[:, 1:].astype(np.int64) - mask[:, :-1].astype(np.int64)) * int(shift)
    v_delta = (mask[:-1, :].astype(np.int64) - mask[1:, :].astype(np.int64)) * int(shift)
    return h_delta, v_delta


def _edge_cost(edge: np.ndarray, flow: np.ndarray, nshortcycle: int) -> np.ndarray:
    cost = np.maximum(np.abs(edge[..., 1].astype(np.int64)), 1)
    offset = edge[..., 0].astype(np.int64)
    dzmax = edge[..., 2].astype(np.int64)
    laycost = edge[..., 3].astype(np.int64)
    dz = np.abs(flow.astype(np.int64) * int(nshortcycle) + offset)
    dzmax = np.where(laycost == -32000, 32000, np.maximum(dzmax, 0))
    falloff = dz - dzmax
    shelf = (dz * dz) // cost
    shelf = np.where((laycost != -32000) & (shelf > laycost), laycost, shelf)
    out = np.where(dz > dzmax, (falloff * falloff) // (2 * cost) + laycost, shelf)
    return np.where(edge[..., 1] == 32000, 0, out).astype(np.int64)


def initial_defo_objective(
    ifgw: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    unwrapped: np.ndarray,
    *,
    nshortcycle: int = 200,
) -> int:
    phase = np.angle(np.asarray(ifgw, dtype=np.complex64)).astype(np.float32)
    labels = np.rint((np.asarray(unwrapped, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    h_desired, v_desired = _desired_flows(phase)
    h_flow = labels[:, 1:] - labels[:, :-1] - h_desired
    v_flow = -(labels[1:, :] - labels[:-1, :] - v_desired)
    horizontal = _edge_cost(np.asarray(colcost), h_flow, nshortcycle).sum()
    vertical = _edge_cost(np.asarray(rowcost), v_flow, nshortcycle).sum()
    return int(horizontal + vertical)
