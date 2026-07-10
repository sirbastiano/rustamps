#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from pystamps.io.mat import read_mat
from pystamps.kernels import run_stage6_unwrap_grid_kernel
from scripts.stage6_hf_components import (
    component_shift_summary,
    oracle_boundary_energy_summary,
    oracle_threshold_shift_summary,
)
from scripts.stage6_hf_core import (
    TWO_PI,
    _component_shift_gain,
    _desired_flows,
    _edge_cost,
    _mask_flow_delta,
    _wrap_phase,
    dense_msd,
    initial_defo_objective,
    label_component_summary,
    label_diff_summary,
    load_native_unwrap,
    save_native_unwrap,
)
from scripts.stage6_hf_flow_diagnostics import edge_flow_diff_summary, flow_dump_summaries, load_snaphu_flow
from scripts.stage6_hf_isolation import component_isolation_summary


def _load_fixture(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nzix = np.asarray(read_mat(root / "uw_grid.mat")["nzix"], dtype=bool)
    nrow, ncol = nzix.shape
    row_elems = (nrow - 1) * ncol * 4
    cost_raw = np.fromfile(root / "snaphu.costinfile", dtype=np.int16)
    expected = row_elems + nrow * (ncol - 1) * 4
    if cost_raw.size != expected:
        raise RuntimeError(f"snaphu.costinfile has {cost_raw.size} int16 values, expected {expected}")
    rowcost = cost_raw[:row_elems].reshape((nrow - 1, ncol, 4))
    colcost = cost_raw[row_elems:].reshape((nrow, ncol - 1, 4))
    ifgw = np.fromfile(root / "snaphu.in", dtype=np.complex64).reshape((nrow, ncol))
    snaphu = np.fromfile(root / "snaphu.out", dtype=np.float32).reshape((nrow, ncol))
    return nzix, ifgw, rowcost, colcost, snaphu


def analyze_fixture(
    root: Path,
    *,
    backend: str = "native",
    nshortcycle: float = 200.0,
    flowfile: Path | None = None,
    threads: int = 0,
    native_file: Path | None = None,
    save_native: Path | None = None,
) -> dict[str, Any]:
    nzix, ifgw, rowcost, colcost, snaphu = _load_fixture(root)
    nrow, ncol = nzix.shape
    if native_file is None:
        native_t0 = time.perf_counter()
        native_payload = run_stage6_unwrap_grid_kernel(
            ifgw,
            rowcost.reshape((nrow - 1, ncol * 4)),
            colcost.reshape((nrow, (ncol - 1) * 4)),
            backend=backend,
            nshortcycle=nshortcycle,
            threads=threads,
        )
        native_seconds = time.perf_counter() - native_t0
        native = np.asarray(native_payload["ifguw"], dtype=np.float32)
        if save_native is not None:
            save_native_unwrap(save_native, native)
        native_source = "kernel"
        native_msd = float(native_payload["msd"])
        native_flow_cycles = int(native_payload.get("flow_cycles", -1))
        native_flow_objective = int(native_payload.get("flow_objective", -1))
        post_label_flow_cycles = int(native_payload.get("post_label_flow_cycles", -1))
        post_label_flow_objective = int(native_payload.get("post_label_flow_objective", -1))
    else:
        native = load_native_unwrap(native_file, (nrow, ncol))
        native_seconds = 0.0
        native_source = str(native_file)
        native_msd = dense_msd(native)
        native_flow_cycles = -1
        native_flow_objective = -1
        post_label_flow_cycles = -1
        post_label_flow_objective = -1
    wrap_diff = np.angle(np.exp(1j * (native - snaphu)))
    summary = {
        "shape": [int(nrow), int(ncol)],
        "nzix_count": int(nzix.sum()),
        "native_source": native_source,
        "native_seconds": float(native_seconds),
        "native_msd": native_msd,
        "native_flow_cycles": native_flow_cycles,
        "native_flow_objective": native_flow_objective,
        "post_label_flow_cycles": post_label_flow_cycles,
        "post_label_flow_objective": post_label_flow_objective,
        "snaphu_msd": dense_msd(snaphu),
        "max_wrap_diff": float(np.nanmax(np.abs(wrap_diff))),
        "native_initial_defo_objective": initial_defo_objective(ifgw, rowcost, colcost, native),
        "snaphu_initial_defo_objective": initial_defo_objective(ifgw, rowcost, colcost, snaphu),
    }
    summary["objective_delta_native_minus_snaphu"] = (
        summary["native_initial_defo_objective"] - summary["snaphu_initial_defo_objective"]
    )
    summary.update(label_diff_summary(native, snaphu))
    rounded_nshortcycle = int(round(nshortcycle))
    summary["component_shift_gains"] = component_shift_summary(
        ifgw, rowcost, colcost, native, snaphu, limit=8, nshortcycle=rounded_nshortcycle
    )
    summary["component_isolation"] = component_isolation_summary(
        ifgw, rowcost, colcost, native, snaphu, limit=8, nshortcycle=rounded_nshortcycle
    )
    summary["oracle_threshold_shifts"] = oracle_threshold_shift_summary(
        ifgw, rowcost, colcost, native, snaphu, nshortcycle=rounded_nshortcycle
    )
    summary["oracle_boundary_energy"] = oracle_boundary_energy_summary(
        ifgw, rowcost, colcost, native, snaphu, nshortcycle=rounded_nshortcycle
    )
    summary["edge_flow_diff"] = edge_flow_diff_summary(ifgw, rowcost, colcost, native, snaphu)
    if flowfile is not None:
        flow = load_snaphu_flow(flowfile, (nrow, ncol))
        summary["flow_dump_match"] = flow_dump_summaries(ifgw, native, snaphu, *flow)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize native Stage 6 vs saved SNAPHU fixture output.")
    parser.add_argument(
        "--root",
        default="inputs_and_outputs/InSAR_dataset_test",
        help="Dataset fixture directory containing uw_grid.mat and snaphu.* files.",
    )
    parser.add_argument("--backend", default="native", choices=("native", "auto"))
    parser.add_argument("--nshortcycle", default=200.0, type=float)
    parser.add_argument("--threads", default=0, type=int, help="Native Stage 6 threads; 0 uses the Rayon default.")
    parser.add_argument("--flowfile", type=Path, help="Optional SNAPHU FLOWFILE to compare against.")
    parser.add_argument("--native-file", type=Path, help="Load cached native unwrap .npy and skip the native solve.")
    parser.add_argument("--save-native", type=Path, help="Save the computed native unwrap .npy for faster follow-up diagnostics.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = analyze_fixture(
        Path(args.root),
        backend=args.backend,
        nshortcycle=args.nshortcycle,
        flowfile=args.flowfile,
        threads=args.threads,
        native_file=args.native_file,
        save_native=args.save_native,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
