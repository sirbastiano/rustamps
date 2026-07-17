#!/usr/bin/env python3
"""Developer-only historical-oracle fixture generator; not a production runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from pystamps.config import load_config
from pystamps.io.mat import read_mat, write_mat
from pystamps.pipeline.stages import run_pipeline
from pystamps.pipeline.types import PipelineContext


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("inputs_and_outputs/validation_runs/native_conda_e2e")
DEFAULT_CONFIG = Path("configs/native-kernels.yaml")
N_SIDE = 8
N_PS = N_SIDE * N_SIDE
SLAVE_DATES = ("20230101", "20230113", "20230125", "20230206", "20230218", "20230302")
MASTER_DATE = "20230107"
MASTER_IX = 2

STAGE_FILES: dict[int, tuple[str, ...]] = {
    1: ("PATCH_1/ps1.mat", "PATCH_1/ph1.mat", "PATCH_1/bp1.mat"),
    2: ("PATCH_1/pm1.mat",),
    3: ("PATCH_1/select1.mat",),
    4: ("PATCH_1/weed1.mat",),
    5: ("PATCH_1/ph2.mat", "ps2.mat", "ph2.mat", "pm2.mat", "ifgstd2.mat"),
    6: ("phuw2.mat", "uw_phaseuw.mat", "uw_grid.mat", "uw_interp.mat"),
    7: ("scla2.mat", "scla_smooth2.mat"),
    8: ("mean_v.mat", "uw_space_time.mat"),
}


def _safe_reset(root: Path) -> Path:
    target = root.expanduser().resolve()
    allowed = (REPO_ROOT / "inputs_and_outputs/validation_runs").resolve()
    if target.parent != allowed or target.name != "native_conda_e2e":
        raise SystemExit(f"refusing to replace non-validation path: {target}")
    if target.exists():
        shutil.rmtree(target)
    (target / "PATCH_1").mkdir(parents=True)
    return target


def _write_phase(path: Path, phase: np.ndarray) -> None:
    values = np.exp(1j * phase).astype(np.complex64).T
    interleaved = np.empty((values.shape[0], values.shape[1] * 2), dtype=">f4")
    interleaved[:, 0::2] = values.real
    interleaved[:, 1::2] = values.imag
    interleaved.tofile(path)


def prepare(root: Path) -> dict[str, Any]:
    root = _safe_reset(root)
    patch = root / "PATCH_1"
    row, col = np.indices((N_SIDE, N_SIDE))
    row_f = row.reshape(-1).astype(np.float64)
    col_f = col.reshape(-1).astype(np.float64)
    ids = np.arange(1, N_PS + 1, dtype=np.float64)
    ij = np.column_stack((ids, row_f + 20.0, col_f + 30.0))
    lonlat = np.column_stack((12.40 + col_f * 0.00072, 41.90 + row_f * 0.00055)).astype(">f4")

    bperp = np.asarray([-120.0, -55.0, 25.0, 85.0, 145.0, 205.0], dtype=np.float64)
    topo = 0.0018 + 0.00035 * np.sin((row_f + 1.0) * 0.7) + 0.0002 * np.cos((col_f + 1.0) * 0.5)
    time = np.arange(1, len(SLAVE_DATES) + 1, dtype=np.float64)
    deform = 0.025 * (row_f[:, None] - col_f[:, None]) * time[None, :] / N_SIDE
    noise = 0.006 * np.sin((ids[:, None] + 2.0 * time[None, :]) * 0.31)
    phase = topo[:, None] * bperp[None, :] + deform + noise

    np.savetxt(patch / "pscands.1.ij", ij, fmt=["%d", "%.0f", "%.0f"])
    np.savetxt(patch / "pscands.1.da", 0.2 + 0.01 * (ids % 7), fmt="%.8f")
    lonlat.tofile(patch / "pscands.1.ll")
    _write_phase(patch / "pscands.1.ph", phase)
    (root / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    (root / "width.txt").write_text("64\n", encoding="utf-8")
    (root / "len.txt").write_text("64\n", encoding="utf-8")
    (root / "day.1.in").write_text("\n".join(SLAVE_DATES) + "\n", encoding="utf-8")
    (root / "master_day.1.in").write_text(MASTER_DATE + "\n", encoding="utf-8")
    (root / "bperp.1.in").write_text("\n".join(str(v) for v in bperp) + "\n", encoding="utf-8")
    (patch / "patch.in").write_text("1 100 1 100\n", encoding="utf-8")
    (patch / "patch_noover.in").write_text("1 100 1 100\n", encoding="utf-8")
    write_mat(
        root / "parms.mat",
        {
            "filter_grid_size": np.asarray(50.0),
            "clap_win": np.asarray(8.0),
            "clap_low_pass_wavelength": np.asarray(200.0),
            "gamma_change_convergence": np.asarray(1.0e9),
            "gamma_max_iterations": np.asarray(1.0),
            "filter_weighting": "P-square",
            "select_method": "PERCENT",
            "percent_rand": np.asarray(100.0),
            "gamma_stdev_reject": np.asarray(0.0),
            "small_baseline_flag": "n",
            "weed_neighbours": "n",
            "weed_zero_elevation": "n",
            "weed_standard_dev": np.asarray(np.pi),
            "weed_max_noise": np.asarray(np.pi),
            "unwrap_method": "3D_FULL",
            "unwrap_patch_phase": "n",
            "unwrap_prefilter_flag": "n",
            "unwrap_grid_size": np.asarray(40.0),
            "unwrap_la_error_flag": "y",
            "unwrap_spatial_cost_func_flag": "n",
            "unwrap_time_win": np.asarray(36.0),
            "scla_deramp": "y",
            "heading": np.asarray(190.0),
            "lambda": np.asarray(0.0555),
            "max_topo_err": np.asarray(15.0),
        },
    )
    return {"root": str(root), "n_ps": N_PS, "n_ifg": len(SLAVE_DATES) + 1}


def _array(root: Path, relative: str, key: str) -> np.ndarray:
    return np.asarray(read_mat(root / relative)[key])


def _finite(name: str, values: np.ndarray, *, allow_some_nan: bool = False) -> None:
    finite = np.isfinite(values)
    if allow_some_nan:
        if not finite.any():
            raise RuntimeError(f"{name} has no finite values")
    elif not finite.all():
        raise RuntimeError(f"{name} contains non-finite values")


def validate_stage(root: Path, stage: int) -> dict[str, Any]:
    missing = [name for name in STAGE_FILES[stage] if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"stage {stage} missing outputs: {', '.join(missing)}")
    ps1 = read_mat(root / "PATCH_1/ps1.mat")
    if int(np.asarray(ps1["n_ps"]).reshape(-1)[0]) != N_PS:
        raise RuntimeError("stage 1 candidate count changed")
    if stage == 1:
        ph = _array(root, "PATCH_1/ph1.mat", "ph")
        if ph.shape != (N_PS, len(SLAVE_DATES) + 1):
            raise RuntimeError(f"unexpected ph1 shape: {ph.shape}")
        _finite("ph1.ph", ph)
        if not np.allclose(ph[:, MASTER_IX - 1], 1.0 + 0.0j):
            raise RuntimeError("stage 1 master column is not unity")
    elif stage == 2:
        pm = read_mat(root / "PATCH_1/pm1.mat")
        coh = np.asarray(pm["coh_ps"]).reshape(-1)
        if coh.shape != (N_PS,) or np.any((coh < 0.0) | (coh > 1.000001)):
            raise RuntimeError("stage 2 coherence output is invalid")
        _finite("pm1.coh_ps", coh)
        _finite("pm1.ph_patch", np.asarray(pm["ph_patch"]))
    elif stage == 3:
        ix = _array(root, "PATCH_1/select1.mat", "ix").reshape(-1).astype(np.int64)
        if ix.size < 4 or ix.min() < 1 or ix.max() > N_PS:
            raise RuntimeError(f"stage 3 selected invalid PS indices: {ix.size}")
    elif stage == 4:
        keep = _array(root, "PATCH_1/weed1.mat", "ix_weed").reshape(-1).astype(bool)
        if keep.size < 4 or np.count_nonzero(keep) < 4:
            raise RuntimeError("stage 4 retained too few PS")
    elif stage == 5:
        ps2 = read_mat(root / "ps2.mat")
        n_ps = int(np.asarray(ps2["n_ps"]).reshape(-1)[0])
        ph2 = _array(root, "ph2.mat", "ph")
        if n_ps < 4 or ph2.shape != (n_ps, len(SLAVE_DATES) + 1):
            raise RuntimeError(f"invalid merged Stage 5 shape: {ph2.shape}")
        _finite("ph2.ph", ph2)
        _finite("ifgstd2.ifg_std", _array(root, "ifgstd2.mat", "ifg_std"))
    elif stage == 6:
        phuw = _array(root, "phuw2.mat", "ph_uw")
        _finite("phuw2.ph_uw", phuw)
        if not np.allclose(phuw[:, MASTER_IX - 1], 0.0):
            raise RuntimeError("stage 6 master column is not zero")
        _finite("uw_grid.ph", _array(root, "uw_grid.mat", "ph"))
        if _array(root, "uw_interp.mat", "edgs").size == 0:
            raise RuntimeError("stage 6 produced no interpolation edges")
    elif stage == 7:
        for relative in ("scla2.mat", "scla_smooth2.mat"):
            _finite(f"{relative}.K_ps_uw", _array(root, relative, "K_ps_uw"))
            _finite(f"{relative}.ph_scla", _array(root, relative, "ph_scla"))
    elif stage == 8:
        mean_v = _array(root, "mean_v.mat", "m")
        _finite("mean_v.m", mean_v)
        noise = _array(root, "uw_space_time.mat", "dph_noise")
        _finite("uw_space_time.dph_noise", noise, allow_some_nan=True)
    return {"stage": stage, "files": list(STAGE_FILES[stage])}


def _reset_stage_outputs(root: Path, stage: int) -> None:
    for current_stage in range(stage, 9):
        for relative in STAGE_FILES[current_stage]:
            (root / relative).unlink(missing_ok=True)


def run_stage(root: Path, stage: int, config: Path) -> dict[str, Any]:
    _reset_stage_outputs(root, stage)
    context = PipelineContext(
        dataset_root=root.resolve(),
        run_config=load_config(config),
        start_step=stage,
        end_step=stage,
    )
    report = run_pipeline(context)
    if report.failures:
        raise RuntimeError("; ".join(item.details for item in report.failures))
    if not report.results or any(item.status != "completed" for item in report.results):
        states = ", ".join(f"{item.scope}:{item.status}" for item in report.results)
        raise RuntimeError(f"stage {stage} did not execute cleanly: {states}")
    validated = validate_stage(root, stage)
    validated["results"] = [
        {"scope": item.scope, "details": item.details, "duration_sec": round(item.duration_sec, 6)}
        for item in report.results
    ]
    return validated


def verify(root: Path) -> dict[str, Any]:
    stages = [validate_stage(root, stage) for stage in sorted(STAGE_FILES)]
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for stage in stages:
        for relative in stage["files"]:
            path = root / relative
            data = path.read_bytes()
            digest.update(relative.encode("utf-8"))
            digest.update(data)
            files.append({"path": relative, "bytes": len(data)})
    payload = {
        "ok": True,
        "backend": "native",
        "n_ps_input": N_PS,
        "n_ifg": len(SLAVE_DATES) + 1,
        "stages": stages,
        "artifact_count": len(files),
        "artifact_bytes": sum(item["bytes"] for item in files),
        "artifact_sha256": digest.hexdigest(),
        "files": files,
    }
    output = root / "native_e2e_output_manifest.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["manifest"] = str(output)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and validate the compact native Stage 1-8 fixture.")
    parser.add_argument("command", choices=("prepare", "stage", "verify"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", type=int, choices=range(1, 9))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "prepare":
        payload = prepare(args.root)
    elif args.command == "stage":
        if args.stage is None:
            raise SystemExit("--stage is required for the stage command")
        payload = run_stage(args.root, args.stage, args.config)
    else:
        payload = verify(args.root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
