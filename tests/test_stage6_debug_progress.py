import copy
from pathlib import Path

import numpy as np

import pystamps.pipeline.ported as ported
from pystamps.io.mat import write_mat


def _write_minimal_stage6_inputs(dataset_root: Path) -> None:
    n_ps = 2
    n_ifg = 3
    master_ix = 2
    write_mat(
        dataset_root / "ps2.mat",
        {
            "n_ps": np.asarray(float(n_ps), dtype=np.float64),
            "n_ifg": np.asarray(float(n_ifg), dtype=np.float64),
            "n_image": np.asarray(float(n_ifg), dtype=np.float64),
            "master_ix": np.asarray(float(master_ix), dtype=np.float64),
            "day": np.asarray([[10.0], [20.0], [30.0]], dtype=np.float64),
            "bperp": np.asarray([[10.0], [0.0], [20.0]], dtype=np.float32),
            "xy": np.asarray([[1.0, 0.0, 0.0], [2.0, 41.0, 0.0]], dtype=np.float32),
            "ij": np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0]], dtype=np.float64),
            "lonlat": np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            "ll0": np.asarray([0.0, 0.0], dtype=np.float64),
            "mean_range": np.asarray(830000.0, dtype=np.float64),
            "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
        },
    )
    write_mat(dataset_root / "ph2.mat", {"ph": np.ones((n_ps, n_ifg), dtype=np.complex64)})
    write_mat(
        dataset_root / "pm2.mat",
        {
            "K_ps": np.asarray([[0.0], [0.0]], dtype=np.float64),
            "C_ps": np.asarray([[0.0], [0.0]], dtype=np.float64),
            "coh_ps": np.asarray([[1.0], [1.0]], dtype=np.float64),
            "ph_patch": np.ones((n_ps, n_ifg - 1), dtype=np.complex64),
            "ph_res": np.zeros((n_ps, n_ifg - 1), dtype=np.float32),
        },
    )
    write_mat(dataset_root / "rc2.mat", {"ph_rc": np.ones((n_ps, n_ifg), dtype=np.complex64)})
    write_mat(dataset_root / "bp2.mat", {"bperp_mat": np.zeros((n_ps, n_ifg - 1), dtype=np.float32)})
    write_mat(
        dataset_root / "uw_grid.mat",
        {
            "ph": np.ones((2, 2), dtype=np.complex64),
            "ph_in": np.ones((2, 2), dtype=np.complex64),
            "nzix": np.asarray([[True, True], [False, False]], dtype=bool),
            "grid_ij": np.asarray([[1.0, 1.0], [1.0, 2.0]], dtype=np.float64),
            "n_ps": np.asarray(2.0, dtype=np.float64),
        },
    )
    write_mat(
        dataset_root / "uw_interp.mat",
        {
            "edgs": np.asarray([[1.0, 1.0, 2.0]], dtype=np.float64),
            "rowix": np.zeros((1, 2), dtype=np.float64),
            "colix": np.asarray([[1.0], [1.0]], dtype=np.float64),
            "Z": np.asarray([[1, 2], [1, 2]], dtype=np.int64),
            "n_edge": np.asarray(1.0, dtype=np.float64),
        },
    )
    write_mat(
        dataset_root / "parms.mat",
        {
            "small_baseline_flag": np.asarray("n"),
            "unwrap_patch_phase": np.asarray("n"),
            "unwrap_method": np.asarray("3D"),
            "unwrap_la_error_flag": np.asarray("y"),
            "unwrap_spatial_cost_func_flag": np.asarray("n"),
            "unwrap_time_win": np.asarray(36.0, dtype=np.float64),
            "lambda": np.asarray(0.0555, dtype=np.float64),
            "max_topo_err": np.asarray(15.0, dtype=np.float64),
        },
    )


def test_stage6_debug_records_each_ifg_before_native_unwrap(monkeypatch, tmp_path: Path) -> None:
    _write_minimal_stage6_inputs(tmp_path)
    debug_path = tmp_path / "stage6_debug.json"
    monkeypatch.setenv("PYSTAMPS_STAGE6_DEBUG_JSON", str(debug_path))

    snapshots: list[dict] = []
    pre_unwrap_snapshots: list[dict] = []

    def capture_debug(path: Path | None, payload: dict | None) -> None:
        del path
        if payload is not None:
            snapshots.append(copy.deepcopy(payload))

    monkeypatch.setattr(ported, "_write_stage6_debug", capture_debug)
    monkeypatch.setattr(
        ported,
        "_compute_active_single_master_uw_space_time",
        lambda *args, **kwargs: (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((1, 2), dtype=np.complex64),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        ported,
        "run_stage6_prepare_cost_offsets_kernel",
        lambda rowcost, colcost, *args, **kwargs: (rowcost.copy(), colcost.copy()),
    )
    monkeypatch.setattr(
        ported,
        "run_stage6_select_ifgw_kernel",
        lambda *args, **kwargs: np.ones((2, 2), dtype=np.complex64),
    )

    def fake_unwrap_grid(*args, **kwargs):
        pre_unwrap_snapshots.append(copy.deepcopy(snapshots[-1]))
        return {"ifguw": np.ones((2, 2), dtype=np.float32), "msd": 0.0}

    monkeypatch.setattr(ported, "run_stage6_unwrap_grid_kernel", fake_unwrap_grid)
    monkeypatch.setattr(
        ported,
        "run_stage6_extract_grid_values_kernel",
        lambda *args, **kwargs: np.asarray([1.0, 1.0], dtype=np.float32),
    )
    monkeypatch.setattr(ported, "run_stage6_ps_grid_indices_kernel", lambda *args, **kwargs: np.asarray([0, 1]))
    monkeypatch.setattr(ported, "run_stage6_reconstruct_ps_phase_kernel", lambda ph_uw_some, *args, **kwargs: ph_uw_some)

    ported.stage6_unwrap(tmp_path, backend="native", enable_mat_cache=False)

    assert [payload["ifg_in_progress"] for payload in pre_unwrap_snapshots] == [1, 2]
    assert [payload["current_ifg_index"] for payload in pre_unwrap_snapshots] == [0, 1]
    assert [payload["ifg_completed"] for payload in pre_unwrap_snapshots] == [0, 1]
    assert {payload["unwrap_loop_phase"] for payload in pre_unwrap_snapshots} == {"unwrap_native"}
