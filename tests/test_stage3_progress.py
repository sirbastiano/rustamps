from __future__ import annotations

from pathlib import Path

import numpy as np

from pystamps.io.mat import write_mat
from pystamps.pipeline import ported


def test_stage3_reestimate_writes_running_progress_debug(tmp_path: Path, monkeypatch) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    coh_bins = np.arange(0.005, 1.0, 0.01, dtype=np.float64)

    write_mat(
        patch_dir / "parms.mat",
        {
            "select_method": ported._matlab_char_row("DENSITY"),
            "density_rand": np.asarray(1.0, dtype=np.float64),
            "small_baseline_flag": ported._matlab_char_row("n"),
            "gamma_stdev_reject": np.asarray(0.0, dtype=np.float64),
            "clap_win": np.asarray(1.0, dtype=np.float64),
            "clap_alpha": np.asarray(1.0, dtype=np.float64),
            "clap_beta": np.asarray(0.3, dtype=np.float64),
            "slc_osf": np.asarray(1.0, dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ps1.mat",
        {
            "n_ps": np.asarray(1.0, dtype=np.float64),
            "master_ix": np.asarray(1.0, dtype=np.float64),
            "bperp": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "xy": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        },
    )
    write_mat(patch_dir / "da1.mat", {"D_A": np.asarray([0.2], dtype=np.float64)})
    write_mat(
        patch_dir / "pm1.mat",
        {
            "coh_ps": np.asarray([0.6], dtype=np.float64),
            "coh_bins": coh_bins,
            "Nr": np.ones(coh_bins.size, dtype=np.float64),
            "ph_patch": np.asarray([[0.5 + 0.0j, 0.25 + 0.0j]], dtype=np.complex64),
            "ph_res": np.zeros((1, 2), dtype=np.float32),
            "K_ps": np.asarray([0.1], dtype=np.float64),
            "C_ps": np.asarray([0.0], dtype=np.float64),
            "ph_grid": np.ones((2, 2, 2), dtype=np.complex64),
            "grid_ij": np.asarray([[1.0, 1.0]], dtype=np.float64),
            "n_trial_wraps": np.asarray(1.0, dtype=np.float64),
            "low_pass": np.ones((1, 1), dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ph1.mat",
        {"ph": np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)},
    )
    write_mat(patch_dir / "bp1.mat", {"bperp_mat": np.asarray([[10.0, 20.0]], dtype=np.float64)})

    monkeypatch.setattr(ported, "_ifg_index_for_selection", lambda ps, parms, **kwargs: np.asarray([1.0, 2.0]))
    real_as_ps_dim = ported._as_ps_dim
    real_as_ps_ifg_complex = ported._as_ps_ifg_complex
    real_as_ps_matrix = ported._as_ps_matrix

    def fake_as_ps_dim(values, n_ps, n_dim, name):
        if name == "ps1.xy":
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
        if name == "pm1.grid_ij":
            return np.asarray([[1.0, 1.0]], dtype=np.float64)
        return real_as_ps_dim(values, n_ps, n_dim, name)

    def fake_as_ps_ifg_complex(values, n_ps, name):
        if name == "pm1.ph_patch":
            return np.asarray([[0.5 + 0.0j, 0.25 + 0.0j]], dtype=np.complex64)
        if name == "ph1.ph":
            return np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)
        return real_as_ps_ifg_complex(values, n_ps, name)

    def fake_as_ps_matrix(values, n_ps, name):
        if name == "pm1.ph_res":
            return np.zeros((1, 2), dtype=np.float32)
        if name == "bp1.bperp_mat":
            return np.asarray([[10.0, 20.0]], dtype=np.float64)
        return real_as_ps_matrix(values, n_ps, name)

    monkeypatch.setattr(ported, "_as_ps_dim", fake_as_ps_dim)
    monkeypatch.setattr(ported, "_as_ps_ifg_complex", fake_as_ps_ifg_complex)
    monkeypatch.setattr(ported, "_as_ps_matrix", fake_as_ps_matrix)
    monkeypatch.setattr(
        ported,
        "run_stage3_coh_threshold_kernel",
        lambda *args, **kwargs: (np.zeros(1, dtype=np.float64), np.asarray([1.0, 0.0], dtype=np.float64)),
    )

    def fake_clap_stack(ph_stack, *, alpha, beta, low_pass, backend="auto", threads=0):
        del alpha, beta, low_pass, backend, threads
        out = np.zeros_like(ph_stack, dtype=np.complex128)
        out[0, 0, :] = np.asarray([0.5 + 0.0j, 0.25 + 0.0j], dtype=np.complex128)
        return out

    monkeypatch.setattr(ported, "run_stage3_clap_filt_patch_stack_kernel", fake_clap_stack)

    def fake_topofit(cpxphase, bperp, n_trial_wraps, *, backend="auto", threads=0, cpu_fallback=None):
        del cpxphase, bperp, n_trial_wraps, backend, threads, cpu_fallback
        return (
            np.asarray([0.1], dtype=np.float64),
            np.asarray([0.2], dtype=np.float64),
            np.asarray([0.9], dtype=np.float64),
            np.ones((1, 2), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "run_stage2_topofit_kernel", fake_topofit)
    debug_writes: list[dict] = []

    def capture_debug(patch: Path, payload: dict | None) -> None:
        assert patch == patch_dir
        assert payload is not None
        debug_writes.append(dict(payload))

    monkeypatch.setattr(ported, "_write_stage3_debug", capture_debug)

    assert ported.stage3_select_ps(patch_dir, backend="native") == "Stage 3 selected 1 PS"

    statuses = [payload["reestimate_status"] for payload in debug_writes]
    assert "running" in statuses
    assert statuses[-1] == "completed"
    running = next(payload for payload in debug_writes if payload["reestimate_status"] == "running")
    assert running["reestimate_progress"]["rows_completed"] == 0
    assert running["reestimate_progress"]["rows_total"] == 1
