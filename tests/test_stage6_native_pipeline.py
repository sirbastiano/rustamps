from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pystamps.pipeline.ported as ported
from pystamps.io.mat import read_mat, write_mat


def test_resolve_snaphu_from_optional_python_package(monkeypatch, tmp_path: Path) -> None:
    package_dir = tmp_path / "snaphu"
    package_dir.mkdir()
    executable = package_dir / "snaphu"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr(ported.shutil, "which", lambda *args: None)
    monkeypatch.setattr(
        ported.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=[str(package_dir)]),
    )

    assert ported._maybe_resolve_external_tool("snaphu") == str(executable.resolve())


def test_ps_ifg_matrix_restores_singleton_axis_squeezed_by_mat_reader() -> None:
    values = np.asarray([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)

    observed = ported._as_ps_ifg_complex(values, 2, "fixture")

    assert observed.shape == (2, 1)
    np.testing.assert_array_equal(observed[:, 0], values)


def test_stage6_auto_prefers_resolvable_snaphu_and_native_does_not(monkeypatch) -> None:
    monkeypatch.setattr(ported, "_maybe_resolve_external_tool", lambda *args, **kwargs: "/opt/bin/snaphu")

    assert ported._select_stage6_unwrap_solver("auto", None) == (True, "/opt/bin/snaphu")
    assert ported._select_stage6_unwrap_solver("native", None) == (False, None)
    assert ported._select_stage6_unwrap_solver("native", None, solver="auto") == (
        True,
        "/opt/bin/snaphu",
    )
    assert ported._select_stage6_unwrap_solver("auto", None, solver="native") == (False, None)

    monkeypatch.setattr(ported, "_maybe_resolve_external_tool", lambda *args, **kwargs: None)
    assert ported._select_stage6_unwrap_solver("auto", None) == (False, None)


@pytest.mark.parametrize("unsupported_flag", ["subtr_tropo", "unwrap_hold_good_values"])
def test_stage6_rejects_unsupported_phase_options(tmp_path: Path, unsupported_flag: str) -> None:
    write_mat(
        tmp_path / "ps2.mat",
        {
            "n_ps": np.asarray(2.0),
            "master_ix": np.asarray(1.0),
        },
    )
    write_mat(tmp_path / "ph2.mat", {"ph": np.ones((2, 2), dtype=np.complex64)})
    write_mat(tmp_path / "parms.mat", {unsupported_flag: np.asarray("y")})

    with pytest.raises(ported.PortedStageError, match=unsupported_flag):
        ported.stage6_unwrap(tmp_path, enable_mat_cache=False)


def test_stage6_unwrap_uses_native_grid_kernel_without_snaphu(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path
    n_ps = 2
    n_ifg = 3
    master_ix = 2
    captured: dict[str, np.ndarray | str] = {}

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
    monkeypatch.setattr(
        ported,
        "_compute_active_single_master_uw_space_time",
        lambda *args, **kwargs: captured.update({"space_time_backend": kwargs["backend"]})
        or (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((1, 2), dtype=np.complex64),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        ported,
        "run_stage6_unwrap_grid_kernel",
        lambda ifgw, rowcost, colcost, **kwargs: captured.update(
            {
                "ifgw": np.asarray(ifgw).copy(),
                "rowcost": np.asarray(rowcost).copy(),
                "colcost": np.asarray(colcost).copy(),
                "backend": kwargs["backend"],
            }
        )
        or {"ifguw": np.asarray([[1.0, 2.0], [9.0, 10.0]], dtype=np.float32), "msd": 7.0},
    )
    monkeypatch.setattr(
        ported,
        "_run_external_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("snaphu should not run")),
    )

    ported.stage6_unwrap(dataset_root, backend="native", enable_mat_cache=False, snaphu_path="/missing/snaphu")

    assert captured["backend"] == "native"
    assert captured["space_time_backend"] == "native"
    np.testing.assert_array_equal(captured["ifgw"], np.ones((2, 2), dtype=np.complex64))
    assert np.asarray(captured["rowcost"]).shape == (1, 8)
    assert np.asarray(captured["colcost"]).shape == (2, 4)
    uw_phaseuw = read_mat(dataset_root / "uw_phaseuw.mat")
    np.testing.assert_allclose(
        np.asarray(uw_phaseuw["ph_uw"], dtype=np.float32),
        np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
    )
    phuw2 = read_mat(dataset_root / "phuw2.mat")
    np.testing.assert_allclose(np.asarray(phuw2["ph_uw"], dtype=np.float32), np.zeros((n_ps, n_ifg), dtype=np.float32))


def test_compute_active_single_master_routes_native_smoothing(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_estimate(dph_space, day, bperp, n_trial_wraps, *, backend, **kwargs):
        captured["estimate_backend"] = backend
        return np.zeros((np.asarray(dph_space).shape[0],), dtype=np.float32)

    def fake_smooth(dph_space, day, time_win, *, backend, chunk_edges=32768, **kwargs):
        captured["smooth_backend"] = backend
        captured["smooth_shape"] = np.asarray(dph_space).shape
        captured["chunk_edges"] = int(chunk_edges)
        return np.ones((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32)

    monkeypatch.setattr(ported, "run_stage6_estimate_la_error_kernel", fake_estimate)
    monkeypatch.setattr(ported, "run_stage6_smooth_3d_full_single_master_kernel", fake_smooth)

    uw_ph = np.asarray([[1 + 0j, 1 + 0j], [1j, -1j]], dtype=np.complex64)
    edgs = np.asarray([[1.0, 1.0, 2.0]], dtype=np.float64)

    _G, _dph_space, dph_smooth, dph_noise, dph_space_uw = ported._compute_active_single_master_uw_space_time(
        uw_ph,
        edgs,
        day=np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
        master_ix=1,
        bperp=np.asarray([100.0, 200.0], dtype=np.float64),
        unwrap_ifg=np.asarray([2, 3], dtype=np.int64),
        time_win=36.0,
        n_trial_wraps=1.0,
        chunk_edges=3,
        backend="native",
    )

    assert captured["estimate_backend"] == "native"
    assert captured["smooth_backend"] == "native"
    assert captured["smooth_shape"] == (1, 2)
    assert captured["chunk_edges"] == 3
    np.testing.assert_array_equal(dph_smooth, np.ones((1, 2), dtype=np.float32))
    np.testing.assert_array_equal(dph_noise, np.zeros((1, 2), dtype=np.float32))
    np.testing.assert_array_equal(dph_space_uw, np.ones((1, 2), dtype=np.float32))
