from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pystamps.pipeline import ported
from pystamps.pipeline.ported import (
    _deramp_unwrapped_phase,
    _select_reference_ps,
    _stage7_mean_velocity_fit,
    _stage7_unwrap_ifg_sets,
    _weighted_affine_fit,
    _weighted_lstsq_shared_design,
    _weighted_slope_fit,
)


def test_weighted_lstsq_shared_design_solves_multi_rhs() -> None:
    G = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 2.0],
            [1.0, 5.0],
            [1.0, 9.0],
        ],
        dtype=np.float64,
    )
    coeffs_true = np.asarray(
        [
            [3.0, -2.0],
            [0.5, 1.25],
        ],
        dtype=np.float64,
    )
    Y = G @ coeffs_true
    cov = np.diag(np.asarray([1.0, 4.0, 9.0, 16.0], dtype=np.float64))

    coeffs = _weighted_lstsq_shared_design(G, Y, cov=cov)

    np.testing.assert_allclose(coeffs, coeffs_true, atol=1e-10, rtol=0.0)


def test_weighted_lstsq_shared_design_can_route_to_native_wrapper(monkeypatch: object) -> None:
    G = np.asarray([[1.0, 0.0], [1.0, 2.0], [1.0, 5.0]], dtype=np.float64)
    Y = np.asarray([[2.0], [4.0], [7.0]], dtype=np.float64)
    cov = np.diag(np.asarray([1.0, 4.0, 9.0], dtype=np.float64))
    expected = np.asarray([[1.0], [2.0]], dtype=np.float64)
    captured: dict[str, object] = {}

    def fake_weighted_lstsq(
        design: np.ndarray,
        values: np.ndarray,
        covariance: np.ndarray | None = None,
        backend: str = "auto",
        threads: int = 0,
    ) -> np.ndarray:
        captured["design"] = np.asarray(design)
        captured["values"] = np.asarray(values)
        captured["covariance"] = None if covariance is None else np.asarray(covariance)
        captured["backend"] = backend
        captured["threads"] = threads
        return expected

    monkeypatch.setattr(ported, "run_stage8_weighted_lstsq_kernel", fake_weighted_lstsq)

    observed = _weighted_lstsq_shared_design(G, Y, cov=cov, backend="native", threads=3)

    np.testing.assert_allclose(observed, expected)
    np.testing.assert_allclose(captured["design"], G)
    np.testing.assert_allclose(captured["values"], Y)
    np.testing.assert_allclose(captured["covariance"], cov)
    assert captured["backend"] == "native"
    assert captured["threads"] == 3


def test_weighted_affine_fit_can_route_to_native_wrapper(monkeypatch: object) -> None:
    time_diff = np.asarray([-2.0, 0.0, 3.0], dtype=np.float64)
    y = np.asarray([[1.0, 2.0, 5.0], [3.0, 4.0, 9.0]], dtype=np.float64)
    weight = np.asarray([1.0, 4.0, 2.0], dtype=np.float64)
    expected = (np.asarray([1.0, 2.0], dtype=np.float64), np.asarray([3.0, 4.0], dtype=np.float64))
    captured: dict[str, object] = {}

    def fake_affine(
        time_arg: np.ndarray,
        y_arg: np.ndarray,
        weight_arg: np.ndarray,
        backend: str = "auto",
        threads: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        captured["time_diff"] = np.asarray(time_arg)
        captured["y"] = np.asarray(y_arg)
        captured["weight"] = np.asarray(weight_arg)
        captured["backend"] = backend
        captured["threads"] = threads
        return expected

    monkeypatch.setattr(ported, "run_weighted_affine_fit_kernel", fake_affine)

    observed = _weighted_affine_fit(time_diff, y, weight, backend="native", threads=7)

    np.testing.assert_allclose(observed[0], expected[0])
    np.testing.assert_allclose(observed[1], expected[1])
    np.testing.assert_allclose(captured["time_diff"], time_diff)
    np.testing.assert_allclose(captured["y"], y)
    np.testing.assert_allclose(captured["weight"], weight)
    assert captured["backend"] == "native"
    assert captured["threads"] == 7


def test_weighted_slope_fit_can_route_to_native_wrapper(monkeypatch: object) -> None:
    x = np.asarray([-1.0, 2.0, 4.0], dtype=np.float64)
    y = np.asarray([[1.0, 3.0, 5.0], [-2.0, 4.0, 8.0]], dtype=np.float64)
    weight = np.asarray([1.0, np.inf, 2.0], dtype=np.float64)
    expected = np.asarray([1.5, 2.5], dtype=np.float64)
    captured: dict[str, object] = {}

    def fake_slope(
        x_arg: np.ndarray,
        y_arg: np.ndarray,
        weight_arg: np.ndarray,
        backend: str = "auto",
        threads: int = 0,
    ) -> np.ndarray:
        captured["x"] = np.asarray(x_arg)
        captured["y"] = np.asarray(y_arg)
        captured["weight"] = np.asarray(weight_arg)
        captured["backend"] = backend
        captured["threads"] = threads
        return expected

    monkeypatch.setattr(ported, "run_weighted_slope_fit_kernel", fake_slope)

    observed = _weighted_slope_fit(x, y, weight, backend="native", threads=8)

    np.testing.assert_allclose(observed, expected)
    np.testing.assert_allclose(captured["x"], x)
    np.testing.assert_allclose(captured["y"], y)
    np.testing.assert_allclose(captured["weight"], weight)
    assert captured["backend"] == "native"
    assert captured["threads"] == 8


def test_deramp_unwrapped_phase_removes_linear_plane() -> None:
    ps = {
        "n_ps": np.asarray(4.0),
        "xy": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 1000.0, 0.0],
                [3.0, 0.0, 1000.0],
                [4.0, 1000.0, 1000.0],
            ],
            dtype=np.float64,
        ),
    }
    x_km = ps["xy"][:, 1] / 1000.0
    y_km = ps["xy"][:, 2] / 1000.0
    ramp = np.column_stack(
        (
            1.5 * x_km + 0.75 * y_km + 2.0,
            -0.5 * x_km + 1.25 * y_km - 1.0,
        )
    )
    ph = ramp.copy()

    ph_out, ph_ramp = _deramp_unwrapped_phase(ps, ph)

    np.testing.assert_allclose(ph_ramp, ramp, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(ph_out, np.zeros_like(ph), atol=1e-10, rtol=0.0)


def test_deramp_unwrapped_phase_can_route_to_native_wrapper(monkeypatch: object) -> None:
    xy = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1000.0, 0.0],
            [3.0, 0.0, 1000.0],
        ],
        dtype=np.float64,
    )
    ps = {"n_ps": np.asarray(float(xy.shape[0])), "xy": xy}
    ph = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    expected = (np.full_like(ph, 7.0), np.full_like(ph, 8.0))
    captured: dict[str, object] = {}

    def fake_deramp(
        xy_arg: np.ndarray,
        ph_arg: np.ndarray,
        backend: str = "auto",
        threads: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        captured["xy"] = np.asarray(xy_arg)
        captured["ph"] = np.asarray(ph_arg)
        captured["backend"] = backend
        captured["threads"] = threads
        return expected

    monkeypatch.setattr(ported, "run_stage7_deramp_unwrapped_phase_kernel", fake_deramp)

    observed = _deramp_unwrapped_phase(ps, ph, backend="native", threads=4)

    np.testing.assert_allclose(observed[0], expected[0])
    np.testing.assert_allclose(observed[1], expected[1])
    np.testing.assert_allclose(captured["xy"], xy)
    np.testing.assert_allclose(captured["ph"], ph)
    assert captured["backend"] == "native"
    assert captured["threads"] == 4


def test_select_reference_ps_uses_local_coordinate_units_for_radius() -> None:
    ps = {
        "n_ps": np.asarray(2.0),
        "lonlat": np.asarray(
            [
                [0.0, 0.0],
                [0.0009, 0.0],
            ],
            dtype=np.float64,
        ),
        "ll0": np.asarray([0.0, 0.0], dtype=np.float64),
    }
    parms_raw = {
        "ref_lon": np.asarray([-1.0, 1.0], dtype=np.float64),
        "ref_lat": np.asarray([-1.0, 1.0], dtype=np.float64),
        "ref_centre_lonlat": np.asarray([0.0, 0.0], dtype=np.float64),
        "ref_radius": np.asarray(120.0, dtype=np.float64),
    }

    ref_ix = _select_reference_ps(ps, parms_raw)

    np.testing.assert_array_equal(ref_ix, np.asarray([0, 1], dtype=np.int64))


def test_stage7_unwrap_ifg_sets_keeps_master_for_sequential_diffs() -> None:
    unwrap_ifg, solve_ifg = _stage7_unwrap_ifg_sets(n_ifg=5, master_ix=3, drop_set={5})

    np.testing.assert_array_equal(unwrap_ifg, np.asarray([1, 2, 3, 4], dtype=np.int64))
    np.testing.assert_array_equal(solve_ifg, np.asarray([1, 2, 4], dtype=np.int64))


def test_stage7_mean_velocity_fit_uses_full_stack_weights() -> None:
    ph_mean_v = np.asarray(
        [
            [3.0, 0.0, -1.0, 1.0],
            [-2.0, 0.0, 4.0, 7.0],
        ],
        dtype=np.float64,
    )
    day = np.asarray([8.0, 10.0, 13.0, 17.0], dtype=np.float64)
    ifg_std = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)

    m = _stage7_mean_velocity_fit(ph_mean_v, day, master_ix=2, ifg_std=ifg_std)

    time_diff = day - day[1]
    weights = 1.0 / ((ifg_std * np.pi / 180.0) ** 2)
    s0 = float(np.sum(weights))
    s1 = float(np.sum(weights * time_diff))
    s2 = float(np.sum(weights * time_diff * time_diff))
    det = s0 * s2 - s1 * s1
    wy0 = np.sum(ph_mean_v * weights[None, :], axis=1)
    wy1 = np.sum(ph_mean_v * (weights * time_diff)[None, :], axis=1)
    expected = np.vstack(
        (
            ((wy0 * s2 - wy1 * s1) / det).astype(np.float32),
            ((wy1 * s0 - wy0 * s1) / det).astype(np.float32),
        )
    )

    np.testing.assert_allclose(m, expected, atol=1e-10, rtol=0.0)


def test_stage7_mean_velocity_fit_can_route_to_native_wrapper(monkeypatch: object) -> None:
    ph_mean_v = np.asarray([[3.0, 0.0, -1.0], [-2.0, 0.0, 4.0]], dtype=np.float64)
    day = np.asarray([8.0, 10.0, 13.0], dtype=np.float64)
    ifg_std = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
    expected = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    captured: dict[str, object] = {}

    def fake_mean_velocity(
        ph_arg: np.ndarray,
        day_arg: np.ndarray,
        master_ix: int,
        ifg_std_arg: np.ndarray,
        backend: str = "auto",
        threads: int = 0,
    ) -> np.ndarray:
        captured["ph"] = np.asarray(ph_arg)
        captured["day"] = np.asarray(day_arg)
        captured["master_ix"] = master_ix
        captured["ifg_std"] = np.asarray(ifg_std_arg)
        captured["backend"] = backend
        captured["threads"] = threads
        return expected

    monkeypatch.setattr(ported, "run_stage7_mean_velocity_fit_kernel", fake_mean_velocity)

    observed = _stage7_mean_velocity_fit(ph_mean_v, day, master_ix=2, ifg_std=ifg_std, backend="native", threads=5)

    np.testing.assert_allclose(observed, expected)
    np.testing.assert_allclose(captured["ph"], ph_mean_v)
    np.testing.assert_allclose(captured["day"], day)
    np.testing.assert_allclose(captured["ifg_std"], ifg_std)
    assert captured["master_ix"] == 2
    assert captured["backend"] == "native"
    assert captured["threads"] == 5


def test_stage7_calc_scla_deramps_before_centering(monkeypatch: object, tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for filename in ("phuw2.mat", "ps2.mat", "bp2.mat", "ifgstd2.mat", "parms.mat"):
        (dataset_root / filename).touch()

    captured: dict[str, np.ndarray] = {}

    def fake_resolve_file(root: Path, name: str) -> Path | None:
        if root == dataset_root and name == "parms.mat":
            return dataset_root / name
        return None

    def fake_read_mat_cached(path: Path, cache: dict[Path, dict[str, np.ndarray]], enabled: bool = True) -> dict[str, np.ndarray]:
        if path.name == "ps2.mat":
            return {
                "n_ps": np.asarray(2.0),
                "master_ix": np.asarray(1.0),
                "day": np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
                "bperp": np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
                "xy": np.asarray([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float64),
            }
        if path.name == "phuw2.mat":
            return {
                "ph_uw": np.asarray(
                    [
                        [10.0, 12.0, 14.0],
                        [4.0, 6.0, 8.0],
                    ],
                    dtype=np.float32,
                )
            }
        if path.name == "bp2.mat":
            return {
                "bperp_mat": np.asarray(
                    [
                        [0.0, 1.0],
                        [0.0, 2.0],
                    ],
                    dtype=np.float32,
                )
            }
        if path.name == "ifgstd2.mat":
            return {"ifg_std": np.asarray([1.0, 1.0, 1.0], dtype=np.float64)}
        if path.name == "parms.mat":
            return {
                "small_baseline_flag": "n",
                "drop_ifg_index": np.asarray([], dtype=np.int64),
                "scla_drop_index": np.asarray([], dtype=np.int64),
                "scla_deramp": "y",
            }
        raise AssertionError(f"unexpected cached read: {path}")

    def fake_select_reference_ps(ps: dict[str, np.ndarray], parms_raw: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray([0], dtype=np.int64)

    def fake_deramp_unwrapped_phase(ps: dict[str, np.ndarray], ph_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        captured["deramp_input"] = np.asarray(ph_all)
        ph_all_arr = np.asarray(ph_all, dtype=np.float64)
        ph_deramped = ph_all_arr - np.asarray(
            [
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )
        return ph_deramped, np.zeros_like(ph_all_arr, dtype=np.float64)

    def fake_run_stage7_scla_kernel(
        *,
        ph_proc: np.ndarray,
        ph_mean_v: np.ndarray,
        bperp_mat: np.ndarray,
        unwrap_ix: np.ndarray,
        solve_ix: np.ndarray,
        day: np.ndarray,
        master_ix: int,
        ifg_std: np.ndarray,
        backend: str,
        chunk_ps: int,
    ) -> dict[str, np.ndarray]:
        captured["ph_proc"] = np.asarray(ph_proc)
        captured["ph_mean_v"] = np.asarray(ph_mean_v)
        n_ps, n_ifg = np.asarray(ph_proc).shape
        return {
            "K_ps_uw": np.zeros(n_ps, dtype=np.float64),
            "C_ps_uw": np.zeros(n_ps, dtype=np.float32),
            "ph_scla": np.zeros((n_ps, n_ifg), dtype=np.float32),
            "ph_ramp": np.zeros((n_ps, n_ifg), dtype=np.float64),
            "ifg_vcm": np.eye(n_ifg, dtype=np.float64),
            "mean_v": np.zeros(n_ps, dtype=np.float32),
            "m": np.zeros((2, n_ps), dtype=np.float32),
        }

    monkeypatch.setattr(ported, "_resolve_file", fake_resolve_file)
    monkeypatch.setattr(ported, "_read_mat_cached", fake_read_mat_cached)
    monkeypatch.setattr(ported, "_select_reference_ps", fake_select_reference_ps)
    monkeypatch.setattr(ported, "_deramp_unwrapped_phase", fake_deramp_unwrapped_phase)
    monkeypatch.setattr(ported, "_resolve_scla_smooth_edges", lambda *args, **kwargs: np.empty((0, 2), dtype=np.int64))
    monkeypatch.setattr(ported, "run_stage7_scla_kernel", fake_run_stage7_scla_kernel)
    monkeypatch.setattr(ported, "write_mat", lambda *args, **kwargs: None)
    monkeypatch.setattr(ported, "_cache_mat_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(ported, "stage6_unwrap", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stage6_unwrap should not run")))

    ported.stage7_calc_scla(dataset_root, backend="python", chunk_ps=0, enable_mat_cache=True, io_workers=0)

    expected_raw = np.asarray(
        [
            [10.0, 12.0, 14.0],
            [4.0, 6.0, 8.0],
        ],
        dtype=np.float64,
    )
    expected_centered = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-5.5, -5.5, -5.5],
        ],
        dtype=np.float64,
    )
    expected_mean_v = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-6.0, -6.0, -6.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(captured["deramp_input"], expected_raw, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(captured["ph_proc"], expected_centered, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(captured["ph_mean_v"], expected_mean_v, atol=0.0, rtol=0.0)


def test_stage7_calc_scla_rejects_small_baseline_before_writing(monkeypatch: object, tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for filename in ("phuw2.mat", "ps2.mat", "bp2.mat", "ifgstd2.mat", "parms.mat"):
        (dataset_root / filename).touch()

    captured: dict[str, np.ndarray] = {}
    written: dict[str, dict[str, np.ndarray]] = {}

    def fake_resolve_file(root: Path, name: str) -> Path | None:
        if root == dataset_root and name == "parms.mat":
            return dataset_root / name
        return None

    def fake_read_mat_cached(path: Path, cache: dict[Path, dict[str, np.ndarray]], enabled: bool = True) -> dict[str, np.ndarray]:
        if path.name == "ps2.mat":
            return {
                "n_ps": np.asarray(2.0),
                "master_ix": np.asarray(2.0),
                "day": np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
            }
        if path.name == "phuw2.mat":
            return {
                "ph_uw": np.asarray(
                    [
                        [0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                    ],
                    dtype=np.float32,
                )
            }
        if path.name == "bp2.mat":
            return {
                "bperp_mat": np.asarray(
                    [
                        [11.0, 12.0, 13.0],
                        [21.0, 22.0, 23.0],
                    ],
                    dtype=np.float32,
                )
            }
        if path.name == "ifgstd2.mat":
            return {"ifg_std": np.asarray([1.0, 2.0, 3.0], dtype=np.float64)}
        if path.name == "parms.mat":
            return {
                "small_baseline_flag": "y",
                "drop_ifg_index": np.asarray([], dtype=np.int64),
                "scla_drop_index": np.asarray([], dtype=np.int64),
                "scla_deramp": "n",
            }
        raise AssertionError(f"unexpected cached read: {path}")

    def fake_select_reference_ps(ps: dict[str, np.ndarray], parms_raw: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray([], dtype=np.int64)

    def fake_run_stage7_scla_kernel(
        *,
        ph_proc: np.ndarray,
        ph_mean_v: np.ndarray,
        bperp_mat: np.ndarray,
        unwrap_ix: np.ndarray,
        solve_ix: np.ndarray,
        day: np.ndarray,
        master_ix: int,
        ifg_std: np.ndarray,
        backend: str,
        chunk_ps: int,
    ) -> dict[str, np.ndarray]:
        captured["ph_proc"] = np.asarray(ph_proc)
        captured["ph_mean_v"] = np.asarray(ph_mean_v)
        captured["bperp_mat"] = np.asarray(bperp_mat)
        captured["unwrap_ix"] = np.asarray(unwrap_ix)
        captured["solve_ix"] = np.asarray(solve_ix)
        captured["day"] = np.asarray(day)
        captured["ifg_std"] = np.asarray(ifg_std)
        captured["master_ix"] = np.asarray(master_ix)
        captured["backend"] = np.asarray(backend)
        captured["chunk_ps"] = np.asarray(chunk_ps)
        n_ps, n_ifg = np.asarray(ph_proc).shape
        return {
            "K_ps_uw": np.full(n_ps, 1.0, dtype=np.float64),
            "C_ps_uw": np.full(n_ps, 2.0, dtype=np.float32),
            "ph_scla": np.zeros((n_ps, n_ifg), dtype=np.float32),
            "ph_ramp": np.zeros((n_ps, n_ifg), dtype=np.float64),
            "ifg_vcm": np.eye(n_ifg, dtype=np.float64),
            "mean_v": np.full(n_ps, 3.0, dtype=np.float32),
            "m": np.zeros((2, n_ps), dtype=np.float32),
        }

    def fake_write_mat(path: Path, payload: dict[str, np.ndarray]) -> None:
        written[path.name] = payload

    monkeypatch.setattr(ported, "_resolve_file", fake_resolve_file)
    monkeypatch.setattr(ported, "_read_mat_cached", fake_read_mat_cached)
    monkeypatch.setattr(ported, "_select_reference_ps", fake_select_reference_ps)
    monkeypatch.setattr(ported, "_resolve_scla_smooth_edges", lambda *args, **kwargs: np.empty((0, 2), dtype=np.int64))
    monkeypatch.setattr(ported, "run_stage7_scla_kernel", fake_run_stage7_scla_kernel)
    monkeypatch.setattr(ported, "write_mat", fake_write_mat)
    monkeypatch.setattr(ported, "_cache_mat_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(ported, "stage6_unwrap", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stage6_unwrap should not run")))

    with pytest.raises(ported.PortedStageError, match="legacy three-pass workflow"):
        ported.stage7_calc_scla(dataset_root, backend="native", chunk_ps=0, enable_mat_cache=True, io_workers=0)

    assert written == {}


def test_stage7_calc_scla_rejects_small_baseline_before_rebuilding_missing_bp2(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for filename in ("phuw2.mat", "ps2.mat", "ifgstd2.mat", "parms.mat"):
        (dataset_root / filename).touch()

    captured: dict[str, np.ndarray] = {}
    written: dict[str, dict[str, np.ndarray]] = {}

    def fake_resolve_file(root: Path, name: str) -> Path | None:
        if root == dataset_root and name == "parms.mat":
            return dataset_root / name
        return None

    def fake_read_mat_cached(path: Path, cache: dict[Path, dict[str, np.ndarray]], enabled: bool = True) -> dict[str, np.ndarray]:
        if path.name == "ps2.mat":
            return {
                "n_ps": np.asarray(2.0),
                "master_ix": np.asarray(2.0),
                "bperp": np.asarray([31.0, 32.0, 33.0], dtype=np.float64),
                "day": np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
            }
        if path.name == "phuw2.mat":
            return {
                "ph_uw": np.asarray(
                    [
                        [0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                    ],
                    dtype=np.float32,
                )
            }
        if path.name == "ifgstd2.mat":
            return {"ifg_std": np.asarray([1.0, 2.0, 3.0], dtype=np.float64)}
        if path.name == "parms.mat":
            return {
                "small_baseline_flag": "y",
                "drop_ifg_index": np.asarray([], dtype=np.int64),
                "scla_drop_index": np.asarray([], dtype=np.int64),
                "scla_deramp": "n",
            }
        raise AssertionError(f"unexpected cached read: {path}")

    def fake_select_reference_ps(ps: dict[str, np.ndarray], parms_raw: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray([], dtype=np.int64)

    def fake_run_stage7_scla_kernel(
        *,
        ph_proc: np.ndarray,
        ph_mean_v: np.ndarray,
        bperp_mat: np.ndarray,
        unwrap_ix: np.ndarray,
        solve_ix: np.ndarray,
        day: np.ndarray,
        master_ix: int,
        ifg_std: np.ndarray,
        backend: str,
        chunk_ps: int,
    ) -> dict[str, np.ndarray]:
        captured["bperp_mat"] = np.asarray(bperp_mat)
        captured["unwrap_ix"] = np.asarray(unwrap_ix)
        captured["solve_ix"] = np.asarray(solve_ix)
        captured["master_ix"] = np.asarray(master_ix)
        n_ps, n_ifg = np.asarray(ph_proc).shape
        return {
            "K_ps_uw": np.full(n_ps, 1.0, dtype=np.float64),
            "C_ps_uw": np.full(n_ps, 2.0, dtype=np.float32),
            "ph_scla": np.zeros((n_ps, n_ifg), dtype=np.float32),
            "ph_ramp": np.zeros((n_ps, n_ifg), dtype=np.float64),
            "ifg_vcm": np.eye(n_ifg, dtype=np.float64),
            "mean_v": np.full(n_ps, 3.0, dtype=np.float32),
            "m": np.zeros((2, n_ps), dtype=np.float32),
        }

    def fake_write_mat(path: Path, payload: dict[str, np.ndarray]) -> None:
        written[path.name] = payload

    monkeypatch.setattr(ported, "_resolve_file", fake_resolve_file)
    monkeypatch.setattr(ported, "_read_mat_cached", fake_read_mat_cached)
    monkeypatch.setattr(ported, "_select_reference_ps", fake_select_reference_ps)
    monkeypatch.setattr(ported, "_resolve_scla_smooth_edges", lambda *args, **kwargs: np.empty((0, 2), dtype=np.int64))
    monkeypatch.setattr(ported, "run_stage7_scla_kernel", fake_run_stage7_scla_kernel)
    monkeypatch.setattr(ported, "write_mat", fake_write_mat)
    monkeypatch.setattr(ported, "_cache_mat_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(ported, "stage6_unwrap", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stage6_unwrap should not run")))

    with pytest.raises(ported.PortedStageError, match="legacy three-pass workflow"):
        ported.stage7_calc_scla(dataset_root, backend="native", chunk_ps=0, enable_mat_cache=True, io_workers=0)

    assert written == {}


def test_stage7_calc_scla_uses_no_deramp_default_and_writes_smoothed_payload(
    monkeypatch: object, tmp_path: Path
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for filename in ("phuw2.mat", "ps2.mat", "bp2.mat", "ifgstd2.mat", "parms.mat"):
        (dataset_root / filename).touch()

    written: dict[str, dict[str, np.ndarray]] = {}

    def fake_resolve_file(root: Path, name: str) -> Path | None:
        if root == dataset_root and name == "parms.mat":
            return dataset_root / name
        return None

    def fake_read_mat_cached(path: Path, cache: dict[Path, dict[str, np.ndarray]], enabled: bool = True) -> dict[str, np.ndarray]:
        if path.name == "ps2.mat":
            return {
                "n_ps": np.asarray(3.0),
                "master_ix": np.asarray(2.0),
                "day": np.asarray([1.0, 3.0, 6.0], dtype=np.float64),
                "xy": np.asarray(
                    [
                        [1.0, 0.0, 0.0],
                        [2.0, 1.0, 0.0],
                        [3.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
            }
        if path.name == "phuw2.mat":
            return {"ph_uw": np.ones((3, 3), dtype=np.float32)}
        if path.name == "bp2.mat":
            return {
                "bperp_mat": np.asarray(
                    [
                        [10.0, 30.0],
                        [10.0, 30.0],
                        [10.0, 30.0],
                    ],
                    dtype=np.float32,
                )
            }
        if path.name == "ifgstd2.mat":
            return {"ifg_std": np.asarray([1.0, 1.0, 1.0], dtype=np.float64)}
        if path.name == "parms.mat":
            return {
                "small_baseline_flag": "n",
                "drop_ifg_index": np.asarray([], dtype=np.int64),
                "scla_drop_index": np.asarray([], dtype=np.int64),
            }
        raise AssertionError(f"unexpected cached read: {path}")

    def fake_run_stage7_scla_kernel(**kwargs: object) -> dict[str, np.ndarray]:
        return {
            "K_ps_uw": np.asarray([10.0, 1.0, 2.0], dtype=np.float64),
            "C_ps_uw": np.asarray([5.0, 0.0, 2.0], dtype=np.float32),
            "ph_scla": np.zeros((3, 3), dtype=np.float32),
            "ph_ramp": np.zeros((3, 3), dtype=np.float64),
            "ifg_vcm": np.eye(3, dtype=np.float64),
            "mean_v": np.zeros(3, dtype=np.float32),
            "m": np.zeros((2, 3), dtype=np.float32),
        }

    def fake_write_mat(path: Path, payload: dict[str, np.ndarray]) -> None:
        written[path.name] = payload

    monkeypatch.setattr(ported, "_resolve_file", fake_resolve_file)
    monkeypatch.setattr(ported, "_read_mat_cached", fake_read_mat_cached)
    monkeypatch.setattr(ported, "_select_reference_ps", lambda *args, **kwargs: np.asarray([], dtype=np.int64))
    monkeypatch.setattr(ported, "_resolve_scla_smooth_edges", lambda *args, **kwargs: np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int64))
    monkeypatch.setattr(ported, "run_stage7_scla_kernel", fake_run_stage7_scla_kernel)
    monkeypatch.setattr(
        ported,
        "run_stage7_deramp_unwrapped_phase_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy default must not deramp")),
    )
    monkeypatch.setattr(ported, "write_mat", fake_write_mat)
    monkeypatch.setattr(ported, "_cache_mat_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(ported, "stage6_unwrap", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stage6_unwrap should not run")))

    result = ported.stage7_calc_scla(dataset_root, backend="python", enable_mat_cache=True, io_workers=0)

    assert result == "Stage 7 estimated SCLA for 3 PS"
    assert set(written) == {"scla2.mat", "scla_smooth2.mat"}
    assert np.asarray(written["scla2.mat"]["ph_ramp"]).shape == (0, 0)
    smooth = written["scla_smooth2.mat"]
    np.testing.assert_allclose(smooth["K_ps_uw"], np.asarray([[2.0], [2.0], [2.0]], dtype=np.float32), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(smooth["C_ps_uw"], np.asarray([[2.0], [2.0], [2.0]], dtype=np.float32), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        smooth["ph_scla"],
        np.asarray(
            [
                [20.0, 0.0, 60.0],
                [20.0, 0.0, 60.0],
                [20.0, 0.0, 60.0],
            ],
            dtype=np.float32,
        ),
        atol=0.0,
        rtol=0.0,
    )
