from __future__ import annotations

from pathlib import Path

import numpy as np

from pystamps.pipeline import ported
from pystamps.pipeline.ported import _smooth_scla_neighbor_envelope


def test_smooth_scla_neighbor_envelope_clamps_to_neighbor_bounds() -> None:
    k_ps_uw = np.asarray([10.0, 1.0, 2.0], dtype=np.float32)
    c_ps_uw = np.asarray([5.0, 0.0, 2.0], dtype=np.float32)
    edges = np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int64)

    k_out, c_out = _smooth_scla_neighbor_envelope(k_ps_uw, c_ps_uw, edges)

    np.testing.assert_allclose(k_out, np.asarray([2.0, 2.0, 2.0], dtype=np.float32), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(c_out, np.asarray([2.0, 2.0, 2.0], dtype=np.float32), atol=0.0, rtol=0.0)


def test_stage8_mean_velocity_payload_uses_degree_to_radian_weights(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    ps2 = {
        "n_ps": np.asarray(2.0),
        "n_ifg": np.asarray(3.0),
        "master_ix": np.asarray(2.0),
        "day": np.asarray([1.0, 3.0, 7.0], dtype=np.float64),
        "xy": np.asarray([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float64),
    }
    captured: dict[str, np.ndarray] = {}

    def fake_read_mat_cached(path: Path, cache: dict[Path, dict[str, np.ndarray]], enabled: bool = True) -> dict[str, np.ndarray]:
        if path.name == "phuw2.mat":
            return {"ph_uw": np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32)}
        if path.name == "scla2.mat":
            return {"ph_scla": np.zeros((2, 3), dtype=np.float32)}
        if path.name == "ifgstd2.mat":
            return {"ifg_std": np.asarray([18.0, 90.0, 36.0], dtype=np.float64)}
        raise AssertionError(f"unexpected cached read: {path}")

    def fake_weighted_lstsq(
        design: np.ndarray,
        values: np.ndarray,
        covariance: np.ndarray | None = None,
        backend: str = "auto",
    ) -> np.ndarray:
        captured["design"] = design
        captured["values"] = values
        captured["cov"] = np.asarray(covariance)
        captured["backend"] = np.asarray([backend])
        return np.zeros((2, 2), dtype=np.float64)

    monkeypatch.setattr(ported, "_read_mat_cached", fake_read_mat_cached)
    monkeypatch.setattr(ported, "_deramp_unwrapped_phase", lambda ps, ph: (ph, np.zeros_like(ph)))
    monkeypatch.setattr(ported, "_select_reference_ps", lambda ps, parms: np.asarray([], dtype=np.int64))
    monkeypatch.setattr(ported, "run_stage8_weighted_lstsq_kernel", fake_weighted_lstsq)

    payload = ported._stage8_mean_velocity_payload(
        dataset_root,
        ps2,
        {},
        {},
        enable_mat_cache=True,
    )

    expected_cov = np.diag((np.asarray([18.0, 36.0], dtype=np.float64) * np.pi / 180.0) ** 2)
    np.testing.assert_allclose(captured["cov"], expected_cov, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(captured["design"], np.asarray([[1.0, -2.0], [1.0, 4.0]], dtype=np.float64))
    assert captured["backend"][0] == "python"
    assert payload["m"].shape == (2, 2)


def test_stage8_filter_scn_writes_legacy_scn2_without_rerunning_unwrap(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for filename in ("ps2.mat", "phuw2.mat", "scla2.mat", "parms.mat"):
        (dataset_root / filename).touch()

    written: dict[str, dict[str, np.ndarray]] = {}
    captured: dict[str, object] = {}

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
            return {
                "ph_uw": np.arange(9, dtype=np.float32).reshape(3, 3),
            }
        if path.name == "scla2.mat":
            return {
                "ph_scla": np.ones((3, 3), dtype=np.float32),
                "C_ps_uw": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                "ph_ramp": np.full((3, 3), 0.5, dtype=np.float64),
            }
        if path.name == "parms.mat":
            return {
                "small_baseline_flag": "n",
                "drop_ifg_index": np.asarray([1], dtype=np.int64),
                "scn_deramp_ifg": np.asarray([3], dtype=np.int64),
                "scn_time_win": np.asarray(360.0),
                "scn_wavelength": np.asarray(100.0),
            }
        raise AssertionError(f"unexpected cached read: {path}")

    def fake_write_mat(path: Path, payload: dict[str, np.ndarray]) -> None:
        written[path.name] = payload

    monkeypatch.setattr(ported, "_resolve_file", fake_resolve_file)
    monkeypatch.setattr(ported, "_read_mat_cached", fake_read_mat_cached)
    def fake_build_scn_payload(ph_uw: np.ndarray, xy: np.ndarray, day: np.ndarray, **kwargs: object):
        captured.update({"ph_uw": ph_uw, "xy": xy, "day": day, **kwargs})
        return {
            "ph_scn_slave": np.zeros((3, 3), dtype=np.float64),
            "ph_hpt": np.zeros((3, 2), dtype=np.float32),
            "ph_ramp": np.zeros((3, 1), dtype=np.float64),
        }

    monkeypatch.setattr(ported, "build_scn_payload", fake_build_scn_payload)
    monkeypatch.setattr(ported, "write_mat", fake_write_mat)
    monkeypatch.setattr(ported, "_cache_mat_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ported,
        "stage6_unwrap",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Stage 8 must not rerun Stage 6")),
    )

    result = ported.stage8_filter_scn(dataset_root, backend="python", enable_mat_cache=True, io_workers=0)

    assert result == "Stage 8 estimated spatially correlated noise for 3 PS"
    assert set(written) == {"scn2.mat"}
    np.testing.assert_array_equal(captured["unwrap_indices"], np.asarray([1, 2]))
    np.testing.assert_array_equal(captured["deramp_indices"], np.asarray([2]))
    assert captured["master_index"] == 1
    assert captured["time_window"] == 360.0
    assert captured["wavelength"] == 100.0
