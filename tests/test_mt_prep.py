from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import pystamps.prep.mt_prep as mt_prep
from pystamps.prep.mt_prep import prepare_snap_mt_prep_inputs


def _write_par(path: Path, width: int, length: int) -> None:
    path.write_text(
        "\n".join(
            [
                "title:\ttest",
                f"range_samples:\t{width}",
                f"azimuth_lines:\t{length}",
                "image_format:\tFCOMPLEX",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_complex(path: Path, values: np.ndarray) -> None:
    np.asarray(values, dtype=">c8").tofile(path)


def _write_float(path: Path, values: np.ndarray) -> None:
    np.asarray(values, dtype=">f4").tofile(path)


def _load_phase(path: Path, n_rows: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=">f4")
    blocks = raw.reshape(raw.size // (2 * n_rows), n_rows * 2)
    return (blocks[:, 0::2] + 1j * blocks[:, 1::2]).T.astype(np.complex64)


def _fixture_amplitudes() -> dict[str, np.ndarray]:
    base = np.full((3, 4), 10.0, dtype=np.float32)
    amps = {
        "20240101": base,
        "20240113": base * 2.0,
        "20240125": base * 3.0,
    }
    for values in amps.values():
        values[0, 0] = 10.0
        values[1, 2] = 4.0
        values[2, 3] = 0.0
    amps["20240113"][1, 2] = 5.0
    return amps


def _legacy_fixture_stats() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack(list(_fixture_amplitudes().values())).astype(np.float64)
    calibration = np.asarray([values[values > 0.001].mean() for values in stack])
    normalized = stack / calibration[:, None, None]
    normalized_sum = normalized.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        da = np.sqrt(np.maximum(stack.shape[0] * np.sum(normalized**2, axis=0) / normalized_sum**2 - 1.0, 0.0))
    selected = (~np.any(normalized <= 0.00005, axis=0)) & (da < 0.25)
    return da, normalized_sum, np.argwhere(selected)


def _write_fixture(root: Path) -> None:
    (root / "rslc").mkdir(parents=True)
    (root / "diff0").mkdir()
    (root / "geo").mkdir()

    width, length = 4, 3
    amps = _fixture_amplitudes()
    for date in amps:
        _write_complex(root / "rslc" / f"{date}.rslc", amps[date].astype(np.complex64))
        _write_par(root / "rslc" / f"{date}.rslc.par", width, length)

    diff_early = (100 + np.arange(width * length).reshape(length, width)).astype(np.float32)
    diff_late = (200 + np.arange(width * length).reshape(length, width)).astype(np.float32)
    _write_complex(root / "diff0" / "20240113_20240101.diff", diff_early + 1j * (diff_early + 1000))
    _write_complex(root / "diff0" / "20240113_20240125.diff", diff_late + 1j * (diff_late + 1000))

    rows, cols = np.indices((length, width))
    lon = 12.0 + cols / 10.0
    lat = 41.0 + rows / 10.0
    hgt = 70.0 + rows * width + cols
    _write_float(root / "geo" / "20240113.lon", lon)
    _write_float(root / "geo" / "20240113.lat", lat)
    _write_float(root / "geo" / "elevation_dem.rdc", hgt)


def _prepare_fixture(root: Path, *, backend: str) -> mt_prep.MtPrepSummary:
    return prepare_snap_mt_prep_inputs(
        root,
        master_date="20240113",
        amp_dispersion=0.25,
        range_patches=1,
        azimuth_patches=1,
        range_overlap=0,
        azimuth_overlap=0,
        force=True,
        backend=backend,
    )


def _assert_expected_patch(root: Path, summary: mt_prep.MtPrepSummary) -> None:
    patch = root / "PATCH_1"
    expected_da, expected_amp_sum, rows_cols = _legacy_fixture_stats()
    linear = rows_cols[:, 0] * 4 + rows_cols[:, 1]
    assert summary.patch_count == 1
    assert summary.candidate_count == 9
    assert (root / "patch.list").read_text(encoding="utf-8") == "PATCH_1\n"
    expected_ij = np.column_stack((np.arange(1, rows_cols.shape[0] + 1), rows_cols))
    np.testing.assert_array_equal(np.loadtxt(patch / "pscands.1.ij"), expected_ij)
    np.testing.assert_array_equal(
        np.fromfile(patch / "pscands.1.ij.int", dtype=">i4").reshape(-1, 2), rows_cols[:, ::-1]
    )
    np.testing.assert_allclose(np.loadtxt(patch / "pscands.1.da"), expected_da[tuple(rows_cols.T)])
    expected_ll = np.column_stack((12.0 + rows_cols[:, 1] / 10.0, 41.0 + rows_cols[:, 0] / 10.0))
    np.testing.assert_allclose(np.fromfile(patch / "pscands.1.ll", dtype=">f4").reshape(-1, 2), expected_ll)
    np.testing.assert_allclose(np.fromfile(patch / "pscands.1.hgt", dtype=">f4"), 70.0 + linear)
    np.testing.assert_allclose(
        np.fromfile(patch / "mean_amp.flt", dtype=np.float32).reshape(3, 4), expected_amp_sum
    )
    np.testing.assert_allclose(
        _load_phase(patch / "pscands.1.ph", rows_cols.shape[0]),
        np.column_stack((100 + linear + 1j * (1100 + linear), 200 + linear + 1j * (1200 + linear))),
    )


def _patch_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted((root / "PATCH_1").iterdir())
        if path.is_file()
    } | {"patch.list": (root / "patch.list").read_bytes()}


def test_prepare_snap_mt_prep_inputs_writes_patch_candidates(tmp_path: Path) -> None:
    root = tmp_path / "INSAR_20240113"
    _write_fixture(root)

    summary = _prepare_fixture(root, backend="python")

    _assert_expected_patch(root, summary)


def test_prepare_snap_mt_prep_inputs_auto_falls_back_without_native(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "INSAR_20240113"
    _write_fixture(root)
    monkeypatch.setattr(mt_prep, "_native_export", lambda: None)

    summary = _prepare_fixture(root, backend="auto")

    _assert_expected_patch(root, summary)


def test_prepare_snap_mt_prep_inputs_native_requires_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "INSAR_20240113"
    _write_fixture(root)
    monkeypatch.setattr(mt_prep, "_native_export", lambda: None)

    with pytest.raises(mt_prep.MtPrepError, match="Native mt_prep backend requested"):
        _prepare_fixture(root, backend="native")


def test_prepare_snap_mt_prep_inputs_native_matches_python(tmp_path: Path) -> None:
    native_fn = mt_prep._native_export()
    if native_fn is None:
        pytest.skip("native mt_prep export is not available")

    python_root = tmp_path / "python" / "INSAR_20240113"
    native_root = tmp_path / "native" / "INSAR_20240113"
    _write_fixture(python_root)
    shutil.copytree(python_root, native_root)

    python_summary = _prepare_fixture(python_root, backend="python")
    native_summary = _prepare_fixture(native_root, backend="native")

    assert native_summary.patch_count == python_summary.patch_count
    assert native_summary.candidate_count == python_summary.candidate_count
    assert native_summary.patch_rows == python_summary.patch_rows
    assert _patch_files(native_root) == _patch_files(python_root)


def test_prepare_snap_mt_prep_inputs_rejects_unknown_backend(tmp_path: Path) -> None:
    root = tmp_path / "INSAR_20240113"
    _write_fixture(root)

    with pytest.raises(mt_prep.MtPrepError, match="Unsupported mt_prep backend"):
        _prepare_fixture(root, backend="bogus")


def test_candidate_selection_rejects_any_near_zero_acquisition(tmp_path: Path) -> None:
    root = tmp_path / "INSAR_20240101"
    (root / "rslc").mkdir(parents=True)
    (root / "geo").mkdir()
    files: list[Path] = []
    for index in range(10):
        path = root / "rslc" / f"{index:08d}.rslc"
        _write_complex(path, np.asarray([[10.0, 0.0 if index == 0 else 10.0]], dtype=np.complex64))
        files.append(path)
    for name in ("20240101.lon", "20240101.lat", "elevation_dem.rdc"):
        _write_float(root / "geo" / name, np.ones((1, 2), dtype=np.float32))

    mask, da, normalized_sum = mt_prep._candidate_arrays(root, "20240101", files, (1, 2), 0.4)

    assert da[0, 1] == pytest.approx(1.0 / 3.0)
    assert da[0, 1] < 0.4
    np.testing.assert_array_equal(mask, np.asarray([[True, False]]))
    np.testing.assert_allclose(normalized_sum, np.asarray([[10.0, 9.0]], dtype=np.float32))
