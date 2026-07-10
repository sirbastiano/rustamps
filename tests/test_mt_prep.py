from __future__ import annotations

from pathlib import Path

import numpy as np

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


def test_prepare_snap_mt_prep_inputs_writes_patch_candidates(tmp_path: Path) -> None:
    root = tmp_path / "INSAR_20240113"
    (root / "rslc").mkdir(parents=True)
    (root / "diff0").mkdir()
    (root / "geo").mkdir()

    width, length = 4, 3
    dates = ["20240101", "20240113", "20240125"]
    base = np.full((length, width), 10.0, dtype=np.float32)
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
    for date in dates:
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

    summary = prepare_snap_mt_prep_inputs(
        root,
        master_date="20240113",
        amp_dispersion=0.25,
        range_patches=1,
        azimuth_patches=1,
        range_overlap=0,
        azimuth_overlap=0,
        force=True,
    )

    patch = root / "PATCH_1"
    assert summary.patch_count == 1
    assert summary.candidate_count == 2
    assert (root / "patch.list").read_text(encoding="utf-8") == "PATCH_1\n"
    np.testing.assert_array_equal(np.loadtxt(patch / "pscands.1.ij"), np.asarray([[1, 0, 0], [2, 1, 2]]))
    np.testing.assert_array_equal(np.fromfile(patch / "pscands.1.ij.int", dtype=">i4"), np.asarray([0, 0, 2, 1]))
    np.testing.assert_allclose(np.loadtxt(patch / "pscands.1.da"), np.asarray([0.0, np.std([4, 5, 4]) / np.mean([4, 5, 4])]))
    np.testing.assert_allclose(np.fromfile(patch / "pscands.1.ll", dtype=">f4").reshape(-1, 2), np.asarray([[12.0, 41.0], [12.2, 41.1]]))
    np.testing.assert_allclose(np.fromfile(patch / "pscands.1.hgt", dtype=">f4"), np.asarray([70.0, 76.0]))
    np.testing.assert_allclose(
        _load_phase(patch / "pscands.1.ph", 2),
        np.asarray([[100 + 1100j, 200 + 1200j], [106 + 1106j, 206 + 1206j]], dtype=np.complex64),
    )
