from __future__ import annotations

import numpy as np
import pytest
from scipy import spatial

from pystamps.pipeline.scn import build_scn_payload


def _legacy_edge_high_pass(phase: np.ndarray, xy: np.ndarray, day: np.ndarray, master: int, window: float) -> np.ndarray:
    simplices = spatial.Delaunay(xy).simplices
    edges = np.unique(
        np.sort(
            np.concatenate(
                (
                    simplices[:, [0, 1]],
                    simplices[:, [1, 2]],
                    simplices[:, [0, 2]],
                )
            ),
            axis=1,
        ),
        axis=0,
    )
    incidence = np.zeros((edges.shape[0], xy.shape[0]), dtype=np.float64)
    incidence[np.arange(edges.shape[0]), edges[:, 0]] = -1.0
    incidence[np.arange(edges.shape[0]), edges[:, 1]] = 1.0
    pair_phase = incidence @ phase
    low = np.empty_like(pair_phase)
    for column in range(day.size):
        weights = np.exp(-((day[column] - day) ** 2) / (2.0 * window**2))
        weights[master] = 0.0
        weights /= np.sum(weights)
        low[:, column] = pair_phase @ weights
    solution = np.linalg.lstsq(incidence[:, 1:], pair_phase - low, rcond=None)[0]
    return np.vstack((np.zeros((1, day.size)), solution))


def _brute_spatial(phase: np.ndarray, xy: np.ndarray, wavelength: float) -> np.ndarray:
    output = np.empty(phase.shape, dtype=np.float64)
    radius_sq = (4.0 * wavelength) ** 2
    for index in range(xy.shape[0]):
        dist_sq = np.sum((xy - xy[index]) ** 2, axis=1)
        keep = dist_sq < radius_sq
        weights = np.exp(-dist_sq[keep] / (2.0 * wavelength**2))
        weights /= np.sum(weights)
        output[index] = weights @ phase[keep]
    return output - output[0]


def test_temporal_shortcut_matches_legacy_triangulation_solve() -> None:
    xy = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [2.0, 2.0], [1.0, 1.0]])
    day = np.asarray([0.0, 12.0, 30.0, 55.0])
    phase = np.asarray(
        [
            [0.1, 0.3, 0.0, 0.8],
            [0.5, 0.2, 0.0, 1.4],
            [-0.2, 0.7, 0.0, 0.6],
            [1.2, -0.1, 0.0, 2.0],
            [0.4, 0.5, 0.0, 1.1],
        ],
        dtype=np.float64,
    )

    payload = build_scn_payload(
        phase,
        xy,
        day,
        master_index=2,
        time_window=20.0,
        wavelength=0.2,
        chunk_points=2,
    )
    expected = _legacy_edge_high_pass(phase, xy, day, master=2, window=20.0)

    np.testing.assert_allclose(payload["ph_hpt"], expected.astype(np.float32), atol=2e-7, rtol=2e-7)


def test_payload_matches_legacy_corrections_deramp_and_spatial_filter() -> None:
    xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [4.0, 0.0]])
    day = np.asarray([0.0, 5.0, 11.0, 20.0])
    ph_uw = np.asarray(
        [
            [1.0, 0.0, 2.0, 5.0],
            [2.0, 0.0, np.nan, 8.0],
            [0.0, 0.0, 1.0, 4.0],
            [3.0, 0.0, 0.5, 10.0],
        ],
        dtype=np.float32,
    )
    ph_scla = np.full(ph_uw.shape, 0.25, dtype=np.float32)
    c_ps = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    scla_ramp = np.zeros(ph_uw.shape, dtype=np.float64)
    scla_ramp[:, 3] = np.asarray([0.0, 0.1, 0.2, 0.3])
    unwrap = np.asarray([0, 1, 3])

    payload = build_scn_payload(
        ph_uw,
        xy,
        day,
        master_index=1,
        unwrap_indices=unwrap,
        time_window=8.0,
        wavelength=1.0,
        ph_scla=ph_scla,
        c_ps_uw=c_ps,
        scla_ramp=scla_ramp,
        deramp_indices=[3],
        chunk_points=2,
    )

    corrected = ph_uw[:, unwrap].astype(np.float64) - ph_scla[:, unwrap] - c_ps[:, None] - scla_ramp[:, unwrap]
    corrected[np.isnan(corrected)] = 0.0
    design = np.column_stack((np.ones(xy.shape[0]), xy))
    ramp = design @ np.linalg.lstsq(design, corrected[:, 2], rcond=None)[0]
    corrected[:, 2] -= ramp
    high = _legacy_edge_high_pass(corrected, xy, day[unwrap], master=1, window=8.0)
    high[:, 2] += ramp
    high = high.astype(np.float32)
    spatial_phase = _brute_spatial(high, xy, wavelength=1.0)
    expected_slave = np.zeros(ph_uw.shape, dtype=np.float64)
    expected_slave[:, unwrap] = spatial_phase
    expected_slave[:, 1] = 0.0

    np.testing.assert_allclose(payload["ph_ramp"], ramp[:, None], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(payload["ph_hpt"], high, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(payload["ph_scn_slave"], expected_slave, atol=2e-7, rtol=2e-7)
    assert payload["ph_hpt"].dtype == np.float32
    assert payload["ph_scn_slave"].dtype == np.float64


def test_temporal_weights_remain_finite_when_naive_gaussian_underflows() -> None:
    phase = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    payload = build_scn_payload(
        phase,
        np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        np.asarray([0.0, 1e9]),
        master_index=0,
        time_window=1e-3,
        wavelength=1.0,
    )

    assert np.all(np.isfinite(payload["ph_hpt"]))
    assert np.all(np.isfinite(payload["ph_scn_slave"]))


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("time_window", 0.0, "time_window"),
        ("wavelength", -1.0, "wavelength"),
        ("master_index", 2, "master_index"),
        ("unwrap_indices", [2], "unwrap_indices"),
    ],
)
def test_rejects_invalid_scientific_parameters(keyword: str, value: object, message: str) -> None:
    arguments: dict[str, object] = {
        "master_index": 0,
        "time_window": 1.0,
        "wavelength": 1.0,
    }
    arguments[keyword] = value
    with pytest.raises(ValueError, match=message):
        build_scn_payload(
            np.zeros((2, 2)),
            np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            np.asarray([0.0, 1.0]),
            **arguments,
        )
