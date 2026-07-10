from __future__ import annotations

import numpy as np

from scripts.stage6_cut_window_diagnostics import cut_window_candidate_summary


def test_cut_window_summary_reports_uncovered_tall_component() -> None:
    ifgw = np.ones((3, 3), dtype=np.complex64)
    rowcost = np.zeros((2, 3, 4), dtype=np.int16)
    colcost = np.asarray(
        [
            [[-400, 1000, 32000, -32000], [-400, 1000, 32000, -32000]],
            [[-400, 1000, 32000, -32000], [-400, 1000, 32000, -32000]],
            [[-400, 1000, 32000, -32000], [-400, 1000, 32000, -32000]],
        ],
        dtype=np.int16,
    )
    native = np.zeros((3, 3), dtype=np.float32)
    snaphu = np.zeros((3, 3), dtype=np.float32)
    snaphu[:, 1] = 2.0 * np.pi

    summary = cut_window_candidate_summary(ifgw, rowcost, colcost, native, snaphu, max_cells=6)

    component = summary["components"]["-1"][0]
    assert summary["cut_side"] == 2
    assert component["bbox_height"] == 3
    assert component["bbox_width"] == 1
    assert component["full_cover_candidate_count"] == 0
    assert component["max_overlap_pixels"] == 2
    assert component["max_overlap_fraction"] == 2 / 3


def test_cut_window_summary_reports_full_candidate_cover() -> None:
    ifgw = np.ones((2, 3), dtype=np.complex64)
    rowcost = np.zeros((1, 3, 4), dtype=np.int16)
    colcost = np.asarray(
        [
            [[-400, 1000, 32000, -32000], [-400, 1000, 32000, -32000]],
            [[-400, 1000, 32000, -32000], [-400, 1000, 32000, -32000]],
        ],
        dtype=np.int16,
    )
    native = np.zeros((2, 3), dtype=np.float32)
    snaphu = np.zeros((2, 3), dtype=np.float32)
    snaphu[:, 1] = 2.0 * np.pi

    summary = cut_window_candidate_summary(ifgw, rowcost, colcost, native, snaphu, max_cells=4)

    component = summary["components"]["-1"][0]
    assert summary["cut_side"] == 2
    assert component["size"] == 2
    assert component["full_cover_candidate_count"] > 0
    assert component["max_overlap_fraction"] == 1.0
    assert component["best_overlap_window"]["height"] == 2
    assert component["best_overlap_window"]["width"] == 2
