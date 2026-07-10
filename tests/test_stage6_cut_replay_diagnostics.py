from __future__ import annotations

import numpy as np

from scripts.stage6_cut_replay_diagnostics import replay_binary_cut_patch


def test_replay_binary_cut_patch_selects_improving_cell() -> None:
    ifgw = np.ones((1, 2), dtype=np.complex64)
    rowcost = np.zeros((0, 2, 4), dtype=np.int16)
    colcost = np.asarray([[[-400, 1000, 32000, -32000]]], dtype=np.int16)
    native = np.zeros((1, 2), dtype=np.float32)

    summary = replay_binary_cut_patch(ifgw, rowcost, colcost, native, (0, 0, 1, 2), shift=1)

    assert summary["selected_gain"] == 120
    assert summary["selected_count"] == 1
    assert summary["selected_mask"] == [[False, True]]
    assert summary["pair_terms"] == 1
    assert summary["skipped_non_submodular_pairs"] == 0
