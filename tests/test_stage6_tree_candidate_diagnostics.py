from __future__ import annotations

import numpy as np

from scripts.stage6_tree_candidate_diagnostics import tree_candidate_summary_from_flows


def test_tree_candidate_summary_detects_remount_exposed_cycle() -> None:
    nrow = 3
    ncol = 4
    h_flow = np.asarray([[-3, -1, 0], [3, 0, -1], [-1, 0, 1]], dtype=np.int64)
    v_flow = np.asarray([[-3, 2, 2, 1], [-1, 1, -1, 2]], dtype=np.int64)
    rowcost = np.tile(np.asarray([0, 1000, 32000, -32000], dtype=np.int16), (2, 4, 1))
    colcost = np.tile(np.asarray([0, 1000, 32000, -32000], dtype=np.int16), (3, 3, 1))

    summary = tree_candidate_summary_from_flows(rowcost, colcost, h_flow, v_flow)

    assert summary["status"] == "ok"
    assert summary["negative_cycle"] is True
    assert summary["retained_tree_negative_cycle"] is False
    assert summary["negative_arc_candidate_count"] > 0
    assert summary["reduced_cost_remounts"] > 0
    assert summary["remounted_tree_negative_cycle"] is True


def test_tree_candidate_summary_can_skip_full_cycle_scan_only() -> None:
    nrow = 3
    ncol = 4
    h_flow = np.zeros((nrow, ncol - 1), dtype=np.int64)
    v_flow = np.zeros((nrow - 1, ncol), dtype=np.int64)
    rowcost = np.tile(np.asarray([0, 1000, 32000, -32000], dtype=np.int16), (2, 4, 1))
    colcost = np.tile(np.asarray([0, 1000, 32000, -32000], dtype=np.int16), (3, 3, 1))

    summary = tree_candidate_summary_from_flows(
        rowcost,
        colcost,
        h_flow,
        v_flow,
        max_nodes=20,
        cycle_check_max_nodes=1,
    )

    assert summary["status"] == "ok"
    assert summary["negative_cycle"] is None
    assert summary["retained_tree_negative_cycle"] is False
    assert summary["remounted_tree_negative_cycle"] is False
