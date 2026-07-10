from __future__ import annotations

import numpy as np

from scripts.stage6_residual_cycle_diagnostics import residual_cycle_summary


def test_residual_cycle_summary_detects_negative_cycle() -> None:
    ifgw = np.ones((2, 2), dtype=np.complex64)
    rowcost = np.asarray([[[0, 1000, 32000, -32000], [0, 1000, 32000, -32000]]], dtype=np.int16)
    colcost = np.asarray(
        [
            [[0, 1000, 32000, -32000]],
            [[0, 1000, 32000, -32000]],
        ],
        dtype=np.int16,
    )
    unwrapped = np.asarray([[0.0, 0.0], [0.0, 2.0 * np.pi]], dtype=np.float32)

    summary = residual_cycle_summary(ifgw, rowcost, colcost, unwrapped)

    assert summary["status"] == "ok"
    assert summary["negative_cycle"] is True
    assert summary["last_relaxed_cost"] < 0


def test_residual_cycle_summary_reports_local_optimum() -> None:
    ifgw = np.ones((2, 2), dtype=np.complex64)
    rowcost = np.asarray([[[0, 1000, 32000, -32000], [0, 1000, 32000, -32000]]], dtype=np.int16)
    colcost = np.asarray(
        [
            [[0, 1000, 32000, -32000]],
            [[0, 1000, 32000, -32000]],
        ],
        dtype=np.int16,
    )
    unwrapped = np.zeros((2, 2), dtype=np.float32)

    summary = residual_cycle_summary(ifgw, rowcost, colcost, unwrapped)

    assert summary["status"] == "ok"
    assert summary["negative_cycle"] is False
    assert summary["last_relaxed_cost"] is None


def test_residual_cycle_summary_skips_large_graph() -> None:
    ifgw = np.ones((4, 4), dtype=np.complex64)
    rowcost = np.zeros((3, 4, 4), dtype=np.int16)
    colcost = np.zeros((4, 3, 4), dtype=np.int16)
    unwrapped = np.zeros((4, 4), dtype=np.float32)

    summary = residual_cycle_summary(ifgw, rowcost, colcost, unwrapped, max_nodes=2)

    assert summary["status"] == "skipped"
    assert summary["node_count"] == 10
