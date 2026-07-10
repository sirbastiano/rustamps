from __future__ import annotations

import numpy as np

from scripts.stage6_hf_flow_diagnostics import edge_flow_diff_summary


def test_edge_flow_diff_summary_includes_global_flow_distribution() -> None:
    ifgw = np.ones((1, 2), dtype=np.complex64)
    rowcost = np.zeros((0, 2, 4), dtype=np.int16)
    colcost = np.asarray([[[-200, 1000, 32000, -32000]]], dtype=np.int16)
    native = np.zeros((1, 2), dtype=np.float32)
    snaphu = np.asarray([[0.0, 2.0 * np.pi]], dtype=np.float32)

    summary = edge_flow_diff_summary(ifgw, rowcost, colcost, native, snaphu)

    assert summary["inferred_distribution"] == {
        "native_abs_counts": [[0, 1]],
        "snaphu_abs_counts": [[1, 1]],
        "delta_counts": [[1, 1]],
        "changed_edges": 1,
    }
