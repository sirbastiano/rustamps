from __future__ import annotations

import numpy as np

from scripts.stage6_flow_component_diagnostics import changed_flow_component_summary


def test_changed_flow_component_summary_groups_edges_by_incident_cells() -> None:
    ifgw = np.ones((2, 2), dtype=np.complex64)
    rowcost = np.asarray([[[0, 32000, 32000, -32000], [0, 32000, 32000, -32000]]], dtype=np.int16)
    colcost = np.asarray(
        [
            [[0, 32000, 32000, -32000]],
            [[0, 32000, 32000, -32000]],
        ],
        dtype=np.int16,
    )
    native = np.zeros((2, 2), dtype=np.float32)
    snaphu = np.asarray([[0.0, 2.0 * np.pi], [2.0 * np.pi, 2.0 * np.pi]], dtype=np.float32)

    summary = changed_flow_component_summary(ifgw, rowcost, colcost, native, snaphu)

    assert summary["component_count"] == 1
    assert summary["total_changed_edges"] == 2
    assert summary["components"] == [
        {
            "bbox": [0, 1, 0, 1],
            "cell_count": 3,
            "changed_edges": 2,
            "changed_h": 1,
            "changed_v": 1,
            "delta_counts": [[-1, 1], [1, 1]],
            "native_cost_on_changed": 0,
            "native_minus_snaphu_cost": 0,
            "snaphu_cost_on_changed": 0,
        }
    ]
