from __future__ import annotations

import numpy as np

from scripts.stage6_native_candidate_diagnostics import potential_component_rectangles


def test_potential_component_rectangles_cover_tall_native_structure() -> None:
    ifgw = np.ones((3, 4), dtype=np.complex64)
    rowcost = np.zeros((2, 4, 4), dtype=np.int16)
    colcost = np.zeros((3, 3, 4), dtype=np.int16)
    colcost[:, 1, :] = np.asarray([-400, 1000, 32000, -32000], dtype=np.int16)
    native = np.zeros((3, 4), dtype=np.float32)

    rectangles = potential_component_rectangles(
        ifgw,
        rowcost,
        colcost,
        native,
        threshold=1,
        max_cells=16,
    )

    assert rectangles[0]["bbox"] == [0, 2, 1, 2]
    assert rectangles[0]["cells"] == 6
    assert rectangles[0]["score"] > 0
