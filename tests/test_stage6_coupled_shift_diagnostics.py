from __future__ import annotations

import numpy as np

from pystamps.kernels import run_stage6_unwrap_grid_kernel
from scripts.stage6_coupled_shift_diagnostics import coupled_threshold_summary
from scripts.stage6_hf_diagnostics import initial_defo_objective


def test_coupled_threshold_summary_reports_negative_required_step() -> None:
    ifgw = np.ones((1, 3), dtype=np.complex64)
    rowcost = np.zeros((0, 3, 4), dtype=np.int16)
    colcost = np.asarray(
        [[[-600, 1000, 32000, -32000], [0, 1000, 32000, -32000]]],
        dtype=np.int16,
    )
    native = np.zeros((1, 3), dtype=np.float32)
    snaphu = np.asarray([[0.0, 2.0 * np.pi, -2.0 * np.pi]], dtype=np.float32)

    summary = coupled_threshold_summary(ifgw, rowcost, colcost, native, snaphu)

    assert summary["sequential_gain"] == 40
    assert summary["positive_gain"] == 160
    assert summary["negative_gain"] == -120
    assert summary["negative_step_count"] == 1
    assert summary["requires_coupled_acceptance"] is True


def test_coupled_threshold_synthetic_case_is_not_an_objective_oracle() -> None:
    ifgw = np.ones((1, 3), dtype=np.complex64)
    rowcost = np.zeros((0, 12), dtype=np.int16)
    colcost = np.asarray(
        [[-600, 1000, 32000, -32000, 0, 1000, 32000, -32000]],
        dtype=np.int16,
    )
    oracle = np.asarray([[0.0, 2.0 * np.pi, -2.0 * np.pi]], dtype=np.float32)

    native = run_stage6_unwrap_grid_kernel(ifgw, rowcost, colcost, backend="native")["ifguw"]

    rowcost_grid = rowcost.reshape(0, 3, 4)
    colcost_grid = colcost.reshape(1, 2, 4)
    native_objective = initial_defo_objective(ifgw, rowcost_grid, colcost_grid, native)
    oracle_objective = initial_defo_objective(ifgw, rowcost_grid, colcost_grid, oracle)
    assert native_objective < oracle_objective
