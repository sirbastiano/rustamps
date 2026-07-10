from __future__ import annotations

import numpy as np
import pytest

from scripts import stage6_hf_diagnostics


def test_dense_msd_skips_zero_neighbor_diffs() -> None:
    values = np.asarray([[1.0, 1.0], [3.0, 1.0]], dtype=np.float32)

    assert stage6_hf_diagnostics.dense_msd(values) == 4.0


def test_label_diff_summary_reports_integer_cycle_offsets() -> None:
    twopi = np.float32(2.0 * np.pi)
    snaphu = np.zeros((2, 3), dtype=np.float32)
    native = np.asarray([[0.0, twopi, twopi], [0.0, -twopi, 0.0]], dtype=np.float32)

    summary = stage6_hf_diagnostics.label_diff_summary(native, snaphu)

    assert summary["diff_min"] == -1
    assert summary["diff_max"] == 1
    assert summary["diff_unique"] == 3
    assert summary["top_counts"] == [[0, 3], [1, 2], [-1, 1]]
    assert summary["change_edges_h"] == 3
    assert summary["change_edges_v"] == 2


def test_initial_defo_objective_uses_wrapped_phase_cycle_labels() -> None:
    ifgw = np.ones((1, 2), dtype=np.complex64)
    rowcost = np.zeros((0, 2, 4), dtype=np.int16)
    colcost = np.asarray([[[-200, 1000, 32000, -32000]]], dtype=np.int16)
    unwrapped_flat = np.zeros((1, 2), dtype=np.float32)
    unwrapped_shift = np.asarray([[0.0, 2.0 * np.pi]], dtype=np.float32)

    flat = stage6_hf_diagnostics.initial_defo_objective(
        ifgw,
        rowcost,
        colcost,
        unwrapped_flat,
    )
    shifted = stage6_hf_diagnostics.initial_defo_objective(
        ifgw,
        rowcost,
        colcost,
        unwrapped_shift,
    )

    assert shifted < flat


def test_component_shift_summary_scores_boundary_objective_gain() -> None:
    ifgw = np.ones((1, 2), dtype=np.complex64)
    rowcost = np.zeros((0, 2, 4), dtype=np.int16)
    colcost = np.asarray([[[-200, 1000, 32000, -32000]]], dtype=np.int16)
    native = np.zeros((1, 2), dtype=np.float32)
    snaphu = np.asarray([[0.0, 2.0 * np.pi]], dtype=np.float32)

    summary = stage6_hf_diagnostics.component_shift_summary(ifgw, rowcost, colcost, native, snaphu)

    component = summary["-1"]["top"][0]
    assert component["size"] == 1
    assert component["shift"] == 1
    assert component["gain"] == (
        stage6_hf_diagnostics.initial_defo_objective(ifgw, rowcost, colcost, native)
        - stage6_hf_diagnostics.initial_defo_objective(ifgw, rowcost, colcost, snaphu)
    )
    assert component["gain"] > 0


def test_component_isolation_summary_reports_zero_boundary_blocker() -> None:
    ifgw = np.ones((2, 3), dtype=np.complex64)
    rowcost = np.asarray(
        [
            [[0, 1000, 32000, -32000], [0, 1000, 32000, -32000], [0, 1000, 32000, -32000]],
        ],
        dtype=np.int16,
    )
    colcost = np.asarray(
        [
            [[-400, 1000, 32000, -32000], [0, 1000, 32000, -32000]],
            [[0, 1000, 32000, -32000], [0, 1000, 32000, -32000]],
        ],
        dtype=np.int16,
    )
    native = np.zeros((2, 3), dtype=np.float32)
    snaphu = np.zeros((2, 3), dtype=np.float32)
    snaphu[0, 1] = 2.0 * np.pi

    summary = stage6_hf_diagnostics.component_isolation_summary(ifgw, rowcost, colcost, native, snaphu)

    component = summary["-1"]["top"][0]
    assert summary["barrier_thresholds"] == [160, 80, 40, 20, 10, 5]
    assert component["boundary_min"] == 0
    assert component["isolated_thresholds"] == []
    assert component["gain"] > 0


def test_component_isolation_summary_reports_cut_window_geometry() -> None:
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

    summary = stage6_hf_diagnostics.component_isolation_summary(
        ifgw,
        rowcost,
        colcost,
        native,
        snaphu,
        cut_max_cells=6,
    )

    component = summary["-1"]["top"][0]
    assert summary["cut_max_cells"] == 6
    assert summary["cut_square_side"] == 2
    assert component["bbox_height"] == 3
    assert component["bbox_width"] == 1
    assert component["bbox_cells"] == 3
    assert component["fits_cut_cell_budget"] is True
    assert component["fits_square_cut_window"] is False


def test_edge_flow_diff_summary_reports_costed_flow_deltas() -> None:
    ifgw = np.ones((1, 2), dtype=np.complex64)
    rowcost = np.zeros((0, 2, 4), dtype=np.int16)
    colcost = np.asarray([[[-200, 1000, 32000, -32000]]], dtype=np.int16)
    native = np.zeros((1, 2), dtype=np.float32)
    snaphu = np.asarray([[0.0, 2.0 * np.pi]], dtype=np.float32)

    summary = stage6_hf_diagnostics.edge_flow_diff_summary(ifgw, rowcost, colcost, native, snaphu)

    assert summary["horizontal"]["changed_edges"] == 1
    assert summary["vertical"]["changed_edges"] == 0
    assert summary["horizontal"]["delta_counts"] == [[1, 1]]
    assert summary["horizontal"]["native_cost_on_changed"] > summary["horizontal"]["snaphu_cost_on_changed"]
    assert summary["total_native_cost_on_changed"] > summary["total_snaphu_cost_on_changed"]


def test_edge_flow_diff_changed_costs_account_for_objective_delta() -> None:
    ifgw = np.ones((2, 2), dtype=np.complex64)
    rowcost = np.asarray(
        [
            [[0, 1000, 32000, -32000], [0, 1000, 32000, -32000]],
        ],
        dtype=np.int16,
    )
    colcost = np.asarray(
        [
            [[-200, 1000, 32000, -32000]],
            [[0, 1000, 32000, -32000]],
        ],
        dtype=np.int16,
    )
    native = np.zeros((2, 2), dtype=np.float32)
    snaphu = np.asarray([[0.0, 2.0 * np.pi], [0.0, 2.0 * np.pi]], dtype=np.float32)

    native_objective = stage6_hf_diagnostics.initial_defo_objective(ifgw, rowcost, colcost, native)
    snaphu_objective = stage6_hf_diagnostics.initial_defo_objective(ifgw, rowcost, colcost, snaphu)
    summary = stage6_hf_diagnostics.edge_flow_diff_summary(ifgw, rowcost, colcost, native, snaphu)

    changed_delta = summary["total_native_cost_on_changed"] - summary["total_snaphu_cost_on_changed"]
    assert summary["total_changed_edges"] == 2
    assert changed_delta == native_objective - snaphu_objective


def test_native_unwrap_cache_round_trips_and_validates_shape(tmp_path) -> None:
    native = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "native.npy"

    stage6_hf_diagnostics.save_native_unwrap(path, native)

    np.testing.assert_array_equal(
        stage6_hf_diagnostics.load_native_unwrap(path, (2, 2)),
        native,
    )
    with pytest.raises(ValueError, match="shape"):
        stage6_hf_diagnostics.load_native_unwrap(path, (1, 4))


def test_oracle_threshold_shift_summary_scores_nested_snaphu_corrections() -> None:
    ifgw = np.ones((1, 4), dtype=np.complex64)
    rowcost = np.zeros((0, 4, 4), dtype=np.int16)
    colcost = np.asarray(
        [[[0, 1000, 32000, -32000], [0, 1000, 32000, -32000], [0, 1000, 32000, -32000]]],
        dtype=np.int16,
    )
    native = np.zeros((1, 4), dtype=np.float32)
    snaphu = np.asarray([[0.0, 2.0 * np.pi, 2.0 * np.pi, 4.0 * np.pi]], dtype=np.float32)

    summary = stage6_hf_diagnostics.oracle_threshold_shift_summary(ifgw, rowcost, colcost, native, snaphu)

    assert summary["correction_min"] == 0
    assert summary["correction_max"] == 2
    assert summary["thresholds"][0]["shift"] == 1
    assert summary["thresholds"][0]["pixels"] == 3
    assert summary["thresholds"][1]["shift"] == 1
    assert summary["thresholds"][1]["pixels"] == 1
    assert summary["sequential_gain"] == (
        stage6_hf_diagnostics.initial_defo_objective(ifgw, rowcost, colcost, native)
        - stage6_hf_diagnostics.initial_defo_objective(ifgw, rowcost, colcost, snaphu)
    )


def test_oracle_boundary_energy_summary_scores_native_boundary_costs() -> None:
    ifgw = np.ones((1, 4), dtype=np.complex64)
    rowcost = np.zeros((0, 4, 4), dtype=np.int16)
    colcost = np.asarray(
        [[[200, 1000, 32000, -32000], [0, 1000, 32000, -32000], [400, 1000, 32000, -32000]]],
        dtype=np.int16,
    )
    native = np.zeros((1, 4), dtype=np.float32)
    snaphu = np.asarray([[0.0, 2.0 * np.pi, 2.0 * np.pi, 4.0 * np.pi]], dtype=np.float32)

    summary = stage6_hf_diagnostics.oracle_boundary_energy_summary(ifgw, rowcost, colcost, native, snaphu)

    assert summary["total_native_edge_cost"] == 200
    assert summary["thresholds"][0]["shift"] == 1
    assert summary["thresholds"][0]["boundary_edges"] == 1
    assert summary["thresholds"][0]["boundary_native_cost"] == 40
    assert summary["thresholds"][1]["level"] == 2
    assert summary["thresholds"][1]["boundary_native_cost"] == 160

