from __future__ import annotations

import numpy as np

from scripts import stage6_hf_diagnostics
from scripts.stage6_hf_flow_diagnostics import (
    flow_distribution_summary,
    flow_dump_match_summary,
    flow_dump_summaries,
    load_snaphu_flow,
)


def test_flow_dump_match_summary_checks_snaphu_row_column_orientation() -> None:
    ifgw = np.ones((2, 2), dtype=np.complex64)
    unwrapped = np.asarray([[0.0, 2.0 * np.pi], [4.0 * np.pi, 6.0 * np.pi]], dtype=np.float32)
    row_flow = np.asarray([[-2, -2]], dtype=np.int16)
    col_flow = np.asarray([[1], [1]], dtype=np.int16)

    exact = flow_dump_match_summary(ifgw, unwrapped, row_flow, col_flow)
    row_mismatch = flow_dump_match_summary(ifgw, unwrapped, row_flow + 1, col_flow)

    assert exact["exact"] is True
    assert exact["row_mismatch_count"] == 0
    assert exact["col_mismatch_count"] == 0
    assert row_mismatch["exact"] is False
    assert row_mismatch["row_mismatch_count"] == 2


def test_flow_dump_summaries_compare_native_and_snaphu_to_dump() -> None:
    ifgw = np.ones((1, 2), dtype=np.complex64)
    row_flow = np.zeros((0, 2), dtype=np.int16)
    col_flow = np.asarray([[1]], dtype=np.int16)
    snaphu = np.asarray([[0.0, 2.0 * np.pi]], dtype=np.float32)
    native = np.zeros((1, 2), dtype=np.float32)

    summary = flow_dump_summaries(ifgw, native, snaphu, row_flow, col_flow)

    assert summary["snaphu"]["exact"] is True
    assert summary["native"]["exact"] is False
    assert summary["native"]["row_mismatch_count"] == 0
    assert summary["native"]["col_mismatch_count"] == 1


def test_flow_distribution_summary_reports_abs_and_delta_counts() -> None:
    native = np.asarray([[0, 1, -2]], dtype=np.int16)
    snaphu = np.asarray([[1, 1, 0]], dtype=np.int16)

    summary = flow_distribution_summary(native, snaphu)

    assert summary["native_abs_counts"] == [[0, 1], [1, 1], [2, 1]]
    assert summary["snaphu_abs_counts"] == [[0, 1], [1, 2]]
    assert summary["delta_counts"] == [[0, 1], [1, 1], [2, 1]]
    assert summary["changed_edges"] == 2


def test_load_snaphu_flow_splits_row_then_column_values(tmp_path) -> None:
    path = tmp_path / "snaphu.flow"
    np.arange(7, dtype=np.int16).tofile(path)

    row_flow, col_flow = load_snaphu_flow(path, (2, 3))

    np.testing.assert_array_equal(row_flow, np.asarray([[0, 1, 2]], dtype=np.int16))
    np.testing.assert_array_equal(col_flow, np.asarray([[3, 4], [5, 6]], dtype=np.int16))


def test_analyze_fixture_includes_optional_flow_dump_summaries(tmp_path, monkeypatch) -> None:
    ifgw = np.ones((1, 2), dtype=np.complex64)
    rowcost = np.zeros((0, 2, 4), dtype=np.int16)
    colcost = np.asarray([[[-200, 1000, 32000, -32000]]], dtype=np.int16)
    snaphu = np.asarray([[0.0, 2.0 * np.pi]], dtype=np.float32)
    native = np.zeros((1, 2), dtype=np.float32)
    flowfile = tmp_path / "snaphu.flow"
    np.asarray([1], dtype=np.int16).tofile(flowfile)
    monkeypatch.setattr(
        stage6_hf_diagnostics,
        "_load_fixture",
        lambda _root: (np.ones((1, 2), dtype=bool), ifgw, rowcost, colcost, snaphu),
    )
    observed = {}

    def fake_kernel(*_args, **kwargs):
        observed["threads"] = kwargs["threads"]
        return {"ifguw": native, "msd": 0.0}

    monkeypatch.setattr(stage6_hf_diagnostics, "run_stage6_unwrap_grid_kernel", fake_kernel)

    summary = stage6_hf_diagnostics.analyze_fixture(tmp_path, flowfile=flowfile, threads=3)

    assert observed["threads"] == 3
    assert summary["flow_dump_match"]["snaphu"]["exact"] is True
    assert summary["flow_dump_match"]["native"]["exact"] is False
