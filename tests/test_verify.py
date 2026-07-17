from pathlib import Path

import numpy as np

from pystamps.config import ToleranceConfig
from pystamps.io.mat import write_mat
from pystamps.verify import FileComparison, VerificationReport, _compare_mat, classify_failures, summarize_failures


def test_classify_failures_groups_downstream_residuals() -> None:
    report = VerificationReport(
        comparisons=[
            FileComparison("PATCH_1/select1.mat", False, "Value mismatch for key 'C_ps2', max_abs=2.79e-05"),
            FileComparison("phuw2.mat", False, "Value mismatch for key 'msd', max_abs=14.9361"),
            FileComparison("uw_space_time.mat", False, "Wrap mismatch for key 'dph_noise', wrapped_max_abs=6.26338"),
            FileComparison("uw_interp.mat", True, "Matched 1 numeric keys"),
        ]
    )

    failures = classify_failures(report)

    assert [failure.failure_class for failure in failures] == [
        "stage3_patch_boundary",
        "unwrap_smoothing",
        "unwrapped_noise_statistics",
    ]
    assert [failure.failing_key for failure in failures] == ["C_ps2", "msd", "dph_noise"]


def test_summarize_failures_includes_trace_guidance() -> None:
    report = VerificationReport(
        comparisons=[
            FileComparison("ifgstd2.mat", False, "Value mismatch for key 'ifg_std', max_abs=0.125"),
            FileComparison("mean_v.mat", False, "Value mismatch for key 'm', max_abs=8.3154"),
        ]
    )

    summary = summarize_failures(report)

    assert summary["failed"] == 2
    assert [group["failure_class"] for group in summary["groups"]] == [
        "unwrap_smoothing",
        "unwrapped_noise_statistics",
    ]
    assert summary["first_boundary_failure"]["path"] == "ifgstd2.mat"
    assert summary["trace"]["stage3_4_residual_present"] is False
    assert summary["trace"]["stage3_4_coupling_evidence_present"] is False


def test_summarize_failures_prioritizes_earliest_stage_boundary() -> None:
    report = VerificationReport(
        comparisons=[
            FileComparison(
                "uw_space_time.mat",
                False,
                "Shape mismatch for key 'dph_noise': (3, 4) != (5, 4)",
                failure_kind="shape_mismatch",
                failing_key="dph_noise",
                shape_run=(3, 4),
                shape_oracle=(5, 4),
            ),
            FileComparison(
                "PATCH_1/pm1.mat",
                False,
                "Value mismatch for key 'C_ps', max_abs=1.25",
                failure_kind="value_mismatch",
                failing_key="C_ps",
                shape_run=(2,),
                shape_oracle=(2,),
                max_abs=1.25,
            ),
        ]
    )

    summary = summarize_failures(report)

    assert summary["first_boundary_failure"] == {
        "path": "PATCH_1/pm1.mat",
        "message": "Value mismatch for key 'C_ps', max_abs=1.25",
        "stage_scope": "stage2",
        "failure_class": "stage2_patch_boundary",
        "label": "Stage 2 patch boundary",
        "failing_key": "C_ps",
        "failure_kind": "value_mismatch",
        "shape_run": [2],
        "shape_oracle": [2],
        "max_abs": 1.25,
        "guidance": (
            "pm1.mat diverges before later patch stages; fix stage-2 parity before changing stage-3/4 or "
            "downstream code."
        ),
    }
    assert summary["trace"]["stage2_residual_present"] is True


def _compare_payloads(tmp_path: Path, run_payload: dict[str, np.ndarray], golden_payload: dict[str, np.ndarray]) -> bool:
    run_path = tmp_path / "run.mat"
    golden_path = tmp_path / "golden.mat"
    write_mat(run_path, run_payload)
    write_mat(golden_path, golden_payload)
    ok, _, _ = _compare_mat(run_path, golden_path, ToleranceConfig())
    return ok


def test_verify_compares_complex_phase_magnitude(tmp_path: Path) -> None:
    assert not _compare_payloads(
        tmp_path,
        {"ph": np.asarray([[0.0 + 2.0j]])},
        {"ph": np.asarray([[0.0 + 1.0j]])},
    )


def test_verify_does_not_wrap_unwrapped_phase_products(tmp_path: Path) -> None:
    for key in ("ph_uw", "dph_space_uw"):
        assert not _compare_payloads(
            tmp_path,
            {key: np.asarray([[2.0 * np.pi]])},
            {key: np.asarray([[0.0]])},
        )


def test_verify_retains_wrap_equivalence_for_angular_noise(tmp_path: Path) -> None:
    assert _compare_payloads(
        tmp_path,
        {"dph_noise": np.asarray([[2.0 * np.pi + 0.25]])},
        {"dph_noise": np.asarray([[0.25]])},
    )
