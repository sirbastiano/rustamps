from pystamps.verify import FileComparison, VerificationReport, classify_failures, summarize_failures


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
        "upstream_patch_residual",
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
    assert summary["trace"]["stage3_4_residual_present"] is False
    assert summary["trace"]["stage3_4_coupling_evidence_present"] is False
