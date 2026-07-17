from pathlib import Path

import pytest

from pystamps.config import RunConfig
from pystamps.pipeline.stages import STAGE_DEFS, _run_patch_stage, run_pipeline
from pystamps.pipeline.types import PipelineContext, StageResult


def _context(root: Path, start_step: int, end_step: int) -> PipelineContext:
    return PipelineContext(
        dataset_root=root,
        run_config=RunConfig(),
        start_step=start_step,
        end_step=end_step,
    )


class _Executor:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> "_Executor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_explicit_patch_stage_reruns_with_existing_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "pm1.mat").write_bytes(b"stale")
    calls: list[int] = []
    monkeypatch.setattr(
        "pystamps.pipeline.stages._run_ported_patch_stage",
        lambda stage_id, *args, **kwargs: calls.append(stage_id) or "recomputed",
    )

    result = _run_patch_stage(STAGE_DEFS[1], tmp_path, _context(tmp_path, 2, 2), patch_count=1)

    assert result.status == "completed"
    assert calls == [2]


def test_pipeline_stops_before_dependent_stage_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch = tmp_path / "PATCH_1"
    patch.mkdir()

    class _Dataset:
        root = tmp_path
        patches = [patch]

    calls: list[int] = []

    def fail_stage(stage, patch_dir, context, patch_count):
        calls.append(stage.stage_id)
        return StageResult(stage.stage_id, "patch", patch_dir.name, "failed", "boom")

    monkeypatch.setattr("pystamps.pipeline.stages.discover_dataset", lambda root: _Dataset())
    monkeypatch.setattr("pystamps.pipeline.stages.HybridExecutor", _Executor)
    monkeypatch.setattr("pystamps.pipeline.stages._run_patch_stage_timed", fail_stage)

    report = run_pipeline(_context(tmp_path, 2, 3))

    assert calls == [2]
    assert len(report.failures) == 1


def test_explicit_merged_stage_forces_recomputation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Dataset:
        root = tmp_path
        patches: list[Path] = []

    force_values: list[bool] = []

    def merged(stage, dataset_root, context, *, force_run=False):
        force_values.append(force_run)
        return StageResult(stage.stage_id, "merged", dataset_root.name, "completed", "ok")

    monkeypatch.setattr("pystamps.pipeline.stages.discover_dataset", lambda root: _Dataset())
    monkeypatch.setattr("pystamps.pipeline.stages.HybridExecutor", _Executor)
    monkeypatch.setattr("pystamps.pipeline.stages._run_merged_stage_timed", merged)

    report = run_pipeline(_context(tmp_path, 6, 6))

    assert not report.failures
    assert force_values == [True]
