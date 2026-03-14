from pathlib import Path

import pytest

from pystamps.config import CompatibilityConfig, RunConfig, RuntimeConfig
from pystamps.pipeline.stages import STAGE_DEFS, StageExecutionError, _normalize_backend, _task_kind_for_stage
from pystamps.pipeline.types import PipelineContext
from pystamps.runtime.executor import HybridExecutor


def _ctx(backend: str = "auto", strict_reference: bool = False) -> PipelineContext:
    return PipelineContext(
        dataset_root=Path("."),
        run_config=RunConfig(
            runtime=RuntimeConfig(backend=backend),
            compat=CompatibilityConfig(strict_reference=strict_reference),
        ),
        start_step=1,
        end_step=8,
        dry_run=False,
    )


def test_backend_normalization_aliases() -> None:
    assert _normalize_backend("auto") == "auto"
    assert _normalize_backend("threads") == "threads"
    assert _normalize_backend("thread") == "threads"
    assert _normalize_backend("io") == "threads"
    assert _normalize_backend("processes") == "processes"
    assert _normalize_backend("process") == "processes"
    assert _normalize_backend("cpu") == "processes"
    assert _normalize_backend("gpu") == "gpu"
    assert _normalize_backend("native") == "native"


def test_backend_normalization_rejects_invalid() -> None:
    with pytest.raises(StageExecutionError, match="Unsupported runtime backend"):
        _normalize_backend("bogus")


def test_task_kind_auto_mode() -> None:
    context = _ctx("auto")
    assert _task_kind_for_stage(STAGE_DEFS[0], context, patch_count=4) == "io"
    assert _task_kind_for_stage(STAGE_DEFS[1], context, patch_count=4) == "cpu"
    assert _task_kind_for_stage(STAGE_DEFS[1], context, patch_count=1) == "io"
    assert _task_kind_for_stage(STAGE_DEFS[5], context, patch_count=4) == "io"


def test_task_kind_threads_mode() -> None:
    context = _ctx("threads")
    for stage in STAGE_DEFS:
        assert _task_kind_for_stage(stage, context, patch_count=4) == "io"


def test_task_kind_processes_mode() -> None:
    context = _ctx("processes")
    for stage in STAGE_DEFS:
        assert _task_kind_for_stage(stage, context, patch_count=4) == "cpu"


def test_task_kind_gpu_mode() -> None:
    context = _ctx("gpu")
    for stage in STAGE_DEFS:
        assert _task_kind_for_stage(stage, context, patch_count=4) == "io"


def test_task_kind_native_mode() -> None:
    context = _ctx("native")
    for stage in STAGE_DEFS:
        assert _task_kind_for_stage(stage, context, patch_count=4) == "cpu"


def test_task_kind_strict_reference_forces_io() -> None:
    context = _ctx("processes", strict_reference=True)
    for stage in STAGE_DEFS:
        assert _task_kind_for_stage(stage, context, patch_count=4) == "io"


def test_hybrid_executor_lazily_creates_process_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _FakeFuture:
        def __init__(self, value: object) -> None:
            self._value = value

        def result(self) -> object:
            return self._value

    class _FakeThreadPool:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append("thread_init")

        def submit(self, fn: object, *args: object, **kwargs: object) -> _FakeFuture:
            return _FakeFuture(fn(*args, **kwargs))

        def shutdown(self, **kwargs: object) -> None:
            calls.append("thread_shutdown")

    class _FakeProcessPool:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append("process_init")

        def submit(self, fn: object, *args: object, **kwargs: object) -> _FakeFuture:
            return _FakeFuture(fn(*args, **kwargs))

        def shutdown(self, **kwargs: object) -> None:
            calls.append("process_shutdown")

    monkeypatch.setattr("pystamps.runtime.executor.ThreadPoolExecutor", _FakeThreadPool)
    monkeypatch.setattr("pystamps.runtime.executor.ProcessPoolExecutor", _FakeProcessPool)

    with HybridExecutor(io_workers=2, cpu_workers=2) as executor:
        assert "process_init" not in calls
        assert executor.submit("io", lambda: 1).result() == 1
        assert "process_init" not in calls
        assert executor.submit("cpu", lambda: 2).result() == 2
        assert calls.count("process_init") == 1
