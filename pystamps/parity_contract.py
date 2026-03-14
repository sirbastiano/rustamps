from __future__ import annotations

from pathlib import Path
from typing import Any

from pystamps.io.dataset import discover_dataset


DEFAULT_REQUIRED_DATASETS: tuple[str, ...] = (
    "inputs_and_outputs/InSAR_dataset_test_stage8diag",
    "inputs_and_outputs/InSAR_dataset_test",
)

STAGE1_VERIFY_PATTERNS: tuple[str, ...] = (
    "PATCH_*/ps1.mat",
    "PATCH_*/ph1.mat",
    "PATCH_*/bp1.mat",
    "PATCH_*/da1.mat",
    "PATCH_*/hgt1.mat",
)

STAGE25_VERIFY_PATTERNS: tuple[str, ...] = (
    "PATCH_*/pm1.mat",
    "PATCH_*/select1.mat",
    "PATCH_*/weed1.mat",
    "PATCH_*/ps2.mat",
    "PATCH_*/ph2.mat",
    "PATCH_*/pm2.mat",
    "PATCH_*/bp2.mat",
    "PATCH_*/hgt2.mat",
    "PATCH_*/la2.mat",
    "PATCH_*/rc2.mat",
    "PATCH_*/psver.mat",
    "ps2.mat",
    "ph2.mat",
    "pm2.mat",
    "bp2.mat",
    "hgt2.mat",
    "la2.mat",
    "rc2.mat",
    "psver.mat",
)
STAGE2_VERIFY_PATTERNS: tuple[str, ...] = ("PATCH_*/pm1.mat",)
STAGE3_VERIFY_PATTERNS: tuple[str, ...] = ("PATCH_*/select1.mat",)
STAGE4_VERIFY_PATTERNS: tuple[str, ...] = ("PATCH_*/weed1.mat",)

STAGE6_VERIFY_PATTERNS: tuple[str, ...] = (
    "ifgstd2.mat",
    "phuw2.mat",
    "uw_phaseuw.mat",
    "uw_grid.mat",
    "uw_interp.mat",
)

STAGE68_VERIFY_PATTERNS: tuple[str, ...] = STAGE6_VERIFY_PATTERNS + (
    "scla2.mat",
    "scla_smooth2.mat",
    "mean_v.mat",
    "mv2.mat",
    "uw_space_time.mat",
)

STAGE25_CLEAN_PATTERNS = STAGE25_VERIFY_PATTERNS
STAGE2_CLEAN_PATTERNS = STAGE2_VERIFY_PATTERNS
STAGE3_CLEAN_PATTERNS = STAGE3_VERIFY_PATTERNS
STAGE4_CLEAN_PATTERNS = STAGE4_VERIFY_PATTERNS
STAGE6_CLEAN_PATTERNS = STAGE6_VERIFY_PATTERNS
STAGE68_CLEAN_PATTERNS = STAGE68_VERIFY_PATTERNS
STAGE28_CLEAN_PATTERNS = STAGE25_CLEAN_PATTERNS + STAGE68_CLEAN_PATTERNS
FULL_CLEAN_PATTERNS = STAGE1_VERIFY_PATTERNS + STAGE28_CLEAN_PATTERNS
STAGE78_VERIFY_PATTERNS: tuple[str, ...] = (
    "scla2.mat",
    "scla_smooth2.mat",
    "mean_v.mat",
    "mv2.mat",
    "uw_space_time.mat",
)
STAGE78_CLEAN_PATTERNS = STAGE78_VERIFY_PATTERNS
FULL_VERIFY_PATTERNS = STAGE1_VERIFY_PATTERNS + STAGE25_VERIFY_PATTERNS + STAGE68_VERIFY_PATTERNS

REQUIRED_WORKFLOWS: dict[str, dict[str, Any]] = {
    "stage2_only": {
        "kind": "dataset_workflow",
        "audit_name_suffix": "stage2_only",
        "start_step": 2,
        "end_step": 2,
        "verify_patterns": list(STAGE2_VERIFY_PATTERNS),
    },
    "stage3_only": {
        "kind": "dataset_workflow",
        "audit_name_suffix": "stage3_only",
        "start_step": 3,
        "end_step": 3,
        "verify_patterns": list(STAGE3_VERIFY_PATTERNS),
    },
    "stage4_only": {
        "kind": "dataset_workflow",
        "audit_name_suffix": "stage4_only",
        "start_step": 4,
        "end_step": 4,
        "verify_patterns": list(STAGE4_VERIFY_PATTERNS),
    },
    "stage2_5": {
        "kind": "dataset_workflow",
        "audit_name_suffix": "stage2_5",
        "start_step": 2,
        "end_step": 5,
        "verify_patterns": list(STAGE25_VERIFY_PATTERNS),
    },
    "stage6_8": {
        "kind": "dataset_workflow",
        "audit_name_suffix": "stage6_8",
        "start_step": 6,
        "end_step": 8,
        "verify_patterns": list(STAGE68_VERIFY_PATTERNS),
    },
    "full_validation": {
        "kind": "campaign",
        "driver": "scripts/validate_audit.py",
        "output_artifact": "inputs_and_outputs/validation_runs/latest_audit.json",
        "required_dataset_names": [Path(path).name for path in DEFAULT_REQUIRED_DATASETS],
        "includes_dataset_workflow_suffixes": [
            "stage2_only",
            "stage3_only",
            "stage4_only",
            "stage2_5",
            "stage6_8",
        ],
    },
    "pytest_smoke": {
        "kind": "repo_workflow",
        "command": "uv run pytest -q",
    },
}

CANONICAL_ARTIFACTS: dict[str, dict[str, Any]] = {
    "stage2_weighting_snapshot": {
        "kind": "stage2_debug_snapshot",
        "path": "inputs_and_outputs/validation_runs/stage2_weighting_snapshot.json",
        "source_patch": "PATCH_1",
        "required_fields": [
            "Nr",
            "Na",
            "low_coh_thresh",
            "Nr_max_nz_ix",
            "coh_ps",
            "prand",
            "prand_hi",
            "prand_ps",
            "weighting",
        ],
    },
    "stage2_weighting_oracle": {
        "kind": "stage2_weighting_oracle",
        "path": "inputs_and_outputs/validation_runs/stage2_weighting_oracle.json",
        "source_artifact": "stage2_weighting_snapshot",
        "required_outputs": [
            "prand",
            "prand_hi",
            "prand_ps",
            "weighting",
        ],
    },
    "stage2_weighting_compare": {
        "kind": "stage2_weighting_compare",
        "path": "inputs_and_outputs/validation_runs/stage2_weighting_compare.json",
        "driver": "scripts/stage2_weighting_harness.py",
        "snapshot_artifact": "stage2_weighting_snapshot",
        "oracle_artifact": "stage2_weighting_oracle",
        "expected_max_abs": 0.0,
    },
}


def _is_dataset_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "patch.list").exists():
        return True
    return any(child.is_dir() and child.name.startswith("PATCH_") for child in path.iterdir())


def discover_golden_datasets(inputs_root: str | Path) -> list[Path]:
    root = Path(inputs_root).expanduser().resolve()
    if not root.exists():
        return []
    datasets = [path for path in sorted(root.iterdir(), key=lambda item: item.name) if _is_dataset_dir(path)]
    return datasets


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    return str(path.relative_to(repo_root)).replace("\\", "/")


def _dataset_payload(dataset: Path, repo_root: Path) -> dict[str, Any]:
    layout = discover_dataset(dataset)
    discovery_signals: list[str] = []
    if layout.patch_list_file is not None:
        discovery_signals.append("patch.list")
    if layout.patches:
        discovery_signals.append("PATCH_*")
    return {
        "name": dataset.name,
        "path": _relative_to_repo(dataset, repo_root),
        "required": _relative_to_repo(dataset, repo_root) in DEFAULT_REQUIRED_DATASETS,
        "patch_count": len(layout.patches),
        "patches": [patch.name for patch in layout.patches],
        "has_patch_list": layout.patch_list_file is not None,
        "discovery_signals": discovery_signals,
    }


def build_parity_contract(inputs_root: str | Path) -> dict[str, Any]:
    root = Path(inputs_root).expanduser().resolve()
    repo_root = root.parent
    datasets = discover_golden_datasets(root)
    return {
        "contract_version": 1,
        "inputs_root": _relative_to_repo(root, repo_root),
        "dataset_discovery_rule": "Immediate child directories of inputs_root with patch.list or at least one PATCH_* directory.",
        "required_dataset_names": [Path(path).name for path in DEFAULT_REQUIRED_DATASETS],
        "required_dataset_paths": list(DEFAULT_REQUIRED_DATASETS),
        "required_workflow_names": list(REQUIRED_WORKFLOWS),
        "artifacts": CANONICAL_ARTIFACTS,
        "datasets": [_dataset_payload(dataset, repo_root) for dataset in datasets],
        "workflows": REQUIRED_WORKFLOWS,
    }
