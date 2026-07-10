from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
TESTING_DOC = (REPO_ROOT / "docs" / "testing.html").read_text(encoding="utf-8")
VERIFICATION_DOC = (REPO_ROOT / "docs" / "verification.html").read_text(encoding="utf-8")
RELEASE_DOC = (REPO_ROOT / "docs" / "release.md").read_text(encoding="utf-8")
PARITY_DOC = (REPO_ROOT / "parity.md").read_text(encoding="utf-8")
MANIFEST = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
MAKEFILE = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
PYPROJECT = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
ENVIRONMENT = (REPO_ROOT / "environment.yml").read_text(encoding="utf-8")


def test_readme_documents_fresh_clone_validation_separately() -> None:
    assert "Fresh-clone validation commands:" in README
    assert "uv run pytest -q" in README
    assert "uv run --with build python -m build --sdist --wheel" in README
    assert "uv run --with twine python -m twine check dist/*" in README
    assert "Rust toolchain" in README
    assert "platform wheels for the Rust extension" in README
    assert "uv run python scripts/validate_audit.py \\\n  --datasets inputs_and_outputs/InSAR_dataset_test" not in README
    assert "inputs_and_outputs/InSAR_dataset_test_stage8diag" in README
    assert "inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag" in README
    assert "inputs_and_outputs/InSAR_dataset_small_baseline_stage7" in README
    assert "https://huggingface.co/datasets/mdelgadoblasco/InSAR_dataset_test/tree/main" in README
    assert "optional repo assets" in README


def test_readme_and_makefile_expose_the_same_local_entrypoints() -> None:
    assert "make setup" in README
    assert "make test" in README
    assert "make build" in README
    assert "make twine-check" in README
    assert "make fetch-insar-dataset" in README
    assert "make import-insar-dataset" in README
    assert "make audit" in README
    assert "make native-conda-env-check" in README
    assert "make native-conda-check" in README
    assert "make native-conda-kernel-check" in README
    assert "make native-conda-step-validate" in README
    assert "make native-conda-audit-hf" in README
    assert "make native-conda-stage6-fixture" in README
    assert "make native-conda-audit" in README
    assert "make native-conda-verify" in README
    assert "make parity-loop" in README
    assert "make verify" in README
    assert "make benchmark" in README

    assert (
        ".PHONY: setup test test-impl build twine-check fetch-insar-dataset import-insar-dataset audit native-conda-env-check native-conda-check "
        "native-conda-kernel-check native-conda-step-validate native-conda-audit-hf native-conda-stage6-fixture native-conda-audit native-conda-verify verify benchmark parity-loop"
    ) in MAKEFILE
    assert "PARITY_ENV = OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=." in MAKEFILE
    assert (
        "AUDIT_DATASETS = inputs_and_outputs/InSAR_dataset_test_stage8diag "
        "inputs_and_outputs/InSAR_dataset_test "
        "inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag "
        "inputs_and_outputs/InSAR_dataset_small_baseline_stage7"
    ) in MAKEFILE
    assert "AUDIT_OUTPUT = inputs_and_outputs/validation_runs/latest_audit.json" in MAKEFILE
    assert "HF_DATASET_REPO = mdelgadoblasco/InSAR_dataset_test" in MAKEFILE
    assert "HF_DATASET_DEST = inputs_and_outputs/InSAR_dataset_test" in MAKEFILE
    assert "HF_DATASET_ARCHIVE =" in MAKEFILE
    assert "HF_PYTHON ?= python" in MAKEFILE
    assert "HF_NATIVE_AUDIT_OUTPUT = inputs_and_outputs/validation_runs/latest_native_hf_audit.json" in MAKEFILE
    assert "NATIVE_CONFIG = configs/native-kernels.yaml" in MAKEFILE
    assert "NATIVE_AUDIT_OUTPUT = inputs_and_outputs/validation_runs/latest_native_conda_audit.json" in MAKEFILE
    assert "VERIFY_RUN = inputs_and_outputs/RUN_FULL_GATE_1e10" in MAKEFILE
    assert "VERIFY_GOLDEN = inputs_and_outputs/InSAR_dataset_test" in MAKEFILE
    assert "BENCHMARK_DATASET = inputs_and_outputs/InSAR_dataset_test_stage8diag" in MAKEFILE
    assert "uv sync" in MAKEFILE
    assert "uv run pytest -q" in MAKEFILE
    assert "uv run --with build python -m build --sdist --wheel" in MAKEFILE
    assert "uv run --with twine python -m twine check dist/*" in MAKEFILE
    assert (
        "$(HF_PYTHON) scripts/download_hf_dataset.py --backend huggingface "
        "--repo $(HF_DATASET_REPO) --destination $(HF_DATASET_DEST)"
    ) in MAKEFILE
    assert 'Incomplete Hugging Face dataset: $(HF_DATASET_DEST). Run: make fetch-insar-dataset' in MAKEFILE
    assert "Incomplete required dataset: $$dataset. Expected patch.list or PATCH_1" in MAKEFILE
    assert "Incomplete run dataset: $(VERIFY_RUN). Expected patch.list or PATCH_1" in MAKEFILE
    assert "Incomplete golden dataset: $(VERIFY_GOLDEN). Expected patch.list or PATCH_1" in MAKEFILE
    assert "huggingface_hub" in ENVIRONMENT
    assert 'python scripts/import_dataset_archive.py --archive "$(HF_DATASET_ARCHIVE)" --destination "$(HF_DATASET_DEST)" --overwrite' in MAKEFILE
    assert "uv run python scripts/validate_audit.py" in MAKEFILE
    assert "$(CONDA) run -n pystamps-rust python -c \"import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)\"" in MAKEFILE
    assert "$(CONDA) run -n pystamps-rust cargo check" in MAKEFILE
    assert "$(CONDA) run -n pystamps-rust python -m pystamps.cli describe-backends" in MAKEFILE
    assert "$(CONDA) run -n pystamps-rust python scripts/validate_audit.py" in MAKEFILE
    assert "--config $(NATIVE_CONFIG)" in MAKEFILE
    assert "--datasets $(HF_DATASET_DEST)" in MAKEFILE
    assert "--allow-subset" in MAKEFILE
    assert "uv run python scripts/parity_bug_loop.py" in MAKEFILE
    assert "--datasets $(AUDIT_DATASETS)" in MAKEFILE
    assert "--audit-output $(AUDIT_OUTPUT)" in MAKEFILE
    assert "--output inputs_and_outputs/validation_runs/latest_parity_loop.json" in MAKEFILE
    assert "uv run pystamps verify --run $(VERIFY_RUN) --golden $(VERIFY_GOLDEN)" in MAKEFILE
    assert "uv run python scripts/benchmark_backends.py" in MAKEFILE


def test_release_docs_reference_the_supported_parity_gate() -> None:
    assert "make audit" not in RELEASE_DOC
    assert "make verify" not in RELEASE_DOC
    assert "Do not substitute a Makefile target" in RELEASE_DOC
    assert "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=." in RELEASE_DOC
    assert "inputs_and_outputs/InSAR_dataset_test_stage8diag" in RELEASE_DOC
    assert "inputs_and_outputs/InSAR_dataset_test" in RELEASE_DOC
    assert "inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag" in RELEASE_DOC
    assert "inputs_and_outputs/InSAR_dataset_small_baseline_stage7" in RELEASE_DOC
    assert "--output inputs_and_outputs/validation_runs/latest_audit.json" in RELEASE_DOC
    assert "run_root" in RELEASE_DOC
    assert "manual restart" in RELEASE_DOC
    assert "Rust toolchain" in RELEASE_DOC
    assert "cibuildwheel" in RELEASE_DOC
    assert "platform wheels" in RELEASE_DOC


def test_testing_docs_call_out_optional_dataset_workflows() -> None:
    assert "skip cleanly when the local validation datasets are absent" in TESTING_DOC
    assert "do not guess a Makefile, CI wrapper, or reduced audit dataset list" in TESTING_DOC
    assert "documented audited dataset set" in TESTING_DOC
    assert "uv run --with build python -m build --sdist --wheel" in TESTING_DOC
    assert "uv run --with twine python -m twine check dist/*" in TESTING_DOC
    assert "uv run --with cibuildwheel python -m cibuildwheel --platform" in TESTING_DOC
    assert "uv run pystamps verify --run RUN_COPY --golden ./inputs_and_outputs/InSAR_dataset_test" in TESTING_DOC
    assert "latest_audit.json" in TESTING_DOC
    assert "stale-output reuse keeps the validation gate red" in TESTING_DOC


def test_verification_and_parity_docs_define_the_final_oracle_backed_contract() -> None:
    assert "make audit" in VERIFICATION_DOC
    assert "scripts/validate_audit.py" in VERIFICATION_DOC
    assert "pystamps/data/audited_workflow_manifest.json" in VERIFICATION_DOC
    assert "make parity-loop" in VERIFICATION_DOC
    assert "scripts/parity_bug_loop.py" in VERIFICATION_DOC
    assert "cpp_wrapper" in VERIFICATION_DOC
    assert "matlab_source" in VERIFICATION_DOC
    assert "manual_references" in VERIFICATION_DOC
    assert "inputs_and_outputs/InSAR_dataset_test_stage8diag \\\n      inputs_and_outputs/InSAR_dataset_test" not in VERIFICATION_DOC

    assert "# Oracle-backed parity contract" in PARITY_DOC
    assert "pystamps/data/oracle_contract.json" in PARITY_DOC
    assert "pystamps/data/audited_workflow_manifest.json" in PARITY_DOC
    assert "RUN_FULL_GATE_1e10" in PARITY_DOC
    assert "legacy_post" in PARITY_DOC
    assert "small_baseline" in PARITY_DOC
    assert "latest_audit.json" in PARITY_DOC
    assert "latest_parity_loop.json" in PARITY_DOC
    assert "Status: blocked by environment prerequisites." not in PARITY_DOC
    assert "triangle:missing" not in PARITY_DOC
    assert "snaphu:missing" not in PARITY_DOC


def test_manifest_excludes_generated_release_artifacts() -> None:
    assert "prune dist" in MANIFEST
    assert "prune build" in MANIFEST
    assert "prune .codex" in MANIFEST
    assert "prune .github" in MANIFEST


def test_packaging_contract_prefers_rust_sources_and_excludes_cython_package_data() -> None:
    assert "include Cargo.toml" in MANIFEST
    assert "recursive-include src *.rs" in MANIFEST
    assert "recursive-include pystamps/data *.json" in MANIFEST
    assert "recursive-include pystamps *.pyx" not in MANIFEST
    assert "include-package-data = false" in PYPROJECT
    assert '[tool.setuptools.package-data]\npystamps = ["data/*.json"]' in PYPROJECT
    assert "setuptools-rust>=1.10" in PYPROJECT
