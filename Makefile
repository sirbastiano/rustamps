PARITY_ENV = OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
AUDIT_DATASETS = inputs_and_outputs/InSAR_dataset_test_stage8diag inputs_and_outputs/InSAR_dataset_test inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag inputs_and_outputs/InSAR_dataset_small_baseline_stage7
AUDIT_OUTPUT = inputs_and_outputs/validation_runs/latest_audit.json
HF_DATASET_REPO = mdelgadoblasco/InSAR_dataset_test
HF_DATASET_DEST = inputs_and_outputs/InSAR_dataset_test
HF_DATASET_ARCHIVE =
HF_PYTHON ?= python
CONDA ?= conda
HF_NATIVE_AUDIT_OUTPUT = inputs_and_outputs/validation_runs/latest_native_hf_audit.json
NATIVE_CONFIG = configs/native-kernels.yaml
NATIVE_AUDIT_OUTPUT = inputs_and_outputs/validation_runs/latest_native_conda_audit.json
NATIVE_STEP_VALIDATION_OUTPUT = inputs_and_outputs/validation_runs/native_conda_step_validation_latest.json
NATIVE_STEPS =
NATIVE_STAGE2_CACHE = inputs_and_outputs/validation_runs/stage2_random_hist_cache
STAGE6_THREADS ?= 0
STAGE6_FIXTURE_ROOT ?= $(HF_DATASET_DEST)
VERIFY_RUN = inputs_and_outputs/RUN_FULL_GATE_1e10
VERIFY_GOLDEN = inputs_and_outputs/InSAR_dataset_test
BENCHMARK_DATASET = inputs_and_outputs/InSAR_dataset_test_stage8diag

.PHONY: setup test test-impl build twine-check fetch-insar-dataset import-insar-dataset audit native-conda-env-check native-conda-check native-conda-kernel-check native-conda-step-validate native-conda-audit-hf native-conda-stage6-fixture native-conda-audit native-conda-verify verify benchmark parity-loop

setup:
	uv sync

test:
	uv run pytest -q

test-impl:
	uv run pytest -q tests/test_cli.py tests/test_verify.py tests/test_validate_audit.py tests/test_stage7_ported.py tests/test_kernels_accelerated.py tests/test_dataset.py

build:
	uv run --with build python -m build --sdist --wheel

twine-check:
	uv run --with twine python -m twine check dist/*

fetch-insar-dataset:
	$(HF_PYTHON) scripts/download_hf_dataset.py --backend huggingface --repo $(HF_DATASET_REPO) --destination $(HF_DATASET_DEST)

import-insar-dataset:
	@test -n "$(HF_DATASET_ARCHIVE)" || { echo "Set HF_DATASET_ARCHIVE=/path/to/InSAR_dataset_test archive"; exit 1; }
	python scripts/import_dataset_archive.py --archive "$(HF_DATASET_ARCHIVE)" --destination "$(HF_DATASET_DEST)" --overwrite

audit:
	$(PARITY_ENV) uv run python scripts/validate_audit.py \
		--datasets $(AUDIT_DATASETS) \
		--output $(AUDIT_OUTPUT)

native-conda-env-check:
	$(CONDA) run -n pystamps-rust python -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"
	$(CONDA) run -n pystamps-rust python -m pystamps.cli describe-backends

native-conda-check:
	$(CONDA) run -n pystamps-rust cargo check
	$(CONDA) run -n pystamps-rust python -m pystamps.cli describe-backends

native-conda-kernel-check:
	$(CONDA) run -n pystamps-rust cargo fmt --check
	$(CONDA) run -n pystamps-rust cargo test --lib
	$(CONDA) run -n pystamps-rust python setup.py build_ext --inplace
	PYTHONPATH=. $(CONDA) run -n pystamps-rust python -c "import pystamps.kernels._stage2_native as native; assert hasattr(native, 'stage6_unwrap_grid'); print('stage6_unwrap_grid available')"

native-conda-step-validate:
	$(PARITY_ENV) $(CONDA) run -n pystamps-rust python scripts/native_conda_step_validate.py --output $(NATIVE_STEP_VALIDATION_OUTPUT) $(foreach step,$(NATIVE_STEPS),--step $(step))

native-conda-audit:
	@for dataset in $(AUDIT_DATASETS); do \
		test -d "$$dataset" || { echo "Missing required dataset: $$dataset"; exit 1; }; \
		test -f "$$dataset/patch.list" -o -d "$$dataset/PATCH_1" || { echo "Incomplete required dataset: $$dataset. Expected patch.list or PATCH_1"; exit 1; }; \
	done
	$(PARITY_ENV) PYSTAMPS_STAGE2_RANDOM_HIST_CACHE=$(NATIVE_STAGE2_CACHE) $(CONDA) run -n pystamps-rust python scripts/validate_audit.py \
		--config $(NATIVE_CONFIG) \
		--datasets $(AUDIT_DATASETS) \
		--output $(NATIVE_AUDIT_OUTPUT)

native-conda-audit-hf:
	@test -d "$(HF_DATASET_DEST)" || { echo "Missing Hugging Face dataset: $(HF_DATASET_DEST). Run: make fetch-insar-dataset"; exit 1; }
	@test -f "$(HF_DATASET_DEST)/patch.list" -o -d "$(HF_DATASET_DEST)/PATCH_1" || { echo "Incomplete Hugging Face dataset: $(HF_DATASET_DEST). Run: make fetch-insar-dataset"; exit 1; }
	$(PARITY_ENV) PYSTAMPS_STAGE2_RANDOM_HIST_CACHE=$(NATIVE_STAGE2_CACHE) $(CONDA) run -n pystamps-rust python scripts/validate_audit.py \
		--config $(NATIVE_CONFIG) \
		--datasets $(HF_DATASET_DEST) \
		--allow-subset \
		--output $(HF_NATIVE_AUDIT_OUTPUT)

native-conda-stage6-fixture:
	@test -f "$(STAGE6_FIXTURE_ROOT)/uw_grid.mat" -a -f "$(STAGE6_FIXTURE_ROOT)/snaphu.in" -a -f "$(STAGE6_FIXTURE_ROOT)/snaphu.costinfile" -a -f "$(STAGE6_FIXTURE_ROOT)/snaphu.out" || { echo "Missing Stage 6 fixture files in $(STAGE6_FIXTURE_ROOT). Run: make fetch-insar-dataset or set STAGE6_FIXTURE_ROOT=inputs_and_outputs/validation_runs/stage6_fixture_minimal"; exit 1; }
	$(PARITY_ENV) $(CONDA) run -n pystamps-rust python scripts/stage6_hf_diagnostics.py --root $(STAGE6_FIXTURE_ROOT) --threads $(STAGE6_THREADS)

native-conda-verify:
	@test -d "$(VERIFY_RUN)" || { echo "Missing run dataset: $(VERIFY_RUN)"; exit 1; }
	@test -d "$(VERIFY_GOLDEN)" || { echo "Missing golden dataset: $(VERIFY_GOLDEN)"; exit 1; }
	@test -f "$(VERIFY_RUN)/patch.list" -o -d "$(VERIFY_RUN)/PATCH_1" || { echo "Incomplete run dataset: $(VERIFY_RUN). Expected patch.list or PATCH_1"; exit 1; }
	@test -f "$(VERIFY_GOLDEN)/patch.list" -o -d "$(VERIFY_GOLDEN)/PATCH_1" || { echo "Incomplete golden dataset: $(VERIFY_GOLDEN). Expected patch.list or PATCH_1"; exit 1; }
	$(PARITY_ENV) $(CONDA) run -n pystamps-rust python -m pystamps.cli verify --run $(VERIFY_RUN) --golden $(VERIFY_GOLDEN)

verify:
	$(PARITY_ENV) uv run pystamps verify --run $(VERIFY_RUN) --golden $(VERIFY_GOLDEN)

benchmark:
	uv run python scripts/benchmark_backends.py \
		--dataset $(BENCHMARK_DATASET) \
		--start-step 1 --end-step 8 \
		--repeat 3 --warmup 1

parity-loop:
	$(PARITY_ENV) uv run python scripts/parity_bug_loop.py \
		--datasets $(AUDIT_DATASETS) \
		--audit-output $(AUDIT_OUTPUT) \
		--output inputs_and_outputs/validation_runs/latest_parity_loop.json
