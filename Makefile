CARGO ?= cargo
UV ?= uv
UV_ORACLE = $(UV) --project oracle

PARITY_ENV = OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
AUDIT_DATASETS = inputs_and_outputs/InSAR_dataset_test_stage8diag inputs_and_outputs/InSAR_dataset_test inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag inputs_and_outputs/InSAR_dataset_small_baseline_stage7
AUDIT_OUTPUT = inputs_and_outputs/validation_runs/latest_audit.json
HF_DATASET_REPO = mdelgadoblasco/InSAR_dataset_test
HF_DATASET_DEST = inputs_and_outputs/InSAR_dataset_test
HF_DATASET_ARCHIVE =
VERIFY_RUN = inputs_and_outputs/RUN_FULL_GATE_1e10
VERIFY_GOLDEN = inputs_and_outputs/InSAR_dataset_test
BENCHMARK_DATASET = inputs_and_outputs/InSAR_dataset_test_stage8diag

.PHONY: format test build verify oracle-setup oracle-test oracle-fetch-insar-dataset oracle-import-insar-dataset oracle-audit oracle-verify oracle-benchmark oracle-parity-loop

format:
	$(CARGO) fmt --all -- --check

test:
	$(CARGO) test --workspace --locked

build:
	$(CARGO) build --release --locked

verify:
	@test -d "$(VERIFY_RUN)" || { echo "Missing run dataset: $(VERIFY_RUN)"; exit 1; }
	@test -d "$(VERIFY_GOLDEN)" || { echo "Missing golden dataset: $(VERIFY_GOLDEN)"; exit 1; }
	$(CARGO) run --release --locked -- verify --run "$(VERIFY_RUN)" --golden "$(VERIFY_GOLDEN)"

# The remaining targets exercise the source-only Python oracle. They never
# install a Python package and are not part of the production runtime.
oracle-setup:
	$(UV_ORACLE) sync --locked

oracle-test:
	$(PARITY_ENV) $(UV_ORACLE) run --locked pytest -c oracle/pyproject.toml tests

oracle-fetch-insar-dataset:
	PYTHONPATH=. $(UV_ORACLE) run --locked python scripts/download_hf_dataset.py --backend huggingface --repo $(HF_DATASET_REPO) --destination $(HF_DATASET_DEST)

oracle-import-insar-dataset:
	@test -n "$(HF_DATASET_ARCHIVE)" || { echo "Set HF_DATASET_ARCHIVE=/path/to/InSAR_dataset_test archive"; exit 1; }
	PYTHONPATH=. $(UV_ORACLE) run --locked python scripts/import_dataset_archive.py --archive "$(HF_DATASET_ARCHIVE)" --destination "$(HF_DATASET_DEST)" --overwrite

oracle-audit:
	$(PARITY_ENV) $(UV_ORACLE) run --locked python scripts/validate_audit.py \
		--datasets $(AUDIT_DATASETS) \
		--output $(AUDIT_OUTPUT)

oracle-verify:
	@test -d "$(VERIFY_RUN)" || { echo "Missing run dataset: $(VERIFY_RUN)"; exit 1; }
	@test -d "$(VERIFY_GOLDEN)" || { echo "Missing golden dataset: $(VERIFY_GOLDEN)"; exit 1; }
	$(PARITY_ENV) $(UV_ORACLE) run --locked python -m pystamps.cli verify --run "$(VERIFY_RUN)" --golden "$(VERIFY_GOLDEN)"

oracle-benchmark:
	$(PARITY_ENV) $(UV_ORACLE) run --locked python scripts/benchmark_backends.py \
		--dataset $(BENCHMARK_DATASET) \
		--start-step 1 --end-step 8 \
		--repeat 3 --warmup 1

oracle-parity-loop:
	$(PARITY_ENV) $(UV_ORACLE) run --locked python scripts/parity_bug_loop.py \
		--datasets $(AUDIT_DATASETS) \
		--audit-output $(AUDIT_OUTPUT) \
		--output inputs_and_outputs/validation_runs/latest_parity_loop.json
