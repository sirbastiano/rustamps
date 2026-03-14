# ExecPlan

## Goal
- Stabilize the single supported audit driver so the parity validation loop fails fast on missing datasets and always emits a deterministic machine-readable result contract.

## Scope / Non-goals
- In scope: `scripts/validate_audit.py`, parity-contract metadata consumed by that script, focused regression tests, and minimal operational docs tied directly to the supported audit entrypoint.
- Out of scope: fixing parity mismatches inside workflow outputs, changing the verify comparator, or broad CLI/package refactors beyond what the audit contract requires.

## Invariants and contracts to preserve
- `scripts/validate_audit.py` remains the supported full-validation audit entrypoint referenced by the contract and docs.
- The required datasets remain `inputs_and_outputs/InSAR_dataset_test_stage8diag` and `inputs_and_outputs/InSAR_dataset_test`.
- The audit output must stay JSON-serializable, written to `--output` when requested, and exit non-zero on missing datasets or failed workflows.
- The parity contract remains the source of truth for required datasets, workflows, and canonical audit artifact metadata.

## Files / layers likely to change
- `scripts/validate_audit.py`
- `pystamps/parity_contract.py`
- `tests/test_parity_contract.py`
- `tests/` audit-focused regression coverage
- `README.md`

## Ordered steps
1. Confirm the current audit entrypoint, required datasets, and workflow metadata in `scripts/validate_audit.py`, `pystamps/parity_contract.py`, and README references.
2. Tighten the audit driver so it validates required datasets before running verification, records structured workflow/audit results, and marks completion/interruption state explicitly.
3. Extend parity-contract metadata only where needed so the supported entrypoint and output artifact are explicit and aligned with the audit driver.
4. Add focused tests for the success contract and missing-dataset fast-fail behavior.
5. Run targeted tests first, then the required repo quality gates for this story, followed by a brief security/performance/regression review before commit.

## Validation plan
- `uv run pytest -q tests/test_parity_contract.py tests/test_validate_audit.py`
- `uv run pytest -q`
- `uv run --with build python -m build --sdist --wheel`
- `uv run --with twine python -m twine check dist/*`
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. uv run python scripts/validate_audit.py --datasets inputs_and_outputs/InSAR_dataset_test_stage8diag inputs_and_outputs/InSAR_dataset_test --output inputs_and_outputs/validation_runs/latest_audit.json`
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. uv run pystamps verify --run <run-copy> --golden ./inputs_and_outputs/InSAR_dataset_test`

## Rollback / recovery
- Revert the audit-driver and parity-contract hunks together. No schema or persistent-state rollback is required beyond deleting a newly written audit JSON artifact.

## Risks / blockers
- The repo already has unrelated uncommitted changes; edits must stay tightly scoped so this story does not absorb earlier work accidentally.
- Full audit and verify commands depend on local datasets that may be absent or partially populated; missing assets must be reported as validation evidence, not hidden.
- The verify gate requires a concrete run-copy path; if no run copy exists in the local dataset tree, that command may remain blocked by missing local artifacts.

## Definition of done
- One supported audit entrypoint is explicit in code/docs and writes a deterministic JSON audit artifact.
- Missing required datasets fail fast with a clear report and non-zero exit.
- The audit result records completion/interruption state plus enough workflow detail to identify failing workflows.
- Focused regression coverage and required validation commands provide evidence for the contract.
