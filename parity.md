# Oracle-backed parity contract

This repository keeps parity claims tied to explicit oracle metadata and audited workflow manifests, not to ad hoc notebook output.

Primary contract files:

- `pystamps/data/oracle_contract.json`
- `pystamps/data/audited_workflow_manifest.json`

Required local parity artifacts are repository-local and may be absent from a fresh clone. The supported verification run root for the full gate is `inputs_and_outputs/RUN_FULL_GATE_1e10`.

Audit outputs:

- `inputs_and_outputs/validation_runs/latest_audit.json`
- `inputs_and_outputs/validation_runs/latest_parity_loop.json`

Oracle classes include `legacy_post`, `small_baseline`, maintained manual references, MATLAB/StaMPS source outputs, and wrapper outputs where declared by the JSON contract files.

Use the documented audit and verify commands in `docs/verification.html` and `docs/release.md` for release-grade parity evidence.
