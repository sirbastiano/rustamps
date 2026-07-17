# Architecture Snapshot

pySTAMPS has one production boundary: a standalone Rust binary reads and
writes a StaMPS-style dataset directory. That directory, including its MATLAB
artifacts and stage markers, is the source of truth.

For scientific details and current fail-closed compatibility boundaries, see
[native_runtime.md](native_runtime.md).

## Crate boundaries

| Layer | Rust crate or path | Responsibility |
| --- | --- | --- |
| CLI | `crates/pystamps-cli` | Parse commands and print JSON reports |
| Pipeline | `crates/pystamps-pipeline` | Load configuration, schedule stages, invalidate downstream products, and commit outputs |
| Algorithms | `crates/pystamps-core` | Native numerical kernels for preparation and Stages 1–8 |
| Dataset I/O | `crates/pystamps-io` | Discover layouts and read/write MATLAB-compatible artifacts |
| Verification | `crates/pystamps-verify` | Compare production artifacts under strict or scientific tolerances |

The root Cargo package assembles these crates into the `pystamps` binary. It
has no Python extension boundary and does not spawn an external scientific
tool. Pure-Rust MAT v5/v7.3 handling keeps system HDF5 outside the runtime
dependency graph.

## Pipeline model

Stages mirror the StaMPS stage range 1 through 8.

| Stage | Scope | Intent | Completion artifact |
| --- | --- | --- | --- |
| 1 | patch | Initial load | `PATCH_*/ps1.mat` |
| 2 | patch | Estimate gamma | `PATCH_*/pm1.mat` |
| 3 | patch | Select PS pixels | `PATCH_*/select1.mat` |
| 4 | patch | Weed adjacent pixels | `PATCH_*/weed1.mat` |
| 5 | patch and merged | Correct phase and merge | `PATCH_*/ph2.mat`, root `ifgstd2.mat` |
| 6 | merged | Unwrap phase | root `phuw2.mat` |
| 7 | merged | Calculate SCLA | root `scla2.mat` |
| 8 | merged | Filter SCN | root `scn2.mat` |

Stages 1–5 execute for every discovered `PATCH_*` directory. Stage 5 then
aggregates patch products at the dataset root. Stages 6–8 operate on merged
artifacts.

## Scheduling and publication

`pystamps-pipeline` discovers the dataset once and dispatches every selected
stage to `NativeExecutor`. An explicit positive stage range recomputes the
range. `start_step: 0` is resume mode and reports `skipped_existing` for a
complete stage.

Before recomputation, downstream products are invalidated so an old success
marker cannot survive changed upstream data. A stage transaction publishes
its completion artifact only after all outputs are written successfully.

The normal result statuses are:

| Status | Meaning |
| --- | --- |
| `planned` | Dry-run selected the stage without writing |
| `completed` | Native execution committed the complete output bundle |
| `skipped_existing` | Resume mode found the completion artifact |
| `failed` | Execution or publication failed |

## Native configuration

The production configuration accepts only native execution. `auto` remains a
compatibility alias and normalizes to `native`; Python, CUDA, and external
solver values fail during configuration loading.

```yaml
runtime:
  backend: native
  stage2_kernel_backend: native
  stage6_solver: native
  cpu_workers: 0
  stage6_grid_scale: 1.0
  stage6_max_flow_passes: 0
```

`cpu_workers: 0` uses the detected Rayon CPU budget. A positive value creates
a bounded pool. Large MAT-file reads are intentionally conservative to limit
peak memory.

`stage6_grid_scale: 1.0` preserves the configured unwrap grid. Values above
one trade spatial resolution for fewer flow cells. A zero
`stage6_max_flow_passes` solves to convergence. Both values are part of the
Stage 6 scientific checkpoint fingerprint; shipped profiles leave the flow
solve converged.

Inspect the compiled runtime boundary with:

```bash
pystamps describe-backends
```

## Stage 6 checkpoint flow

Stage 6 separates reusable intermediate state from the final artifact:

1. Fingerprinted grid, interpolation, and space-time caches are loaded or
   rebuilt from current scientific inputs.
2. Each non-master solve interferogram is unwrapped natively and written as an
   atomic checkpoint below `.pystamps-stage6/`.
3. Invalid, corrupt, or mismatched checkpoints are recomputed.
4. Only after every solve succeeds are the merged products committed and
   `phuw2.mat` published.

This supports interruption and bounded-memory parallel completion without
allowing a partial Stage 6 result to look complete.

## Verification architecture

The verifier compares artifact presence, exact structural keys, dimensions,
types, and values:

```bash
pystamps verify --run RUN_DIR --golden GOLDEN_DIR
pystamps verify --run RUN_DIR --golden GOLDEN_DIR --profile scientific
```

Strict mode is the default. Scientific mode applies configured tolerances,
bounded outlier fractions, hard maximum-error caps, and wrapping only to keys
whose scientific contract is cyclic. Unwrapped phase is never wrapped merely
to make a comparison pass. `--through-stage N` limits the artifact contract
to a completed pipeline prefix.

## Development oracle boundary

The source Python tree and `oracle/pyproject.toml` exist only to reproduce the
historical implementation during development. They are not a production
package, backend, or fallback. Explicit `make oracle-*` targets own those
workflows; the native binary never imports them.
