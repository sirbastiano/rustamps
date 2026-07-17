# pySTAMPS Native Pipeline and Science Guide

pySTAMPS is a standalone Rust implementation of the StaMPS-style single-master
processing workflow. The production binary reads and writes the established
MATLAB artifact layout, but it does not load Python, MATLAB, SNAPHU, Triangle,
or another executable at runtime.

This guide describes the supported native path. The retained historical
implementation is a developer-only oracle and is not a production backend or
fallback.

## Scientific boundary

The dataset directory is the processing state. Patch products live below
`PATCH_*`; merged products live at the dataset root. Run on a writable copy:

```bash
cp -a /path/to/source_dataset /path/to/run_dataset
pystamps status --dataset /path/to/run_dataset
```

MAT values preserve the StaMPS contracts that matter downstream:

- MATLAB artifact dimensions and column-major serialization;
- one-based identifiers at file boundaries and zero-based Rust indices;
- complex phase, sparse, logical, character, typed-empty, NaN, and Inf values;
- zero master and dropped-interferogram columns in `phuw2.mat`;
- transactional publication of each stage completion artifact.

## Pipeline stages

| Stage | Scope | Scientific purpose | Completion artifact |
| --- | --- | --- | --- |
| 1 | patch | Load candidates, phase, baselines, height, and metadata | `PATCH_*/ps1.mat` |
| 2 | patch | Estimate coherence and topographic-error terms | `PATCH_*/pm1.mat` |
| 3 | patch | Select persistent-scatterer candidates | `PATCH_*/select1.mat` |
| 4 | patch | Weed weak, adjacent, or duplicate candidates | `PATCH_*/weed1.mat` |
| 5 | patch/root | Correct phase and merge patches | root `ifgstd2.mat` |
| 6 | root | Unwrap phase with native integer-flow optimization | `phuw2.mat` |
| 7 | root | Estimate spatially correlated look-angle error | `scla2.mat` |
| 8 | root | Apply final space-time noise filtering | `scn2.mat` |

Stages 1–5 process all discovered patches. Stage 5 then creates merged root
products, and Stages 6–8 operate on those merged arrays.

For a compatible SNAP export that has not been prepared, synthesize native
patch inputs first:

```bash
pystamps prep snap --dataset /path/to/run_dataset
```

The preparation command derives wavelength and heading from the master RSLC
metadata and writes them into `parms.mat` when missing. Invalid or incomplete
sensor metadata is rejected rather than replaced with a plausible constant.

## Execute and resume

Rehearse the selected range without writes, then run it:

```bash
pystamps run --dataset /path/to/run_dataset \
  --start-step 1 --end-step 8 --dry-run
pystamps run --dataset /path/to/run_dataset \
  --start-step 1 --end-step 8
```

An explicit positive start step recomputes the selected range and invalidates
dependent later artifacts. Start step `0` resumes from valid completion
artifacts:

```bash
pystamps run --dataset /path/to/run_dataset --start-step 0 --end-step 8
```

Stage 6 additionally writes atomic, fingerprinted per-interferogram
checkpoints below `.pystamps-stage6/`. Changed phase, geometry, baselines,
dates, selection, or solver settings invalidate affected checkpoints.
`phuw2.mat` is published only after every requested interferogram succeeds.

## Native Stage 6 solver

Stage 6 uses the in-process Rust grid solver. It minimizes the StaMPS/SNAPHU-
style integer-flow objective without executing SNAPHU. Flow optimization runs
to convergence; a positive `stage6_max_flow_passes` is rejected because a
bounded solve did not meet scientific validation.

The default grid preserves `unwrap_grid_size`:

```yaml
runtime:
  backend: native
  stage2_kernel_backend: native
  stage6_solver: native
  cpu_workers: 0
  stage6_ifg_workers: 0
  stage6_grid_scale: 1.0
  stage6_max_flow_passes: 0
```

`cpu_workers: 0` uses the available CPU budget. A positive value bounds the
Rayon pool and may reduce peak memory. `stage6_ifg_workers: 0` adaptively uses
up to four independent solves with a nine-million-active-cell budget and three
Rayon threads per solve. Explicit `1`, `2`, or `4` values are safe upper bounds.
This scheduling-only field is excluded from scientific fingerprints.

Stage 6 writes `.pystamps-stage6/timing-v1-<fingerprint>.json` with input,
grid, interpolation, space-time, cost, solve/output, and per-IFG core timings.

### Speed/accuracy profiles

Coarsening the Stage 6 grid is the supported speed/accuracy trade:

- `configs/stage6-balanced.yaml` uses scale `4.0`;
- `configs/stage6-fast.yaml` uses scale `10.0`;
- experimental profiles use scale `15.0` or `20.0` with Stage-6-only bounds;
- all retain a converged flow solve.

A scale multiplies the configured grid spacing. Scale `4.0` therefore reduces
dense grid cells by roughly sixteen times; scale `10.0` reduces them by
roughly one hundred times. This changes spatial sampling, so accept a profile
only after scientific comparison with a strict-grid run for the dataset.
Use `--through-stage 6` for the experimental profiles; do not interpret their
bounded Stage 6 comparison as approval of downstream SCLA or SCN differences.
When the golden tree includes grid caches, also pass `--final-products-only`.

```bash
pystamps --config configs/stage6-fast.yaml run \
  --dataset /path/to/run_dataset --start-step 6 --end-step 8
```

## Verification

Strict verification compares artifact presence, keys, dimensions, types, and
values:

```bash
pystamps verify \
  --run /path/to/run_dataset \
  --golden /path/to/reference_dataset
```

Limit the contract to a completed prefix with `--through-stage 1` through
`--through-stage 8`. The scientific profile permits only configured bounded
numeric outliers and still enforces hard error caps and structural equality:

```bash
pystamps --config configs/stage6-fast.yaml verify \
  --run /path/to/run_dataset \
  --golden /path/to/reference_dataset \
  --profile scientific --final-products-only --through-stage 6
```

Coarser-grid checks use `--final-products-only` so expected cache-shape changes
do not hide or replace the final `phuw2.mat` comparison.

Wrapped equivalence is used only for configured cyclic keys. Unwrapped phase
is never wrapped merely to make a comparison pass. Command completion by
itself is not evidence of scientific parity; retain the verifier report with
the run.

## Fail-closed compatibility

Unsupported branches fail before writing scientifically different stand-ins.
The current boundary includes small-baseline processing, an external Stage 6
solver, Stage 7 L1/tropospheric/legacy APS alternatives, non-degree-1
deramping, and Stage 8 kriging substitution. See
[`native_runtime.md`](native_runtime.md) for the detailed list.

Legacy `auto` values normalize to native. Python, CUDA, external-solver,
per-kernel override, and reference-replay configuration values are rejected.
Use `pystamps verify` to compare a run with an independently produced oracle.

## Developer oracle

Reference reproduction is isolated under `oracle/` and explicit `make
oracle-*` targets. It may require development-only tools, but none are loaded,
installed, or spawned by the production Rust binary.

```bash
make oracle-setup
make oracle-test
```

For installation and the shortest production workflow, see
[`getting_started.md`](getting_started.md). For crate boundaries and checkpoint
publication, see [`architecture.md`](architecture.md).
