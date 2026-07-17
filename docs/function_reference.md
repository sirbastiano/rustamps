# Native Command and Configuration Reference

The supported public interface is the standalone `rustamps` Rust binary. It
has no production Python API or dynamically selected scientific backend.

## Global option

```text
rustamps [--config CONFIG] COMMAND
```

`--config` accepts YAML or JSON. Unknown fields and unsupported legacy values
are rejected during loading.

## Commands

### `rustamps status`

Inspect the available dataset layout and stage completion artifacts without
changing files.

```bash
rustamps status --dataset DATASET
```

### `rustamps run`

Plan or execute a stage range.

```text
rustamps run --dataset DATASET
             [--start-step 0..8]
             [--end-step 1..8]
             [--dry-run]
             [--cpu-workers N]
```

The default range is 1–8. A positive start step recomputes the selected range
and invalidates dependent later products. Start step `0` resumes from existing
valid completion artifacts. `--cpu-workers 0` uses the available CPU budget.

### `rustamps prep snap`

Convert a compatible SNAP export into native patch inputs.

```text
rustamps prep snap --dataset DATASET
  [--master-date YYYYMMDD]
  [--amp-dispersion VALUE]
  [--range-patches N] [--azimuth-patches N]
  [--range-overlap N] [--azimuth-overlap N]
  [--force]
```

Defaults are amplitude dispersion `0.4`, one range patch, one azimuth patch,
and 50-pixel overlaps. Existing output is preserved unless `--force` is used.

### `rustamps verify`

Compare production artifacts with a golden dataset.

```text
rustamps verify --run RUN --golden GOLDEN
  [--profile strict|scientific]
  [--through-stage 1..8]
  [--final-products-only]
```

Strict is the default profile. `--through-stage` limits both expected
artifacts and comparisons to a completed pipeline prefix.
`--final-products-only` explicitly excludes grid/cache and smoothing
intermediates while retaining final products for every included stage.

### Inspection commands

```bash
rustamps describe-backends
rustamps describe-inputs --stage all
rustamps describe-inputs --stage 1 --dataset DATASET --patch PATCH_1
rustamps list-legacy --stamps-root /path/to/StaMPS
```

`describe-backends` reports the standalone native provider and its runtime
dependency boundary. `describe-inputs` lists stage contracts. `list-legacy`
only inventories an existing StaMPS tree; it does not execute its programs.

## Runtime configuration

These runtime fields affect native production behavior:

| Field | Accepted value | Meaning |
| --- | --- | --- |
| `runtime.backend` | `native` or legacy alias `auto` | Native executor |
| `runtime.stage2_kernel_backend` | `native` or `auto` | Native Stage 2 kernels |
| `runtime.stage6_solver` | `native`, `auto`, or `backend` | Native in-process flow solver |
| `runtime.cpu_workers` | non-negative integer | `0` uses the available CPU budget |
| `runtime.stage6_ifg_workers` | `0`, `1`, `2`, or `4` | Adaptive or bounded independent IFG solves |
| `runtime.stage6_grid_scale` | finite number greater than zero | Multiplies Stage 6 grid spacing |
| `runtime.stage6_max_flow_passes` | `0` | Solve integer flow to convergence |

Minimal explicit YAML:

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

External solvers, Python/CUDA providers, reference replay, per-kernel
overrides, and non-default legacy no-op fields fail configuration validation.

## Verification tolerance configuration

`tolerance` controls numeric comparison, not pipeline calculations:

| Field | Purpose |
| --- | --- |
| `profile` | `strict` or `scientific` |
| `rtol`, `atol` | finite non-negative relative and absolute tolerances |
| `wrap_equivalence`, `wrap_period`, `wrap_keys` | cyclic-value comparison policy |
| `exact_keys` | keys that must compare exactly |
| `max_outlier_fraction` | bounded scientific outlier fraction |
| `max_abs` | required hard cap when outliers are allowed |
| `key_tolerances` | per-key numeric policy overrides |

Example scientific policy:

```yaml
tolerance:
  profile: scientific
  rtol: 1.0e-5
  atol: 1.0e-7
  max_outlier_fraction: 0.001
  max_abs: 0.05
  key_tolerances:
    ph_uw:
      atol: 0.001
      max_outlier_fraction: 0.001
      max_abs: 0.05
```

Nonzero outlier fractions are rejected in strict mode and require a finite
`max_abs` hard cap in scientific mode.

## Artifact publication

Stages publish completion artifacts only after their complete output bundle
has been written. Stage 6 additionally uses atomic per-interferogram
checkpoints below `.pystamps-stage6/`; only checkpoints with matching input and
solver fingerprints are reused.

The historical source tree is a development oracle, not a supported runtime
API. Use the explicit `make oracle-*` targets only when reproducing reference
results.
