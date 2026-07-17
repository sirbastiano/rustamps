# Getting Started With Rustamps

Rustamps is a standalone Rust implementation of the StaMPS-style processing
pipeline. The production command does not load Python or invoke external
programs at runtime.

## Install with Conda

The `rustamps` 0.3.0 development-channel Conda package installs only the
compiled Rust command. It does not install or load Python, a system HDF5
library, SNAPHU, or another external scientific executable. Create a clean
environment with:

```bash
conda create -n rustamps -c sirbastiano/label/dev -c conda-forge rustamps=0.3.0
conda run -n rustamps rustamps --version
conda run -n rustamps rustamps describe-backends
```

`describe-backends` must report an empty `runtime_external_dependencies` list.
Compiled packages are produced for `linux-64`, `linux-aarch64`, `osx-64`,
`osx-arm64`, `win-64`, and `win-arm64`. Linux packages require glibc 2.17 or
newer, and macOS packages require macOS 11 or newer. They are neither musl nor
`noarch` packages. Windows ARM64 remains experimental and stays on
`sirbastiano/label/dev` until native ARM64 CI passes. Promotion to the `main`
label is a separate, explicit release action.

## Build the native binary from source

Install Rust 1.89 or newer, clone the repository, and build with the locked
dependency graph:

```bash
git clone https://github.com/sirbastiano/rustamps.git
cd rustamps
cargo build --release --locked
cargo install --path . --locked
rustamps describe-backends
```

`describe-backends` should report the standalone `native` provider and an
empty `runtime_external_dependencies` list. Python, a Python package install,
MATLAB, and external triangulation or phase-unwrapping programs are not
production prerequisites.

The supported source-install targets are 64-bit little-endian Linux, macOS,
and Windows on x86_64 or ARM64. Linux GNU and musl are both gated. The portable
CI matrix performs native tests, a release build, `cargo install`, and an
installed-command smoke test for each supported target.

## Prepare a run copy

Always process a copy so the reference inputs remain unchanged:

```bash
export SOURCE_DATASET=/path/to/your_dataset
export RUN_DATASET=/path/to/run_dataset
cp -a "$SOURCE_DATASET" "$RUN_DATASET"

rustamps status --dataset "$RUN_DATASET"
rustamps run --dataset "$RUN_DATASET" --start-step 1 --end-step 8 --dry-run
```

On Windows PowerShell, replace the `export` lines with `$env:NAME = 'value'`
and `cp -a` with `Copy-Item -Recurse`.

For a compatible SNAP export that has not yet been converted into patch
inputs, use the native preparation command first:

```bash
rustamps prep snap --dataset "$RUN_DATASET"
```

Use `rustamps prep snap --help` for master-date, patch-grid, overlap, and
amplitude-dispersion options.

## Run the pipeline

Run all eight stages with:

```bash
rustamps run --dataset "$RUN_DATASET" --start-step 1 --end-step 8
```

An explicit positive stage range recomputes the requested stages and
invalidates dependent later products. To resume automatically from existing
completion artifacts, use `--start-step 0`:

```bash
rustamps run --dataset "$RUN_DATASET" --start-step 0 --end-step 8
```

You can also run a controlled range, for example:

```bash
rustamps run --dataset "$RUN_DATASET" --start-step 6 --end-step 8
```

Stage 6 stores fingerprinted, atomic per-interferogram checkpoints below
`.pystamps-stage6/`. An interrupted solve reuses only checkpoints whose inputs
and solver settings still match; `phuw2.mat` is published only after every
interferogram succeeds.

## Configure native execution

The defaults are native. A compact explicit configuration is:

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

Run with the checked-in strict native profile:

```bash
rustamps --config configs/native-kernels.yaml run \
  --dataset "$RUN_DATASET" --start-step 1 --end-step 8
```

Worker count `0` uses the available CPU budget. Reduce `--cpu-workers` when
memory is constrained. `stage6_ifg_workers: 0` adaptively runs up to four
independent solves subject to Rayon-thread and grid-cell budgets; `1`, `2`, or
`4` is an explicit upper bound. The native Stage 6 solver is always used; legacy
`auto` backend values normalize to native, while external solver selections
are rejected before processing begins.

For a deliberate speed/accuracy trade, use `configs/stage6-balanced.yaml` or
`configs/stage6-fast.yaml`. They use coarser Stage 6 grids while retaining the
converged flow solve, so validate their output scientifically against a strict
run before adopting them for a dataset.

The `stage6-experimental-15x.yaml` and `stage6-experimental-20x.yaml` profiles
contain bounded Stage 6 tolerances measured on the bundled patch. Verify them
with `--final-products-only --through-stage 6`; their Stage 7–8 output is
intentionally not waived.

## Verify results

Strict verification is the default:

```bash
export GOLDEN_DATASET=/path/to/reference_dataset
rustamps verify --run "$RUN_DATASET" --golden "$GOLDEN_DATASET"
```

Use `--through-stage 1` through `--through-stage 8` to limit comparison to a
completed prefix. The scientific profile permits only configured bounded
numeric outliers and still enforces artifact structure and hard error caps:

```bash
rustamps --config configs/stage6-fast.yaml verify \
  --run "$RUN_DATASET" --golden "$GOLDEN_DATASET" \
  --profile scientific --final-products-only --through-stage 6
```

`--final-products-only` excludes grid/cache intermediates only when explicitly
requested; final stage products and their scientific tolerances remain checked.

Command completion alone is not a parity claim; retain the verifier report
with scientific results.

## Developer-only Python oracle

The historical Python implementation is an isolated source oracle for tests
and comparisons. It is not installed as `rustamps`, is not loaded by the Rust
binary, and is not part of production deployment.

```bash
make oracle-setup
make oracle-test
```

Developers may mirror the public comparison dataset with
`make oracle-fetch-insar-dataset`. This target uses the isolated
`oracle/pyproject.toml` environment solely to acquire reference data.

## Stage map

| Stage | Typical intent | Completion artifact |
| --- | --- | --- |
| 1 | Prepare candidate-level patch artifacts | `PATCH_*/ps1.mat` |
| 2 | Compute quality metrics and model terms | `PATCH_*/pm1.mat` |
| 3 | Select persistent candidates | `PATCH_*/select1.mat` |
| 4 | Weed weak or redundant candidates | `PATCH_*/weed1.mat` |
| 5 | Correct phase and merge patches | `ifgstd2.mat` |
| 6 | Unwrap temporal products | `phuw2.mat` |
| 7 | Estimate SCLA correction terms | `scla2.mat` |
| 8 | Apply final space-time filtering | `scn2.mat` |
