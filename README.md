<div align="center">

<img src="docs/assets/pystamps-logo.svg" alt="pySTAMPS" style="width: 200px; height: auto; max-width: 100%;" />

# pySTAMPS

Standalone Rust implementation of the StaMPS persistent-scatterer workflow.

</div>

The production runtime covers SNAP preparation, StaMPS-compatible Stages 1–8,
MAT-file I/O, dataset status, and numerical verification. It does not invoke
Python, MATLAB, SNAPHU, Triangle, or other external executables. The historical
Python and PyO3 sources remain in the repository only as an audit oracle; Cargo
does not compile or install them.

## Install

### Conda package

The unpublished `pystamps` 0.2.0 Conda package contains the native Rust CLI only;
it does not install Python, a system HDF5 library, SNAPHU, or another scientific
executable. No Conda artifact has been uploaded yet. After the first validated
development-channel release, install and smoke-test it in a clean environment:

```bash
conda create -n pystamps -c sirbastiano/label/dev -c conda-forge pystamps=0.2.0
conda run -n pystamps pystamps --version
conda run -n pystamps pystamps describe-backends
```

The compiled package targets `linux-64`, `linux-aarch64`, `osx-64`,
`osx-arm64`, `win-64`, and `win-arm64`. Conda's Linux packages require glibc
2.17 or newer, and the macOS packages require macOS 11 or newer. The
source-install matrix remains the installation route for Linux musl. This is
not a `noarch` package. Windows ARM64 remains experimental and on the `dev`
label until its artifact passes native ARM64 CI. Promotion of the exact tested
artifacts to the `main` label requires an explicit release decision.

### Build from source

Install Rust 1.89 or newer, then build and install directly from the repository:

```bash
git clone https://github.com/sirbastiano/pystamps.git
cd pystamps
cargo install --path . --locked
pystamps --help
```

For a checkout-local release build:

```bash
cargo build --release --locked
cargo run --release --locked -- --help
```

No Python environment or system HDF5 library is required. The Rust dependency
graph contains the numerical, FFT, MAT v5/v7.3, and parallel-processing crates
used by the binary.

Source installation is gated on 64-bit little-endian Linux (GNU and musl),
macOS, and Windows, on both x86_64 and ARM64. The native matrix in
[portable-rust.yml](.github/workflows/portable-rust.yml) builds, tests, installs,
and smoke-tests each target with the minimum supported Rust release. Other
architectures, BSD, mobile, and WebAssembly are not currently supported.

## Prepare SNAP input

Run preparation on a writeable dataset directory containing the SNAP-exported
rasters and metadata:

```bash
pystamps prep snap \
  --dataset /path/to/dataset \
  --amp-dispersion 0.4 \
  --range-patches 1 \
  --azimuth-patches 1
```

Use `--master-date YYYYMMDD` when the master cannot be inferred and `--force`
to replace an existing prepared layout.

## Run the pipeline

Always work on a copy when comparing with a reference result:

```bash
cp -a /path/to/source_dataset /path/to/run_dataset
pystamps status --dataset /path/to/run_dataset
pystamps run --dataset /path/to/run_dataset --start-step 1 --end-step 8
```

The example uses POSIX `cp`; in PowerShell, use
`Copy-Item -Recurse SOURCE_DATASET RUN_DATASET`.

Run any contiguous stage range by changing `--start-step` and `--end-step`.
Preview the planned writes with `--dry-run`:

```bash
pystamps run \
  --dataset /path/to/run_dataset \
  --start-step 3 \
  --end-step 5 \
  --dry-run
```

The default configuration is native-only. An explicit YAML file can tune worker
counts and scientific tolerances:

```bash
pystamps --config configs/native-kernels.yaml run \
  --dataset /path/to/run_dataset \
  --start-step 1 \
  --end-step 8
```

`--cpu-workers 0` uses the available Rayon threads. Restrict it when memory or
shared-machine load matters. MAT reads are intentionally conservative and do
not expose a separate I/O-worker pool.

`runtime.stage6_ifg_workers` controls independent Stage 6 solves: `0` selects
an adaptive, cell-budgeted schedule, while `1`, `2`, or `4` sets an upper
bound. Scheduling does not alter scientific checkpoint fingerprints.

## Verify scientific output

Compare a run tree with a retained golden tree:

```bash
pystamps verify \
  --run /path/to/run_dataset \
  --golden /path/to/golden_dataset
```

Verification handles real and complex arrays, wrapped phase, NaN/Inf, sparse
arrays, and character data. It reports every failed artifact and exits nonzero
when tolerances are exceeded.

The default compares every production artifact present in the golden tree. For
a stage-scoped run, use `--through-stage 6` (or another stage in `1..8`) to
exclude later-stage golden products without silently intersecting the trees.
Add `--final-products-only` explicitly when comparing a coarser grid: strict
verification still includes grid/cache intermediates by default.

Explicit reruns invalidate later-stage products before processing. Stage 2 and
Stage 6 caches are fingerprinted from their complete scientific inputs, so a
same-size dataset or an unchanged baseline span cannot reuse stale results.

The native Stage 6 solver is self-contained. It follows the same integer-flow
scientific model without delegating to SNAPHU; small floating-point or solver
path differences should be evaluated through the verifier rather than by file
hash alone.

High-quality Stage 6 flow optimization remains expensive on very large stacks.
Its preprocessing and per-interferogram solve checkpoints are reusable,
fingerprinted, checksummed, and atomically written. The final `phuw2.mat` is
still published transactionally only after every interferogram finishes.
Each run also writes a machine-readable phase and per-IFG timing report below
`.pystamps-stage6/`.
`configs/stage6-balanced.yaml` and `configs/stage6-fast.yaml` provide explicit
coarser-grid profiles for users who prefer a large speed gain. Both retain a
converged flow solve, and their results should be verified against the strict
default before adoption for a dataset.

The measured `stage6-experimental-15x.yaml` and
`stage6-experimental-20x.yaml` profiles are more aggressive and carry
Stage-6-only bounds. Use `--through-stage 6` when checking those bounds; later
stages remain strict and must be assessed separately. Pass
`--final-products-only` when the golden tree contains grid intermediates.

The validated production path is the available single-master workflow
(`small_baseline_flag='n'`). Small-baseline Stage 2–7 branches and nonstandard
Stage 6 modes such as patch-phase, hold-good, tropo subtraction, disabled
look-angle estimation, or custom spatial costs are rejected before output is
written; the application never substitutes a scientifically different mode.
The optional Goldstein prefilter uses the faster conservative CLAP spectrum,
which deliberately preserves weak spectral bins and can produce small wrapped
phase differences from the historical filter.

## Development checks

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo build --release
cargo tree -p pystamps
```

The production dependency audit should show no `pyo3`, `numpy`, Python runtime,
SNAPHU, or Triangle dependency. The root package deliberately sets
`autolib = false` so the legacy `src/lib.rs` PyO3 oracle cannot be discovered by
Cargo accidentally.

## Runtime commands

```text
pystamps prep snap         prepare native SNAP exports
pystamps run               execute Stages 1–8
pystamps status            inspect available artifacts
pystamps verify            compare a run with a golden dataset
pystamps describe-inputs   describe per-stage data scope
pystamps describe-backends report the compiled runtime backend
pystamps list-legacy       inventory an external StaMPS script tree
```

`list-legacy` is read-only inventory support for audits; it never executes the
listed scripts.

## Repository layout

- `crates/pystamps-core`: numerical kernels and scientific stage models
- `crates/pystamps-io`: pure-Rust MAT and SNAP dataset I/O
- `crates/pystamps-pipeline`: transactional stage orchestration
- `crates/pystamps-verify`: tolerant scientific comparison
- `crates/pystamps-cli`: source for the installed `pystamps` binary
- `pystamps/` and `src/`: retained legacy/reference implementations

`oracle/pyproject.toml` only locks dependencies for that source-only oracle and
is marked `tool.uv.package = false`; there is no root Python project. There is
no Python build backend, console entry point, wheel, or sdist. Explicit `make
oracle-*` targets run the reference checks through `PYTHONPATH` without
installing it.

## Notes

- Treat the run directory as mutable and keep the source/golden dataset intact.
- Prefer release builds for realistic Stage 2, Stage 4, and Stage 6 performance.
- Use deterministic retained fixtures when changing solver or filtering logic.
- Runtime output contracts follow StaMPS MAT conventions, including MATLAB-style
  one-based identifiers and column-major array semantics at I/O boundaries.

The standalone contracts and supported modes are documented in
[`docs/native_runtime.md`](docs/native_runtime.md); the reference audit and
full-data measurements are recorded in
[`docs/scientific_audit.md`](docs/scientific_audit.md). Retained Python API
pages are explicitly bannered as historical-oracle references, not production
install or runtime instructions.
