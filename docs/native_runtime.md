# Standalone native runtime

The installed `pystamps` command is a Rust binary. It does not load Python,
MATLAB, SNAPHU, Triangle, a system HDF5 library, or another process at runtime.
The Python and PyO3 trees are retained only to reproduce reference results.

## Build and install

Rust 1.89 or newer is required by the locked pure-Rust HDF5 reader.

```bash
cargo build --release --locked
cargo install --path . --locked
pystamps describe-backends
```

The root Cargo package is the production package. Its dependency graph is the
authoritative runtime dependency list.

## Conda distribution

The unpublished `pystamps` 0.2.0 Conda package is the same standalone Rust CLI, not
a Python binding or environment for the retained oracle. It has no Python,
system HDF5, SNAPHU, or external scientific-program requirement. No artifact
has been uploaded yet. After a validated development-channel upload, a clean
installation and runtime-boundary check is:

```bash
conda create -n pystamps -c sirbastiano/label/dev -c conda-forge pystamps=0.2.0
conda run -n pystamps pystamps --version
conda run -n pystamps pystamps describe-backends
```

The compiled Conda subdirectories are `linux-64`, `linux-aarch64`, `osx-64`,
`osx-arm64`, `win-64`, and `win-arm64`. Linux packages require glibc 2.17 or
newer, and macOS packages require macOS 11 or newer. Conda does not provide a
musl subdirectory, and the compiled CLI cannot be packaged as `noarch`. Windows
ARM64 is experimental and remains on the `dev` label until native ARM64 CI
succeeds. Moving the exact tested artifacts to `main` requires explicit
authorization.

## Supported systems

The supported source-install matrix is deliberately explicit:

- Linux GNU: x86_64 and ARM64;
- Linux musl: x86_64 and ARM64;
- macOS: Intel x86_64 and Apple Silicon ARM64;
- Windows MSVC: x86_64 and ARM64.

All supported targets are 64-bit and little-endian. CI builds, tests, installs,
and launches the binary natively on representative GitHub-hosted runners using
Rust 1.89. The project does not currently claim 32-bit, big-endian, BSD,
mobile, or WebAssembly support. Dataset transactions require a filesystem with
normal same-volume rename semantics; unusual network filesystems and power-loss
durability are outside the portability contract.

The isolated `oracle/pyproject.toml` is only a locked development environment
for the historical oracle (`tool.uv.package = false`). There is no root Python
project. The oracle manifest defines no Python build backend, console script,
or wheel configuration; `setup.py` and the Python source-distribution manifest
are intentionally absent.

## Validated workflow

The native pipeline implements SNAP preparation and StaMPS-compatible Stages
1–8 for the available single-master workflow:

```bash
pystamps prep snap --dataset DATASET
pystamps run --dataset DATASET --start-step 1 --end-step 8
pystamps verify --run DATASET --golden GOLDEN_DATASET
```

Stages write through transactions, so the completion artifact is published
only after every output for that stage has been written successfully. A rerun
removes later-stage products before execution, preventing an obsolete success
marker from surviving changed upstream data. Stage 2 random-reference caches
include the full baseline vector; Stage 6 checkpoints include all scientific
inputs. Changed phase, geometry, baselines, dates, selection, or unwrap
parameters therefore invalidate the affected caches together.

The application rejects unsupported branches before writing scientifically
different substitutes. Current fail-closed cases include:

- `small_baseline_flag='y'` in Stages 2–7;
- legacy Stage 1 geometry/wavelength/oversampling input files outside the SNAP
  metadata synthesis path;
- Stage 2 `quick_est_gamma_flag='n'` and nonzero Stage 5 merge resampling;
- Stage 3 nonzero `gamma_stdev_reject` bootstrap rejection;
- Stage 6 patch-phase, hold-good, tropo subtraction, predefined phase,
  look-angle-off, and custom spatial-cost modes;
- an external or SNAPHU Stage 6 solver;
- Stage 7 L1 SCLA, tropo or legacy `aps2.mat` subtraction, Cartesian
  `ref_x`/`ref_y` bounds, non-degree-1 deramping, and the small-baseline
  three-pass workflow;
- Stage 8 kriging substitution.

The Goldstein prefilter uses the conservative CLAP spectrum. This is the one
intentional speed/accuracy trade: weak spectral bins are preserved, so wrapped
phase can differ slightly from the historical filter while remaining stable.

## Artifact and numeric contracts

- In-memory matrices are row-major; MAT v5/v7.3 boundaries preserve MATLAB
  dimensions and column-major semantics.
- Identifiers stored in MAT artifacts remain one-based where StaMPS expects
  them; Rust kernel indices are zero-based.
- Complex, sparse, logical, character, NaN/Inf, typed-empty, and wrapped-phase
  comparisons are handled by `pystamps verify`.
- The Stage 6 master and dropped interferograms remain zero in `phuw2.mat`, as
  in the reference workflow.
- Stage 7 preserves `K_ps_uw` as `float64` internally and on write rather than
  discarding precision solely to match a historical `float32` cast.

## Performance choices

- Stage 4 duplicate detection uses stable coordinate sorting rather than a
  quadratic scan.
- Stage 4 edge noise is evaluated edge-wise in parallel without repeatedly
  allocating edge-by-interferogram matrices.
- Stage 6 reuses persistent tree state and saturates each selected negative
  cycle before one pivot count, matching the upstream optimizer contract.
- Stage 6 adaptively unwraps up to four independent interferograms, reserving
  three Rayon threads per solve and limiting active grids to nine million cells.
- Tree adjacency, hierarchy, root-cost, and traversal buffers are rebuilt in
  place without changing cycle candidates, ordering, or convergence.
- Stage 8 uses a spatial hash grid for the finite-radius Gaussian filter.

Use release builds for measurements. Worker count `0` means the available
Rayon pool; constrain it on memory-limited shared machines.

The default `stage6_grid_scale: 1.0` preserves the dataset's StaMPS
`unwrap_grid_size`. For an explicit speed/accuracy trade, run with
`--config configs/stage6-balanced.yaml`; its scale `4.0` multiplies the grid
spacing and therefore reduces dense flow cells by about sixteen times. On a
dataset configured at 20 m this is an 80 m grid, still finer than StaMPS's
historical 200 m default. Shipped profiles leave
`stage6_max_flow_passes: 0` so the flow optimizer converges. Both settings are
part of the scientific checkpoint fingerprint, and a balanced result should
be accepted only after comparison with a strict run.

`configs/stage6-fast.yaml` uses scale `10.0`; for a 20 m dataset that restores
the historical StaMPS 200 m default. It is intended for the large-speed-gain
case and carries the same scientific-verification requirement.

On the 77,850-PS, 76-interferogram M3 Pro gate, a fresh controlled 200 m solve
took 461.90 seconds with the former fixed two-worker scheduler. The new portable
release took 813.72, 462.60, and 254.48 seconds with one, two, and four workers,
respectively. Four workers remained strictly identical to the validated output
and used 728 MB peak RSS versus 700 MB before the change. An M3-specific
`target-cpu=native` build was rejected as a recommendation because it took
290.45 seconds, 14.1% slower than the portable release.

Experimental 300 m and 400 m grids completed Stage 6 in 113.00 and 64.68
seconds. Relative to 200 m, 1.504% and 2.873% of unwrapped samples changed by
integer cycles, capped at 4π; wrapped phase remained within 1.3e-6 radians.
Stage 7 regression propagates those cycle choices broadly, so both coarser
settings remain opt-in and must be checked through Stage 8 for each dataset.
Use `--final-products-only --through-stage 6` to compare their Stage 6 final
product without treating expected grid-cache shape changes as phase failures.

Preprocessing and per-interferogram checkpoints carry solver/cache schemas,
complete input fingerprints, and payload checksums; `phuw2.mat` is not
published until every solve completes. Every run records phase and per-IFG
timings in `.pystamps-stage6/timing-v1-<fingerprint>.json`.

## Verification gate

Before publishing a binary, run:

```bash
cargo fmt --all -- --check
cargo test --workspace --locked
cargo build --release --locked
cargo tree -p pystamps
```

Then run at least one fresh-data comparison, not only checkpoint reuse. The
verifier reports artifact/key context and absolute or wrapped-phase error.

For optional oracle-backed development checks, use the explicitly named
`make oracle-*` targets. They import the checkout through `PYTHONPATH` and do
not install or publish a Python package.
