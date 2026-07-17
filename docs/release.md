# Native release process

`rustamps` is released as a Rust binary. Python wheels, sdists, PyPI, PyO3,
SNAPHU, Triangle, and system HDF5 are outside the production release surface.
The retained Python tree is a source-only scientific oracle.

## Prerequisites

- Rust 1.89 or newer;
- a clean worktree at the candidate revision;
- the retained validation datasets for the scientific gate;
- target toolchains for every platform being released.

## Candidate gate

Run the complete Cargo gate from the repository root:

```bash
cargo fmt --all -- --check
cargo test --workspace --locked
cargo build --release --locked
cargo tree -p rustamps -e normal,build
```

The dependency tree must not contain `pyo3`, `numpy`, `hdf5-sys`, a Python
runtime, or wrappers for SNAPHU or Triangle. On macOS, `otool -L
target/release/rustamps` should list only operating-system libraries. Use the
corresponding loader inspection on Linux or Windows.

The `Rustamps portable installation` workflow must also pass its GNU/musl Linux,
Intel/ARM macOS, and x64/ARM Windows rows. Each row uses Rust 1.89, runs the
workspace tests, builds a release binary, installs from the checkout, and
launches the installed command.

Smoke-test the installed command, not only the checkout binary. On POSIX:

```bash
cargo install --path . --locked --root /tmp/rustamps-release
/tmp/rustamps-release/bin/rustamps --help
/tmp/rustamps-release/bin/rustamps describe-backends
```

On Windows PowerShell:

```powershell
$root = Join-Path $env:TEMP 'rustamps-release'
cargo install --path . --locked --root $root
$binary = Join-Path $root 'bin\rustamps.exe'
& $binary --help
& $binary describe-backends
```

## Scientific gate

Create a fresh writable copy of the retained input, execute the native stage
range being released, and compare it with the golden tree:

```bash
cp -a GOLDEN_DATASET RUN_DATASET
cargo run --release --locked -- run \
  --dataset RUN_DATASET --start-step 1 --end-step 8
cargo run --release --locked -- verify \
  --run RUN_DATASET --golden GOLDEN_DATASET
```

The example uses POSIX copy syntax. On Windows, create the run copy with
`Copy-Item -Recurse GOLDEN_DATASET RUN_DATASET` before running the same Cargo
commands.

Do not substitute a cache-only rerun. Long Stage 6 validation may be resumed
from its fingerprinted per-interferogram checkpoints, but the final comparison
must cover a completed fresh-data run. Record tolerances, elapsed time, peak
memory, and any accepted scientific deviation with the candidate revision.

The non-installable historical oracle can provide an additional comparison:

```bash
make oracle-setup
make oracle-test
make oracle-audit
```

These commands are development evidence only; their Python environment is not
bundled with or required by the release binary.

## Publish

### Conda package

The Conda distribution is `rustamps` 0.3.0 and contains only the compiled
native CLI plus package and license metadata. It must not contain or depend on
Python, a system HDF5 library, SNAPHU, the oracle tree, or another scientific
executable. Conda Linux artifacts target glibc and are not interchangeable with
the Cargo-built musl binaries. Linux packages require glibc 2.17 or newer,
macOS packages require macOS 11 or newer, and the package is not `noarch`.

Build and validate exactly one package for each supported Conda subdirectory:
`linux-64`, `linux-aarch64`, `osx-64`, `osx-arm64`, `win-64`, and `win-arm64`.
For every artifact, create a clean temporary environment and run:

```bash
conda create -n rustamps-package-test -c LOCAL_CHANNEL -c conda-forge rustamps=0.3.0
conda run -n rustamps-package-test rustamps --version
conda run -n rustamps-package-test rustamps --help
conda run -n rustamps-package-test rustamps describe-backends
```

Assert that `describe-backends` reports an empty
`runtime_external_dependencies` list and that the artifact architecture and
Conda subdirectory match. The Cargo, recipe, source-tag, and installed-command
versions must agree. Rebuilding 0.3.0 requires incrementing the Conda build
number; do not overwrite an existing artifact.

Authorized uploads publish the exact locally and natively tested artifacts to
`sirbastiano/label/dev`.
Windows ARM64 remains experimental on `dev` until its package passes native
ARM64 CI. Before adding `ANACONDA_API_KEY`, configure the GitHub `anaconda-dev`
environment with a required reviewer, prevent self-review, and restrict its
deployment rule to release tags. Protect `v*` tags from update or deletion with
a repository ruleset, and store the key as an environment secret rather than a
repository secret. Verify the channel with a clean installation:

```bash
conda create -n rustamps -c sirbastiano/label/dev -c conda-forge rustamps=0.3.0
conda run -n rustamps rustamps describe-backends
```

Promotion to the `main` label is a separate explicit action. Promote the exact
tested filenames without rebuilding them; never expose publishing credentials
to pull-request or ordinary branch jobs.

### Standalone archives

1. Tag the exact gated revision as `vX.Y.Z`.
2. Build each supported target from that tag with `cargo build --release
   --locked --target TARGET`.
3. Archive the `rustamps` executable with `LICENSE`, `README.md`, and
   `docs/native_runtime.md`.
4. Verify the archive checksum and rerun `rustamps --help` after extraction.
5. Publish checksums beside the platform archives.

A release remains blocked by a failed Cargo check, an incomplete or stale
scientific run, an unexpected dynamic library, or an undocumented verifier
failure.
