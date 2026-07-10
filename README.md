<div align="center">

<img src="docs/assets/pystamps-logo.svg" alt="pySTAMPS" style="width: 200px; height: auto; max-width: 100%;" />

# pySTAMPS

Python-first STA(MPS)-style runtime for staged InSAR/PS processing, verification, and deterministic audit checks.

Run staged pipelines, inspect dataset progress, and validate outputs against a reference dataset.

<p align="center">
  <a href="https://sirbastiano.github.io/pystamps/"><img src="https://img.shields.io/badge/-Documentation-0f172a?style=for-the-badge&logo=readme&logoColor=white&labelColor=0f172a" alt="Documentation" style="height: 34px;" /></a>
  <a href="https://sirbastiano.github.io/pystamps/quickstart.html"><img src="https://img.shields.io/badge/-Quick%20Start-0f172a?style=for-the-badge&logo=firefoxbrowser&logoColor=white&labelColor=0f172a" alt="Quick Start" style="height: 34px;" /></a>
  <a href="https://sirbastiano.github.io/pystamps/api/pystamps.html"><img src="https://img.shields.io/badge/-API%20Reference-0f172a?style=for-the-badge&logo=python&logoColor=white&labelColor=0f172a" alt="API Reference" style="height: 34px;" /></a>
</p>

</div>

## Install

Prerequisites: Python 3.12+ and `uv` on `PATH` for the default source workflow. Install `uv` first if needed: <https://docs.astral.sh/uv/getting-started/installation/>.

From source:

```bash
git clone git@github.com:sirbastiano/pystamps.git
cd pystamps
uv sync
uv run pystamps describe-backends
```

Native Rust conda install:

```bash
git clone git@github.com:sirbastiano/pystamps.git
cd pystamps
conda env create -f environment.yml
conda activate pystamps-rust
python -m pip install -e ".[dev]"
make native-conda-kernel-check
make native-conda-step-validate
```

The conda environment also installs `huggingface_hub`, which is used by the dataset mirror target below.
Run `make native-conda-env-check` after creating or updating the environment to verify both `huggingface_hub` and pySTAMPS backend discovery.
Run `make native-conda-kernel-check` after editing Rust kernels; it format-checks Rust, runs Rust unit tests, rebuilds the PyO3 extension, and imports `stage6_unwrap_grid`.
Run `make native-conda-step-validate` for the short, always-runnable gate. It checks Rust formatting/build/tests, rebuilds and imports the PyO3 extension, generates a 64-PS fixture, and executes and validates connected Stages 1-8 under `configs/native-kernels.yaml`, one process at a time. The report at `inputs_and_outputs/validation_runs/native_conda_step_validation_latest.json` includes elapsed time and peak RSS for every step in decimal GB and binary GiB.
After a complete run, rerun one retained-fixture step with `make native-conda-step-validate NATIVE_STEPS=e2e-stage6`; list names with `conda run -n pystamps-rust python scripts/native_conda_step_validate.py --list`.
If `conda` is not on the noninteractive shell `PATH`, pass it explicitly, for example:

```bash
make native-conda-check CONDA=/opt/miniconda3/bin/conda
make native-conda-kernel-check CONDA=/opt/miniconda3/bin/conda
make native-conda-step-validate CONDA=/opt/miniconda3/bin/conda
```

To update an existing environment after `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```

Editable installs use `python -m pip install -e .` or `python -m pip install -e "[dev]"`.
`cargo` is required only for editable/source installs that build the Rust extension. Wheels from PyPI may avoid local compilation.
Source builds need a Rust toolchain. Release builds publish platform wheels for the Rust extension where supported.
After activating `pystamps-rust`, use `pystamps ...` directly. Outside that environment, keep using `uv run pystamps ...` from the checkout.

## Validation

Fresh-clone validation commands:

```bash
uv run pytest -q
uv run --with build python -m build --sdist --wheel
uv run --with twine python -m twine check dist/*
```

Local entrypoints:

```bash
make setup
make test
make build
make twine-check
make fetch-insar-dataset
make import-insar-dataset
make audit
make native-conda-env-check
make native-conda-check
make native-conda-kernel-check
make native-conda-step-validate
make native-conda-audit-hf
make native-conda-stage6-fixture
make native-conda-audit
make native-conda-verify
make parity-loop
make verify
make benchmark
```

Dataset-backed audit workflows use the documented optional repo assets, including:

- `inputs_and_outputs/InSAR_dataset_test_stage8diag`
- `inputs_and_outputs/InSAR_dataset_test`
- `inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag`
- `inputs_and_outputs/InSAR_dataset_small_baseline_stage7`

The public Hugging Face source for the single-master test dataset is `https://huggingface.co/datasets/mdelgadoblasco/InSAR_dataset_test/tree/main` and can be mirrored with the official Hugging Face Python API:

```bash
make fetch-insar-dataset
```

That target downloads `mdelgadoblasco/InSAR_dataset_test` into `inputs_and_outputs/InSAR_dataset_test` via `huggingface_hub.snapshot_download`. Override `HF_DATASET_DEST` if you need a different local path.
If you want to force the conda environment interpreter for the fetch, run `make fetch-insar-dataset HF_PYTHON="conda run -n pystamps-rust python"`.
If the runtime has no network access, download the Hugging Face repository archive in a browser and import it with:

```bash
make import-insar-dataset HF_DATASET_ARCHIVE=/path/to/InSAR_dataset_test.zip
```

After mirroring that dataset, run the deep native Rust/conda audit for that Hugging Face dataset with:

```bash
make native-conda-audit-hf
```

Recorded native status:

- Local runs of `conda env update -f environment.yml --prune`, `python -m pip install -e ".[dev]"`, `make native-conda-env-check CONDA=/opt/miniconda3/bin/conda`, `make native-conda-check CONDA=/opt/miniconda3/bin/conda`, and `make native-conda-kernel-check CONDA=/opt/miniconda3/bin/conda` pass in the `pystamps-rust` environment with Rust/Cargo 1.96.1.
- On 2026-07-10, the connected short gate passed all 16 steps in `14.8-23.2s` measured step time across warm-cache and rebuild runs. Peak RSS was `0.118 GB` (`0.110 GiB`); the final manifest validated 19 artifacts for 64 PS, 7 IFGs, and a 97-arc Stage 8 model.
- A prior completed Stage 8 native resume took `5222.5s`; its internal native Stage 6 unwrap reported `5116.1s` for 75 IFGs (`68.2s/IFG`) and `snaphu_external=0.0s`.
- Reproduce saved Stage 6 fixture timing with `make native-conda-stage6-fixture CONDA=/opt/miniconda3/bin/conda STAGE6_FIXTURE_ROOT=inputs_and_outputs/validation_runs/stage6_fixture_minimal STAGE6_THREADS=<threads>`. Current saved 1773x4378 HF fixture estimates are `779.31s` with `STAGE6_THREADS=1` and `442.86s` with `STAGE6_THREADS=0`, so max Rayon is about `1.76x` faster for that candidate. The retained window1024 diagnostic is slower but closer to parity: `1289.85s` one-thread and `867.83s` max-thread, about `1.49x`.
- Output verification is not yet full-chain parity-clean. The current documented blockers are the reduced strict Stage 2 residual on `PATCH_1` (`C_ps max=0.0005214810371398926`, shared by Python and native reruns, so not isolated to Rust dispatch) and the Stage 6 modeled objective delta (`66152`). The retained Stage 6 SNAPHU-core fixture passes the opt-in stable dense-MSD gate with strict wrap agreement, but the exact `diff != 0` dense MSD is not solver evidence because SNAPHU float32 output carries tiny nonzero differences on otherwise flat neighbor edges. Recent Stage 6 diagnostics show the remaining high-gain label islands are tiny favorable subregions inside much larger same-label plateaus; larger cut windows can cover them, but the retained side-256 run moved away from legacy dense-MSD parity despite lowering the modeled objective.

Use `make native-conda-audit` and `make native-conda-verify` when the full `inputs_and_outputs` parity set, including `RUN_FULL_GATE_1e10`, is present.

## Run by stage

Set a local dataset path and always work on a writeable copy:

```bash
export DATASET_SOURCE=/path/to/original_dataset
export DATASET_COPY=/path/to/dataset_copy
cp -a "$DATASET_SOURCE" "$DATASET_COPY"
```

First, check status and verify what can execute:

```bash
uv run pystamps status --dataset "$DATASET_COPY"
```

Run a single stage or stage range:

```bash
uv run pystamps run --dataset "$DATASET_COPY" --start-step 1 --end-step 1      # stage 1 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 2 --end-step 2      # stage 2 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 3 --end-step 3      # stage 3 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 4 --end-step 4      # stage 4 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 5 --end-step 5      # stage 5 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 6 --end-step 6      # stage 6 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 7 --end-step 7      # stage 7 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 8 --end-step 8      # stage 8 only
uv run pystamps run --dataset "$DATASET_COPY" --start-step 1 --end-step 8          # full pipeline
```

From the activated `pystamps-rust` conda environment, launch the same chain with native kernels using `pystamps --config configs/native-kernels.yaml run --dataset "$DATASET_COPY" --start-step 1 --end-step 8`. Change the two step values to run one stage at a time.

Use `--dry-run` to preview actions without writing:

```bash
uv run pystamps run --dataset "$DATASET_COPY" --start-step 1 --end-step 8 --dry-run
```

## Verify a run

```bash
export RUN_COPY=/path/to/run_copy
export GOLDEN_DATASET=/path/to/golden_dataset
uv run pystamps verify --run "$RUN_COPY" --golden "$GOLDEN_DATASET"
```

## Stage-backend profile (optional)

```bash
uv run pystamps describe-backends
```

Use the checked-in `configs/native-kernels.yaml`, or copy the same profile into a local `native-kernels.yaml`:

```bash
cat > native-kernels.yaml <<'YAML'
runtime:
  backend: native
  stage2_kernel_backend: native
  stage2_native_threads: 0
  kernel_backend_overrides:
    stage2_clap_filter_kernel: native
    stage2_grid_accumulate: native
    stage2_grid_indices: native
    stage2_histogram: native
    stage2_normalize_complex: native
    stage2_normalize_phase_matrix: native
    stage2_ph_weight_block: native
    stage2_topofit: native
    stage2_topofit_coh_row_invariant: native
    stage2_topofit_row_invariant: native
    stage3_clap_filt_grid: native
    stage3_clap_filt_grid_stack: native
    stage3_clap_filt_patch: native
    stage3_clap_filt_patch_stack: native
    stage3_coh_threshold: native
    stage3_select_ifg_index: native
    stage3_wrap_filt: native
    stage3_wrap_filt_global: native
    stage4_adjacent_component_keep: native
    stage4_duplicate_keep: native
    stage4_edge_stats: native
    stage4_phase_correction: native
    stage4_weed_ifg_index: native
    stage5_duplicate_keep: native
    stage5_format_merged_rc2: native
    stage5_ifg_std: native
    stage5_patch_keep_mask: native
    stage5_rc2_correction: native
    stage6_estimate_la_error: native
    stage6_extract_grid_values: native
    stage6_grid_accumulate: native
    stage6_prepare_cost_offsets: native
    stage6_ps_grid_indices: native
    stage6_reconstruct_ps_phase: native
    stage6_select_ifgw: native
    stage6_single_master_ifg_geometry: native
    stage6_smooth_3d_full_single_master: native
    stage6_unwrap_grid: native
    stage6_unwrap_ifg_sets: native
    stage7_center_to_reference: native
    stage7_deramp_unwrapped_phase: native
    stage7_mean_velocity_fit: native
    stage7_scla: native
    stage7_scla_smooth: native
    stage8_edge_noise: native
    stage8_weighted_lstsq: native
    weighted_affine_fit: native
    weighted_slope_fit: native
  io_workers: 1
  cpu_workers: 0
  stage7_chunk_ps: 100000
  stage8_chunk_edges: 200000
YAML
```

This profile uses `runtime.backend: native` to select compiled Rust/CPU kernels and run them in-process.
The checked-in validation profile uses `io_workers: 1` to avoid concurrent large MAT-file reads.

```bash
uv run pystamps --config configs/native-kernels.yaml run --dataset "$DATASET_COPY" --start-step 2 --end-step 8
```

When `pystamps-rust` is active, the same run is:

```bash
pystamps --config configs/native-kernels.yaml run --dataset "$DATASET_COPY" --start-step 2 --end-step 8
```

Use `python` backends for reference behavior in debugging, and `native` for the compiled Rust/CPU path.

## Benchmarking and audit checkpoints

```bash
make benchmark
make audit
```

`make audit` reads the manifest in `pystamps/data/audited_workflow_manifest.json`.

## Notes

- Do not point docs or examples at a fixed repository dataset path.
- Always treat outputs in your run tree as authoritative; avoid running on your only source copy.
- Optional repo assets are kept for parity and offline reproducibility, not required for runtime usage.

## Read the docs

- [Pipeline and science guide](https://sirbastiano.github.io/pystamps/pipeline-science-guide.html)
- [Quick Start](https://sirbastiano.github.io/pystamps/quickstart.html)
- [Getting Started](https://sirbastiano.github.io/pystamps/getting-started.html)
- [Usage](https://sirbastiano.github.io/pystamps/usage.html)
- [Configuration](https://sirbastiano.github.io/pystamps/configuration.html)
- [Architecture](https://sirbastiano.github.io/pystamps/architecture.html)
- [Verification](https://sirbastiano.github.io/pystamps/verification.html)
- [API Reference](https://sirbastiano.github.io/pystamps/api/pystamps.html)
- [Release workflow](https://sirbastiano.github.io/pystamps/release.md)

## Notebooks

- `notebooks/start_here.ipynb`
- `notebooks/00_pystamps_beginner_walkthrough.ipynb`
