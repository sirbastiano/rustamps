# Getting Started With pySTAMPS

This is the shortest path to run pySTAMPS without tying documentation to repository-owned datasets.

## What pySTAMPS does

pySTAMPS executes a numbered stage pipeline (`1` to `8`) over a dataset directory.
It detects existing artifacts, so stage ranges can be run incrementally:

- `status`: inspect dataset and stage state
- `run`: execute a selected range
- `verify`: compare a run result against a reference dataset

## Install

```bash
git clone git@github.com:sirbastiano/pystamps.git
cd pystamps
uv sync
uv run pystamps describe-backends
```

If you need to build the native Rust extension from source, use the repository Conda environment. It installs Python plus Rust/Cargo:

```bash
git clone git@github.com:sirbastiano/pystamps.git
cd pystamps
conda env create -f environment.yml
conda activate pystamps-rust
python -m pip install -e ".[dev]"
cargo check
python setup.py build_ext --inplace
pystamps describe-backends
```

Run `make native-conda-env-check` after creating or updating the environment to confirm that `huggingface_hub` and pySTAMPS backend discovery are available.
After Rust kernel edits, run `make native-conda-kernel-check` to format-check Rust, run the Rust unit tests, rebuild the PyO3 extension, and verify that Python can import `stage6_unwrap_grid`.
If `conda` is not on the noninteractive shell `PATH`, pass the executable explicitly:

```bash
make native-conda-check CONDA=/opt/miniconda3/bin/conda
make native-conda-kernel-check CONDA=/opt/miniconda3/bin/conda
```

Update that environment later with:

```bash
conda env update -f environment.yml --prune
```

## Basic run workflow (copy-first)

```bash
export SOURCE_DATASET=/path/to/your_dataset
export RUN_DATASET=/path/to/run_dataset
cp -a "$SOURCE_DATASET" "$RUN_DATASET"

# inspect current progress
uv run pystamps status --dataset "$RUN_DATASET"

# rehearse without writing outputs
uv run pystamps run --dataset "$RUN_DATASET" --start-step 1 --end-step 8 --dry-run
```

Run individual stages by number when you need controlled execution:

```bash
uv run pystamps run --dataset "$RUN_DATASET" --start-step 1 --end-step 1
uv run pystamps run --dataset "$RUN_DATASET" --start-step 2 --end-step 2
uv run pystamps run --dataset "$RUN_DATASET" --start-step 3 --end-step 3
uv run pystamps run --dataset "$RUN_DATASET" --start-step 4 --end-step 4
uv run pystamps run --dataset "$RUN_DATASET" --start-step 5 --end-step 5
uv run pystamps run --dataset "$RUN_DATASET" --start-step 6 --end-step 6
uv run pystamps run --dataset "$RUN_DATASET" --start-step 7 --end-step 7
uv run pystamps run --dataset "$RUN_DATASET" --start-step 8 --end-step 8
```

For full processing:

```bash
uv run pystamps run --dataset "$RUN_DATASET" --start-step 1 --end-step 8
```

## Configure kernel backends

```bash
uv run pystamps describe-backends
```

```yaml
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
```

This profile uses `runtime.backend: native` to select compiled Rust/CPU kernels and run them in-process.
The checked-in validation profile uses `io_workers: 1` to avoid concurrent large MAT-file reads.

Use the checked-in `configs/native-kernels.yaml`, or save that YAML as `native-kernels.yaml`, then run:

```bash
uv run pystamps --config configs/native-kernels.yaml run --dataset "$RUN_DATASET" --start-step 2 --end-step 8
```

Inside the activated `pystamps-rust` conda environment, run the same command without `uv run`:

```bash
pystamps --config configs/native-kernels.yaml run --dataset "$RUN_DATASET" --start-step 2 --end-step 8
```

Current local native status for the public HF test dataset:

- `make native-conda-env-check CONDA=/opt/miniconda3/bin/conda`, `make native-conda-check CONDA=/opt/miniconda3/bin/conda`, and `make native-conda-kernel-check CONDA=/opt/miniconda3/bin/conda` pass in the `pystamps-rust` environment.
- A prior completed native Stage 8 resume measured `5222.5s`; its internal Stage 6 unwrap reported `5116.1s` for 75 IFGs (`68.2s/IFG`) with `snaphu_external=0.0s`.
- Reproduce saved Stage 6 fixture timing with `make native-conda-stage6-fixture STAGE6_THREADS=<threads> STAGE6_FIXTURE_ROOT=inputs_and_outputs/validation_runs/stage6_fixture_minimal CONDA=/opt/miniconda3/bin/conda`; full-budget local diagnostics are `1298.77s` wall time with `STAGE6_THREADS=1` (`1289.85s` inside the native call) and `876.86s` wall time with `STAGE6_THREADS=0` (`867.83s` inside the native call) on the 1773x4378 HF fixture, so the current default/max Rayon speed estimate is about `1.48x` by wall time.
- Native output verification is not yet full-chain parity-clean. The current documented blockers are the reduced strict Stage 2 residual on `PATCH_1` (`C_ps max=0.0005214810371398926`, shared by Python and native reruns, so not isolated to Rust dispatch) and the Stage 6 modeled objective delta (`66152`). The retained Stage 6 SNAPHU-core fixture passes the opt-in stable dense-MSD gate with strict wrap agreement, but exact `diff != 0` dense MSD is not solver evidence because SNAPHU float32 output carries tiny nonzero differences on otherwise flat neighbor edges. Recent Stage 6 diagnostics show the remaining high-gain label islands are tiny favorable subregions inside much larger same-label plateaus; larger cut windows can cover them, but the retained side-256 run moved away from legacy dense-MSD parity despite lowering the modeled objective.

Use `python` in place of `native` for reference execution paths if you are debugging numerical differences.

## Verify

Mirror the public single-master test dataset when you need local validation data:

```bash
make fetch-insar-dataset
```

Source: `https://huggingface.co/datasets/mdelgadoblasco/InSAR_dataset_test/tree/main`.
The fetch target uses `huggingface_hub.snapshot_download`; the `pystamps-rust` conda environment installs that package.
Use `make fetch-insar-dataset HF_PYTHON="conda run -n pystamps-rust python"` to force the conda interpreter.

```bash
export GOLDEN_DATASET=/path/to/reference_dataset
uv run pystamps verify --run "$RUN_DATASET" --golden "$GOLDEN_DATASET"
```

## Stage meaning (quick map)

| Stage | Typical intent |
|---|---|
| 1 | Prepare candidate-level patch artifacts |
| 2 | Compute quality metrics and model terms |
| 3 | Select persistent candidates |
| 4 | Weed out weak/redundant candidates |
| 5 | Merge patch outputs into dataset-level products |
| 6 | Unwrap temporal products |
| 7 | Estimate SCLA correction terms |
| 8 | Apply final space-time filtering |
