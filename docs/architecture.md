# Architecture Snapshot

pySTAMPS is organized around one invariant: a StaMPS-style dataset directory is the source of truth. The CLI, runtime scheduler, ported stages, optimized kernels, and verification tools all read from or write to that directory.

For the full teaching guide, read [pipeline_science_guide.md](pipeline_science_guide.md).

## Runtime layers

| Layer | Main modules | Responsibility |
| --- | --- | --- |
| CLI | `pystamps.cli` | Parse commands, load config, print JSON reports |
| Configuration | `pystamps.config` | Normalize runtime, kernel, tolerance, tool, and compatibility settings |
| Dataset I/O | `pystamps.io.dataset`, `pystamps.io.mat` | Discover patches and read/write MATLAB-compatible artifacts |
| Pipeline orchestration | `pystamps.pipeline.stages` | Select stages, skip existing artifacts, schedule patch or merged work |
| Scientific stages | `pystamps.pipeline.ported` | Implement StaMPS-style stage behavior in Python |
| Kernels | `pystamps.kernels` | Dispatch hot numerical kernels to Python, native Rust/CPU, or CUDA providers |
| Runtime execution | `pystamps.runtime` | Provide hybrid thread/process execution primitives |
| Verification | `pystamps.verify`, `scripts/validate_audit.py` | Compare run outputs against golden datasets and audit manifests |

## Pipeline model

Stages mirror the StaMPS stage range 1 through 8.

| Stage | Scope | Pipeline name | Expected progress artifact |
| --- | --- | --- | --- |
| 1 | patch | Initial load | `PATCH_*/ps1.mat` |
| 2 | patch | Estimate gamma | `PATCH_*/pm1.mat` |
| 3 | patch | Select PS pixels | `PATCH_*/select1.mat` |
| 4 | patch | Weed adjacent pixels | `PATCH_*/weed1.mat` |
| 5 | patch and merged | Correct phase and merge | `PATCH_*/ph2.mat`, root `ifgstd2.mat` |
| 6 | merged | Unwrap phase | root `phuw2.mat` |
| 7 | merged | Calculate SCLA | root `scla2.mat` |
| 8 | merged | Filter SCN | root `uw_space_time.mat` |

Patch-scoped stages run once per discovered `PATCH_*` directory. Merged stages run once at the dataset root. Stage 5 has both patch promotion and merged aggregation behavior.

## Artifact-driven scheduling

Before running a stage, `pystamps.pipeline.stages` checks the expected artifact or stage bundle. If the artifacts already exist, the result status is `skipped_existing`. This makes the pipeline resumable and safe for inspection, but it also means benchmark or backend experiments must use a dataset copy that actually needs the target outputs.

The normal stage result statuses are:

| Status | Meaning |
| --- | --- |
| `planned` | Dry-run selected the stage but did not execute it |
| `completed` | Stage executed or strict reference replay copied the expected bundle |
| `skipped_existing` | Expected artifacts were already present |
| `skipped` | No artifact mapping exists for that scope |
| `failed` | Stage raised an execution error |

## Backend and kernel architecture

pySTAMPS separates runtime scheduling from numerical kernel selection.

Runtime backend values:

- `auto`: choose a practical scheduling mode by stage
- `threads`: use the I/O-oriented path
- `processes`: use CPU process workers where appropriate
- `gpu`: keep GPU-capable work in-process
- `native`: prefer CPU-oriented scheduling

Kernel backend values:

- `python`: reference NumPy/Python implementation
- `native`: compiled Rust/CPU implementation
- `cuda`: CuPy/CUDA implementation where registered and available
- `auto`: prefer optimized available providers with fallback rules

Current optimized kernel names are:

- `stage2_clap_filter_kernel`
- `stage2_grid_accumulate`
- `stage2_grid_indices`
- `stage2_histogram`
- `stage2_normalize_complex`
- `stage2_normalize_phase_matrix`
- `stage2_ph_weight_block`
- `stage2_topofit`
- `stage2_topofit_row_invariant`
- `stage2_topofit_coh_row_invariant`
- `stage3_clap_filt_grid`
- `stage3_clap_filt_grid_stack`
- `stage3_clap_filt_patch`
- `stage3_wrap_filt`
- `stage3_wrap_filt_global`
- `stage3_coh_threshold`
- `stage3_select_ifg_index`
- `stage4_adjacent_component_keep`
- `stage4_duplicate_keep`
- `stage4_edge_stats`
- `stage4_phase_correction`
- `stage4_weed_ifg_index`
- `stage5_duplicate_keep`
- `stage5_format_merged_rc2`
- `stage5_ifg_std`
- `stage5_patch_keep_mask`
- `stage5_rc2_correction`
- `stage6_estimate_la_error`
- `stage6_extract_grid_values`
- `stage6_grid_accumulate`
- `stage6_smooth_3d_full_single_master`
- `stage6_single_master_ifg_geometry`
- `stage6_unwrap_grid`
- `stage6_unwrap_ifg_sets`
- `stage7_deramp_unwrapped_phase`
- `stage7_mean_velocity_fit`
- `stage7_scla`
- `stage7_scla_smooth`
- `stage8_edge_noise`
- `stage8_weighted_lstsq`

Inspect local availability with:

```bash
uv run pystamps describe-backends
```

## Config flow

`pystamps --config CONFIG.yaml ...` loads a YAML or JSON config into `RunConfig`.

Common runtime fields:

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
    stage6_smooth_3d_full_single_master: native
    stage6_single_master_ifg_geometry: native
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

`backend: native` selects compiled Rust/CPU kernels and schedules them in-process.
The native validation profile uses one patch worker to avoid concurrent large MAT-file reads.

`cpu_workers: 0` means use the detected CPU budget. `stage2_native_threads: 0` lets native stage-2 execution use the configured CPU budget while avoiding patch-level oversubscription.

## Verification architecture

Single-run verification compares one run tree to one golden tree:

```bash
uv run pystamps verify --run RUN_DIR --golden GOLDEN_DIR
```

Full parity audit is handled by:

```bash
make audit
```

The audit dataset list is owned by `pystamps/data/audited_workflow_manifest.json`. Oracle precedence is owned by `pystamps/data/oracle_contract.json`. These files are the compatibility contract for broad parity claims.

## Practical boundaries

- The package implements stages 1 through 8 in `pystamps.pipeline.ported`.
- The optimized native extension accelerates selected hot kernels, not every line of the pipeline.
- Stage 3 still performs selection orchestration and CLAP stack preparation in Python, but grid-stack, grid, patch CLAP, local wrapped-phase, and global wrapped-phase filtering, IFG-index selection, threshold histograms, and re-estimation topofit solves route through native-capable kernel dispatchers when `backend` selects an accelerated provider.
- External tools such as `triangle` and legacy/fallback `snaphu` are still required for execution paths that select them; the supported native Stage 6 path avoids `snaphu`.
- Parity should be claimed from `verify` or audit evidence, not from command completion alone.
- Speed should be claimed from `make benchmark` or `scripts/benchmark_backends.py`, not from a skipped pipeline run.
