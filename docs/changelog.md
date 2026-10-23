# Changelog

Changes that affect scientific output, artifacts, or the supported workflow.
Parity statements refer to the reference run `InSAR_dataset_test`
(Hugging Face revision `cabed239`) and StaMPS revision `c159eb8`.

## Unreleased

### Parity with StaMPS

- **Stage 2 incidence angle.** The mean incidence now follows
  `ps_est_gamma_quick.m`: `inc1.mat`, then `mean(la)+0.052`, then 21°.
  `ps.mean_incidence` is no longer read here; Stage 6 still uses it, as
  `ps_unwrap.m` does. On PATCH_1 this changes `n_trial_wraps` from 0.784 to
  0.727, the iteration count from 5 to 8, and the selected PS from 80928 to
  80938.
- **Stage 5 merge metadata.** The merged `ps2.mat` takes `bperp`, `ll0`,
  `mean_incidence` and `mean_range` from the last patch, as
  `ps_merge_patches.m` does, instead of the first. `ll0` is no longer
  recomputed, because `ps_setref` uses it for `ref_radius` selection.
- **Stage 6 Goldstein prefilter.** The prefilter follows `wrap_filt.m`: the
  spectrum is weighted by the median-normalised, smoothed response raised to
  `unwrap_gold_alpha`, and the padding is `round(n_win/4)`. The earlier
  CLAP-based approximation did not attenuate weak bins.
- **Stage 6 rerun.** Re-running Stage 6 keeps `scla2.mat` and `scla2.npz`, so
  the usual 6 → 7 → 6 sequence leaves the Stage 7 estimate in place. Re-running
  Stage 5 or an earlier stage still removes them.

### Artifacts

- Stage 7 also writes `scla2.npz`, a NumPy archive with the same arrays as
  `scla2.mat`. The MAT files remain the inputs for later stages and for
  verification.
- Mask variables `ix_weed` and `ix_weed2` use the MATLAB logical type.

### Cache invalidation

- The Stage 6 cache schema versions were increased (grid, interpolation and
  space-time caches: 2; solve checkpoints: 5). Caches written by earlier builds
  are rebuilt on the next run.

### Notebooks

- Notebooks `00`-`04`, `start_here`, `start_here_rust` and `backend_rust_explanation` now run the real `rustamps`
  binary on the PATCH_1 golden inputs (Stages 1-5) and end in assertions: counts from the original `STAMPS.log`,
  bit-exact index/mask/coordinate arrays, declared float bounds for the rest, negative controls, determinism across
  worker counts and the runtime boundary. They need a built binary (`cargo build --release`, or `RUSTAMPS_BIN`) and the
  golden PATCH_1 files; missing inputs fail the notebook. Shared helpers: `notebooks/rustamps_nb.py`.
- Notebook `05` (SNAP/CDSE preprocessing) stops at a prerequisite gate when SNAP `gpt`, CDSE credentials, `phidown` or
  `sarpyx` are missing; it is not proven here. Stages 6-8 are not covered by any notebook.

### Verified result

On PATCH_1, running Stages 1–5 in Rust from the raw `pscands.1.*` inputs gives
81428 → 80938 → 79227 → 77888 PS, with exact `ix`, `keep_ix`, `ix_weed`,
`ix_weed2`, `ij`, `xy`, `ph2` and `bp2`. `K_ps` and `coh_ps` still differ by up
to 1.2e-4, because MATLAB evaluates the topographic fit in single precision.
Stages 6–8 have not been compared numerically.

## 0.3.0

- First standalone Rust release: SNAP preparation and the single-master
  Stages 1–8, MAT v5/v7.3 I/O, transactional stage publication, checkpointed
  Stage 6 and the `verify` command.
