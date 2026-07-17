# Rustamps pipeline science guide

Rustamps implements the supported single-master persistent-scatterer workflow
in eight native stages. Stages 1–4 work inside each patch, Stage 5 finishes and
merges the patches, and Stages 6–8 work on the merged dataset.

The production binary does not load Python, MATLAB, SNAPHU, Triangle, or
another scientific executable at runtime. The retained historical
implementation is a developer-only oracle, not a fallback backend.

## Terms used below

- An **acquisition** is one radar image captured on one date.
- The **reference** or **master** is the acquisition to which every secondary
  image is aligned.
- An **interferogram** is the phase difference between the reference and one
  secondary.
- A **persistent scatterer (PS)** is a pixel whose radar response remains
  stable enough to follow phase through time.
- **Wrapped phase** repeats from −π to +π. Unwrapping restores the missing
  whole 2π cycles.
- **Perpendicular baseline** is cross-track satellite separation. Residual
  elevation error changes phase in proportion to it.
- A **patch** is a slightly overlapping piece of the scene processed before
  the global merge.

## Before Stage 1: prepare candidates

`rustamps prep snap` converts a compatible SNAP StaMPS export into the raw
point files used by Stage 1:

1. Read each co-registered RSLC and normalize its amplitude.
2. Compute amplitude dispersion at every pixel.
3. Reject near-zero signal and invalid longitude, latitude, or height.
4. Keep pixels below the configured dispersion threshold.
5. Partition them into overlapping patches and extract wrapped phase across
   all reference–secondary interferograms.
6. Write sensor heading and wavelength into `parms.mat`.

It publishes `PATCH_*/pscands.1.*`, `patch.list`, and `parms.mat`. Preparation
checks the raster tree and basic dimensions. Stage 1 later performs the full
baseline and sensor-geometry check. See [Input data from SNAP](inputs.html).

## Stage 1: organize the observations

**Purpose:** align every candidate’s phase, geometry, date, and baseline.

1. Read candidate row/column, complex phase, longitude/latitude, height, and
   amplitude dispersion.
2. Read acquisition dates, the single reference, baseline files, and reference
   sensor geometry.
3. Sort acquisitions chronologically and create the reference column in the
   expected position.
4. Convert longitude/latitude to a local metric XY system, remove unusable
   rows, and spatially sort the candidates.
5. Apply the same reindexing to every related array.

It writes `ps1.mat`, `ph1.mat`, `bp1.mat`, `psver.mat`, and optional
`da1.mat`, `hgt1.mat`, and `la1.mat`. `ps1.mat` is the completion marker.

## Stage 2: estimate phase stability

**Purpose:** measure how consistently each candidate’s phase can be explained.

1. Remove the reference phase so interferograms share one phase origin.
2. Grid nearby candidates and CLAP-filter the spatial phase.
3. Fit `K`, the phase term proportional to perpendicular baseline. This
   represents residual topographic or look-angle error.
4. Fit constant phase `C`, calculate residuals, and convert residual
   consistency into temporal coherence.
5. Repeat with P-square random-noise or SNR weighting until convergence or the
   configured iteration limit.

It writes `pm1.mat` with coherence, `K`/`C`, filtered patch phase, residuals,
and random-coherence reference data.

## Stage 3: select persistent scatterers

**Purpose:** derive a data-aware coherence cutoff and apply it.

1. Group candidates into amplitude-dispersion bins.
2. Compare observed coherence with the distribution expected from random
   phase.
3. Choose a threshold that limits expected random points by density or
   percentage.
4. Keep candidates strictly above that threshold.
5. Normally refilter/refit the selected subset, recalculate the cutoff, and
   reject unstable topographic-error estimates.

It writes `select1.mat` with source indices, threshold information, and the
final keep mask.

## Stage 4: weed unreliable or redundant points

**Purpose:** remove selected points that fail spatial or temporal checks.

1. Apply Stage 3 selection.
2. Optionally keep only the best adjacent candidate; remove duplicate
   locations and optionally zero-height points.
3. Correct phase with the estimated topographic term.
4. Connect nearby points with a Delaunay network and measure temporal phase
   noise on its edges.
5. Reject points beyond the configured standard-deviation or maximum-noise
   limits.

It writes `weed1.mat` with the final patch keep masks and noise measurements.

## Stage 5: promote and merge patches

**Purpose:** produce one scene without duplicated overlap points.

1. Apply all selection/weeding masks to phase, geometry, baseline, height, and
   model arrays.
2. Correct/rereference phase and write each patch’s version-2 products.
3. Use non-overlap bounds to assign one owner to shared patch borders.
4. Resolve remaining duplicate coordinates and join the owned rows.
5. Recompute/sort merged XY and estimate residual standard deviation for each
   interferogram.

It writes patch and root `ps2.mat`, `ph2.mat`, `pm2.mat`, `bp2.mat`, and
related products. Root `ifgstd2.mat` is the completion marker. Stages 6–8 use
the merged root from this point onward.

## Stage 6: unwrap phase

**Purpose:** recover missing whole 2π cycles from wrapped phase.

1. Load merged phase/geometry, exclude the reference and configured dropped
   interferograms, and apply available corrections.
2. Coalesce nearby PS onto an unwrap grid and optionally Goldstein-filter it.
3. Build spatial neighbor edges and estimate space-time edge phase and
   uncertainty.
4. Convert uncertainty to integer-flow costs and solve every interferogram to
   convergence.
5. Interpolate the grid solution back to the original PS, restore required
   corrections, and keep reference/dropped columns zero.

It writes reusable `uw_*.mat` intermediates and final `phuw2.mat`.
Fingerprinted per-interferogram checkpoints below `.pystamps-stage6/` are
reused only when phase, geometry, dates, baselines, selection, and solver
settings still match. `phuw2.mat` appears only after every solve succeeds.

## Stage 7: estimate spatially correlated look-angle error

**Purpose:** estimate unwrapped phase that still follows baseline because
elevation/look angle is slightly wrong.

1. Optionally remove a degree-1 ramp and center phase on a geographic reference
   area.
2. Fit unwrapped phase against each PS baseline history with the L2 method.
3. Account for acquisition timing and Stage 5 interferogram noise.
4. Compare each `K`/`C` estimate with Delaunay neighbors and clamp extremes to
   a spatially plausible envelope.
5. Build the correction predicted by the smoothed field.

It writes `scla2.mat` and `scla_smooth2.mat`.

## Stage 8: estimate space-time correlated noise

**Purpose:** isolate broad phase patterns shared by nearby PS and changing
through time.

1. Subtract Stage 7 SCLA, constant, and optional ramp terms.
2. Optionally remove a spatial plane from selected interferograms.
3. Apply a temporal Gaussian high-pass.
4. Apply a spatial Gaussian low-pass to retain nearby correlated structure.
5. Reference the estimate consistently and force reference-acquisition noise
   to zero.

It writes `scn2.mat` with `ph_scn_slave`, `ph_hpt`, and `ph_ramp`.
`scn2.mat` is a correction/noise product, not by itself a velocity map.

## Run, resume, and invalidate

Preview writes, run a range, or resume from existing markers:

```bash
rustamps run --dataset DATASET --start-step 1 --end-step 8 --dry-run
rustamps run --dataset DATASET --start-step 3 --end-step 6
rustamps run --dataset DATASET --start-step 0 --end-step 8
```

A positive start stage explicitly reruns from there. Before writing, Rustamps
removes dependent later products. A patch rerun invalidates merged output; a
merged rerun removes only later merged products. Every bundle is transactional,
so a failed stage does not publish a false completion marker.

## Speed and verification

`stage6_grid_scale` is the reviewed speed/accuracy trade. Larger values make a
coarser unwrap grid but still solve integer flow to convergence. Compare a
coarse profile with a strict-grid result for each dataset:

```bash
rustamps --config configs/stage6-fast.yaml run \
  --dataset DATASET --start-step 6 --end-step 8
rustamps --config configs/stage6-fast.yaml verify \
  --run DATASET --golden STRICT_RESULT \
  --profile scientific --final-products-only --through-stage 6
```

Strict verification checks structure and configured numeric tolerances.
Scientific verification permits only explicitly bounded outliers with a hard
absolute cap. Unwrapped `ph_uw` is never wrapped merely to pass comparison.

## Supported boundary

The validated end-to-end contract is single-master PS processing. Small
baseline mode, external unwrapping, reference replay, non-native providers,
Stage 7 atmospheric subtraction, and Stage 8 kriging fail before publication.
See [Configuration](configuration.html) and
[the native runtime contract](native_runtime.md) for exact supported values.
