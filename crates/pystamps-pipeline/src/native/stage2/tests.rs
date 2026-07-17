use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use num_complex::Complex32;
use pystamps_io::{read_mat, write_mat, MatArray, MatFile, MatValue};

use super::*;
use crate::native::mat::{complex32, numeric_f64, shape};
use crate::{NativeExecutor, StageExecutor};

struct TempDir(PathBuf);

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

impl TempDir {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "pystamps-stage2-test-{}-{nonce}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn synthetic_stage2_writes_replayable_pm_contract_from_cached_reference() {
    assert_eq!(reference::random_samples(), 300_000);
    let temp = TempDir::new();
    let patch = temp.0.join("PATCH_1");
    fs::create_dir(&patch).unwrap();
    write_inputs(&patch);

    let params = Params::load(&patch).unwrap();
    let loaded = input::load(&patch).unwrap();
    let wraps = options::trial_wraps(&loaded, &params).unwrap();
    write_reference_cache(&patch, wraps, &loaded.nominal_bperp);

    let started = Instant::now();
    let summary = NativeExecutor
        .run_patch(2, &patch, &RunConfig::default())
        .unwrap();
    let elapsed = started.elapsed().as_secs_f64();
    assert!(summary.contains("random reference cached"));
    assert!(
        elapsed < 10.0,
        "cached two-point Stage 2 took {elapsed:.3}s"
    );

    let pm = read_mat(patch.join("pm1.mat")).unwrap();
    for (key, expected) in [
        ("K_ps", vec![2, 1]),
        ("C_ps", vec![2, 1]),
        ("coh_ps", vec![2, 1]),
        ("N_opt", vec![2, 1]),
        ("ph_res", vec![2, 2]),
        ("ph_patch", vec![2, 2]),
        ("ph_weight", vec![2, 2]),
        ("grid_ij", vec![2, 2]),
        ("Nr", vec![1, 100]),
        ("coh_bins", vec![1, 100]),
    ] {
        assert_eq!(shape(&pm, key).unwrap(), expected, "wrong {key} shape");
    }
    assert_grid_replays(&pm);
    let weights = complex32(&pm, "ph_weight").unwrap();
    for value in &weights[..2] {
        assert!((value.norm() - 5.0).abs() <= 1e-5);
    }
    for value in &weights[2..] {
        assert!((value.norm() - 4.0).abs() <= 1e-5);
    }
    assert_eq!(numeric_f64(&pm, "Nr").unwrap(), vec![1.0; 100]);
    assert_eq!(numeric_f64(&pm, "i_loop").unwrap(), vec![1.0]);
    assert!(numeric_f64(&pm, "n_trial_wraps").unwrap()[0].is_finite());
}

#[test]
fn small_baseline_mode_fails_before_partial_output() {
    let temp = TempDir::new();
    let patch = temp.0.join("PATCH_1");
    fs::create_dir(&patch).unwrap();
    let mut parms = MatFile::new();
    parms.insert(
        "small_baseline_flag".to_owned(),
        MatValue::U8(MatArray {
            shape: vec![1, 1],
            values: vec![b'y'],
        }),
    );
    write_mat(patch.join("parms.mat"), &parms).unwrap();
    let error = NativeExecutor
        .run_patch(2, &patch, &RunConfig::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("small-baseline Stage 2 is not supported"));
    assert!(!patch.join("pm1.mat").exists());
}

#[test]
fn full_gamma_mode_fails_before_partial_output() {
    let temp = TempDir::new();
    let patch = temp.0.join("PATCH_1");
    fs::create_dir(&patch).unwrap();
    let mut parms = MatFile::new();
    parms.insert(
        "quick_est_gamma_flag".to_owned(),
        MatValue::U8(MatArray {
            shape: vec![1, 1],
            values: vec![b'n'],
        }),
    );
    write_mat(patch.join("parms.mat"), &parms).unwrap();

    let error = NativeExecutor
        .run_patch(2, &patch, &RunConfig::default())
        .unwrap_err()
        .to_string();

    assert!(error.contains("quick_est_gamma_flag='n'"));
    assert!(!patch.join("pm1.mat").exists());
}

#[test]
fn legacy_patch_cache_without_baseline_identity_is_not_reused() {
    let patch = Path::new("inputs_and_outputs/InSAR_dataset_test/PATCH_1");
    if !patch.join("pm1.mat").is_file() {
        return;
    }
    let params = Params::load(patch).unwrap();
    let loaded = input::load(patch).unwrap();
    let computed = options::trial_wraps(&loaded, &params).unwrap();
    let cached = reference::load_or_generate(patch, &loaded.nominal_bperp, computed).unwrap();
    assert!(!cached.cache_hit);
    let pm = read_mat(patch.join("pm1.mat")).unwrap();
    let saved = numeric_f64(&pm, "n_trial_wraps").unwrap()[0];
    assert!((computed as f32 as f64 - saved).abs() <= f32::EPSILON as f64);
}

#[test]
fn cache_rejects_changed_inner_baseline_with_unchanged_span() {
    let temp = TempDir::new();
    let patch = temp.0.join("PATCH_1");
    fs::create_dir(&patch).unwrap();
    let original = [-20.0, 0.0, 25.0];
    let changed = [-20.0, 1.0, 25.0];
    let wraps = 0.75;
    write_reference_cache(&patch, wraps, &original);
    let bins = reference::coherence_bins();

    assert!(reference::load_pm_cache(
        &patch,
        &bins,
        wraps,
        reference::bperp_fingerprint(&original),
    )
    .is_some());
    assert!(
        reference::load_pm_cache(&patch, &bins, wraps, reference::bperp_fingerprint(&changed),)
            .is_none()
    );
}

pub(super) fn write_inputs(patch: &Path) {
    let mut ps = MatFile::new();
    insert_f64(&mut ps, "n_ps", vec![1, 1], vec![2.0]);
    insert_f64(&mut ps, "master_ix", vec![1, 1], vec![2.0]);
    insert_f64(&mut ps, "bperp", vec![3, 1], vec![-20.0, 0.0, 25.0]);
    insert_f64(
        &mut ps,
        "xy",
        vec![2, 3],
        vec![1.0, 0.0, 0.0, 2.0, 30.0, 20.0],
    );
    insert_f64(&mut ps, "mean_range", vec![1, 1], vec![830_000.0]);
    insert_f64(
        &mut ps,
        "mean_incidence",
        vec![1, 1],
        vec![23_f64.to_radians()],
    );
    write_mat(patch.join("ps1.mat"), &ps).unwrap();

    let phase = [-20.0_f32, 0.0, 25.0]
        .into_iter()
        .chain([-20.0_f32, 0.0, 25.0])
        .enumerate()
        .map(|(index, baseline)| {
            let angle = baseline * 0.01 + if index >= 3 { 0.2 } else { 0.0 };
            Complex32::new(angle.cos(), angle.sin())
        })
        .collect();
    let mut ph = MatFile::new();
    ph.insert(
        "ph".to_owned(),
        MatValue::ComplexF32(MatArray {
            shape: vec![2, 3],
            values: phase,
        }),
    );
    write_mat(patch.join("ph1.mat"), &ph).unwrap();

    let mut bp = MatFile::new();
    insert_f64(
        &mut bp,
        "bperp_mat",
        vec![2, 2],
        vec![-20.0, 25.0, -20.0, 25.0],
    );
    write_mat(patch.join("bp1.mat"), &bp).unwrap();
    let mut da = MatFile::new();
    insert_f64(&mut da, "D_A", vec![2, 1], vec![0.2, 0.25]);
    write_mat(patch.join("da1.mat"), &da).unwrap();
    let mut parms = MatFile::new();
    insert_f64(&mut parms, "gamma_max_iterations", vec![1, 1], vec![1.0]);
    insert_f64(&mut parms, "lambda", vec![1, 1], vec![0.0555]);
    write_mat(patch.join("parms.mat"), &parms).unwrap();
}

fn write_reference_cache(patch: &Path, wraps: f64, bperp: &[f64]) {
    let mut cache = MatFile::new();
    insert_f64(
        &mut cache,
        "coh_bins",
        vec![1, 100],
        reference::coherence_bins(),
    );
    insert_f64(&mut cache, "Nr", vec![1, 100], vec![1.0; 100]);
    insert_f64(&mut cache, "Nr_max_nz_ix", vec![1, 1], vec![100.0]);
    cache.insert(
        "n_trial_wraps".to_owned(),
        MatValue::F32(MatArray {
            shape: vec![1, 1],
            values: vec![wraps as f32],
        }),
    );
    insert_f64(
        &mut cache,
        "random_bperp_fingerprint",
        vec![1, 1],
        vec![reference::bperp_fingerprint(bperp) as f64],
    );
    write_mat(patch.join("pm1.mat"), &cache).unwrap();
}

fn assert_grid_replays(pm: &BTreeMap<String, MatValue>) {
    let grid_shape = shape(pm, "ph_grid").unwrap();
    assert_eq!(grid_shape.len(), 3);
    let n_ifg = grid_shape[2];
    let indices = numeric_f64(pm, "grid_ij").unwrap();
    let weights = complex32(pm, "ph_weight").unwrap();
    let actual = complex32(pm, "ph_grid").unwrap();
    let mut replay = vec![Complex32::new(0.0, 0.0); actual.len()];
    for point in 0..indices.len() / 2 {
        let row = indices[point * 2] as usize - 1;
        let col = indices[point * 2 + 1] as usize - 1;
        for ifg in 0..n_ifg {
            replay[(row * grid_shape[1] + col) * n_ifg + ifg] += weights[point * n_ifg + ifg];
        }
    }
    for (index, (left, right)) in actual.iter().zip(replay).enumerate() {
        assert!(
            (*left - right).norm() <= 1e-6,
            "grid mismatch at {index}: actual={left:?} replay={right:?}, shape={grid_shape:?}, indices={indices:?}, weights={weights:?}"
        );
    }
}

fn insert_f64(file: &mut MatFile, key: &str, shape: Vec<usize>, values: Vec<f64>) {
    file.insert(key.to_owned(), MatValue::F64(MatArray { shape, values }));
}
