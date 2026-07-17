use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use num_complex::Complex32;
use pystamps_io::{read_mat, write_mat, MatArray, MatFile, MatValue};

use crate::RunConfig;

use super::run;

#[test]
fn fresh_single_master_run_builds_checkpoints_and_scientific_phase_output() {
    let root = fixture("fresh");
    let expected = write_inputs(&root, false);
    let mut config = RunConfig::default();
    config.runtime.cpu_workers = 1;

    let message = run(&root, &config).unwrap();

    assert!(message.contains("9 PS across 3 interferograms"));
    for name in [
        "uw_grid.mat",
        "uw_interp.mat",
        "uw_space_time.mat",
        "uw_phaseuw.mat",
        "phuw2.mat",
    ] {
        assert!(root.join(name).is_file(), "missing {name}");
    }
    for name in ["uw_grid.mat", "uw_interp.mat", "uw_space_time.mat"] {
        let checkpoint = read_mat(root.join(name)).unwrap();
        assert!(checkpoint.contains_key("pystamps_stage6_cache_schema"));
        assert!(checkpoint.contains_key("payload_checksum"));
    }
    let output = read_mat(root.join("phuw2.mat")).unwrap();
    let shape = output.get("ph_uw").and_then(MatValue::shape).unwrap();
    assert_eq!(shape, [9, 3]);
    let values = real_f32(&output, "ph_uw");
    for row in 0..9 {
        assert!((values[row * 3] - expected[row * 3]).abs() < 2.0e-4);
        assert_eq!(values[row * 3 + 1], 0.0);
        assert!((values[row * 3 + 2] - expected[row * 3 + 2]).abs() < 2.0e-4);
    }
    let staging = root.join(".pystamps-tmp");
    assert!(!staging.exists() || fs::read_dir(staging).unwrap().next().is_none());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn sequential_and_parallel_solves_are_bitwise_equal() {
    let sequential = fixture("sequential");
    let parallel = fixture("parallel");
    write_inputs(&sequential, false);
    write_inputs(&parallel, false);
    let mut sequential_config = RunConfig::default();
    sequential_config.runtime.cpu_workers = 1;
    sequential_config.runtime.stage6_ifg_workers = 1;
    let mut parallel_config = RunConfig::default();
    parallel_config.runtime.stage6_ifg_workers = 4;

    run(&sequential, &sequential_config).unwrap();
    run(&parallel, &parallel_config).unwrap();

    for name in ["uw_phaseuw.mat", "phuw2.mat"] {
        let expected = read_mat(sequential.join(name)).unwrap();
        let actual = read_mat(parallel.join(name)).unwrap();
        assert_eq!(actual, expected, "parallel drift in {name}");
    }
    fs::remove_dir_all(sequential).unwrap();
    fs::remove_dir_all(parallel).unwrap();
}

#[test]
fn valid_stage6_checkpoints_are_reused_without_external_tools() {
    let root = fixture("checkpoint");
    write_inputs(&root, false);
    let mut config = RunConfig::default();
    config.runtime.cpu_workers = 1;
    run(&root, &config).unwrap();
    let names = ["uw_grid.mat", "uw_interp.mat", "uw_space_time.mat"];
    let before = names
        .iter()
        .map(|name| fs::read(root.join(name)).unwrap())
        .collect::<Vec<_>>();

    run(&root, &config).unwrap();

    for (name, bytes) in names.iter().zip(before) {
        assert_eq!(fs::read(root.join(name)).unwrap(), bytes);
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn changed_scientific_input_invalidates_all_stage6_checkpoints() {
    let root = fixture("checkpoint-invalidation");
    write_inputs(&root, false);
    let mut config = RunConfig::default();
    config.runtime.cpu_workers = 1;
    run(&root, &config).unwrap();
    let names = ["uw_grid.mat", "uw_interp.mat", "uw_space_time.mat"];
    let before = names
        .iter()
        .map(|name| fs::read(root.join(name)).unwrap())
        .collect::<Vec<_>>();
    let mut params = read_mat(root.join("parms.mat")).unwrap();
    params.insert("unwrap_grid_size".to_owned(), scalar(5.0));
    write_mat(root.join("parms.mat"), &params).unwrap();

    run(&root, &config).unwrap();

    for (name, bytes) in names.iter().zip(before) {
        assert_ne!(fs::read(root.join(name)).unwrap(), bytes, "stale {name}");
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn parseable_cache_corruption_fails_checksum_validation() {
    let root = fixture("cache-checksum");
    write_inputs(&root, false);
    let mut config = RunConfig::default();
    config.runtime.cpu_workers = 1;
    run(&root, &config).unwrap();
    let path = root.join("uw_interp.mat");
    let mut checkpoint = read_mat(&path).unwrap();
    match checkpoint.get_mut("Z").unwrap() {
        MatValue::F64(array) => {
            array.values[0] = if array.values[0] == 1.0 { 2.0 } else { 1.0 };
        }
        _ => panic!("uw_interp.Z is not f64"),
    }
    write_mat(&path, &checkpoint).unwrap();

    let error = run(&root, &config).unwrap_err();

    assert!(error.to_string().contains("payload checksum"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn fresh_goldstein_prefilter_path_produces_finite_phase() {
    let root = fixture("prefilter");
    write_inputs(&root, false);
    let mut params = read_mat(root.join("parms.mat")).unwrap();
    params.insert("unwrap_prefilter_flag".to_owned(), text("y"));
    params.insert("unwrap_gold_n_win".to_owned(), scalar(2.0));
    write_mat(root.join("parms.mat"), &params).unwrap();
    let mut config = RunConfig::default();
    config.runtime.cpu_workers = 1;

    run(&root, &config).unwrap();

    let output = read_mat(root.join("phuw2.mat")).unwrap();
    assert!(real_f32(&output, "ph_uw")
        .iter()
        .all(|value| value.is_finite()));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn unsupported_small_baseline_mode_is_explicit() {
    let root = fixture("unsupported");
    let mut params = BTreeMap::new();
    params.insert("small_baseline_flag".to_owned(), text("y"));
    write_mat(root.join("parms.mat"), &params).unwrap();

    let error = run(&root, &RunConfig::default()).unwrap_err().to_string();

    assert!(error.contains("small_baseline_flag"));
    fs::remove_dir_all(root).unwrap();
}

pub(super) fn write_inputs(root: &Path, small_baseline: bool) -> Vec<f32> {
    let n_ps = 9;
    let n_ifg = 3;
    let mut xy = Vec::with_capacity(n_ps * 3);
    let mut phase = Vec::with_capacity(n_ps * n_ifg);
    let mut angles = Vec::with_capacity(n_ps * n_ifg);
    for row in 0..3 {
        for col in 0..3 {
            xy.extend_from_slice(&[
                (row * 3 + col + 1) as f64,
                col as f64 * 10.0,
                row as f64 * 10.0,
            ]);
            let row_angles = [
                0.05 * col as f32 + 0.03 * row as f32,
                0.0,
                -0.04 * col as f32 + 0.02 * row as f32,
            ];
            for angle in row_angles {
                angles.push(angle);
                phase.push(Complex32::from_polar(1.0, angle));
            }
        }
    }
    let mut ps = BTreeMap::new();
    ps.insert("n_ps".to_owned(), scalar(n_ps as f64));
    ps.insert("n_ifg".to_owned(), scalar(n_ifg as f64));
    ps.insert("master_ix".to_owned(), scalar(2.0));
    ps.insert("day".to_owned(), f64s(vec![n_ifg], vec![0.0, 12.0, 24.0]));
    ps.insert(
        "bperp".to_owned(),
        f64s(vec![n_ifg], vec![-30.0, 0.0, 40.0]),
    );
    ps.insert("xy".to_owned(), f64s(vec![n_ps, 3], xy));
    ps.insert("mean_range".to_owned(), scalar(830_000.0));
    ps.insert("mean_incidence".to_owned(), scalar(23_f64.to_radians()));
    write_mat(root.join("ps2.mat"), &ps).unwrap();
    let mut ph = BTreeMap::new();
    ph.insert(
        "ph".to_owned(),
        MatValue::ComplexF32(MatArray {
            shape: vec![n_ps, n_ifg],
            values: phase,
        }),
    );
    write_mat(root.join("ph2.mat"), &ph).unwrap();
    let mut pm = BTreeMap::new();
    pm.insert(
        "ph_patch".to_owned(),
        MatValue::ComplexF32(MatArray {
            shape: vec![n_ps, n_ifg - 1],
            values: vec![Complex32::new(1.0, 0.0); n_ps * (n_ifg - 1)],
        }),
    );
    pm.insert("K_ps".to_owned(), f64s(vec![n_ps], vec![0.0; n_ps]));
    pm.insert("C_ps".to_owned(), f64s(vec![n_ps], vec![0.0; n_ps]));
    write_mat(root.join("pm2.mat"), &pm).unwrap();
    let mut params = BTreeMap::new();
    params.insert(
        "small_baseline_flag".to_owned(),
        text(if small_baseline { "y" } else { "n" }),
    );
    params.insert("unwrap_method".to_owned(), text("3D_FULL"));
    params.insert("unwrap_la_error_flag".to_owned(), text("y"));
    params.insert("unwrap_spatial_cost_func_flag".to_owned(), text("n"));
    params.insert("unwrap_prefilter_flag".to_owned(), text("n"));
    params.insert("unwrap_grid_size".to_owned(), scalar(10.0));
    params.insert("unwrap_time_win".to_owned(), scalar(36.0));
    params.insert("max_topo_err".to_owned(), scalar(20.0));
    params.insert("lambda".to_owned(), scalar(0.0555));
    write_mat(root.join("parms.mat"), &params).unwrap();
    angles
}

pub(super) fn fixture(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "pystamps-stage6-{label}-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn scalar(value: f64) -> MatValue {
    f64s(vec![1, 1], vec![value])
}

fn f64s(shape: Vec<usize>, values: Vec<f64>) -> MatValue {
    MatValue::F64(MatArray { shape, values })
}

fn text(value: &str) -> MatValue {
    MatValue::U8(MatArray {
        shape: vec![1, value.len()],
        values: value.as_bytes().to_vec(),
    })
}

pub(super) fn real_f32(file: &MatFile, key: &str) -> Vec<f32> {
    match file.get(key).unwrap() {
        MatValue::F32(array) => array.values.clone(),
        MatValue::F64(array) => array.values.iter().map(|&value| value as f32).collect(),
        _ => panic!("{key} is not real"),
    }
}
