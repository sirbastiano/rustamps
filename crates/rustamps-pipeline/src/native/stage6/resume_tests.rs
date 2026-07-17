use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use rustamps_io::{read_mat, write_mat, MatArray, MatValue};

use crate::RunConfig;

use super::run;
use super::tests::{fixture, real_f32, write_inputs};

#[test]
fn partial_solve_resume_preserves_completed_checkpoint_and_exact_output() {
    let root = fixture("partial-resume");
    write_inputs(&root, false);
    let mut config = RunConfig::default();
    config.runtime.cpu_workers = 1;
    run(&root, &config).unwrap();
    let expected_phase = real_f32(&read_mat(root.join("phuw2.mat")).unwrap(), "ph_uw");
    let expected_msd = real_f32(&read_mat(root.join("phuw2.mat")).unwrap(), "msd");
    let files = solve_files(&root);
    assert_eq!(files.len(), 2);
    let retained = fs::read(&files[0]).unwrap();
    fs::remove_file(&files[1]).unwrap();
    fs::remove_file(root.join("uw_phaseuw.mat")).unwrap();
    fs::remove_file(root.join("phuw2.mat")).unwrap();

    run(&root, &config).unwrap();

    assert_eq!(fs::read(&files[0]).unwrap(), retained);
    assert_eq!(solve_files(&root).len(), 2);
    let output = read_mat(root.join("phuw2.mat")).unwrap();
    assert_eq!(real_f32(&output, "ph_uw"), expected_phase);
    assert_eq!(real_f32(&output, "msd"), expected_msd);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn payload_checksum_rejects_and_recomputes_bit_changed_checkpoint() {
    let root = fixture("checksum");
    write_inputs(&root, false);
    let mut config = RunConfig::default();
    config.runtime.cpu_workers = 1;
    run(&root, &config).unwrap();
    let expected = real_f32(&read_mat(root.join("phuw2.mat")).unwrap(), "ph_uw");
    let checkpoint = solve_files(&root).remove(0);
    let mut corrupt = read_mat(&checkpoint).unwrap();
    match corrupt.get_mut("ph_uw").unwrap() {
        MatValue::F32(array) => array.values[0] += 1.0,
        _ => panic!("solve checkpoint ph_uw is not f32"),
    }
    write_mat(&checkpoint, &corrupt).unwrap();
    fs::remove_file(root.join("uw_phaseuw.mat")).unwrap();
    fs::remove_file(root.join("phuw2.mat")).unwrap();

    run(&root, &config).unwrap();

    assert_eq!(
        real_f32(&read_mat(root.join("phuw2.mat")).unwrap(), "ph_uw"),
        expected
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn changed_scientific_input_uses_a_distinct_solve_cache_namespace() {
    let root = fixture("solve-fingerprint");
    write_inputs(&root, false);
    let mut config = RunConfig::default();
    config.runtime.cpu_workers = 1;
    run(&root, &config).unwrap();
    assert_eq!(solve_directories(&root).len(), 1);
    let mut params = read_mat(root.join("parms.mat")).unwrap();
    if let Some(MatValue::F64(value)) = params.get_mut("unwrap_grid_size") {
        value.values[0] = 5.0;
    } else {
        panic!("unwrap_grid_size is not f64");
    }
    write_mat(root.join("parms.mat"), &params).unwrap();

    run(&root, &config).unwrap();

    assert_eq!(solve_directories(&root).len(), 2);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn balanced_grid_scale_uses_a_distinct_solve_cache_namespace() {
    let root = fixture("grid-scale-fingerprint");
    write_inputs(&root, false);
    let mut strict = RunConfig::default();
    strict.runtime.cpu_workers = 1;
    run(&root, &strict).unwrap();
    assert_eq!(solve_directories(&root).len(), 1);
    let mut balanced = strict;
    balanced.runtime.stage6_grid_scale = 2.0;

    run(&root, &balanced).unwrap();

    assert_eq!(solve_directories(&root).len(), 2);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn scheduling_is_fingerprint_neutral_and_writes_machine_readable_timings() {
    let root = fixture("schedule-fingerprint");
    write_inputs(&root, false);
    let mut config = RunConfig::default();
    config.runtime.stage6_ifg_workers = 1;
    run(&root, &config).unwrap();
    let expected = read_mat(root.join("phuw2.mat")).unwrap();

    for workers in [2, 4] {
        config.runtime.stage6_ifg_workers = workers;
        run(&root, &config).unwrap();
        assert_eq!(read_mat(root.join("phuw2.mat")).unwrap(), expected);
        assert_eq!(solve_directories(&root).len(), 1);
    }

    let timing_path = fs::read_dir(root.join(".pystamps-stage6"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("timing-v1-") && name.ends_with(".json"))
        })
        .unwrap();
    let timing: serde_json::Value =
        serde_json::from_slice(&fs::read(timing_path).unwrap()).unwrap();
    assert_eq!(timing["schema_version"], 1);
    assert_eq!(timing["requested_ifg_workers"], 4);
    assert_eq!(timing["resumed_ifgs"], 2);
    assert_eq!(timing["solved_ifgs"], 0);
    assert_eq!(timing["interferograms"].as_array().unwrap().len(), 2);
    assert!(timing["interferograms"]
        .as_array()
        .unwrap()
        .iter()
        .all(|ifg| ifg["resumed"] == true));
    for ifg in timing["interferograms"].as_array().unwrap() {
        for phase in [
            "prepare_sec",
            "core_sec",
            "decode_sec",
            "initial_flow_sec",
            "initial_label_sec",
            "post_flow_sec",
            "final_label_sec",
            "msd_sec",
            "extract_sec",
            "total_sec",
        ] {
            assert!(ifg[phase].as_f64().is_some(), "missing IFG timing {phase}");
        }
    }
    for phase in [
        "input_sec",
        "grid_sec",
        "interpolation_sec",
        "space_time_sec",
        "costs_sec",
        "solve_wall_sec",
        "solve_output_sec",
        "total_sec",
    ] {
        let seconds = timing["phases"][phase].as_f64().unwrap();
        assert!(seconds.is_finite() && seconds >= 0.0, "invalid {phase}");
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn bounded_flow_passes_use_a_distinct_solve_cache_namespace() {
    let root = fixture("flow-pass-fingerprint");
    write_inputs(&root, false);
    let mut strict = RunConfig::default();
    strict.runtime.cpu_workers = 1;
    run(&root, &strict).unwrap();
    assert_eq!(solve_directories(&root).len(), 1);
    let mut bounded = strict;
    bounded.runtime.stage6_max_flow_passes = 1;

    run(&root, &bounded).unwrap();

    assert_eq!(solve_directories(&root).len(), 2);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn stale_scla_feedback_with_an_old_ps_count_is_ignored() {
    let root = fixture("stale-scla");
    write_inputs(&root, false);
    let mut config = RunConfig::default();
    config.runtime.cpu_workers = 1;
    run(&root, &config).unwrap();
    let expected = real_f32(&read_mat(root.join("phuw2.mat")).unwrap(), "ph_uw");
    let mut scla = BTreeMap::new();
    scla.insert("K_ps_uw".to_owned(), f64s(vec![1], vec![0.5]));
    scla.insert("C_ps_uw".to_owned(), f64s(vec![9], vec![0.25; 9]));
    scla.insert("ph_ramp".to_owned(), f64s(vec![9, 3], vec![1.0; 27]));
    write_mat(root.join("scla_smooth2.mat"), &scla).unwrap();
    let mut params = read_mat(root.join("parms.mat")).unwrap();
    params.insert(
        "scla_deramp".to_owned(),
        MatValue::U8(MatArray {
            shape: vec![1, 1],
            values: vec![b'y'],
        }),
    );
    write_mat(root.join("parms.mat"), &params).unwrap();
    run(&root, &config).unwrap();

    assert_eq!(
        real_f32(&read_mat(root.join("phuw2.mat")).unwrap(), "ph_uw"),
        expected
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn valid_scla_k_without_master_correction_fails_before_output() {
    let root = fixture("partial-scla");
    write_inputs(&root, false);
    let scla = BTreeMap::from([("K_ps_uw".to_owned(), f64s(vec![9], vec![0.5; 9]))]);
    write_mat(root.join("scla_smooth2.mat"), &scla).unwrap();

    let error = run(&root, &RunConfig::default()).unwrap_err().to_string();

    assert!(error.contains("C_ps_uw is required"));
    assert!(!root.join("uw_grid.mat").exists());
    assert!(!root.join("phuw2.mat").exists());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn deramp_feedback_requires_a_complete_ramp_before_output() {
    let root = fixture("missing-ramp");
    write_inputs(&root, false);
    let scla = BTreeMap::from([
        ("K_ps_uw".to_owned(), f64s(vec![9], vec![0.5; 9])),
        ("C_ps_uw".to_owned(), f64s(vec![9], vec![0.25; 9])),
    ]);
    write_mat(root.join("scla_smooth2.mat"), &scla).unwrap();
    let mut params = read_mat(root.join("parms.mat")).unwrap();
    params.insert(
        "scla_deramp".to_owned(),
        MatValue::U8(MatArray {
            shape: vec![1, 1],
            values: vec![b'y'],
        }),
    );
    write_mat(root.join("parms.mat"), &params).unwrap();

    let error = run(&root, &RunConfig::default()).unwrap_err().to_string();

    assert!(error.contains("ph_ramp is required"));
    assert!(!root.join("uw_grid.mat").exists());
    assert!(!root.join("phuw2.mat").exists());
    fs::remove_dir_all(root).unwrap();
}

fn solve_files(root: &Path) -> Vec<PathBuf> {
    let mut files = solve_directories(root)
        .into_iter()
        .flat_map(|directory| {
            fs::read_dir(directory)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|path| path.extension().is_some_and(|value| value == "mat"))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn solve_directories(root: &Path) -> Vec<PathBuf> {
    let mut directories = fs::read_dir(root.join(".pystamps-stage6"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    directories.sort();
    directories
}

fn f64s(shape: Vec<usize>, values: Vec<f64>) -> MatValue {
    MatValue::F64(MatArray { shape, values })
}
