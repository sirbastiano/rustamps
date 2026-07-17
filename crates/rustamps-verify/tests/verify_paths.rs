use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use num_complex::Complex64;
use rustamps_io::{write_mat, MatArray, MatValue};
use rustamps_pipeline::config::{ToleranceConfig, VerificationProfile};
use rustamps_verify::{
    verify_paths, verify_paths_through_stage, verify_paths_with_scope, VerifyError,
};

struct Fixture {
    path: PathBuf,
}

impl Fixture {
    fn new(label: &str) -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "rustamps-verify-{label}-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn patch(&self) -> PathBuf {
        let path = self.path.join("PATCH_1");
        fs::create_dir_all(&path).unwrap();
        path
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn array<T>(shape: &[usize], values: Vec<T>) -> MatArray<T> {
    MatArray {
        shape: shape.to_vec(),
        values,
    }
}

fn write(path: impl AsRef<Path>, value: MatValue) {
    write_mat(path, &BTreeMap::from([("value".to_owned(), value)])).unwrap();
}

#[test]
fn verifies_patch_artifacts_and_reports_missing_root_artifacts() {
    let run = Fixture::new("run");
    let golden = Fixture::new("golden");
    let value = MatValue::F64(array(&[1, 2], vec![1.0, 2.0]));
    write(golden.patch().join("ps1.mat"), value.clone());
    write(run.patch().join("ps1.mat"), value.clone());
    write(golden.path.join("ps2.mat"), value);

    let report = verify_paths(&run.path, &golden.path, &ToleranceConfig::default()).unwrap();
    assert_eq!(report.comparisons.len(), 2);
    assert!(report
        .comparisons
        .iter()
        .any(|item| item.path == "PATCH_1/ps1.mat" && item.ok));
    assert!(report
        .comparisons
        .iter()
        .any(|item| item.path == "ps2.mat" && !item.ok));
    assert!(!report.ok());
}

#[test]
fn verifies_cross_precision_real_values_and_complex_tolerance() {
    let run = Fixture::new("run-types");
    let golden = Fixture::new("golden-types");
    write(
        golden.path.join("ps2.mat"),
        MatValue::F64(array(&[2], vec![1.0, 2.0])),
    );
    write(
        run.path.join("ps2.mat"),
        MatValue::F32(array(&[2], vec![1.0, 2.0])),
    );
    let report = verify_paths(&run.path, &golden.path, &ToleranceConfig::default()).unwrap();
    assert!(report.ok());

    write(
        golden.path.join("ph2.mat"),
        MatValue::ComplexF64(array(&[1], vec![Complex64::new(2.0, -3.0)])),
    );
    write(
        run.path.join("ph2.mat"),
        MatValue::ComplexF64(array(&[1], vec![Complex64::new(2.1, -3.0)])),
    );
    let report = verify_paths(&run.path, &golden.path, &ToleranceConfig::default()).unwrap();
    let failed = report
        .comparisons
        .iter()
        .find(|item| item.path == "ph2.mat")
        .unwrap();
    assert!(!failed.ok);
    assert_eq!(failed.failing_key.as_deref(), Some("value"));
}

#[test]
fn empty_or_missing_roots_are_not_successful() {
    let run = Fixture::new("run-empty");
    let golden = Fixture::new("golden-empty");
    let report = verify_paths(&run.path, &golden.path, &ToleranceConfig::default()).unwrap();
    assert!(!report.ok());
    assert_eq!(report.comparisons[0].path, "<dataset>");

    let missing = golden.path.join("missing");
    assert!(matches!(
        verify_paths(&missing, &golden.path, &ToleranceConfig::default()),
        Err(VerifyError::MissingRoot(_))
    ));
}

#[test]
fn final_product_scope_excludes_grid_cache_artifacts_explicitly() {
    let run = Fixture::new("run-final-products");
    let golden = Fixture::new("golden-final-products");
    run.patch();
    golden.patch();
    let value = MatValue::F32(array(&[1], vec![1.0]));
    write(golden.path.join("phuw2.mat"), value.clone());
    write(run.path.join("phuw2.mat"), value.clone());
    write(golden.path.join("uw_grid.mat"), value);

    let all = verify_paths_through_stage(&run.path, &golden.path, &ToleranceConfig::default(), 6)
        .unwrap();
    assert!(!all.ok());

    let final_only = verify_paths_with_scope(
        &run.path,
        &golden.path,
        &ToleranceConfig::default(),
        Some(6),
        true,
    )
    .unwrap();
    assert!(final_only.ok());
    assert_eq!(final_only.comparisons.len(), 1);
    assert_eq!(final_only.comparisons[0].path, "phuw2.mat");
}

#[test]
fn rejects_invalid_outlier_policy_before_reading_artifacts() {
    let run = Fixture::new("run-invalid-tolerance");
    let golden = Fixture::new("golden-invalid-tolerance");
    let tolerance = ToleranceConfig {
        profile: VerificationProfile::Scientific,
        max_outlier_fraction: 0.01,
        max_abs: None,
        ..ToleranceConfig::default()
    };
    assert!(matches!(
        verify_paths(&run.path, &golden.path, &tolerance),
        Err(VerifyError::InvalidTolerance(_))
    ));
}

#[test]
fn reports_tolerated_scientific_outliers() {
    let run = Fixture::new("run-outliers");
    let golden = Fixture::new("golden-outliers");
    let expected = BTreeMap::from([(
        "ph_res".to_owned(),
        MatValue::F64(array(&[100, 1], vec![0.0; 100])),
    )]);
    let mut observed_values = vec![0.0; 100];
    observed_values[0] = 0.05;
    let observed = BTreeMap::from([(
        "ph_res".to_owned(),
        MatValue::F64(array(&[100, 1], observed_values)),
    )]);
    write_mat(golden.path.join("ph2.mat"), &expected).unwrap();
    write_mat(run.path.join("ph2.mat"), &observed).unwrap();
    let tolerance = ToleranceConfig {
        profile: VerificationProfile::Scientific,
        rtol: 0.0,
        atol: 0.001,
        max_outlier_fraction: 0.01,
        max_abs: Some(0.05),
        ..ToleranceConfig::default()
    };
    let report = verify_paths(&run.path, &golden.path, &tolerance).unwrap();
    assert!(report.ok());
    assert_eq!(report.comparisons[0].outliers[0].count, 1);
}

#[test]
fn through_stage_compares_full_golden_without_later_products() {
    let run = Fixture::new("run-through-stage");
    let golden = Fixture::new("golden-through-stage");
    let value = MatValue::F64(array(&[1], vec![1.0]));
    for name in ["ps2.mat", "phuw2.mat"] {
        write(golden.path.join(name), value.clone());
        write(run.path.join(name), value.clone());
    }
    for name in ["scla2.mat", "scn2.mat"] {
        write(golden.path.join(name), value.clone());
    }

    assert!(
        !verify_paths(&run.path, &golden.path, &ToleranceConfig::default())
            .unwrap()
            .ok()
    );
    let scoped =
        verify_paths_through_stage(&run.path, &golden.path, &ToleranceConfig::default(), 6)
            .unwrap();
    assert!(scoped.ok());
    assert_eq!(scoped.comparisons.len(), 2);
}

#[test]
fn rejects_invalid_through_stage() {
    let run = Fixture::new("run-invalid-stage");
    let golden = Fixture::new("golden-invalid-stage");
    assert!(matches!(
        verify_paths_through_stage(&run.path, &golden.path, &ToleranceConfig::default(), 0),
        Err(VerifyError::InvalidStage(0))
    ));
}
