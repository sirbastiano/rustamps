use std::collections::BTreeMap;

use num_complex::Complex64;
use pystamps_io::{MatArray, MatFile, MatValue};
use pystamps_pipeline::config::{KeyToleranceConfig, ToleranceConfig, VerificationProfile};

use crate::value_compare::compare_file;

fn array<T>(values: Vec<T>) -> MatArray<T> {
    MatArray {
        shape: vec![values.len(), 1],
        values,
    }
}

fn payload(key: &str, value: MatValue) -> MatFile {
    BTreeMap::from([(key.to_owned(), value)])
}

#[test]
fn float_stored_selection_indices_are_exact() {
    let tolerance = ToleranceConfig {
        rtol: 1.0,
        atol: 1.0,
        ..ToleranceConfig::default()
    };
    for key in [
        "ix",
        "sort_ix",
        "edgs",
        "n_edge",
        "rowix",
        "colix",
        "pystamps_input_fingerprint",
        "random_bperp_fingerprint",
    ] {
        let expected = payload(key, MatValue::F64(array(vec![1.0, 2.0])));
        let observed = payload(key, MatValue::F64(array(vec![1.0, 2.001])));
        assert!(
            !compare_file("structural.mat", &observed, &expected, &tolerance).ok,
            "{key} must be exact"
        );
    }

    let expected = payload("coh_ps", MatValue::F64(array(vec![0.8])));
    let observed = payload("coh_ps", MatValue::F64(array(vec![0.81])));
    assert!(compare_file("pm1.mat", &observed, &expected, &tolerance).ok);
}

#[test]
fn scientific_profile_wraps_residuals_but_not_unwrapped_phase() {
    let tolerance = ToleranceConfig {
        profile: VerificationProfile::Scientific,
        rtol: 0.0,
        atol: 1e-10,
        ..ToleranceConfig::default()
    };
    let expected = payload("ph_res", MatValue::F64(array(vec![0.2])));
    let observed = payload(
        "ph_res",
        MatValue::F64(array(vec![0.2 + std::f64::consts::TAU])),
    );
    assert!(compare_file("pm1.mat", &observed, &expected, &tolerance).ok);

    let expected = payload("ph_uw", MatValue::F64(array(vec![0.2])));
    let observed = payload(
        "ph_uw",
        MatValue::F64(array(vec![0.2 + std::f64::consts::TAU])),
    );
    assert!(!compare_file("phuw2.mat", &observed, &expected, &tolerance).ok);
}

#[test]
fn bounded_outliers_are_inclusive_and_hard_capped() {
    let expected = payload("phase", MatValue::F64(array(vec![0.0; 100])));
    let mut values = vec![0.0; 100];
    values[0] = 0.1;
    values[1] = -0.1;
    let observed = payload("phase", MatValue::F64(array(values)));
    let mut tolerance = ToleranceConfig {
        profile: VerificationProfile::Scientific,
        rtol: 0.0,
        atol: 0.01,
        max_outlier_fraction: 0.02,
        max_abs: Some(0.1),
        ..ToleranceConfig::default()
    };
    let report = compare_file("phase.mat", &observed, &expected, &tolerance);
    assert!(report.ok);
    assert_eq!(report.outliers[0].count, 2);

    tolerance.max_outlier_fraction = 0.019;
    assert!(!compare_file("phase.mat", &observed, &expected, &tolerance).ok);
    tolerance.max_outlier_fraction = 0.02;
    tolerance.max_abs = Some(0.099);
    assert!(!compare_file("phase.mat", &observed, &expected, &tolerance).ok);

    let mut expected_values = vec![f64::NAN; 100];
    expected_values[98..].fill(0.0);
    let mut observed_values = expected_values.clone();
    observed_values[98] = 0.05;
    let expected = payload("phase", MatValue::F64(array(expected_values)));
    let observed = payload("phase", MatValue::F64(array(observed_values)));
    tolerance.max_abs = Some(0.1);
    assert!(!compare_file("phase.mat", &observed, &expected, &tolerance).ok);
}

#[test]
fn longest_per_key_override_applies_only_to_matching_key() {
    let mut tolerance = ToleranceConfig {
        rtol: 0.0,
        atol: 1e-7,
        ..ToleranceConfig::default()
    };
    tolerance.key_tolerances.insert(
        "C_ps".to_owned(),
        KeyToleranceConfig {
            atol: Some(0.01),
            ..KeyToleranceConfig::default()
        },
    );
    let expected = payload("C_ps", MatValue::F64(array(vec![0.0])));
    let observed = payload("C_ps", MatValue::F64(array(vec![0.005])));
    assert!(compare_file("pm1.mat", &observed, &expected, &tolerance).ok);
    let expected = payload("K_ps", MatValue::F64(array(vec![0.0])));
    let observed = payload("K_ps", MatValue::F64(array(vec![0.005])));
    assert!(!compare_file("pm1.mat", &observed, &expected, &tolerance).ok);
}

#[test]
fn undefined_zero_complex_phase_is_not_wrap_equivalent() {
    let expected = payload(
        "dph_noise",
        MatValue::ComplexF64(array(vec![Complex64::new(1.0, 0.0)])),
    );
    let observed = payload(
        "dph_noise",
        MatValue::ComplexF64(array(vec![Complex64::new(0.0, 0.0)])),
    );
    let report = compare_file(
        "scn2.mat",
        &observed,
        &expected,
        &ToleranceConfig::default(),
    );
    assert!(!report.ok);
    assert_eq!(report.max_abs, Some(f64::INFINITY));
}
