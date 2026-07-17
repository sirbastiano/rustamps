use std::collections::BTreeMap;

use num_complex::Complex64;
use pystamps_io::mat::MatSparse;
use pystamps_io::{MatArray, MatFile, MatValue};
use pystamps_pipeline::config::ToleranceConfig;

use super::*;

fn array<T>(shape: &[usize], values: Vec<T>) -> MatArray<T> {
    MatArray {
        shape: shape.to_vec(),
        values,
    }
}

fn payload(key: &str, value: MatValue) -> MatFile {
    BTreeMap::from([(key.to_owned(), value)])
}

#[test]
fn compares_complex_values_and_reports_full_difference() {
    let tolerance = ToleranceConfig {
        rtol: 0.0,
        atol: 0.01,
        ..ToleranceConfig::default()
    };
    let expected = payload(
        "phase",
        MatValue::ComplexF64(array(
            &[2],
            vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        )),
    );
    let close = payload(
        "phase",
        MatValue::ComplexF64(array(
            &[2],
            vec![Complex64::new(1.005, 2.0), Complex64::new(3.0, 4.005)],
        )),
    );
    assert!(compare_file("x.mat", &close, &expected, &tolerance).ok);

    let far = payload(
        "phase",
        MatValue::ComplexF64(array(
            &[2],
            vec![Complex64::new(1.02, 2.0), Complex64::new(30.0, 40.0)],
        )),
    );
    let report = compare_file("x.mat", &far, &expected, &tolerance);
    assert!(!report.ok);
    assert_eq!(report.failing_key.as_deref(), Some("phase"));
    assert!(report.max_abs.unwrap() > 40.0);
}

#[test]
fn wraps_real_and_complex_phase_against_zero_tolerance_scale() {
    let tolerance = ToleranceConfig {
        rtol: 100.0,
        atol: 1e-7,
        wrap_keys: vec!["dph_noise".to_owned()],
        ..ToleranceConfig::default()
    };
    let expected = payload("dph_noise", MatValue::F64(array(&[1], vec![0.0])));
    let shifted = payload(
        "dph_noise",
        MatValue::F64(array(&[1], vec![std::f64::consts::TAU + 0.1])),
    );
    assert!(!compare_file("x.mat", &shifted, &expected, &tolerance).ok);

    let expected = payload(
        "dph_noise",
        MatValue::ComplexF64(array(&[1], vec![Complex64::from_polar(1.0, 0.2)])),
    );
    let equivalent = payload(
        "dph_noise",
        MatValue::ComplexF64(array(
            &[1],
            vec![Complex64::from_polar(9.0, std::f64::consts::TAU + 0.2)],
        )),
    );
    assert!(compare_file("x.mat", &equivalent, &expected, &tolerance).ok);
}

#[test]
fn handles_equal_nan_and_infinity_but_rejects_opposite_infinity() {
    let expected = payload(
        "values",
        MatValue::F64(array(&[2], vec![f64::NAN, f64::INFINITY])),
    );
    let same = expected.clone();
    assert!(compare_file("x.mat", &same, &expected, &ToleranceConfig::default()).ok);

    let wrong = payload(
        "values",
        MatValue::F64(array(&[2], vec![f64::NAN, f64::NEG_INFINITY])),
    );
    let report = compare_file("x.mat", &wrong, &expected, &ToleranceConfig::default());
    assert_eq!(report.max_abs, Some(f64::INFINITY));
}

#[test]
fn compares_sparse_structure_and_zero_imaginary_storage() {
    let expected = payload(
        "matrix",
        MatValue::Sparse(MatSparse {
            rows: 3,
            cols: 2,
            row_indices: vec![0, 2],
            column_offsets: vec![0, 1, 2],
            real: vec![2.0, 4.0],
            imag: Some(vec![0.0, 0.0]),
        }),
    );
    let equivalent = payload(
        "matrix",
        MatValue::Sparse(MatSparse {
            rows: 3,
            cols: 2,
            row_indices: vec![0, 2],
            column_offsets: vec![0, 1, 2],
            real: vec![2.0, 4.0],
            imag: None,
        }),
    );
    assert!(compare_file("x.mat", &equivalent, &expected, &ToleranceConfig::default()).ok);

    let mut wrong = equivalent;
    if let MatValue::Sparse(value) = wrong.get_mut("matrix").unwrap() {
        value.row_indices[1] = 1;
    }
    let report = compare_file("x.mat", &wrong, &expected, &ToleranceConfig::default());
    assert_eq!(report.failing_key.as_deref(), Some("matrix.ir"));
}

#[test]
fn recurses_through_structs_cells_and_char_arrays() {
    let expected_nested = MatValue::Struct(BTreeMap::from([(
        "labels".to_owned(),
        MatValue::Cell(vec![MatValue::Char(array(
            &[1, 2],
            vec!['O' as u16, 'K' as u16],
        ))]),
    )]));
    let mut observed_nested = expected_nested.clone();
    if let MatValue::Struct(fields) = &mut observed_nested {
        if let MatValue::Cell(values) = fields.get_mut("labels").unwrap() {
            if let MatValue::Char(text) = &mut values[0] {
                text.values[1] = 'X' as u16;
            }
        }
    }
    let report = compare_file(
        "x.mat",
        &payload("meta", observed_nested),
        &payload("meta", expected_nested),
        &ToleranceConfig::default(),
    );
    assert_eq!(report.failing_key.as_deref(), Some("meta.labels[0]"));
}

#[test]
fn reports_shape_and_missing_nested_field_precisely() {
    let expected = payload(
        "meta",
        MatValue::Struct(BTreeMap::from([(
            "value".to_owned(),
            MatValue::U8(array(&[1], vec![1])),
        )])),
    );
    let observed = payload("meta", MatValue::Struct(BTreeMap::new()));
    let report = compare_file("x.mat", &observed, &expected, &ToleranceConfig::default());
    assert_eq!(report.failing_key.as_deref(), Some("meta.value"));

    let expected = payload("x", MatValue::I32(array(&[1, 2], vec![1, 2])));
    let observed = payload("x", MatValue::I32(array(&[2, 1], vec![1, 2])));
    let report = compare_file("x.mat", &observed, &expected, &ToleranceConfig::default());
    assert!(report.message.starts_with("shape mismatch"));
}
