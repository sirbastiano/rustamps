use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use hdf5_pure::mat::{EmptyMarkerEncoding, MatBuilder, Options};
use num_complex::{Complex32, Complex64};
use pystamps_io::mat::{detect_format, MatSparse};
use pystamps_io::{
    read_mat, write_mat, write_mat_with_format, MatArray, MatError, MatFormat, MatValue,
};

fn array<T>(shape: &[usize], values: Vec<T>) -> MatArray<T> {
    MatArray {
        shape: shape.to_vec(),
        values,
    }
}

fn temporary_mat(label: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "pystamps-{label}-{}-{stamp}.mat",
        std::process::id()
    ))
}

#[test]
fn level5_round_trip_preserves_order_types_complex_and_sparse() {
    let path = temporary_mat("v5");
    let payload = BTreeMap::from([
        (
            "bool".to_owned(),
            MatValue::Bool(array(&[2, 2], vec![true, false, false, true])),
        ),
        (
            "complex".to_owned(),
            MatValue::ComplexF64(array(
                &[2, 2],
                vec![
                    Complex64::new(1.0, -1.0),
                    Complex64::new(2.0, -2.0),
                    Complex64::new(3.0, -3.0),
                    Complex64::new(4.0, -4.0),
                ],
            )),
        ),
        (
            "double".to_owned(),
            MatValue::F64(array(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        ),
        (
            "integer".to_owned(),
            MatValue::I16(array(&[1, 3], vec![-2, 0, 7])),
        ),
        (
            "sparse".to_owned(),
            MatValue::Sparse(MatSparse {
                rows: 3,
                cols: 2,
                row_indices: vec![0, 2],
                column_offsets: vec![0, 1, 2],
                real: vec![2.0, 4.0],
                imag: Some(vec![-1.0, 1.0]),
            }),
        ),
    ]);

    write_mat(&path, &payload).unwrap();
    assert_eq!(detect_format(&path).unwrap(), MatFormat::Level5);
    assert_eq!(read_mat(&path).unwrap(), payload);
    fs::remove_file(path).unwrap();
}

#[test]
fn v73_fixture_reads_matlab_shapes_and_column_major_values() {
    let path = temporary_mat("v73");
    let mut builder = MatBuilder::new(Options::default());
    builder
        .write_f64("double", &[2, 3], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        .unwrap();
    builder
        .write_complex_f32(
            "complex",
            &[2, 2],
            &[(1.0, -1.0), (3.0, -3.0), (2.0, -2.0), (4.0, -4.0)],
        )
        .unwrap();
    builder
        .write_logical("logical", &[2, 2], &[1, 1, 0, 0])
        .unwrap();
    builder.write_char("text", "μm").unwrap();
    fs::write(&path, builder.finish().unwrap()).unwrap();

    assert_eq!(detect_format(&path).unwrap(), MatFormat::V73);
    let payload = read_mat(&path).unwrap();
    assert_eq!(
        payload.get("double"),
        Some(&MatValue::F64(array(
            &[2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )))
    );
    assert_eq!(
        payload.get("complex"),
        Some(&MatValue::ComplexF32(array(
            &[2, 2],
            vec![
                Complex32::new(1.0, -1.0),
                Complex32::new(2.0, -2.0),
                Complex32::new(3.0, -3.0),
                Complex32::new(4.0, -4.0),
            ],
        )))
    );
    assert_eq!(
        payload.get("logical"),
        Some(&MatValue::Bool(array(
            &[2, 2],
            vec![true, false, true, false]
        )))
    );
    assert_eq!(
        payload.get("text").unwrap().shape(),
        Some([1, 2].as_slice())
    );
    assert_eq!(
        payload.get("text"),
        Some(&MatValue::Char(array(
            &[1, 2],
            "μm".encode_utf16().collect()
        )))
    );
    fs::remove_file(path).unwrap();
}

#[test]
fn automatic_writer_routes_char_to_v73() {
    let path = temporary_mat("v5-char");
    let payload = BTreeMap::from([(
        "text".to_owned(),
        MatValue::Char(array(&[1, 3], "μm!".encode_utf16().collect())),
    )]);
    write_mat(&path, &payload).unwrap();
    assert_eq!(detect_format(&path).unwrap(), MatFormat::V73);
    assert_eq!(read_mat(&path).unwrap(), payload);
    fs::remove_file(path).unwrap();
}

#[test]
fn forced_level5_char_returns_error_without_panicking() {
    let path = temporary_mat("v5-char-error");
    let payload = BTreeMap::from([(
        "text".to_owned(),
        MatValue::Char(array(&[1, 1], vec!['x' as u16])),
    )]);
    let error = write_mat_with_format(&path, &payload, MatFormat::Level5).unwrap_err();
    assert!(matches!(error, MatError::Unsupported(_)));
    assert!(!path.exists());
}

#[test]
fn failed_mat_rewrite_preserves_existing_file() {
    let path = temporary_mat("atomic-rewrite");
    let original = BTreeMap::from([("x".to_owned(), MatValue::F64(array(&[1, 1], vec![4.0])))]);
    write_mat(&path, &original).unwrap();
    let invalid = BTreeMap::from([(
        "text".to_owned(),
        MatValue::Char(array(&[1, 1], vec!['x' as u16])),
    )]);

    assert!(write_mat_with_format(&path, &invalid, MatFormat::Level5).is_err());
    assert_eq!(read_mat(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}

#[test]
fn explicit_v73_round_trip_preserves_empty_shape() {
    let path = temporary_mat("v73-empty");
    let payload = BTreeMap::from([("empty".to_owned(), MatValue::F64(array(&[0, 3], vec![])))]);
    write_mat_with_format(&path, &payload, MatFormat::V73).unwrap();
    assert_eq!(detect_format(&path).unwrap(), MatFormat::V73);
    assert_eq!(read_mat(&path).unwrap(), payload);
    fs::remove_file(path).unwrap();
}

#[test]
fn automatic_writer_preserves_typed_empty_array() {
    let path = temporary_mat("auto-empty");
    let payload = BTreeMap::from([("empty".to_owned(), MatValue::F64(array(&[64, 0], vec![])))]);
    write_mat(&path, &payload).unwrap();
    assert_eq!(detect_format(&path).unwrap(), MatFormat::V73);
    assert_eq!(read_mat(&path).unwrap(), payload);
    fs::remove_file(path).unwrap();
}

#[test]
fn v73_reads_data_as_dims_empty_markers() {
    let path = temporary_mat("v73-empty-dims");
    let mut options = Options::default();
    options.empty_marker_encoding = EmptyMarkerEncoding::DataAsDims;
    let mut builder = MatBuilder::new(options);
    builder.write_f32("empty", &[0, 4], &[]).unwrap();
    fs::write(&path, builder.finish().unwrap()).unwrap();
    assert_eq!(
        read_mat(&path).unwrap().get("empty"),
        Some(&MatValue::F32(array(&[0, 4], vec![])))
    );
    fs::remove_file(path).unwrap();
}

#[test]
fn level5_reader_accepts_scipy_header_without_comma() {
    let path = temporary_mat("v5-scipy-header");
    let payload = BTreeMap::from([("x".to_owned(), MatValue::F64(array(&[1, 1], vec![2.0])))]);
    write_mat_with_format(&path, &payload, MatFormat::Level5).unwrap();
    let mut bytes = fs::read(&path).unwrap();
    bytes[19] = b' ';
    bytes[20] = b'P';
    fs::write(&path, bytes).unwrap();
    assert_eq!(read_mat(&path).unwrap(), payload);
    fs::remove_file(path).unwrap();
}
