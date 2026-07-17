use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use pystamps_io::{write_mat, MatArray, MatFile, MatValue};

use super::run;
use crate::RunConfig;

struct TempDir(PathBuf);

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

impl TempDir {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "pystamps-stage7-options-{}-{nonce}-{}",
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

fn text(value: &str) -> MatValue {
    MatValue::U8(MatArray {
        shape: vec![1, value.len()],
        values: value.as_bytes().to_vec(),
    })
}

fn assert_rejected(key: &str, value: &str, expected: &str) {
    let temp = TempDir::new();
    let mut params = MatFile::new();
    params.insert(key.to_owned(), text(value));
    write_mat(temp.0.join("parms.mat"), &params).unwrap();

    let error = run(&temp.0, &RunConfig::default()).unwrap_err().to_string();

    assert!(error.contains(expected), "unexpected error: {error}");
    assert!(!temp.0.join("scla2.mat").exists());
    assert!(!temp.0.join("scla_smooth2.mat").exists());
}

fn assert_numeric_rejected(key: &str, values: Vec<f64>, expected: &str) {
    let temp = TempDir::new();
    let mut params = MatFile::new();
    params.insert(
        key.to_owned(),
        MatValue::F64(MatArray {
            shape: vec![1, values.len()],
            values,
        }),
    );
    write_mat(temp.0.join("parms.mat"), &params).unwrap();

    let error = run(&temp.0, &RunConfig::default()).unwrap_err().to_string();
    assert!(error.contains(expected), "unexpected error: {error}");
    assert!(!temp.0.join("scla2.mat").exists());
    assert!(!temp.0.join("scla_smooth2.mat").exists());
}

#[test]
fn l1_scla_method_fails_before_artifact_reads_or_writes() {
    assert_rejected("scla_method", "L1", "scla_method=L1");
}

#[test]
fn tropo_subtraction_fails_before_artifact_reads_or_writes() {
    assert_rejected("subtr_tropo", "y", "subtr_tropo='y'");
}

#[test]
fn cartesian_reference_bounds_fail_before_artifact_reads_or_writes() {
    assert_numeric_rejected("ref_x", vec![0.0, 1.0], "ref_x/ref_y");
}

#[test]
fn malformed_lonlat_bounds_fail_before_artifact_reads_or_writes() {
    assert_numeric_rejected("ref_lon", vec![1.0], "exactly two ordered bounds");
    assert_numeric_rejected(
        "ref_lat",
        vec![f64::NEG_INFINITY, 1.0],
        "bounds must be finite",
    );
}

#[test]
fn malformed_circular_reference_fails_before_artifact_reads_or_writes() {
    let temp = TempDir::new();
    let mut params = MatFile::new();
    for (key, values) in [
        ("ref_radius", vec![100.0]),
        ("ref_centre_lonlat", vec![12.0]),
    ] {
        params.insert(
            key.to_owned(),
            MatValue::F64(MatArray {
                shape: vec![1, values.len()],
                values,
            }),
        );
    }
    write_mat(temp.0.join("parms.mat"), &params).unwrap();

    let error = run(&temp.0, &RunConfig::default()).unwrap_err().to_string();
    assert!(
        error.contains("exactly two finite values"),
        "unexpected error: {error}"
    );
    assert!(!temp.0.join("scla2.mat").exists());
    assert!(!temp.0.join("scla_smooth2.mat").exists());
}

#[test]
fn unsupported_deramp_degree_fails_before_artifact_reads_or_writes() {
    let temp = TempDir::new();
    let mut params = MatFile::new();
    params.insert("scla_deramp".to_owned(), text("y"));
    write_mat(temp.0.join("parms.mat"), &params).unwrap();
    let mut degree = MatFile::new();
    degree.insert(
        "degree".to_owned(),
        MatValue::F64(MatArray {
            shape: vec![1, 1],
            values: vec![2.0],
        }),
    );
    write_mat(temp.0.join("deramp_degree.mat"), &degree).unwrap();

    let error = run(&temp.0, &RunConfig::default()).unwrap_err().to_string();
    assert!(
        error.contains("deramp degree 1"),
        "unexpected error: {error}"
    );
    assert!(!temp.0.join("scla2.mat").exists());
}

#[test]
fn legacy_aps_fails_before_artifact_reads_or_writes() {
    let temp = TempDir::new();
    fs::write(temp.0.join("aps2.mat"), []).unwrap();

    let error = run(&temp.0, &RunConfig::default()).unwrap_err().to_string();
    assert!(error.contains("aps2.mat"), "unexpected error: {error}");
    assert!(!temp.0.join("scla2.mat").exists());
}

#[test]
fn stamps_unbounded_lonlat_defaults_remain_supported() {
    let temp = TempDir::new();
    let mut params = MatFile::new();
    for key in ["ref_lon", "ref_lat"] {
        params.insert(
            key.to_owned(),
            MatValue::F64(MatArray {
                shape: vec![1, 2],
                values: vec![f64::NEG_INFINITY, f64::INFINITY],
            }),
        );
    }
    write_mat(temp.0.join("parms.mat"), &params).unwrap();
    let params = crate::native::params::Params::load(&temp.0).unwrap();
    super::reference::validate(&params).unwrap();
}
