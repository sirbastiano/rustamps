use std::fs;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rustamps_core::stage7::Stage7Outputs;
use rustamps_io::{read_mat, write_mat, MatArray, MatFile, MatValue};

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
            "rustamps-stage7-options-{}-{nonce}-{}",
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
    assert!(!temp.0.join("scla2.npz").exists());
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
    assert!(!temp.0.join("scla2.npz").exists());
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

fn write_output_fixture(root: &Path, ramp: &[f64]) -> Result<(), String> {
    super::output::write(
        root,
        2,
        3,
        Stage7Outputs {
            k_ps_uw: vec![1.25, -2.5],
            c_ps_uw: vec![3.5, -4.25],
            ph_scla: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ifg_vcm: vec![1.0, 0.25, 0.5, 0.25, 2.0, 0.75, 0.5, 0.75, 3.0],
        },
        ramp,
        vec![1.0, -2.0],
        vec![3.0, -4.0],
        vec![0.0; 6],
    )
}

#[test]
fn numpy_export_matches_mat_names_types_shapes_and_row_order() {
    for ramp in [vec![], vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5]] {
        let temp = TempDir::new();
        write_output_fixture(&temp.0, &ramp).unwrap();
        assert!(temp.0.join("scla_smooth2.mat").is_file());
        let mat = read_mat(temp.0.join("scla2.mat")).unwrap();
        let mut archive =
            zip::ZipArchive::new(fs::File::open(temp.0.join("scla2.npz")).unwrap()).unwrap();
        assert_eq!(archive.len(), mat.len());
        for (key, value) in mat {
            let (shape, dtype, values): (_, _, Vec<u8>) = match value {
                MatValue::F32(array) => (
                    array.shape,
                    "<f4",
                    array.values.iter().flat_map(|x| x.to_le_bytes()).collect(),
                ),
                MatValue::F64(array) => (
                    array.shape,
                    "<f8",
                    array.values.iter().flat_map(|x| x.to_le_bytes()).collect(),
                ),
                _ => panic!("unexpected MAT type for {key}"),
            };
            let mut bytes = Vec::new();
            archive
                .by_name(&format!("{key}.npy"))
                .unwrap()
                .read_to_end(&mut bytes)
                .unwrap();
            assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
            let data_start = 10 + usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
            assert_eq!(data_start % 64, 0);
            assert_eq!(bytes[data_start - 1], b'\n');
            let header = std::str::from_utf8(&bytes[10..data_start]).unwrap();
            assert_eq!(
                header.trim(),
                format!(
                    "{{'descr': '{dtype}', 'fortran_order': False, 'shape': ({}, {}, ), }}",
                    shape[0], shape[1]
                )
            );
            assert_eq!(&bytes[data_start..], values, "{key}");
        }
    }
}

#[test]
fn failed_numpy_export_preserves_previous_stage7_bundle() {
    let temp = TempDir::new();
    let names = ["scla2.mat", "scla2.npz", "scla_smooth2.mat"];
    for name in names {
        fs::write(temp.0.join(name), format!("previous {name}")).unwrap();
    }
    let error = write_output_fixture(&temp.0, &[1.0]).unwrap_err();
    assert!(
        error.contains("shape mismatch"),
        "unexpected error: {error}"
    );
    for name in names {
        assert_eq!(
            fs::read(temp.0.join(name)).unwrap(),
            format!("previous {name}").as_bytes()
        );
    }
    assert_eq!(
        fs::read_dir(temp.0.join(".pystamps-tmp")).unwrap().count(),
        0
    );
}
