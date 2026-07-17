use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use num_complex::Complex32;
use pystamps_core::stages::stage1::Matrix;
use pystamps_core::stages::stage5::{Stage5Merged, Stage5Row};
use pystamps_io::{read_mat, write_mat, MatFile};

use super::super::mat::{f32_array, f64_array, scalar, shape};

struct Temp(PathBuf);

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

impl Temp {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "pystamps-stage5-{}-{nonce}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn merged_rc2_is_written_ps_major() {
    let temp = Temp::new();
    let mut base = MatFile::new();
    base.insert(
        "bperp".into(),
        f32_array(vec![3, 1], vec![-10.0, 0.0, 10.0]),
    );
    base.insert("day".into(), f64_array(vec![3, 1], vec![1.0, 2.0, 3.0]));
    base.insert("ll0".into(), f64_array(vec![1, 2], vec![12.0, 45.0]));
    for (key, value) in [
        ("master_day", 2.0),
        ("master_ix", 2.0),
        ("n_ifg", 3.0),
        ("n_image", 3.0),
    ] {
        base.insert(key.into(), scalar(value));
    }
    let merged = Stage5Merged {
        rows: vec![Stage5Row {
            ij: [1.0, 10.0, 20.0],
            lonlat: [12.0, 45.0],
            phase: vec![Complex32::new(1.0, 0.0); 3],
            k_ps: 0.01,
            c_ps: 0.02,
            coherence: 0.9,
            phase_patch: vec![Complex32::new(1.0, 0.0); 2],
            phase_residual: vec![0.0; 2],
            bperp: Some(vec![-10.0, 10.0]),
            height: None,
            look_angle: None,
            amplitude_dispersion: None,
        }],
        xy: Matrix::new(1, 3, vec![1.0, 0.0, 0.0]).unwrap(),
        xy_origin: [12.0, 45.0],
    };
    fs::write(temp.0.join("hgt2.mat"), b"stale").unwrap();
    super::write::merged(&temp.0, &base, merged, 2, 3).unwrap();
    let rc = read_mat(temp.0.join("rc2.mat")).unwrap();
    assert_eq!(shape(&rc, "ph_rc").unwrap(), [1, 3]);
    assert_eq!(shape(&rc, "ph_reref").unwrap(), [1, 3]);
    let ps = read_mat(temp.0.join("ps2.mat")).unwrap();
    assert_eq!(
        super::super::mat::numeric_f64(&ps, "ll0").unwrap(),
        [12.0, 45.0]
    );
    assert!(!temp.0.join("hgt2.mat").exists());
    let std = read_mat(temp.0.join("ifgstd2.mat")).unwrap();
    assert_eq!(shape(&std, "ifg_std").unwrap(), [3, 1]);
}

#[test]
fn multi_patch_merge_requires_every_ownership_file_before_patch_io() {
    let temp = Temp::new();
    let first = temp.0.join("PATCH_1");
    let second = temp.0.join("PATCH_2");
    fs::create_dir_all(&first).unwrap();
    fs::create_dir_all(&second).unwrap();
    fs::write(first.join("patch_noover.in"), b"1 10 1 10\n").unwrap();
    let mut params = MatFile::new();
    params.insert("heading".to_owned(), scalar(190.0));
    write_mat(temp.0.join("parms.mat"), &params).unwrap();

    let error = super::run_merged(&temp.0, &crate::RunConfig::default()).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("missing required Stage 5 patch ownership artifact"));
    assert!(message.contains("PATCH_2/patch_noover.in"));
    assert!(!temp.0.join("ps2.mat").exists());
    assert!(!temp.0.join("ifgstd2.mat").exists());
    assert!(!temp.0.join(".pystamps-tmp").exists());
}
