use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use num_complex::Complex32;
use rustamps_io::{read_mat, write_mat, MatArray, MatFile, MatValue};

use super::execute;

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("rustamps-stage4-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn f64_value(shape: &[usize], values: Vec<f64>) -> MatValue {
    MatValue::F64(MatArray {
        shape: shape.to_vec(),
        values,
    })
}

fn complex_value(shape: &[usize], values: Vec<Complex32>) -> MatValue {
    MatValue::ComplexF32(MatArray {
        shape: shape.to_vec(),
        values,
    })
}

fn scalar(value: f64) -> MatValue {
    f64_value(&[1, 1], vec![value])
}

fn text(value: &str) -> MatValue {
    MatValue::U8(MatArray {
        shape: vec![1, value.len()],
        values: value.as_bytes().to_vec(),
    })
}

fn write(path: &Path, payload: MatFile) {
    write_mat(path, &payload).unwrap();
}

fn fixture(root: &Path) {
    let mut ps = MatFile::new();
    ps.insert("n_ps".into(), scalar(3.0));
    ps.insert("n_ifg".into(), scalar(3.0));
    ps.insert("master_ix".into(), scalar(2.0));
    ps.insert(
        "ij".into(),
        f64_value(&[3, 3], vec![1.0, 1.0, 1.0, 2.0, 5.0, 1.0, 3.0, 1.0, 5.0]),
    );
    ps.insert(
        "xy".into(),
        f64_value(
            &[3, 3],
            vec![1.0, 0.0, 0.0, 2.0, 100.0, 0.0, 3.0, 0.0, 100.0],
        ),
    );
    ps.insert("bperp".into(), f64_value(&[3, 1], vec![-10.0, 0.0, 10.0]));
    ps.insert("day".into(), f64_value(&[3, 1], vec![1.0, 2.0, 3.0]));
    write(&root.join("ps1.mat"), ps);

    let mut select = MatFile::new();
    select.insert("ix".into(), f64_value(&[3, 1], vec![1.0, 2.0, 3.0]));
    select.insert(
        "keep_ix".into(),
        MatValue::Bool(MatArray {
            shape: vec![3, 1],
            values: vec![true; 3],
        }),
    );
    select.insert("coh_ps2".into(), f64_value(&[3, 1], vec![0.9, 0.8, 0.7]));
    select.insert("K_ps2".into(), f64_value(&[3, 1], vec![0.0; 3]));
    select.insert("C_ps2".into(), f64_value(&[3, 1], vec![0.0; 3]));
    write(&root.join("select1.mat"), select);

    let mut ph = MatFile::new();
    ph.insert(
        "ph".into(),
        complex_value(&[3, 3], vec![Complex32::new(1.0, 0.0); 9]),
    );
    write(&root.join("ph1.mat"), ph);
    let mut bp = MatFile::new();
    bp.insert("bperp_mat".into(), f64_value(&[3, 3], vec![0.0; 9]));
    write(&root.join("bp1.mat"), bp);

    let mut params = MatFile::new();
    params.insert("small_baseline_flag".into(), text("n"));
    params.insert("weed_neighbours".into(), text("n"));
    params.insert("weed_zero_elevation".into(), text("n"));
    params.insert("weed_standard_dev".into(), scalar(1.0));
    params.insert("weed_max_noise".into(), scalar(1.0));
    params.insert("drop_ifg_index".into(), f64_value(&[1, 1], vec![2.0]));
    write(&root.join("parms.mat"), params);
}

#[test]
fn builds_native_delaunay_edges_and_writes_full_weed_contract() {
    let temp = TempDir::new();
    fixture(&temp.0);
    assert_eq!(
        execute(&temp.0).unwrap(),
        "Stage 4 retained 3/3 selected PS"
    );
    let output = read_mat(temp.0.join("weed1.mat")).unwrap();
    assert_eq!(
        output.keys().map(String::as_str).collect::<Vec<_>>(),
        ["ifg_index", "ix_weed", "ix_weed2", "ps_max", "ps_std"]
    );
    assert_eq!(output["ix_weed"].shape(), Some([3, 1].as_slice()));
    assert_eq!(output["ix_weed2"].shape(), Some([3, 1].as_slice()));
    assert_eq!(
        super::super::mat::numeric_f64(&output, "ifg_index").unwrap(),
        vec![1.0, 3.0]
    );
    assert!(fs::read_dir(temp.0.join(".pystamps-tmp"))
        .unwrap()
        .next()
        .is_none());
}
