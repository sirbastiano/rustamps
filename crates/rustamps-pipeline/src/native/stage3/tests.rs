use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use num_complex::Complex32;
use rustamps_io::{read_mat, write_mat, MatArray, MatFile, MatValue};

use super::execute;

struct TempDir(PathBuf);

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

impl TempDir {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "rustamps-stage3-{}-{nonce}-{}",
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

fn f64_value(shape: &[usize], values: Vec<f64>) -> MatValue {
    MatValue::F64(MatArray {
        shape: shape.to_vec(),
        values,
    })
}

fn f32_value(shape: &[usize], values: Vec<f32>) -> MatValue {
    MatValue::F32(MatArray {
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

fn fixture(root: &Path, gamma_reject: f64) {
    let baseline = [-10.0, 0.0, 10.0];
    let k = 0.03_f64;
    let phase_row = baseline
        .iter()
        .map(|value| Complex32::new((k * value).cos() as f32, (k * value).sin() as f32))
        .collect::<Vec<_>>();
    let mut ps = MatFile::new();
    ps.insert("n_ps".into(), scalar(2.0));
    ps.insert("n_ifg".into(), scalar(3.0));
    ps.insert("master_ix".into(), scalar(2.0));
    ps.insert(
        "xy".into(),
        f64_value(&[2, 3], vec![1.0, 0.0, 0.0, 2.0, 1000.0, 1000.0]),
    );
    ps.insert("bperp".into(), f64_value(&[3, 1], baseline.to_vec()));
    write(&root.join("ps1.mat"), ps);

    let mut pm = MatFile::new();
    pm.insert("coh_ps".into(), f64_value(&[2, 1], vec![0.9, 0.85]));
    pm.insert("K_ps".into(), f64_value(&[2, 1], vec![k, k]));
    pm.insert("C_ps".into(), f64_value(&[2, 1], vec![0.0, 0.0]));
    pm.insert(
        "ph_patch".into(),
        complex_value(&[2, 3], vec![Complex32::new(1.0, 0.0); 6]),
    );
    pm.insert("ph_res".into(), f32_value(&[2, 3], vec![0.0; 6]));
    pm.insert(
        "ph_grid".into(),
        complex_value(&[4, 4, 3], vec![Complex32::new(1.0, 0.0); 48]),
    );
    pm.insert(
        "grid_ij".into(),
        f64_value(&[2, 2], vec![2.0, 2.0, 3.0, 3.0]),
    );
    let mut low_pass = vec![0.0; 16];
    low_pass[0] = 1.0;
    pm.insert("low_pass".into(), f64_value(&[4, 4], low_pass));
    pm.insert("n_trial_wraps".into(), scalar(1.0));
    pm.insert(
        "coh_bins".into(),
        f64_value(
            &[1, 100],
            (0..100).map(|index| 0.005 + index as f64 * 0.01).collect(),
        ),
    );
    pm.insert("Nr".into(), f64_value(&[1, 100], vec![1.0; 100]));
    write(&root.join("pm1.mat"), pm);

    let mut ph = MatFile::new();
    ph.insert(
        "ph".into(),
        complex_value(&[2, 3], [phase_row.clone(), phase_row].concat()),
    );
    write(&root.join("ph1.mat"), ph);
    let mut bp = MatFile::new();
    bp.insert("bperp_mat".into(), f64_value(&[2, 3], baseline.repeat(2)));
    write(&root.join("bp1.mat"), bp);

    let mut params = MatFile::new();
    params.insert("small_baseline_flag".into(), text("y"));
    params.insert("select_method".into(), text("DENSITY"));
    params.insert("density_rand".into(), scalar(20.0));
    params.insert("quick_est_gamma_flag".into(), text("y"));
    params.insert("select_reest_gamma_flag".into(), text("y"));
    params.insert("gamma_stdev_reject".into(), scalar(gamma_reject));
    params.insert("clap_win".into(), scalar(4.0));
    params.insert("clap_alpha".into(), scalar(1.0));
    params.insert("clap_beta".into(), scalar(0.0));
    params.insert("slc_osf".into(), scalar(1.0));
    write(&root.join("parms.mat"), params);
}

fn assert_invalid_pm(edit: impl FnOnce(&mut MatFile), expected: &str) {
    let temp = TempDir::new();
    fixture(&temp.0, 0.0);
    let mut pm = read_mat(temp.0.join("pm1.mat")).unwrap();
    edit(&mut pm);
    write(&temp.0.join("pm1.mat"), pm);
    let error = execute(&temp.0).unwrap_err();
    assert!(error.contains(expected), "unexpected error: {error}");
    assert!(!temp.0.join("select1.mat").exists());
}

#[test]
fn writes_full_contract_after_default_native_reestimation() {
    let temp = TempDir::new();
    fixture(&temp.0, 0.0);
    assert_eq!(execute(&temp.0).unwrap(), "Stage 3 selected 2 PS");
    let output = read_mat(temp.0.join("select1.mat")).unwrap();
    let expected = [
        "C_ps2",
        "K_ps2",
        "clap_alpha",
        "clap_beta",
        "coh_ps2",
        "coh_thresh",
        "coh_thresh_coeffs",
        "gamma_stdev_reject",
        "ifg_index",
        "ix",
        "keep_ix",
        "max_percent_rand",
        "n_win",
        "ph_patch2",
        "ph_res2",
        "small_baseline_flag",
    ];
    assert_eq!(
        output.keys().map(String::as_str).collect::<Vec<_>>(),
        expected
    );
    assert_eq!(output["ix"].shape(), Some([2, 1].as_slice()));
    assert_eq!(output["K_ps2"].shape(), Some([2, 1].as_slice()));
    assert_eq!(output["ifg_index"].shape(), Some([1, 3].as_slice()));
    assert_eq!(
        super::super::mat::numeric_f64(&output, "ifg_index").unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn rejects_unimplemented_bootstrap_without_partial_artifact() {
    let temp = TempDir::new();
    fixture(&temp.0, 0.1);
    let error = execute(&temp.0).unwrap_err();
    assert!(error.contains("bootstrap rejection"));
    assert!(!temp.0.join("select1.mat").exists());
}

#[test]
fn null_in_dropped_ifg_still_rejects_ps_during_reestimation() {
    let temp = TempDir::new();
    fixture(&temp.0, 0.0);
    let mut params = read_mat(temp.0.join("parms.mat")).unwrap();
    params.insert("drop_ifg_index".into(), f64_value(&[1, 1], vec![2.0]));
    write(&temp.0.join("parms.mat"), params);
    let mut phase = read_mat(temp.0.join("ph1.mat")).unwrap();
    let MatValue::ComplexF32(values) = phase.get_mut("ph").unwrap() else {
        panic!("fixture ph must be complex f32")
    };
    values.values[1] = Complex32::new(0.0, 0.0);
    write(&temp.0.join("ph1.mat"), phase);

    execute(&temp.0).unwrap();
    let output = read_mat(temp.0.join("select1.mat")).unwrap();
    let MatValue::Bool(keep) = &output["keep_ix"] else {
        panic!("keep_ix must be logical")
    };
    assert_eq!(keep.values, [false, true]);
}

#[test]
fn rejects_missing_stage2_distribution_without_partial_output() {
    assert_invalid_pm(
        |pm| {
            pm.remove("Nr");
        },
        "missing required Stage 3 input pm1.Nr",
    );
}

#[test]
fn rejects_malformed_stage2_distributions_without_partial_output() {
    assert_invalid_pm(
        |pm| {
            pm.insert("Nr".into(), f64_value(&[0, 0], Vec::new()));
        },
        "pm1.Nr must be nonempty and finite",
    );
    assert_invalid_pm(
        |pm| {
            pm.insert("coh_bins".into(), f64_value(&[1, 1], vec![f64::NAN]));
        },
        "pm1.coh_bins must be nonempty and finite",
    );
    assert_invalid_pm(
        |pm| {
            pm.insert("Nr".into(), f64_value(&[1, 2], vec![1.0, 1.0]));
        },
        "pm1.Nr and pm1.coh_bins must have equal lengths",
    );
}
