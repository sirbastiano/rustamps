use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rustamps_io::{read_mat, write_mat, MatArray, MatValue};

use super::{reference, tests::write_inputs};
use crate::native::mat::{numeric_f64, shape};
use crate::{NativeExecutor, RunConfig, StageExecutor};

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "rustamps-stage2-snr-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(path.join("PATCH_1")).unwrap();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn snr_bypasses_random_reference_and_preserves_pm_schema() {
    let temp = TempDir::new();
    let patch = temp.0.join("PATCH_1");
    write_inputs(&patch);
    let mut params = read_mat(patch.join("parms.mat")).unwrap();
    params.insert(
        "filter_weighting".to_owned(),
        MatValue::U8(MatArray {
            shape: vec![1, 3],
            values: b"SNR".to_vec(),
        }),
    );
    write_mat(patch.join("parms.mat"), &params).unwrap();

    let summary = NativeExecutor
        .run_patch(2, &patch, &RunConfig::default())
        .unwrap();
    assert!(summary.contains("random reference bypassed for SNR"));

    let pm = read_mat(patch.join("pm1.mat")).unwrap();
    assert_eq!(shape(&pm, "Nr").unwrap(), [1, 100]);
    assert_eq!(shape(&pm, "coh_bins").unwrap(), [1, 100]);
    assert_eq!(numeric_f64(&pm, "Nr").unwrap(), vec![0.0; 100]);
    assert_eq!(numeric_f64(&pm, "Nr_max_nz_ix").unwrap(), [1.0]);
    let wraps = numeric_f64(&pm, "n_trial_wraps").unwrap()[0];
    let fingerprint = numeric_f64(&pm, "random_bperp_fingerprint").unwrap()[0] as u64;
    assert!(
        reference::load_pm_cache(&patch, &reference::coherence_bins(), wraps, fingerprint,)
            .is_none()
    );
}
