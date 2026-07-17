use std::collections::BTreeMap;
use std::path::Path;

use pystamps_core::stage7::Stage7Outputs;
use pystamps_io::{write_mat, StageTransaction};

use super::super::mat::{f32_array, f64_array};

#[allow(clippy::too_many_arguments)]
pub fn write(
    root: &Path,
    n_ps: usize,
    n_ifg: usize,
    estimated: Stage7Outputs,
    ramp: &[f64],
    smooth_k: Vec<f64>,
    smooth_c: Vec<f32>,
    smooth_phase: Vec<f32>,
) -> Result<(), String> {
    let transaction = StageTransaction::begin(root, "stage7").map_err(|error| error.to_string())?;
    let ramp_shape = if ramp.is_empty() {
        vec![0, 0]
    } else {
        vec![n_ps, n_ifg]
    };
    let mut scla = BTreeMap::new();
    scla.insert(
        "K_ps_uw".to_owned(),
        f64_array(vec![n_ps, 1], estimated.k_ps_uw),
    );
    scla.insert(
        "C_ps_uw".to_owned(),
        f32_array(vec![n_ps, 1], estimated.c_ps_uw),
    );
    scla.insert(
        "ph_scla".to_owned(),
        f32_array(vec![n_ps, n_ifg], estimated.ph_scla),
    );
    scla.insert(
        "ph_ramp".to_owned(),
        f64_array(ramp_shape.clone(), ramp.to_vec()),
    );
    scla.insert(
        "ifg_vcm".to_owned(),
        f64_array(vec![n_ifg, n_ifg], estimated.ifg_vcm),
    );

    let mut smooth = BTreeMap::new();
    smooth.insert("K_ps_uw".to_owned(), f64_array(vec![n_ps, 1], smooth_k));
    smooth.insert("C_ps_uw".to_owned(), f32_array(vec![n_ps, 1], smooth_c));
    smooth.insert(
        "ph_scla".to_owned(),
        f32_array(vec![n_ps, n_ifg], smooth_phase),
    );
    smooth.insert("ph_ramp".to_owned(), f64_array(ramp_shape, ramp.to_vec()));
    write_mat(transaction.path("scla_smooth2.mat"), &smooth).map_err(|error| error.to_string())?;
    write_mat(transaction.path("scla2.mat"), &scla).map_err(|error| error.to_string())?;
    transaction
        .commit(&["scla_smooth2.mat", "scla2.mat"], "scla2.mat")
        .map_err(|error| error.to_string())
}
