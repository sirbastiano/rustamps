use std::collections::BTreeMap;
use std::path::Path;

use rustamps_core::stages::stage2::{
    accumulate_weighted_grid, grid_indices, non_master_phase, normalize_phase_matrix,
    phase_weight_block, NativeStage2Options, NativeStage2Result, PsquareReference, Stage2Input,
};
use rustamps_io::{write_mat, MatArray, MatFile, MatValue, StageTransaction};

use super::super::mat::{complex32_array, f32_array, f64_array, scalar};

pub fn write(
    patch: &Path,
    input: &Stage2Input,
    options: &NativeStage2Options,
    reference: &PsquareReference,
    bperp_fingerprint: u64,
    result: &NativeStage2Result,
) -> Result<(), String> {
    let phase = normalize_phase_matrix(&non_master_phase(input)).0;
    let grid_k = result
        .output
        .filter_k_ps
        .iter()
        .map(|value| if value.is_finite() { *value } else { 0.0 })
        .collect::<Vec<_>>();
    let grid_weighting = result
        .output
        .weighting
        .iter()
        .map(|value| if value.is_finite() { *value } else { 0.0 })
        .collect::<Vec<_>>();
    let ph_weight = phase_weight_block(&phase, &input.bperp_mat, &grid_k, &grid_weighting)
        .map_err(|error| error.to_string())?;
    let layout = grid_indices(&input.xy, options.grid_size).map_err(|error| error.to_string())?;
    let ph_grid =
        accumulate_weighted_grid(&ph_weight, &layout).map_err(|error| error.to_string())?;
    let n_ps = input.phase.rows;
    let n_ifg = ph_weight.cols;
    let random_distribution = result
        .final_psquare
        .as_ref()
        .map(|value| value.scaled_random_distribution.clone())
        .unwrap_or_else(|| reference.random_distribution.clone());

    let mut payload = MatFile::new();
    payload.insert(
        "K_ps".to_owned(),
        f64_array(vec![n_ps, 1], result.output.k_ps.clone()),
    );
    payload.insert(
        "C_ps".to_owned(),
        f64_array(vec![n_ps, 1], result.output.c_ps.clone()),
    );
    payload.insert(
        "coh_ps".to_owned(),
        f64_array(vec![n_ps, 1], result.output.coherence.clone()),
    );
    payload.insert(
        "N_opt".to_owned(),
        f64_array(
            vec![n_ps, 1],
            result
                .output
                .k_ps
                .iter()
                .map(|value| if value.is_finite() { 1.0 } else { 0.0 })
                .collect(),
        ),
    );
    payload.insert(
        "ph_res".to_owned(),
        f32_array(
            vec![n_ps, n_ifg],
            result.output.phase_residual.values.clone(),
        ),
    );
    payload.insert(
        "ph_patch".to_owned(),
        complex32_array(vec![n_ps, n_ifg], result.output.phase_patch.values.clone()),
    );
    payload.insert("step_number".to_owned(), scalar(1.0));
    payload.insert(
        "ph_grid".to_owned(),
        complex32_array(
            vec![ph_grid.rows, ph_grid.cols, ph_grid.planes],
            ph_grid.values,
        ),
    );
    payload.insert(
        "n_trial_wraps".to_owned(),
        MatValue::F32(MatArray {
            shape: vec![1, 1],
            values: vec![options.n_trial_wraps as f32],
        }),
    );
    payload.insert(
        "grid_ij".to_owned(),
        MatValue::I64(MatArray {
            shape: vec![n_ps, 2],
            values: layout
                .indices
                .iter()
                .flat_map(|[row, col]| [*row as i64 + 1, *col as i64 + 1])
                .collect(),
        }),
    );
    payload.insert("grid_size".to_owned(), scalar(f64::from(options.grid_size)));
    payload.insert(
        "low_pass".to_owned(),
        f64_array(
            vec![options.low_pass.rows, options.low_pass.cols],
            options.low_pass.values.clone(),
        ),
    );
    payload.insert("i_loop".to_owned(), scalar(result.output.iterations as f64));
    payload.insert(
        "ph_weight".to_owned(),
        complex32_array(vec![n_ps, n_ifg], ph_weight.values),
    );
    payload.insert(
        "Nr".to_owned(),
        f64_array(vec![1, random_distribution.len()], random_distribution),
    );
    payload.insert(
        "Nr_max_nz_ix".to_owned(),
        scalar(reference.last_nonzero_random_bin_one_based as f64),
    );
    payload.insert(
        "random_bperp_fingerprint".to_owned(),
        scalar(bperp_fingerprint as f64),
    );
    payload.insert(
        "coh_bins".to_owned(),
        f64_array(
            vec![1, reference.coherence_bins.len()],
            reference.coherence_bins.clone(),
        ),
    );
    payload.insert(
        "coh_ps_save".to_owned(),
        f64_array(vec![n_ps, 1], result.output.coherence.clone()),
    );
    payload.insert(
        "gamma_change_save".to_owned(),
        scalar(result.output.gamma_rms),
    );

    commit_pm(patch, payload)
}

fn commit_pm(patch: &Path, payload: BTreeMap<String, MatValue>) -> Result<(), String> {
    let transaction =
        StageTransaction::begin(patch, "stage2").map_err(|error| error.to_string())?;
    write_mat(transaction.path("pm1.mat"), &payload).map_err(|error| error.to_string())?;
    transaction
        .commit(&["pm1.mat"], "pm1.mat")
        .map_err(|error| error.to_string())
}
