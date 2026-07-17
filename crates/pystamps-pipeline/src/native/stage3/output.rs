use std::path::Path;

use pystamps_core::stages::stage3::Stage3Output;
use pystamps_io::{write_mat, MatArray, MatFile, MatValue, StageTransaction};

use super::super::mat::{complex32_array, f32_array, f64_array, scalar};

pub(super) struct Metadata<'a> {
    pub coefficients: &'a [f64],
    pub clap_alpha: f64,
    pub clap_beta: f64,
    pub window: f64,
    pub maximum_random: f64,
    pub gamma_stdev_reject: f64,
    pub small_baseline: bool,
    pub ifg_index: &'a [f64],
}

pub(super) fn write(
    patch: &Path,
    output: Stage3Output,
    metadata: &Metadata<'_>,
) -> Result<(), String> {
    let selected = output.selected_ix.len();
    let ifg = output.phase_patch.cols;
    let mut payload = MatFile::new();
    payload.insert(
        "ix".to_owned(),
        f64_array(
            vec![selected, 1],
            output
                .selected_ix
                .into_iter()
                .map(|value| value as f64)
                .collect(),
        ),
    );
    payload.insert(
        "keep_ix".to_owned(),
        MatValue::Bool(MatArray {
            shape: vec![selected, 1],
            values: output.keep_ix,
        }),
    );
    payload.insert(
        "ph_patch2".to_owned(),
        complex32_array(vec![selected, ifg], output.phase_patch.values),
    );
    payload.insert(
        "ph_res2".to_owned(),
        f32_array(
            vec![selected, output.phase_residual.cols],
            output.phase_residual.values,
        ),
    );
    payload.insert(
        "K_ps2".to_owned(),
        f64_array(vec![selected, 1], output.k_ps),
    );
    payload.insert(
        "C_ps2".to_owned(),
        f64_array(vec![selected, 1], output.c_ps),
    );
    payload.insert(
        "coh_ps2".to_owned(),
        f64_array(vec![selected, 1], output.coherence),
    );
    payload.insert(
        "coh_thresh".to_owned(),
        f64_array(vec![selected, 1], output.coherence_threshold),
    );
    let coefficient_shape = if metadata.coefficients.is_empty() {
        vec![0, 0]
    } else {
        vec![1, metadata.coefficients.len()]
    };
    payload.insert(
        "coh_thresh_coeffs".to_owned(),
        f64_array(coefficient_shape, metadata.coefficients.to_vec()),
    );
    payload.insert("clap_alpha".to_owned(), scalar(metadata.clap_alpha));
    payload.insert("clap_beta".to_owned(), scalar(metadata.clap_beta));
    payload.insert("n_win".to_owned(), scalar(metadata.window));
    payload.insert(
        "max_percent_rand".to_owned(),
        f32_array(vec![1, 1], vec![metadata.maximum_random as f32]),
    );
    payload.insert(
        "gamma_stdev_reject".to_owned(),
        scalar(metadata.gamma_stdev_reject),
    );
    let flag = if metadata.small_baseline { "y" } else { "n" };
    payload.insert(
        "small_baseline_flag".to_owned(),
        MatValue::Char(MatArray {
            shape: vec![1, 1],
            values: flag.encode_utf16().collect(),
        }),
    );
    payload.insert(
        "ifg_index".to_owned(),
        f64_array(
            vec![1, metadata.ifg_index.len()],
            metadata.ifg_index.to_vec(),
        ),
    );
    let transaction =
        StageTransaction::begin(patch, "stage3").map_err(|error| error.to_string())?;
    write_mat(transaction.path("select1.mat"), &payload).map_err(|error| error.to_string())?;
    transaction
        .commit(&["select1.mat"], "select1.mat")
        .map_err(|error| error.to_string())
}
