use std::path::Path;

use num_complex::Complex32;
use pystamps_core::stages::stage1::Matrix;
use pystamps_core::stages::stage5::{
    format_merged_rc2, ifg_standard_deviation, rc2_correction, Stage5Merged,
};
use pystamps_io::{write_mat, MatFile, StageTransaction};

use super::super::mat::{complex32_array, f32_array, f64_array, scalar};
use super::write::{base_ps, baseline_matrix, write_optional_patch};

pub fn write(
    root: &Path,
    base: &MatFile,
    merged: Stage5Merged,
    master_ix: usize,
    n_ifg: usize,
) -> Result<(), String> {
    let transaction =
        StageTransaction::begin(root, "stage5-merge").map_err(|error| error.to_string())?;
    let phase = matrix(&merged, n_ifg, |row| &row.phase)?;
    let patch_phase = matrix(&merged, n_ifg - 1, |row| &row.phase_patch)?;
    let baseline = baseline_matrix(&merged.rows, base, n_ifg, master_ix)?;
    let k = merged.rows.iter().map(|row| row.k_ps).collect::<Vec<_>>();
    let c = merged.rows.iter().map(|row| row.c_ps).collect::<Vec<_>>();
    let corrected = rc2_correction(&phase, &patch_phase, &baseline, &k, &c, false, master_ix)
        .map_err(|error| error.to_string())?;
    let full_patch = insert_column(&patch_phase, master_ix - 1, Complex32::new(1.0, 0.0));
    let full_baseline = insert_column(&baseline, master_ix - 1, 0.0);
    let ifg_std = ifg_standard_deviation(&phase, &full_patch, &full_baseline, &k, &c)
        .map_err(|error| error.to_string())?;
    write_files(
        &transaction,
        base,
        &merged,
        phase,
        patch_phase,
        baseline,
        corrected.phase_corrected,
        corrected.phase_rereferenced,
        ifg_std,
    )?;
    transaction
        .commit_with_removals(
            &file_names(&merged),
            "ifgstd2.mat",
            &optional_removals(&merged),
        )
        .map_err(|error| error.to_string())
}

fn matrix(
    merged: &Stage5Merged,
    columns: usize,
    values: impl Fn(&pystamps_core::stages::stage5::Stage5Row) -> &[Complex32],
) -> Result<Matrix<Complex32>, String> {
    Matrix::new(
        merged.rows.len(),
        columns,
        merged.rows.iter().flat_map(values).copied().collect(),
    )
    .map_err(|error| error.to_string())
}

fn insert_column<T: Copy>(matrix: &Matrix<T>, column: usize, value: T) -> Matrix<T> {
    let mut values = Vec::with_capacity(matrix.rows * (matrix.cols + 1));
    for row in 0..matrix.rows {
        values.extend_from_slice(&matrix.row(row)[..column]);
        values.push(value);
        values.extend_from_slice(&matrix.row(row)[column..]);
    }
    Matrix {
        rows: matrix.rows,
        cols: matrix.cols + 1,
        values,
    }
}

#[allow(clippy::too_many_arguments)]
fn write_files(
    transaction: &StageTransaction,
    base: &MatFile,
    merged: &Stage5Merged,
    phase: Matrix<Complex32>,
    patch_phase: Matrix<Complex32>,
    baseline: Matrix<f64>,
    corrected: Matrix<Complex32>,
    rereferenced: Option<Matrix<Complex32>>,
    ifg_std: Vec<f32>,
) -> Result<(), String> {
    let n_ps = merged.rows.len();
    let n_ifg = phase.cols;
    let mut ps = base_ps(base, n_ps, n_ifg)?;
    // StaMPS recomputes merged XY from this midpoint but accidentally retains a
    // patch-local ll0. Keep the coordinate payload internally consistent.
    ps.insert(
        "ll0".to_owned(),
        f64_array(vec![1, 2], merged.xy_origin.to_vec()),
    );
    ps.insert(
        "ij".to_owned(),
        f64_array(
            vec![n_ps, 3],
            merged.rows.iter().flat_map(|row| row.ij).collect(),
        ),
    );
    ps.insert(
        "lonlat".to_owned(),
        f64_array(
            vec![n_ps, 2],
            merged.rows.iter().flat_map(|row| row.lonlat).collect(),
        ),
    );
    ps.insert(
        "xy".to_owned(),
        f32_array(vec![n_ps, 3], merged.xy.values.clone()),
    );
    let mut ph = MatFile::new();
    ph.insert(
        "ph".to_owned(),
        complex32_array(vec![n_ps, n_ifg], phase.values),
    );
    let mut pm = MatFile::new();
    pm.insert(
        "K_ps".to_owned(),
        f64_array(
            vec![n_ps, 1],
            merged.rows.iter().map(|row| row.k_ps).collect(),
        ),
    );
    pm.insert(
        "C_ps".to_owned(),
        f64_array(
            vec![n_ps, 1],
            merged.rows.iter().map(|row| row.c_ps).collect(),
        ),
    );
    pm.insert(
        "coh_ps".to_owned(),
        f64_array(
            vec![n_ps, 1],
            merged.rows.iter().map(|row| row.coherence).collect(),
        ),
    );
    pm.insert(
        "ph_patch".to_owned(),
        complex32_array(vec![n_ps, n_ifg - 1], patch_phase.values),
    );
    pm.insert(
        "ph_res".to_owned(),
        f32_array(
            vec![n_ps, n_ifg - 1],
            merged
                .rows
                .iter()
                .flat_map(|row| row.phase_residual.iter().copied())
                .collect(),
        ),
    );
    let mut bp = MatFile::new();
    bp.insert(
        "bperp_mat".to_owned(),
        f32_array(
            vec![n_ps, n_ifg - 1],
            baseline.values.iter().map(|&value| value as f32).collect(),
        ),
    );
    let mut rc = MatFile::new();
    rc.insert(
        "ph_rc".to_owned(),
        complex32_array(vec![n_ps, n_ifg], ps_major_rc2(&corrected)),
    );
    if let Some(values) = rereferenced {
        rc.insert(
            "ph_reref".to_owned(),
            complex32_array(vec![n_ps, n_ifg], values.values),
        );
    }
    let mut version = MatFile::new();
    version.insert("psver".to_owned(), scalar(2.0));
    let mut standard_deviation = MatFile::new();
    standard_deviation.insert("ifg_std".to_owned(), f32_array(vec![n_ifg, 1], ifg_std));
    for (name, payload) in [
        ("ps2.mat", &ps),
        ("ph2.mat", &ph),
        ("pm2.mat", &pm),
        ("bp2.mat", &bp),
        ("rc2.mat", &rc),
        ("psver.mat", &version),
        ("ifgstd2.mat", &standard_deviation),
    ] {
        write_mat(transaction.path(name), payload).map_err(|error| error.to_string())?;
    }
    let mut ignored = Vec::new();
    write_optional_patch(transaction, &merged.rows, &mut ignored)
}

fn ps_major_rc2(corrected: &Matrix<Complex32>) -> Vec<Complex32> {
    let formatted = format_merged_rc2(corrected);
    let mut output = vec![Complex32::new(0.0, 0.0); corrected.rows * corrected.cols];
    for interferogram in 0..corrected.cols {
        for row in 0..corrected.rows {
            output[row * corrected.cols + interferogram] =
                formatted.values[interferogram * corrected.rows + row];
        }
    }
    output
}

fn file_names(merged: &Stage5Merged) -> Vec<&'static str> {
    let mut files = vec![
        "ps2.mat",
        "ph2.mat",
        "pm2.mat",
        "bp2.mat",
        "rc2.mat",
        "psver.mat",
    ];
    if merged.rows.iter().all(|row| row.height.is_some()) {
        files.push("hgt2.mat");
    }
    if merged.rows.iter().all(|row| row.look_angle.is_some()) {
        files.push("la2.mat");
    }
    if merged
        .rows
        .iter()
        .all(|row| row.amplitude_dispersion.is_some())
    {
        files.push("da2.mat");
    }
    files.push("ifgstd2.mat");
    files
}

fn optional_removals(merged: &Stage5Merged) -> Vec<&'static str> {
    let mut files = vec!["inc2.mat"];
    if !merged.rows.iter().all(|row| row.height.is_some()) {
        files.push("hgt2.mat");
    }
    if !merged.rows.iter().all(|row| row.look_angle.is_some()) {
        files.push("la2.mat");
    }
    if !merged
        .rows
        .iter()
        .all(|row| row.amplitude_dispersion.is_some())
    {
        files.push("da2.mat");
    }
    files
}
