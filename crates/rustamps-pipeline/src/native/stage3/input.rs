use std::path::Path;

use num_complex::Complex32;
use rustamps_core::stages::stage1::Matrix;
use rustamps_io::{read_mat, MatFile};

use super::super::mat::{complex32, numeric_f64, shape};

pub(super) struct Initial {
    pub ps: MatFile,
    pub pm: MatFile,
    pub n_ps: usize,
    pub n_ifg: usize,
    pub master: usize,
    pub xy: Matrix<f64>,
    pub coherence: Vec<f64>,
    pub k_ps: Vec<f64>,
    pub c_ps: Vec<f64>,
    pub amplitude_dispersion: Vec<f64>,
    pub phase_patch: Matrix<Complex32>,
    pub phase_residual: Matrix<f32>,
    pub coherence_bins: Vec<f64>,
    pub random_distribution: Vec<f64>,
}

pub(super) fn load(patch: &Path, small_baseline: bool) -> Result<Initial, String> {
    let ps = read_required(patch, "ps1.mat")?;
    let pm = read_required(patch, "pm1.mat")?;
    let n_ps = scalar_usize(&ps, "n_ps")?;
    let n_ifg = scalar_usize(&ps, "n_ifg")?;
    if n_ps == 0 || n_ifg == 0 {
        return Err("ps1 n_ps and n_ifg must be positive".to_owned());
    }
    let master_one = scalar_usize(&ps, "master_ix")?;
    if !(1..=n_ifg).contains(&master_one) {
        return Err("ps1.master_ix is outside the interferogram range".to_owned());
    }
    let master = master_one - 1;
    let work_ifg = n_ifg - usize::from(!small_baseline);
    let xy = real_matrix(&ps, "xy", n_ps, 3)?;
    let coherence = vector(&pm, "coh_ps", n_ps)?;
    let k_ps = vector(&pm, "K_ps", n_ps)?;
    let c_ps = vector(&pm, "C_ps", n_ps)?;
    let phase_patch = complex_matrix(&pm, "ph_patch", n_ps, work_ifg)?;
    let phase_residual = real_matrix(&pm, "ph_res", n_ps, work_ifg).map(|matrix| Matrix {
        rows: matrix.rows,
        cols: matrix.cols,
        values: matrix
            .values
            .into_iter()
            .map(|value| value as f32)
            .collect(),
    })?;
    let amplitude_dispersion = if patch.join("da1.mat").is_file() {
        vector(&read_required(patch, "da1.mat")?, "D_A", n_ps)?
    } else {
        vec![1.0; n_ps]
    };
    let coherence_bins = required_finite_vector(&pm, "coh_bins")?;
    let random_distribution = required_finite_vector(&pm, "Nr")?;
    if random_distribution.len() != coherence_bins.len() {
        return Err("pm1.Nr and pm1.coh_bins must have equal lengths".to_owned());
    }
    Ok(Initial {
        ps,
        pm,
        n_ps,
        n_ifg,
        master,
        xy,
        coherence,
        k_ps,
        c_ps,
        amplitude_dispersion,
        phase_patch,
        phase_residual,
        coherence_bins,
        random_distribution,
    })
}

pub(super) fn read_required(patch: &Path, name: &str) -> Result<MatFile, String> {
    let path = patch.join(name);
    if !path.is_file() {
        return Err(format!("missing required Stage 3 artifact {name}"));
    }
    read_mat(path).map_err(|error| error.to_string())
}

pub(super) fn scalar_usize(file: &MatFile, key: &str) -> Result<usize, String> {
    let value = numeric_f64(file, key)?
        .first()
        .copied()
        .ok_or_else(|| format!("{key} is empty"))?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        Err(format!("{key} is not a non-negative integer"))
    } else {
        Ok(value as usize)
    }
}

pub(super) fn vector(file: &MatFile, key: &str, length: usize) -> Result<Vec<f64>, String> {
    let values = numeric_f64(file, key)?;
    if values.len() != length {
        Err(format!(
            "{key} has {} values; expected {length}",
            values.len()
        ))
    } else {
        Ok(values)
    }
}

fn required_finite_vector(file: &MatFile, key: &str) -> Result<Vec<f64>, String> {
    if !file.contains_key(key) {
        return Err(format!("missing required Stage 3 input pm1.{key}"));
    }
    let values = numeric_f64(file, key)?;
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("pm1.{key} must be nonempty and finite"));
    }
    Ok(values)
}

pub(super) fn real_matrix(
    file: &MatFile,
    key: &str,
    rows: usize,
    cols: usize,
) -> Result<Matrix<f64>, String> {
    let dimensions = shape(file, key)?;
    let values = numeric_f64(file, key)?;
    orient_matrix(key, dimensions, values, rows, cols)
}

pub(super) fn complex_matrix(
    file: &MatFile,
    key: &str,
    rows: usize,
    cols: usize,
) -> Result<Matrix<Complex32>, String> {
    let dimensions = shape(file, key)?;
    let values = complex32(file, key)?;
    orient_matrix(key, dimensions, values, rows, cols)
}

fn orient_matrix<T: Copy>(
    key: &str,
    dimensions: Vec<usize>,
    values: Vec<T>,
    rows: usize,
    cols: usize,
) -> Result<Matrix<T>, String> {
    if dimensions == [rows, cols] {
        return Ok(Matrix { rows, cols, values });
    }
    if dimensions != [cols, rows] {
        return Err(format!(
            "{key} has shape {dimensions:?}; expected [{rows}, {cols}]"
        ));
    }
    let mut transposed = Vec::with_capacity(values.len());
    for row in 0..rows {
        for col in 0..cols {
            transposed.push(values[col * rows + row]);
        }
    }
    Ok(Matrix {
        rows,
        cols,
        values: transposed,
    })
}
