use std::path::Path;

use rustamps_core::stages::stage1::Matrix;
use rustamps_core::stages::stage5::{promote_patch, Stage5PatchInput};
use rustamps_io::{read_mat, MatFile};

use crate::{PipelineError, RunConfig};

use super::super::failure;
use super::super::mat::{bools, complex32, numeric_f32, numeric_f64, scalar_f64, shape};
use super::super::params::Params;

pub struct PatchProduct {
    pub ps: MatFile,
    pub rows: rustamps_core::stages::stage5::PromotedPatch,
    pub xy: Vec<f32>,
    pub n_ifg: usize,
    pub master_ix: usize,
}

pub fn run(path: &Path, _config: &RunConfig) -> Result<String, PipelineError> {
    let params = Params::load(path).map_err(|error| failure(5, error))?;
    if params
        .flag("small_baseline_flag", false)
        .map_err(|error| failure(5, error))?
    {
        return Err(failure(5, "small-baseline Stage 5 is not yet supported"));
    }
    let product = load_and_promote(path).map_err(|error| failure(5, error))?;
    let retained = product.rows.rows.len();
    super::write::patch(path, product).map_err(|error| failure(5, error))?;
    Ok(format!("Stage 5 promoted {retained} PS to version 2"))
}

fn load_and_promote(path: &Path) -> Result<PatchProduct, String> {
    let ps = required(path, "ps1.mat")?;
    let pm = required(path, "pm1.mat")?;
    let selection = required(path, "select1.mat")?;
    let weed = required(path, "weed1.mat")?;
    let phase_file = required(path, "ph1.mat")?;
    let n_ps = integer_scalar(&ps, "n_ps")?;
    let n_ifg = integer_scalar(&ps, "n_ifg")?;
    let master_ix = integer_scalar(&ps, "master_ix")?;
    if n_ps == 0 || !(1..=n_ifg).contains(&master_ix) {
        return Err("ps1 contains invalid n_ps/master_ix".to_owned());
    }
    let phase = matrix_complex(&phase_file, "ph", n_ps, n_ifg)?;
    let ij = matrix_f64(&ps, "ij", n_ps, 3)?;
    let lonlat = matrix_f64(&ps, "lonlat", n_ps, 2)?;
    let xy_all = matrix_f32(&ps, "xy", n_ps, 3)?;
    let selected_one = numeric_f64(&selection, "ix")?;
    if selected_one.is_empty() {
        return Err("select1.ix is empty".to_owned());
    }
    let selected = selected_one
        .iter()
        .map(|&value| one_based(value, n_ps, "select1.ix"))
        .collect::<Result<Vec<_>, _>>()?;
    let keep = selection
        .contains_key("keep_ix")
        .then(|| bools(&selection, "keep_ix"))
        .transpose()?
        .unwrap_or_else(|| vec![true; selected.len()]);
    require_len("select1.keep_ix", keep.len(), selected.len())?;
    let selected_kept = selected
        .iter()
        .zip(&keep)
        .filter_map(|(&row, &retain)| retain.then_some(row))
        .collect::<Vec<_>>();
    if !weed.contains_key("ix_weed") {
        return Err("weed1.ix_weed is required for Stage 5 promotion".to_owned());
    }
    let weed_keep = bools(&weed, "ix_weed")?;
    require_len("weed1.ix_weed", weed_keep.len(), selected_kept.len())?;
    let k = select_vector(&numeric_f64(&selection, "K_ps2")?, &keep, "K_ps2")?;
    let c = select_vector(&numeric_f64(&selection, "C_ps2")?, &keep, "C_ps2")?;
    let coherence = select_vector(&numeric_f64(&selection, "coh_ps2")?, &keep, "coh_ps2")?;
    let residual_all = matrix_f32(&selection, "ph_res2", selected.len(), n_ifg - 1)?;
    let residual = select_matrix(&residual_all, &keep);
    let patch_all = matrix_complex(&pm, "ph_patch", n_ps, n_ifg - 1)?;
    let phase_selected = select_rows(&phase, &selected_kept);
    let patch_selected = select_rows(&patch_all, &selected_kept);
    let ij_selected = select_rows(&ij, &selected_kept);
    let lonlat_selected = select_rows(&lonlat, &selected_kept);
    let baseline = optional_matrix_f32(path, "bp1.mat", "bperp_mat", n_ps, n_ifg - 1)?
        .map(|matrix| select_rows(&matrix, &selected_kept));
    let height = optional_vector_f32(path, "hgt1.mat", "hgt", n_ps)?
        .map(|values| select_rows_vector(&values, &selected_kept));
    let look_angle = optional_vector_f64(path, "la1.mat", "la", n_ps)?
        .map(|values| select_rows_vector(&values, &selected_kept));
    let dispersion = optional_vector_f64(path, "da1.mat", "D_A", n_ps)?
        .map(|values| select_rows_vector(&values, &selected_kept));
    let promoted = promote_patch(
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("PATCH"),
        &Stage5PatchInput {
            ij: ij_selected,
            lonlat: lonlat_selected,
            phase: phase_selected,
            k_ps: k,
            c_ps: c,
            coherence,
            phase_patch: patch_selected,
            phase_residual: residual,
            retain: weed_keep.clone(),
            bperp_mat: baseline,
            height,
            look_angle,
            amplitude_dispersion: dispersion,
        },
    )
    .map_err(|error| error.to_string())?;
    let final_sources = selected_kept
        .iter()
        .zip(&weed_keep)
        .filter_map(|(&row, &retain)| retain.then_some(row))
        .collect::<Vec<_>>();
    let xy = select_rows(&xy_all, &final_sources).values;
    Ok(PatchProduct {
        ps,
        rows: promoted,
        xy,
        n_ifg,
        master_ix,
    })
}

fn required(path: &Path, name: &str) -> Result<MatFile, String> {
    let target = path.join(name);
    if !target.is_file() {
        return Err(format!("missing required Stage 5 artifact {name}"));
    }
    read_mat(target).map_err(|error| error.to_string())
}

fn integer_scalar(file: &MatFile, key: &str) -> Result<usize, String> {
    let value = scalar_f64(file, key)?;
    if value < 0.0 || !value.is_finite() || value.fract() != 0.0 {
        Err(format!("{key} is not a non-negative integer"))
    } else {
        Ok(value as usize)
    }
}

fn one_based(value: f64, upper: usize, name: &str) -> Result<usize, String> {
    if !value.is_finite() || value.fract() != 0.0 || value < 1.0 || value > upper as f64 {
        Err(format!("{name} contains invalid index {value}"))
    } else {
        Ok(value as usize - 1)
    }
}

fn matrix_f64(file: &MatFile, key: &str, rows: usize, cols: usize) -> Result<Matrix<f64>, String> {
    require_shape(file, key, rows, cols)?;
    Matrix::new(rows, cols, numeric_f64(file, key)?).map_err(|error| error.to_string())
}

fn matrix_f32(file: &MatFile, key: &str, rows: usize, cols: usize) -> Result<Matrix<f32>, String> {
    require_shape(file, key, rows, cols)?;
    Matrix::new(rows, cols, numeric_f32(file, key)?).map_err(|error| error.to_string())
}

fn matrix_complex(
    file: &MatFile,
    key: &str,
    rows: usize,
    cols: usize,
) -> Result<Matrix<num_complex::Complex32>, String> {
    require_shape(file, key, rows, cols)?;
    Matrix::new(rows, cols, complex32(file, key)?).map_err(|error| error.to_string())
}

fn require_shape(file: &MatFile, key: &str, rows: usize, cols: usize) -> Result<(), String> {
    let dimensions = shape(file, key)?;
    (dimensions == [rows, cols])
        .then_some(())
        .ok_or_else(|| format!("{key} has shape {dimensions:?}; expected [{rows}, {cols}]"))
}

fn require_len(name: &str, actual: usize, expected: usize) -> Result<(), String> {
    (actual == expected)
        .then_some(())
        .ok_or_else(|| format!("{name} has {actual} values; expected {expected}"))
}

fn select_rows<T: Copy>(matrix: &Matrix<T>, rows: &[usize]) -> Matrix<T> {
    Matrix {
        rows: rows.len(),
        cols: matrix.cols,
        values: rows
            .iter()
            .flat_map(|&row| matrix.row(row).iter().copied())
            .collect(),
    }
}

fn select_rows_vector<T: Copy>(values: &[T], rows: &[usize]) -> Vec<T> {
    rows.iter().map(|&row| values[row]).collect()
}

fn select_vector(values: &[f64], keep: &[bool], name: &str) -> Result<Vec<f64>, String> {
    require_len(name, values.len(), keep.len())?;
    Ok(values
        .iter()
        .zip(keep)
        .filter_map(|(&value, &retain)| retain.then_some(value))
        .collect())
}

fn select_matrix<T: Copy>(matrix: &Matrix<T>, keep: &[bool]) -> Matrix<T> {
    let rows = keep
        .iter()
        .enumerate()
        .filter_map(|(row, &retain)| retain.then_some(row))
        .collect::<Vec<_>>();
    select_rows(matrix, &rows)
}

fn optional_matrix_f32(
    path: &Path,
    name: &str,
    key: &str,
    rows: usize,
    cols: usize,
) -> Result<Option<Matrix<f32>>, String> {
    if !path.join(name).is_file() {
        return Ok(None);
    }
    let file = required(path, name)?;
    matrix_f32(&file, key, rows, cols).map(Some)
}

fn optional_vector_f32(
    path: &Path,
    name: &str,
    key: &str,
    rows: usize,
) -> Result<Option<Vec<f32>>, String> {
    if !path.join(name).is_file() {
        return Ok(None);
    }
    let values = numeric_f32(&required(path, name)?, key)?;
    require_len(key, values.len(), rows)?;
    Ok(Some(values))
}

fn optional_vector_f64(
    path: &Path,
    name: &str,
    key: &str,
    rows: usize,
) -> Result<Option<Vec<f64>>, String> {
    if !path.join(name).is_file() {
        return Ok(None);
    }
    let values = numeric_f64(&required(path, name)?, key)?;
    require_len(key, values.len(), rows)?;
    Ok(Some(values))
}
