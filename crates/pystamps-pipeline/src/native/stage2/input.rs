use std::path::Path;

use pystamps_core::stages::stage1::{Complex32, Matrix};
use pystamps_core::stages::stage2::Stage2Input;
use pystamps_io::{read_mat, MatFile};

use super::super::mat::{complex32, numeric_f32, numeric_f64, shape};

pub struct Loaded {
    pub input: Stage2Input,
    pub nominal_bperp: Vec<f64>,
    pub mean_incidence: f64,
}

pub fn load(patch: &Path) -> Result<Loaded, String> {
    for name in ["ps1.mat", "ph1.mat", "bp1.mat"] {
        if !patch.join(name).is_file() {
            return Err(format!("missing required Stage 2 artifact {name}"));
        }
    }
    let ps = read_mat(patch.join("ps1.mat")).map_err(|error| error.to_string())?;
    let n_ps = scalar_usize(&ps, "n_ps")?;
    if n_ps == 0 {
        return Err("ps1.n_ps must be positive".to_owned());
    }
    let master_ix = scalar_usize(&ps, "master_ix")?;
    let ph_file = read_mat(patch.join("ph1.mat")).map_err(|error| error.to_string())?;
    let phase = complex_matrix(&ph_file, "ph", n_ps)?;
    if phase.cols < 2 {
        return Err("ph1.ph must contain a master and at least one interferogram".to_owned());
    }
    if phase
        .values
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("ph1.ph contains non-finite phase values".to_owned());
    }
    if !(1..=phase.cols).contains(&master_ix) {
        return Err(format!(
            "ps1.master_ix={master_ix} is outside ph1.ph columns={}",
            phase.cols
        ));
    }

    let bperp = numeric_f64(&ps, "bperp")?;
    if bperp.len() != phase.cols || bperp.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "ps1.bperp must contain {} finite values; found {}",
            phase.cols,
            bperp.len(),
        ));
    }
    let nominal_bperp = bperp
        .into_iter()
        .enumerate()
        .filter_map(|(index, value)| (index + 1 != master_ix).then_some(value))
        .collect::<Vec<_>>();

    let bp_file = read_mat(patch.join("bp1.mat")).map_err(|error| error.to_string())?;
    let bperp_mat = baseline_matrix(&bp_file, n_ps, phase.cols, master_ix)?;
    let xy = f32_matrix(&ps, "xy", n_ps, 3)?;
    if bperp_mat.values.iter().any(|value| !value.is_finite())
        || xy.values.iter().any(|value| !value.is_finite())
    {
        return Err("Stage 2 geometry contains non-finite values".to_owned());
    }
    let amplitude_dispersion = load_da(patch, n_ps)?;
    let mean_incidence = load_mean_incidence(patch, &ps)?;
    Ok(Loaded {
        input: Stage2Input {
            phase,
            bperp_mat,
            xy,
            amplitude_dispersion,
            master_ix,
            small_baseline: false,
        },
        nominal_bperp,
        mean_incidence,
    })
}

fn baseline_matrix(
    file: &MatFile,
    n_ps: usize,
    n_ifg_full: usize,
    master_ix: usize,
) -> Result<Matrix<f64>, String> {
    let mut matrix = f64_matrix_for_rows(file, "bperp_mat", n_ps)?;
    if matrix.cols == n_ifg_full {
        let mut values = Vec::with_capacity(n_ps * (n_ifg_full - 1));
        for row in 0..n_ps {
            values.extend(
                matrix
                    .row(row)
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &value)| (index + 1 != master_ix).then_some(value)),
            );
        }
        matrix = Matrix::new(n_ps, n_ifg_full - 1, values).map_err(|error| error.to_string())?;
    }
    if matrix.cols != n_ifg_full - 1 {
        return Err(format!(
            "bp1.bperp_mat has {} columns; expected {} or {n_ifg_full}",
            matrix.cols,
            n_ifg_full - 1
        ));
    }
    Ok(matrix)
}

fn load_da(patch: &Path, n_ps: usize) -> Result<Vec<f64>, String> {
    let path = patch.join("da1.mat");
    if !path.is_file() {
        return Ok(vec![1.0; n_ps]);
    }
    let file = read_mat(path).map_err(|error| error.to_string())?;
    let values = numeric_f64(&file, "D_A")?;
    if values.len() != n_ps
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(format!(
            "da1.D_A must contain {n_ps} finite, non-negative values"
        ));
    }
    Ok(values)
}

fn load_mean_incidence(patch: &Path, ps: &MatFile) -> Result<f64, String> {
    if let Some(value) = angle_file_mean(patch, "inc1.mat", "inc", true)? {
        return Ok(value);
    }
    if let Some(value) = optional_scalar(ps, "mean_incidence").filter(|value| value.is_finite()) {
        return Ok(value);
    }
    if let Some(value) = angle_file_mean(patch, "la1.mat", "la", false)? {
        return Ok(value + 0.052);
    }
    Ok(21_f64.to_radians())
}

fn angle_file_mean(
    patch: &Path,
    name: &str,
    key: &str,
    reject_zero: bool,
) -> Result<Option<f64>, String> {
    let path = patch.join(name);
    if !path.is_file() {
        return Ok(None);
    }
    let file = read_mat(path).map_err(|error| error.to_string())?;
    let values = numeric_f64(&file, key)?;
    let valid = values
        .into_iter()
        .filter(|value| value.is_finite() && (!reject_zero || *value != 0.0))
        .collect::<Vec<_>>();
    Ok((!valid.is_empty()).then(|| valid.iter().sum::<f64>() / valid.len() as f64))
}

fn complex_matrix(file: &MatFile, key: &str, rows: usize) -> Result<Matrix<Complex32>, String> {
    let dimensions = shape(file, key)?;
    orient(complex32(file, key)?, dimensions, rows, key)
}

fn f64_matrix_for_rows(file: &MatFile, key: &str, rows: usize) -> Result<Matrix<f64>, String> {
    let dimensions = shape(file, key)?;
    orient(numeric_f64(file, key)?, dimensions, rows, key)
}

fn f32_matrix(file: &MatFile, key: &str, rows: usize, cols: usize) -> Result<Matrix<f32>, String> {
    let dimensions = shape(file, key)?;
    let matrix = orient(numeric_f32(file, key)?, dimensions, rows, key)?;
    if matrix.cols != cols {
        return Err(format!(
            "{key} has {} columns; expected {cols}",
            matrix.cols
        ));
    }
    Ok(matrix)
}

fn orient<T: Copy>(
    values: Vec<T>,
    shape: Vec<usize>,
    rows: usize,
    key: &str,
) -> Result<Matrix<T>, String> {
    if shape.len() != 2 {
        return Err(format!("{key} must be 2-D, found shape {shape:?}"));
    }
    if shape[0] == rows {
        return Matrix::new(shape[0], shape[1], values).map_err(|error| error.to_string());
    }
    if shape[1] != rows {
        return Err(format!(
            "{key} shape {shape:?} is incompatible with n_ps={rows}"
        ));
    }
    let mut transposed = Vec::with_capacity(values.len());
    for row in 0..shape[1] {
        for col in 0..shape[0] {
            transposed.push(values[col * shape[1] + row]);
        }
    }
    Matrix::new(shape[1], shape[0], transposed).map_err(|error| error.to_string())
}

fn scalar_usize(file: &MatFile, key: &str) -> Result<usize, String> {
    let value = optional_scalar(file, key).ok_or_else(|| format!("missing MAT key {key}"))?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        Err(format!("{key} is not a non-negative integer"))
    } else {
        Ok(value as usize)
    }
}

fn optional_scalar(file: &MatFile, key: &str) -> Option<f64> {
    numeric_f64(file, key).ok()?.first().copied()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use pystamps_io::{write_mat, MatArray, MatValue};

    use super::*;

    #[test]
    fn exact_incidence_precedes_legacy_look_angle() {
        let root = std::env::temp_dir().join(format!(
            "pystamps-stage2-incidence-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let mut ps = MatFile::new();
        ps.insert(
            "mean_incidence".to_owned(),
            MatValue::F64(MatArray {
                shape: vec![1, 1],
                values: vec![0.5],
            }),
        );
        let mut la = MatFile::new();
        la.insert(
            "la".to_owned(),
            MatValue::F64(MatArray {
                shape: vec![1, 1],
                values: vec![0.4],
            }),
        );
        write_mat(root.join("la1.mat"), &la).unwrap();
        assert_eq!(load_mean_incidence(&root, &ps).unwrap(), 0.5);
        let _ = fs::remove_dir_all(root);
    }
}
