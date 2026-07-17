use std::path::Path;

use num_complex::Complex32;
use rustamps_core::stages::stage1::Matrix;
use rustamps_io::{read_mat, MatFile, MatValue};

use super::super::mat::{complex32, numeric_f64, shape};

pub(super) struct Loaded {
    pub n_ifg: usize,
    pub master: usize,
    pub selected_ix: Vec<usize>,
    pub selection_keep: Vec<bool>,
    pub coherence: Vec<f64>,
    pub k_ps: Vec<f64>,
    pub c_ps: Vec<f64>,
    pub ij: Matrix<i64>,
    pub xy: Matrix<f64>,
    pub phase: Matrix<Complex32>,
    pub height: Option<Vec<f32>>,
    pub bperp: Vec<f64>,
    pub day: Vec<f64>,
}

pub(super) fn load(patch: &Path, small: bool) -> Result<Loaded, String> {
    let select = read_required(patch, "select1.mat")?;
    let ps = read_required(patch, "ps1.mat")?;
    let ph = read_required(patch, "ph1.mat")?;
    let n_ps = scalar_usize(&ps, "n_ps")?;
    let n_ifg = scalar_usize(&ps, "n_ifg")?;
    if n_ps == 0 || n_ifg == 0 {
        return Err("ps1 n_ps and n_ifg must be positive".to_owned());
    }
    let master_one = scalar_usize(&ps, "master_ix")?;
    if !(1..=n_ifg).contains(&master_one) {
        return Err("ps1.master_ix is outside the interferogram range".to_owned());
    }
    let selected_ix = indices(&select, "ix")?;
    if selected_ix.is_empty() {
        return Err("select1.ix is empty".to_owned());
    }
    if selected_ix.iter().any(|&index| index == 0 || index > n_ps) {
        return Err("select1.ix contains an index outside ps1".to_owned());
    }
    let selection_keep = if select.contains_key("keep_ix") {
        booleans(&select, "keep_ix")?
    } else {
        vec![true; selected_ix.len()]
    };
    if selection_keep.len() != selected_ix.len() {
        return Err("select1.keep_ix length does not match ix".to_owned());
    }
    let selected = selected_ix.len();
    let coherence = vector(&select, "coh_ps2", selected)?;
    let k_ps = vector(&select, "K_ps2", selected)?;
    let c_ps = vector(&select, "C_ps2", selected)?;
    let ij_real = real_matrix(&ps, "ij", n_ps, 3)?;
    let mut ij = Vec::with_capacity(ij_real.values.len());
    for value in ij_real.values {
        if !value.is_finite() || value.fract() != 0.0 {
            return Err("ps1.ij must contain finite integers".to_owned());
        }
        ij.push(value as i64);
    }
    let xy = real_matrix(&ps, "xy", n_ps, 3)?;
    let phase = complex_matrix(&ph, "ph", n_ps, n_ifg)?;
    let height = if patch.join("hgt1.mat").is_file() {
        Some(
            vector(&read_required(patch, "hgt1.mat")?, "hgt", n_ps)?
                .into_iter()
                .map(|value| value as f32)
                .collect(),
        )
    } else {
        None
    };
    validate_optional_bp(patch, n_ps, n_ifg)?;
    let bperp = vector(&ps, "bperp", n_ifg)?;
    let day = if small && !ps.contains_key("day") {
        Vec::new()
    } else {
        vector(&ps, "day", n_ifg)?
    };
    Ok(Loaded {
        n_ifg,
        master: master_one - 1,
        selected_ix,
        selection_keep,
        coherence,
        k_ps,
        c_ps,
        ij: Matrix {
            rows: n_ps,
            cols: 3,
            values: ij,
        },
        xy,
        phase,
        height,
        bperp,
        day,
    })
}

fn read_required(patch: &Path, name: &str) -> Result<MatFile, String> {
    let path = patch.join(name);
    if !path.is_file() {
        return Err(format!("missing required Stage 4 artifact {name}"));
    }
    read_mat(path).map_err(|error| error.to_string())
}

fn scalar_usize(file: &MatFile, key: &str) -> Result<usize, String> {
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

fn indices(file: &MatFile, key: &str) -> Result<Vec<usize>, String> {
    numeric_f64(file, key)?
        .into_iter()
        .map(|value| {
            if value.is_finite() && value >= 1.0 && value.fract() == 0.0 {
                Ok(value as usize)
            } else {
                Err(format!("{key} contains an invalid one-based index"))
            }
        })
        .collect()
}

fn booleans(file: &MatFile, key: &str) -> Result<Vec<bool>, String> {
    match file
        .get(key)
        .ok_or_else(|| format!("missing MAT key {key}"))?
    {
        MatValue::Bool(values) => Ok(values.values.clone()),
        _ => numeric_f64(file, key)
            .map(|values| values.into_iter().map(|value| value != 0.0).collect()),
    }
}

fn vector(file: &MatFile, key: &str, length: usize) -> Result<Vec<f64>, String> {
    let values = numeric_f64(file, key)?;
    if values.len() == length {
        Ok(values)
    } else {
        Err(format!(
            "{key} has {} values; expected {length}",
            values.len()
        ))
    }
}

fn real_matrix(file: &MatFile, key: &str, rows: usize, cols: usize) -> Result<Matrix<f64>, String> {
    orient(key, shape(file, key)?, numeric_f64(file, key)?, rows, cols)
}

fn complex_matrix(
    file: &MatFile,
    key: &str,
    rows: usize,
    cols: usize,
) -> Result<Matrix<Complex32>, String> {
    orient(key, shape(file, key)?, complex32(file, key)?, rows, cols)
}

fn orient<T: Copy>(
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
    let values = (0..rows)
        .flat_map(|row| {
            (0..cols).map({
                let values = &values;
                move |col| values[col * rows + row]
            })
        })
        .collect();
    Ok(Matrix { rows, cols, values })
}

fn validate_optional_bp(patch: &Path, n_ps: usize, n_ifg: usize) -> Result<(), String> {
    if !patch.join("bp1.mat").is_file() {
        return Ok(());
    }
    let bp = read_required(patch, "bp1.mat")?;
    let dimensions = shape(&bp, "bperp_mat")?;
    let count = numeric_f64(&bp, "bperp_mat")?.len();
    let valid_shape = dimensions == [n_ps, n_ifg]
        || dimensions == [n_ifg, n_ps]
        || (n_ifg > 1 && (dimensions == [n_ps, n_ifg - 1] || dimensions == [n_ifg - 1, n_ps]));
    if valid_shape && count == dimensions.iter().product::<usize>() {
        Ok(())
    } else {
        Err(format!("bp1.bperp_mat has invalid shape {dimensions:?}"))
    }
}
