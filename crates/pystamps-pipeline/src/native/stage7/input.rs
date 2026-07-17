use std::path::Path;

use pystamps_io::read_mat;

use super::super::mat::{numeric_f32, numeric_f64, shape};

pub struct Loaded {
    pub n_ps: usize,
    pub n_ifg: usize,
    pub master_index: usize,
    pub ph_uw: Vec<f32>,
    pub bperp: Vec<f64>,
    pub day: Vec<f64>,
    pub ifg_std: Vec<f64>,
    pub xy: Vec<f64>,
    pub lonlat: Vec<f64>,
    pub ll0: Vec<f64>,
}

pub fn load(root: &Path) -> Result<Loaded, String> {
    for name in ["ps2.mat", "phuw2.mat", "ifgstd2.mat"] {
        if !root.join(name).is_file() {
            return Err(format!("missing required Stage 7 artifact {name}"));
        }
    }
    let ps = read_mat(root.join("ps2.mat")).map_err(|error| error.to_string())?;
    let unwrapped = read_mat(root.join("phuw2.mat")).map_err(|error| error.to_string())?;
    let dimensions = shape(&unwrapped, "ph_uw")?;
    if dimensions.len() != 2 {
        return Err(format!("phuw2.ph_uw must be 2-D, found {dimensions:?}"));
    }
    let n_ps = dimensions[0];
    let n_ifg = dimensions[1];
    let declared = scalar_usize(&ps, "n_ps")?;
    if declared != n_ps {
        return Err(format!(
            "ps2.n_ps={declared} does not match phuw2 rows={n_ps}"
        ));
    }
    let master_one = scalar_usize(&ps, "master_ix")?;
    if !(1..=n_ifg).contains(&master_one) {
        return Err("ps2.master_ix is outside phuw2 columns".to_owned());
    }
    let master_index = master_one - 1;
    let phase = numeric_f32(&unwrapped, "ph_uw")?;
    require_len("phuw2.ph_uw", phase.len(), n_ps * n_ifg)?;
    let day = numeric_f64(&ps, "day")?;
    require_len("ps2.day", day.len(), n_ifg)?;
    let ifg = read_mat(root.join("ifgstd2.mat")).map_err(|error| error.to_string())?;
    let ifg_std = numeric_f64(&ifg, "ifg_std")?;
    require_len("ifgstd2.ifg_std", ifg_std.len(), n_ifg)?;
    let xy_full = numeric_f64(&ps, "xy")?;
    require_len("ps2.xy", xy_full.len(), n_ps * 3)?;
    let xy = xy_full
        .chunks_exact(3)
        .flat_map(|row| [row[1], row[2]])
        .collect();
    let lonlat = numeric_f64(&ps, "lonlat")?;
    require_len("ps2.lonlat", lonlat.len(), n_ps * 2)?;
    let ll0 = numeric_f64(&ps, "ll0")?;
    let bperp = load_baseline(root, &ps, n_ps, n_ifg, master_index)?;
    Ok(Loaded {
        n_ps,
        n_ifg,
        master_index,
        ph_uw: phase,
        bperp,
        day,
        ifg_std,
        xy,
        lonlat,
        ll0,
    })
}

fn load_baseline(
    root: &Path,
    ps: &pystamps_io::MatFile,
    n_ps: usize,
    n_ifg: usize,
    master: usize,
) -> Result<Vec<f64>, String> {
    if root.join("bp2.mat").is_file() {
        let bp = read_mat(root.join("bp2.mat")).map_err(|error| error.to_string())?;
        let dimensions = shape(&bp, "bperp_mat")?;
        let mut values = numeric_f64(&bp, "bperp_mat")?;
        if dimensions == [n_ps, n_ifg] {
            force_master_zero(&mut values, n_ifg, master);
            return Ok(values);
        }
        if dimensions != [n_ps, n_ifg - 1] {
            return Err(format!("bp2.bperp_mat has invalid shape {dimensions:?}"));
        }
        return Ok(insert_master(&values, n_ps, n_ifg, master));
    }
    let mut nominal = numeric_f64(ps, "bperp")?;
    require_len("ps2.bperp", nominal.len(), n_ifg)?;
    nominal[master] = 0.0;
    Ok((0..n_ps).flat_map(|_| nominal.iter().copied()).collect())
}

fn force_master_zero(values: &mut [f64], columns: usize, master: usize) {
    for row in values.chunks_exact_mut(columns) {
        row[master] = 0.0;
    }
}

fn insert_master(values: &[f64], rows: usize, columns: usize, master: usize) -> Vec<f64> {
    let mut output = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        let source = &values[row * (columns - 1)..(row + 1) * (columns - 1)];
        output.extend_from_slice(&source[..master]);
        output.push(0.0);
        output.extend_from_slice(&source[master..]);
    }
    output
}

fn scalar_usize(file: &pystamps_io::MatFile, key: &str) -> Result<usize, String> {
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

fn require_len(name: &str, actual: usize, expected: usize) -> Result<(), String> {
    (actual == expected)
        .then_some(())
        .ok_or_else(|| format!("{name} has {actual} values; expected {expected}"))
}

#[cfg(test)]
mod tests {
    use super::force_master_zero;

    #[test]
    fn full_baseline_matrix_has_exact_zero_master_column() {
        let mut baseline = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        force_master_zero(&mut baseline, 3, 1);
        assert_eq!(baseline, [1.0, 0.0, 3.0, 4.0, 0.0, 6.0]);
    }
}
