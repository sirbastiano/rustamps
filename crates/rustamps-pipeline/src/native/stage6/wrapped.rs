use std::path::Path;

use num_complex::Complex32;
use rustamps_io::{read_mat, MatFile};

use super::super::mat::{complex32, numeric_f32, numeric_f64, shape};
use super::super::params::Params;

#[allow(clippy::too_many_arguments)]
pub fn build(
    root: &Path,
    params: &Params,
    ps: &MatFile,
    pm: &MatFile,
    ph2: &[Complex32],
    n_ps: usize,
    n_ifg: usize,
    master: usize,
) -> Result<(Vec<Complex32>, Vec<f32>), String> {
    let bperp = baseline_matrix(root, ps, n_ps, n_ifg, master)?;
    let k_ps = optional_vector(pm, "K_ps", n_ps)?;
    let c_ps = optional_vector(pm, "C_ps", n_ps)?;
    let mut phase = if root.join("rc2.mat").is_file() {
        let rc = read_mat(root.join("rc2.mat")).map_err(|error| error.to_string())?;
        let mut values = complex_matrix(&rc, "ph_rc", n_ps, n_ifg)?;
        if let Some(k) = &k_ps {
            for row in 0..n_ps {
                for col in 0..n_ifg {
                    values[row * n_ifg + col] *= polar(k[row] * bperp[row * n_ifg + col]);
                }
            }
        }
        values
    } else {
        require_shape(pm, "ph_patch", &[n_ps, n_ifg - 1])?;
        let patch = complex32(pm, "ph_patch")?;
        let mut values = ph2.to_vec();
        for row in 0..n_ps {
            for col in 0..n_ifg {
                let patch_value = if col == master {
                    Complex32::new(1.0, 0.0)
                } else {
                    let source = if col < master { col } else { col - 1 };
                    patch[row * (n_ifg - 1) + source]
                };
                values[row * n_ifg + col] *= patch_value.conj();
                if let Some(k) = &k_ps {
                    let constant = c_ps.as_ref().map_or(0.0, |values| values[row]);
                    values[row * n_ifg + col] *=
                        polar(-(k[row] * bperp[row * n_ifg + col] + constant));
                }
            }
        }
        values
    };
    let mut restore = vec![0.0_f32; n_ps * n_ifg];
    apply_scla(root, params, &bperp, n_ps, n_ifg, &mut phase, &mut restore)?;
    for value in &mut phase {
        let magnitude = value.norm();
        if magnitude != 0.0 {
            *value /= magnitude;
        }
    }
    Ok((phase, restore))
}

fn baseline_matrix(
    root: &Path,
    ps: &MatFile,
    n_ps: usize,
    n_ifg: usize,
    master: usize,
) -> Result<Vec<f32>, String> {
    if root.join("bp2.mat").is_file() {
        let bp = read_mat(root.join("bp2.mat")).map_err(|error| error.to_string())?;
        require_shape(&bp, "bperp_mat", &[n_ps, n_ifg - 1])?;
        let source = numeric_f32(&bp, "bperp_mat")?;
        let mut output = Vec::with_capacity(n_ps * n_ifg);
        for row in 0..n_ps {
            let values = &source[row * (n_ifg - 1)..(row + 1) * (n_ifg - 1)];
            output.extend_from_slice(&values[..master]);
            output.push(0.0);
            output.extend_from_slice(&values[master..]);
        }
        Ok(output)
    } else {
        let nominal = numeric_f32(ps, "bperp")?;
        if nominal.len() != n_ifg {
            return Err("ps2.bperp must match ph2.ph columns".to_owned());
        }
        Ok((0..n_ps).flat_map(|_| nominal.iter().copied()).collect())
    }
}

fn apply_scla(
    root: &Path,
    params: &Params,
    bperp: &[f32],
    n_ps: usize,
    n_ifg: usize,
    phase: &mut [Complex32],
    restore: &mut [f32],
) -> Result<(), String> {
    let path = root.join("scla_smooth2.mat");
    if !path.is_file() {
        return Ok(());
    }
    let file = read_mat(path).map_err(|error| error.to_string())?;
    let Some(k) = optional_feedback_vector(&file, "K_ps_uw", n_ps)? else {
        return Ok(());
    };
    let c = required_feedback_vector(&file, "C_ps_uw", n_ps)?;
    let ramp = if params.flag("scla_deramp", false)? {
        if !file.contains_key("ph_ramp") {
            return Err("scla_smooth2.ph_ramp is required when scla_deramp='y'".to_owned());
        }
        let dimensions = shape(&file, "ph_ramp")?;
        if dimensions == [n_ps, n_ifg] {
            Some(numeric_f32(&file, "ph_ramp")?)
        } else {
            return Err(format!(
                "scla_smooth2.ph_ramp shape {dimensions:?}; expected [{n_ps}, {n_ifg}]"
            ));
        }
    } else {
        None
    };
    for row in 0..n_ps {
        for col in 0..n_ifg {
            let index = row * n_ifg + col;
            let correction = k[row] * bperp[index];
            phase[index] *= polar(-correction);
            restore[index] += correction;
        }
    }
    for row in 0..n_ps {
        for col in 0..n_ifg {
            let index = row * n_ifg + col;
            phase[index] *= polar(-c[row]);
            restore[index] += c[row];
        }
    }
    if let Some(ramp) = ramp {
        for index in 0..phase.len() {
            phase[index] *= polar(-ramp[index]);
            restore[index] += ramp[index];
        }
    }
    Ok(())
}

fn required_feedback_vector(
    file: &MatFile,
    key: &str,
    expected: usize,
) -> Result<Vec<f32>, String> {
    optional_feedback_vector(file, key, expected)?
        .ok_or_else(|| format!("scla_smooth2.{key} is required when K_ps_uw is valid"))
}

fn optional_feedback_vector(
    file: &MatFile,
    key: &str,
    expected: usize,
) -> Result<Option<Vec<f32>>, String> {
    if !file.contains_key(key) {
        return Ok(None);
    }
    let values = numeric_f64(file, key)?
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    if values.len() == expected {
        Ok(Some(values))
    } else {
        eprintln!(
            "Stage 6: ignoring scla_smooth2.{key} with {} values; expected {expected}",
            values.len()
        );
        Ok(None)
    }
}

fn optional_vector(file: &MatFile, key: &str, expected: usize) -> Result<Option<Vec<f32>>, String> {
    if !file.contains_key(key) {
        return Ok(None);
    }
    let values = numeric_f64(file, key)?
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    if values.len() != expected {
        Err(format!(
            "{key} has {} values; expected {expected}",
            values.len()
        ))
    } else {
        Ok(Some(values))
    }
}

fn require_shape(file: &MatFile, key: &str, expected: &[usize]) -> Result<(), String> {
    let actual = shape(file, key)?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!("{key} has shape {actual:?}; expected {expected:?}"))
    }
}

fn complex_matrix(
    file: &MatFile,
    key: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<Complex32>, String> {
    let dimensions = shape(file, key)?;
    let values = complex32(file, key)?;
    if dimensions == [rows, cols] {
        return Ok(values);
    }
    if dimensions != [cols, rows] {
        return Err(format!(
            "{key} has shape {dimensions:?}; expected [{rows}, {cols}] or its transpose"
        ));
    }
    let mut output = vec![Complex32::new(0.0, 0.0); values.len()];
    for row in 0..rows {
        for col in 0..cols {
            output[row * cols + col] = values[col * rows + row];
        }
    }
    Ok(output)
}

fn polar(angle: f32) -> Complex32 {
    Complex32::new(angle.cos(), angle.sin())
}
