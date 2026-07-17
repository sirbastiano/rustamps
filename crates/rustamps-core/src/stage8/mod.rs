//! Pure-Rust implementation of legacy `ps_scn_filt`.
//!
//! Matrices are row-major and all indices are zero-based.

mod spatial;
mod temporal;

use std::fmt;

use rayon::prelude::*;

use crate::stage7::qr::QrSolver;

#[derive(Clone, Debug)]
pub struct ScnInputs<'a> {
    pub ph_uw: &'a [f32],
    pub xy: &'a [f64],
    pub day: &'a [f64],
    pub n_ps: usize,
    pub n_ifg: usize,
    pub ph_scla: Option<&'a [f32]>,
    pub c_ps_uw: Option<&'a [f32]>,
    pub scla_ramp: Option<&'a [f64]>,
}

#[derive(Clone, Debug)]
pub struct ScnConfig<'a> {
    pub master_index: usize,
    pub unwrap_indices: &'a [usize],
    pub deramp_indices: &'a [usize],
    pub time_window: f64,
    pub wavelength: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScnOutputs {
    pub ph_scn_slave: Vec<f64>,
    pub ph_hpt: Vec<f32>,
    pub ph_ramp: Vec<f64>,
    pub n_unwrap: usize,
    pub n_deramp: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScnError(String);

impl ScnError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ScnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ScnError {}

pub fn estimate_scn(input: &ScnInputs<'_>, config: &ScnConfig<'_>) -> Result<ScnOutputs, ScnError> {
    validate(input, config)?;
    let unwrap = normalized_indices(config.unwrap_indices, input.n_ifg, "unwrap_indices")?;
    if unwrap.is_empty() {
        return Err(ScnError::new("unwrap_indices must not be empty"));
    }
    let requested_deramp =
        normalized_indices(config.deramp_indices, input.n_ifg, "deramp_indices")?;
    let deramp = requested_deramp
        .into_iter()
        .filter(|index| unwrap.binary_search(index).is_ok())
        .collect::<Vec<_>>();
    let deramp_local = deramp
        .iter()
        .map(|index| unwrap.binary_search(index).expect("intersected index"))
        .collect::<Vec<_>>();
    let mut ramp_for_column = vec![None; unwrap.len()];
    for (ramp_column, &phase_column) in deramp_local.iter().enumerate() {
        ramp_for_column[phase_column] = Some(ramp_column);
    }

    (0..input.n_ps).into_par_iter().try_for_each(|row| {
        for &full_column in &unwrap {
            let value = corrected_value(input, row, full_column);
            if !value.is_finite() {
                return Err(ScnError::new("corrected unwrapped phase contains infinity"));
            }
        }
        Ok(())
    })?;

    let ramp_coefficients = if deramp.is_empty() {
        Vec::new()
    } else {
        let mut design = Vec::with_capacity(input.n_ps * 3);
        for row in 0..input.n_ps {
            design.extend_from_slice(&[1.0, input.xy[row * 2], input.xy[row * 2 + 1]]);
        }
        let solver = QrSolver::factor(&design, input.n_ps, 3).map_err(ScnError::new)?;
        deramp
            .iter()
            .map(|&full_column| solver.solve_with(|row| corrected_value(input, row, full_column)))
            .collect::<Vec<_>>()
    };

    let mut ph_ramp = vec![0.0; input.n_ps * deramp.len()];
    if !deramp.is_empty() {
        ph_ramp
            .par_chunks_mut(deramp.len())
            .enumerate()
            .for_each(|(row, output)| {
                let x = input.xy[row * 2];
                let y = input.xy[row * 2 + 1];
                for (column, coefficients) in ramp_coefficients.iter().enumerate() {
                    output[column] = coefficients[0] + coefficients[1] * x + coefficients[2] * y;
                }
            });
    }

    let selected_day = unwrap
        .iter()
        .map(|&index| input.day[index])
        .collect::<Vec<_>>();
    let local_master = unwrap.binary_search(&config.master_index).ok();
    let weights = temporal::gaussian_weights(&selected_day, local_master, config.time_window)
        .map_err(ScnError::new)?;

    let mut reference_values = vec![0.0; unwrap.len()];
    fill_corrected_row(input, &unwrap, 0, &mut reference_values);
    for (phase_column, ramp_column) in ramp_for_column.iter().enumerate() {
        if let Some(ramp_column) = ramp_column {
            reference_values[phase_column] -= ph_ramp[*ramp_column];
        }
    }
    let mut reference_high_pass = vec![0.0; unwrap.len()];
    temporal::high_pass_into(&reference_values, &weights, &mut reference_high_pass);

    let mut ph_hpt = vec![0.0_f32; input.n_ps * unwrap.len()];
    ph_hpt
        .par_chunks_mut(unwrap.len())
        .enumerate()
        .for_each_init(
            || (vec![0.0; unwrap.len()], vec![0.0; unwrap.len()]),
            |(values, high_pass), (row, output)| {
                fill_corrected_row(input, &unwrap, row, values);
                let ramp_row = &ph_ramp[row * deramp.len()..(row + 1) * deramp.len()];
                for (phase_column, ramp_column) in ramp_for_column.iter().enumerate() {
                    if let Some(ramp_column) = ramp_column {
                        values[phase_column] -= ramp_row[*ramp_column];
                    }
                }
                temporal::high_pass_into(values, &weights, high_pass);
                for column in 0..unwrap.len() {
                    let mut value = high_pass[column] - reference_high_pass[column];
                    if let Some(ramp_column) = ramp_for_column[column] {
                        value += ramp_row[ramp_column];
                    }
                    output[column] = value as f32;
                }
            },
        );
    if ph_hpt.iter().any(|value| !value.is_finite()) {
        return Err(ScnError::new(
            "temporal high-pass output overflowed its f32 contract",
        ));
    }

    let selected_scn = spatial::gaussian_low_pass(
        &ph_hpt,
        input.xy,
        input.n_ps,
        unwrap.len(),
        config.wavelength,
    )
    .map_err(ScnError::new)?;
    let full_unwrap = unwrap.len() == input.n_ifg;
    let mut ph_scn_slave = if full_unwrap {
        selected_scn
    } else {
        let mut full = vec![0.0; input.n_ps * input.n_ifg];
        full.par_chunks_mut(input.n_ifg)
            .enumerate()
            .for_each(|(row, output)| {
                let selected = &selected_scn[row * unwrap.len()..(row + 1) * unwrap.len()];
                for (local, &full_column) in unwrap.iter().enumerate() {
                    output[full_column] = selected[local];
                }
            });
        full
    };
    ph_scn_slave
        .par_chunks_mut(input.n_ifg)
        .for_each(|row| row[config.master_index] = 0.0);

    Ok(ScnOutputs {
        ph_scn_slave,
        ph_hpt,
        ph_ramp,
        n_unwrap: unwrap.len(),
        n_deramp: deramp.len(),
    })
}

fn corrected_value(input: &ScnInputs<'_>, row: usize, column: usize) -> f64 {
    let index = row * input.n_ifg + column;
    let mut value = input.ph_uw[index] as f64;
    if let Some(correction) = input.ph_scla {
        value -= correction[index] as f64;
    }
    if let Some(constant) = input.c_ps_uw {
        value -= constant[row] as f64;
    }
    if let Some(ramp) = input.scla_ramp {
        value -= ramp[index];
    }
    if value.is_nan() {
        0.0
    } else {
        value
    }
}

fn fill_corrected_row(input: &ScnInputs<'_>, unwrap: &[usize], row: usize, output: &mut [f64]) {
    for (local, &full) in unwrap.iter().enumerate() {
        output[local] = corrected_value(input, row, full);
    }
}

fn normalized_indices(values: &[usize], upper: usize, name: &str) -> Result<Vec<usize>, ScnError> {
    if values.iter().any(|&index| index >= upper) {
        return Err(ScnError::new(format!(
            "{name} contains an out-of-range index"
        )));
    }
    let mut output = values.to_vec();
    output.sort_unstable();
    output.dedup();
    Ok(output)
}

fn validate(input: &ScnInputs<'_>, config: &ScnConfig<'_>) -> Result<(), ScnError> {
    let matrix_len = input
        .n_ps
        .checked_mul(input.n_ifg)
        .ok_or_else(|| ScnError::new("SCN matrix shape overflows usize"))?;
    if input.n_ps == 0 || input.n_ifg == 0 || input.ph_uw.len() != matrix_len {
        return Err(ScnError::new(
            "ph_uw must be a non-empty n_ps by n_ifg matrix",
        ));
    }
    if input.xy.len() != input.n_ps * 2 || input.xy.iter().any(|value| !value.is_finite()) {
        return Err(ScnError::new("xy must be a finite n_ps by 2 matrix"));
    }
    if input.day.len() != input.n_ifg || input.day.iter().any(|value| !value.is_finite()) {
        return Err(ScnError::new(
            "day must contain one finite value per interferogram",
        ));
    }
    if config.master_index >= input.n_ifg
        || !config.time_window.is_finite()
        || config.time_window <= 0.0
        || !config.wavelength.is_finite()
        || config.wavelength <= 0.0
    {
        return Err(ScnError::new(
            "master index, time window, or wavelength is invalid",
        ));
    }
    for (correction, name) in [
        (input.ph_scla.map(|v| v.len()), "ph_scla"),
        (input.scla_ramp.map(|v| v.len()), "scla_ramp"),
    ] {
        if correction.is_some_and(|length| length != matrix_len) {
            return Err(ScnError::new(format!("{name} must match ph_uw")));
        }
    }
    if input
        .c_ps_uw
        .is_some_and(|values| values.len() != input.n_ps)
    {
        return Err(ScnError::new("c_ps_uw must contain one value per PS"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
