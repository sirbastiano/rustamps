use std::collections::BTreeSet;
use std::path::Path;

use super::super::mat::{complex32, numeric_f64, shape};
use super::super::params::Params;
use super::input::{complex_matrix, read_required, real_matrix, Initial};
use super::threshold::ThresholdContext;
use pystamps_core::stages::stage1::Matrix;
use pystamps_core::stages::stage2::{butterworth_low_pass, ComplexGrid, GridLayout};
use pystamps_core::stages::stage3::{
    apply_reestimate, reestimate_gamma_native, NativeReestimateInput, NativeReestimateOptions,
    Stage3Output,
};

pub(super) fn ifg_indices(
    data: &Initial,
    params: &Params,
    small: bool,
) -> Result<(Vec<usize>, Vec<f64>), String> {
    let dropped = params
        .indices("drop_ifg_index")?
        .into_iter()
        .collect::<BTreeSet<_>>();
    if dropped.iter().any(|&index| index >= data.n_ifg) {
        return Err("drop_ifg_index exceeds the interferogram count".to_owned());
    }
    let mut zero_based = Vec::new();
    for source in 0..data.n_ifg {
        if dropped.contains(&source) || (!small && source == data.master) {
            continue;
        }
        zero_based.push(if !small && source > data.master {
            source - 1
        } else {
            source
        });
    }
    let one_based = zero_based.iter().map(|index| (index + 1) as f64).collect();
    Ok((zero_based, one_based))
}

pub(super) fn run(
    patch: &Path,
    data: &Initial,
    params: &Params,
    threshold: &ThresholdContext,
    selected_rows: &[usize],
    small: bool,
) -> Result<(Stage3Output, Vec<f64>), String> {
    let context = load_context(patch, data, small)?;
    let (interferograms, _) = ifg_indices(data, params, small)?;
    if interferograms.is_empty() {
        return Err("no interferograms remain for gamma re-estimation".to_owned());
    }
    let options = options(data, params)?;
    let initial_threshold = threshold.calculate(data, &data.coherence)?.threshold;
    let mut estimated = reestimate_gamma_native(
        &NativeReestimateInput {
            source_phase: &context.phase,
            phase_grid: &context.grid,
            grid_layout: &context.layout,
            per_pixel_bperp: &context.bperp_mat,
            nominal_bperp: &context.bperp,
            selected_rows,
            interferogram_indices: &interferograms,
            coherence_threshold: &initial_threshold,
        },
        &options,
    )
    .map_err(|error| error.to_string())?;
    let mut coherence = data.coherence.clone();
    for (&source, &value) in selected_rows.iter().zip(&estimated.coherence) {
        coherence[source] = value;
    }
    let recalculated = threshold.calculate(data, &coherence)?;
    estimated.coherence_threshold = selected_rows
        .iter()
        .map(|&row| recalculated.threshold[row])
        .collect();
    let output = apply_reestimate(&data.k_ps, estimated).map_err(|error| error.to_string())?;
    Ok((output, recalculated.linear_coefficients))
}

struct Context {
    phase: Matrix<num_complex::Complex32>,
    grid: ComplexGrid,
    layout: GridLayout,
    bperp_mat: Matrix<f64>,
    bperp: Vec<f64>,
}

fn load_context(patch: &Path, data: &Initial, small: bool) -> Result<Context, String> {
    let ph = read_required(patch, "ph1.mat")?;
    let full_phase = complex_matrix(&ph, "ph", data.n_ps, data.n_ifg)?;
    let phase = if small {
        full_phase
    } else {
        remove_column(&full_phase, data.master)
    };
    let grid_shape = shape(&data.pm, "ph_grid")?;
    let grid_values = complex32(&data.pm, "ph_grid")?;
    let work_ifg = phase.cols;
    let (grid_rows, grid_cols) = match grid_shape.as_slice() {
        [rows, cols, planes] if *planes == work_ifg => (*rows, *cols),
        [rows, cols] if work_ifg == 1 => (*rows, *cols),
        _ => return Err(format!("pm1.ph_grid has invalid shape {grid_shape:?}")),
    };
    if grid_values.len() != grid_rows * grid_cols * work_ifg {
        return Err("pm1.ph_grid value count does not match its shape".to_owned());
    }
    let grid_ij = real_matrix(&data.pm, "grid_ij", data.n_ps, 2)?;
    let mut indices = Vec::with_capacity(data.n_ps);
    for row in 0..data.n_ps {
        let pair = grid_ij.row(row);
        if pair
            .iter()
            .any(|value| !value.is_finite() || *value < 1.0 || value.fract() != 0.0)
        {
            return Err("pm1.grid_ij must contain positive one-based integers".to_owned());
        }
        let index = [pair[0] as usize - 1, pair[1] as usize - 1];
        if index[0] >= grid_rows || index[1] >= grid_cols {
            return Err("pm1.grid_ij exceeds ph_grid bounds".to_owned());
        }
        indices.push(index);
    }
    let bp = read_required(patch, "bp1.mat")?;
    let bperp_mat = baseline_matrix(&bp, data, small)?;
    let nominal = numeric_f64(&data.ps, "bperp")?;
    let bperp = if small {
        require_length("ps1.bperp", nominal, data.n_ifg)?
    } else if nominal.len() == data.n_ifg {
        nominal
            .into_iter()
            .enumerate()
            .filter_map(|(index, value)| (index != data.master).then_some(value))
            .collect()
    } else {
        require_length("ps1.bperp", nominal, work_ifg)?
    };
    Ok(Context {
        phase,
        grid: ComplexGrid {
            rows: grid_rows,
            cols: grid_cols,
            planes: work_ifg,
            values: grid_values,
        },
        layout: GridLayout {
            indices,
            rows: grid_rows,
            cols: grid_cols,
        },
        bperp_mat,
        bperp,
    })
}

fn baseline_matrix(
    bp: &pystamps_io::MatFile,
    data: &Initial,
    small: bool,
) -> Result<Matrix<f64>, String> {
    let dimensions = shape(bp, "bperp_mat")?;
    let work_ifg = data.n_ifg - usize::from(!small);
    if dimensions == [data.n_ps, work_ifg] || dimensions == [work_ifg, data.n_ps] {
        return real_matrix(bp, "bperp_mat", data.n_ps, work_ifg);
    }
    let full = real_matrix(bp, "bperp_mat", data.n_ps, data.n_ifg)?;
    if small {
        Ok(full)
    } else {
        Ok(remove_column(&full, data.master))
    }
}

fn options(data: &Initial, params: &Params) -> Result<NativeReestimateOptions, String> {
    let window = positive_integer(params.scalar("clap_win", 32.0)?, "clap_win")?;
    let low_pass =
        if data.pm.contains_key("low_pass") && shape(&data.pm, "low_pass")? == [window, window] {
            real_matrix(&data.pm, "low_pass", window, window)?
        } else {
            butterworth_low_pass(
                window,
                params.scalar("filter_grid_size", 50.0)?,
                params.scalar("clap_low_pass_wavelength", 800.0)?,
            )
        };
    let n_trial_wraps = data
        .pm
        .get("n_trial_wraps")
        .map(|_| numeric_f64(&data.pm, "n_trial_wraps"))
        .transpose()?
        .and_then(|values| values.first().copied())
        .unwrap_or(0.0);
    Ok(NativeReestimateOptions {
        clap_window: window,
        clap_alpha: params.scalar("clap_alpha", 1.0)?,
        clap_beta: params.scalar("clap_beta", 0.3)?,
        low_pass,
        slc_oversampling: positive_integer(params.scalar("slc_osf", 1.0)?, "slc_osf")?,
        n_trial_wraps,
    })
}

fn positive_integer(value: f64, name: &str) -> Result<usize, String> {
    if value.is_finite() && value >= 1.0 && value.fract() == 0.0 {
        Ok(value as usize)
    } else {
        Err(format!("{name} must be a positive integer"))
    }
}

fn remove_column<T: Copy>(matrix: &Matrix<T>, remove: usize) -> Matrix<T> {
    let values = (0..matrix.rows)
        .flat_map(|row| {
            matrix
                .row(row)
                .iter()
                .enumerate()
                .filter_map(move |(column, value)| (column != remove).then_some(*value))
        })
        .collect();
    Matrix {
        rows: matrix.rows,
        cols: matrix.cols - 1,
        values,
    }
}

fn require_length(name: &str, values: Vec<f64>, expected: usize) -> Result<Vec<f64>, String> {
    if values.len() == expected {
        Ok(values)
    } else {
        Err(format!(
            "{name} has {} values; expected {expected}",
            values.len()
        ))
    }
}
