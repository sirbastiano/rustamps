use super::{ReestimatedSelection, Stage3Error};
use crate::stages::stage1::{Complex32, Matrix};
use crate::stages::stage2::{clap_filter_patch, topofit_batch, ComplexGrid, GridLayout};
use num_complex::Complex64;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct NativeReestimateInput<'a> {
    pub source_phase: &'a Matrix<Complex32>,
    pub phase_grid: &'a ComplexGrid,
    pub grid_layout: &'a GridLayout,
    pub per_pixel_bperp: &'a Matrix<f64>,
    pub nominal_bperp: &'a [f64],
    pub selected_rows: &'a [usize],
    pub interferogram_indices: &'a [usize],
    pub coherence_threshold: &'a [f64],
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeReestimateOptions {
    pub clap_window: usize,
    pub clap_alpha: f64,
    pub clap_beta: f64,
    pub low_pass: Matrix<f64>,
    pub slc_oversampling: usize,
    pub n_trial_wraps: f64,
}

fn window_start(coordinate: usize, extent: usize, window: usize) -> Option<usize> {
    let coordinate_one_based = coordinate + 1;
    let half = window / 2;
    let mut start = coordinate_one_based.saturating_sub(half).max(1) as isize;
    let end = start + window as isize - 1;
    if end > extent as isize {
        start -= end - extent as isize;
    }
    (start >= 1).then_some(start as usize - 1)
}

fn filtered_patch_row(
    source_row: usize,
    input: &NativeReestimateInput<'_>,
    options: &NativeReestimateOptions,
) -> Result<Vec<Complex32>, Stage3Error> {
    let [grid_row, grid_col] = input.grid_layout.indices[source_row];
    let Some(row_start) = window_start(grid_row, input.phase_grid.rows, options.clap_window) else {
        return Ok(vec![Complex32::new(0.0, 0.0); input.phase_grid.planes]);
    };
    let Some(col_start) = window_start(grid_col, input.phase_grid.cols, options.clap_window) else {
        return Ok(vec![Complex32::new(0.0, 0.0); input.phase_grid.planes]);
    };
    let local_row = grid_row - row_start;
    let local_col = grid_col - col_start;
    let mut output = Vec::with_capacity(input.phase_grid.planes);
    for plane in 0..input.phase_grid.planes {
        let mut patch = vec![Complex64::new(0.0, 0.0); options.clap_window * options.clap_window];
        for row in 0..options.clap_window {
            for col in 0..options.clap_window {
                let value = input.phase_grid.values[((row_start + row) * input.phase_grid.cols
                    + col_start
                    + col)
                    * input.phase_grid.planes
                    + plane];
                patch[row * options.clap_window + col] =
                    Complex64::new(f64::from(value.re), f64::from(value.im));
            }
        }
        patch[local_row * options.clap_window + local_col] = Complex64::new(0.0, 0.0);
        if plane == 0 && options.slc_oversampling > 1 {
            let radius = options.slc_oversampling - 1;
            for row in
                local_row.saturating_sub(radius)..=(local_row + radius).min(options.clap_window - 1)
            {
                for col in local_col.saturating_sub(radius)
                    ..=(local_col + radius).min(options.clap_window - 1)
                {
                    patch[row * options.clap_window + col] = Complex64::new(0.0, 0.0);
                }
            }
        }
        let filtered = clap_filter_patch(
            &patch,
            options.clap_window,
            options.clap_window,
            options.clap_alpha,
            options.clap_beta,
            &options.low_pass.values,
        )
        .map_err(|error| Stage3Error::ReestimateRequired(error.to_string()))?;
        let value = filtered[local_row * options.clap_window + local_col];
        output.push(Complex32::new(value.re as f32, value.im as f32));
    }
    Ok(output)
}

pub fn reestimate_gamma_native(
    input: &NativeReestimateInput<'_>,
    options: &NativeReestimateOptions,
) -> Result<ReestimatedSelection, Stage3Error> {
    let rows = input.source_phase.rows;
    let columns = input.source_phase.cols;
    if rows == 0
        || columns == 0
        || input.phase_grid.planes != columns
        || input.phase_grid.rows != input.grid_layout.rows
        || input.phase_grid.cols != input.grid_layout.cols
        || input.grid_layout.indices.len() != rows
        || input.per_pixel_bperp.rows != rows
        || input.per_pixel_bperp.cols != columns
        || input.nominal_bperp.len() != columns
        || input.coherence_threshold.len() != rows
        || options.clap_window == 0
        || options.low_pass.rows != options.clap_window
        || options.low_pass.cols != options.clap_window
        || input.selected_rows.iter().any(|&row| row >= rows)
        || input.interferogram_indices.is_empty()
        || input
            .interferogram_indices
            .iter()
            .any(|&column| column >= columns)
    {
        return Err(Stage3Error::InvalidInput(
            "invalid native gamma re-estimation input",
        ));
    }
    let patch_rows = input
        .selected_rows
        .par_iter()
        .map(|&row| filtered_patch_row(row, input, options))
        .collect::<Result<Vec<_>, _>>()?;
    let mut k_ps = vec![f64::NAN; input.selected_rows.len()];
    let mut c_ps = vec![0.0; input.selected_rows.len()];
    let mut coherence = vec![f64::NAN; input.selected_rows.len()];
    let mut residual = vec![0.0_f32; input.selected_rows.len() * columns];
    let mut fit_phase = Vec::new();
    let mut fit_baseline = Vec::new();
    let mut fit_rows = Vec::new();
    for (selected, (&source, patch)) in input.selected_rows.iter().zip(&patch_rows).enumerate() {
        if input
            .source_phase
            .row(source)
            .iter()
            .zip(patch)
            .any(|(&phase, &patch)| phase * patch.conj() == Complex32::new(0.0, 0.0))
        {
            continue;
        }
        let mut row = Vec::with_capacity(input.interferogram_indices.len());
        for &column in input.interferogram_indices {
            let mut value = input.source_phase.row(source)[column] * patch[column].conj();
            value /= value.norm();
            row.push(value);
        }
        fit_phase.extend(row);
        fit_baseline.extend(
            input
                .interferogram_indices
                .iter()
                .map(|&column| input.per_pixel_bperp.row(source)[column]),
        );
        fit_rows.push(selected);
    }
    if !fit_rows.is_empty() {
        let fit = topofit_batch(
            &Matrix {
                rows: fit_rows.len(),
                cols: input.interferogram_indices.len(),
                values: fit_phase,
            },
            &Matrix {
                rows: fit_rows.len(),
                cols: input.interferogram_indices.len(),
                values: fit_baseline,
            },
            options.n_trial_wraps,
        )
        .map_err(|error| Stage3Error::ReestimateRequired(error.to_string()))?;
        for (fit_row, &selected) in fit_rows.iter().enumerate() {
            k_ps[selected] = fit.k_ps[fit_row];
            c_ps[selected] = fit.c_ps[fit_row];
            coherence[selected] = fit.coherence[fit_row];
            for (fit_col, &column) in input.interferogram_indices.iter().enumerate() {
                residual[selected * columns + column] = fit.residual.row(fit_row)[fit_col].arg();
            }
        }
    }
    let mut range = input
        .nominal_bperp
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - input
            .nominal_bperp
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
    if range <= 0.0 {
        range = 1.0;
    }
    Ok(ReestimatedSelection {
        source_rows: input.selected_rows.to_vec(),
        coherence,
        k_ps,
        c_ps,
        phase_patch: Matrix {
            rows: patch_rows.len(),
            cols: columns,
            values: patch_rows.into_iter().flatten().collect(),
        },
        phase_residual: Matrix {
            rows: input.selected_rows.len(),
            cols: columns,
            values: residual,
        },
        coherence_threshold: input
            .selected_rows
            .iter()
            .map(|&row| input.coherence_threshold[row])
            .collect(),
        bperp_range: range,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_reestimation_path_filters_then_refits_selected_ps() {
        let baseline = [-10.0, 0.0, 10.0];
        let expected_k = 0.03_f64;
        let source = Matrix::new(
            1,
            3,
            baseline
                .iter()
                .map(|value| {
                    Complex32::new(
                        (expected_k * value).cos() as f32,
                        (expected_k * value).sin() as f32,
                    )
                })
                .collect(),
        )
        .unwrap();
        let grid = ComplexGrid {
            rows: 8,
            cols: 8,
            planes: 3,
            values: vec![Complex32::new(1.0, 0.0); 8 * 8 * 3],
        };
        let layout = GridLayout {
            indices: vec![[3, 3]],
            rows: 8,
            cols: 8,
        };
        let mut low_pass = vec![0.0; 64];
        low_pass[0] = 1.0;
        let result = reestimate_gamma_native(
            &NativeReestimateInput {
                source_phase: &source,
                phase_grid: &grid,
                grid_layout: &layout,
                per_pixel_bperp: &Matrix::new(1, 3, baseline.to_vec()).unwrap(),
                nominal_bperp: &baseline,
                selected_rows: &[0],
                interferogram_indices: &[0, 1, 2],
                coherence_threshold: &[0.5],
            },
            &NativeReestimateOptions {
                clap_window: 8,
                clap_alpha: 1.0,
                clap_beta: 0.0,
                low_pass: Matrix::new(8, 8, low_pass).unwrap(),
                slc_oversampling: 1,
                n_trial_wraps: 1.0,
            },
        )
        .unwrap();
        assert!((result.k_ps[0] - expected_k).abs() < 1e-5);
        assert!(result.coherence[0] > 0.999);
    }
}
