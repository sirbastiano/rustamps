use super::{
    accumulate_weighted_grid, clap_filter_grid_stack, grid_indices, non_master_phase,
    normalize_phase_matrix, phase_weight_block, psquare_weighting, run_stage2,
    signal_noise_weighting, topofit_batch, ComplexGrid, PsquareOutput, PsquareReference,
    Stage2Config, Stage2Error, Stage2Input, Stage2Iteration, Stage2Kernel, Stage2Output,
    Stage2State,
};
use crate::stages::stage1::{Complex32, Matrix};

#[derive(Clone, Debug, PartialEq)]
pub enum NativeWeighting {
    Psquare(PsquareReference),
    SignalNoise,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeStage2Options {
    pub grid_size: f32,
    pub clap_alpha: f64,
    pub clap_beta: f64,
    pub clap_window: usize,
    pub clap_padding: usize,
    pub low_pass: Matrix<f64>,
    pub n_trial_wraps: f64,
    pub weighting: NativeWeighting,
}

impl NativeStage2Options {
    pub fn signal_noise_defaults() -> Self {
        Self {
            grid_size: 50.0,
            clap_alpha: 1.0,
            clap_beta: 0.3,
            clap_window: 24,
            clap_padding: 8,
            low_pass: butterworth_low_pass(32, 50.0, 800.0),
            n_trial_wraps: 0.0,
            weighting: NativeWeighting::SignalNoise,
        }
    }
}

pub struct NativeStage2Kernel {
    pub options: NativeStage2Options,
    last_psquare: Option<PsquareOutput>,
    previous_psquare: Option<PsquareOutput>,
}

impl NativeStage2Kernel {
    pub fn new(options: NativeStage2Options) -> Self {
        Self {
            options,
            last_psquare: None,
            previous_psquare: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeStage2Result {
    pub output: Stage2Output,
    pub final_psquare: Option<PsquareOutput>,
}

pub fn butterworth_low_pass(size: usize, grid_size: f64, wavelength: f64) -> Matrix<f64> {
    let frequency_zero = 1.0 / wavelength;
    let frequency = (0..size)
        .map(|index| (index as f64 - size as f64 / 2.0) / (grid_size * size as f64))
        .collect::<Vec<_>>();
    let one_dimensional = frequency
        .iter()
        .map(|value| 1.0 / (1.0 + (value / frequency_zero).powi(10)))
        .collect::<Vec<_>>();
    let mut values = vec![0.0; size * size];
    for row in 0..size {
        for col in 0..size {
            let source_row = (row + size / 2) % size;
            let source_col = (col + size / 2) % size;
            values[row * size + col] = one_dimensional[source_row] * one_dimensional[source_col];
        }
    }
    Matrix {
        rows: size,
        cols: size,
        values,
    }
}

fn extract_patch(grid: &ComplexGrid, layout: &super::GridLayout) -> Matrix<Complex32> {
    let mut values = Vec::with_capacity(layout.indices.len() * grid.planes);
    for &[row, col] in &layout.indices {
        for plane in 0..grid.planes {
            let mut value = grid.values[(row * grid.cols + col) * grid.planes + plane];
            let magnitude = value.norm();
            if magnitude != 0.0 {
                value /= magnitude;
            }
            values.push(value);
        }
    }
    Matrix {
        rows: layout.indices.len(),
        cols: grid.planes,
        values,
    }
}

impl Stage2Kernel for NativeStage2Kernel {
    fn estimate(
        &mut self,
        input: &Stage2Input,
        state: &Stage2State,
    ) -> Result<Stage2Iteration, Stage2Error> {
        let phase_source = non_master_phase(input);
        let (phase, amplitude) = normalize_phase_matrix(&phase_source);
        let layout = grid_indices(&input.xy, self.options.grid_size)?;
        let weighted = phase_weight_block(&phase, &input.bperp_mat, &state.k_ps, &state.weighting)?;
        let grid = accumulate_weighted_grid(&weighted, &layout)?;
        let filtered = clap_filter_grid_stack(
            &grid,
            self.options.clap_alpha,
            self.options.clap_beta,
            self.options.clap_window,
            self.options.clap_padding,
            &self.options.low_pass,
        )?;
        let phase_patch = extract_patch(&filtered, &layout);
        let psdph = Matrix {
            rows: phase.rows,
            cols: phase.cols,
            values: phase
                .values
                .iter()
                .zip(&phase_patch.values)
                .map(|(&source, &patch)| source * patch.conj())
                .collect(),
        };
        let valid = super::valid_all_ifg_rows(&psdph);
        let fit = topofit_batch(&psdph, &input.bperp_mat, self.options.n_trial_wraps)?;
        let mut k_ps = fit.k_ps;
        let mut c_ps = fit.c_ps;
        let mut coherence = fit.coherence;
        let mut residual_angle = Matrix {
            rows: fit.residual.rows,
            cols: fit.residual.cols,
            values: fit
                .residual
                .values
                .iter()
                .map(|value| value.arg())
                .collect(),
        };
        for row in 0..phase.rows {
            if valid[row] {
                continue;
            }
            k_ps[row] = f64::NAN;
            c_ps[row] = 0.0;
            coherence[row] = 0.0;
            residual_angle.values[row * phase.cols..(row + 1) * phase.cols].fill(0.0);
        }
        let weighting = match &self.options.weighting {
            NativeWeighting::Psquare(reference) => {
                let output = psquare_weighting(&coherence, reference)?;
                let weighting = output.weighting.clone();
                self.previous_psquare = self.last_psquare.replace(output);
                weighting
            }
            NativeWeighting::SignalNoise => {
                self.last_psquare = None;
                self.previous_psquare = None;
                signal_noise_weighting(&amplitude, &residual_angle)
            }
        };
        Ok(Stage2Iteration {
            k_ps,
            c_ps,
            coherence,
            weighting,
            phase_residual: residual_angle,
            phase_patch,
        })
    }
}

pub fn run_stage2_native(
    input: &Stage2Input,
    config: &Stage2Config,
    options: NativeStage2Options,
) -> Result<Stage2Output, Stage2Error> {
    Ok(run_stage2_native_detailed(input, config, options)?.output)
}

pub fn run_stage2_native_detailed(
    input: &Stage2Input,
    config: &Stage2Config,
    mut options: NativeStage2Options,
) -> Result<NativeStage2Result, Stage2Error> {
    options.n_trial_wraps = config.n_trial_wraps;
    let mut kernel = NativeStage2Kernel::new(options);
    let output = run_stage2(input, config, &mut kernel)?;
    Ok(NativeStage2Result {
        output,
        final_psquare: kernel.previous_psquare,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn butterworth_filter_is_shifted_to_fft_zero_frequency() {
        let filter = butterworth_low_pass(32, 50.0, 800.0);
        assert_eq!(filter.values[0], 1.0);
        assert!(filter.values[16 * 32 + 16] < filter.values[0]);
    }
}
