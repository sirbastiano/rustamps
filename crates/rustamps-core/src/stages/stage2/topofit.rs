use super::Stage2Error;
use crate::stages::stage1::{Complex32, Matrix};
use num_complex::Complex64;
use rayon::prelude::*;

const QUARTER_PI: f64 = std::f64::consts::PI / 4.0;
const NEAR_MAX_TOLERANCE: f64 = 2.0e-4;

#[derive(Clone, Debug, PartialEq)]
pub struct TopofitOutput {
    pub k_ps: Vec<f64>,
    pub c_ps: Vec<f64>,
    pub coherence: Vec<f64>,
    pub residual: Matrix<Complex32>,
}

struct RowData {
    columns: Vec<usize>,
    phase: Vec<Complex64>,
    bperp: Vec<f64>,
    amplitude: Vec<f64>,
    weighted_bperp: Vec<f64>,
    linear_denominator: f64,
    bperp_range: f64,
    amplitude_sum: f64,
    width: usize,
}

struct RowResult {
    k: f64,
    c: f64,
    coherence: f64,
    residual: Vec<Complex32>,
}

fn collect(phase: &[Complex32], bperp: &[f64]) -> RowData {
    let mut row = RowData {
        columns: Vec::new(),
        phase: Vec::new(),
        bperp: Vec::new(),
        amplitude: Vec::new(),
        weighted_bperp: Vec::new(),
        linear_denominator: 0.0,
        bperp_range: 1.0,
        amplitude_sum: 0.0,
        width: phase.len(),
    };
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for (column, (&value, &baseline)) in phase.iter().zip(bperp).enumerate() {
        if value == Complex32::new(0.0, 0.0) {
            continue;
        }
        let value64 = Complex64::new(f64::from(value.re), f64::from(value.im));
        let amplitude = value64.norm();
        let weighted_bperp = amplitude * baseline;
        row.columns.push(column);
        row.phase.push(value64);
        row.bperp.push(baseline);
        row.amplitude.push(amplitude);
        row.weighted_bperp.push(weighted_bperp);
        row.linear_denominator += weighted_bperp * weighted_bperp;
        row.amplitude_sum += amplitude;
        low = low.min(baseline);
        high = high.max(baseline);
    }
    if row.linear_denominator == 0.0 {
        row.linear_denominator = 1.0;
    }
    if row.amplitude_sum == 0.0 {
        row.amplitude_sum = 1.0;
    }
    if !row.phase.is_empty() {
        row.bperp_range = high - low;
        if row.bperp_range == 0.0 {
            row.bperp_range = 1.0;
        }
    }
    row
}

fn trial_values(n_trial_wraps: f64) -> Vec<f64> {
    let extent = (8.0 * n_trial_wraps).ceil() as i64;
    (-extent..=extent).map(|value| value as f64).collect()
}

fn trial_coherence(row: &RowData, trials: &[f64]) -> Vec<f64> {
    trials
        .iter()
        .map(|&trial| {
            let mut sum = Complex64::new(0.0, 0.0);
            for index in 0..row.phase.len() {
                let angle = row.bperp[index] / row.bperp_range * QUARTER_PI * trial;
                sum += row.phase[index] * Complex64::from_polar(1.0, -angle);
            }
            sum.norm() / row.amplitude_sum
        })
        .collect()
}

fn near_maxima(coherence: &[f64]) -> Vec<usize> {
    if coherence.len() <= 1 {
        return vec![0];
    }
    let maximum = coherence.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let last = coherence.len() - 1;
    let mut candidates = (0..coherence.len())
        .filter(|&index| {
            let local = (index == 0 || coherence[index] >= coherence[index - 1])
                && (index == last || coherence[index] >= coherence[index + 1]);
            local && coherence[index] >= maximum - NEAR_MAX_TOLERANCE
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        candidates.push(
            coherence
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0,
        );
    }
    candidates
}

fn refine(row: &RowData, coarse_k: f64) -> RowResult {
    let mut offset = Complex64::new(0.0, 0.0);
    for index in 0..row.phase.len() {
        offset += row.phase[index] * Complex64::from_polar(1.0, -coarse_k * row.bperp[index]);
    }
    let mut numerator = 0.0;
    for index in 0..row.phase.len() {
        let residual = row.phase[index] * Complex64::from_polar(1.0, -coarse_k * row.bperp[index]);
        numerator +=
            row.weighted_bperp[index] * row.amplitude[index] * (residual * offset.conj()).arg();
    }
    let k = coarse_k + numerator / row.linear_denominator;
    let mut mean = Complex64::new(0.0, 0.0);
    let mut denominator = 0.0;
    let mut residual = vec![Complex32::new(0.0, 0.0); row.width];
    for (index, &column) in row.columns.iter().enumerate() {
        let value = row.phase[index] * Complex64::from_polar(1.0, -k * row.bperp[index]);
        mean += value;
        denominator += value.norm();
        residual[column] = Complex32::new(value.re as f32, value.im as f32);
    }
    if denominator == 0.0 {
        denominator = 1.0;
    }
    RowResult {
        k,
        c: mean.arg(),
        coherence: mean.norm() / denominator,
        residual,
    }
}

fn solve_row(phase: &[Complex32], bperp: &[f64], trials: &[f64]) -> RowResult {
    let row = collect(phase, bperp);
    if row.phase.is_empty() {
        return RowResult {
            k: f64::NAN,
            c: f64::NAN,
            coherence: f64::NAN,
            residual: vec![Complex32::new(0.0, 0.0); phase.len()],
        };
    }
    let coherence = trial_coherence(&row, trials);
    let candidates = near_maxima(&coherence);
    let best = candidates
        .into_iter()
        .max_by(|&left, &right| coherence[left].total_cmp(&coherence[right]))
        .unwrap();
    refine(&row, QUARTER_PI / row.bperp_range * trials[best])
}

pub fn topofit_batch(
    phase: &Matrix<Complex32>,
    bperp: &Matrix<f64>,
    n_trial_wraps: f64,
) -> Result<TopofitOutput, Stage2Error> {
    if phase.rows == 0 || phase.cols == 0 || bperp.rows != phase.rows || bperp.cols != phase.cols {
        return Err(Stage2Error::InvalidInput(
            "topofit phase and baseline shapes must match",
        ));
    }
    if !n_trial_wraps.is_finite() || n_trial_wraps < 0.0 {
        return Err(Stage2Error::InvalidInput(
            "n_trial_wraps must be finite and non-negative",
        ));
    }
    let trials = trial_values(n_trial_wraps);
    let rows = (0..phase.rows)
        .into_par_iter()
        .map(|row| solve_row(phase.row(row), bperp.row(row), &trials))
        .collect::<Vec<_>>();
    Ok(TopofitOutput {
        k_ps: rows.iter().map(|row| row.k).collect(),
        c_ps: rows.iter().map(|row| row.c).collect(),
        coherence: rows.iter().map(|row| row.coherence).collect(),
        residual: Matrix {
            rows: phase.rows,
            cols: phase.cols,
            values: rows.into_iter().flat_map(|row| row.residual).collect(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topofit_recovers_a_known_baseline_ramp() {
        let bperp = Matrix::new(1, 5, vec![-20.0, -10.0, 0.0, 10.0, 20.0]).unwrap();
        let expected = 0.03_f64;
        let phase = Matrix::new(
            1,
            5,
            bperp
                .values
                .iter()
                .map(|b| Complex32::new((expected * b).cos() as f32, (expected * b).sin() as f32))
                .collect(),
        )
        .unwrap();
        let result = topofit_batch(&phase, &bperp, 1.0).unwrap();
        assert!((result.k_ps[0] - expected).abs() < 1e-5);
        assert!(result.coherence[0] > 0.9999);
    }
}
