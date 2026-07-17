use super::fit::{
    standard_deviation_and_maximum, variance_columns_complex, variance_columns_real,
    weighted_slope_complex, weighted_slope_real, wrap_phase,
};
use super::Stage4Error;
use crate::stages::stage1::{Complex32, Matrix};
use num_complex::Complex64;
use rayon::prelude::*;

use super::noise_affine::{affine_moments, affine_one, EdgeScratch};

#[derive(Clone, Debug, PartialEq)]
pub struct Stage4Noise {
    pub ps_std: Vec<f64>,
    pub ps_max: Vec<f64>,
}

fn inverse_variance(variance: &[f64]) -> Vec<f64> {
    variance
        .iter()
        .map(|&value| {
            if value == 0.0 {
                f64::INFINITY
            } else {
                1.0 / value
            }
        })
        .collect()
}

fn edge_phase(
    phase: &Matrix<Complex32>,
    edges: &[[usize; 2]],
) -> Result<Vec<Complex64>, Stage4Error> {
    if edges.iter().flatten().any(|&node| node >= phase.rows) {
        return Err(Stage4Error::InvalidInput("edge node is out of bounds"));
    }
    let mut values = vec![Complex64::new(0.0, 0.0); edges.len() * phase.cols];
    values
        .par_chunks_mut(phase.cols)
        .zip(edges.par_iter())
        .for_each(|(output, &[first, second])| {
            for ifg in 0..phase.cols {
                let a = phase.row(first)[ifg];
                let b = phase.row(second)[ifg];
                output[ifg] = Complex64::new(f64::from(b.re), f64::from(b.im))
                    * Complex64::new(f64::from(a.re), f64::from(a.im)).conj();
            }
        });
    Ok(values)
}

fn small_baseline_noise(
    phase: &[Complex64],
    edges: usize,
    interferograms: usize,
    bperp: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let variance = variance_columns_complex(phase, edges, interferograms, usize::from(edges > 1));
    let slopes = weighted_slope_complex(
        bperp,
        phase,
        edges,
        interferograms,
        &inverse_variance(&variance),
    );
    let mut angles = vec![0.0; phase.len()];
    for edge in 0..edges {
        for ifg in 0..interferograms {
            angles[edge * interferograms + ifg] =
                (phase[edge * interferograms + ifg] - slopes[edge] * bperp[ifg]).arg();
        }
    }
    standard_deviation_and_maximum(
        &angles,
        edges,
        interferograms,
        usize::from(interferograms > 1),
    )
}

fn temporal_weights(day: &[f64], time_window: f64) -> (Vec<f64>, Vec<f64>) {
    let count = day.len();
    let mut difference = vec![0.0; count * count];
    let mut weights = vec![0.0; count * count];
    for output in 0..count {
        let mut sum = 0.0;
        for source in 0..count {
            let delta = day[output] - day[source];
            difference[output * count + source] = delta;
            let weight = (-(delta * delta) / (2.0 * time_window.max(1e-6).powi(2))).exp();
            weights[output * count + source] = weight;
            sum += weight;
        }
        if sum <= 0.0 {
            weights[output * count..(output + 1) * count].fill(1.0 / count as f64);
        } else {
            weights[output * count..(output + 1) * count]
                .iter_mut()
                .for_each(|value| *value /= sum);
        }
    }
    (difference, weights)
}

pub(super) fn single_master_noise(
    phase: &[Complex64],
    edges: usize,
    interferograms: usize,
    bperp: &[f64],
    day: &[f64],
    time_window: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (time_difference, weights) = temporal_weights(day, time_window);
    let mut noise = vec![0.0; phase.len()];
    let mut leave_one_out_noise = vec![0.0; phase.len()];
    let moments = (0..interferograms)
        .map(|output| {
            let range = output * interferograms..(output + 1) * interferograms;
            affine_moments(&time_difference[range.clone()], &weights[range])
        })
        .collect::<Vec<_>>();
    noise
        .par_chunks_mut(interferograms)
        .zip(leave_one_out_noise.par_chunks_mut(interferograms))
        .zip(phase.par_chunks(interferograms))
        .for_each_init(
            || EdgeScratch {
                smooth: vec![Complex64::new(0.0, 0.0); interferograms],
                adjusted: vec![0.0; interferograms],
                detrended: vec![0.0; interferograms],
            },
            |scratch, ((noise_row, leave_out_row), phase_row)| {
                for output in 0..interferograms {
                    let weight = &weights[output * interferograms..(output + 1) * interferograms];
                    scratch.smooth[output] = phase_row
                        .iter()
                        .zip(weight)
                        .map(|(value, weight)| *value * *weight)
                        .sum();
                }
                for output in 0..interferograms {
                    let range = output * interferograms..(output + 1) * interferograms;
                    let difference = &time_difference[range.clone()];
                    let weight = &weights[range];
                    for source in 0..interferograms {
                        scratch.adjusted[source] =
                            (phase_row[source] * scratch.smooth[output].conj()).arg();
                    }
                    let (intercept, slope) =
                        affine_one(difference, &scratch.adjusted, weight, moments[output]);
                    for source in 0..interferograms {
                        scratch.detrended[source] = wrap_phase(
                            scratch.adjusted[source] - intercept - slope * difference[source],
                        );
                    }
                    let (second_intercept, _) =
                        affine_one(difference, &scratch.detrended, weight, moments[output]);
                    let fitted = scratch.smooth[output]
                        * Complex64::from_polar(1.0, intercept + second_intercept);
                    noise_row[output] = (phase_row[output] * fitted.conj()).arg();
                    let leave_out = scratch.smooth[output]
                        - phase_row[output] * weights[output * interferograms + output];
                    leave_out_row[output] = (phase_row[output] * leave_out.conj()).arg();
                }
            },
        );
    let variance = variance_columns_real(
        &leave_one_out_noise,
        edges,
        interferograms,
        usize::from(edges > 1),
    );
    let slopes = weighted_slope_real(
        bperp,
        &noise,
        edges,
        interferograms,
        &inverse_variance(&variance),
    );
    noise
        .par_chunks_mut(interferograms)
        .zip(slopes.par_iter())
        .for_each(|(row, slope)| {
            for ifg in 0..interferograms {
                row[ifg] -= slope * bperp[ifg];
            }
        });
    standard_deviation_and_maximum(
        &noise,
        edges,
        interferograms,
        usize::from(interferograms > 1),
    )
}

pub fn edge_noise_statistics(
    phase: &Matrix<Complex32>,
    edges: &[[usize; 2]],
    bperp: &[f64],
    day: &[f64],
    time_window: f64,
    small_baseline: bool,
) -> Result<Stage4Noise, Stage4Error> {
    if phase.cols != bperp.len() || (!small_baseline && phase.cols != day.len()) {
        return Err(Stage4Error::InvalidInput(
            "edge-noise interferogram shape mismatch",
        ));
    }
    let mut ps_std = vec![f64::INFINITY; phase.rows];
    let mut ps_max = vec![f64::INFINITY; phase.rows];
    if edges.is_empty() || phase.cols == 0 {
        return Ok(Stage4Noise { ps_std, ps_max });
    }
    let edge_phase = edge_phase(phase, edges)?;
    let (edge_std, edge_max) = if small_baseline {
        small_baseline_noise(&edge_phase, edges.len(), phase.cols, bperp)
    } else {
        single_master_noise(
            &edge_phase,
            edges.len(),
            phase.cols,
            bperp,
            day,
            time_window,
        )
    };
    for (edge, &[first, second]) in edges.iter().enumerate() {
        for node in [first, second] {
            ps_std[node] = ps_std[node].min(edge_std[edge]);
            ps_max[node] = ps_max[node].min(edge_max[edge]);
        }
    }
    Ok(Stage4Noise { ps_std, ps_max })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_small_baseline_phase_has_zero_edge_noise() {
        let phase = Matrix::new(3, 3, vec![Complex32::new(1.0, 0.0); 9]).unwrap();
        let result = edge_noise_statistics(
            &phase,
            &[[0, 1], [1, 2]],
            &[-10.0, 0.0, 10.0],
            &[],
            730.0,
            true,
        )
        .unwrap();
        assert_eq!(result.ps_std, vec![0.0; 3]);
        assert_eq!(result.ps_max, vec![0.0; 3]);
    }
}
