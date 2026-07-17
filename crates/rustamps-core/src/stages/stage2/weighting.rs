use super::{histogram_with_centers, Stage2Error};
use crate::stages::stage1::Matrix;

#[derive(Clone, Debug, PartialEq)]
pub struct PsquareReference {
    pub coherence_bins: Vec<f64>,
    pub random_distribution: Vec<f64>,
    pub low_coherence_bins: usize,
    pub last_nonzero_random_bin_one_based: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PsquareOutput {
    pub observed_distribution: Vec<f64>,
    pub scaled_random_distribution: Vec<f64>,
    pub random_probability: Vec<f64>,
    pub weighting: Vec<f64>,
}

fn gaussian_window(length: usize) -> Vec<f64> {
    let standard_deviation = (length as f64 - 1.0) / 5.0;
    let center = (length as f64 - 1.0) / 2.0;
    (0..length)
        .map(|index| {
            let x = (index as f64 - center) / standard_deviation;
            (-0.5 * x * x).exp()
        })
        .collect()
}

fn fir_filter(coefficients: &[f64], values: &[f64]) -> Vec<f64> {
    (0..values.len())
        .map(|index| {
            coefficients
                .iter()
                .enumerate()
                .take(index + 1)
                .map(|(lag, coefficient)| coefficient * values[index - lag])
                .sum()
        })
        .collect()
}

fn bandlimited(offset: f64) -> f64 {
    let omega = std::f64::consts::PI * 0.5;
    if offset.abs() < 1e-14 {
        2.0 * omega
    } else {
        2.0 * (omega * offset).sin() / offset
    }
}

fn solve(mut matrix: Vec<f64>, mut rhs: Vec<f64>, size: usize) -> Option<Vec<f64>> {
    for col in 0..size {
        let pivot = (col..size).max_by(|&a, &b| {
            matrix[a * size + col]
                .abs()
                .total_cmp(&matrix[b * size + col].abs())
        })?;
        if matrix[pivot * size + col].abs() <= f64::EPSILON {
            return None;
        }
        for index in 0..size {
            matrix.swap(col * size + index, pivot * size + index);
        }
        rhs.swap(col, pivot);
        let diagonal = matrix[col * size + col];
        for index in col..size {
            matrix[col * size + index] /= diagonal;
        }
        rhs[col] /= diagonal;
        for row in 0..size {
            if row == col {
                continue;
            }
            let factor = matrix[row * size + col];
            for index in col..size {
                matrix[row * size + index] -= factor * matrix[col * size + index];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    Some(rhs)
}

fn matlab_interp_filter(factor: usize) -> Result<Vec<f64>, Stage2Error> {
    let order = 4usize;
    let delay = factor * order;
    let samples = (-4..4).map(f64::from).collect::<Vec<_>>();
    let mut normal = vec![0.0; samples.len() * samples.len()];
    for row in 0..samples.len() {
        for col in 0..samples.len() {
            normal[row * samples.len() + col] = bandlimited(samples[row] - samples[col]);
        }
    }
    let mut coefficients = vec![0.0; 2 * delay + 1];
    coefficients[delay] = 1.0;
    for phase in 1..factor {
        let rhs = samples
            .iter()
            .map(|sample| bandlimited(*sample + phase as f64 / factor as f64))
            .collect();
        let taps = solve(normal.clone(), rhs, samples.len())
            .ok_or_else(|| Stage2Error::Kernel("singular MATLAB interpolation filter".into()))?;
        for (sample, tap) in (-4..4).zip(taps) {
            coefficients[(delay as isize + phase as isize + sample * factor as isize) as usize] =
                tap;
        }
    }
    let reversed = coefficients.iter().copied().rev().collect::<Vec<_>>();
    for (value, mirror) in coefficients.iter_mut().zip(reversed) {
        *value = (*value + mirror) * 0.5;
    }
    Ok(coefficients)
}

fn matlab_interp(values: &[f64], factor: usize) -> Result<Vec<f64>, Stage2Error> {
    let filter = matlab_interp_filter(factor)?;
    let delay = (filter.len() - 1) / 2;
    let mut upsampled = vec![0.0; values.len() * factor + delay];
    for (index, &value) in values.iter().enumerate() {
        upsampled[index * factor] = value;
    }
    let filtered = fir_filter(&filter, &upsampled);
    Ok(filtered[delay..].to_vec())
}

pub fn psquare_weighting(
    coherence: &[f64],
    reference: &PsquareReference,
) -> Result<PsquareOutput, Stage2Error> {
    if reference.coherence_bins.len() != reference.random_distribution.len()
        || reference.coherence_bins.is_empty()
    {
        return Err(Stage2Error::InvalidInput(
            "invalid P-square reference histogram",
        ));
    }
    let observed = histogram_with_centers(coherence, &reference.coherence_bins);
    let low = reference.low_coherence_bins.min(observed.len());
    let random_low: f64 = reference.random_distribution[..low].iter().sum();
    let observed_low: f64 = observed[..low].iter().sum();
    let scale = if random_low > 0.0 {
        observed_low / random_low
    } else {
        1.0
    };
    let scaled_random = reference
        .random_distribution
        .iter()
        .map(|value| value * scale)
        .collect::<Vec<_>>();
    let mut probability = scaled_random
        .iter()
        .zip(&observed)
        .map(|(&random, &actual)| (random / if actual == 0.0 { 1.0 } else { actual }).min(1.0))
        .collect::<Vec<_>>();
    probability[..low].fill(1.0);
    let last = reference
        .last_nonzero_random_bin_one_based
        .min(probability.len());
    probability[last..].fill(0.0);
    let window = gaussian_window(7);
    let normalization: f64 = window.iter().sum();
    let mut padded = vec![1.0; 7];
    padded.extend_from_slice(&probability);
    probability = fir_filter(&window, &padded)[7..]
        .iter()
        .map(|value| value / normalization)
        .collect();
    let mut interpolation_input = vec![1.0];
    interpolation_input.extend_from_slice(&probability);
    let mut high_resolution = matlab_interp(&interpolation_input, 10)?;
    high_resolution.truncate(high_resolution.len().saturating_sub(9));
    let weighting = coherence
        .iter()
        .map(|&value| {
            let index = (value * 1000.0)
                .round()
                .clamp(0.0, (high_resolution.len() - 1) as f64) as usize;
            (1.0 - high_resolution[index]).powi(2)
        })
        .collect();
    Ok(PsquareOutput {
        observed_distribution: observed,
        scaled_random_distribution: scaled_random,
        random_probability: probability,
        weighting,
    })
}

pub fn signal_noise_weighting(amplitude: &Matrix<f32>, residual_angle: &Matrix<f32>) -> Vec<f64> {
    (0..amplitude.rows)
        .map(|row| {
            let g = amplitude
                .row(row)
                .iter()
                .zip(residual_angle.row(row))
                .map(|(&amp, &phase)| f64::from(amp) * f64::from(phase).cos())
                .sum::<f64>()
                / amplitude.cols as f64;
            let mean_square = amplitude
                .row(row)
                .iter()
                .map(|&value| f64::from(value).powi(2))
                .sum::<f64>()
                / amplitude.cols as f64;
            let sigma = (0.5 * (mean_square - g * g)).sqrt();
            if sigma == 0.0 {
                0.0
            } else {
                g / sigma
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_has_matlab_length_contract() {
        let values = matlab_interp(&[1.0, 2.0, 3.0], 10).unwrap();
        assert_eq!(values.len(), 30);
        for (index, expected) in [
            (0, 1.0),
            (1, 1.076_054_830_445_159_8),
            (10, 2.0),
            (19, 3.084_419_620_794_768),
            (29, 0.262_675_596_487_962_96),
        ] {
            assert!((values[index] - expected).abs() < 1e-12);
        }
    }
}
