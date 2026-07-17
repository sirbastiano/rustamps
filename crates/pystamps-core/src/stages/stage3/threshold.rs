use super::{SelectMethod, Stage3Error};
use crate::stages::stage2::histogram_with_centers;

#[derive(Clone, Debug)]
pub struct CoherenceThresholdInput<'a> {
    pub coherence: &'a [f64],
    pub amplitude_dispersion: &'a [f64],
    pub dispersion_edges: &'a [f64],
    pub coherence_bins: &'a [f64],
    pub random_distribution: &'a [f64],
    pub low_coherence_bins: usize,
    pub maximum_random: f64,
    pub method: SelectMethod,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CoherenceThresholdOutput {
    pub threshold: Vec<f64>,
    pub linear_coefficients: Vec<f64>,
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

fn cubic_fit(x: &[f64], y: &[f64], evaluate_at: f64) -> f64 {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let mut deviation = if x.len() > 1 {
        (x.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt()
    } else {
        1.0
    };
    if !deviation.is_finite() || deviation == 0.0 {
        deviation = 1.0;
    }
    let mut gram = vec![0.0; 16];
    let mut rhs = vec![0.0; 4];
    for (&x_value, &y_value) in x.iter().zip(y) {
        let scaled = (x_value - mean) / deviation;
        let basis = [scaled.powi(3), scaled.powi(2), scaled, 1.0];
        for row in 0..4 {
            rhs[row] += basis[row] * y_value;
            for col in 0..4 {
                gram[row * 4 + col] += basis[row] * basis[col];
            }
        }
    }
    let Some(coefficients) = solve(gram, rhs, 4) else {
        return f64::NAN;
    };
    let scaled = (evaluate_at - mean) / deviation;
    coefficients
        .iter()
        .fold(0.0, |value, coefficient| value * scaled + coefficient)
}

pub fn coherence_threshold(
    input: &CoherenceThresholdInput<'_>,
) -> Result<CoherenceThresholdOutput, Stage3Error> {
    if input.coherence.len() != input.amplitude_dispersion.len()
        || input.coherence_bins.len() != input.random_distribution.len()
        || input.coherence_bins.is_empty()
        || input.dispersion_edges.len() < 2
    {
        return Err(Stage3Error::InvalidInput(
            "invalid coherence-threshold arrays",
        ));
    }
    let bin_count = input.dispersion_edges.len() - 1;
    let mut minimum_coherence = vec![f64::NAN; bin_count];
    let mut mean_dispersion = vec![f64::NAN; bin_count];
    for bin in 0..bin_count {
        let low = input.dispersion_edges[bin];
        let high = input.dispersion_edges[bin + 1];
        let indices = input
            .amplitude_dispersion
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value > low && value <= high).then_some(index))
            .collect::<Vec<_>>();
        let coherence = indices
            .iter()
            .filter_map(|&index| {
                let value = input.coherence[index];
                (value.is_finite() && value != 0.0).then_some(value)
            })
            .collect::<Vec<_>>();
        if indices.is_empty() || coherence.is_empty() {
            continue;
        }
        mean_dispersion[bin] = indices
            .iter()
            .map(|&index| input.amplitude_dispersion[index])
            .sum::<f64>()
            / indices.len() as f64;
        let observed = histogram_with_centers(&coherence, input.coherence_bins);
        let low_bins = input.low_coherence_bins.min(observed.len());
        let random_low: f64 = input.random_distribution[..low_bins].iter().sum();
        let observed_low: f64 = observed[..low_bins].iter().sum();
        let scale = if random_low > 0.0 {
            observed_low / random_low
        } else {
            1.0
        };
        let random = input
            .random_distribution
            .iter()
            .map(|value| value * scale)
            .collect::<Vec<_>>();
        let mut tail_random = 0.0;
        let mut tail_observed = 0.0;
        let mut random_metric = vec![0.0; observed.len()];
        for index in (0..observed.len()).rev() {
            tail_random += random[index];
            tail_observed += if observed[index] == 0.0 {
                1.0
            } else {
                observed[index]
            };
            random_metric[index] = match input.method {
                SelectMethod::Percent => tail_random / tail_observed * 100.0,
                SelectMethod::Density => tail_random,
            };
        }
        let Some(first_ok) = random_metric
            .iter()
            .position(|value| *value < input.maximum_random)
        else {
            minimum_coherence[bin] = 1.0;
            continue;
        };
        let first_one_based = first_ok + 1;
        let fit_start_one_based = first_one_based as isize - 3;
        if fit_start_one_based <= 0 {
            continue;
        }
        let fit_end_one_based = (first_one_based + 2).min(100).min(random_metric.len());
        let start = fit_start_one_based as usize - 1;
        if fit_end_one_based <= start || fit_end_one_based - start < 4 {
            continue;
        }
        let x = &random_metric[start..fit_end_one_based];
        let y = (fit_start_one_based as usize..=fit_end_one_based)
            .map(|index| index as f64 * 0.01)
            .collect::<Vec<_>>();
        minimum_coherence[bin] = cubic_fit(x, &y, input.maximum_random);
    }
    let valid = mean_dispersion
        .iter()
        .zip(&minimum_coherence)
        .filter_map(|(&dispersion, &threshold)| {
            (dispersion.is_finite() && threshold.is_finite()).then_some((dispersion, threshold))
        })
        .collect::<Vec<_>>();
    let mut coefficients = Vec::new();
    let mut threshold = if valid.is_empty() {
        vec![0.3; input.coherence.len()]
    } else if valid.len() == 1 {
        vec![valid[0].1; input.coherence.len()]
    } else {
        let count = valid.len() as f64;
        let sum_x: f64 = valid.iter().map(|value| value.0).sum();
        let sum_y: f64 = valid.iter().map(|value| value.1).sum();
        let sum_xx: f64 = valid.iter().map(|value| value.0 * value.0).sum();
        let sum_xy: f64 = valid.iter().map(|value| value.0 * value.1).sum();
        let denominator = count * sum_xx - sum_x * sum_x;
        let (slope, intercept) = if denominator == 0.0 {
            (0.0, sum_y / count)
        } else {
            let slope = (count * sum_xy - sum_x * sum_y) / denominator;
            (slope, (sum_y - slope * sum_x) / count)
        };
        if slope > 0.0 {
            coefficients = vec![slope, intercept];
            input
                .amplitude_dispersion
                .iter()
                .map(|value| slope * value + intercept)
                .collect()
        } else {
            vec![slope * 0.35 + intercept; input.coherence.len()]
        }
    };
    threshold
        .iter_mut()
        .for_each(|value| *value = value.max(0.0));
    Ok(CoherenceThresholdOutput {
        threshold,
        linear_coefficients: coefficients,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_valid_bins_use_reference_threshold() {
        let output = coherence_threshold(&CoherenceThresholdInput {
            coherence: &[0.0, 0.0],
            amplitude_dispersion: &[1.0, 1.0],
            dispersion_edges: &[0.0, 1.0],
            coherence_bins: &[0.0, 1.0],
            random_distribution: &[1.0, 1.0],
            low_coherence_bins: 1,
            maximum_random: 1.0,
            method: SelectMethod::Percent,
        })
        .unwrap();
        assert_eq!(output.threshold, vec![0.3, 0.3]);
    }
}
