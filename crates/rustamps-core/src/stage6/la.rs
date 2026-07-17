use num_complex::{Complex32, Complex64};
use std::f64::consts::PI;

use super::{require_shape, Stage6Error};

fn insert_master_index(day: &[f64]) -> usize {
    day.iter()
        .enumerate()
        .filter(|(_, value)| **value > 0.0)
        .min_by(|left, right| left.1.total_cmp(right.1))
        .map_or_else(|| day.len().saturating_sub(1), |(index, _)| index)
}

fn temp_value(row: &[Complex32], insert: usize, mean_abs: f64, column: usize) -> Complex64 {
    if column < insert {
        Complex64::new(row[column].re.into(), row[column].im.into())
    } else if column == insert {
        Complex64::new(mean_abs, 0.0)
    } else {
        Complex64::new(row[column - 1].re.into(), row[column - 1].im.into())
    }
}

pub fn estimate_la_error_single_master(
    dph_space: &[Complex32],
    n_edge: usize,
    n_ifg: usize,
    day: &[f64],
    bperp: &[f64],
    n_trial_wraps: f64,
) -> Result<Vec<f32>, Stage6Error> {
    require_shape(dph_space, n_edge, n_ifg, "dph_space")?;
    if n_ifg == 0 || day.len() != n_ifg || bperp.len() != n_ifg {
        return Err(Stage6Error::new(
            "day and bperp must align with non-empty phase columns",
        ));
    }
    if day.iter().chain(bperp).any(|value| !value.is_finite())
        || !n_trial_wraps.is_finite()
        || n_trial_wraps < 0.0
    {
        return Err(Stage6Error::new(
            "look-angle inputs must be finite and trial wraps non-negative",
        ));
    }
    if n_edge == 0 {
        return Ok(Vec::new());
    }

    let insert = insert_master_index(day);
    let mut full_bperp = Vec::with_capacity(n_ifg + 1);
    full_bperp.extend_from_slice(&bperp[..insert]);
    full_bperp.push(0.0);
    full_bperp.extend_from_slice(&bperp[insert..]);
    let differences = full_bperp
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    let original_range = range(bperp);
    let difference_range = range(&differences);
    let scaled_wraps = if original_range == 0.0 {
        n_trial_wraps
    } else {
        n_trial_wraps * difference_range / original_range
    };
    let selected_indices = differences
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value != 0.0).then_some(index))
        .collect::<Vec<_>>();
    let selected_bperp = selected_indices
        .iter()
        .map(|&index| differences[index])
        .collect::<Vec<_>>();
    let safe_range = difference_range.max(1.0e-12);
    let trial_limit = (8.0 * scaled_wraps).ceil() as i32;
    let trial_multipliers = (-trial_limit..=trial_limit).collect::<Vec<_>>();
    let trial_phase = selected_bperp
        .iter()
        .map(|value| value / safe_range * PI / 4.0)
        .collect::<Vec<_>>();

    Ok(dph_space
        .chunks_exact(n_ifg)
        .map(|row| {
            let (value, coherence) = estimate_row(
                row,
                insert,
                &selected_indices,
                &selected_bperp,
                safe_range,
                &trial_multipliers,
                &trial_phase,
            );
            if coherence < 0.31 {
                0.0
            } else {
                value
            }
        })
        .collect())
}

fn range(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - values.iter().copied().fold(f64::INFINITY, f64::min)
}

#[allow(clippy::too_many_arguments)]
fn estimate_row(
    row: &[Complex32],
    insert: usize,
    selected_indices: &[usize],
    bperp_difference: &[f64],
    safe_range: f64,
    trial_multipliers: &[i32],
    trial_phase: &[f64],
) -> (f32, f32) {
    if bperp_difference.is_empty() || trial_multipliers.is_empty() {
        return (0.0, 0.0);
    }
    let mean_abs =
        row.iter().map(|value| value.norm() as f64).sum::<f64>() / row.len().max(1) as f64;
    let full = (0..row.len())
        .map(|column| {
            let mut value = temp_value(row, insert, mean_abs, column + 1)
                * temp_value(row, insert, mean_abs, column).conj();
            let amplitude = value.norm();
            if amplitude == 0.0 {
                Complex64::new(0.0, 0.0)
            } else {
                value /= amplitude;
                value
            }
        })
        .collect::<Vec<_>>();
    let selected = selected_indices
        .iter()
        .map(|&index| full[index])
        .collect::<Vec<_>>();
    let denominator = selected.iter().map(|value| value.norm()).sum::<f64>();
    if denominator == 0.0 {
        return (0.0, 0.0);
    }
    let trials = trial_multipliers
        .iter()
        .map(|trial| {
            selected
                .iter()
                .zip(trial_phase)
                .map(|(value, phase)| {
                    let angle = -phase * f64::from(*trial);
                    *value * Complex64::new(angle.cos(), angle.sin())
                })
                .sum::<Complex64>()
                .norm()
                / denominator
        })
        .collect::<Vec<_>>();
    let (peak_index, peak) = trials
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((0, 0.0));
    let peak_start = (0..peak_index)
        .filter(|&index| trials[index + 1] < trials[index])
        .map(|index| index + 1)
        .next_back()
        .unwrap_or(0);
    let peak_end = (peak_index..trials.len().saturating_sub(1))
        .find(|&index| trials[index + 1] > trials[index])
        .unwrap_or(trials.len() - 1);
    let next_peak = trials
        .iter()
        .enumerate()
        .filter(|(index, _)| *index < peak_start || *index > peak_end)
        .map(|(_, value)| *value)
        .fold(0.0_f64, f64::max);
    if peak - next_peak <= 0.1 {
        return (0.0, 0.0);
    }

    let k0 = PI / 4.0 / safe_range * f64::from(trial_multipliers[peak_index]);
    let residual = selected
        .iter()
        .zip(bperp_difference)
        .map(|(value, baseline)| {
            let angle = -k0 * baseline;
            *value * Complex64::new(angle.cos(), angle.sin())
        })
        .collect::<Vec<_>>();
    let offset = residual.iter().copied().sum::<Complex64>();
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for ((value, baseline), residual) in selected.iter().zip(bperp_difference).zip(residual) {
        let weight = value.norm();
        let weighted_baseline = weight * baseline;
        let phase = residual * offset.conj();
        numerator += weighted_baseline * weight * phase.im.atan2(phase.re);
        denominator += weighted_baseline * weighted_baseline;
    }
    let estimate = k0
        + if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        };
    let corrected = selected
        .iter()
        .zip(bperp_difference)
        .map(|(value, baseline)| {
            let angle = -estimate * baseline;
            *value * Complex64::new(angle.cos(), angle.sin())
        })
        .collect::<Vec<_>>();
    let sum = corrected.iter().copied().sum::<Complex64>();
    let magnitude = corrected.iter().map(|value| value.norm()).sum::<f64>();
    (
        estimate as f32,
        (sum.norm() / magnitude.max(f64::EPSILON)) as f32,
    )
}
