use num_complex::Complex64;

use super::fit::{
    standard_deviation_and_maximum, variance_columns_real, weighted_affine, weighted_slope_real,
    wrap_phase,
};
use super::noise::single_master_noise;

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

fn serial_reference(
    phase: &[Complex64],
    edges: usize,
    interferograms: usize,
    bperp: &[f64],
    day: &[f64],
    time_window: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (time_difference, weights) = temporal_weights(day, time_window);
    let mut smooth = vec![Complex64::new(0.0, 0.0); phase.len()];
    for edge in 0..edges {
        for output in 0..interferograms {
            for source in 0..interferograms {
                smooth[edge * interferograms + output] += phase[edge * interferograms + source]
                    * weights[output * interferograms + source];
            }
        }
    }
    let mut leave_one_out = smooth.clone();
    for edge in 0..edges {
        for ifg in 0..interferograms {
            leave_one_out[edge * interferograms + ifg] -=
                phase[edge * interferograms + ifg] * weights[ifg * interferograms + ifg];
        }
    }
    let mut fitted = vec![Complex64::new(0.0, 0.0); phase.len()];
    for output in 0..interferograms {
        let range = output * interferograms..(output + 1) * interferograms;
        let difference = &time_difference[range.clone()];
        let weight = &weights[range];
        let mut adjusted = vec![0.0; edges * interferograms];
        for edge in 0..edges {
            let mean = smooth[edge * interferograms + output];
            for source in 0..interferograms {
                adjusted[edge * interferograms + source] =
                    (phase[edge * interferograms + source] * mean.conj()).arg();
            }
        }
        let (intercept, slope) =
            weighted_affine(difference, &adjusted, edges, interferograms, weight);
        let mut detrended = vec![0.0; adjusted.len()];
        for edge in 0..edges {
            for source in 0..interferograms {
                detrended[edge * interferograms + source] = wrap_phase(
                    adjusted[edge * interferograms + source]
                        - intercept[edge]
                        - slope[edge] * difference[source],
                );
            }
        }
        let (second_intercept, _) =
            weighted_affine(difference, &detrended, edges, interferograms, weight);
        for edge in 0..edges {
            fitted[edge * interferograms + output] = smooth[edge * interferograms + output]
                * Complex64::from_polar(1.0, intercept[edge] + second_intercept[edge]);
        }
    }
    let mut noise = vec![0.0; phase.len()];
    let mut leave_one_out_noise = vec![0.0; phase.len()];
    for index in 0..phase.len() {
        noise[index] = (phase[index] * fitted[index].conj()).arg();
        leave_one_out_noise[index] = (phase[index] * leave_one_out[index].conj()).arg();
    }
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
    for edge in 0..edges {
        for ifg in 0..interferograms {
            noise[edge * interferograms + ifg] -= slopes[edge] * bperp[ifg];
        }
    }
    standard_deviation_and_maximum(
        &noise,
        edges,
        interferograms,
        usize::from(interferograms > 1),
    )
}

#[test]
fn edge_parallel_single_master_matches_serial_reference() {
    let edges = 17;
    let interferograms = 9;
    let phase = (0..edges * interferograms)
        .map(|index| Complex64::from_polar(1.0, (index as f64 * 0.371).sin()))
        .collect::<Vec<_>>();
    let bperp = (0..interferograms)
        .map(|index| index as f64 * 13.0 - 47.0)
        .collect::<Vec<_>>();
    let day = (0..interferograms)
        .map(|index| 730_000.0 + index as f64 * 24.0)
        .collect::<Vec<_>>();
    let expected = serial_reference(&phase, edges, interferograms, &bperp, &day, 730.0);
    let observed = single_master_noise(&phase, edges, interferograms, &bperp, &day, 730.0);
    for (left, right) in expected.0.iter().zip(&observed.0) {
        assert!(
            (left - right).abs() < 1e-12,
            "std mismatch: {left} vs {right}"
        );
    }
    for (left, right) in expected.1.iter().zip(&observed.1) {
        assert!(
            (left - right).abs() < 1e-12,
            "max mismatch: {left} vs {right}"
        );
    }
}
