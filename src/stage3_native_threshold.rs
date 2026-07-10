use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn histogram_with_centers(values: &[f64], centers: &[f64]) -> Vec<f64> {
    if centers.is_empty() {
        return Vec::new();
    }
    if centers.len() == 1 {
        return vec![values.len() as f64];
    }
    let mids: Vec<f64> = centers
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) / 2.0)
        .collect();
    let mut counts = vec![0.0_f64; centers.len()];
    for &value in values {
        let mut lo = 0usize;
        let mut hi = mids.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if mids[mid] < value {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        counts[lo.min(centers.len() - 1)] += 1.0;
    }
    counts
}

fn solve_linear(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = a[col * n + col].abs();
        for row in (col + 1)..n {
            let value_abs = a[row * n + col].abs();
            if value_abs > pivot_abs {
                pivot = row;
                pivot_abs = value_abs;
            }
        }
        if pivot_abs <= f64::EPSILON {
            return None;
        }
        if pivot != col {
            for c in 0..n {
                a.swap(col * n + c, pivot * n + c);
            }
            b.swap(col, pivot);
        }
        let diag = a[col * n + col];
        for c in col..n {
            a[col * n + c] /= diag;
        }
        b[col] /= diag;
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * n + col];
            if factor == 0.0 {
                continue;
            }
            for c in col..n {
                a[row * n + c] -= factor * a[col * n + c];
            }
            b[row] -= factor * b[col];
        }
    }
    Some(b)
}

fn polyfit_eval_centered(x: &[f64], y: &[f64], degree: usize, x_eval: f64) -> f64 {
    if x.is_empty() || y.is_empty() || x.len() != y.len() {
        return f64::NAN;
    }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let mut std = 1.0;
    if x.len() > 1 {
        let var = x
            .iter()
            .map(|value| {
                let diff = *value - mean;
                diff * diff
            })
            .sum::<f64>()
            / (x.len() as f64 - 1.0);
        std = var.sqrt();
        if !std.is_finite() || std == 0.0 {
            std = 1.0;
        }
    }
    let n_coeff = degree + 1;
    let mut gram = vec![0.0_f64; n_coeff * n_coeff];
    let mut rhs = vec![0.0_f64; n_coeff];
    for (&x_value, &y_value) in x.iter().zip(y.iter()) {
        let scaled = (x_value - mean) / std;
        let mut powers = vec![1.0_f64; n_coeff];
        for ix in 1..n_coeff {
            powers[ix] = powers[ix - 1] * scaled;
        }
        for row in 0..n_coeff {
            let row_basis = powers[degree - row];
            rhs[row] += row_basis * y_value;
            for col in 0..n_coeff {
                gram[row * n_coeff + col] += row_basis * powers[degree - col];
            }
        }
    }
    let Some(coeffs) = solve_linear(gram, rhs, n_coeff) else {
        return f64::NAN;
    };
    let scaled_eval = (x_eval - mean) / std;
    coeffs
        .iter()
        .fold(0.0_f64, |acc, coeff| acc * scaled_eval + coeff)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn stage3_coh_threshold<'py>(
    py: Python<'py>,
    coh_values: PyReadonlyArray1<f64>,
    d_a: PyReadonlyArray1<f64>,
    d_a_max: PyReadonlyArray1<f64>,
    coh_bins: PyReadonlyArray1<f64>,
    nr_dist: PyReadonlyArray1<f64>,
    low_coh_thresh: usize,
    max_percent_rand: f64,
    select_method: &str,
    _threads: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let coh_view = coh_values.as_array();
    let d_a_view = d_a.as_array();
    let d_a_max_view = d_a_max.as_array();
    let coh_bins_view = coh_bins.as_array();
    let nr_view = nr_dist.as_array();
    let coh = coh_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("coh_values must be contiguous"))?;
    let da = d_a_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("D_A must be contiguous"))?;
    let da_max = d_a_max_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("D_A_max must be contiguous"))?;
    let bins = coh_bins_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("coh_bins must be contiguous"))?;
    let nr_dist = nr_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Nr_dist must be contiguous"))?;
    if coh.len() != da.len() {
        return Err(PyValueError::new_err(
            "coh_values and D_A must have matching lengths",
        ));
    }
    if nr_dist.len() != bins.len() {
        return Err(PyValueError::new_err(
            "Nr_dist and coh_bins must have matching lengths",
        ));
    }
    if da_max.is_empty() {
        return Ok((
            vec![0.3_f64; coh.len()].into_pyarray(py),
            Vec::<f64>::new().into_pyarray(py),
        ));
    }

    let n_bin = da_max.len().saturating_sub(1);
    let mut min_coh = vec![f64::NAN; n_bin];
    let mut da_mean = vec![f64::NAN; n_bin];
    let percent_mode = select_method.eq_ignore_ascii_case("PERCENT");

    for i in 0..n_bin {
        let lo = da_max[i];
        let hi = da_max[i + 1];
        let mut coh_chunk = Vec::new();
        let mut da_sum = 0.0;
        let mut da_count = 0usize;
        for (&coh_value, &da_value) in coh.iter().zip(da.iter()) {
            if da_value > lo && da_value <= hi {
                da_sum += da_value;
                da_count += 1;
                if coh_value.is_finite() && coh_value != 0.0 {
                    coh_chunk.push(coh_value);
                }
            }
        }
        if da_count == 0 || coh_chunk.is_empty() {
            continue;
        }
        da_mean[i] = da_sum / da_count as f64;
        let na = histogram_with_centers(&coh_chunk, bins);
        let low_cut = low_coh_thresh.min(na.len());
        let denom: f64 = nr_dist.iter().take(low_cut).sum();
        let na_low_sum: f64 = na.iter().take(low_cut).sum();
        let scale = if denom > 0.0 { na_low_sum / denom } else { 1.0 };
        let nr: Vec<f64> = nr_dist.iter().map(|value| value * scale).collect();
        let mut na_safe = na;
        for value in &mut na_safe {
            if *value == 0.0 {
                *value = 1.0;
            }
        }
        let mut percent_rand = vec![0.0_f64; na_safe.len()];
        let mut nr_cumsum = 0.0;
        let mut na_cumsum = 0.0;
        for rev_ix in (0..na_safe.len()).rev() {
            nr_cumsum += nr[rev_ix];
            na_cumsum += na_safe[rev_ix];
            percent_rand[rev_ix] = if percent_mode {
                nr_cumsum / na_cumsum * 100.0
            } else {
                nr_cumsum
            };
        }
        let Some(min_ok_ix) = percent_rand
            .iter()
            .position(|value| *value < max_percent_rand)
        else {
            min_coh[i] = 1.0;
            continue;
        };
        let min_ok_1b = min_ok_ix + 1;
        let min_fit_ix = min_ok_1b as isize - 3;
        if min_fit_ix <= 0 {
            continue;
        }
        let max_fit_ix = (min_ok_1b + 2).min(100);
        let start = min_fit_ix as usize - 1;
        let end = max_fit_ix.min(percent_rand.len());
        if end <= start || end - start < 4 {
            continue;
        }
        let xs = &percent_rand[start..end];
        let ys: Vec<f64> = (min_fit_ix as usize..=end)
            .map(|ix| ix as f64 * 0.01)
            .collect();
        min_coh[i] = polyfit_eval_centered(xs, &ys, 3, max_percent_rand);
    }

    let valid: Vec<(f64, f64)> = min_coh
        .iter()
        .zip(da_mean.iter())
        .filter_map(|(&min_value, &mean_value)| {
            if min_value.is_nan() || mean_value.is_nan() {
                None
            } else {
                Some((mean_value, min_value))
            }
        })
        .collect();
    let mut coeffs = Vec::new();
    let mut threshold = if valid.is_empty() {
        vec![0.3_f64; coh.len()]
    } else if valid.len() == 1 {
        vec![valid[0].1; coh.len()]
    } else {
        let n = valid.len() as f64;
        let sum_x: f64 = valid.iter().map(|pair| pair.0).sum();
        let sum_y: f64 = valid.iter().map(|pair| pair.1).sum();
        let sum_xx: f64 = valid.iter().map(|pair| pair.0 * pair.0).sum();
        let sum_xy: f64 = valid.iter().map(|pair| pair.0 * pair.1).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom == 0.0 {
            vec![sum_y / n; coh.len()]
        } else {
            let slope = (n * sum_xy - sum_x * sum_y) / denom;
            let intercept = (sum_y - slope * sum_x) / n;
            if slope > 0.0 {
                coeffs.push(slope);
                coeffs.push(intercept);
                da.iter().map(|value| slope * *value + intercept).collect()
            } else {
                vec![slope * 0.35 + intercept; coh.len()]
            }
        }
    };
    for value in &mut threshold {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
    Ok((threshold.into_pyarray(py), coeffs.into_pyarray(py)))
}
