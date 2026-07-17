use num_complex::Complex64;

pub(super) fn variance_columns_real(
    data: &[f64],
    rows: usize,
    cols: usize,
    ddof: usize,
) -> Vec<f64> {
    let denominator = rows.saturating_sub(ddof);
    (0..cols)
        .map(|col| {
            if denominator == 0 {
                return 0.0;
            }
            let mean = (0..rows).map(|row| data[row * cols + col]).sum::<f64>() / rows as f64;
            (0..rows)
                .map(|row| (data[row * cols + col] - mean).powi(2))
                .sum::<f64>()
                / denominator as f64
        })
        .collect()
}

pub(super) fn variance_columns_complex(
    data: &[Complex64],
    rows: usize,
    cols: usize,
    ddof: usize,
) -> Vec<f64> {
    let denominator = rows.saturating_sub(ddof);
    (0..cols)
        .map(|col| {
            if denominator == 0 {
                return 0.0;
            }
            let mean = (0..rows)
                .map(|row| data[row * cols + col])
                .sum::<Complex64>()
                / rows as f64;
            (0..rows)
                .map(|row| (data[row * cols + col] - mean).norm_sqr())
                .sum::<f64>()
                / denominator as f64
        })
        .collect()
}

pub(super) fn standard_deviation_and_maximum(
    data: &[f64],
    _rows: usize,
    cols: usize,
    ddof: usize,
) -> (Vec<f64>, Vec<f64>) {
    let denominator = cols.saturating_sub(ddof);
    let pairs = data
        .chunks_exact(cols)
        .map(|row| {
            let mean = row.iter().sum::<f64>() / cols as f64;
            let deviation = if denominator == 0 {
                0.0
            } else {
                (row.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / denominator as f64)
                    .sqrt()
            };
            let maximum = row
                .iter()
                .fold(0.0_f64, |current, value| current.max(value.abs()));
            (deviation, maximum)
        })
        .collect::<Vec<_>>();
    (
        pairs.iter().map(|value| value.0).collect(),
        pairs.iter().map(|value| value.1).collect(),
    )
}

fn fit_columns(weights: &[f64]) -> Vec<usize> {
    let infinite = weights
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.is_infinite().then_some(index))
        .collect::<Vec<_>>();
    if infinite.is_empty() {
        weights
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (value.is_finite() && *value > 0.0).then_some(index))
            .collect()
    } else {
        infinite
    }
}

pub(super) fn weighted_slope_real(
    x: &[f64],
    y: &[f64],
    rows: usize,
    cols: usize,
    weights: &[f64],
) -> Vec<f64> {
    let selected = fit_columns(weights);
    let infinite = selected
        .first()
        .is_some_and(|index| weights[*index].is_infinite());
    let denominator: f64 = selected
        .iter()
        .map(|&col| (if infinite { 1.0 } else { weights[col] }) * x[col] * x[col])
        .sum();
    if denominator == 0.0 {
        return vec![0.0; rows];
    }
    (0..rows)
        .map(|row| {
            selected
                .iter()
                .map(|&col| {
                    y[row * cols + col] * (if infinite { 1.0 } else { weights[col] }) * x[col]
                })
                .sum::<f64>()
                / denominator
        })
        .collect()
}

pub(super) fn weighted_slope_complex(
    x: &[f64],
    y: &[Complex64],
    rows: usize,
    cols: usize,
    weights: &[f64],
) -> Vec<Complex64> {
    let selected = fit_columns(weights);
    let infinite = selected
        .first()
        .is_some_and(|index| weights[*index].is_infinite());
    let denominator: f64 = selected
        .iter()
        .map(|&col| (if infinite { 1.0 } else { weights[col] }) * x[col] * x[col])
        .sum();
    if denominator == 0.0 {
        return vec![Complex64::new(0.0, 0.0); rows];
    }
    (0..rows)
        .map(|row| {
            selected
                .iter()
                .map(|&col| {
                    y[row * cols + col] * ((if infinite { 1.0 } else { weights[col] }) * x[col])
                })
                .sum::<Complex64>()
                / denominator
        })
        .collect()
}

#[cfg(test)]
pub(super) fn weighted_affine(
    x: &[f64],
    y: &[f64],
    rows: usize,
    cols: usize,
    weights: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let s0: f64 = weights.iter().sum();
    let s1: f64 = weights
        .iter()
        .zip(x)
        .map(|(weight, value)| weight * value)
        .sum();
    let s2: f64 = weights
        .iter()
        .zip(x)
        .map(|(weight, value)| weight * value * value)
        .sum();
    let determinant = s0 * s2 - s1 * s1;
    let mut intercept = vec![0.0; rows];
    let mut slope = vec![0.0; rows];
    for row in 0..rows {
        let wy0 = (0..cols)
            .map(|col| y[row * cols + col] * weights[col])
            .sum::<f64>();
        let wy1 = (0..cols)
            .map(|col| y[row * cols + col] * weights[col] * x[col])
            .sum::<f64>();
        if determinant == 0.0 {
            if s0 != 0.0 {
                intercept[row] = wy0 / s0;
            }
        } else {
            intercept[row] = (wy0 * s2 - wy1 * s1) / determinant;
            slope[row] = (wy1 * s0 - wy0 * s1) / determinant;
        }
    }
    (intercept, slope)
}

pub(super) fn wrap_phase(value: f64) -> f64 {
    value.sin().atan2(value.cos())
}
