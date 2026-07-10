use num_complex::Complex64;

pub(super) fn variance_cols_real(
    data: &[f64],
    n_row: usize,
    n_col: usize,
    ddof: usize,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_col];
    if n_row == 0 || n_col == 0 {
        return out;
    }
    let denom = n_row.saturating_sub(ddof);
    if denom == 0 {
        return out;
    }
    for col_ix in 0..n_col {
        let mut mean = 0.0_f64;
        for row_ix in 0..n_row {
            mean += data[row_ix * n_col + col_ix];
        }
        mean /= n_row as f64;
        let mut accum = 0.0_f64;
        for row_ix in 0..n_row {
            let delta = data[row_ix * n_col + col_ix] - mean;
            accum += delta * delta;
        }
        out[col_ix] = accum / denom as f64;
    }
    out
}

pub(super) fn variance_cols_complex(
    data: &[Complex64],
    n_row: usize,
    n_col: usize,
    ddof: usize,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_col];
    if n_row == 0 || n_col == 0 {
        return out;
    }
    let denom = n_row.saturating_sub(ddof);
    if denom == 0 {
        return out;
    }
    for col_ix in 0..n_col {
        let mut mean = Complex64::new(0.0, 0.0);
        for row_ix in 0..n_row {
            mean += data[row_ix * n_col + col_ix];
        }
        mean /= n_row as f64;
        let mut accum = 0.0_f64;
        for row_ix in 0..n_row {
            let delta = data[row_ix * n_col + col_ix] - mean;
            accum += delta.norm_sqr();
        }
        out[col_ix] = accum / denom as f64;
    }
    out
}

pub(super) fn std_max_rows_real(
    data: &[f64],
    n_row: usize,
    n_col: usize,
    ddof: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut std = vec![0.0_f64; n_row];
    let mut max_abs = vec![0.0_f64; n_row];
    if n_row == 0 || n_col == 0 {
        return (std, max_abs);
    }
    let denom = n_col.saturating_sub(ddof);
    for row_ix in 0..n_row {
        let row = &data[row_ix * n_col..(row_ix + 1) * n_col];
        let mean = row.iter().copied().sum::<f64>() / n_col as f64;
        let mut accum = 0.0_f64;
        let mut max_value = 0.0_f64;
        for &value in row {
            let delta = value - mean;
            accum += delta * delta;
            max_value = max_value.max(value.abs());
        }
        std[row_ix] = if denom == 0 {
            0.0
        } else {
            (accum / denom as f64).sqrt()
        };
        max_abs[row_ix] = max_value;
    }
    (std, max_abs)
}
