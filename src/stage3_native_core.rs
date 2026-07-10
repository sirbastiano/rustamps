use num_complex::Complex64;
use rustfft::FftPlanner;

pub(super) fn fft2_in_place(values: &mut [Complex64], n_row: usize, n_col: usize, inverse: bool) {
    let mut planner = FftPlanner::<f64>::new();
    let row_fft = if inverse {
        planner.plan_fft_inverse(n_col)
    } else {
        planner.plan_fft_forward(n_col)
    };
    for row in 0..n_row {
        let start = row * n_col;
        row_fft.process(&mut values[start..start + n_col]);
    }

    let col_fft = if inverse {
        planner.plan_fft_inverse(n_row)
    } else {
        planner.plan_fft_forward(n_row)
    };
    let mut column = vec![Complex64::new(0.0, 0.0); n_row];
    for col in 0..n_col {
        for row in 0..n_row {
            column[row] = values[row * n_col + col];
        }
        col_fft.process(&mut column);
        for row in 0..n_row {
            values[row * n_col + col] = column[row];
        }
    }

    if inverse {
        let scale = (n_row * n_col) as f64;
        for value in values {
            *value /= scale;
        }
    }
}

pub(super) fn roll_real(
    values: &[f64],
    n_row: usize,
    n_col: usize,
    row_shift: isize,
    col_shift: isize,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_row * n_col];
    for row in 0..n_row {
        for col in 0..n_col {
            let src_row = (row as isize - row_shift).rem_euclid(n_row as isize) as usize;
            let src_col = (col as isize - col_shift).rem_euclid(n_col as isize) as usize;
            out[row * n_col + col] = values[src_row * n_col + src_col];
        }
    }
    out
}

pub(super) fn clap_filter_kernel_values() -> [f64; 49] {
    let alpha = 2.5_f64;
    let std = (7.0_f64 - 1.0) / (2.0 * alpha);
    let center = (7.0_f64 - 1.0) / 2.0;
    let mut g = [0.0_f64; 7];
    for (idx, value) in g.iter_mut().enumerate() {
        let x = (idx as f64 - center) / std;
        *value = (-0.5 * x * x).exp();
    }
    let mut out = [0.0_f64; 49];
    for row in 0..7 {
        for col in 0..7 {
            out[row * 7 + col] = g[row] * g[col];
        }
    }
    out
}

pub(super) fn gaussian_window(n: usize, alpha: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 || alpha <= 0.0 {
        return vec![1.0; n];
    }
    let std = (n as f64 - 1.0) / (2.0 * alpha);
    let center = (n as f64 - 1.0) / 2.0;
    (0..n)
        .map(|idx| {
            let x = (idx as f64 - center) / std;
            (-0.5 * x * x).exp()
        })
        .collect()
}

pub(super) fn convolve_same_7x7(
    values: &[f64],
    n_row: usize,
    n_col: usize,
    kernel: &[f64; 49],
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_row * n_col];
    for row in 0..n_row {
        for col in 0..n_col {
            let mut acc = 0.0_f64;
            for kr in 0..7 {
                let src_r = row as isize + kr as isize - 3;
                if src_r < 0 || src_r >= n_row as isize {
                    continue;
                }
                for kc in 0..7 {
                    let src_c = col as isize + kc as isize - 3;
                    if src_c < 0 || src_c >= n_col as isize {
                        continue;
                    }
                    acc += values[src_r as usize * n_col + src_c as usize] * kernel[kr * 7 + kc];
                }
            }
            out[row * n_col + col] = acc;
        }
    }
    out
}

pub(super) fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

pub(crate) fn clap_filter_patch_values(
    ph_values: &[Complex64],
    n_row: usize,
    n_col: usize,
    alpha: f64,
    beta: f64,
    low_pass: &[f64],
) -> Vec<Complex64> {
    let mut ph_fft = ph_values
        .iter()
        .map(|value| {
            if value.re.is_nan() || value.im.is_nan() {
                Complex64::new(0.0, 0.0)
            } else {
                *value
            }
        })
        .collect::<Vec<_>>();
    fft2_in_place(&mut ph_fft, n_row, n_col, false);

    let h = ph_fft.iter().map(|value| value.norm()).collect::<Vec<_>>();
    let h_shifted = roll_real(&h, n_row, n_col, (n_row / 2) as isize, (n_col / 2) as isize);
    let kernel = clap_filter_kernel_values();
    let h_conv = convolve_same_7x7(&h_shifted, n_row, n_col, &kernel);
    let mut h = roll_real(
        &h_conv,
        n_row,
        n_col,
        -((n_row / 2) as isize),
        -((n_col / 2) as isize),
    );

    let mut h_for_median = h.clone();
    let mean_h = median(&mut h_for_median);
    if mean_h != 0.0 {
        for value in &mut h {
            *value /= mean_h;
        }
    }
    for value in &mut h {
        *value = value.powf(alpha) - 1.0;
        if *value < 0.0 {
            *value = 0.0;
        }
    }

    for (idx, value) in ph_fft.iter_mut().enumerate() {
        *value *= h[idx] * beta + low_pass[idx];
    }
    fft2_in_place(&mut ph_fft, n_row, n_col, true);
    ph_fft
}
