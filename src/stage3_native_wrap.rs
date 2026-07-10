use num_complex::Complex64;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::stage3_native_core::{
    clap_filter_kernel_values, convolve_same_7x7, fft2_in_place, gaussian_window, median, roll_real,
};

fn wrap_weight(n_win: usize, row: usize, col: usize, row_shift: usize, col_shift: usize) -> f64 {
    if row < row_shift || col < col_shift {
        return 0.0;
    }
    let base_row = row - row_shift;
    let base_col = col - col_shift;
    if base_row >= n_win || base_col >= n_win {
        return 0.0;
    }
    base_row.min(n_win - 1 - base_row) as f64 + base_col.min(n_win - 1 - base_col) as f64 + 2.0
}

fn low_pass_filter_values(n: usize) -> Vec<f64> {
    let g = gaussian_window(n, 16.0);
    let mut outer = vec![0.0_f64; n * n];
    for row in 0..n {
        for col in 0..n {
            outer[row * n + col] = g[row] * g[col];
        }
    }
    roll_real(&outer, n, n, -((n / 2) as isize), -((n / 2) as isize))
}

fn restore_input_magnitude(values: &mut [Complex64], input: &[Complex64]) {
    for (idx, value) in values.iter_mut().enumerate() {
        let in_value = input[idx];
        let magnitude = if in_value.re.is_nan() || in_value.im.is_nan() {
            0.0
        } else {
            in_value.norm()
        };
        let angle = value.im.atan2(value.re);
        *value = Complex64::new(magnitude * angle.cos(), magnitude * angle.sin());
    }
}

fn wrap_filter_values(
    ph_slice: &[Complex64],
    n_i: usize,
    n_j: usize,
    n_win: usize,
    alpha: f64,
    n_pad: usize,
    want_low: bool,
    global_counts: bool,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let n_win_ex = n_win + n_pad;
    let mut out = vec![Complex64::new(0.0, 0.0); n_i * n_j];
    let mut out_low = vec![Complex64::new(0.0, 0.0); n_i * n_j];
    let n_inc = (n_win / 2).max(1);
    let mut n_win_blocks_i = ((n_i + n_inc - 1) / n_inc) as isize - 1;
    let mut n_win_blocks_j = ((n_j + n_inc - 1) / n_inc) as isize - 1;
    if global_counts {
        n_win_blocks_i = n_win_blocks_i.max(1);
        n_win_blocks_j = n_win_blocks_j.max(1);
    }
    if n_win_blocks_i <= 0 || n_win_blocks_j <= 0 {
        restore_input_magnitude(&mut out, ph_slice);
        if want_low {
            restore_input_magnitude(&mut out_low, ph_slice);
        }
        return (out, out_low);
    }

    let kernel = clap_filter_kernel_values();
    let low_filter = if want_low {
        low_pass_filter_values(n_win_ex)
    } else {
        Vec::new()
    };
    let mut ph_bit = vec![Complex64::new(0.0, 0.0); n_win_ex * n_win_ex];
    for ix1 in 0..n_win_blocks_i as usize {
        let mut i1 = ix1 * n_inc;
        let mut i2 = i1 + n_win;
        let mut row_shift = 0usize;
        if i2 > n_i {
            row_shift = i2 - n_i;
            i2 = n_i;
            i1 = n_i - n_win;
        }
        for ix2 in 0..n_win_blocks_j as usize {
            let mut j1 = ix2 * n_inc;
            let mut j2 = j1 + n_win;
            let mut col_shift = 0usize;
            if j2 > n_j {
                col_shift = j2 - n_j;
                j2 = n_j;
                j1 = n_j - n_win;
            }

            ph_bit.fill(Complex64::new(0.0, 0.0));
            for row in 0..n_win {
                for col in 0..n_win {
                    let value = ph_slice[(i1 + row) * n_j + (j1 + col)];
                    ph_bit[row * n_win_ex + col] = if value.re.is_nan() || value.im.is_nan() {
                        Complex64::new(0.0, 0.0)
                    } else {
                        value
                    };
                }
            }

            let mut ph_fft = ph_bit.clone();
            fft2_in_place(&mut ph_fft, n_win_ex, n_win_ex, false);
            let h = ph_fft.iter().map(|value| value.norm()).collect::<Vec<_>>();
            let h_shifted = roll_real(
                &h,
                n_win_ex,
                n_win_ex,
                (n_win_ex / 2) as isize,
                (n_win_ex / 2) as isize,
            );
            let h_conv = convolve_same_7x7(&h_shifted, n_win_ex, n_win_ex, &kernel);
            let mut h = roll_real(
                &h_conv,
                n_win_ex,
                n_win_ex,
                -((n_win_ex / 2) as isize),
                -((n_win_ex / 2) as isize),
            );
            let mut h_for_median = h.clone();
            let mean_h = median(&mut h_for_median);
            if mean_h != 0.0 {
                for value in &mut h {
                    *value /= mean_h;
                }
            }
            for value in &mut h {
                *value = value.powf(alpha);
            }

            let mut filtered = ph_fft.clone();
            for (idx, value) in filtered.iter_mut().enumerate() {
                *value *= h[idx];
            }
            fft2_in_place(&mut filtered, n_win_ex, n_win_ex, true);
            for row in 0..(i2 - i1) {
                for col in 0..(j2 - j1) {
                    let weight = wrap_weight(n_win, row, col, row_shift, col_shift);
                    out[(i1 + row) * n_j + (j1 + col)] += filtered[row * n_win_ex + col] * weight;
                }
            }

            if want_low {
                let mut filtered_low = ph_fft.clone();
                for (idx, value) in filtered_low.iter_mut().enumerate() {
                    *value *= low_filter[idx];
                }
                fft2_in_place(&mut filtered_low, n_win_ex, n_win_ex, true);
                for row in 0..(i2 - i1) {
                    for col in 0..(j2 - j1) {
                        let weight = wrap_weight(n_win, row, col, row_shift, col_shift);
                        out_low[(i1 + row) * n_j + (j1 + col)] +=
                            filtered_low[row * n_win_ex + col] * weight;
                    }
                }
            }
        }
    }

    restore_input_magnitude(&mut out, ph_slice);
    if want_low {
        restore_input_magnitude(&mut out_low, ph_slice);
    }
    (out, out_low)
}

#[pyfunction]
pub fn stage3_wrap_filt<'py>(
    py: Python<'py>,
    ph: PyReadonlyArray2<Complex64>,
    n_win: usize,
    alpha: f64,
    n_pad: usize,
    want_low: bool,
    _threads: usize,
) -> PyResult<(
    Bound<'py, PyArray2<Complex64>>,
    Bound<'py, PyArray2<Complex64>>,
)> {
    let ph_view = ph.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err("ph must be a 2-D complex grid"));
    }
    if n_win <= 1 {
        return Err(PyValueError::new_err("n_win must be greater than 1"));
    }
    if n_win % 2 != 0 {
        return Err(PyValueError::new_err(
            "n_win must be even for native wrap filtering",
        ));
    }
    let n_i = ph_view.shape()[0];
    let n_j = ph_view.shape()[1];
    if n_i < n_win || n_j < n_win {
        return Err(PyValueError::new_err(
            "ph shape must be at least n_win by n_win for native wrap filtering",
        ));
    }
    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph must be C-contiguous"))?;
    let (out, out_low) =
        wrap_filter_values(ph_slice, n_i, n_j, n_win, alpha, n_pad, want_low, false);

    Ok((
        Array2::from_shape_vec((n_i, n_j), out)
            .map_err(|err| {
                PyValueError::new_err(format!("failed to build stage3 wrap filter output: {err}"))
            })?
            .into_pyarray(py),
        Array2::from_shape_vec((n_i, n_j), out_low)
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "failed to build stage3 wrap low-pass output: {err}"
                ))
            })?
            .into_pyarray(py),
    ))
}

#[pyfunction]
pub fn stage3_wrap_filt_global<'py>(
    py: Python<'py>,
    ph: PyReadonlyArray2<Complex64>,
    n_win: usize,
    alpha: f64,
    n_pad: usize,
    want_low: bool,
    _threads: usize,
) -> PyResult<(
    Bound<'py, PyArray2<Complex64>>,
    Bound<'py, PyArray2<Complex64>>,
)> {
    let ph_view = ph.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err("ph must be a 2-D complex grid"));
    }
    if n_win == 0 {
        return Err(PyValueError::new_err("n_win must be positive"));
    }
    if n_win % 2 != 0 {
        return Err(PyValueError::new_err(
            "n_win must be even for native global wrap filtering",
        ));
    }
    let n_i = ph_view.shape()[0];
    let n_j = ph_view.shape()[1];
    if n_i < n_win || n_j < n_win {
        return Err(PyValueError::new_err(
            "ph shape must be at least n_win by n_win for native global wrap filtering",
        ));
    }
    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph must be C-contiguous"))?;
    let (out, out_low) =
        wrap_filter_values(ph_slice, n_i, n_j, n_win, alpha, n_pad, want_low, true);

    Ok((
        Array2::from_shape_vec((n_i, n_j), out)
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "failed to build stage3 global wrap filter output: {err}"
                ))
            })?
            .into_pyarray(py),
        Array2::from_shape_vec((n_i, n_j), out_low)
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "failed to build stage3 global wrap low-pass output: {err}"
                ))
            })?
            .into_pyarray(py),
    ))
}
