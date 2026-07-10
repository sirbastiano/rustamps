use num_complex::Complex64;
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::stage3_native_core::clap_filter_patch_values;

fn clap_weight(n_win: usize, row: usize, col: usize, row_shift: usize, col_shift: usize) -> f64 {
    if row < row_shift || col < col_shift {
        return 0.0;
    }
    let base_row = row - row_shift;
    let base_col = col - col_shift;
    if base_row >= n_win || base_col >= n_win {
        return 0.0;
    }
    base_row.min(n_win - 1 - base_row) as f64 + base_col.min(n_win - 1 - base_col) as f64 + 1e-6
}

fn clap_filter_grid_values(
    ph_slice: &[Complex64],
    n_i: usize,
    n_j: usize,
    alpha: f64,
    beta: f64,
    n_win: usize,
    n_pad: usize,
    low_slice: &[f64],
) -> Vec<Complex64> {
    let n_win_ex = n_win + n_pad;
    let mut out = vec![Complex64::new(0.0, 0.0); n_i * n_j];
    let n_inc = (n_win / 4).max(1);
    let n_win_i = ((n_i + n_inc - 1) / n_inc) as isize - 3;
    let n_win_j = ((n_j + n_inc - 1) / n_inc) as isize - 3;
    if n_win_i <= 0 || n_win_j <= 0 {
        return out;
    }

    let mut ph_bit = vec![Complex64::new(0.0, 0.0); n_win_ex * n_win_ex];
    for ix1 in 0..n_win_i as usize {
        let mut i1 = ix1 * n_inc;
        let mut i2 = i1 + n_win;
        let mut row_shift = 0usize;
        if i2 > n_i {
            row_shift = i2 - n_i;
            i2 = n_i;
            i1 = n_i - n_win;
        }
        for ix2 in 0..n_win_j as usize {
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
            let ph_filt =
                clap_filter_patch_values(&ph_bit, n_win_ex, n_win_ex, alpha, beta, low_slice);
            for row in 0..(i2 - i1) {
                for col in 0..(j2 - j1) {
                    let weight = clap_weight(n_win, row, col, row_shift, col_shift);
                    out[(i1 + row) * n_j + (j1 + col)] += ph_filt[row * n_win_ex + col] * weight;
                }
            }
        }
    }
    out
}

#[pyfunction]
pub fn stage3_clap_filt_grid<'py>(
    py: Python<'py>,
    ph: PyReadonlyArray2<Complex64>,
    alpha: f64,
    beta: f64,
    n_win: usize,
    n_pad: usize,
    low_pass: PyReadonlyArray2<f64>,
    _threads: usize,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    let ph_view = ph.as_array();
    let low_view = low_pass.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err("ph must be a 2-D complex grid"));
    }
    if n_win == 0 {
        return Err(PyValueError::new_err("n_win must be positive"));
    }
    if n_win % 2 != 0 {
        return Err(PyValueError::new_err(
            "n_win must be even for native clap grid filtering",
        ));
    }
    let n_win_ex = n_win + n_pad;
    if low_view.shape() != [n_win_ex, n_win_ex] {
        return Err(PyValueError::new_err(
            "low_pass shape must match n_win + n_pad",
        ));
    }
    let n_i = ph_view.shape()[0];
    let n_j = ph_view.shape()[1];
    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph must be C-contiguous"))?;
    let low_slice = low_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("low_pass must be C-contiguous"))?;

    let out = clap_filter_grid_values(ph_slice, n_i, n_j, alpha, beta, n_win, n_pad, low_slice);

    Ok(Array2::from_shape_vec((n_i, n_j), out)
        .map_err(|err| {
            PyValueError::new_err(format!("failed to build stage3 clap grid output: {err}"))
        })?
        .into_pyarray(py))
}

#[pyfunction]
pub fn stage3_clap_filt_grid_stack<'py>(
    py: Python<'py>,
    ph_stack: PyReadonlyArray3<Complex64>,
    alpha: f64,
    beta: f64,
    n_win: usize,
    n_pad: usize,
    low_pass: PyReadonlyArray2<f64>,
    _threads: usize,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    let ph_view = ph_stack.as_array();
    let low_view = low_pass.as_array();
    if ph_view.ndim() != 3 {
        return Err(PyValueError::new_err(
            "ph_stack must be a 3-D complex stack",
        ));
    }
    if n_win == 0 {
        return Err(PyValueError::new_err("n_win must be positive"));
    }
    if n_win % 2 != 0 {
        return Err(PyValueError::new_err(
            "n_win must be even for native clap grid stack filtering",
        ));
    }
    let n_win_ex = n_win + n_pad;
    if low_view.shape() != [n_win_ex, n_win_ex] {
        return Err(PyValueError::new_err(
            "low_pass shape must match n_win + n_pad",
        ));
    }
    let n_i = ph_view.shape()[0];
    let n_j = ph_view.shape()[1];
    let n_ifg = ph_view.shape()[2];
    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_stack must be C-contiguous"))?;
    let low_slice = low_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("low_pass must be C-contiguous"))?;

    let mut out = vec![Complex64::new(0.0, 0.0); n_i * n_j * n_ifg];
    let mut plane = vec![Complex64::new(0.0, 0.0); n_i * n_j];
    for i_ifg in 0..n_ifg {
        for row in 0..n_i {
            for col in 0..n_j {
                plane[row * n_j + col] = ph_slice[(row * n_j + col) * n_ifg + i_ifg];
            }
        }
        let filtered =
            clap_filter_grid_values(&plane, n_i, n_j, alpha, beta, n_win, n_pad, low_slice);
        for row in 0..n_i {
            for col in 0..n_j {
                out[(row * n_j + col) * n_ifg + i_ifg] = filtered[row * n_j + col];
            }
        }
    }

    Ok(Array3::from_shape_vec((n_i, n_j, n_ifg), out)
        .map_err(|err| {
            PyValueError::new_err(format!(
                "failed to build stage3 clap grid stack output: {err}"
            ))
        })?
        .into_pyarray(py))
}
