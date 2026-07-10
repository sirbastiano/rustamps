use num_complex::Complex64;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) fn wrap_phase(value: f64) -> f64 {
    value.sin().atan2(value.cos())
}

pub(crate) fn weighted_affine_fit_rows(
    time_diff: &[f64],
    y: &[f64],
    n_row: usize,
    n_col: usize,
    w: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut intercept = vec![0.0_f64; n_row];
    let mut slope = vec![0.0_f64; n_row];
    if n_row == 0 || n_col == 0 {
        return (intercept, slope);
    }

    let s0: f64 = w.iter().copied().sum();
    let s1: f64 = w
        .iter()
        .zip(time_diff.iter())
        .map(|(&wi, &ti)| wi * ti)
        .sum();
    let s2: f64 = w
        .iter()
        .zip(time_diff.iter())
        .map(|(&wi, &ti)| wi * ti * ti)
        .sum();
    let det = s0 * s2 - s1 * s1;
    if det == 0.0 {
        if s0 != 0.0 {
            for row_ix in 0..n_row {
                let mut base = 0.0_f64;
                for col_ix in 0..n_col {
                    base += y[row_ix * n_col + col_ix] * w[col_ix];
                }
                intercept[row_ix] = base / s0;
            }
        }
        return (intercept, slope);
    }

    for row_ix in 0..n_row {
        let mut wy0 = 0.0_f64;
        let mut wy1 = 0.0_f64;
        for col_ix in 0..n_col {
            let value = y[row_ix * n_col + col_ix];
            let weight = w[col_ix];
            wy0 += value * weight;
            wy1 += value * weight * time_diff[col_ix];
        }
        intercept[row_ix] = (wy0 * s2 - wy1 * s1) / det;
        slope[row_ix] = (wy1 * s0 - wy0 * s1) / det;
    }
    (intercept, slope)
}

pub(crate) fn weighted_slope_fit_rows_real(
    x: &[f64],
    y: &[f64],
    n_row: usize,
    n_col: usize,
    w: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_row];
    if n_row == 0 || n_col == 0 {
        return out;
    }

    let inf_idx: Vec<usize> = w
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| if value.is_infinite() { Some(idx) } else { None })
        .collect();
    if !inf_idx.is_empty() {
        let den: f64 = inf_idx.iter().map(|&idx| x[idx] * x[idx]).sum();
        if den == 0.0 {
            return out;
        }
        for row_ix in 0..n_row {
            let mut num = 0.0_f64;
            for &col_ix in &inf_idx {
                num += y[row_ix * n_col + col_ix] * x[col_ix];
            }
            out[row_ix] = num / den;
        }
        return out;
    }

    let pos_idx: Vec<usize> = w
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| {
            if value.is_finite() && value > 0.0 {
                Some(idx)
            } else {
                None
            }
        })
        .collect();
    if pos_idx.is_empty() {
        return out;
    }

    let den: f64 = pos_idx.iter().map(|&idx| w[idx] * x[idx] * x[idx]).sum();
    if den == 0.0 {
        return out;
    }
    for row_ix in 0..n_row {
        let mut num = 0.0_f64;
        for &col_ix in &pos_idx {
            num += y[row_ix * n_col + col_ix] * w[col_ix] * x[col_ix];
        }
        out[row_ix] = num / den;
    }
    out
}

pub(crate) fn weighted_slope_fit_rows_complex(
    x: &[f64],
    y: &[Complex64],
    n_row: usize,
    n_col: usize,
    w: &[f64],
) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n_row];
    if n_row == 0 || n_col == 0 {
        return out;
    }

    let inf_idx: Vec<usize> = w
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| if value.is_infinite() { Some(idx) } else { None })
        .collect();
    if !inf_idx.is_empty() {
        let den: f64 = inf_idx.iter().map(|&idx| x[idx] * x[idx]).sum();
        if den == 0.0 {
            return out;
        }
        for row_ix in 0..n_row {
            let mut num = Complex64::new(0.0, 0.0);
            for &col_ix in &inf_idx {
                num += y[row_ix * n_col + col_ix] * x[col_ix];
            }
            out[row_ix] = num / den;
        }
        return out;
    }

    let pos_idx: Vec<usize> = w
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| {
            if value.is_finite() && value > 0.0 {
                Some(idx)
            } else {
                None
            }
        })
        .collect();
    if pos_idx.is_empty() {
        return out;
    }

    let den: f64 = pos_idx.iter().map(|&idx| w[idx] * x[idx] * x[idx]).sum();
    if den == 0.0 {
        return out;
    }
    for row_ix in 0..n_row {
        let mut num = Complex64::new(0.0, 0.0);
        for &col_ix in &pos_idx {
            num += y[row_ix * n_col + col_ix] * (w[col_ix] * x[col_ix]);
        }
        out[row_ix] = num / den;
    }
    out
}

#[pyfunction]
pub fn weighted_affine_fit<'py>(
    py: Python<'py>,
    time_diff: PyReadonlyArray1<f64>,
    y: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let time_view = time_diff.as_array();
    let y_view = y.as_array();
    let weight_view = weights.as_array();
    let n_row = y_view.shape()[0];
    let n_col = y_view.shape()[1];
    if time_view.len() != n_col || weight_view.len() != n_col {
        return Err(PyValueError::new_err(
            "weighted_affine_fit expects time/weights length to match target columns",
        ));
    }
    let time_slice = time_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("time_diff must be contiguous"))?;
    let y_slice = y_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("y must be C-contiguous"))?;
    let weight_slice = weight_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("weights must be contiguous"))?;

    let (intercept, slope) =
        weighted_affine_fit_rows(time_slice, y_slice, n_row, n_col, weight_slice);
    let dict = PyDict::new(py);
    dict.set_item("intercept", Array1::from_vec(intercept).into_pyarray(py))?;
    dict.set_item("slope", Array1::from_vec(slope).into_pyarray(py))?;
    Ok(dict)
}

#[pyfunction]
pub fn weighted_slope_fit_real<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x_view = x.as_array();
    let y_view = y.as_array();
    let weight_view = weights.as_array();
    let n_row = y_view.shape()[0];
    let n_col = y_view.shape()[1];
    if x_view.len() != n_col || weight_view.len() != n_col {
        return Err(PyValueError::new_err(
            "weighted_slope_fit_real expects x/weights length to match target columns",
        ));
    }
    let x_slice = x_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("x must be contiguous"))?;
    let y_slice = y_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("y must be C-contiguous"))?;
    let weight_slice = weight_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("weights must be contiguous"))?;
    let out = weighted_slope_fit_rows_real(x_slice, y_slice, n_row, n_col, weight_slice);
    Ok(Array1::from_vec(out).into_pyarray(py))
}

#[pyfunction]
pub fn weighted_slope_fit_complex<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray2<Complex64>,
    weights: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let x_view = x.as_array();
    let y_view = y.as_array();
    let weight_view = weights.as_array();
    let n_row = y_view.shape()[0];
    let n_col = y_view.shape()[1];
    if x_view.len() != n_col || weight_view.len() != n_col {
        return Err(PyValueError::new_err(
            "weighted_slope_fit_complex expects x/weights length to match target columns",
        ));
    }
    let x_slice = x_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("x must be contiguous"))?;
    let y_slice = y_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("y must be C-contiguous"))?;
    let weight_slice = weight_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("weights must be contiguous"))?;
    let out = weighted_slope_fit_rows_complex(x_slice, y_slice, n_row, n_col, weight_slice);
    Ok(Array1::from_vec(out).into_pyarray(py))
}
