use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{invert_small_matrix_with_jitter, solve_linear_system};

#[pyfunction]
pub fn stage8_weighted_lstsq_diagonal<'py>(
    py: Python<'py>,
    design: PyReadonlyArray2<f64>,
    values: PyReadonlyArray2<f64>,
    variances: PyReadonlyArray1<f64>,
    _threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let design_view = design.as_array();
    let values_view = values.as_array();
    let variance_view = variances.as_array();
    if design_view.ndim() != 2 || values_view.ndim() != 2 {
        return Err(PyValueError::new_err(
            "stage8_weighted_lstsq_diagonal expects 2-D design and values",
        ));
    }
    let n_obs = design_view.shape()[0];
    let n_coeff = design_view.shape()[1];
    if values_view.shape()[0] != n_obs || variance_view.len() != n_obs {
        return Err(PyValueError::new_err(
            "stage8_weighted_lstsq_diagonal expects values/covariance aligned with design rows",
        ));
    }
    let n_rhs = values_view.shape()[1];
    let design_slice = design_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("design must be C-contiguous"))?;
    let values_slice = values_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("values must be C-contiguous"))?;
    let variance_slice = variance_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("variances must be contiguous"))?;

    let mut gram = vec![0.0_f64; n_coeff * n_coeff];
    let mut rhs = vec![0.0_f64; n_coeff * n_rhs];
    for obs in 0..n_obs {
        let variance = if variance_slice[obs] == 0.0 {
            1.0
        } else {
            variance_slice[obs]
        };
        let weight = if variance.is_finite() && variance > 0.0 {
            1.0 / variance
        } else {
            1.0
        };
        for left in 0..n_coeff {
            let left_val = design_slice[obs * n_coeff + left];
            for right in 0..n_coeff {
                gram[left * n_coeff + right] +=
                    weight * left_val * design_slice[obs * n_coeff + right];
            }
            for col in 0..n_rhs {
                rhs[left * n_rhs + col] += weight * left_val * values_slice[obs * n_rhs + col];
            }
        }
    }

    let mut out = vec![0.0_f64; n_coeff * n_rhs];
    for col in 0..n_rhs {
        let col_rhs: Vec<f64> = (0..n_coeff).map(|row| rhs[row * n_rhs + col]).collect();
        let solution = solve_linear_system(gram.clone(), col_rhs, n_coeff).ok_or_else(|| {
            PyValueError::new_err("stage8 weighted least-squares system is singular")
        })?;
        for row in 0..n_coeff {
            out[row * n_rhs + col] = solution[row];
        }
    }

    Ok(Array2::from_shape_vec((n_coeff, n_rhs), out)
        .map_err(|err| {
            PyValueError::new_err(format!(
                "failed to build stage8 weighted-lstsq output: {err}"
            ))
        })?
        .into_pyarray(py))
}

#[pyfunction]
pub fn stage8_weighted_lstsq_full<'py>(
    py: Python<'py>,
    design: PyReadonlyArray2<f64>,
    values: PyReadonlyArray2<f64>,
    covariance: PyReadonlyArray2<f64>,
    _threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let design_view = design.as_array();
    let values_view = values.as_array();
    let covariance_view = covariance.as_array();
    if design_view.ndim() != 2 || values_view.ndim() != 2 || covariance_view.ndim() != 2 {
        return Err(PyValueError::new_err(
            "stage8_weighted_lstsq_full expects 2-D design, values, and covariance",
        ));
    }
    let n_obs = design_view.shape()[0];
    let n_coeff = design_view.shape()[1];
    if values_view.shape()[0] != n_obs
        || covariance_view.shape()[0] != n_obs
        || covariance_view.shape()[1] != n_obs
    {
        return Err(PyValueError::new_err(
            "stage8_weighted_lstsq_full expects values/covariance aligned with design rows",
        ));
    }
    let n_rhs = values_view.shape()[1];
    let design_slice = design_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("design must be C-contiguous"))?;
    let values_slice = values_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("values must be C-contiguous"))?;
    let covariance_slice = covariance_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("covariance must be C-contiguous"))?;

    let covariance_inv = invert_small_matrix_with_jitter(covariance_slice, n_obs);
    let mut gram = vec![0.0_f64; n_coeff * n_coeff];
    for left in 0..n_coeff {
        for right in 0..n_coeff {
            let mut value = 0.0_f64;
            for row in 0..n_obs {
                let left_value = design_slice[row * n_coeff + left];
                for col in 0..n_obs {
                    value += left_value
                        * covariance_inv[row * n_obs + col]
                        * design_slice[col * n_coeff + right];
                }
            }
            gram[left * n_coeff + right] = value;
        }
    }

    let mut out = vec![0.0_f64; n_coeff * n_rhs];
    for rhs_col in 0..n_rhs {
        let mut rhs = vec![0.0_f64; n_coeff];
        for coeff_ix in 0..n_coeff {
            let mut value = 0.0_f64;
            for row in 0..n_obs {
                let design_value = design_slice[row * n_coeff + coeff_ix];
                for col in 0..n_obs {
                    value += design_value
                        * covariance_inv[row * n_obs + col]
                        * values_slice[col * n_rhs + rhs_col];
                }
            }
            rhs[coeff_ix] = value;
        }
        let solution = solve_linear_system(gram.clone(), rhs, n_coeff).ok_or_else(|| {
            PyValueError::new_err(
                "stage8 weighted full-covariance least-squares system is singular",
            )
        })?;
        for row in 0..n_coeff {
            out[row * n_rhs + rhs_col] = solution[row];
        }
    }

    Ok(Array2::from_shape_vec((n_coeff, n_rhs), out)
        .map_err(|err| {
            PyValueError::new_err(format!(
                "failed to build stage8 weighted-lstsq full-covariance output: {err}"
            ))
        })?
        .into_pyarray(py))
}
