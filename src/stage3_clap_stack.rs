use num_complex::Complex64;
use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::stage3_native::clap_filter_patch_values;

#[pyfunction]
pub fn stage3_clap_filt_patch_stack<'py>(
    py: Python<'py>,
    ph_stack: PyReadonlyArray3<Complex64>,
    alpha: f64,
    beta: f64,
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
    let shape = ph_view.shape();
    let n_row = shape[0];
    let n_col = shape[1];
    let n_ifg = shape[2];
    if n_row == 0 || n_col == 0 || n_ifg == 0 {
        return Err(PyValueError::new_err("ph_stack must be non-empty"));
    }
    if low_view.shape() != [n_row, n_col] {
        return Err(PyValueError::new_err(
            "low_pass shape must match ph_stack rows and columns",
        ));
    }
    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_stack must be C-contiguous"))?;
    let low_slice = low_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("low_pass must be C-contiguous"))?;

    let values = py.detach(move || {
        let mut out = vec![Complex64::new(0.0, 0.0); n_row * n_col * n_ifg];
        let mut plane = vec![Complex64::new(0.0, 0.0); n_row * n_col];
        for ifg in 0..n_ifg {
            for row in 0..n_row {
                for col in 0..n_col {
                    plane[row * n_col + col] = ph_slice[(row * n_col + col) * n_ifg + ifg];
                }
            }
            let filtered = clap_filter_patch_values(&plane, n_row, n_col, alpha, beta, low_slice);
            for row in 0..n_row {
                for col in 0..n_col {
                    out[(row * n_col + col) * n_ifg + ifg] = filtered[row * n_col + col];
                }
            }
        }
        out
    });
    Ok(Array3::from_shape_vec((n_row, n_col, n_ifg), values)
        .map_err(|err| PyValueError::new_err(format!("failed to build clap stack output: {err}")))?
        .into_pyarray(py))
}
