use num_complex::Complex64;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[path = "stage3_native_core.rs"]
mod stage3_native_core;
#[path = "stage3_native_grid.rs"]
mod stage3_native_grid;
#[path = "stage3_native_threshold.rs"]
mod stage3_native_threshold;
#[path = "stage3_native_wrap.rs"]
mod stage3_native_wrap;

pub(crate) use self::stage3_native_core::clap_filter_patch_values;
pub use self::stage3_native_grid::{stage3_clap_filt_grid, stage3_clap_filt_grid_stack};
pub use self::stage3_native_threshold::stage3_coh_threshold;
pub use self::stage3_native_wrap::{stage3_wrap_filt, stage3_wrap_filt_global};

#[pyfunction]
pub fn stage3_select_ifg_index<'py>(
    py: Python<'py>,
    n_ifg: i64,
    master_ix: i64,
    drop_ifg_index: PyReadonlyArray1<i64>,
    small_baseline: bool,
    _threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if n_ifg < 0 {
        return Err(PyValueError::new_err("n_ifg must be non-negative"));
    }
    if !small_baseline && (master_ix < 1 || master_ix > n_ifg) {
        return Err(PyValueError::new_err(
            "master_ix must be 1-based within n_ifg",
        ));
    }

    let drop_view = drop_ifg_index.as_array();
    let drop_slice = drop_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("drop_ifg_index must be contiguous"))?;

    let mut out = Vec::with_capacity(n_ifg as usize);
    'ifg_loop: for ifg in 1..=n_ifg {
        for &drop in drop_slice {
            if ifg == drop {
                continue 'ifg_loop;
            }
        }
        if !small_baseline && ifg == master_ix {
            continue;
        }
        let value = if !small_baseline && ifg > master_ix {
            ifg - 1
        } else {
            ifg
        };
        out.push(value as f64);
    }

    Ok(out.into_pyarray(py))
}

#[pyfunction]
pub fn stage3_clap_filt_patch<'py>(
    py: Python<'py>,
    ph: PyReadonlyArray2<Complex64>,
    alpha: f64,
    beta: f64,
    low_pass: PyReadonlyArray2<f64>,
    _threads: usize,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    let ph_view = ph.as_array();
    let low_view = low_pass.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err("ph must be a 2-D complex matrix"));
    }
    if low_view.shape() != ph_view.shape() {
        return Err(PyValueError::new_err("low_pass shape must match ph"));
    }
    let n_row = ph_view.shape()[0];
    let n_col = ph_view.shape()[1];
    if n_row == 0 || n_col == 0 {
        return Err(PyValueError::new_err("ph must be non-empty"));
    }
    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph must be C-contiguous"))?;
    let low_slice = low_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("low_pass must be C-contiguous"))?;

    let ph_fft = clap_filter_patch_values(ph_slice, n_row, n_col, alpha, beta, low_slice);

    Ok(Array2::from_shape_vec((n_row, n_col), ph_fft)
        .map_err(|err| {
            PyValueError::new_err(format!("failed to build stage3 clap patch output: {err}"))
        })?
        .into_pyarray(py))
}
