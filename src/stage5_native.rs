use num_complex::Complex32;
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::f32::consts::PI;

#[path = "stage5_native_filters.rs"]
mod stage5_native_filters;

pub use self::stage5_native_filters::{stage5_duplicate_keep, stage5_patch_keep_mask};

const TWO_PI: f32 = 2.0 * PI;

fn wrap_phase(value: f32) -> f32 {
    (value + PI).rem_euclid(TWO_PI) - PI
}

#[pyfunction(signature = (ph2, ph_patch, bperp, k_ps, c_ps, threads = 0))]
pub fn stage5_ifg_std<'py>(
    py: Python<'py>,
    ph2: PyReadonlyArray2<Complex32>,
    ph_patch: PyReadonlyArray2<Complex32>,
    bperp: PyReadonlyArray2<f64>,
    k_ps: PyReadonlyArray1<f64>,
    c_ps: PyReadonlyArray1<f64>,
    threads: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
    let _ = threads;
    let ph2_view = ph2.as_array();
    let ph_patch_view = ph_patch.as_array();
    let bperp_view = bperp.as_array();
    if ph2_view.ndim() != 2 {
        return Err(PyValueError::new_err(
            "stage5_ifg_std expects a 2-D ph2 matrix",
        ));
    }
    let n_ps = ph2_view.shape()[0];
    let n_ifg = ph2_view.shape()[1];
    if n_ps == 0 || n_ifg == 0 {
        return Err(PyValueError::new_err(
            "stage5_ifg_std expects a non-empty ph2 matrix",
        ));
    }
    if ph_patch_view.shape() != ph2_view.shape() || bperp_view.shape() != ph2_view.shape() {
        return Err(PyValueError::new_err(
            "stage5_ifg_std expects ph2, ph_patch, and bperp with matching shapes",
        ));
    }
    let k_view = k_ps.as_array();
    let c_view = c_ps.as_array();
    if k_view.len() != n_ps || c_view.len() != n_ps {
        return Err(PyValueError::new_err(
            "stage5_ifg_std expects k_ps and c_ps length to match ph2 rows",
        ));
    }

    let ph2_slice = ph2_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph2 must be C-contiguous"))?;
    let ph_patch_slice = ph_patch_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_patch must be C-contiguous"))?;
    let bperp_slice = bperp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp must be C-contiguous"))?;
    let k_slice = k_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("k_ps must be contiguous"))?;
    let c_slice = c_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("c_ps must be contiguous"))?;

    let out = py.detach(move || {
        let mut sums = vec![0.0_f64; n_ifg];
        for row in 0..n_ps {
            let k = k_slice[row] as f32;
            let c = c_slice[row] as f32;
            let row_offset = row * n_ifg;
            for col in 0..n_ifg {
                let idx = row_offset + col;
                let phase = ph2_slice[idx].im.atan2(ph2_slice[idx].re);
                let patch_phase = ph_patch_slice[idx].im.atan2(ph_patch_slice[idx].re);
                let correction = k * bperp_slice[idx] as f32 + c;
                let diff = wrap_phase(phase - patch_phase - correction);
                sums[col] += f64::from(diff) * f64::from(diff);
            }
        }
        let scale = 180.0_f64 / std::f64::consts::PI;
        sums.into_iter()
            .map(|sum| ((sum / n_ps as f64).sqrt() * scale) as f32)
            .collect::<Vec<f32>>()
    });
    Ok(Array1::from_vec(out).into_pyarray(py))
}

#[pyfunction]
pub fn stage5_format_merged_rc2<'py>(
    py: Python<'py>,
    rc2_all: PyReadonlyArray2<Complex32>,
    _threads: usize,
) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
    let rc_view = rc2_all.as_array();
    if rc_view.ndim() != 2 {
        return Err(PyValueError::new_err(
            "stage5_format_merged_rc2 expects a 2-D matrix",
        ));
    }
    let n_row = rc_view.shape()[0];
    let n_col = rc_view.shape()[1];
    let rc_slice = rc_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("rc2_all must be C-contiguous"))?;

    let out = py.detach(move || {
        let mut values = vec![Complex32::new(0.0, 0.0); n_col * n_row];
        for row in 0..n_row {
            for col in 0..n_col {
                let value = rc_slice[row * n_col + col];
                let norm = value.norm();
                let out_value = if norm != 0.0 { value / norm } else { value };
                values[col * n_row + row] = out_value;
            }
        }
        values
    });
    Ok(Array2::from_shape_vec((n_col, n_row), out)
        .map_err(|err| PyValueError::new_err(format!("failed to build stage5 rc2 payload: {err}")))?
        .into_pyarray(py))
}

#[pyfunction]
pub fn stage5_rc2_correction<'py>(
    py: Python<'py>,
    ph2: PyReadonlyArray2<Complex32>,
    ph_patch: PyReadonlyArray2<Complex32>,
    bperp: PyReadonlyArray2<f64>,
    k_ps: PyReadonlyArray1<f64>,
    c_ps: PyReadonlyArray1<f64>,
    small_baseline: bool,
    master_ix: i64,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = threads;
    let ph_view = ph2.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err("ph2 must be a 2-D complex matrix"));
    }
    let n_ps = ph_view.shape()[0];
    let n_ifg = ph_view.shape()[1];
    let b_view = bperp.as_array();
    let expected_b_cols = if small_baseline {
        n_ifg
    } else {
        n_ifg.saturating_sub(1)
    };
    if b_view.ndim() != 2 || b_view.shape()[0] != n_ps || b_view.shape()[1] != expected_b_cols {
        return Err(PyValueError::new_err(
            "bperp shape is incompatible with ph2 and small_baseline mode",
        ));
    }
    let patch_view = ph_patch.as_array();
    if !small_baseline
        && (patch_view.ndim() != 2
            || patch_view.shape()[0] != n_ps
            || patch_view.shape()[1] != n_ifg.saturating_sub(1))
    {
        return Err(PyValueError::new_err(
            "ph_patch shape is incompatible with single-master rc2 correction",
        ));
    }
    let k_view = k_ps.as_array();
    let c_view = c_ps.as_array();
    if k_view.len() != n_ps || c_view.len() != n_ps {
        return Err(PyValueError::new_err(
            "k_ps and c_ps lengths must match ph2 rows",
        ));
    }
    if !small_baseline && (master_ix < 1 || master_ix as usize > n_ifg) {
        return Err(PyValueError::new_err(
            "master_ix must be a valid 1-based ph2 column",
        ));
    }
    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph2 must be C-contiguous"))?;
    let patch_slice = patch_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_patch must be C-contiguous"))?;
    let b_slice = b_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp must be C-contiguous"))?;
    let k_slice = k_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("k_ps must be contiguous"))?;
    let c_slice = c_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("c_ps must be contiguous"))?;

    let mut ph_rc = Vec::with_capacity(n_ps * n_ifg);
    for row in 0..n_ps {
        for col in 0..n_ifg {
            let b = if small_baseline {
                b_slice[row * n_ifg + col]
            } else {
                let master_col = (master_ix - 1) as usize;
                if col < master_col {
                    b_slice[row * (n_ifg - 1) + col]
                } else if col == master_col {
                    0.0
                } else {
                    b_slice[row * (n_ifg - 1) + col - 1]
                }
            };
            let phase = if small_baseline {
                -(k_slice[row] * b)
            } else {
                -(k_slice[row] * b + c_slice[row])
            };
            let ramp = Complex32::new(phase.cos() as f32, phase.sin() as f32);
            ph_rc.push(ph_slice[row * n_ifg + col] * ramp);
        }
    }

    let dict = PyDict::new(py);
    dict.set_item(
        "ph_rc",
        Array2::from_shape_vec((n_ps, n_ifg), ph_rc)
            .map_err(|err| {
                PyValueError::new_err(format!("failed to build stage5 ph_rc output: {err}"))
            })?
            .into_pyarray(py),
    )?;

    if !small_baseline {
        let master_col = (master_ix - 1) as usize;
        let mut ph_reref = Vec::with_capacity(n_ps * n_ifg);
        for row in 0..n_ps {
            for col in 0..n_ifg {
                if col < master_col {
                    ph_reref.push(patch_slice[row * (n_ifg - 1) + col]);
                } else if col == master_col {
                    ph_reref.push(Complex32::new(1.0, 0.0));
                } else {
                    ph_reref.push(patch_slice[row * (n_ifg - 1) + col - 1]);
                }
            }
        }
        dict.set_item(
            "ph_reref",
            Array2::from_shape_vec((n_ps, n_ifg), ph_reref)
                .map_err(|err| {
                    PyValueError::new_err(format!("failed to build stage5 ph_reref output: {err}"))
                })?
                .into_pyarray(py),
        )?;
    }

    Ok(dict)
}
