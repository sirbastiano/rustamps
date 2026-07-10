use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn stage7_center_to_reference<'py>(
    py: Python<'py>,
    ph: PyReadonlyArray2<f64>,
    ref_ix: PyReadonlyArray1<i64>,
    _threads: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let ph_view = ph.as_array();
    let ref_view = ref_ix.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err(
            "stage7_center_to_reference expects a 2-D phase matrix",
        ));
    }
    let n_ps = ph_view.shape()[0];
    let n_ifg = ph_view.shape()[1];
    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph must be C-contiguous"))?;
    let ref_slice = ref_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ref_ix must be contiguous"))?;
    if ref_slice.is_empty() {
        return Ok(Array2::from_shape_vec((n_ps, n_ifg), ph_slice.to_vec())
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "failed to build stage7_center_to_reference output: {err}"
                ))
            })?
            .into_pyarray(py));
    }

    let mut refs = Vec::with_capacity(ref_slice.len());
    for &ref_value in ref_slice {
        let idx = if ref_value < 0 {
            n_ps as i64 + ref_value
        } else {
            ref_value
        };
        if idx < 0 || idx >= n_ps as i64 {
            return Err(PyValueError::new_err(
                "stage7_center_to_reference ref_ix entries must be within phase rows",
            ));
        }
        refs.push(idx as usize);
    }

    let mut means = vec![f64::NAN; n_ifg];
    for col in 0..n_ifg {
        let mut sum = 0.0_f64;
        let mut count = 0_usize;
        for &row in &refs {
            let value = ph_slice[row * n_ifg + col];
            if !value.is_nan() {
                sum += value;
                count += 1;
            }
        }
        if count > 0 {
            means[col] = sum / count as f64;
        }
    }

    let mut out = vec![0.0_f64; n_ps * n_ifg];
    for row in 0..n_ps {
        for col in 0..n_ifg {
            out[row * n_ifg + col] = ph_slice[row * n_ifg + col] - means[col];
        }
    }

    Ok(Array2::from_shape_vec((n_ps, n_ifg), out)
        .map_err(|err| {
            PyValueError::new_err(format!(
                "failed to build stage7_center_to_reference output: {err}"
            ))
        })?
        .into_pyarray(py))
}

#[pyfunction]
pub fn stage7_scla_smooth<'py>(
    py: Python<'py>,
    k_ps_uw: PyReadonlyArray1<f64>,
    c_ps_uw: PyReadonlyArray1<f64>,
    edges: PyReadonlyArray2<i64>,
    _threads: usize,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let k_view = k_ps_uw.as_array();
    let c_view = c_ps_uw.as_array();
    let edge_view = edges.as_array();
    let k_slice = k_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("k_ps_uw must be contiguous"))?;
    let c_slice = c_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("c_ps_uw must be contiguous"))?;
    let edge_slice = edge_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("edges must be C-contiguous"))?;
    if k_slice.len() != c_slice.len() {
        return Err(PyValueError::new_err(
            "k_ps_uw and c_ps_uw must have matching lengths",
        ));
    }
    if edge_view.ndim() != 2 || edge_view.shape()[1] != 2 {
        return Err(PyValueError::new_err(
            "stage7_scla_smooth edges must have shape (n_edge, 2)",
        ));
    }

    let n_ps = k_slice.len();
    let mut k_min = vec![f64::INFINITY; n_ps];
    let mut k_max = vec![f64::NEG_INFINITY; n_ps];
    let mut c_min = vec![f64::INFINITY; n_ps];
    let mut c_max = vec![f64::NEG_INFINITY; n_ps];

    for edge in edge_slice.chunks_exact(2) {
        let a_raw = edge[0];
        let b_raw = edge[1];
        if a_raw < 0 || b_raw < 0 {
            continue;
        }
        let a = a_raw as usize;
        let b = b_raw as usize;
        if a >= n_ps || b >= n_ps || a == b {
            continue;
        }
        k_min[a] = k_min[a].min(k_slice[b]);
        k_min[b] = k_min[b].min(k_slice[a]);
        k_max[a] = k_max[a].max(k_slice[b]);
        k_max[b] = k_max[b].max(k_slice[a]);
        c_min[a] = c_min[a].min(c_slice[b]);
        c_min[b] = c_min[b].min(c_slice[a]);
        c_max[a] = c_max[a].max(c_slice[b]);
        c_max[b] = c_max[b].max(c_slice[a]);
    }

    let mut k_out = Vec::with_capacity(n_ps);
    let mut c_out = Vec::with_capacity(n_ps);
    for idx in 0..n_ps {
        let mut k_val = k_slice[idx];
        if k_max[idx].is_finite() && k_val > k_max[idx] {
            k_val = k_max[idx];
        }
        if k_min[idx].is_finite() && k_val < k_min[idx] {
            k_val = k_min[idx];
        }
        let mut c_val = c_slice[idx];
        if c_max[idx].is_finite() && c_val > c_max[idx] {
            c_val = c_max[idx];
        }
        if c_min[idx].is_finite() && c_val < c_min[idx] {
            c_val = c_min[idx];
        }
        k_out.push(k_val as f32);
        c_out.push(c_val as f32);
    }

    Ok((k_out.into_pyarray(py), c_out.into_pyarray(py)))
}

#[pyfunction]
pub fn stage7_mean_velocity_fit<'py>(
    py: Python<'py>,
    ph_mean_v: PyReadonlyArray2<f64>,
    day: PyReadonlyArray1<f64>,
    master_ix: usize,
    ifg_std: PyReadonlyArray1<f64>,
    _threads: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let ph_view = ph_mean_v.as_array();
    let day_view = day.as_array();
    let std_view = ifg_std.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err(
            "stage7_mean_velocity_fit expects a 2-D phase matrix",
        ));
    }
    let n_ps = ph_view.shape()[0];
    let n_ifg = ph_view.shape()[1];
    if master_ix == 0 || master_ix > n_ifg {
        return Err(PyValueError::new_err(
            "stage7_mean_velocity_fit master_ix must be 1-based within the phase width",
        ));
    }
    if day_view.len() != n_ifg || std_view.len() != n_ifg {
        return Err(PyValueError::new_err(
            "stage7_mean_velocity_fit day/ifg_std length must match phase width",
        ));
    }
    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_mean_v must be C-contiguous"))?;
    let day_slice = day_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("day must be contiguous"))?;
    let std_slice = std_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ifg_std must be contiguous"))?;

    let master_zero = day_slice[master_ix - 1];
    let time_diff: Vec<f64> = day_slice.iter().map(|value| *value - master_zero).collect();
    let weights: Vec<f64> = std_slice
        .iter()
        .map(|std| {
            if *std > 0.0 {
                let rad = *std * std::f64::consts::PI / 180.0;
                1.0 / (rad * rad)
            } else {
                0.0
            }
        })
        .collect();
    let s0: f64 = weights.iter().sum();
    let s1: f64 = weights.iter().zip(&time_diff).map(|(w, t)| *w * *t).sum();
    let s2: f64 = weights
        .iter()
        .zip(&time_diff)
        .map(|(w, t)| *w * *t * *t)
        .sum();
    let det = s0 * s2 - s1 * s1;

    let mut out = vec![0.0_f32; 2 * n_ps];
    for ps_ix in 0..n_ps {
        let row = &ph_slice[(ps_ix * n_ifg)..((ps_ix + 1) * n_ifg)];
        let wy0: f64 = row.iter().zip(&weights).map(|(y, w)| *y * *w).sum();
        let wy1: f64 = row
            .iter()
            .zip(weights.iter().zip(&time_diff))
            .map(|(y, (w, t))| *y * *w * *t)
            .sum();
        let (intercept, slope) = if det == 0.0 {
            let intercept = if s0 != 0.0 { wy0 / s0 } else { 0.0 };
            (intercept, 0.0)
        } else {
            ((wy0 * s2 - wy1 * s1) / det, (wy1 * s0 - wy0 * s1) / det)
        };
        out[ps_ix] = intercept as f32;
        out[n_ps + ps_ix] = slope as f32;
    }

    let array = Array2::from_shape_vec((2, n_ps), out).map_err(|_| {
        PyValueError::new_err("stage7_mean_velocity_fit output shape construction failed")
    })?;
    Ok(array.into_pyarray(py))
}
