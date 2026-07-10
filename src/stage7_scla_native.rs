use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{parse_indices, stage7_outputs};

#[pyfunction(signature = (ph_proc, ph_mean_v, bperp_mat, unwrap_ix, solve_ix, day, master_ix, ifg_std, threads = 0))]
pub fn stage7_scla_parity<'py>(
    py: Python<'py>,
    ph_proc: PyReadonlyArray2<f64>,
    ph_mean_v: PyReadonlyArray2<f64>,
    bperp_mat: PyReadonlyArray2<f64>,
    unwrap_ix: PyReadonlyArray1<i64>,
    solve_ix: PyReadonlyArray1<i64>,
    day: PyReadonlyArray1<f64>,
    master_ix: usize,
    ifg_std: PyReadonlyArray1<f64>,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let ph_proc_view = ph_proc.as_array();
    let ph_mean_v_view = ph_mean_v.as_array();
    let bperp_view = bperp_mat.as_array();
    if ph_proc_view.ndim() != 2 || ph_mean_v_view.ndim() != 2 || bperp_view.ndim() != 2 {
        return Err(PyValueError::new_err(
            "stage7_scla_parity expects 2-D ph_proc, ph_mean_v, and bperp_mat",
        ));
    }
    if ph_proc_view.shape() != ph_mean_v_view.shape() || ph_proc_view.shape() != bperp_view.shape()
    {
        return Err(PyValueError::new_err(
            "stage7_scla_parity expects ph_proc, ph_mean_v, and bperp_mat with matching shapes",
        ));
    }

    let ph_proc_slice = ph_proc_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_proc must be C-contiguous"))?;
    let ph_mean_v_slice = ph_mean_v_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_mean_v must be C-contiguous"))?;
    let bperp_slice = bperp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp_mat must be C-contiguous"))?;
    let unwrap_view = unwrap_ix.as_array();
    let unwrap_slice = unwrap_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("unwrap_ix must be contiguous"))?;
    let solve_view = solve_ix.as_array();
    let solve_slice = solve_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("solve_ix must be contiguous"))?;
    let day_view = day.as_array();
    let day_slice = day_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("day must be contiguous"))?;
    let ifg_std_view = ifg_std.as_array();
    let ifg_std_slice = ifg_std_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ifg_std must be contiguous"))?;

    let n_ps = ph_proc_view.shape()[0];
    let n_ifg = ph_proc_view.shape()[1];
    let unwrap_idx = parse_indices(unwrap_slice, n_ifg, "unwrap_ix")?;
    let solve_idx = parse_indices(solve_slice, n_ifg, "solve_ix")?;
    let outputs = py.detach(move || {
        stage7_outputs(
            ph_proc_slice,
            ph_mean_v_slice,
            bperp_slice,
            n_ps,
            n_ifg,
            &unwrap_idx,
            &solve_idx,
            day_slice,
            master_ix,
            ifg_std_slice,
            threads,
        )
    })?;

    let dict = PyDict::new(py);
    dict.set_item(
        "K_ps_uw",
        Array1::from_vec(outputs.k_ps_uw).into_pyarray(py),
    )?;
    dict.set_item(
        "C_ps_uw",
        Array1::from_vec(outputs.c_ps_uw).into_pyarray(py),
    )?;
    dict.set_item(
        "ph_scla",
        Array2::from_shape_vec((n_ps, n_ifg), outputs.ph_scla)
            .map_err(|err| {
                PyValueError::new_err(format!("failed to build stage7 ph_scla output: {err}"))
            })?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ifg_vcm",
        Array2::from_shape_vec((n_ifg, n_ifg), outputs.ifg_vcm)
            .map_err(|err| {
                PyValueError::new_err(format!("failed to build stage7 ifg_vcm output: {err}"))
            })?
            .into_pyarray(py),
    )?;
    dict.set_item("mean_v", Array1::from_vec(outputs.mean_v).into_pyarray(py))?;
    dict.set_item(
        "m",
        Array2::from_shape_vec((2, n_ps), outputs.m)
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "failed to build stage7 mean-velocity output: {err}"
                ))
            })?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ph_ramp",
        Array2::from_shape_vec((n_ps, n_ifg), outputs.ph_ramp)
            .map_err(|err| {
                PyValueError::new_err(format!("failed to build stage7 ph_ramp output: {err}"))
            })?
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[pyfunction(signature = (ph_uw, bperp_mat, no_master, day, master_ix, chunk_ps = 0, threads = 0))]
pub fn stage7_scla<'py>(
    py: Python<'py>,
    ph_uw: PyReadonlyArray2<f32>,
    bperp_mat: PyReadonlyArray2<f32>,
    no_master: PyReadonlyArray1<bool>,
    day: PyReadonlyArray1<f64>,
    master_ix: usize,
    chunk_ps: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let ph_view = ph_uw.as_array();
    let bperp_view = bperp_mat.as_array();
    let no_master_view = no_master.as_array();
    if ph_view.ndim() != 2 || bperp_view.ndim() != 2 {
        return Err(PyValueError::new_err(
            "stage7_scla expects 2-D ph_uw and bperp_mat",
        ));
    }
    let n_ps = ph_view.shape()[0];
    let n_ifg = ph_view.shape()[1];
    if no_master_view.len() != n_ifg || day.as_array().len() != n_ifg {
        return Err(PyValueError::new_err(
            "stage7_scla no_master/day length must match ph_uw width",
        ));
    }

    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_uw must be C-contiguous"))?;
    let bperp_slice = bperp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp_mat must be C-contiguous"))?;
    let no_master_slice = no_master_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("no_master must be contiguous"))?;
    let day_view = day.as_array();
    let day_slice = day_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("day must be contiguous"))?;
    let _ = chunk_ps;

    let mut unwrap_idx = Vec::new();
    for (ifg_ix, &keep) in no_master_slice.iter().enumerate() {
        if keep {
            unwrap_idx.push(ifg_ix);
        }
    }
    let solve_idx = unwrap_idx.clone();
    let ph_proc64: Vec<f64> = ph_slice.iter().map(|&value| value as f64).collect();
    let bperp64: Vec<f64> = bperp_slice.iter().map(|&value| value as f64).collect();
    let ifg_std = vec![1.0_f64; n_ifg];

    let outputs = py.detach(move || {
        stage7_outputs(
            &ph_proc64,
            &ph_proc64,
            &bperp64,
            n_ps,
            n_ifg,
            &unwrap_idx,
            &solve_idx,
            day_slice,
            master_ix,
            &ifg_std,
            threads,
        )
    })?;

    let dict = PyDict::new(py);
    dict.set_item(
        "K_ps_uw",
        Array1::from_vec(outputs.k_ps_uw).into_pyarray(py),
    )?;
    dict.set_item(
        "C_ps_uw",
        Array1::from_vec(outputs.c_ps_uw).into_pyarray(py),
    )?;
    dict.set_item(
        "ph_scla",
        Array2::from_shape_vec((n_ps, n_ifg), outputs.ph_scla)
            .map_err(|err| {
                PyValueError::new_err(format!("failed to build stage7 shim ph_scla output: {err}"))
            })?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ifg_vcm",
        Array2::from_shape_vec((n_ifg, n_ifg), outputs.ifg_vcm)
            .map_err(|err| {
                PyValueError::new_err(format!("failed to build stage7 shim ifg_vcm output: {err}"))
            })?
            .into_pyarray(py),
    )?;
    dict.set_item("mean_v", Array1::from_vec(outputs.mean_v).into_pyarray(py))?;
    dict.set_item(
        "m",
        Array2::from_shape_vec((2, n_ps), outputs.m)
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "failed to build stage7 shim mean-velocity output: {err}"
                ))
            })?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ph_ramp",
        Array2::from_shape_vec((n_ps, n_ifg), outputs.ph_ramp)
            .map_err(|err| {
                PyValueError::new_err(format!("failed to build stage7 shim ph_ramp output: {err}"))
            })?
            .into_pyarray(py),
    )?;
    Ok(dict)
}
