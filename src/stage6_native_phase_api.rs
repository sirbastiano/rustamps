use num_complex::Complex32;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction(signature = (rowcost_base, colcost_base, rowix, colix, wrapped_space_uw, dph_smooth, nshortcycle = 200.0, threads = 0))]
pub fn stage6_prepare_cost_offsets<'py>(
    py: Python<'py>,
    rowcost_base: PyReadonlyArray2<i16>,
    colcost_base: PyReadonlyArray2<i16>,
    rowix: PyReadonlyArray2<f64>,
    colix: PyReadonlyArray2<f64>,
    wrapped_space_uw: PyReadonlyArray1<f32>,
    dph_smooth: PyReadonlyArray1<f32>,
    nshortcycle: f64,
    threads: usize,
) -> PyResult<(Bound<'py, PyArray2<i16>>, Bound<'py, PyArray2<i16>>)> {
    let _ = threads;
    if nshortcycle <= 0.0 || !nshortcycle.is_finite() {
        return Err(PyValueError::new_err(
            "nshortcycle must be a positive finite value",
        ));
    }
    let rowcost_view = rowcost_base.as_array();
    let colcost_view = colcost_base.as_array();
    let rowix_view = rowix.as_array();
    let colix_view = colix.as_array();
    let wrapped_view = wrapped_space_uw.as_array();
    let smooth_view = dph_smooth.as_array();

    if rowcost_view.ndim() != 2 || colcost_view.ndim() != 2 {
        return Err(PyValueError::new_err(
            "stage6_prepare_cost_offsets expects 2-D cost matrices",
        ));
    }
    if rowcost_view.shape()[1] % 4 != 0 || colcost_view.shape()[1] % 4 != 0 {
        return Err(PyValueError::new_err(
            "stage6_prepare_cost_offsets cost widths must be multiples of 4",
        ));
    }
    let row_shape = [rowcost_view.shape()[0], rowcost_view.shape()[1] / 4];
    let col_shape = [colcost_view.shape()[0], colcost_view.shape()[1] / 4];
    if rowix_view.shape() != row_shape.as_slice() || colix_view.shape() != col_shape.as_slice() {
        return Err(PyValueError::new_err(
            "stage6_prepare_cost_offsets edge index shapes must match cost arc columns",
        ));
    }
    if wrapped_view.len() != smooth_view.len() {
        return Err(PyValueError::new_err(
            "stage6_prepare_cost_offsets wrapped_space_uw and dph_smooth must have matching lengths",
        ));
    }

    let rowcost_slice = rowcost_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("rowcost_base must be C-contiguous"))?;
    let colcost_slice = colcost_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("colcost_base must be C-contiguous"))?;
    let rowix_slice = rowix_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("rowix must be C-contiguous"))?;
    let colix_slice = colix_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("colix must be C-contiguous"))?;
    let wrapped_slice = wrapped_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("wrapped_space_uw must be contiguous"))?;
    let smooth_slice = smooth_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("dph_smooth must be contiguous"))?;

    let mut rowcost = rowcost_slice.to_vec();
    let mut colcost = colcost_slice.to_vec();
    let n_edge = wrapped_slice.len();
    let scale = nshortcycle / std::f64::consts::TAU;

    for row in 0..row_shape[0] {
        for col in 0..row_shape[1] {
            let edge_value = rowix_slice[row * row_shape[1] + col];
            if !edge_value.is_finite() || edge_value == 0.0 {
                continue;
            }
            let edge_ix = edge_value.abs() as usize;
            if edge_ix == 0 || edge_ix > n_edge {
                return Err(PyValueError::new_err(
                    "stage6_prepare_cost_offsets rowix edge index exceeds phase vectors",
                ));
            }
            let offset = (f64::from(wrapped_slice[edge_ix - 1])
                - f64::from(smooth_slice[edge_ix - 1]))
                * edge_value.signum()
                * scale;
            let cost_ix = row * rowcost_view.shape()[1] + col * 4;
            rowcost[cost_ix] = -(offset.round_ties_even() as i16);
        }
    }

    for row in 0..col_shape[0] {
        for col in 0..col_shape[1] {
            let edge_value = colix_slice[row * col_shape[1] + col];
            if !edge_value.is_finite() || edge_value == 0.0 {
                continue;
            }
            let edge_ix = edge_value.abs() as usize;
            if edge_ix == 0 || edge_ix > n_edge {
                return Err(PyValueError::new_err(
                    "stage6_prepare_cost_offsets colix edge index exceeds phase vectors",
                ));
            }
            let offset = (f64::from(wrapped_slice[edge_ix - 1])
                - f64::from(smooth_slice[edge_ix - 1]))
                * edge_value.signum()
                * scale;
            let cost_ix = row * colcost_view.shape()[1] + col * 4;
            colcost[cost_ix] = offset.round_ties_even() as i16;
        }
    }

    let row_arr =
        Array2::from_shape_vec((rowcost_view.shape()[0], rowcost_view.shape()[1]), rowcost)
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "failed to build stage6 rowcost offset output: {err}"
                ))
            })?
            .into_pyarray(py);
    let col_arr =
        Array2::from_shape_vec((colcost_view.shape()[0], colcost_view.shape()[1]), colcost)
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "failed to build stage6 colcost offset output: {err}"
                ))
            })?
            .into_pyarray(py);
    Ok((row_arr, col_arr))
}

#[pyfunction(signature = (ph_uw_grid, ps_grid_idx, ph_in, phase_restore = None, threads = 0))]
pub fn stage6_reconstruct_ps_phase<'py>(
    py: Python<'py>,
    ph_uw_grid: PyReadonlyArray2<f32>,
    ps_grid_idx: PyReadonlyArray1<i64>,
    ph_in: PyReadonlyArray2<Complex32>,
    phase_restore: Option<PyReadonlyArray2<f32>>,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let _ = threads;
    let grid_view = ph_uw_grid.as_array();
    let idx_view = ps_grid_idx.as_array();
    let ph_in_view = ph_in.as_array();
    if grid_view.ndim() != 2 || ph_in_view.ndim() != 2 {
        return Err(PyValueError::new_err(
            "stage6_reconstruct_ps_phase expects 2-D phase inputs",
        ));
    }
    let n_grid_ps = grid_view.shape()[0];
    let n_ifg = grid_view.shape()[1];
    let n_ps = idx_view.len();
    if ph_in_view.shape() != [n_ps, n_ifg] {
        return Err(PyValueError::new_err(
            "stage6_reconstruct_ps_phase ph_in shape must match (n_ps, n_ifg)",
        ));
    }
    let restore_view = phase_restore.as_ref().map(|arr| arr.as_array());
    if let Some(view) = restore_view.as_ref() {
        if view.shape() != [n_ps, n_ifg] {
            return Err(PyValueError::new_err(
                "stage6_reconstruct_ps_phase phase_restore shape must match ph_in",
            ));
        }
    }

    let grid_slice = grid_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_uw_grid must be C-contiguous"))?;
    let idx_slice = idx_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ps_grid_idx must be contiguous"))?;
    let ph_in_slice = ph_in_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_in must be C-contiguous"))?;
    let restore_slice = match restore_view.as_ref() {
        Some(view) => Some(
            view.as_slice()
                .ok_or_else(|| PyValueError::new_err("phase_restore must be C-contiguous"))?,
        ),
        None => None,
    };

    let mut out = vec![f32::NAN; n_ps * n_ifg];
    for ps_ix in 0..n_ps {
        let grid_one_based = idx_slice[ps_ix];
        if grid_one_based <= 0 {
            continue;
        }
        let grid_ix = (grid_one_based - 1) as usize;
        if grid_ix >= n_grid_ps {
            return Err(PyValueError::new_err(
                "stage6_reconstruct_ps_phase ps_grid_idx exceeds grid rows",
            ));
        }
        for ifg_ix in 0..n_ifg {
            let pix = grid_slice[grid_ix * n_ifg + ifg_ix];
            let wrapped = ph_in_slice[ps_ix * n_ifg + ifg_ix];
            let cos_pix = pix.cos();
            let sin_pix = pix.sin();
            let real = wrapped.re * cos_pix + wrapped.im * sin_pix;
            let imag = wrapped.im * cos_pix - wrapped.re * sin_pix;
            let correction = imag.atan2(real);
            let mut value = pix + correction;
            if let Some(restore) = restore_slice {
                value += restore[ps_ix * n_ifg + ifg_ix];
            }
            out[ps_ix * n_ifg + ifg_ix] = value;
        }
    }

    Ok(Array2::from_shape_vec((n_ps, n_ifg), out)
        .map_err(|err| {
            PyValueError::new_err(format!(
                "failed to build stage6_reconstruct_ps_phase output: {err}"
            ))
        })?
        .into_pyarray(py))
}
