use num_complex::Complex32;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub fn stage6_unwrap_ifg_sets<'py>(
    py: Python<'py>,
    n_ifg: i64,
    master_ix: i64,
    drop_ifg_index: PyReadonlyArray1<i64>,
    small_baseline: bool,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = threads;
    if n_ifg < 0 {
        return Err(PyValueError::new_err("n_ifg must be non-negative"));
    }
    if !small_baseline && (master_ix < 1 || master_ix > n_ifg) {
        return Err(PyValueError::new_err(
            "master_ix must be 1-based within the interferogram stack",
        ));
    }
    let drop_view = drop_ifg_index.as_array();
    let drop_slice = drop_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("drop_ifg_index must be contiguous"))?;

    let mut unwrap_ifg = Vec::with_capacity(n_ifg as usize);
    let mut solve_ifg = Vec::with_capacity(n_ifg as usize);
    'ifg_loop: for ifg in 1..=n_ifg {
        for &drop in drop_slice {
            if ifg == drop {
                continue 'ifg_loop;
            }
        }
        unwrap_ifg.push(ifg);
        if small_baseline || ifg != master_ix {
            solve_ifg.push(ifg);
        }
    }

    let dict = PyDict::new(py);
    dict.set_item("unwrap_ifg", unwrap_ifg.into_pyarray(py))?;
    dict.set_item("solve_ifg", solve_ifg.into_pyarray(py))?;
    Ok(dict)
}

#[pyfunction]
pub fn stage6_single_master_ifg_geometry<'py>(
    py: Python<'py>,
    n_ifg: i64,
    master_ix: i64,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = threads;
    if n_ifg < 0 {
        return Err(PyValueError::new_err("n_ifg must be non-negative"));
    }
    if master_ix < 1 || master_ix > n_ifg {
        return Err(PyValueError::new_err(
            "master_ix must be 1-based within the interferogram stack",
        ));
    }

    let count = n_ifg.saturating_sub(1) as usize;
    let mut unwrap_ifg = Vec::with_capacity(count);
    let mut ifgday_ix = Vec::with_capacity(count * 2);
    for ifg in 1..=n_ifg {
        if ifg == master_ix {
            continue;
        }
        unwrap_ifg.push(ifg);
        ifgday_ix.push(master_ix);
        ifgday_ix.push(ifg);
    }

    let ifgday_arr = Array2::from_shape_vec((count, 2), ifgday_ix).map_err(|err| {
        PyValueError::new_err(format!("failed to build stage6 ifgday_ix output: {err}"))
    })?;
    let dict = PyDict::new(py);
    dict.set_item("unwrap_ifg", unwrap_ifg.into_pyarray(py))?;
    dict.set_item("ifgday_ix", ifgday_arr.into_pyarray(py))?;
    Ok(dict)
}

#[pyfunction]
pub fn stage6_grid_accumulate<'py>(
    py: Python<'py>,
    ph_in: PyReadonlyArray2<Complex32>,
    grid_lin: PyReadonlyArray1<i64>,
    n_cells: i64,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
    let _ = threads;
    if n_cells < 0 {
        return Err(PyValueError::new_err("n_cells must be non-negative"));
    }
    let ph_view = ph_in.as_array();
    let grid_view = grid_lin.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err(
            "stage6_grid_accumulate expects a 2-D ph_in matrix",
        ));
    }
    let n_ps = ph_view.shape()[0];
    let n_ifg = ph_view.shape()[1];
    if grid_view.len() != n_ps {
        return Err(PyValueError::new_err(
            "stage6_grid_accumulate expects grid_lin length to match ph_in rows",
        ));
    }
    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_in must be C-contiguous"))?;
    let grid_slice = grid_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("grid_lin must be contiguous"))?;

    let cell_count = n_cells as usize;
    let mut values = vec![Complex32::new(0.0, 0.0); cell_count * n_ifg];
    for row in 0..n_ps {
        let cell = grid_slice[row];
        if cell < 0 || cell >= n_cells {
            return Err(PyValueError::new_err(
                "stage6_grid_accumulate grid_lin contains an out-of-range cell index",
            ));
        }
        let cell_usize = cell as usize;
        for col in 0..n_ifg {
            values[cell_usize * n_ifg + col] += ph_slice[row * n_ifg + col];
        }
    }

    Ok(Array2::from_shape_vec((cell_count, n_ifg), values)
        .map_err(|err| PyValueError::new_err(format!("failed to build stage6 grid output: {err}")))?
        .into_pyarray(py))
}

#[pyfunction]
pub fn stage6_extract_grid_values<'py>(
    py: Python<'py>,
    ifguw: PyReadonlyArray2<f32>,
    nzix: PyReadonlyArray2<bool>,
    _threads: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let grid = ifguw.as_array();
    let mask = nzix.as_array();
    if grid.shape() != mask.shape() {
        return Err(PyValueError::new_err(
            "stage6_extract_grid_values expects ifguw and nzix with matching 2-D shapes",
        ));
    }
    let grid_slice = grid
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ifguw must be C-contiguous"))?;
    let mask_slice = mask
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("nzix must be C-contiguous"))?;
    let nrow = grid.shape()[0];
    let ncol = grid.shape()[1];
    let mut out = Vec::with_capacity(mask_slice.iter().filter(|&&keep| keep).count());
    for col in 0..ncol {
        for row in 0..nrow {
            let offset = row * ncol + col;
            if mask_slice[offset] {
                out.push(grid_slice[offset]);
            }
        }
    }
    Ok(out.into_pyarray(py))
}

#[pyfunction]
pub fn stage6_ps_grid_indices<'py>(
    py: Python<'py>,
    nzix: PyReadonlyArray2<bool>,
    grid_ij: PyReadonlyArray2<i64>,
    _threads: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let mask_view = nzix.as_array();
    let grid_view = grid_ij.as_array();
    if mask_view.ndim() != 2 || grid_view.ndim() != 2 || grid_view.shape()[1] != 2 {
        return Err(PyValueError::new_err(
            "stage6_ps_grid_indices expects nzix as 2-D and grid_ij with shape (n_ps, 2)",
        ));
    }
    let nrow = mask_view.shape()[0];
    let ncol = mask_view.shape()[1];
    let n_ps = grid_view.shape()[0];
    let mask_slice = mask_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("nzix must be C-contiguous"))?;
    let grid_slice = grid_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("grid_ij must be C-contiguous"))?;

    let mut gridix = vec![0_i64; nrow * ncol];
    let mut next_id = 1_i64;
    for col in 0..ncol {
        for row in 0..nrow {
            let offset = row * ncol + col;
            if mask_slice[offset] {
                gridix[offset] = next_id;
                next_id += 1;
            }
        }
    }

    let mut out = Vec::with_capacity(n_ps);
    for ps_ix in 0..n_ps {
        let row_one = grid_slice[ps_ix * 2];
        let col_one = grid_slice[ps_ix * 2 + 1];
        if row_one <= 0 || col_one <= 0 || row_one as usize > nrow || col_one as usize > ncol {
            return Err(PyValueError::new_err(
                "stage6_ps_grid_indices grid_ij entries must be 1-based within nzix",
            ));
        }
        let row = (row_one - 1) as usize;
        let col = (col_one - 1) as usize;
        out.push(gridix[row * ncol + col]);
    }

    Ok(out.into_pyarray(py))
}

#[pyfunction]
pub fn stage6_select_ifgw<'py>(
    py: Python<'py>,
    uw_ph: PyReadonlyArray2<Complex32>,
    z: PyReadonlyArray2<i64>,
    ifg_ix: usize,
    _threads: usize,
) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
    let ph_view = uw_ph.as_array();
    let z_view = z.as_array();
    if ph_view.ndim() != 2 || z_view.ndim() != 2 {
        return Err(PyValueError::new_err(
            "stage6_select_ifgw expects 2-D uw_ph and Z arrays",
        ));
    }
    let n_grid_ps = ph_view.shape()[0];
    let n_ifg = ph_view.shape()[1];
    if ifg_ix >= n_ifg {
        return Err(PyValueError::new_err(
            "stage6_select_ifgw ifg_ix must be within uw_ph columns",
        ));
    }
    let nrow = z_view.shape()[0];
    let ncol = z_view.shape()[1];
    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("uw_ph must be C-contiguous"))?;
    let z_slice = z_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Z must be C-contiguous"))?;

    let mut out = Vec::with_capacity(nrow * ncol);
    for &z_value in z_slice {
        if z_value <= 0 || z_value as usize > n_grid_ps {
            return Err(PyValueError::new_err(
                "stage6_select_ifgw Z entries must be 1-based within uw_ph rows",
            ));
        }
        let grid_ix = (z_value - 1) as usize;
        out.push(ph_slice[grid_ix * n_ifg + ifg_ix]);
    }

    Ok(Array2::from_shape_vec((nrow, ncol), out)
        .map_err(|err| {
            PyValueError::new_err(format!("failed to build stage6_select_ifgw output: {err}"))
        })?
        .into_pyarray(py))
}
