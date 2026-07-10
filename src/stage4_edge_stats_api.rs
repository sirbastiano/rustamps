use num_complex::Complex64;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::stage4_edge_stats_core::stage4_edge_stats_outputs;
use crate::parse_indices;

#[pyfunction(signature = (ph_weed, node_a, node_b, bperp, day, time_win, small_baseline, threads = 0))]
pub fn stage4_edge_stats<'py>(
    py: Python<'py>,
    ph_weed: PyReadonlyArray2<Complex64>,
    node_a: PyReadonlyArray1<i64>,
    node_b: PyReadonlyArray1<i64>,
    bperp: PyReadonlyArray1<f64>,
    day: PyReadonlyArray1<f64>,
    time_win: f64,
    small_baseline: bool,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let ph_view = ph_weed.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err("ph_weed must be a 2-D matrix"));
    }
    let node_a_view = node_a.as_array();
    let node_b_view = node_b.as_array();
    if node_a_view.len() != node_b_view.len() {
        return Err(PyValueError::new_err(
            "node_a and node_b must have matching lengths",
        ));
    }
    let bperp_view = bperp.as_array();
    let day_view = day.as_array();

    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_weed must be C-contiguous"))?;
    let node_a_slice = node_a_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("node_a must be contiguous"))?;
    let node_b_slice = node_b_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("node_b must be contiguous"))?;
    let bperp_slice = bperp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp must be contiguous"))?;
    let day_slice = day_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("day must be contiguous"))?;

    let n_node = ph_view.shape()[0];
    let n_ifg = ph_view.shape()[1];
    if bperp_slice.len() != n_ifg {
        return Err(PyValueError::new_err(
            "stage4_edge_stats bperp length must match phase width",
        ));
    }
    if !small_baseline && day_slice.len() != n_ifg {
        return Err(PyValueError::new_err(
            "stage4_edge_stats day length must match phase width",
        ));
    }
    let edge_a = parse_indices(node_a_slice, n_node, "node_a")?;
    let edge_b = parse_indices(node_b_slice, n_node, "node_b")?;
    let (ps_std, ps_max) = py.detach(move || {
        stage4_edge_stats_outputs(
            ph_slice,
            n_node,
            n_ifg,
            &edge_a,
            &edge_b,
            bperp_slice,
            day_slice,
            time_win,
            small_baseline,
            threads,
        )
    })?;

    let dict = PyDict::new(py);
    dict.set_item("ps_std", Array1::from_vec(ps_std).into_pyarray(py))?;
    dict.set_item("ps_max", Array1::from_vec(ps_max).into_pyarray(py))?;
    Ok(dict)
}
