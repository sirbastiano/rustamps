use num_complex::Complex32;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::{build_pool, parse_indices, STAGE8_NOISE_SCALE};

#[pyfunction(signature = (uw_ph, node_a, node_b, chunk_edges = 0, threads = 0))]
pub fn stage8_edge_noise<'py>(
    py: Python<'py>,
    uw_ph: PyReadonlyArray2<Complex32>,
    node_a: PyReadonlyArray1<i64>,
    node_b: PyReadonlyArray1<i64>,
    chunk_edges: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let ph_view = uw_ph.as_array();
    let node_a_view = node_a.as_array();
    let node_b_view = node_b.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err("uw_ph must be a 2-D matrix"));
    }
    if node_a_view.len() != node_b_view.len() {
        return Err(PyValueError::new_err(
            "node_a and node_b must have matching lengths",
        ));
    }

    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("uw_ph must be C-contiguous"))?;
    let node_a_slice = node_a_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("node_a must be contiguous"))?;
    let node_b_slice = node_b_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("node_b must be contiguous"))?;
    let n_node = ph_view.shape()[0];
    let n_ifg = ph_view.shape()[1];
    let edge_a = parse_indices(node_a_slice, n_node, "node_a")?;
    let edge_b = parse_indices(node_b_slice, n_node, "node_b")?;
    let n_edge = edge_a.len();
    let _ = chunk_edges;
    let pool = build_pool(threads)?;

    let rows = py.detach(move || {
        let compute = || {
            (0..n_edge)
                .into_par_iter()
                .map(|edge_ix| edge_noise_row(ph_slice, n_ifg, edge_a[edge_ix], edge_b[edge_ix]))
                .collect::<Vec<_>>()
        };
        match pool {
            Some(pool) => pool.install(compute),
            None => (0..n_edge)
                .map(|edge_ix| edge_noise_row(ph_slice, n_ifg, edge_a[edge_ix], edge_b[edge_ix]))
                .collect::<Vec<_>>(),
        }
    });

    let mut dph_noise = vec![0.0_f32; n_edge * n_ifg];
    let mut dph_space_uw = vec![0.0_f32; n_edge * n_ifg];
    for (edge_ix, (noise_row, space_row)) in rows.into_iter().enumerate() {
        dph_noise[edge_ix * n_ifg..(edge_ix + 1) * n_ifg].copy_from_slice(&noise_row);
        dph_space_uw[edge_ix * n_ifg..(edge_ix + 1) * n_ifg].copy_from_slice(&space_row);
    }

    let dict = PyDict::new(py);
    dict.set_item(
        "dph_noise",
        Array2::from_shape_vec((n_edge, n_ifg), dph_noise)
            .map_err(|err| {
                PyValueError::new_err(format!("failed to build stage8 dph_noise output: {err}"))
            })?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "dph_space_uw",
        Array2::from_shape_vec((n_edge, n_ifg), dph_space_uw)
            .map_err(|err| {
                PyValueError::new_err(format!("failed to build stage8 dph_space_uw output: {err}"))
            })?
            .into_pyarray(py),
    )?;
    Ok(dict)
}

fn edge_noise_row(
    ph_slice: &[Complex32],
    n_ifg: usize,
    a_ix: usize,
    b_ix: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut dph_space = vec![0.0_f32; n_ifg];
    let mut sum = 0.0_f64;
    for ifg_ix in 0..n_ifg {
        let left = ph_slice[a_ix * n_ifg + ifg_ix];
        let right = ph_slice[b_ix * n_ifg + ifg_ix];
        let phase = (right * left.conj()).arg();
        dph_space[ifg_ix] = phase;
        sum += phase as f64;
    }
    let mean = if n_ifg == 0 {
        0.0_f32
    } else {
        (sum / n_ifg as f64) as f32
    };
    let dph_noise = dph_space
        .iter()
        .map(|&value| (value - mean) * STAGE8_NOISE_SCALE)
        .collect();
    (dph_noise, dph_space)
}
