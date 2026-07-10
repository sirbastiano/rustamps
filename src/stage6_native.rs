use num_complex::Complex32;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::stage6_component_shift::{
    refine_labels_by_barrier_component_shifts, refine_labels_by_component_shifts,
};
use crate::stage6_cut::refine_labels_by_binary_cuts;
use crate::stage6_label_flow::absorb_label_corrections;
use crate::stage6_patch::refine_labels_by_patch_shifts;

#[path = "stage6_native_boundary.rs"]
mod stage6_native_boundary;
#[path = "stage6_native_curl.rs"]
mod stage6_native_curl;
#[path = "stage6_native_decode.rs"]
mod stage6_native_decode;
#[path = "stage6_native_edges.rs"]
mod stage6_native_edges;
#[path = "stage6_native_flow.rs"]
mod stage6_native_flow;
#[path = "stage6_native_graph.rs"]
mod stage6_native_graph;
#[path = "stage6_native_grid_api.rs"]
mod stage6_native_grid_api;
#[path = "stage6_native_labels.rs"]
mod stage6_native_labels;
#[path = "stage6_native_phase_api.rs"]
mod stage6_native_phase_api;

use self::stage6_native_decode::{decode_cost_edges, decode_wrapped_phases, TWO_PI};
pub(crate) use self::stage6_native_edges::{
    apply_edge_correction, defo_edge_cost, edge_increment_cost, edge_label_energy, edge_weight,
    horizontal_index, rounded_delta, vertical_index, EdgeDatum,
};
use self::stage6_native_flow::edge_flow_objective;
pub(crate) use self::stage6_native_flow::optimize_edge_flows_with_parallel;
#[cfg(test)]
pub(crate) use self::stage6_native_flow::{
    optimize_edge_flows, snaphu_capped_batch_limit, snaphu_continue_capped_batches,
    snaphu_flow_increments, snaphu_flow_tree_cycle_limit, snaphu_max_nflow_cycles,
};
pub(crate) use self::stage6_native_graph::reseed_labels_from_edge_deltas;
use self::stage6_native_graph::{build_adjacency, seed_labels_from_adjacency};
pub use self::stage6_native_grid_api::{
    stage6_extract_grid_values, stage6_grid_accumulate, stage6_ps_grid_indices, stage6_select_ifgw,
    stage6_single_master_ifg_geometry, stage6_unwrap_ifg_sets,
};
use self::stage6_native_labels::{refine_labels, refine_labels_by_line_shifts};
pub use self::stage6_native_phase_api::{stage6_prepare_cost_offsets, stage6_reconstruct_ps_phase};

fn unwrap_grid(
    ifgw: &[Complex32],
    nrow: usize,
    ncol: usize,
    rowcost: &[i16],
    colcost: &[i16],
    nshortcycle: f32,
    parallel: bool,
) -> (Vec<f32>, usize, i64, usize, i64) {
    let node_count = nrow * ncol;
    let phases = decode_wrapped_phases(ifgw, nrow, ncol);
    let (mut horizontal, mut vertical) =
        decode_cost_edges(&phases, nrow, ncol, rowcost, colcost, nshortcycle);
    let flow_cycles =
        optimize_edge_flows_with_parallel(&mut horizontal, &mut vertical, nrow, ncol, parallel);
    let flow_objective = edge_flow_objective(&horizontal, &vertical);

    let adjacency = build_adjacency(&horizontal, &vertical, nrow, ncol);
    let mut labels = seed_labels_from_adjacency(node_count, &adjacency);
    refine_labels(&mut labels, &adjacency);
    refine_labels_by_line_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    refine_labels_by_binary_cuts(&mut labels, &horizontal, &vertical, nrow, ncol);
    refine_labels_by_patch_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    refine_labels_by_component_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    refine_labels_by_barrier_component_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    refine_labels(&mut labels, &adjacency);
    drop(adjacency);
    let mut post_horizontal = horizontal.clone();
    let mut post_vertical = vertical.clone();
    absorb_label_corrections(
        &labels,
        &mut post_horizontal,
        &mut post_vertical,
        nrow,
        ncol,
    );
    let post_label_flow_cycles = optimize_edge_flows_with_parallel(
        &mut post_horizontal,
        &mut post_vertical,
        nrow,
        ncol,
        parallel,
    );
    let post_label_flow_objective = edge_flow_objective(&post_horizontal, &post_vertical);
    reseed_labels_from_edge_deltas(&mut labels, &post_horizontal, &post_vertical, nrow, ncol);
    refine_labels_by_line_shifts(&mut labels, &post_horizontal, &post_vertical, nrow, ncol);
    refine_labels_by_binary_cuts(&mut labels, &post_horizontal, &post_vertical, nrow, ncol);
    refine_labels_by_patch_shifts(&mut labels, &post_horizontal, &post_vertical, nrow, ncol);
    refine_labels_by_component_shifts(&mut labels, &post_horizontal, &post_vertical, nrow, ncol);
    refine_labels_by_barrier_component_shifts(
        &mut labels,
        &post_horizontal,
        &post_vertical,
        nrow,
        ncol,
    );

    let unwrapped = phases
        .into_iter()
        .zip(labels)
        .map(|(phase, label)| phase + TWO_PI * label as f32)
        .collect();
    (
        unwrapped,
        flow_cycles,
        flow_objective,
        post_label_flow_cycles,
        post_label_flow_objective,
    )
}

fn neighbor_msd(values: &[f32], nrow: usize, ncol: usize) -> f64 {
    let mut accum = 0.0_f64;
    let mut count = 0_usize;
    if nrow > 1 {
        for row in 0..(nrow - 1) {
            for col in 0..ncol {
                let diff = values[row * ncol + col] - values[(row + 1) * ncol + col];
                if diff != 0.0 {
                    accum += f64::from(diff) * f64::from(diff);
                    count += 1;
                }
            }
        }
    }
    if ncol > 1 {
        for row in 0..nrow {
            for col in 0..(ncol - 1) {
                let diff = values[row * ncol + col] - values[row * ncol + col + 1];
                if diff != 0.0 {
                    accum += f64::from(diff) * f64::from(diff);
                    count += 1;
                }
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        accum / count as f64
    }
}

#[pyfunction(signature = (ifgw, rowcost, colcost, nshortcycle = 200.0, threads = 0))]
pub fn stage6_unwrap_grid<'py>(
    py: Python<'py>,
    ifgw: PyReadonlyArray2<Complex32>,
    rowcost: PyReadonlyArray2<i16>,
    colcost: PyReadonlyArray2<i16>,
    nshortcycle: f64,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = threads;
    if nshortcycle <= 0.0 || !nshortcycle.is_finite() {
        return Err(PyValueError::new_err(
            "nshortcycle must be a positive finite value",
        ));
    }
    let ifgw_view = ifgw.as_array();
    let rowcost_view = rowcost.as_array();
    let colcost_view = colcost.as_array();
    let shape = ifgw_view.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(
            "stage6_unwrap_grid expects a 2-D ifgw grid",
        ));
    }
    let nrow = shape[0];
    let ncol = shape[1];
    if nrow == 0 || ncol == 0 {
        return Err(PyValueError::new_err(
            "stage6_unwrap_grid expects a non-empty ifgw grid",
        ));
    }
    let expected_rowcost_shape = [nrow - 1, ncol * 4];
    let expected_colcost_shape = [nrow, (ncol - 1) * 4];
    if rowcost_view.shape() != expected_rowcost_shape.as_slice() {
        return Err(PyValueError::new_err(
            "stage6_unwrap_grid rowcost shape must be (nrow - 1, ncol * 4)",
        ));
    }
    if colcost_view.shape() != expected_colcost_shape.as_slice() {
        return Err(PyValueError::new_err(
            "stage6_unwrap_grid colcost shape must be (nrow, (ncol - 1) * 4)",
        ));
    }
    let ifgw_slice = ifgw_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ifgw must be C-contiguous"))?;
    let rowcost_slice = rowcost_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("rowcost must be C-contiguous"))?;
    let colcost_slice = colcost_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("colcost must be C-contiguous"))?;

    let nshortcycle32 = nshortcycle as f32;
    let pool = crate::build_pool(threads)?;
    let parallel = threads != 1;
    let (unwrapped, flow_cycles, flow_objective, post_label_flow_cycles, post_label_flow_objective) =
        py.detach(move || {
            let run = || {
                unwrap_grid(
                    ifgw_slice,
                    nrow,
                    ncol,
                    rowcost_slice,
                    colcost_slice,
                    nshortcycle32,
                    parallel,
                )
            };
            if let Some(pool) = pool {
                pool.install(run)
            } else {
                run()
            }
        });
    let msd = neighbor_msd(&unwrapped, nrow, ncol);
    let arr = Array2::from_shape_vec((nrow, ncol), unwrapped)
        .map_err(|err| PyValueError::new_err(format!("failed to build stage6 output: {err}")))?;
    let dict = PyDict::new(py);
    dict.set_item("ifguw", arr.into_pyarray(py))?;
    dict.set_item("msd", msd)?;
    dict.set_item("flow_cycles", flow_cycles)?;
    dict.set_item("flow_objective", flow_objective)?;
    dict.set_item("post_label_flow_cycles", post_label_flow_cycles)?;
    dict.set_item("post_label_flow_objective", post_label_flow_objective)?;
    Ok(dict)
}
