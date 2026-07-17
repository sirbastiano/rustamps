use num_complex::Complex32;
use std::time::Instant;

use crate::stage6::unwrap::component_shift::{
    refine_labels_by_barrier_component_shifts, refine_labels_by_component_shifts,
};
use crate::stage6::unwrap::cut::refine_labels_by_binary_cuts;
use crate::stage6::unwrap::label_flow::absorb_label_corrections;
use crate::stage6::unwrap::patch::refine_labels_by_patch_shifts;

#[path = "stage6_native_decode.rs"]
mod stage6_native_decode;
#[path = "stage6_native_edges.rs"]
mod stage6_native_edges;
#[path = "stage6_native_flow.rs"]
mod stage6_native_flow;
#[path = "stage6_native_graph.rs"]
mod stage6_native_graph;
#[path = "stage6_native_labels.rs"]
mod stage6_native_labels;

use self::stage6_native_decode::{decode_cost_edges, decode_wrapped_phases, TWO_PI};
pub(crate) use self::stage6_native_edges::{
    apply_edge_correction, defo_edge_cost, edge_label_energy, edge_weight, horizontal_index,
    rounded_delta, vertical_index, EdgeDatum,
};
use self::stage6_native_flow::edge_flow_objective;
pub(crate) use self::stage6_native_flow::optimize_edge_flows_with_parallel;
pub(crate) use self::stage6_native_graph::reseed_labels_from_edge_deltas;
use self::stage6_native_graph::{build_adjacency, seed_labels_from_adjacency};
use self::stage6_native_labels::{refine_labels, refine_labels_by_line_shifts};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct NativeUnwrapTimings {
    pub decode_sec: f64,
    pub initial_flow_sec: f64,
    pub initial_label_sec: f64,
    pub post_flow_sec: f64,
    pub final_label_sec: f64,
}

pub(super) struct NativeUnwrapResult {
    pub ifguw: Vec<f32>,
    pub flow_cycles: usize,
    pub flow_objective: i64,
    pub post_cycles: usize,
    pub post_objective: i64,
    pub timings: NativeUnwrapTimings,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn unwrap_grid(
    ifgw: &[Complex32],
    nrow: usize,
    ncol: usize,
    rowcost: &[i16],
    colcost: &[i16],
    nshortcycle: f32,
    parallel: bool,
    max_flow_passes: Option<usize>,
) -> NativeUnwrapResult {
    let decode_started = Instant::now();
    let phases = decode_wrapped_phases(ifgw, nrow, ncol);
    let active = ifgw
        .iter()
        .map(|value| value.re != 0.0 || value.im != 0.0)
        .collect::<Vec<_>>();
    let (mut horizontal, mut vertical) =
        decode_cost_edges(&phases, &active, nrow, ncol, rowcost, colcost, nshortcycle);
    let decode_sec = decode_started.elapsed().as_secs_f64();

    let initial_flow_started = Instant::now();
    let flow_cycles = optimize_edge_flows_with_parallel(
        &mut horizontal,
        &mut vertical,
        nrow,
        ncol,
        parallel,
        max_flow_passes,
    );
    let flow_objective = edge_flow_objective(&horizontal, &vertical);
    let initial_flow_sec = initial_flow_started.elapsed().as_secs_f64();

    let initial_label_started = Instant::now();
    let adjacency = build_adjacency(&horizontal, &vertical, nrow, ncol);
    let mut labels = seed_labels_from_adjacency(nrow * ncol, &adjacency);
    refine_labels(&mut labels, &adjacency);
    refine_labels_by_line_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    refine_labels_by_binary_cuts(&mut labels, &horizontal, &vertical, nrow, ncol);
    refine_labels_by_patch_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    refine_labels_by_component_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    refine_labels_by_barrier_component_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    refine_labels(&mut labels, &adjacency);
    drop(adjacency);

    absorb_label_corrections(&labels, &mut horizontal, &mut vertical, nrow, ncol);
    let initial_label_sec = initial_label_started.elapsed().as_secs_f64();

    let post_flow_started = Instant::now();
    let post_cycles = optimize_edge_flows_with_parallel(
        &mut horizontal,
        &mut vertical,
        nrow,
        ncol,
        parallel,
        max_flow_passes,
    );
    let post_flow_sec = post_flow_started.elapsed().as_secs_f64();

    let final_label_started = Instant::now();
    reseed_labels_from_edge_deltas(&mut labels, &horizontal, &vertical, nrow, ncol);
    refine_labels_by_line_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    refine_labels_by_binary_cuts(&mut labels, &horizontal, &vertical, nrow, ncol);
    refine_labels_by_patch_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    refine_labels_by_component_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    refine_labels_by_barrier_component_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    absorb_label_corrections(&labels, &mut horizontal, &mut vertical, nrow, ncol);
    let post_objective = edge_flow_objective(&horizontal, &vertical);
    let output = phases
        .into_iter()
        .zip(labels)
        .map(|(phase, label)| phase + TWO_PI * label as f32)
        .collect();
    let final_label_sec = final_label_started.elapsed().as_secs_f64();
    NativeUnwrapResult {
        ifguw: output,
        flow_cycles,
        flow_objective,
        post_cycles,
        post_objective,
        timings: NativeUnwrapTimings {
            decode_sec,
            initial_flow_sec,
            initial_label_sec,
            post_flow_sec,
            final_label_sec,
        },
    }
}

pub(super) fn neighbor_msd(values: &[f32], nrow: usize, ncol: usize) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for row in 0..nrow.saturating_sub(1) {
        for col in 0..ncol {
            let difference = values[row * ncol + col] - values[(row + 1) * ncol + col];
            if difference != 0.0 {
                sum += f64::from(difference) * f64::from(difference);
                count += 1;
            }
        }
    }
    for row in 0..nrow {
        for col in 0..ncol.saturating_sub(1) {
            let difference = values[row * ncol + col] - values[row * ncol + col + 1];
            if difference != 0.0 {
                sum += f64::from(difference) * f64::from(difference);
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}
