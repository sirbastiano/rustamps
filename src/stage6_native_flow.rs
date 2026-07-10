use super::stage6_native_boundary::route_residue_to_boundary;
use super::stage6_native_curl::balance_local_curl;
use super::{defo_edge_cost, EdgeDatum};
use crate::stage6_flow::pair_neighbor_residues;
use crate::stage6_tree_compact::optimize_tree_cycles_compact_with_nflow_parallel;

const SNAPHU_DEFAULT_MAX_CYCLE_FRACTION: f64 = 0.00004;
const MIN_FLOW_TREE_CYCLE_LIMIT: usize = 28;
const FLOW_BATCH_CONTINUE_NODE_LIMIT: usize = 8192;
const LARGE_FLOW_BATCH_LIMIT: usize = 3;
const SNAPHU_DEFAULT_MAX_FLOW: i32 = 4;

#[cfg(test)]
pub(crate) fn optimize_edge_flows(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) -> usize {
    optimize_edge_flows_with_parallel(horizontal, vertical, nrow, ncol, false)
}

pub(crate) fn snaphu_flow_increments(most_flow: i32) -> Vec<i32> {
    let upper = most_flow.max(1).min(SNAPHU_DEFAULT_MAX_FLOW);
    (1..=upper).collect()
}

pub(crate) fn snaphu_max_nflow_cycles(nrow: usize, ncol: usize) -> usize {
    ((nrow as f64 * ncol as f64 * SNAPHU_DEFAULT_MAX_CYCLE_FRACTION).ceil() as usize).max(1)
}

pub(crate) fn snaphu_flow_tree_cycle_limit(nrow: usize, ncol: usize) -> usize {
    (snaphu_max_nflow_cycles(nrow, ncol) * SNAPHU_DEFAULT_MAX_FLOW as usize)
        .max(MIN_FLOW_TREE_CYCLE_LIMIT)
}

pub(crate) fn snaphu_continue_capped_batches(nrow: usize, ncol: usize) -> bool {
    nrow.saturating_sub(1) * ncol.saturating_sub(1) + 1 <= FLOW_BATCH_CONTINUE_NODE_LIMIT
}

pub(crate) fn snaphu_capped_batch_limit(nrow: usize, ncol: usize) -> usize {
    if snaphu_continue_capped_batches(nrow, ncol) {
        usize::MAX
    } else {
        LARGE_FLOW_BATCH_LIMIT
    }
}

fn max_abs_edge_flow(horizontal: &[Option<EdgeDatum>], vertical: &[Option<EdgeDatum>]) -> i32 {
    horizontal
        .iter()
        .chain(vertical.iter())
        .filter_map(|edge| edge.map(|datum| datum.flow.abs()))
        .max()
        .unwrap_or(1)
}

pub(crate) fn optimize_edge_flows_with_parallel(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    parallel: bool,
) -> usize {
    pair_neighbor_residues(horizontal, vertical, nrow, ncol);
    route_residue_to_boundary(horizontal, vertical, nrow, ncol);
    balance_local_curl(horizontal, vertical, nrow, ncol);
    let max_cycles = snaphu_flow_tree_cycle_limit(nrow, ncol);
    let capped_batch_limit = snaphu_capped_batch_limit(nrow, ncol);
    snaphu_flow_increments(max_abs_edge_flow(horizontal, vertical))
        .into_iter()
        .map(|nflow| {
            let mut total = 0;
            let mut batches = 0;
            loop {
                let applied = optimize_tree_cycles_compact_with_nflow_parallel(
                    horizontal, vertical, nrow, ncol, nflow, max_cycles, parallel,
                );
                batches += 1;
                total += applied;
                if applied < max_cycles || batches >= capped_batch_limit {
                    break total;
                }
            }
        })
        .sum()
}

pub(super) fn edge_flow_objective(
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
) -> i64 {
    horizontal
        .iter()
        .chain(vertical.iter())
        .filter_map(|edge| *edge)
        .map(|edge| {
            defo_edge_cost(
                edge.cost,
                edge.offset,
                edge.dzmax,
                edge.laycost,
                edge.nshortcycle,
                edge.flow,
            )
        })
        .sum()
}
