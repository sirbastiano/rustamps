use super::{defo_edge_cost, EdgeDatum};
use crate::stage6::unwrap::mst::apply_mst_flows;
use crate::stage6::unwrap::mst_flow::mst_initial_flows;
use crate::stage6::unwrap::residual_view::CompactResidualView;
use crate::stage6::unwrap::residue::edge_residues;
use crate::stage6::unwrap::tree_compact::{
    optimize_tree_cycles_compact_with_state, CompactTreeState,
};

const SNAPHU_DEFAULT_MAX_CYCLE_FRACTION: f64 = 0.00001;
const MIN_FLOW_TREE_CYCLE_LIMIT: usize = 28;
const SNAPHU_DEFAULT_MAX_FLOW: i32 = 4;

#[cfg(test)]
pub(crate) fn optimize_edge_flows(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) -> usize {
    optimize_edge_flows_with_parallel(horizontal, vertical, nrow, ncol, false, None)
}

#[cfg(test)]
pub(crate) fn snaphu_flow_increments(most_flow: i32) -> Vec<i32> {
    let upper = most_flow.max(1).min(SNAPHU_DEFAULT_MAX_FLOW);
    (1..=upper).collect()
}

pub(crate) fn snaphu_max_nflow_cycles(nrow: usize, ncol: usize) -> usize {
    (SNAPHU_DEFAULT_MAX_CYCLE_FRACTION * nrow as f64 * ncol as f64).round_ties_even() as usize
}

#[cfg(test)]
mod tests {
    use super::snaphu_max_nflow_cycles;

    #[test]
    fn max_nflow_cycles_preserves_upstream_zero_for_small_grids() {
        assert_eq!(snaphu_max_nflow_cycles(93, 236), 0);
        assert_eq!(snaphu_max_nflow_cycles(500, 500), 2);
        assert_eq!(snaphu_max_nflow_cycles(350, 1000), 4);
        assert_eq!(snaphu_max_nflow_cycles(200, 3250), 6);
    }
}

pub(crate) fn snaphu_flow_tree_cycle_limit(nrow: usize, ncol: usize) -> usize {
    (snaphu_max_nflow_cycles(nrow, ncol) * SNAPHU_DEFAULT_MAX_FLOW as usize)
        .max(MIN_FLOW_TREE_CYCLE_LIMIT)
}

pub(crate) fn next_flow_increment(nflow: i32, most_flow: i32) -> i32 {
    let next = nflow.max(1) + 1;
    if next > SNAPHU_DEFAULT_MAX_FLOW || next > most_flow.max(1) {
        1
    } else {
        next
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

fn optimize_flow_increment(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    nflow: i32,
    max_cycles: usize,
    parallel: bool,
    state: &mut CompactTreeState,
    max_passes: Option<usize>,
) -> (usize, usize) {
    let mut total = 0_usize;
    let mut passes = 0_usize;
    loop {
        if max_passes.is_some_and(|limit| passes >= limit) {
            return (total, passes);
        }
        let applied = optimize_tree_cycles_compact_with_state(
            horizontal, vertical, nrow, ncol, nflow, max_cycles, parallel, state,
        );
        passes += 1;
        total = total.saturating_add(applied);
        if applied == 0 {
            return (total, passes);
        }
    }
}

pub(crate) fn optimize_edge_flows_with_parallel(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    parallel: bool,
    max_passes: Option<usize>,
) -> usize {
    let residue = edge_residues(horizontal, vertical, nrow, ncol);
    let (rowflow, colflow) = mst_initial_flows(&residue, horizontal, vertical, nrow, ncol);
    apply_mst_flows(horizontal, vertical, nrow, ncol, &rowflow, &colflow);
    let max_cycles = snaphu_flow_tree_cycle_limit(nrow, ncol);
    let max_nflow_cycles = snaphu_max_nflow_cycles(nrow, ncol);
    let mut state = {
        let view = CompactResidualView::new(horizontal, vertical, nrow, ncol);
        CompactTreeState::new(&view)
    };
    let mut total = 0_usize;
    let mut nflow = 1_i32;
    let mut nflow_done = 0_i32;
    let mut minimum_objective = edge_flow_objective(horizontal, vertical);
    let mut nondecreasing_iterations = 0_i32;
    let mut passes = 0_usize;

    loop {
        let remaining = max_passes.map(|limit| limit.saturating_sub(passes));
        let (applied, used) = optimize_flow_increment(
            horizontal, vertical, nrow, ncol, nflow, max_cycles, parallel, &mut state, remaining,
        );
        passes += used;
        total = total.saturating_add(applied);
        if max_passes.is_some_and(|limit| passes >= limit) {
            break;
        }
        if applied <= max_nflow_cycles {
            nflow_done += 1;
        } else {
            nflow_done = 1;
        }

        let most_flow = max_abs_edge_flow(horizontal, vertical).max(1);
        let objective = edge_flow_objective(horizontal, vertical);
        if objective > minimum_objective {
            nondecreasing_iterations += 1;
        } else {
            minimum_objective = objective;
            nondecreasing_iterations = 0;
        }
        if nondecreasing_iterations >= 2 * most_flow
            || nflow_done >= SNAPHU_DEFAULT_MAX_FLOW
            || nflow_done >= most_flow
        {
            break;
        }
        nflow = next_flow_increment(nflow, most_flow);
    }
    total
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
