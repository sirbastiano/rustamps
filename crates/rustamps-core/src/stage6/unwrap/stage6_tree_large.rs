use crate::stage6::unwrap::native::EdgeDatum;
use crate::stage6::unwrap::residual_view::{
    saturate_compact_residual_cycle_with_nflow, CompactResidualView,
};

use super::pivot_compact_tree_on_cycle_fast;
use super::stage6_tree_basis::CompactTreeBasis;

const REBASE_NODE_LIMIT: usize = 8192;
const CANDIDATE_BATCH_LIMIT: usize = 32;

pub(crate) struct CompactTreeState {
    pub(super) node_count: usize,
    pub(super) trees: [Vec<usize>; 2],
}

impl CompactTreeState {
    pub(crate) fn new(view: &CompactResidualView<'_>) -> Self {
        Self {
            node_count: view.node_count(),
            trees: [
                super::spanning_tree_arc_indices_compact_with_order(view, super::ArcOrder::Forward),
                super::spanning_tree_arc_indices_compact_with_order(
                    view,
                    super::ArcOrder::NegativeFirst,
                ),
            ],
        }
    }
}

pub(super) fn optimize_large_tree_cycles(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    nflow: i32,
    max_cycles: usize,
    parallel: bool,
    node_count: usize,
    tree: &mut Vec<usize>,
) -> usize {
    let rebase_tree = node_count <= REBASE_NODE_LIMIT;
    let candidate_limit = if rebase_tree {
        1
    } else {
        CANDIDATE_BATCH_LIMIT
    };
    if max_cycles == 0 {
        return 0;
    }
    let mut applied = 0;
    let mut basis = {
        let view = CompactResidualView::with_nflow(horizontal, vertical, nrow, ncol, nflow);
        let Some(basis) = CompactTreeBasis::new(&view, tree) else {
            return 0;
        };
        basis
    };
    let mut basis_ready = true;
    while applied < max_cycles {
        if !basis_ready {
            let rebuilt = {
                let view = CompactResidualView::with_nflow(horizontal, vertical, nrow, ncol, nflow);
                basis.rebuild(&view, tree)
            };
            if rebuilt.is_none() {
                break;
            }
        }
        basis_ready = false;
        let view = CompactResidualView::with_nflow(horizontal, vertical, nrow, ncol, nflow);
        let cycles = if candidate_limit == 1 {
            let cycle = if parallel {
                basis.find_negative_cycle_parallel(&view)
            } else {
                basis.find_negative_cycle(&view)
            };
            cycle.into_iter().collect::<Vec<_>>()
        } else {
            basis.negative_cycles(&view, candidate_limit.min(max_cycles - applied), parallel)
        };
        if cycles.is_empty() {
            break;
        }

        let mut batch_applied = 0;
        let mut first_cycle = None;
        for cycle in cycles {
            let increments = saturate_compact_residual_cycle_with_nflow(
                horizontal, vertical, nrow, ncol, &cycle, nflow,
            );
            if increments == 0 {
                continue;
            }
            if first_cycle.is_none() {
                first_cycle = Some(cycle.clone());
            }
            applied += 1;
            batch_applied += 1;
            if rebase_tree || applied == max_cycles {
                break;
            }
        }
        if batch_applied == 0 {
            break;
        }
        let Some(cycle) = first_cycle else {
            break;
        };
        if rebase_tree {
            if !pivot_compact_tree_on_cycle_fast(tree, &cycle) {
                break;
            }
            let view = CompactResidualView::with_nflow(horizontal, vertical, nrow, ncol, nflow);
            if basis.rebuild(&view, tree).is_none() {
                break;
            }
            basis_ready = true;
        } else if !pivot_compact_tree_on_cycle_fast(tree, &cycle) {
            break;
        }
    }
    applied
}
