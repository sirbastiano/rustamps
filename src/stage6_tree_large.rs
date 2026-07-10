use crate::stage6_native::EdgeDatum;
use crate::stage6_residual_view::{apply_compact_residual_cycle_with_nflow, CompactResidualView};

use super::stage6_tree_basis::CompactTreeBasis;
use super::{pivot_compact_tree_on_cycle, pivot_compact_tree_on_cycle_fast};

const REBASE_NODE_LIMIT: usize = 8192;
const CANDIDATE_BATCH_LIMIT: usize = 8;

pub(super) fn optimize_large_tree_cycles(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    nflow: i32,
    max_cycles: usize,
    parallel: bool,
    node_count: usize,
    mut tree: Vec<usize>,
) -> usize {
    let rebase_tree = node_count <= REBASE_NODE_LIMIT;
    let candidate_limit = if rebase_tree {
        1
    } else {
        CANDIDATE_BATCH_LIMIT
    };
    let mut applied = 0;
    while applied < max_cycles {
        let Some(basis) = ({
            let view = CompactResidualView::with_nflow(horizontal, vertical, nrow, ncol, nflow);
            CompactTreeBasis::new(&view, &tree)
        }) else {
            break;
        };
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
            let view = CompactResidualView::with_nflow(horizontal, vertical, nrow, ncol, nflow);
            if compact_cycle_cost(&view, &cycle).unwrap_or(0) >= 0 {
                continue;
            }
            if first_cycle.is_none() {
                first_cycle = Some(cycle.clone());
            }
            apply_compact_residual_cycle_with_nflow(
                horizontal, vertical, nrow, ncol, &cycle, nflow,
            );
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
            let view = CompactResidualView::with_nflow(horizontal, vertical, nrow, ncol, nflow);
            if !pivot_compact_tree_on_cycle(&view, &mut tree, &cycle) {
                break;
            }
        } else if !pivot_compact_tree_on_cycle_fast(&mut tree, &cycle) {
            break;
        }
    }
    applied
}

fn compact_cycle_cost(view: &CompactResidualView<'_>, cycle: &[usize]) -> Option<i64> {
    cycle.iter().try_fold(0_i64, |cost, &index| {
        Some(cost + i64::from(view.arc(index)?.cost))
    })
}
