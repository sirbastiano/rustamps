use crate::stage6_local_cycles::cancel_local_negative_cycles_with_nflow;
use crate::stage6_native::EdgeDatum;
use crate::stage6_residual::cancel_negative_cycles_with_nflow;
use crate::stage6_residual_view::{apply_compact_residual_cycle_with_nflow, CompactResidualView};

#[path = "stage6_compact_dsu.rs"]
mod stage6_compact_dsu;
#[path = "stage6_tree_basis.rs"]
mod stage6_tree_basis;
#[path = "stage6_tree_large.rs"]
mod stage6_tree_large;
#[cfg(test)]
#[path = "stage6_tree_remount.rs"]
mod stage6_tree_remount;
use self::stage6_compact_dsu::CompactDisjointSet;
#[cfg(test)]
pub(crate) use self::stage6_tree_basis::CompactTreeBasis;
use self::stage6_tree_basis::CompactTreeBasis as CompactTreeBasisImpl;
use self::stage6_tree_large::optimize_large_tree_cycles;
#[cfg(test)]
pub(crate) use self::stage6_tree_remount::{
    relax_compact_tree_by_reduced_cost, relax_compact_tree_by_reduced_cost_candidates,
};

const SMALL_GRAPH_EXACT_CYCLE_NODES: usize = 4096;
const TREE_ORDERS: [ArcOrder; 2] = [ArcOrder::Forward, ArcOrder::NegativeFirst];

#[derive(Clone, Copy)]
enum ArcOrder {
    Forward,
    NegativeFirst,
}

pub(crate) fn spanning_tree_arc_indices_compact(view: &CompactResidualView<'_>) -> Vec<usize> {
    spanning_tree_arc_indices_compact_with_order(view, ArcOrder::Forward)
}

fn spanning_tree_arc_indices_compact_with_order(
    view: &CompactResidualView<'_>,
    order: ArcOrder,
) -> Vec<usize> {
    let node_count = view.node_count();
    let mut dsu = CompactDisjointSet::new(node_count);
    let mut tree = Vec::with_capacity(node_count.saturating_sub(1));

    let mut visit = |index| {
        let Some(arc) = view.arc(index) else {
            return false;
        };
        if arc.from >= node_count || arc.to >= node_count || arc.from == arc.to {
            return false;
        }
        if dsu.union(arc.from, arc.to) {
            tree.push(index);
            return tree.len() + 1 == node_count;
        }
        false
    };

    match order {
        ArcOrder::Forward => {
            for index in 0..view.arc_count() {
                if visit(index) {
                    break;
                }
            }
        }
        ArcOrder::NegativeFirst => {
            let mut negative = Vec::new();
            for index in 0..view.arc_count() {
                if let Some(arc) = view.arc(index) {
                    if arc.cost < 0 {
                        negative.push((arc.cost, index));
                    }
                }
            }
            negative.sort_unstable();
            for (_cost, index) in negative {
                if visit(index) {
                    return tree;
                }
            }
            for index in 0..view.arc_count() {
                if visit(index) {
                    break;
                }
            }
        }
    }
    tree
}

pub(crate) fn compact_tree_edge_mask(
    view: &CompactResidualView<'_>,
    tree_arc_indices: &[usize],
) -> Vec<bool> {
    let mut mask = vec![false; view.arc_count()];
    for &index in tree_arc_indices {
        if view.arc(index).is_some() {
            mask[index] = true;
            let reverse = index ^ 1;
            if reverse < mask.len() && view.arc(reverse).is_some() {
                mask[reverse] = true;
            }
        }
    }
    mask
}

pub(crate) fn find_negative_tree_cycle_compact(
    view: &CompactResidualView<'_>,
    tree_arc_indices: &[usize],
) -> Option<Vec<usize>> {
    CompactTreeBasisImpl::new(view, tree_arc_indices)?.find_negative_cycle(view)
}

pub(crate) fn pivot_compact_tree_on_cycle(
    view: &CompactResidualView<'_>,
    tree_arc_indices: &mut [usize],
    cycle: &[usize],
) -> bool {
    let Some(&entering) = cycle.first() else {
        return false;
    };
    if view.arc(entering).is_none() {
        return false;
    }
    if !pivot_compact_tree_on_cycle_fast(tree_arc_indices, cycle) {
        return false;
    }
    CompactTreeBasisImpl::new(view, tree_arc_indices).is_some()
}

pub(super) fn pivot_compact_tree_on_cycle_fast(
    tree_arc_indices: &mut [usize],
    cycle: &[usize],
) -> bool {
    let Some(&entering) = cycle.first() else {
        return false;
    };
    let entering_pair = entering / 2;
    if tree_arc_indices
        .iter()
        .any(|&index| index / 2 == entering_pair)
    {
        return false;
    }
    let Some(remove_pos) = cycle.iter().skip(1).find_map(|&arc_index| {
        let pair = arc_index / 2;
        tree_arc_indices
            .iter()
            .position(|&tree_index| tree_index / 2 == pair)
    }) else {
        return false;
    };
    tree_arc_indices[remove_pos] = entering;
    true
}

pub(crate) fn optimize_tree_cycles_compact(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    max_cycles: usize,
) -> usize {
    optimize_tree_cycles_compact_with_nflow(horizontal, vertical, nrow, ncol, 1, max_cycles)
}

pub(crate) fn optimize_tree_cycles_compact_with_nflow(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    nflow: i32,
    max_cycles: usize,
) -> usize {
    optimize_tree_cycles_compact_with_nflow_parallel(
        horizontal, vertical, nrow, ncol, nflow, max_cycles, false,
    )
}

pub(crate) fn optimize_tree_cycles_compact_with_nflow_parallel(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    nflow: i32,
    max_cycles: usize,
    parallel: bool,
) -> usize {
    let node_count =
        CompactResidualView::with_nflow(horizontal, vertical, nrow, ncol, nflow).node_count();
    let mut applied = 0;
    let orders = if nflow.abs().max(1) == 1 {
        &TREE_ORDERS[..]
    } else {
        &TREE_ORDERS[..1]
    };
    let pivot_tree = node_count <= SMALL_GRAPH_EXACT_CYCLE_NODES;
    let mut connected = false;

    for &order in orders {
        let mut tree = {
            let view = CompactResidualView::with_nflow(horizontal, vertical, nrow, ncol, nflow);
            spanning_tree_arc_indices_compact_with_order(&view, order)
        };
        if tree.len() + 1 != node_count {
            continue;
        }
        connected = true;
        if pivot_tree {
            while applied < max_cycles {
                let Some(basis) = ({
                    let view =
                        CompactResidualView::with_nflow(horizontal, vertical, nrow, ncol, nflow);
                    CompactTreeBasisImpl::new(&view, &tree)
                }) else {
                    break;
                };
                let view = CompactResidualView::with_nflow(horizontal, vertical, nrow, ncol, nflow);
                let cycle = if parallel {
                    basis.find_negative_cycle_parallel(&view)
                } else {
                    basis.find_negative_cycle(&view)
                };
                let Some(cycle) = cycle else {
                    break;
                };
                pivot_compact_tree_on_cycle_fast(&mut tree, &cycle);
                apply_compact_residual_cycle_with_nflow(
                    horizontal, vertical, nrow, ncol, &cycle, nflow,
                );
                applied += 1;
            }
        } else {
            applied += optimize_large_tree_cycles(
                horizontal,
                vertical,
                nrow,
                ncol,
                nflow,
                max_cycles - applied,
                parallel,
                node_count,
                tree,
            );
        }
        if applied == max_cycles {
            break;
        }
    }
    if !connected {
        return 0;
    }
    if node_count <= SMALL_GRAPH_EXACT_CYCLE_NODES && applied < max_cycles {
        applied += cancel_negative_cycles_with_nflow(
            horizontal,
            vertical,
            nrow,
            ncol,
            nflow,
            max_cycles - applied,
        );
    } else if applied < max_cycles {
        applied += cancel_local_negative_cycles_with_nflow(
            horizontal,
            vertical,
            nrow,
            ncol,
            nflow,
            max_cycles - applied,
        );
    }
    applied
}
