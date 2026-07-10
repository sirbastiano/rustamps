use crate::stage6_native::{defo_edge_cost, horizontal_index, vertical_index, EdgeDatum};
use crate::stage6_residual::{build_unit_residual_arcs, find_negative_unit_cycle};
use crate::stage6_residual_view::CompactResidualView;
use crate::stage6_residue::edge_residues;
use crate::stage6_tree_cycle::{
    find_negative_tree_cycle_compact, optimize_tree_cycles_compact,
    relax_compact_tree_by_reduced_cost, relax_compact_tree_by_reduced_cost_candidates,
    spanning_tree_arc_indices_compact, CompactTreeBasis,
};

fn edge(flow_sign: i32, flow: i32) -> EdgeDatum {
    EdgeDatum {
        cost: 1000,
        desired_delta: 0.0,
        offset: 0,
        dzmax: 32000,
        laycost: -32000,
        nshortcycle: 200,
        flow_sign,
        flow,
    }
}

fn objective(horizontal: &[Option<EdgeDatum>], vertical: &[Option<EdgeDatum>]) -> i64 {
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

fn negative_first_tree(view: &CompactResidualView<'_>) -> Vec<usize> {
    let node_count = view.node_count();
    let mut parent = (0..node_count).collect::<Vec<_>>();
    fn find(parent: &mut [usize], node: usize) -> usize {
        if parent[node] != node {
            parent[node] = find(parent, parent[node]);
        }
        parent[node]
    }
    fn union(parent: &mut [usize], left: usize, right: usize) -> bool {
        let lroot = find(parent, left);
        let rroot = find(parent, right);
        if lroot == rroot {
            return false;
        }
        parent[rroot] = lroot;
        true
    }

    let mut tree = Vec::new();
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
        let Some(arc) = view.arc(index) else {
            continue;
        };
        if union(&mut parent, arc.from, arc.to) {
            tree.push(index);
        }
    }
    for index in 0..view.arc_count() {
        let Some(arc) = view.arc(index) else {
            continue;
        };
        if union(&mut parent, arc.from, arc.to) {
            tree.push(index);
            if tree.len() + 1 == node_count {
                break;
            }
        }
    }
    tree
}

#[test]
fn large_tree_optimizer_reduces_cycle_missed_by_retained_tree_bases() {
    let flows = [3, 3, 0, 0, -3, -2, -1, -1, 2, -1, -2, -1, 0];
    let small_ncol = 5;
    let mut small_horizontal = vec![Some(edge(1, 0)); 2 * (small_ncol - 1)];
    let mut small_vertical = vec![Some(edge(-1, 0)); small_ncol];
    for row in 0..2 {
        for col in 0..(small_ncol - 1) {
            small_horizontal[horizontal_index(row, col, small_ncol)] =
                Some(edge(1, flows[row * (small_ncol - 1) + col]));
        }
    }
    for col in 0..small_ncol {
        small_vertical[vertical_index(0, col, small_ncol)] =
            Some(edge(-1, flows[2 * (small_ncol - 1) + col]));
    }
    let small_view = CompactResidualView::new(&small_horizontal, &small_vertical, 2, small_ncol);
    let small_arcs = build_unit_residual_arcs(&small_horizontal, &small_vertical, 2, small_ncol);
    assert!(find_negative_unit_cycle(&small_arcs, small_view.node_count()).is_some());

    let forward_tree = spanning_tree_arc_indices_compact(&small_view);
    let negative_tree = negative_first_tree(&small_view);
    assert!(find_negative_tree_cycle_compact(&small_view, &forward_tree).is_none());
    assert!(CompactTreeBasis::new(&small_view, &negative_tree)
        .and_then(|basis| basis.find_negative_cycle(&small_view))
        .is_none());

    let nrow = 2;
    let ncol = 9000;
    let mut horizontal = vec![Some(edge(1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(-1, 0)); ncol];
    for row in 0..2 {
        for col in 0..(small_ncol - 1) {
            horizontal[horizontal_index(row, col, ncol)] =
                Some(edge(1, flows[row * (small_ncol - 1) + col]));
        }
    }
    for col in 0..small_ncol {
        vertical[vertical_index(0, col, ncol)] = Some(edge(-1, flows[2 * (small_ncol - 1) + col]));
    }
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);
    let before_objective = objective(&horizontal, &vertical);

    let applied = optimize_tree_cycles_compact(&mut horizontal, &mut vertical, nrow, ncol, 8);

    assert!(applied > 0);
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
    assert!(objective(&horizontal, &vertical) < before_objective);
}

#[test]
fn reduced_cost_remount_exposes_cycle_missed_by_retained_tree_bases() {
    let nrow = 3;
    let ncol = 4;
    let hflows = [-3, -1, 0, 3, 0, -1, -1, 0, 1];
    let vflows = [-3, 2, 2, 1, -1, 1, -1, 2];
    let mut horizontal = vec![Some(edge(1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(-1, 0)); (nrow - 1) * ncol];
    for (index, flow) in hflows.into_iter().enumerate() {
        let row = index / (ncol - 1);
        let col = index % (ncol - 1);
        horizontal[horizontal_index(row, col, ncol)] = Some(edge(1, flow));
    }
    for (index, flow) in vflows.into_iter().enumerate() {
        let row = index / ncol;
        let col = index % ncol;
        vertical[vertical_index(row, col, ncol)] = Some(edge(-1, flow));
    }
    let view = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    assert!(find_negative_unit_cycle(&arcs, view.node_count()).is_some());

    let mut tree = spanning_tree_arc_indices_compact(&view);
    let negative_tree = negative_first_tree(&view);
    assert!(find_negative_tree_cycle_compact(&view, &tree).is_none());
    assert!(CompactTreeBasis::new(&view, &negative_tree)
        .and_then(|basis| basis.find_negative_cycle(&view))
        .is_none());

    let remounts = relax_compact_tree_by_reduced_cost(&view, &mut tree, 8);

    assert!(remounts > 0);
    assert!(find_negative_tree_cycle_compact(&view, &tree).is_some());
}

#[test]
fn candidate_limited_reduced_cost_remount_exposes_same_cycle() {
    let nrow = 3;
    let ncol = 4;
    let hflows = [-3, -1, 0, 3, 0, -1, -1, 0, 1];
    let vflows = [-3, 2, 2, 1, -1, 1, -1, 2];
    let mut horizontal = vec![Some(edge(1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(-1, 0)); (nrow - 1) * ncol];
    for (index, flow) in hflows.into_iter().enumerate() {
        horizontal[horizontal_index(index / (ncol - 1), index % (ncol - 1), ncol)] =
            Some(edge(1, flow));
    }
    for (index, flow) in vflows.into_iter().enumerate() {
        vertical[vertical_index(index / ncol, index % ncol, ncol)] = Some(edge(-1, flow));
    }
    let view = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);
    let mut tree = spanning_tree_arc_indices_compact(&view);
    assert!(find_negative_tree_cycle_compact(&view, &tree).is_none());

    let candidates = (0..view.arc_count())
        .filter(|&index| view.arc(index).is_some_and(|arc| arc.cost < 0))
        .collect::<Vec<_>>();
    let remounts = relax_compact_tree_by_reduced_cost_candidates(&view, &mut tree, &candidates, 8);

    assert!(remounts > 0);
    assert!(find_negative_tree_cycle_compact(&view, &tree).is_some());
}
