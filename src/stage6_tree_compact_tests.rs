use crate::stage6_native::{defo_edge_cost, horizontal_index, EdgeDatum};
use crate::stage6_residual::{build_unit_residual_arcs, find_negative_unit_cycle};
use crate::stage6_residual_view::{apply_compact_residual_cycle_with_nflow, CompactResidualView};
use crate::stage6_residue::edge_residues;
use crate::stage6_tree_cycle::{
    find_negative_tree_cycle_compact, optimize_tree_cycles_compact,
    optimize_tree_cycles_compact_with_nflow, pivot_compact_tree_on_cycle,
    spanning_tree_arc_indices, spanning_tree_arc_indices_compact, CompactTreeBasis,
};

fn edge(cost: i32, flow_sign: i32, flow: i32) -> EdgeDatum {
    EdgeDatum {
        cost,
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

#[test]
fn compact_spanning_tree_matches_materialized_tree() {
    let nrow = 3;
    let ncol = 3;
    let horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    let view = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);

    assert_eq!(
        spanning_tree_arc_indices_compact(&view),
        spanning_tree_arc_indices(&arcs, view.node_count())
    );
}

#[test]
fn compact_optimizer_builds_tree_and_reduces_objective() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(1000, 1, 1));
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);
    let before_objective = objective(&horizontal, &vertical);

    let applied = optimize_tree_cycles_compact(&mut horizontal, &mut vertical, nrow, ncol, 8);

    assert_eq!(applied, 1);
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
    assert!(objective(&horizontal, &vertical) < before_objective);
}

#[test]
fn compact_optimizer_can_apply_larger_flow_steps() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(1000, 1, 3));
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);
    let before_objective = objective(&horizontal, &vertical);

    let applied =
        optimize_tree_cycles_compact_with_nflow(&mut horizontal, &mut vertical, nrow, ncol, 2, 8);

    assert_eq!(applied, 1);
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
    assert!(objective(&horizontal, &vertical) < before_objective);
}

#[test]
fn compact_tree_pivot_replaces_tree_arc_with_entering_cycle_arc() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(1000, 1, 1));
    let view = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);
    let mut tree = spanning_tree_arc_indices_compact(&view);
    let cycle = find_negative_tree_cycle_compact(&view, &tree).expect("negative cycle");
    let entering_pair = cycle[0] / 2;

    assert!(pivot_compact_tree_on_cycle(&view, &mut tree, &cycle));

    assert_eq!(tree.len() + 1, view.node_count());
    assert!(tree.iter().any(|&index| index / 2 == entering_pair));
    assert!(CompactTreeBasis::new(&view, &tree).is_some());
}

#[test]
fn compact_optimizer_pivots_tree_basis_between_cycles() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = [-2, 1, 3, 3, 3, -3]
        .into_iter()
        .map(|flow| Some(edge(1000, 1, flow)))
        .collect::<Vec<_>>();
    let mut vertical = [-1, -3, 0, 3, 0, 0]
        .into_iter()
        .map(|flow| Some(edge(1000, -1, flow)))
        .collect::<Vec<_>>();
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);

    let applied = optimize_tree_cycles_compact(&mut horizontal, &mut vertical, nrow, ncol, 8);

    assert_eq!(applied, 8);
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
    assert!(objective(&horizontal, &vertical) <= 1000);
}

#[test]
fn compact_optimizer_pivots_large_tree_basis_between_batches() {
    let nrow = 3;
    let ncol = 5000;
    let mut horizontal = vec![Some(edge(1000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(1000, -1, 0)); (nrow - 1) * ncol];
    for (index, flow) in [-2, 1, 3, 3, 3, -3].into_iter().enumerate() {
        let row = index / 2;
        let col = index % 2;
        horizontal[horizontal_index(row, col, ncol)] = Some(edge(1000, 1, flow));
    }
    for (index, flow) in [-1, -3, 0, 3, 0, 0].into_iter().enumerate() {
        vertical[index] = Some(edge(1000, -1, flow));
    }
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);

    let applied = optimize_tree_cycles_compact(&mut horizontal, &mut vertical, nrow, ncol, 8);

    assert_eq!(applied, 8);
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
    assert!(objective(&horizontal, &vertical) <= 1000);
}

#[test]
fn compact_tree_basis_returns_candidate_batch_from_one_scan() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = [-3, 1, -1, 3, -2, 0]
        .into_iter()
        .map(|flow| Some(edge(1000, 1, flow)))
        .collect::<Vec<_>>();
    let mut vertical = [2, 2, 3, 2, -1, 1]
        .into_iter()
        .map(|flow| Some(edge(1000, -1, flow)))
        .collect::<Vec<_>>();
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);
    let before_objective = objective(&horizontal, &vertical);
    let view = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);
    let tree = spanning_tree_arc_indices_compact(&view);
    let basis = CompactTreeBasis::new(&view, &tree).expect("connected test tree");
    let cycles = basis.negative_cycles(&view, 3, false);

    assert_eq!(cycles.len(), 3);
    for cycle in &cycles {
        apply_compact_residual_cycle_with_nflow(
            &mut horizontal,
            &mut vertical,
            nrow,
            ncol,
            cycle,
            1,
        );
    }
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
    assert!(objective(&horizontal, &vertical) < before_objective);
}

#[test]
fn compact_optimizer_falls_back_when_fixed_tree_misses_negative_cycle() {
    let nrow = 2;
    let ncol = 4;
    let mut horizontal = vec![Some(edge(1000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(1000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(0, 0, ncol)] = Some(edge(1000, 1, -1));
    horizontal[horizontal_index(1, 1, ncol)] = Some(edge(1000, 1, -1));
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);
    let before_objective = objective(&horizontal, &vertical);
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    let view = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);
    let tree = spanning_tree_arc_indices_compact(&view);

    assert!(find_negative_tree_cycle_compact(&view, &tree).is_none());
    assert!(find_negative_unit_cycle(&arcs, view.node_count()).is_some());

    let applied = optimize_tree_cycles_compact(&mut horizontal, &mut vertical, nrow, ncol, 8);

    assert!(applied > 0);
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
    assert!(objective(&horizontal, &vertical) < before_objective);
}

#[test]
fn compact_optimizer_falls_back_for_larger_flow_steps() {
    let nrow = 2;
    let ncol = 4;
    let mut horizontal = vec![Some(edge(1000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(1000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(0, 0, ncol)] = Some(edge(1000, 1, -2));
    horizontal[horizontal_index(1, 1, ncol)] = Some(edge(1000, 1, -2));
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);
    let before_objective = objective(&horizontal, &vertical);
    let view = CompactResidualView::with_nflow(&horizontal, &vertical, nrow, ncol, 2);
    let tree = spanning_tree_arc_indices_compact(&view);

    assert!(find_negative_tree_cycle_compact(&view, &tree).is_none());

    let applied =
        optimize_tree_cycles_compact_with_nflow(&mut horizontal, &mut vertical, nrow, ncol, 2, 8);

    assert!(applied > 0);
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
    assert!(objective(&horizontal, &vertical) < before_objective);
}

#[test]
fn compact_optimizer_tries_alternate_tree_on_large_graph() {
    let nrow = 2;
    let ncol = 9000;
    let mut horizontal = vec![Some(edge(1000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(1000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(0, 0, ncol)] = Some(edge(1000, 1, -1));
    horizontal[horizontal_index(1, 1, ncol)] = Some(edge(1000, 1, -1));
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
fn cached_compact_tree_basis_matches_uncached_cycle_search_after_cost_change() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(1000, 1, 1));
    let view = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);
    let tree = spanning_tree_arc_indices_compact(&view);
    let basis = CompactTreeBasis::new(&view, &tree).expect("connected test tree");

    assert_eq!(
        basis.find_negative_cycle(&view).is_some(),
        find_negative_tree_cycle_compact(&view, &tree).is_some()
    );

    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(1000, 1, -1));
    let changed_view = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);
    assert_eq!(
        basis.find_negative_cycle(&changed_view).is_some(),
        find_negative_tree_cycle_compact(&changed_view, &tree).is_some()
    );
}
