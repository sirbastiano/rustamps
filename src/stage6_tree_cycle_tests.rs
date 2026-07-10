use crate::stage6_native::{defo_edge_cost, horizontal_index, vertical_index, EdgeDatum};
use crate::stage6_residual::{apply_residual_cycle, build_unit_residual_arcs, residual_cycle_cost};
use crate::stage6_residual_view::{apply_compact_residual_cycle, CompactResidualView};
use crate::stage6_residue::edge_residues;
use crate::stage6_tree_cycle::{
    cancel_negative_tree_cycles_with_tree, compact_tree_edge_mask, find_negative_tree_cycle,
    find_negative_tree_cycle_compact, find_negative_tree_cycle_fast, optimize_tree_cycles,
    spanning_tree_arc_indices, tree_cycle_for_arc, tree_edge_mask,
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

fn arc_index(arcs: &[crate::stage6_residual::ResidualArc], from: usize, to: usize) -> usize {
    arcs.iter()
        .position(|arc| arc.from == from && arc.to == to)
        .unwrap()
}

#[test]
fn tree_cycle_for_arc_uses_tree_path_back_to_source() {
    let nrow = 3;
    let ncol = 3;
    let horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    let tree = [
        arc_index(&arcs, 0, 1),
        arc_index(&arcs, 1, 3),
        arc_index(&arcs, 2, 3),
        arc_index(&arcs, 4, 0),
    ];
    let non_tree = arc_index(&arcs, 2, 0);

    let cycle = tree_cycle_for_arc(&arcs, 5, &tree, non_tree).unwrap();
    let endpoints: Vec<(usize, usize)> = cycle
        .iter()
        .map(|&index| (arcs[index].from, arcs[index].to))
        .collect();

    assert_eq!(endpoints, vec![(2, 0), (0, 1), (1, 3), (3, 2)]);
}

#[test]
fn negative_tree_cycle_reduces_objective_and_preserves_residues() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(1000, 1, 1));
    vertical[vertical_index(0, 1, ncol)] = Some(edge(32000, -1, 0));
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);
    let before_objective = objective(&horizontal, &vertical);
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    let tree = [
        arc_index(&arcs, 0, 1),
        arc_index(&arcs, 1, 3),
        arc_index(&arcs, 2, 3),
        arc_index(&arcs, 4, 0),
    ];

    let cycle = find_negative_tree_cycle(&arcs, 5, &tree).unwrap();

    assert!(residual_cycle_cost(&arcs, &cycle) < 0);
    apply_residual_cycle(&mut horizontal, &mut vertical, &arcs, &cycle);
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
    assert!(objective(&horizontal, &vertical) < before_objective);
}

#[test]
fn fast_negative_tree_cycle_matches_tree_cycle_cost() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(1000, 1, 1));
    vertical[vertical_index(0, 1, ncol)] = Some(edge(32000, -1, 0));
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    let tree = [
        arc_index(&arcs, 0, 1),
        arc_index(&arcs, 1, 3),
        arc_index(&arcs, 2, 3),
        arc_index(&arcs, 4, 0),
    ];

    let cycle = find_negative_tree_cycle_fast(&arcs, 5, &tree).unwrap();

    assert!(residual_cycle_cost(&arcs, &cycle) < 0);
}

#[test]
fn spanning_tree_arc_indices_connects_all_dual_nodes() {
    let nrow = 3;
    let ncol = 3;
    let horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);

    let tree = spanning_tree_arc_indices(&arcs, 5);

    assert_eq!(tree.len(), 4);
}

#[test]
fn tree_edge_mask_marks_both_residual_directions() {
    let nrow = 3;
    let ncol = 3;
    let horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    let tree = [arc_index(&arcs, 0, 1)];

    let mask = tree_edge_mask(&arcs, &tree);

    assert!(mask[arc_index(&arcs, 0, 1)]);
    assert!(mask[arc_index(&arcs, 1, 0)]);
    assert!(!mask[arc_index(&arcs, 2, 0)]);
}

#[test]
fn compact_tree_edge_mask_matches_materialized_mask() {
    let nrow = 3;
    let ncol = 3;
    let horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    let compact = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);
    let tree = [
        arc_index(&arcs, 0, 1),
        arc_index(&arcs, 1, 3),
        arc_index(&arcs, 2, 3),
        arc_index(&arcs, 4, 0),
    ];

    assert_eq!(
        compact_tree_edge_mask(&compact, &tree),
        tree_edge_mask(&arcs, &tree)
    );
}

#[test]
fn compact_negative_tree_cycle_matches_materialized_cost() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(1000, 1, 1));
    vertical[vertical_index(0, 1, ncol)] = Some(edge(32000, -1, 0));
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    let compact = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);
    let tree = [
        arc_index(&arcs, 0, 1),
        arc_index(&arcs, 1, 3),
        arc_index(&arcs, 2, 3),
        arc_index(&arcs, 4, 0),
    ];

    let compact_cycle = find_negative_tree_cycle_compact(&compact, &tree).unwrap();

    let cost: i32 = compact_cycle
        .iter()
        .map(|&index| compact.arc(index).unwrap().cost)
        .sum();
    assert!(cost < 0);
}

#[test]
fn compact_cycle_application_reduces_objective_and_preserves_residues() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(1000, 1, 1));
    vertical[vertical_index(0, 1, ncol)] = Some(edge(32000, -1, 0));
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);
    let before_objective = objective(&horizontal, &vertical);
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    let tree = [
        arc_index(&arcs, 0, 1),
        arc_index(&arcs, 1, 3),
        arc_index(&arcs, 2, 3),
        arc_index(&arcs, 4, 0),
    ];
    let compact = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);
    let compact_cycle = find_negative_tree_cycle_compact(&compact, &tree).unwrap();

    apply_compact_residual_cycle(&mut horizontal, &mut vertical, nrow, ncol, &compact_cycle);

    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
    assert!(objective(&horizontal, &vertical) < before_objective);
}

#[test]
fn tree_cycle_cancellation_stops_at_tree_local_optimum() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(1000, 1, 1));
    let before = objective(&horizontal, &vertical);
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    let tree = [
        arc_index(&arcs, 0, 1),
        arc_index(&arcs, 1, 3),
        arc_index(&arcs, 2, 3),
        arc_index(&arcs, 4, 0),
    ];

    let applied =
        cancel_negative_tree_cycles_with_tree(&mut horizontal, &mut vertical, nrow, ncol, &tree, 8);
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);

    assert_eq!(applied, 1);
    assert!(objective(&horizontal, &vertical) < before);
    assert!(find_negative_tree_cycle(&arcs, 5, &tree).is_none());
}

#[test]
fn optimize_tree_cycles_builds_tree_and_reduces_objective() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(1000, 1, 1));
    let before = objective(&horizontal, &vertical);

    let applied = optimize_tree_cycles(&mut horizontal, &mut vertical, nrow, ncol, 8);

    assert_eq!(applied, 1);
    assert!(objective(&horizontal, &vertical) < before);
}
