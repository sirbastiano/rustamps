use crate::stage6_native::{horizontal_index, vertical_index, EdgeDatum};
use crate::stage6_residual::{build_unit_residual_arcs, residual_cycle_cost};
use crate::stage6_tree_cycle::tree_cycle_for_arc;
use crate::stage6_tree_path::TreePathCosts;

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

fn arc_index(arcs: &[crate::stage6_residual::ResidualArc], from: usize, to: usize) -> usize {
    arcs.iter()
        .position(|arc| arc.from == from && arc.to == to)
        .unwrap()
}

#[test]
fn tree_path_cost_matches_explicit_tree_cycle_path_cost() {
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
    let non_tree = arc_index(&arcs, 2, 0);
    let explicit = tree_cycle_for_arc(&arcs, 5, &tree, non_tree).unwrap();
    let path_cost = residual_cycle_cost(&arcs, &explicit[1..]);

    let fast = TreePathCosts::new(&arcs, 5, &tree).unwrap();

    assert_eq!(
        fast.path_cost(arcs[non_tree].to, arcs[non_tree].from),
        Some(i64::from(path_cost)),
    );
}

#[test]
fn tree_path_cost_handles_reverse_tree_traversal() {
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
    let fast = TreePathCosts::new(&arcs, 5, &tree).unwrap();

    assert_eq!(
        fast.path_cost(3, 0),
        Some(i64::from(
            arcs[arc_index(&arcs, 3, 1)].cost + arcs[arc_index(&arcs, 1, 0)].cost,
        )),
    );
}

#[test]
fn tree_path_cost_rejects_disconnected_tree() {
    let nrow = 3;
    let ncol = 3;
    let horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    let tree = [arc_index(&arcs, 0, 1), arc_index(&arcs, 1, 3)];

    assert!(TreePathCosts::new(&arcs, 5, &tree).is_none());
}

#[test]
fn tree_path_cost_exposes_flat_table_shape() {
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

    let fast = TreePathCosts::new(&arcs, 5, &tree).unwrap();

    assert_eq!(fast.table_shape(), (4, 5));
}

#[test]
fn tree_path_cost_stores_directed_costs_per_node_not_per_level() {
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

    let fast = TreePathCosts::new(&arcs, 5, &tree).unwrap();

    assert_eq!(fast.directed_cost_storage_len(), 10);
}
