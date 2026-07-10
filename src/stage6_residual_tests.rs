use crate::stage6_native::{defo_edge_cost, horizontal_index, vertical_index, EdgeDatum};
use crate::stage6_residual::{
    apply_residual_cycle, build_unit_residual_arcs, cancel_negative_unit_cycles,
    find_negative_unit_cycle, residual_cycle_cost,
};
use crate::stage6_residue::edge_residues;

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
fn residual_cycle_application_preserves_residues() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    let cycle = [
        arcs.iter()
            .position(|arc| arc.from == 0 && arc.to == 1)
            .unwrap(),
        arcs.iter()
            .position(|arc| arc.from == 1 && arc.to == 3)
            .unwrap(),
        arcs.iter()
            .position(|arc| arc.from == 3 && arc.to == 2)
            .unwrap(),
        arcs.iter()
            .position(|arc| arc.from == 2 && arc.to == 0)
            .unwrap(),
    ];
    let before = edge_residues(&horizontal, &vertical, nrow, ncol);

    apply_residual_cycle(&mut horizontal, &mut vertical, &arcs, &cycle);

    assert_eq!(edge_residues(&horizontal, &vertical, nrow, ncol), before);
}

#[test]
fn negative_unit_cycle_is_absent_when_all_increments_are_nonnegative() {
    let nrow = 3;
    let ncol = 3;
    let horizontal = vec![Some(edge(1000, 1, 0)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge(1000, -1, 0)); (nrow - 1) * ncol];
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);

    assert!(find_negative_unit_cycle(&arcs, (nrow - 1) * (ncol - 1) + 1).is_none());
}

#[test]
fn negative_unit_cycle_reduces_defo_objective() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(1000, 1, 1));
    vertical[vertical_index(0, 1, ncol)] = Some(edge(32000, -1, 0));
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    let cycle = find_negative_unit_cycle(&arcs, (nrow - 1) * (ncol - 1) + 1)
        .expect("expected a cost-reducing residual cycle");
    let before = objective(&horizontal, &vertical);

    apply_residual_cycle(&mut horizontal, &mut vertical, &arcs, &cycle);

    assert!(residual_cycle_cost(&arcs, &cycle) < 0);
    assert!(objective(&horizontal, &vertical) < before);
}

#[test]
fn bounded_cycle_cancellation_stops_at_local_unit_optimum() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(1000, 1, 1));
    let before = objective(&horizontal, &vertical);

    let iterations = cancel_negative_unit_cycles(&mut horizontal, &mut vertical, nrow, ncol, 8);
    let arcs = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);

    assert_eq!(iterations, 1);
    assert!(objective(&horizontal, &vertical) < before);
    assert!(find_negative_unit_cycle(&arcs, (nrow - 1) * (ncol - 1) + 1).is_none());
}
