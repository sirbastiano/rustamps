use crate::stage6_native::{
    defo_edge_cost, horizontal_index, optimize_edge_flows, optimize_edge_flows_with_parallel,
    snaphu_capped_batch_limit, snaphu_continue_capped_batches, snaphu_flow_increments,
    snaphu_flow_tree_cycle_limit, snaphu_max_nflow_cycles, EdgeDatum,
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

fn edge_with_offset(cost: i32, offset: i32, dzmax: i32, laycost: i32, flow: i32) -> EdgeDatum {
    EdgeDatum {
        cost,
        desired_delta: 0.0,
        offset,
        dzmax,
        laycost,
        nshortcycle: 200,
        flow_sign: 1,
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
fn stage6_flow_increment_schedule_matches_snaphu_default() {
    assert_eq!(snaphu_flow_increments(0), vec![1]);
    assert_eq!(snaphu_flow_increments(2), vec![1, 2]);
    assert_eq!(snaphu_flow_increments(3), vec![1, 2, 3]);
    assert_eq!(snaphu_flow_increments(9), vec![1, 2, 3, 4]);
}

#[test]
fn stage6_flow_cycle_threshold_matches_snaphu_default_fraction() {
    assert_eq!(snaphu_max_nflow_cycles(3, 3), 1);
    assert_eq!(snaphu_max_nflow_cycles(1773, 4378), 311);
}

#[test]
fn stage6_flow_tree_cycle_limit_allows_one_default_maxflow_sweep() {
    assert_eq!(snaphu_flow_tree_cycle_limit(3, 3), 28);
    assert_eq!(snaphu_flow_tree_cycle_limit(1773, 4378), 1244);
}

#[test]
fn stage6_capped_batch_continuation_is_bounded_to_medium_grids() {
    assert!(snaphu_continue_capped_batches(2, 141));
    assert!(!snaphu_continue_capped_batches(1773, 4378));
    assert_eq!(snaphu_capped_batch_limit(2, 141), usize::MAX);
    assert_eq!(snaphu_capped_batch_limit(1773, 4378), 3);
}

#[test]
fn stage6_edge_flow_optimizer_runs_compact_tree_cycles() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(1000, 1, 1));
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);
    let before_objective = objective(&horizontal, &vertical);

    let applied = optimize_edge_flows(&mut horizontal, &mut vertical, nrow, ncol);

    assert!(applied > 0);
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
    assert!(objective(&horizontal, &vertical) < before_objective);
}

#[test]
fn stage6_edge_flow_optimizer_tries_larger_flow_steps() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(32000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(32000, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge_with_offset(1, -1000, 1000, 1, 7));
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);
    let before_objective = objective(&horizontal, &vertical);

    let applied = optimize_edge_flows(&mut horizontal, &mut vertical, nrow, ncol);

    assert!(applied > 0);
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
    assert!(objective(&horizontal, &vertical) < before_objective);
}

#[test]
fn stage6_edge_flow_optimizer_runs_more_than_eight_unit_cycles() {
    let nrow = 2;
    let ncol = 41;
    let mut horizontal = vec![Some(edge(1000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(1000, -1, 0)); (nrow - 1) * ncol];
    for block in 0..10 {
        let col = block * 4;
        horizontal[horizontal_index(0, col, ncol)] = Some(edge(1000, 1, -1));
        horizontal[horizontal_index(1, col + 1, ncol)] = Some(edge(1000, 1, -1));
    }
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);
    let before_objective = objective(&horizontal, &vertical);

    let applied = optimize_edge_flows(&mut horizontal, &mut vertical, nrow, ncol);

    assert!(applied > 8);
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
    assert!(objective(&horizontal, &vertical) < before_objective);
}

#[test]
fn stage6_edge_flow_optimizer_continues_after_cycle_batch_limit() {
    let nrow = 2;
    let ncol = 141;
    let mut horizontal = vec![Some(edge(1000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(1000, -1, 0)); (nrow - 1) * ncol];
    for block in 0..35 {
        let col = block * 4;
        horizontal[horizontal_index(0, col, ncol)] = Some(edge(1000, 1, -1));
        horizontal[horizontal_index(1, col + 1, ncol)] = Some(edge(1000, 1, -1));
    }
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);

    let applied = optimize_edge_flows(&mut horizontal, &mut vertical, nrow, ncol);

    assert!(applied > snaphu_flow_tree_cycle_limit(nrow, ncol));
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
}

#[test]
fn stage6_edge_flow_optimizer_allows_bounded_large_grid_continuation() {
    let nrow = 2;
    let ncol = 9000;
    let mut horizontal = vec![Some(edge(1000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(1000, -1, 0)); (nrow - 1) * ncol];
    for block in 0..35 {
        let col = block * 4;
        horizontal[horizontal_index(0, col, ncol)] = Some(edge(1000, 1, -1));
        horizontal[horizontal_index(1, col + 1, ncol)] = Some(edge(1000, 1, -1));
    }
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);

    let applied = optimize_edge_flows(&mut horizontal, &mut vertical, nrow, ncol);

    assert!(applied > snaphu_flow_tree_cycle_limit(nrow, ncol));
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
}

#[test]
fn stage6_edge_flow_parallel_path_matches_serial_objective() {
    let nrow = 2;
    let ncol = 13;
    let mut horizontal = vec![Some(edge(1000, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(1000, -1, 0)); (nrow - 1) * ncol];
    for block in 0..3 {
        let col = block * 4;
        horizontal[horizontal_index(0, col, ncol)] = Some(edge(1000, 1, -1));
        horizontal[horizontal_index(1, col + 1, ncol)] = Some(edge(1000, 1, -1));
    }
    let mut serial_h = horizontal.clone();
    let mut serial_v = vertical.clone();
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);

    let serial_applied = optimize_edge_flows(&mut serial_h, &mut serial_v, nrow, ncol);
    let parallel_applied =
        optimize_edge_flows_with_parallel(&mut horizontal, &mut vertical, nrow, ncol, true);

    assert_eq!(parallel_applied, serial_applied);
    assert_eq!(
        objective(&horizontal, &vertical),
        objective(&serial_h, &serial_v)
    );
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
}
