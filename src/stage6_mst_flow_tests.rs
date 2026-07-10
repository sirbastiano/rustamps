use crate::stage6_mst_flow::{mst_initial_flows, shortest_path_initial_flows};
use crate::stage6_native::{horizontal_index, vertical_index, EdgeDatum};

fn edge(cost: i32) -> EdgeDatum {
    EdgeDatum {
        cost,
        desired_delta: 0.0,
        offset: 0,
        dzmax: 32000,
        laycost: -32000,
        nshortcycle: 200,
        flow_sign: 1,
        flow: 0,
    }
}

#[test]
fn mst_initial_flows_pair_adjacent_opposite_residues() {
    let nrow = 2;
    let ncol = 3;
    let horizontal = vec![Some(edge(1000)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge(1000)); (nrow - 1) * ncol];

    let (rowflow, colflow) = mst_initial_flows(&[1, -1], &horizontal, &vertical, nrow, ncol);

    assert_eq!(rowflow, vec![0, 1, 0]);
    assert_eq!(colflow, vec![0, 0, 0, 0]);
}

#[test]
fn mst_initial_flows_routes_single_residue_to_cheapest_boundary() {
    let nrow = 2;
    let ncol = 2;
    let mut horizontal = vec![Some(edge(1)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(1)); (nrow - 1) * ncol];
    horizontal[horizontal_index(0, 0, ncol)] = Some(edge(1));
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(1));
    vertical[vertical_index(0, 0, ncol)] = Some(edge(1000));
    vertical[vertical_index(0, 1, ncol)] = Some(edge(1));

    let (rowflow, colflow) = mst_initial_flows(&[1], &horizontal, &vertical, nrow, ncol);

    assert_eq!(rowflow, vec![-1, 0]);
    assert_eq!(colflow, vec![0, 0]);
}

#[test]
fn mst_initial_flows_returns_zero_for_balanced_empty_residue_grid() {
    let nrow = 2;
    let ncol = 2;
    let horizontal = vec![Some(edge(1000)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge(1000)); (nrow - 1) * ncol];

    let (rowflow, colflow) = mst_initial_flows(&[0], &horizontal, &vertical, nrow, ncol);

    assert_eq!(rowflow, vec![0, 0]);
    assert_eq!(colflow, vec![0, 0]);
}

#[test]
fn shortest_path_initial_flows_matches_simple_adjacent_residue_contract() {
    let nrow = 2;
    let ncol = 3;
    let horizontal = vec![Some(edge(1000)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge(1000)); (nrow - 1) * ncol];

    let (rowflow, colflow) =
        shortest_path_initial_flows(&[1, -1], &horizontal, &vertical, nrow, ncol);

    assert_eq!(rowflow, vec![0, 1, 0]);
    assert_eq!(colflow, vec![0, 0, 0, 0]);
}
