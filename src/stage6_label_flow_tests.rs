use crate::stage6_label_flow::absorb_label_corrections;
use crate::stage6_native::{
    defo_edge_cost, edge_label_energy, horizontal_index, reseed_labels_from_edge_deltas,
    vertical_index, EdgeDatum,
};
use crate::stage6_residue::edge_residues;

fn edge(cost: i32, offset: i32, flow_sign: i32, flow: i32) -> EdgeDatum {
    EdgeDatum {
        cost,
        desired_delta: 0.0,
        offset,
        dzmax: 32000,
        laycost: -32000,
        nshortcycle: 200,
        flow_sign,
        flow,
    }
}

fn flow_objective(horizontal: &[Option<EdgeDatum>], vertical: &[Option<EdgeDatum>]) -> i64 {
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
fn label_correction_absorption_preserves_residues_and_label_objective() {
    let nrow = 2;
    let ncol = 2;
    let labels = vec![0, 2, -1, 1];
    let mut horizontal = vec![Some(edge(7, 100, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(11, -200, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(1, 0, ncol)] = Some(edge(13, 300, 1, 1));
    vertical[vertical_index(0, 1, ncol)] = Some(edge(17, -400, -1, -2));
    let before_residue = edge_residues(&horizontal, &vertical, nrow, ncol);
    let label_objective = edge_label_energy(horizontal[0].unwrap(), labels[0], labels[1])
        + edge_label_energy(horizontal[1].unwrap(), labels[2], labels[3])
        + edge_label_energy(vertical[0].unwrap(), labels[0], labels[2])
        + edge_label_energy(vertical[1].unwrap(), labels[1], labels[3]);

    let changed = absorb_label_corrections(&labels, &mut horizontal, &mut vertical, nrow, ncol);

    assert!(changed > 0);
    assert_eq!(
        edge_residues(&horizontal, &vertical, nrow, ncol),
        before_residue
    );
    assert_eq!(flow_objective(&horizontal, &vertical), label_objective);
}

#[test]
fn label_reseed_follows_grid_edge_delta_orientation() {
    let nrow = 2;
    let ncol = 2;
    let mut labels = vec![5, 5, 5, 5];
    let mut horizontal = vec![Some(edge(7, 0, 1, 0)); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge(11, 0, -1, 0)); (nrow - 1) * ncol];
    horizontal[horizontal_index(0, 0, ncol)]
        .as_mut()
        .unwrap()
        .desired_delta = 1.0;
    horizontal[horizontal_index(1, 0, ncol)]
        .as_mut()
        .unwrap()
        .desired_delta = 1.0;
    vertical[vertical_index(0, 0, ncol)]
        .as_mut()
        .unwrap()
        .desired_delta = 2.0;
    vertical[vertical_index(0, 1, ncol)]
        .as_mut()
        .unwrap()
        .desired_delta = 2.0;

    reseed_labels_from_edge_deltas(&mut labels, &horizontal, &vertical, nrow, ncol);

    assert_eq!(labels, [5, 6, 7, 8]);
}
