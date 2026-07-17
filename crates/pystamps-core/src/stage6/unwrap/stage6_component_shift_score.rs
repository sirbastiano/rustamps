use crate::stage6::unwrap::native::{
    edge_label_energy, horizontal_index, vertical_index, EdgeDatum,
};

pub(super) fn positive_edge_mean_energy(
    labels: &[i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) -> i64 {
    let mut total = 0_i64;
    let mut count = 0_i64;
    for row in 0..nrow {
        for col in 0..ncol.saturating_sub(1) {
            if let Some(edge) = horizontal[horizontal_index(row, col, ncol)] {
                let energy =
                    edge_label_energy(edge, labels[row * ncol + col], labels[row * ncol + col + 1]);
                if energy > 0 {
                    total += energy;
                    count += 1;
                }
            }
        }
    }
    for row in 0..nrow.saturating_sub(1) {
        for col in 0..ncol {
            if let Some(edge) = vertical[vertical_index(row, col, ncol)] {
                let energy = edge_label_energy(
                    edge,
                    labels[row * ncol + col],
                    labels[(row + 1) * ncol + col],
                );
                if energy > 0 {
                    total += energy;
                    count += 1;
                }
            }
        }
    }
    if count == 0 {
        0
    } else {
        (total / count).max(1)
    }
}

pub(super) fn component_shift_gain(
    labels: &[i32],
    component: &[usize],
    mark: &[bool],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    shift: i32,
) -> i64 {
    let mut gain = 0_i64;
    for &node in component {
        let row = node / ncol;
        let col = node - row * ncol;
        let shifted = labels[node] + shift;
        if col + 1 < ncol && !mark[node + 1] {
            if let Some(edge) = horizontal[horizontal_index(row, col, ncol)] {
                gain += edge_label_energy(edge, labels[node], labels[node + 1])
                    - edge_label_energy(edge, shifted, labels[node + 1]);
            }
        }
        if col > 0 && !mark[node - 1] {
            if let Some(edge) = horizontal[horizontal_index(row, col - 1, ncol)] {
                gain += edge_label_energy(edge, labels[node - 1], labels[node])
                    - edge_label_energy(edge, labels[node - 1], shifted);
            }
        }
        if row + 1 < nrow && !mark[node + ncol] {
            if let Some(edge) = vertical[vertical_index(row, col, ncol)] {
                gain += edge_label_energy(edge, labels[node], labels[node + ncol])
                    - edge_label_energy(edge, shifted, labels[node + ncol]);
            }
        }
        if row > 0 && !mark[node - ncol] {
            if let Some(edge) = vertical[vertical_index(row - 1, col, ncol)] {
                gain += edge_label_energy(edge, labels[node - ncol], labels[node])
                    - edge_label_energy(edge, labels[node - ncol], shifted);
            }
        }
    }
    gain
}
