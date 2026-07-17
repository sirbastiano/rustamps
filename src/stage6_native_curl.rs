use super::{
    apply_edge_correction, edge_weight, horizontal_index, rounded_delta, vertical_index, EdgeDatum,
};

pub(crate) fn plaquette_curl(
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    ncol: usize,
    row: usize,
    col: usize,
) -> Option<i32> {
    let top = horizontal[horizontal_index(row, col, ncol)]?;
    let right = vertical[vertical_index(row, col + 1, ncol)]?;
    let bottom = horizontal[horizontal_index(row + 1, col, ncol)]?;
    let left = vertical[vertical_index(row, col, ncol)]?;
    Some(rounded_delta(top) + rounded_delta(right) - rounded_delta(bottom) - rounded_delta(left))
}

#[allow(dead_code)]
pub(crate) fn balance_local_curl(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) {
    if nrow < 2 || ncol < 2 {
        return;
    }
    for _ in 0..16 {
        let mut changed = false;
        for row in 0..(nrow - 1) {
            for col in 0..(ncol - 1) {
                let top_ix = horizontal_index(row, col, ncol);
                let right_ix = vertical_index(row, col + 1, ncol);
                let bottom_ix = horizontal_index(row + 1, col, ncol);
                let left_ix = vertical_index(row, col, ncol);
                let (Some(top), Some(right), Some(bottom), Some(left)) = (
                    horizontal[top_ix],
                    vertical[right_ix],
                    horizontal[bottom_ix],
                    vertical[left_ix],
                ) else {
                    continue;
                };
                let Some(curl) = plaquette_curl(horizontal, vertical, ncol, row, col) else {
                    continue;
                };
                if curl == 0 {
                    continue;
                }

                let mut best_slot = 0_u8;
                let mut best_weight = edge_weight(top.cost);
                for (slot, edge) in [(1_u8, right), (2_u8, bottom), (3_u8, left)] {
                    let weight = edge_weight(edge.cost);
                    if weight < best_weight {
                        best_slot = slot;
                        best_weight = weight;
                    }
                }
                match best_slot {
                    0 => apply_edge_correction(&mut horizontal[top_ix], -curl),
                    1 => apply_edge_correction(&mut vertical[right_ix], -curl),
                    2 => apply_edge_correction(&mut horizontal[bottom_ix], curl),
                    _ => apply_edge_correction(&mut vertical[left_ix], curl),
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}
