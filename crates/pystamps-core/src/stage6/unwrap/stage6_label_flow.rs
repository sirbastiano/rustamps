use crate::stage6::unwrap::native::{
    apply_edge_correction, horizontal_index, rounded_delta, vertical_index, EdgeDatum,
};

pub(crate) fn absorb_label_corrections(
    labels: &[i32],
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) -> usize {
    if labels.len() != nrow * ncol {
        return 0;
    }
    let mut changed = 0;
    if ncol > 1 {
        for row in 0..nrow {
            for col in 0..(ncol - 1) {
                let index = horizontal_index(row, col, ncol);
                let Some(edge) = horizontal[index] else {
                    continue;
                };
                let left = labels[row * ncol + col];
                let right = labels[row * ncol + col + 1];
                let delta = right - left - rounded_delta(edge);
                if delta != 0 {
                    apply_edge_correction(&mut horizontal[index], delta);
                    changed += 1;
                }
            }
        }
    }
    if nrow > 1 {
        for row in 0..(nrow - 1) {
            for col in 0..ncol {
                let index = vertical_index(row, col, ncol);
                let Some(edge) = vertical[index] else {
                    continue;
                };
                let upper = labels[row * ncol + col];
                let lower = labels[(row + 1) * ncol + col];
                let delta = lower - upper - rounded_delta(edge);
                if delta != 0 {
                    apply_edge_correction(&mut vertical[index], delta);
                    changed += 1;
                }
            }
        }
    }
    changed
}
