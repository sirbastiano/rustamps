use crate::stage6_native::{edge_label_energy, horizontal_index, vertical_index, EdgeDatum};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

const MAX_PATCH_CELLS: usize = 6;
const LARGE_GRID_CANDIDATES: usize = 20000;

fn insert_candidate(
    heap: &mut BinaryHeap<Reverse<(i64, u8, usize, usize)>>,
    limit: usize,
    energy: i64,
    axis: u8,
    row: usize,
    col: usize,
) {
    if energy <= 0 || limit == 0 {
        return;
    }
    let item = Reverse((energy, axis, row, col));
    if heap.len() < limit {
        heap.push(item);
    } else if heap.peek().is_some_and(|lowest| item.0 .0 > lowest.0 .0) {
        heap.pop();
        heap.push(item);
    }
}

fn collect_candidates(
    labels: &[i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) -> Vec<(u8, usize, usize)> {
    let edge_count = horizontal.len() + vertical.len();
    let limit = if labels.len() <= 4096 {
        edge_count
    } else {
        LARGE_GRID_CANDIDATES.min(edge_count)
    };
    let mut heap = BinaryHeap::new();

    for row in 0..nrow {
        for col in 0..ncol.saturating_sub(1) {
            let Some(edge) = horizontal[horizontal_index(row, col, ncol)] else {
                continue;
            };
            let left = labels[row * ncol + col];
            let right = labels[row * ncol + col + 1];
            insert_candidate(
                &mut heap,
                limit,
                edge_label_energy(edge, left, right),
                1,
                row,
                col,
            );
        }
    }
    for row in 0..nrow.saturating_sub(1) {
        for col in 0..ncol {
            let Some(edge) = vertical[vertical_index(row, col, ncol)] else {
                continue;
            };
            let upper = labels[row * ncol + col];
            let lower = labels[(row + 1) * ncol + col];
            insert_candidate(
                &mut heap,
                limit,
                edge_label_energy(edge, upper, lower),
                2,
                row,
                col,
            );
        }
    }

    heap.into_iter()
        .map(|Reverse((_energy, axis, row, col))| (axis, row, col))
        .collect()
}

fn edge_patch_cost(
    labels: &[i32],
    deltas: &[i32],
    patch: (usize, usize, usize, usize),
    ncol: usize,
    edge: EdgeDatum,
    from: (usize, usize),
    to: (usize, usize),
) -> i64 {
    let from_ix = from.0 * ncol + from.1;
    let to_ix = to.0 * ncol + to.1;
    let mut from_label = labels[from_ix];
    let mut to_label = labels[to_ix];
    let (row0, col0, height, width) = patch;
    if from.0 >= row0 && from.0 < row0 + height && from.1 >= col0 && from.1 < col0 + width {
        from_label += deltas[(from.0 - row0) * width + (from.1 - col0)];
    }
    if to.0 >= row0 && to.0 < row0 + height && to.1 >= col0 && to.1 < col0 + width {
        to_label += deltas[(to.0 - row0) * width + (to.1 - col0)];
    }
    edge_label_energy(edge, from_label, to_label)
}

fn patch_energy(
    labels: &[i32],
    deltas: &[i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    patch: (usize, usize, usize, usize),
) -> i64 {
    let (row0, col0, height, width) = patch;
    let row_end = row0 + height;
    let col_end = col0 + width;
    let mut total = 0_i64;

    let h_row_start = row0;
    let h_row_end = row_end.min(nrow);
    let h_col_start = col0.saturating_sub(1);
    let h_col_end = col_end.min(ncol.saturating_sub(1));
    for row in h_row_start..h_row_end {
        for col in h_col_start..h_col_end {
            if let Some(edge) = horizontal[horizontal_index(row, col, ncol)] {
                total += edge_patch_cost(
                    labels,
                    deltas,
                    patch,
                    ncol,
                    edge,
                    (row, col),
                    (row, col + 1),
                );
            }
        }
    }

    let v_row_start = row0.saturating_sub(1);
    let v_row_end = row_end.min(nrow.saturating_sub(1));
    let v_col_start = col0;
    let v_col_end = col_end.min(ncol);
    for row in v_row_start..v_row_end {
        for col in v_col_start..v_col_end {
            if let Some(edge) = vertical[vertical_index(row, col, ncol)] {
                total += edge_patch_cost(
                    labels,
                    deltas,
                    patch,
                    ncol,
                    edge,
                    (row, col),
                    (row + 1, col),
                );
            }
        }
    }
    total
}

fn try_patch(
    labels: &mut [i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    patch: (usize, usize, usize, usize),
) -> bool {
    let (_row0, _col0, height, width) = patch;
    let cell_count = height * width;
    if cell_count == 0 || cell_count > MAX_PATCH_CELLS {
        return false;
    }
    let zero_deltas = vec![0_i32; cell_count];
    let baseline = patch_energy(
        labels,
        &zero_deltas,
        horizontal,
        vertical,
        nrow,
        ncol,
        patch,
    );
    let mut best = baseline;
    let mut best_deltas = zero_deltas;
    let mut deltas = vec![0_i32; cell_count];
    let mut combinations = 1_usize;
    for _ in 0..cell_count {
        combinations *= 3;
    }
    for mut code in 1..combinations {
        for delta in &mut deltas {
            *delta = match code % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            };
            code /= 3;
        }
        let energy = patch_energy(labels, &deltas, horizontal, vertical, nrow, ncol, patch);
        if energy < best {
            best = energy;
            best_deltas.copy_from_slice(&deltas);
        }
    }
    if best >= baseline {
        return false;
    }
    let (row0, col0, _height, _width) = patch;
    for row in 0..height {
        for col in 0..width {
            labels[(row0 + row) * ncol + col0 + col] += best_deltas[row * width + col];
        }
    }
    true
}

fn candidate_patches(
    axis: u8,
    row: usize,
    col: usize,
    nrow: usize,
    ncol: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let (row_min, row_max, col_min, col_max) = if axis == 1 {
        (row, row, col, col + 1)
    } else {
        (row, row + 1, col, col)
    };
    let mut out = Vec::new();
    for height in 1..=3 {
        for width in 1..=3 {
            if height * width > MAX_PATCH_CELLS || height * width < 2 {
                continue;
            }
            if height > nrow || width > ncol {
                continue;
            }
            let row_start_min = row_max.saturating_add(1).saturating_sub(height);
            let row_start_max = row_min.min(nrow - height);
            let col_start_min = col_max.saturating_add(1).saturating_sub(width);
            let col_start_max = col_min.min(ncol - width);
            if row_start_min > row_start_max || col_start_min > col_start_max {
                continue;
            }
            for patch_row in row_start_min..=row_start_max {
                for patch_col in col_start_min..=col_start_max {
                    let patch = (patch_row, patch_col, height, width);
                    if !out.contains(&patch) {
                        out.push(patch);
                    }
                }
            }
        }
    }
    out
}

pub(crate) fn refine_labels_by_patch_shifts(
    labels: &mut [i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) {
    if labels.is_empty() || nrow == 0 || ncol == 0 {
        return;
    }
    for _ in 0..2 {
        let mut changed = false;
        for (axis, row, col) in collect_candidates(labels, horizontal, vertical, nrow, ncol) {
            for patch in candidate_patches(axis, row, col, nrow, ncol) {
                changed |= try_patch(labels, horizontal, vertical, nrow, ncol, patch);
            }
        }
        if !changed {
            break;
        }
    }
}
