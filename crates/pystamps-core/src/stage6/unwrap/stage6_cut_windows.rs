use crate::stage6::unwrap::native::{
    edge_label_energy, horizontal_index, vertical_index, EdgeDatum,
};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

const LARGE_GRID_CUT_WINDOWS: usize = 1024;

pub(super) fn cut_windows(
    labels: &[i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    max_cells: usize,
) -> Vec<(usize, usize, usize, usize)> {
    if nrow.saturating_mul(ncol) <= max_cells {
        return vec![(0, 0, nrow, ncol)];
    }
    let side = (max_cells as f64).sqrt() as usize;
    let limit = LARGE_GRID_CUT_WINDOWS.min(horizontal.len() + vertical.len());
    let mut edge_heap = BinaryHeap::new();
    for row in 0..nrow {
        for col in 0..ncol.saturating_sub(1) {
            if let Some(edge) = horizontal[horizontal_index(row, col, ncol)] {
                let score = edge_cut_potential(
                    edge,
                    labels[row * ncol + col],
                    labels[row * ncol + col + 1],
                );
                insert_candidate(&mut edge_heap, limit, score, 1, row, col);
            }
        }
    }
    for row in 0..nrow.saturating_sub(1) {
        for col in 0..ncol {
            if let Some(edge) = vertical[vertical_index(row, col, ncol)] {
                let score = edge_cut_potential(
                    edge,
                    labels[row * ncol + col],
                    labels[(row + 1) * ncol + col],
                );
                insert_candidate(&mut edge_heap, limit, score, 2, row, col);
            }
        }
    }
    let mut windows = Vec::new();
    for window in aggregate_windows(labels, horizontal, vertical, nrow, ncol, side, limit) {
        if windows.len() >= limit {
            break;
        }
        if !windows.contains(&window) {
            windows.push(window);
        }
    }
    for Reverse((_energy, axis, row, col)) in edge_heap {
        if windows.len() >= limit {
            break;
        }
        let window = candidate_window(axis, row, col, nrow, ncol, side);
        if !windows.contains(&window) {
            windows.push(window);
        }
    }
    windows
}

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

fn edge_cut_potential(edge: EdgeDatum, from_label: i32, to_label: i32) -> i64 {
    let old = edge_label_energy(edge, from_label, to_label);
    [
        edge_label_energy(edge, from_label - 1, to_label),
        edge_label_energy(edge, from_label + 1, to_label),
        edge_label_energy(edge, from_label, to_label - 1),
        edge_label_energy(edge, from_label, to_label + 1),
    ]
    .into_iter()
    .map(|new| old - new)
    .max()
    .unwrap_or(0)
    .max(0)
}

fn insert_window_candidate(
    heap: &mut BinaryHeap<Reverse<(i64, usize, usize)>>,
    limit: usize,
    score: i64,
    row: usize,
    col: usize,
) {
    if score <= 0 || limit == 0 {
        return;
    }
    let item = Reverse((score, row, col));
    if heap.len() < limit {
        heap.push(item);
    } else if heap.peek().is_some_and(|lowest| item.0 .0 > lowest.0 .0) {
        heap.pop();
        heap.push(item);
    }
}

fn candidate_window(
    axis: u8,
    row: usize,
    col: usize,
    nrow: usize,
    ncol: usize,
    side: usize,
) -> (usize, usize, usize, usize) {
    let row_min = row;
    let row_max = if axis == 2 { row + 1 } else { row };
    let col_min = col;
    let col_max = if axis == 1 { col + 1 } else { col };
    let height = side.min(nrow);
    let width = side.min(ncol);
    let row0 = row_min.min(nrow - height).min(row_max);
    let col0 = col_min.min(ncol - width).min(col_max);
    (row0, col0, height, width)
}

fn aggregate_windows(
    labels: &[i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    side: usize,
    limit: usize,
) -> Vec<(usize, usize, usize, usize)> {
    if limit == 0 {
        return Vec::new();
    }
    let height = side.min(nrow);
    let width = side.min(ncol);
    if height == 0 || width == 0 {
        return Vec::new();
    }
    let row_starts = window_starts(nrow, height);
    let col_starts = window_starts(ncol, width);
    let mut heap = BinaryHeap::new();

    for &row0 in &row_starts {
        for &col0 in &col_starts {
            let score = aggregate_window_score(
                labels, horizontal, vertical, nrow, ncol, row0, col0, height, width,
            );
            insert_window_candidate(&mut heap, limit, score, row0, col0);
        }
    }

    heap.into_iter()
        .map(|Reverse((_score, row, col))| (row, col, height, width))
        .collect()
}

fn window_starts(limit: usize, size: usize) -> Vec<usize> {
    if limit <= size {
        return vec![0];
    }
    let max_start = limit - size;
    let step = (size / 2).max(1);
    let mut starts = Vec::new();
    let mut value = 0_usize;
    while value < max_start {
        starts.push(value);
        value = value.saturating_add(step);
    }
    if starts.last().copied() != Some(max_start) {
        starts.push(max_start);
    }
    starts
}

fn aggregate_window_score(
    labels: &[i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    row0: usize,
    col0: usize,
    height: usize,
    width: usize,
) -> i64 {
    let row1 = row0 + height;
    let col1 = col0 + width;
    let mut score = 0_i64;
    for row in row0..row1 {
        for col in col0.saturating_sub(1)..col1.min(ncol.saturating_sub(1)) {
            if let Some(edge) = horizontal[horizontal_index(row, col, ncol)] {
                score += edge_cut_potential(
                    edge,
                    labels[row * ncol + col],
                    labels[row * ncol + col + 1],
                );
            }
        }
    }
    for row in row0.saturating_sub(1)..row1.min(nrow.saturating_sub(1)) {
        for col in col0..col1 {
            if let Some(edge) = vertical[vertical_index(row, col, ncol)] {
                score += edge_cut_potential(
                    edge,
                    labels[row * ncol + col],
                    labels[(row + 1) * ncol + col],
                );
            }
        }
    }
    score
}
