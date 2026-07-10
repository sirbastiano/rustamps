use crate::stage6_cut_graph::Dinic;
use crate::stage6_native::{edge_label_energy, horizontal_index, vertical_index, EdgeDatum};

#[path = "stage6_cut_windows.rs"]
mod stage6_cut_windows;
use self::stage6_cut_windows::cut_windows;

#[cfg(test)]
#[path = "stage6_cut_tests.rs"]
mod stage6_cut_tests;

const MAX_CUT_CELLS: usize = 16384;
const INF_CAP: i64 = 1_i64 << 58;

fn add_unary(graph: &mut Dinic, source: usize, sink: usize, node: usize, d0: i64, d1: i64) {
    let base = d0.min(d1);
    graph.add_arc(source, node, d1 - base);
    graph.add_arc(node, sink, d0 - base);
}

fn add_unary_delta(graph: &mut Dinic, source: usize, sink: usize, node: usize, coeff: i64) {
    add_unary(graph, source, sink, node, 0, coeff);
}

fn add_pair(
    graph: &mut Dinic,
    source: usize,
    sink: usize,
    from: usize,
    to: usize,
    costs: [i64; 4],
) {
    let [e00, e10, e01, e11] = costs;
    let w = e10 + e01 - e00 - e11;
    if w < 0 {
        return;
    }
    add_unary_delta(graph, source, sink, from, 2 * (e10 - e00) - w);
    add_unary_delta(graph, source, sink, to, 2 * (e01 - e00) - w);
    graph.add_cut_edge(from, to, w);
}

fn local_index(row: usize, col: usize, row0: usize, col0: usize, width: usize) -> usize {
    (row - row0) * width + (col - col0)
}

fn cell_delta(
    selected: Option<&[bool]>,
    row: usize,
    col: usize,
    patch: (usize, usize, usize, usize),
    shift: i32,
) -> i32 {
    let (row0, col0, height, width) = patch;
    if row < row0 || row >= row0 + height || col < col0 || col >= col0 + width {
        return 0;
    }
    selected
        .and_then(|values| values.get(local_index(row, col, row0, col0, width)))
        .copied()
        .unwrap_or(false) as i32
        * shift
}

fn window_energy(
    labels: &[i32],
    selected: Option<&[bool]>,
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    patch: (usize, usize, usize, usize),
    shift: i32,
) -> i64 {
    let (row0, col0, height, width) = patch;
    let row1 = row0 + height;
    let col1 = col0 + width;
    let mut total = 0_i64;
    for row in row0..row1 {
        for col in col0.saturating_sub(1)..col1.min(ncol.saturating_sub(1)) {
            if let Some(edge) = horizontal[horizontal_index(row, col, ncol)] {
                let left = labels[row * ncol + col] + cell_delta(selected, row, col, patch, shift);
                let right =
                    labels[row * ncol + col + 1] + cell_delta(selected, row, col + 1, patch, shift);
                total += edge_label_energy(edge, left, right);
            }
        }
    }
    for row in row0.saturating_sub(1)..row1.min(nrow.saturating_sub(1)) {
        for col in col0..col1 {
            if let Some(edge) = vertical[vertical_index(row, col, ncol)] {
                let upper = labels[row * ncol + col] + cell_delta(selected, row, col, patch, shift);
                let lower = labels[(row + 1) * ncol + col]
                    + cell_delta(selected, row + 1, col, patch, shift);
                total += edge_label_energy(edge, upper, lower);
            }
        }
    }
    total
}

fn add_edge_terms(
    graph: &mut Dinic,
    labels: &[i32],
    source: usize,
    sink: usize,
    edge: EdgeDatum,
    from: (usize, usize),
    to: (usize, usize),
    patch: (usize, usize, usize, usize),
    ncol: usize,
    shift: i32,
) {
    let (row0, col0, height, width) = patch;
    let from_in =
        from.0 >= row0 && from.0 < row0 + height && from.1 >= col0 && from.1 < col0 + width;
    let to_in = to.0 >= row0 && to.0 < row0 + height && to.1 >= col0 && to.1 < col0 + width;
    let from_label = labels[from.0 * ncol + from.1];
    let to_label = labels[to.0 * ncol + to.1];
    if from_in && to_in {
        let from_ix = local_index(from.0, from.1, row0, col0, width);
        let to_ix = local_index(to.0, to.1, row0, col0, width);
        add_pair(
            graph,
            source,
            sink,
            from_ix,
            to_ix,
            [
                edge_label_energy(edge, from_label, to_label),
                edge_label_energy(edge, from_label + shift, to_label),
                edge_label_energy(edge, from_label, to_label + shift),
                edge_label_energy(edge, from_label + shift, to_label + shift),
            ],
        );
    } else if from_in {
        let node = local_index(from.0, from.1, row0, col0, width);
        add_unary(
            graph,
            source,
            sink,
            node,
            2 * edge_label_energy(edge, from_label, to_label),
            2 * edge_label_energy(edge, from_label + shift, to_label),
        );
    } else if to_in {
        let node = local_index(to.0, to.1, row0, col0, width);
        add_unary(
            graph,
            source,
            sink,
            node,
            2 * edge_label_energy(edge, from_label, to_label),
            2 * edge_label_energy(edge, from_label, to_label + shift),
        );
    }
}

fn try_cut_window(
    labels: &mut [i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    patch: (usize, usize, usize, usize),
    shift: i32,
) -> bool {
    let (row0, col0, height, width) = patch;
    let cells = height * width;
    if cells == 0 || cells > MAX_CUT_CELLS {
        return false;
    }
    let source = cells;
    let sink = cells + 1;
    let mut graph = Dinic::new(cells + 2);
    if row0 == 0 && col0 == 0 {
        add_unary(&mut graph, source, sink, 0, 0, INF_CAP);
    }
    for row in row0..row0 + height {
        for col in col0.saturating_sub(1)..(col0 + width).min(ncol.saturating_sub(1)) {
            if let Some(edge) = horizontal[horizontal_index(row, col, ncol)] {
                add_edge_terms(
                    &mut graph,
                    labels,
                    source,
                    sink,
                    edge,
                    (row, col),
                    (row, col + 1),
                    patch,
                    ncol,
                    shift,
                );
            }
        }
    }
    for row in row0.saturating_sub(1)..(row0 + height).min(nrow.saturating_sub(1)) {
        for col in col0..col0 + width {
            if let Some(edge) = vertical[vertical_index(row, col, ncol)] {
                add_edge_terms(
                    &mut graph,
                    labels,
                    source,
                    sink,
                    edge,
                    (row, col),
                    (row + 1, col),
                    patch,
                    ncol,
                    shift,
                );
            }
        }
    }
    graph.max_flow(source, sink);
    let reachable = graph.reachable_from(source);
    let selected: Vec<bool> = reachable[..cells].iter().map(|seen| !*seen).collect();
    if !selected.iter().any(|value| *value) {
        return false;
    }
    let before = window_energy(labels, None, horizontal, vertical, nrow, ncol, patch, shift);
    let after = window_energy(
        labels,
        Some(&selected),
        horizontal,
        vertical,
        nrow,
        ncol,
        patch,
        shift,
    );
    if after >= before {
        return false;
    }
    for row in 0..height {
        for col in 0..width {
            if selected[row * width + col] {
                labels[(row0 + row) * ncol + col0 + col] += shift;
            }
        }
    }
    true
}

pub(crate) fn refine_labels_by_binary_cuts(
    labels: &mut [i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) {
    if labels.is_empty() {
        return;
    }
    let windows = cut_windows(labels, horizontal, vertical, nrow, ncol, MAX_CUT_CELLS);
    for _ in 0..3 {
        let mut changed = false;
        for &window in &windows {
            changed |= try_cut_window(labels, horizontal, vertical, nrow, ncol, window, 1);
            changed |= try_cut_window(labels, horizontal, vertical, nrow, ncol, window, -1);
        }
        if !changed {
            break;
        }
    }
}
