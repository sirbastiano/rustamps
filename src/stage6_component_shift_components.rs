use crate::stage6_native::{edge_label_energy, horizontal_index, vertical_index, EdgeDatum};

pub(super) fn collect_barrier_component_ids(
    labels: &[i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    barrier: i64,
    max_components: usize,
) -> Option<(Vec<usize>, usize)> {
    let cell_count = nrow.saturating_mul(ncol);
    let mut component_ids = vec![usize::MAX; cell_count];
    let mut stack = Vec::new();
    let mut component_count = 0_usize;

    for seed in 0..cell_count {
        if component_ids[seed] != usize::MAX {
            continue;
        }
        if component_count >= max_components {
            return None;
        }
        component_ids[seed] = component_count;
        stack.push(seed);
        while let Some(node) = stack.pop() {
            let row = node / ncol;
            let col = node - row * ncol;
            if col > 0 {
                let edge_ix = horizontal_index(row, col - 1, ncol);
                if let Some(edge) = horizontal[edge_ix] {
                    push_barrier_component_id(
                        node - 1,
                        edge,
                        labels[node - 1],
                        labels[node],
                        barrier,
                        component_count,
                        &mut component_ids,
                        &mut stack,
                    );
                }
            }
            if col + 1 < ncol {
                let edge_ix = horizontal_index(row, col, ncol);
                if let Some(edge) = horizontal[edge_ix] {
                    push_barrier_component_id(
                        node + 1,
                        edge,
                        labels[node],
                        labels[node + 1],
                        barrier,
                        component_count,
                        &mut component_ids,
                        &mut stack,
                    );
                }
            }
            if row > 0 {
                let edge_ix = vertical_index(row - 1, col, ncol);
                if let Some(edge) = vertical[edge_ix] {
                    push_barrier_component_id(
                        node - ncol,
                        edge,
                        labels[node - ncol],
                        labels[node],
                        barrier,
                        component_count,
                        &mut component_ids,
                        &mut stack,
                    );
                }
            }
            if row + 1 < nrow {
                let edge_ix = vertical_index(row, col, ncol);
                if let Some(edge) = vertical[edge_ix] {
                    push_barrier_component_id(
                        node + ncol,
                        edge,
                        labels[node],
                        labels[node + ncol],
                        barrier,
                        component_count,
                        &mut component_ids,
                        &mut stack,
                    );
                }
            }
        }
        component_count += 1;
    }
    Some((component_ids, component_count))
}

fn push_barrier_component_id(
    node: usize,
    edge: EdgeDatum,
    from_label: i32,
    to_label: i32,
    barrier: i64,
    component: usize,
    component_ids: &mut [usize],
    stack: &mut Vec<usize>,
) {
    if component_ids[node] != usize::MAX {
        return;
    }
    if edge_label_energy(edge, from_label, to_label) < barrier {
        component_ids[node] = component;
        stack.push(node);
    }
}

pub(super) fn collect_same_label_component(
    seed: usize,
    labels: &[i32],
    nrow: usize,
    ncol: usize,
    visited: &mut [bool],
    stack: &mut Vec<usize>,
    component: &mut Vec<usize>,
) {
    stack.clear();
    component.clear();
    let label = labels[seed];
    visited[seed] = true;
    stack.push(seed);
    while let Some(node) = stack.pop() {
        component.push(node);
        let row = node / ncol;
        let col = node - row * ncol;
        if col > 0 {
            push_same_label(node - 1, label, labels, visited, stack);
        }
        if col + 1 < ncol {
            push_same_label(node + 1, label, labels, visited, stack);
        }
        if row > 0 {
            push_same_label(node - ncol, label, labels, visited, stack);
        }
        if row + 1 < nrow {
            push_same_label(node + ncol, label, labels, visited, stack);
        }
    }
}

pub(super) fn collect_barrier_component(
    seed: usize,
    labels: &[i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    barrier: i64,
    visited: &mut [bool],
    stack: &mut Vec<usize>,
    component: &mut Vec<usize>,
) {
    stack.clear();
    component.clear();
    visited[seed] = true;
    stack.push(seed);
    while let Some(node) = stack.pop() {
        component.push(node);
        let row = node / ncol;
        let col = node - row * ncol;
        if col > 0 {
            let edge_ix = horizontal_index(row, col - 1, ncol);
            if let Some(edge) = horizontal[edge_ix] {
                push_across_nonbarrier_edge(
                    node - 1,
                    edge,
                    labels[node - 1],
                    labels[node],
                    barrier,
                    visited,
                    stack,
                );
            }
        }
        if col + 1 < ncol {
            let edge_ix = horizontal_index(row, col, ncol);
            if let Some(edge) = horizontal[edge_ix] {
                push_across_nonbarrier_edge(
                    node + 1,
                    edge,
                    labels[node],
                    labels[node + 1],
                    barrier,
                    visited,
                    stack,
                );
            }
        }
        if row > 0 {
            let edge_ix = vertical_index(row - 1, col, ncol);
            if let Some(edge) = vertical[edge_ix] {
                push_across_nonbarrier_edge(
                    node - ncol,
                    edge,
                    labels[node - ncol],
                    labels[node],
                    barrier,
                    visited,
                    stack,
                );
            }
        }
        if row + 1 < nrow {
            let edge_ix = vertical_index(row, col, ncol);
            if let Some(edge) = vertical[edge_ix] {
                push_across_nonbarrier_edge(
                    node + ncol,
                    edge,
                    labels[node],
                    labels[node + ncol],
                    barrier,
                    visited,
                    stack,
                );
            }
        }
    }
}

fn push_across_nonbarrier_edge(
    node: usize,
    edge: EdgeDatum,
    from_label: i32,
    to_label: i32,
    barrier: i64,
    visited: &mut [bool],
    stack: &mut Vec<usize>,
) {
    if visited[node] {
        return;
    }
    if edge_label_energy(edge, from_label, to_label) < barrier {
        visited[node] = true;
        stack.push(node);
    }
}

fn push_same_label(
    node: usize,
    label: i32,
    labels: &[i32],
    visited: &mut [bool],
    stack: &mut Vec<usize>,
) {
    if !visited[node] && labels[node] == label {
        visited[node] = true;
        stack.push(node);
    }
}
