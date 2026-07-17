use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::{horizontal_index, vertical_index, EdgeDatum};

#[derive(Clone, Copy)]
pub(super) struct Edge {
    pub(super) from: usize,
    pub(super) to: usize,
    pub(super) cost: i32,
    pub(super) desired_delta: f32,
    pub(super) offset: i32,
    pub(super) dzmax: i32,
    pub(super) laycost: i32,
    pub(super) nshortcycle: i32,
    pub(super) flow_sign: i32,
    pub(super) flow: i32,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct QueueEdge {
    cost: i32,
    from: usize,
    to: usize,
    desired_delta_bits: u32,
}

impl Ord for QueueEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .cmp(&self.cost)
            .then_with(|| other.from.cmp(&self.from))
            .then_with(|| other.to.cmp(&self.to))
    }
}

impl PartialOrd for QueueEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn push_edges(
    node: usize,
    adjacency: &[Vec<Edge>],
    visited: &[bool],
    heap: &mut BinaryHeap<QueueEdge>,
) {
    for edge in &adjacency[node] {
        if !visited[edge.to] {
            heap.push(QueueEdge {
                cost: edge.cost,
                from: edge.from,
                to: edge.to,
                desired_delta_bits: edge.desired_delta.to_bits(),
            });
        }
    }
}

fn push_grid_label_edge(
    cost: i32,
    from: usize,
    to: usize,
    desired_delta: f32,
    visited: &[bool],
    heap: &mut BinaryHeap<QueueEdge>,
) {
    if !visited[to] {
        heap.push(QueueEdge {
            cost,
            from,
            to,
            desired_delta_bits: desired_delta.to_bits(),
        });
    }
}

fn push_grid_label_edges(
    node: usize,
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    visited: &[bool],
    heap: &mut BinaryHeap<QueueEdge>,
) {
    let row = node / ncol;
    let col = node - row * ncol;
    if col + 1 < ncol {
        if let Some(edge) = horizontal[horizontal_index(row, col, ncol)] {
            push_grid_label_edge(edge.cost, node, node + 1, edge.desired_delta, visited, heap);
        }
    }
    if col > 0 {
        if let Some(edge) = horizontal[horizontal_index(row, col - 1, ncol)] {
            push_grid_label_edge(
                edge.cost,
                node,
                node - 1,
                -edge.desired_delta,
                visited,
                heap,
            );
        }
    }
    if row + 1 < nrow {
        if let Some(edge) = vertical[vertical_index(row, col, ncol)] {
            push_grid_label_edge(
                edge.cost,
                node,
                node + ncol,
                edge.desired_delta,
                visited,
                heap,
            );
        }
    }
    if row > 0 {
        if let Some(edge) = vertical[vertical_index(row - 1, col, ncol)] {
            push_grid_label_edge(
                edge.cost,
                node,
                node - ncol,
                -edge.desired_delta,
                visited,
                heap,
            );
        }
    }
}

pub(crate) fn reseed_labels_from_edge_deltas(
    labels: &mut [i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) {
    if labels.len() != nrow * ncol {
        return;
    }
    let mut visited = vec![false; labels.len()];
    let mut heap = BinaryHeap::new();
    for seed in 0..labels.len() {
        if visited[seed] {
            continue;
        }
        visited[seed] = true;
        push_grid_label_edges(seed, horizontal, vertical, nrow, ncol, &visited, &mut heap);
        while let Some(edge) = heap.pop() {
            if visited[edge.to] {
                continue;
            }
            let desired_delta = f32::from_bits(edge.desired_delta_bits);
            labels[edge.to] = labels[edge.from] + desired_delta.round() as i32;
            visited[edge.to] = true;
            push_grid_label_edges(
                edge.to, horizontal, vertical, nrow, ncol, &visited, &mut heap,
            );
        }
    }
}

pub(super) fn build_adjacency(
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) -> Vec<Vec<Edge>> {
    let mut adjacency = vec![Vec::<Edge>::new(); nrow * ncol];
    for row in 0..nrow.saturating_sub(1) {
        for col in 0..ncol {
            if let Some(edge) = vertical[vertical_index(row, col, ncol)] {
                let upper = row * ncol + col;
                let lower = (row + 1) * ncol + col;
                adjacency[upper].push(Edge {
                    from: upper,
                    to: lower,
                    cost: edge.cost,
                    desired_delta: edge.desired_delta,
                    offset: edge.offset,
                    dzmax: edge.dzmax,
                    laycost: edge.laycost,
                    nshortcycle: edge.nshortcycle,
                    flow_sign: -1,
                    flow: edge.flow,
                });
                adjacency[lower].push(Edge {
                    from: lower,
                    to: upper,
                    cost: edge.cost,
                    desired_delta: -edge.desired_delta,
                    offset: edge.offset,
                    dzmax: edge.dzmax,
                    laycost: edge.laycost,
                    nshortcycle: edge.nshortcycle,
                    flow_sign: 1,
                    flow: edge.flow,
                });
            }
        }
    }
    for row in 0..nrow {
        for col in 0..ncol.saturating_sub(1) {
            if let Some(edge) = horizontal[horizontal_index(row, col, ncol)] {
                let left = row * ncol + col;
                let right = row * ncol + col + 1;
                adjacency[left].push(Edge {
                    from: left,
                    to: right,
                    cost: edge.cost,
                    desired_delta: edge.desired_delta,
                    offset: edge.offset,
                    dzmax: edge.dzmax,
                    laycost: edge.laycost,
                    nshortcycle: edge.nshortcycle,
                    flow_sign: 1,
                    flow: edge.flow,
                });
                adjacency[right].push(Edge {
                    from: right,
                    to: left,
                    cost: edge.cost,
                    desired_delta: -edge.desired_delta,
                    offset: edge.offset,
                    dzmax: edge.dzmax,
                    laycost: edge.laycost,
                    nshortcycle: edge.nshortcycle,
                    flow_sign: -1,
                    flow: edge.flow,
                });
            }
        }
    }
    adjacency
}

pub(super) fn seed_labels_from_adjacency(node_count: usize, adjacency: &[Vec<Edge>]) -> Vec<i32> {
    let mut labels = vec![0_i32; node_count];
    let mut visited = vec![false; node_count];
    let mut heap = BinaryHeap::new();
    for seed in 0..node_count {
        if visited[seed] {
            continue;
        }
        visited[seed] = true;
        push_edges(seed, adjacency, &visited, &mut heap);
        while let Some(edge) = heap.pop() {
            if visited[edge.to] {
                continue;
            }
            let desired_delta = f32::from_bits(edge.desired_delta_bits);
            labels[edge.to] = labels[edge.from] + desired_delta.round() as i32;
            visited[edge.to] = true;
            push_edges(edge.to, adjacency, &visited, &mut heap);
        }
    }
    labels
}
