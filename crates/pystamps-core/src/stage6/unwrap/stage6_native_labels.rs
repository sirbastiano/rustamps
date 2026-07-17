use super::stage6_native_graph::Edge;
use super::{
    defo_edge_cost, edge_label_energy, edge_weight, horizontal_index, vertical_index, EdgeDatum,
};

fn local_label_energy(node: usize, candidate: i32, labels: &[i32], adjacency: &[Vec<Edge>]) -> f64 {
    let mut energy = 0.0_f64;
    for edge in &adjacency[node] {
        let label_delta = labels[edge.to] - candidate;
        let flow = edge.flow + edge.flow_sign * (label_delta - edge.desired_delta.round() as i32);
        energy += defo_edge_cost(
            edge.cost,
            edge.offset,
            edge.dzmax,
            edge.laycost,
            edge.nshortcycle,
            flow,
        ) as f64;
    }
    energy
}

pub(super) fn refine_labels(labels: &mut [i32], adjacency: &[Vec<Edge>]) {
    if labels.is_empty() {
        return;
    }
    for _ in 0..32 {
        let mut changed = false;
        for node in 0..labels.len() {
            if adjacency[node].is_empty() {
                continue;
            }
            let current = labels[node];
            let mut weight_sum = 0.0_f64;
            let mut target_sum = 0.0_f64;
            for edge in &adjacency[node] {
                let weight = edge_weight(edge.cost);
                weight_sum += weight;
                target_sum += weight * (f64::from(labels[edge.to]) - f64::from(edge.desired_delta));
            }
            if weight_sum == 0.0 {
                continue;
            }
            let target = target_sum / weight_sum;
            let target_floor = target.floor() as i32;
            let mut candidates = [
                current - 2,
                current - 1,
                current,
                current + 1,
                current + 2,
                target_floor - 2,
                target_floor - 1,
                target_floor,
                target_floor + 1,
                target_floor + 2,
                target.round() as i32,
            ];
            candidates.sort_unstable();
            let mut best = current;
            let mut best_energy = local_label_energy(node, current, labels, adjacency);
            for candidate in candidates {
                let energy = local_label_energy(node, candidate, labels, adjacency);
                if energy + 1.0e-9 < best_energy {
                    best = candidate;
                    best_energy = energy;
                }
            }
            if best != current {
                labels[node] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

pub(super) fn refine_labels_by_line_shifts(
    labels: &mut [i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) {
    if labels.is_empty() {
        return;
    }
    for _ in 0..32 {
        let mut best_gain = 0_i64;
        let mut best_axis = 0_u8;
        let mut best_split = 0_usize;
        let mut best_delta = 0_i32;

        for split in 1..ncol {
            for delta in [-1, 1] {
                let mut gain = 0_i64;
                for row in 0..nrow {
                    let Some(edge) = horizontal[horizontal_index(row, split - 1, ncol)] else {
                        continue;
                    };
                    let left = labels[row * ncol + split - 1];
                    let right = labels[row * ncol + split];
                    gain += edge_label_energy(edge, left, right)
                        - edge_label_energy(edge, left, right + delta);
                }
                if gain > best_gain {
                    best_gain = gain;
                    best_axis = 1;
                    best_split = split;
                    best_delta = delta;
                }
            }
        }

        for split in 1..nrow {
            for delta in [-1, 1] {
                let mut gain = 0_i64;
                for col in 0..ncol {
                    let Some(edge) = vertical[vertical_index(split - 1, col, ncol)] else {
                        continue;
                    };
                    let upper = labels[(split - 1) * ncol + col];
                    let lower = labels[split * ncol + col];
                    gain += edge_label_energy(edge, upper, lower)
                        - edge_label_energy(edge, upper, lower + delta);
                }
                if gain > best_gain {
                    best_gain = gain;
                    best_axis = 2;
                    best_split = split;
                    best_delta = delta;
                }
            }
        }

        if best_gain <= 0 {
            break;
        }
        if best_axis == 1 {
            for row in 0..nrow {
                for col in best_split..ncol {
                    labels[row * ncol + col] += best_delta;
                }
            }
        } else if best_axis == 2 {
            for row in best_split..nrow {
                for col in 0..ncol {
                    labels[row * ncol + col] += best_delta;
                }
            }
        }
    }
}
