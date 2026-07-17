use crate::stage6::unwrap::cut_graph::Dinic;
use crate::stage6::unwrap::native::{
    edge_label_energy, horizontal_index, vertical_index, EdgeDatum,
};

use super::stage6_component_shift_components::collect_barrier_component_ids;
use super::{
    CUT_INF_CAP, MAX_BARRIER_COMPONENT_CUT_BOUNDARIES, MAX_BARRIER_COMPONENT_CUT_COMPONENTS,
};

fn add_cut_unary(graph: &mut Dinic, source: usize, sink: usize, node: usize, d0: i64, d1: i64) {
    let base = d0.min(d1);
    graph.add_arc(source, node, d1 - base);
    graph.add_arc(node, sink, d0 - base);
}

fn add_cut_unary_delta(graph: &mut Dinic, source: usize, sink: usize, node: usize, coeff: i64) {
    add_cut_unary(graph, source, sink, node, 0, coeff);
}

fn add_component_cut_pair(
    graph: &mut Dinic,
    source: usize,
    sink: usize,
    from: usize,
    to: usize,
    costs: [i64; 4],
) {
    let [e00, e10, e01, e11] = costs;
    let weight = e10 + e01 - e00 - e11;
    if weight < 0 {
        return;
    }
    add_cut_unary_delta(graph, source, sink, from, 2 * (e10 - e00) - weight);
    add_cut_unary_delta(graph, source, sink, to, 2 * (e01 - e00) - weight);
    graph.add_cut_edge(from, to, weight);
}

pub(super) fn refine_labels_by_barrier_component_cut(
    labels: &mut [i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    barrier: i64,
) -> bool {
    let Some((component_ids, component_count)) = collect_barrier_component_ids(
        labels,
        horizontal,
        vertical,
        nrow,
        ncol,
        barrier,
        MAX_BARRIER_COMPONENT_CUT_COMPONENTS,
    ) else {
        return false;
    };
    if component_count <= 1 {
        return false;
    }

    let mut best_gain = 0_i64;
    let mut best_shift = 0_i32;
    let mut best_selected = Vec::new();
    for shift in [-1, 1] {
        let Some((gain, selected)) = barrier_component_cut_selection(
            labels,
            &component_ids,
            component_count,
            horizontal,
            vertical,
            nrow,
            ncol,
            shift,
        ) else {
            continue;
        };
        if gain > best_gain {
            best_gain = gain;
            best_shift = shift;
            best_selected = selected;
        }
    }
    if best_gain <= 0 {
        return false;
    }
    for (node, &component) in component_ids.iter().enumerate() {
        if best_selected[component] {
            labels[node] += best_shift;
        }
    }
    true
}

fn barrier_component_cut_selection(
    labels: &[i32],
    component_ids: &[usize],
    component_count: usize,
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    shift: i32,
) -> Option<(i64, Vec<bool>)> {
    let source = component_count;
    let sink = component_count + 1;
    let mut graph = Dinic::new(component_count + 2);
    let mut boundary_count = 0_usize;
    add_cut_unary(&mut graph, source, sink, component_ids[0], 0, CUT_INF_CAP);

    for row in 0..nrow {
        for col in 0..ncol.saturating_sub(1) {
            let Some(edge) = horizontal[horizontal_index(row, col, ncol)] else {
                continue;
            };
            let left = row * ncol + col;
            let right = left + 1;
            let left_component = component_ids[left];
            let right_component = component_ids[right];
            if left_component == right_component {
                continue;
            }
            boundary_count += 1;
            if boundary_count > MAX_BARRIER_COMPONENT_CUT_BOUNDARIES {
                return None;
            }
            let left_label = labels[left];
            let right_label = labels[right];
            add_component_cut_pair(
                &mut graph,
                source,
                sink,
                left_component,
                right_component,
                [
                    edge_label_energy(edge, left_label, right_label),
                    edge_label_energy(edge, left_label + shift, right_label),
                    edge_label_energy(edge, left_label, right_label + shift),
                    edge_label_energy(edge, left_label + shift, right_label + shift),
                ],
            );
        }
    }
    for row in 0..nrow.saturating_sub(1) {
        for col in 0..ncol {
            let Some(edge) = vertical[vertical_index(row, col, ncol)] else {
                continue;
            };
            let upper = row * ncol + col;
            let lower = upper + ncol;
            let upper_component = component_ids[upper];
            let lower_component = component_ids[lower];
            if upper_component == lower_component {
                continue;
            }
            boundary_count += 1;
            if boundary_count > MAX_BARRIER_COMPONENT_CUT_BOUNDARIES {
                return None;
            }
            let upper_label = labels[upper];
            let lower_label = labels[lower];
            add_component_cut_pair(
                &mut graph,
                source,
                sink,
                upper_component,
                lower_component,
                [
                    edge_label_energy(edge, upper_label, lower_label),
                    edge_label_energy(edge, upper_label + shift, lower_label),
                    edge_label_energy(edge, upper_label, lower_label + shift),
                    edge_label_energy(edge, upper_label + shift, lower_label + shift),
                ],
            );
        }
    }
    if boundary_count == 0 {
        return None;
    }

    graph.max_flow(source, sink);
    let reachable = graph.reachable_from(source);
    let selected: Vec<bool> = reachable[..component_count]
        .iter()
        .map(|seen| !*seen)
        .collect();
    if !selected.iter().any(|value| *value) || selected.iter().all(|value| *value) {
        return None;
    }
    let gain = barrier_component_cut_gain(
        labels,
        component_ids,
        &selected,
        horizontal,
        vertical,
        nrow,
        ncol,
        shift,
    );
    if gain > 0 {
        Some((gain, selected))
    } else {
        None
    }
}

fn barrier_component_cut_gain(
    labels: &[i32],
    component_ids: &[usize],
    selected: &[bool],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    shift: i32,
) -> i64 {
    let mut gain = 0_i64;
    for row in 0..nrow {
        for col in 0..ncol.saturating_sub(1) {
            let Some(edge) = horizontal[horizontal_index(row, col, ncol)] else {
                continue;
            };
            let left = row * ncol + col;
            let right = left + 1;
            let left_selected = selected[component_ids[left]];
            let right_selected = selected[component_ids[right]];
            if left_selected == right_selected {
                continue;
            }
            let left_label = labels[left];
            let right_label = labels[right];
            gain += edge_label_energy(edge, left_label, right_label)
                - edge_label_energy(
                    edge,
                    left_label + if left_selected { shift } else { 0 },
                    right_label + if right_selected { shift } else { 0 },
                );
        }
    }
    for row in 0..nrow.saturating_sub(1) {
        for col in 0..ncol {
            let Some(edge) = vertical[vertical_index(row, col, ncol)] else {
                continue;
            };
            let upper = row * ncol + col;
            let lower = upper + ncol;
            let upper_selected = selected[component_ids[upper]];
            let lower_selected = selected[component_ids[lower]];
            if upper_selected == lower_selected {
                continue;
            }
            let upper_label = labels[upper];
            let lower_label = labels[lower];
            gain += edge_label_energy(edge, upper_label, lower_label)
                - edge_label_energy(
                    edge,
                    upper_label + if upper_selected { shift } else { 0 },
                    lower_label + if lower_selected { shift } else { 0 },
                );
        }
    }
    gain
}
