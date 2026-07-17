use crate::stage6::unwrap::native::{horizontal_index, vertical_index, EdgeDatum};
use crate::stage6::unwrap::residual::{
    find_negative_unit_cycle, residual_arc_cost, saturate_residual_cycle, ResidualArc,
};
use std::collections::{BinaryHeap, HashSet};

const LOCAL_WINDOW_RADIUS: usize = 8;
const LOCAL_WINDOW_LIMIT: usize = 256;

pub(crate) fn cancel_local_negative_cycles_with_nflow(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    nflow: i32,
    max_cycles: usize,
) -> usize {
    if max_cycles == 0 || nrow < 2 || ncol < 2 {
        return 0;
    }
    let windows = candidate_windows(horizontal, vertical, nrow, ncol, nflow);
    let mut applied = 0;
    for (r0, c0, r1, c1) in windows {
        while applied < max_cycles {
            let arcs =
                window_residual_arcs(horizontal, vertical, nrow, ncol, nflow, r0, c0, r1, c1);
            let node_count = (r1 - r0) * (c1 - c0) + 1;
            let Some(cycle) = find_negative_unit_cycle(&arcs, node_count) else {
                break;
            };
            if saturate_residual_cycle(horizontal, vertical, &arcs, &cycle) == 0 {
                break;
            }
            applied += 1;
        }
        if applied == max_cycles {
            break;
        }
    }
    applied
}

fn candidate_windows(
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    nflow: i32,
) -> Vec<(usize, usize, usize, usize)> {
    let prn = nrow - 1;
    let pcn = ncol - 1;
    let mut candidates = BinaryHeap::new();
    for row in 0..nrow {
        for col in 0..pcn {
            let Some(edge) = horizontal[horizontal_index(row, col, ncol)] else {
                continue;
            };
            if let Some(cost) = best_negative_residual_cost(edge, nflow) {
                let center_row = row.saturating_sub(1).min(prn - 1);
                push_candidate(&mut candidates, cost, center_row, col);
            }
        }
    }
    for row in 0..prn {
        for col in 0..ncol {
            let Some(edge) = vertical[vertical_index(row, col, ncol)] else {
                continue;
            };
            if let Some(cost) = best_negative_residual_cost(edge, nflow) {
                let center_col = col.saturating_sub(1).min(pcn - 1);
                push_candidate(&mut candidates, cost, row, center_col);
            }
        }
    }

    let mut seen = HashSet::new();
    let mut windows = Vec::new();
    for (_cost, row, col) in candidates.into_sorted_vec() {
        let r0 = row.saturating_sub(LOCAL_WINDOW_RADIUS);
        let c0 = col.saturating_sub(LOCAL_WINDOW_RADIUS);
        let r1 = (row + LOCAL_WINDOW_RADIUS + 1).min(prn);
        let c1 = (col + LOCAL_WINDOW_RADIUS + 1).min(pcn);
        if r0 < r1 && c0 < c1 && seen.insert((r0, c0, r1, c1)) {
            windows.push((r0, c0, r1, c1));
        }
    }
    windows
}

fn best_negative_residual_cost(edge: EdgeDatum, nflow: i32) -> Option<i32> {
    let step = nflow.abs().max(1);
    let best = residual_arc_cost(edge, step).min(residual_arc_cost(edge, -step));
    (best < 0).then_some(best)
}

fn push_candidate(
    candidates: &mut BinaryHeap<(i32, usize, usize)>,
    cost: i32,
    row: usize,
    col: usize,
) {
    let candidate = (cost, row, col);
    if candidates.len() < LOCAL_WINDOW_LIMIT {
        candidates.push(candidate);
    } else if candidates.peek().is_some_and(|worst| candidate < *worst) {
        candidates.pop();
        candidates.push(candidate);
    }
}

fn local_node(row: usize, col: usize, r0: usize, c0: usize, width: usize) -> usize {
    (row - r0) * width + (col - c0)
}

fn push_arc_pair(
    arcs: &mut Vec<ResidualArc>,
    from: usize,
    to: usize,
    is_horizontal: bool,
    edge_index: usize,
    edge: EdgeDatum,
    step: i32,
) {
    arcs.push(ResidualArc {
        from,
        to,
        cost: residual_arc_cost(edge, step),
        is_horizontal,
        edge_index,
        correction_delta: step,
    });
    arcs.push(ResidualArc {
        from: to,
        to: from,
        cost: residual_arc_cost(edge, -step),
        is_horizontal,
        edge_index,
        correction_delta: -step,
    });
}

fn window_residual_arcs(
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    nflow: i32,
    r0: usize,
    c0: usize,
    r1: usize,
    c1: usize,
) -> Vec<ResidualArc> {
    let prn = nrow - 1;
    let pcn = ncol - 1;
    let width = c1 - c0;
    let ground = (r1 - r0) * width;
    let step = nflow.abs().max(1);
    let mut arcs = Vec::new();

    for col in c0..c1 {
        if r0 == 0 {
            if let Some(edge) = horizontal[horizontal_index(0, col, ncol)] {
                push_arc_pair(
                    &mut arcs,
                    ground,
                    local_node(0, col, r0, c0, width),
                    true,
                    horizontal_index(0, col, ncol),
                    edge,
                    step,
                );
            }
        }
        for row in (r0 + 1)..r1 {
            if let Some(edge) = horizontal[horizontal_index(row, col, ncol)] {
                push_arc_pair(
                    &mut arcs,
                    local_node(row - 1, col, r0, c0, width),
                    local_node(row, col, r0, c0, width),
                    true,
                    horizontal_index(row, col, ncol),
                    edge,
                    step,
                );
            }
        }
        if r1 == prn {
            if let Some(edge) = horizontal[horizontal_index(nrow - 1, col, ncol)] {
                push_arc_pair(
                    &mut arcs,
                    local_node(prn - 1, col, r0, c0, width),
                    ground,
                    true,
                    horizontal_index(nrow - 1, col, ncol),
                    edge,
                    step,
                );
            }
        }
    }

    for row in r0..r1 {
        if c0 == 0 {
            if let Some(edge) = vertical[vertical_index(row, 0, ncol)] {
                push_arc_pair(
                    &mut arcs,
                    local_node(row, 0, r0, c0, width),
                    ground,
                    false,
                    vertical_index(row, 0, ncol),
                    edge,
                    step,
                );
            }
        }
        for col in (c0 + 1)..c1 {
            if let Some(edge) = vertical[vertical_index(row, col, ncol)] {
                push_arc_pair(
                    &mut arcs,
                    local_node(row, col, r0, c0, width),
                    local_node(row, col - 1, r0, c0, width),
                    false,
                    vertical_index(row, col, ncol),
                    edge,
                    step,
                );
            }
        }
        if c1 == pcn {
            if let Some(edge) = vertical[vertical_index(row, ncol - 1, ncol)] {
                push_arc_pair(
                    &mut arcs,
                    ground,
                    local_node(row, pcn - 1, r0, c0, width),
                    false,
                    vertical_index(row, ncol - 1, ncol),
                    edge,
                    step,
                );
            }
        }
    }

    arcs
}
