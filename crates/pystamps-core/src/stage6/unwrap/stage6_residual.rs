use crate::stage6::unwrap::incr_cost::defo_incremental_costs;
use crate::stage6::unwrap::native::{
    apply_edge_correction, horizontal_index, vertical_index, EdgeDatum,
};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResidualArc {
    pub(crate) from: usize,
    pub(crate) to: usize,
    pub(crate) cost: i32,
    pub(crate) is_horizontal: bool,
    pub(crate) edge_index: usize,
    pub(crate) correction_delta: i32,
}

#[inline(always)]
pub(crate) fn residual_arc_cost(edge: EdgeDatum, correction_delta: i32) -> i32 {
    let flow_delta = edge.flow_sign * correction_delta;
    let costs = defo_incremental_costs(edge, correction_delta.abs().max(1));
    if flow_delta > 0 {
        i32::from(costs.pos)
    } else if flow_delta < 0 {
        i32::from(costs.neg)
    } else {
        0
    }
}

fn push_arc_pair(
    arcs: &mut Vec<ResidualArc>,
    from: usize,
    to: usize,
    is_horizontal: bool,
    edge_index: usize,
    edge: EdgeDatum,
    correction_step: i32,
) {
    arcs.push(ResidualArc {
        from,
        to,
        cost: residual_arc_cost(edge, correction_step),
        is_horizontal,
        edge_index,
        correction_delta: correction_step,
    });
    arcs.push(ResidualArc {
        from: to,
        to: from,
        cost: residual_arc_cost(edge, -correction_step),
        is_horizontal,
        edge_index,
        correction_delta: -correction_step,
    });
}

pub(crate) fn build_unit_residual_arcs(
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) -> Vec<ResidualArc> {
    build_residual_arcs_with_nflow(horizontal, vertical, nrow, ncol, 1)
}

pub(crate) fn build_residual_arcs_with_nflow(
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    nflow: i32,
) -> Vec<ResidualArc> {
    let prn = nrow.saturating_sub(1);
    let pcn = ncol.saturating_sub(1);
    if prn == 0 || pcn == 0 {
        return Vec::new();
    }
    let correction_step = nflow.abs().max(1);
    let ground = prn * pcn;
    let mut arcs = Vec::with_capacity((horizontal.len() + vertical.len()) * 2);

    for row in 0..nrow {
        for col in 0..pcn {
            let index = horizontal_index(row, col, ncol);
            let Some(edge) = horizontal[index] else {
                continue;
            };
            if row == 0 {
                push_arc_pair(&mut arcs, ground, col, true, index, edge, correction_step);
            } else if row == nrow - 1 {
                push_arc_pair(
                    &mut arcs,
                    (row - 1) * pcn + col,
                    ground,
                    true,
                    index,
                    edge,
                    correction_step,
                );
            } else {
                push_arc_pair(
                    &mut arcs,
                    (row - 1) * pcn + col,
                    row * pcn + col,
                    true,
                    index,
                    edge,
                    correction_step,
                );
            }
        }
    }

    for row in 0..prn {
        for col in 0..ncol {
            let index = vertical_index(row, col, ncol);
            let Some(edge) = vertical[index] else {
                continue;
            };
            if col == 0 {
                push_arc_pair(
                    &mut arcs,
                    row * pcn,
                    ground,
                    false,
                    index,
                    edge,
                    correction_step,
                );
            } else if col == ncol - 1 {
                push_arc_pair(
                    &mut arcs,
                    ground,
                    row * pcn + col - 1,
                    false,
                    index,
                    edge,
                    correction_step,
                );
            } else {
                push_arc_pair(
                    &mut arcs,
                    row * pcn + col,
                    row * pcn + col - 1,
                    false,
                    index,
                    edge,
                    correction_step,
                );
            }
        }
    }
    arcs
}

pub(crate) fn residual_cycle_cost(arcs: &[ResidualArc], cycle: &[usize]) -> i32 {
    cycle.iter().map(|&index| arcs[index].cost).sum()
}

pub(crate) fn find_negative_unit_cycle(
    arcs: &[ResidualArc],
    node_count: usize,
) -> Option<Vec<usize>> {
    if node_count == 0 {
        return None;
    }
    let mut dist = vec![0_i64; node_count];
    let mut pred_node = vec![None::<usize>; node_count];
    let mut pred_arc = vec![None::<usize>; node_count];
    let mut relaxed = None;

    for _ in 0..node_count {
        relaxed = None;
        for (index, arc) in arcs.iter().enumerate() {
            if arc.from >= node_count || arc.to >= node_count {
                continue;
            }
            if pred_arc[arc.from].is_some_and(|prior| {
                arcs[prior].is_horizontal == arc.is_horizontal
                    && arcs[prior].edge_index == arc.edge_index
            }) {
                continue;
            }
            let next = dist[arc.from] + i64::from(arc.cost);
            if next < dist[arc.to] {
                dist[arc.to] = next;
                pred_node[arc.to] = Some(arc.from);
                pred_arc[arc.to] = Some(index);
                relaxed = Some(arc.to);
            }
        }
        if relaxed.is_none() {
            return None;
        }
    }

    let mut node = relaxed?;
    for _ in 0..node_count {
        node = pred_node[node]?;
    }
    let start = node;
    let mut cycle = Vec::new();
    loop {
        let arc = pred_arc[node]?;
        cycle.push(arc);
        node = pred_node[node]?;
        if node == start {
            break;
        }
        if cycle.len() > arcs.len() {
            return None;
        }
    }
    (residual_cycle_cost(arcs, &cycle) < 0).then_some(cycle)
}

pub(crate) fn apply_residual_cycle(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    arcs: &[ResidualArc],
    cycle: &[usize],
) {
    for &index in cycle {
        let arc = arcs[index];
        let edge = if arc.is_horizontal {
            &mut horizontal[arc.edge_index]
        } else {
            &mut vertical[arc.edge_index]
        };
        apply_edge_correction(edge, arc.correction_delta);
    }
}

pub(crate) fn saturate_residual_cycle(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    arcs: &[ResidualArc],
    cycle: &[usize],
) -> usize {
    let mut seen = HashSet::with_capacity(cycle.len());
    if cycle.iter().any(|&index| {
        arcs.get(index)
            .is_none_or(|arc| !seen.insert((arc.is_horizontal, arc.edge_index)))
    }) {
        return 0;
    }
    let mut increments = 0;
    loop {
        let cost = cycle.iter().try_fold(0_i64, |sum, &index| {
            let arc = arcs.get(index)?;
            let edge = if arc.is_horizontal {
                horizontal.get(arc.edge_index).and_then(|edge| *edge)?
            } else {
                vertical.get(arc.edge_index).and_then(|edge| *edge)?
            };
            Some(sum + i64::from(residual_arc_cost(edge, arc.correction_delta)))
        });
        if cost.is_none_or(|value| value >= 0) {
            return increments;
        }
        apply_residual_cycle(horizontal, vertical, arcs, cycle);
        increments += 1;
    }
}

pub(crate) fn cancel_negative_unit_cycles(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    max_cycles: usize,
) -> usize {
    cancel_negative_cycles_with_nflow(horizontal, vertical, nrow, ncol, 1, max_cycles)
}

pub(crate) fn cancel_negative_cycles_with_nflow(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    nflow: i32,
    max_cycles: usize,
) -> usize {
    let node_count = nrow.saturating_sub(1) * ncol.saturating_sub(1) + 1;
    let mut applied = 0;
    for _ in 0..max_cycles {
        let arcs = build_residual_arcs_with_nflow(horizontal, vertical, nrow, ncol, nflow);
        let Some(cycle) = find_negative_unit_cycle(&arcs, node_count) else {
            break;
        };
        if saturate_residual_cycle(horizontal, vertical, &arcs, &cycle) == 0 {
            break;
        }
        applied += 1;
    }
    applied
}
