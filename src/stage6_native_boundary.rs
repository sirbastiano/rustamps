use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::stage6_native_curl::plaquette_curl;
use super::{
    apply_edge_correction, edge_increment_cost, horizontal_index, vertical_index, EdgeDatum,
};

#[cfg(test)]
#[path = "stage6_native_boundary_tests.rs"]
mod boundary_route_tests;

const ROUTE_NONE: u8 = 255;
const ROUTE_UP: u8 = 0;
const ROUTE_RIGHT: u8 = 1;
const ROUTE_DOWN: u8 = 2;
const ROUTE_LEFT: u8 = 3;

#[derive(Clone, Copy, PartialEq)]
struct RouteState {
    cost: f64,
    node: usize,
}

impl Eq for RouteState {}

impl Ord for RouteState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for RouteState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn plaquette_index(row: usize, col: usize, ncol: usize) -> usize {
    row * (ncol - 1) + col
}

fn relax_route(
    node: usize,
    cost: f64,
    first_step: u8,
    dist: &mut [f64],
    next_step: &mut [u8],
    heap: &mut BinaryHeap<RouteState>,
) {
    if cost + 1.0e-12 < dist[node] {
        dist[node] = cost;
        next_step[node] = first_step;
        heap.push(RouteState { cost, node });
    }
}

fn route_step_increment(edge: EdgeDatum, step: u8, signed_amount: i32) -> f64 {
    let desired_delta = match step {
        ROUTE_UP | ROUTE_RIGHT => -signed_amount,
        ROUTE_DOWN | ROUTE_LEFT => signed_amount,
        _ => 0,
    };
    edge_increment_cost(edge, desired_delta).max(0.0)
}

fn compute_boundary_routes_for_amount(
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    signed_amount: i32,
) -> Vec<u8> {
    let prn = nrow.saturating_sub(1);
    let pcn = ncol.saturating_sub(1);
    let plaquette_count = prn * pcn;
    let mut dist = vec![f64::INFINITY; plaquette_count];
    let mut next_step = vec![ROUTE_NONE; plaquette_count];
    let mut heap = BinaryHeap::new();

    for row in 0..prn {
        for col in 0..pcn {
            let node = plaquette_index(row, col, ncol);
            if row == 0 {
                if let Some(edge) = horizontal[horizontal_index(0, col, ncol)] {
                    relax_route(
                        node,
                        route_step_increment(edge, ROUTE_UP, signed_amount),
                        ROUTE_UP,
                        &mut dist,
                        &mut next_step,
                        &mut heap,
                    );
                }
            }
            if row + 1 == prn {
                if let Some(edge) = horizontal[horizontal_index(nrow - 1, col, ncol)] {
                    relax_route(
                        node,
                        route_step_increment(edge, ROUTE_DOWN, signed_amount),
                        ROUTE_DOWN,
                        &mut dist,
                        &mut next_step,
                        &mut heap,
                    );
                }
            }
            if col == 0 {
                if let Some(edge) = vertical[vertical_index(row, 0, ncol)] {
                    relax_route(
                        node,
                        route_step_increment(edge, ROUTE_LEFT, signed_amount),
                        ROUTE_LEFT,
                        &mut dist,
                        &mut next_step,
                        &mut heap,
                    );
                }
            }
            if col + 1 == pcn {
                if let Some(edge) = vertical[vertical_index(row, ncol - 1, ncol)] {
                    relax_route(
                        node,
                        route_step_increment(edge, ROUTE_RIGHT, signed_amount),
                        ROUTE_RIGHT,
                        &mut dist,
                        &mut next_step,
                        &mut heap,
                    );
                }
            }
        }
    }

    while let Some(RouteState { cost, node }) = heap.pop() {
        if cost > dist[node] + 1.0e-12 {
            continue;
        }
        let row = node / pcn;
        let col = node - row * pcn;
        if row > 0 {
            if let Some(edge) = horizontal[horizontal_index(row, col, ncol)] {
                relax_route(
                    plaquette_index(row - 1, col, ncol),
                    cost + route_step_increment(edge, ROUTE_DOWN, signed_amount),
                    ROUTE_DOWN,
                    &mut dist,
                    &mut next_step,
                    &mut heap,
                );
            }
        }
        if row + 1 < prn {
            if let Some(edge) = horizontal[horizontal_index(row + 1, col, ncol)] {
                relax_route(
                    plaquette_index(row + 1, col, ncol),
                    cost + route_step_increment(edge, ROUTE_UP, signed_amount),
                    ROUTE_UP,
                    &mut dist,
                    &mut next_step,
                    &mut heap,
                );
            }
        }
        if col > 0 {
            if let Some(edge) = vertical[vertical_index(row, col, ncol)] {
                relax_route(
                    plaquette_index(row, col - 1, ncol),
                    cost + route_step_increment(edge, ROUTE_RIGHT, signed_amount),
                    ROUTE_RIGHT,
                    &mut dist,
                    &mut next_step,
                    &mut heap,
                );
            }
        }
        if col + 1 < pcn {
            if let Some(edge) = vertical[vertical_index(row, col + 1, ncol)] {
                relax_route(
                    plaquette_index(row, col + 1, ncol),
                    cost + route_step_increment(edge, ROUTE_LEFT, signed_amount),
                    ROUTE_LEFT,
                    &mut dist,
                    &mut next_step,
                    &mut heap,
                );
            }
        }
    }

    next_step
}

fn apply_route_step(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    row: usize,
    col: usize,
    step: u8,
    signed_amount: i32,
) -> Option<(usize, usize)> {
    match step {
        ROUTE_UP => {
            apply_edge_correction(
                &mut horizontal[horizontal_index(row, col, ncol)],
                -signed_amount,
            );
            (row > 0).then_some((row - 1, col))
        }
        ROUTE_RIGHT => {
            apply_edge_correction(
                &mut vertical[vertical_index(row, col + 1, ncol)],
                -signed_amount,
            );
            (col + 1 < ncol - 1).then_some((row, col + 1))
        }
        ROUTE_DOWN => {
            apply_edge_correction(
                &mut horizontal[horizontal_index(row + 1, col, ncol)],
                signed_amount,
            );
            (row + 1 < nrow - 1).then_some((row + 1, col))
        }
        ROUTE_LEFT => {
            apply_edge_correction(&mut vertical[vertical_index(row, col, ncol)], signed_amount);
            (col > 0).then_some((row, col - 1))
        }
        _ => None,
    }
}

pub(crate) fn route_residue_to_boundary(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) {
    if nrow < 2 || ncol < 2 {
        return;
    }
    let routes_pos = compute_boundary_routes_for_amount(horizontal, vertical, nrow, ncol, 1);
    let routes_neg = compute_boundary_routes_for_amount(horizontal, vertical, nrow, ncol, -1);
    let prn = nrow - 1;
    let pcn = ncol - 1;
    for row in 0..prn {
        for col in 0..pcn {
            let Some(curl) = plaquette_curl(horizontal, vertical, ncol, row, col) else {
                continue;
            };
            if curl == 0 {
                continue;
            }
            let routes = if curl > 0 { &routes_pos } else { &routes_neg };

            let mut current_row = row;
            let mut current_col = col;
            let mut guard = 0_usize;
            while guard <= routes.len() {
                guard += 1;
                let route_ix = plaquette_index(current_row, current_col, ncol);
                let step = routes[route_ix];
                if step == ROUTE_NONE {
                    break;
                }
                match apply_route_step(
                    horizontal,
                    vertical,
                    nrow,
                    ncol,
                    current_row,
                    current_col,
                    step,
                    curl,
                ) {
                    Some((next_row, next_col)) => {
                        current_row = next_row;
                        current_col = next_col;
                    }
                    None => break,
                }
            }
        }
    }
}
