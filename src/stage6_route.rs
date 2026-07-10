use crate::stage6_native::{edge_increment_cost, horizontal_index, vertical_index, EdgeDatum};

pub(crate) const STEP_UP: u8 = 0;
pub(crate) const STEP_RIGHT: u8 = 1;
pub(crate) const STEP_DOWN: u8 = 2;
pub(crate) const STEP_LEFT: u8 = 3;
const DETOUR_MARGIN: usize = 16;

fn push_step_cost(
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    row: &mut usize,
    col: &mut usize,
    step: u8,
    signed_amount: i32,
    steps: &mut Vec<u8>,
    cost: &mut f64,
) -> bool {
    match step {
        STEP_UP if *row > 0 => {
            let Some(edge) = horizontal[horizontal_index(*row, *col, ncol)] else {
                return false;
            };
            *cost += edge_increment_cost(edge, -signed_amount);
            *row -= 1;
        }
        STEP_RIGHT if *col + 1 < ncol - 1 => {
            let Some(edge) = vertical[vertical_index(*row, *col + 1, ncol)] else {
                return false;
            };
            *cost += edge_increment_cost(edge, -signed_amount);
            *col += 1;
        }
        STEP_DOWN if *row + 1 < nrow - 1 => {
            let Some(edge) = horizontal[horizontal_index(*row + 1, *col, ncol)] else {
                return false;
            };
            *cost += edge_increment_cost(edge, signed_amount);
            *row += 1;
        }
        STEP_LEFT if *col > 0 => {
            let Some(edge) = vertical[vertical_index(*row, *col, ncol)] else {
                return false;
            };
            *cost += edge_increment_cost(edge, signed_amount);
            *col -= 1;
        }
        _ => return false,
    }
    steps.push(step);
    true
}

fn route_segment(
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    from: &mut (usize, usize),
    target: (usize, usize),
    signed_amount: i32,
    vertical_first: bool,
    steps: &mut Vec<u8>,
    cost: &mut f64,
) -> bool {
    let row_step = [STEP_DOWN, STEP_UP][usize::from(target.0 < from.0)];
    let col_step = [STEP_RIGHT, STEP_LEFT][usize::from(target.1 < from.1)];
    let row_segment = (row_step, from.0.abs_diff(target.0));
    let col_segment = (col_step, from.1.abs_diff(target.1));
    let segments = if vertical_first {
        [row_segment, col_segment]
    } else {
        [col_segment, row_segment]
    };
    for (step, count) in segments {
        for _ in 0..count {
            if !push_step_cost(
                horizontal,
                vertical,
                nrow,
                ncol,
                &mut from.0,
                &mut from.1,
                step,
                signed_amount,
                steps,
                cost,
            ) {
                return false;
            }
        }
    }
    true
}

fn route_points(
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    source: (usize, usize),
    points: &[(usize, usize)],
    signed_amount: i32,
    vertical_first: bool,
) -> Option<(f64, Vec<u8>)> {
    let mut current = source;
    let mut steps = Vec::new();
    let mut cost = 0.0_f64;
    for &target in points {
        if !route_segment(
            horizontal,
            vertical,
            nrow,
            ncol,
            &mut current,
            target,
            signed_amount,
            vertical_first,
            &mut steps,
            &mut cost,
        ) {
            return None;
        }
    }
    Some((cost, steps))
}

fn keep_best(best: &mut Option<(f64, Vec<u8>)>, candidate: Option<(f64, Vec<u8>)>) {
    if let Some((cost, steps)) = candidate {
        if best.as_ref().is_none_or(|(best_cost, _)| cost < *best_cost) {
            *best = Some((cost, steps));
        }
    }
}

pub(crate) fn route_path(
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    source: (usize, usize),
    target: (usize, usize),
    signed_amount: i32,
) -> Option<(f64, Vec<u8>)> {
    let mut best = None;
    for vertical_first in [false, true] {
        keep_best(
            &mut best,
            route_points(
                horizontal,
                vertical,
                nrow,
                ncol,
                source,
                &[target],
                signed_amount,
                vertical_first,
            ),
        );
    }
    if source.0 == target.0 {
        for detour in 1..=DETOUR_MARGIN {
            for row in [source.0.wrapping_sub(detour), source.0 + detour] {
                if row < nrow.saturating_sub(1) {
                    keep_best(
                        &mut best,
                        route_points(
                            horizontal,
                            vertical,
                            nrow,
                            ncol,
                            source,
                            &[(row, source.1), (row, target.1), target],
                            signed_amount,
                            true,
                        ),
                    );
                }
            }
        }
    }
    if source.1 == target.1 {
        for detour in 1..=DETOUR_MARGIN {
            for col in [source.1.wrapping_sub(detour), source.1 + detour] {
                if col < ncol.saturating_sub(1) {
                    keep_best(
                        &mut best,
                        route_points(
                            horizontal,
                            vertical,
                            nrow,
                            ncol,
                            source,
                            &[(source.0, col), (target.0, col), target],
                            signed_amount,
                            false,
                        ),
                    );
                }
            }
        }
    }
    best
}
