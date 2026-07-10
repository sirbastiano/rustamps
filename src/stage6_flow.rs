use crate::stage6_native::{
    apply_edge_correction, horizontal_index, rounded_delta, vertical_index, EdgeDatum,
};
use crate::stage6_route::{route_path, STEP_DOWN, STEP_LEFT, STEP_RIGHT, STEP_UP};
const MAX_PAIR_RADIUS: usize = 256;

struct PairCandidate {
    cost: f64,
    source: (usize, usize),
    target: (usize, usize),
}
fn plaquette_curl(
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    ncol: usize,
    row: usize,
    col: usize,
) -> Option<i32> {
    let top = horizontal[horizontal_index(row, col, ncol)]?;
    let right = vertical[vertical_index(row, col + 1, ncol)]?;
    let bottom = horizontal[horizontal_index(row + 1, col, ncol)]?;
    let left = vertical[vertical_index(row, col, ncol)]?;
    Some(rounded_delta(top) + rounded_delta(right) - rounded_delta(bottom) - rounded_delta(left))
}

fn apply_pair_steps(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    ncol: usize,
    source: (usize, usize),
    steps: &[u8],
    signed_amount: i32,
) {
    let (mut row, mut col) = source;
    for &step in steps {
        match step {
            STEP_UP => {
                apply_edge_correction(
                    &mut horizontal[horizontal_index(row, col, ncol)],
                    -signed_amount,
                );
                row -= 1;
            }
            STEP_RIGHT => {
                apply_edge_correction(
                    &mut vertical[vertical_index(row, col + 1, ncol)],
                    -signed_amount,
                );
                col += 1;
            }
            STEP_DOWN => {
                apply_edge_correction(
                    &mut horizontal[horizontal_index(row + 1, col, ncol)],
                    signed_amount,
                );
                row += 1;
            }
            STEP_LEFT => {
                apply_edge_correction(&mut vertical[vertical_index(row, col, ncol)], signed_amount);
                col -= 1;
            }
            _ => {}
        }
    }
}

fn collect_local_pairs(
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    source: (usize, usize),
    candidates: &mut Vec<PairCandidate>,
) {
    let Some(source_curl) = plaquette_curl(horizontal, vertical, ncol, source.0, source.1) else {
        return;
    };
    if source_curl == 0 {
        return;
    }
    let prn = nrow - 1;
    let pcn = ncol - 1;
    for radius in 1..=MAX_PAIR_RADIUS {
        let row_start = source.0.saturating_sub(radius);
        let row_end = (source.0 + radius).min(prn - 1);
        for row in row_start..=row_end {
            let row_distance = row.abs_diff(source.0);
            let col_distance = radius - row_distance;
            for col in [
                source.1.saturating_sub(col_distance),
                source.1 + col_distance,
            ] {
                if col >= pcn || (row, col) == source {
                    continue;
                }
                let Some(target_curl) = plaquette_curl(horizontal, vertical, ncol, row, col) else {
                    continue;
                };
                if source_curl.signum() + target_curl.signum() != 0 {
                    continue;
                }
                let amount = source_curl.abs().min(target_curl.abs());
                if amount == 0 {
                    continue;
                }
                let Some((cost, steps)) = route_path(
                    horizontal,
                    vertical,
                    nrow,
                    ncol,
                    source,
                    (row, col),
                    source_curl.signum() * amount,
                ) else {
                    continue;
                };
                let _ = steps;
                candidates.push(PairCandidate {
                    cost,
                    source,
                    target: (row, col),
                });
            }
        }
    }
}

fn pair_global_residues(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) -> bool {
    let prn = nrow - 1;
    let pcn = ncol - 1;
    let mut candidates = Vec::new();
    for row in 0..prn {
        for col in 0..pcn {
            collect_local_pairs(
                horizontal,
                vertical,
                nrow,
                ncol,
                (row, col),
                &mut candidates,
            );
        }
    }
    candidates.sort_by(|a, b| {
        a.cost
            .partial_cmp(&b.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut changed = false;
    for candidate in candidates {
        let Some(source_curl) = plaquette_curl(
            horizontal,
            vertical,
            ncol,
            candidate.source.0,
            candidate.source.1,
        ) else {
            continue;
        };
        let Some(target_curl) = plaquette_curl(
            horizontal,
            vertical,
            ncol,
            candidate.target.0,
            candidate.target.1,
        ) else {
            continue;
        };
        if source_curl == 0 || source_curl.signum() + target_curl.signum() != 0 {
            continue;
        }
        let amount = source_curl.abs().min(target_curl.abs());
        if amount == 0 {
            continue;
        }
        let signed_amount = source_curl.signum() * amount;
        let Some((_cost, steps)) = route_path(
            horizontal,
            vertical,
            nrow,
            ncol,
            candidate.source,
            candidate.target,
            signed_amount,
        ) else {
            continue;
        };
        apply_pair_steps(
            horizontal,
            vertical,
            ncol,
            candidate.source,
            &steps,
            signed_amount,
        );
        changed = true;
    }
    changed
}

pub(crate) fn pair_neighbor_residues(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) {
    if nrow < 2 || ncol < 2 {
        return;
    }
    let prn = nrow - 1;
    let pcn = ncol - 1;
    for _ in 0..16 {
        let mut changed = false;
        for row in 0..prn {
            for col in 0..pcn.saturating_sub(1) {
                let Some(left_curl) = plaquette_curl(horizontal, vertical, ncol, row, col) else {
                    continue;
                };
                let Some(right_curl) = plaquette_curl(horizontal, vertical, ncol, row, col + 1)
                else {
                    continue;
                };
                if left_curl.signum() + right_curl.signum() != 0 {
                    continue;
                }
                let amount = left_curl.abs().min(right_curl.abs());
                if amount == 0 {
                    continue;
                }
                let delta = if left_curl > 0 { -amount } else { amount };
                apply_edge_correction(&mut vertical[vertical_index(row, col + 1, ncol)], delta);
                changed = true;
            }
        }
        for row in 0..prn.saturating_sub(1) {
            for col in 0..pcn {
                let Some(top_curl) = plaquette_curl(horizontal, vertical, ncol, row, col) else {
                    continue;
                };
                let Some(bottom_curl) = plaquette_curl(horizontal, vertical, ncol, row + 1, col)
                else {
                    continue;
                };
                if top_curl.signum() + bottom_curl.signum() != 0 {
                    continue;
                }
                let amount = top_curl.abs().min(bottom_curl.abs());
                if amount == 0 {
                    continue;
                }
                let delta = if top_curl > 0 { amount } else { -amount };
                apply_edge_correction(&mut horizontal[horizontal_index(row + 1, col, ncol)], delta);
                changed = true;
            }
        }
        changed |= pair_global_residues(horizontal, vertical, nrow, ncol);
        if !changed {
            break;
        }
    }
}
