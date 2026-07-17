use super::grid::Grid;
use super::interp::Interpolation;
use super::space_time::SpaceTime;

pub struct Costs {
    pub row_indices: Vec<f64>,
    pub col_indices: Vec<f64>,
    pub row_base: Vec<i16>,
    pub col_base: Vec<i16>,
}

pub fn build(grid: &Grid, interp: &Interpolation, space: &SpaceTime, n_ifg: usize) -> Costs {
    let edge_count = interp.edges.len();
    let mut occurrences = vec![0_usize; edge_count];
    for value in interp.row_indices.iter().chain(&interp.col_indices) {
        if value.is_finite() && *value != 0.0 {
            occurrences[value.abs() as usize - 1] += 1;
        }
    }
    let mut variance = vec![f64::NAN; edge_count];
    for edge in 0..edge_count {
        let row = &space.noise[edge * n_ifg..(edge + 1) * n_ifg];
        let mean = row.iter().map(|&value| f64::from(value)).sum::<f64>() / n_ifg as f64;
        let divisor = n_ifg.saturating_sub(usize::from(n_ifg > 1)).max(1) as f64;
        let deviation = (row
            .iter()
            .map(|&value| (f64::from(value) - mean).powi(2))
            .sum::<f64>()
            / divisor)
            .sqrt();
        variance[edge] = (deviation / std::f64::consts::TAU).powi(2);
    }
    let mut row_indices = interp.row_indices.clone();
    let mut col_indices = interp.col_indices.clone();
    mask_bad_edges(&mut row_indices, &variance);
    mask_bad_edges(&mut col_indices, &variance);
    let sigsq = variance
        .iter()
        .zip(&occurrences)
        .map(|(value, count)| {
            let raw = (value * 200.0_f64.powi(2) / 100.0 * *count as f64).round_ties_even();
            if raw.is_finite() {
                raw.clamp(1.0, i16::MAX as f64) as i16
            } else {
                1
            }
        })
        .collect::<Vec<_>>();
    let row_base = base_costs(&row_indices, grid.rows.saturating_sub(1), grid.cols, &sigsq);
    let col_base = base_costs(&col_indices, grid.rows, grid.cols.saturating_sub(1), &sigsq);
    Costs {
        row_indices,
        col_indices,
        row_base,
        col_base,
    }
}

fn mask_bad_edges(indices: &mut [f64], variance: &[f64]) {
    for value in indices {
        if value.is_finite() && *value != 0.0 && !variance[value.abs() as usize - 1].is_finite() {
            *value = f64::NAN;
        }
    }
}

fn base_costs(indices: &[f64], rows: usize, arcs: usize, sigsq: &[i16]) -> Vec<i16> {
    let mut costs = vec![0_i16; rows * arcs * 4];
    for row in 0..rows {
        for col in 0..arcs {
            let edge = indices[row * arcs + col];
            let base = (row * arcs + col) * 4;
            costs[base + 1] = if edge.is_finite() && edge != 0.0 {
                sigsq[edge.abs() as usize - 1]
            } else {
                1
            };
            costs[base + 2] = 32_000;
            costs[base + 3] = if edge.is_nan() { 1 } else { -32_000 };
        }
    }
    costs
}
