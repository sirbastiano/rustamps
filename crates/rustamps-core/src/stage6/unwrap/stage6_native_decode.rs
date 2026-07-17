use std::f32::consts::PI;

use num_complex::Complex32;

use super::{horizontal_index, vertical_index, EdgeDatum};

pub(super) const TWO_PI: f32 = 2.0 * PI;

fn wrap_phase(value: f32) -> f32 {
    (value + PI).rem_euclid(TWO_PI) - PI
}

fn edge_is_valid(marker: i16) -> bool {
    marker != 0
}

fn edge_quality(cost: i16) -> i32 {
    i32::from(cost).abs().max(1)
}

fn desired_label_delta(phases: &[f32], from: usize, to: usize, cycle_hint: f32) -> f32 {
    let wrapped_delta = wrap_phase(phases[to] - phases[from]);
    ((phases[from] + wrapped_delta - phases[to]) / TWO_PI) + cycle_hint
}

pub(super) fn decode_wrapped_phases(ifgw: &[Complex32], nrow: usize, ncol: usize) -> Vec<f32> {
    let mut phases = vec![0.0_f32; nrow * ncol];
    for row in 0..nrow {
        for col in 0..ncol {
            let value = ifgw[row * ncol + col];
            phases[row * ncol + col] = value.im.atan2(value.re);
        }
    }
    phases
}

pub(super) fn decode_cost_edges(
    phases: &[f32],
    active: &[bool],
    nrow: usize,
    ncol: usize,
    rowcost: &[i16],
    colcost: &[i16],
    nshortcycle: f32,
) -> (Vec<Option<EdgeDatum>>, Vec<Option<EdgeDatum>>) {
    let nshortcycle_i32 = nshortcycle.round() as i32;
    let mut horizontal = if ncol > 1 {
        vec![None; nrow * (ncol - 1)]
    } else {
        Vec::new()
    };
    let mut vertical = if nrow > 1 {
        vec![None; (nrow - 1) * ncol]
    } else {
        Vec::new()
    };
    for row in 0..nrow.saturating_sub(1) {
        for col in 0..ncol {
            let base = row * (ncol * 4) + col * 4;
            let upper = row * ncol + col;
            let lower = (row + 1) * ncol + col;
            if active.get(upper).copied().unwrap_or(false)
                && active.get(lower).copied().unwrap_or(false)
                && edge_is_valid(rowcost[base + 3])
            {
                vertical[vertical_index(row, col, ncol)] = Some(EdgeDatum {
                    cost: edge_quality(rowcost[base + 1]),
                    desired_delta: desired_label_delta(phases, upper, lower, 0.0),
                    offset: i32::from(rowcost[base]),
                    dzmax: i32::from(rowcost[base + 2]),
                    laycost: i32::from(rowcost[base + 3]),
                    nshortcycle: nshortcycle_i32,
                    flow_sign: -1,
                    flow: 0,
                });
            }
        }
    }
    for row in 0..nrow {
        for col in 0..ncol.saturating_sub(1) {
            let base = row * ((ncol - 1) * 4) + col * 4;
            let left = row * ncol + col;
            let right = row * ncol + col + 1;
            if active.get(left).copied().unwrap_or(false)
                && active.get(right).copied().unwrap_or(false)
                && edge_is_valid(colcost[base + 3])
            {
                horizontal[horizontal_index(row, col, ncol)] = Some(EdgeDatum {
                    cost: edge_quality(colcost[base + 1]),
                    desired_delta: desired_label_delta(phases, left, right, 0.0),
                    offset: i32::from(colcost[base]),
                    dzmax: i32::from(colcost[base + 2]),
                    laycost: i32::from(colcost[base + 3]),
                    nshortcycle: nshortcycle_i32,
                    flow_sign: 1,
                    flow: 0,
                });
            }
        }
    }
    (horizontal, vertical)
}
