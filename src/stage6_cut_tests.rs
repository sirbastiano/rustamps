use super::refine_labels_by_binary_cuts;
use super::stage6_cut_windows::cut_windows;
use crate::stage6_native::{horizontal_index, EdgeDatum};

fn edge(cost: i32, offset: i32, flow_sign: i32) -> EdgeDatum {
    EdgeDatum {
        cost,
        desired_delta: 0.0,
        offset,
        dzmax: 32000,
        laycost: -32000,
        nshortcycle: 200,
        flow_sign,
        flow: 0,
    }
}

fn edge_with_laycost(cost: i32, offset: i32, laycost: i32, flow_sign: i32) -> EdgeDatum {
    EdgeDatum {
        cost,
        desired_delta: 0.0,
        offset,
        dzmax: 32000,
        laycost,
        nshortcycle: 200,
        flow_sign,
        flow: 0,
    }
}

#[test]
fn binary_cuts_refine_large_grid_by_tiled_windows() {
    let nrow = 130;
    let ncol = 130;
    let mut labels = vec![0_i32; nrow * ncol];
    let mut horizontal = vec![Some(edge(32000, 0, 1)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge(32000, 0, -1)); (nrow - 1) * ncol];
    horizontal[horizontal_index(10, 10, ncol)] = Some(edge(1000, 200, 1));

    refine_labels_by_binary_cuts(&mut labels, &horizontal, &vertical, nrow, ncol);

    let nonzero = labels.iter().filter(|&&label| label != 0).count();
    assert!(nonzero > 0, "expected at least one shifted label");
    assert_eq!(labels[10 * ncol + 11] - labels[10 * ncol + 10], -1);
}

#[test]
fn cut_windows_keeps_distributed_energy_window_when_single_edge_outliers_dominate() {
    let nrow = 70;
    let ncol = 70;
    let labels = vec![0_i32; nrow * ncol];
    let mut horizontal = vec![None; nrow * (ncol - 1)];
    let vertical = vec![None; (nrow - 1) * ncol];

    let mut inserted = 0_usize;
    for row in 0..35 {
        for col in (0..ncol - 1).step_by(4) {
            if (36..=48).contains(&row) && (36..=48).contains(&col) {
                continue;
            }
            horizontal[horizontal_index(row, col, ncol)] = Some(edge(1000, 400, 1));
            inserted += 1;
            if inserted == 600 {
                break;
            }
        }
        if inserted == 600 {
            break;
        }
    }
    for row in 40..48 {
        horizontal[horizontal_index(row, 40, ncol)] = Some(edge(1000, 200, 1));
        horizontal[horizontal_index(row, 47, ncol)] = Some(edge(1000, -200, 1));
    }

    let windows = cut_windows(&labels, &horizontal, &vertical, nrow, ncol, 64);

    assert!(
        windows.iter().any(|&(row, col, height, width)| {
            row <= 40 && row + height > 47 && col <= 40 && col + width > 47
        }),
        "expected an aggregate-energy candidate window to cover the distributed boundary"
    );
}

#[test]
fn cut_windows_scores_reducible_energy_over_flat_plateau_outliers() {
    let nrow = 120;
    let ncol = 120;
    let labels = vec![0_i32; nrow * ncol];
    let mut horizontal = vec![None; nrow * (ncol - 1)];
    let vertical = vec![None; (nrow - 1) * ncol];

    let mut inserted = 0_usize;
    for row in (0..nrow).step_by(4) {
        for col in (0..ncol - 1).step_by(4) {
            if (76..=84).contains(&row) && (76..=84).contains(&col) {
                continue;
            }
            horizontal[horizontal_index(row, col, ncol)] =
                Some(edge_with_laycost(1000, 2000, 1000, 1));
            inserted += 1;
            if inserted == 600 {
                break;
            }
        }
        if inserted == 600 {
            break;
        }
    }
    for row in 80..84 {
        horizontal[horizontal_index(row, 80, ncol)] = Some(edge(1000, 200, 1));
        horizontal[horizontal_index(row, 83, ncol)] = Some(edge(1000, -200, 1));
    }

    let windows = cut_windows(&labels, &horizontal, &vertical, nrow, ncol, 16);

    assert!(
        windows.iter().any(|&(row, col, height, width)| {
            row <= 80 && row + height > 83 && col <= 80 && col + width > 83
        }),
        "expected reducible distributed energy to outrank flat high-cost plateau edges"
    );
}

#[test]
fn cut_windows_keeps_midrank_potential_window_on_large_grids() {
    let nrow = 120;
    let ncol = 120;
    let labels = vec![0_i32; nrow * ncol];
    let mut horizontal = vec![None; nrow * (ncol - 1)];
    let vertical = vec![None; (nrow - 1) * ncol];

    let mut inserted = 0_usize;
    'outer: for row in (0..nrow).step_by(4) {
        for col in (0..ncol - 1).step_by(4) {
            if (96..=104).contains(&row) && (96..=104).contains(&col) {
                continue;
            }
            horizontal[horizontal_index(row, col, ncol)] = Some(edge(1000, 300, 1));
            inserted += 1;
            if inserted == 250 {
                break 'outer;
            }
        }
    }
    horizontal[horizontal_index(100, 100, ncol)] = Some(edge(1000, 120, 1));

    let windows = cut_windows(&labels, &horizontal, &vertical, nrow, ncol, 16);

    assert!(
        windows.iter().any(|&(row, col, height, width)| {
            row <= 100 && row + height > 103 && col <= 100 && col + width > 103
        }),
        "expected midrank potential window to be retained on large grids"
    );
}
