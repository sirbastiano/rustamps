use crate::stage6_component_shift::{
    refine_labels_by_barrier_component_shifts, refine_labels_by_component_shifts,
};
use crate::stage6_native::{edge_label_energy, horizontal_index, vertical_index, EdgeDatum};

fn edge(flow_sign: i32) -> EdgeDatum {
    EdgeDatum {
        cost: 1000,
        desired_delta: 0.0,
        offset: 0,
        dzmax: 32000,
        laycost: -32000,
        nshortcycle: 200,
        flow_sign,
        flow: 0,
    }
}

fn edge_with_offset(flow_sign: i32, offset: i32) -> EdgeDatum {
    EdgeDatum {
        cost: 1000,
        desired_delta: 0.0,
        offset,
        dzmax: 32000,
        laycost: -32000,
        nshortcycle: 200,
        flow_sign,
        flow: 0,
    }
}

fn objective(
    labels: &[i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) -> i64 {
    let mut total = 0_i64;
    for row in 0..nrow {
        for col in 0..ncol.saturating_sub(1) {
            if let Some(edge) = horizontal[horizontal_index(row, col, ncol)] {
                total +=
                    edge_label_energy(edge, labels[row * ncol + col], labels[row * ncol + col + 1]);
            }
        }
    }
    for row in 0..nrow.saturating_sub(1) {
        for col in 0..ncol {
            if let Some(edge) = vertical[vertical_index(row, col, ncol)] {
                total += edge_label_energy(
                    edge,
                    labels[row * ncol + col],
                    labels[(row + 1) * ncol + col],
                );
            }
        }
    }
    total
}

#[test]
fn component_shift_reduces_irregular_region_boundary_objective() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![None; nrow * (ncol - 1)];
    let mut vertical = vec![None; (nrow - 1) * ncol];
    for row in 0..nrow {
        for col in 0..ncol - 1 {
            horizontal[horizontal_index(row, col, ncol)] = Some(edge(1));
        }
    }
    for row in 0..nrow - 1 {
        for col in 0..ncol {
            vertical[vertical_index(row, col, ncol)] = Some(edge(-1));
        }
    }
    let mut labels = vec![1, 1, 1, 1, 0, 0, 1, 0, 0];
    let before = objective(&labels, &horizontal, &vertical, nrow, ncol);

    let applied =
        refine_labels_by_component_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    let after = objective(&labels, &horizontal, &vertical, nrow, ncol);

    assert!(applied > 0);
    assert!(after < before);
    assert!(labels.iter().all(|value| *value == labels[0]));
}

#[test]
fn barrier_component_shift_splits_equal_label_grid_on_high_cost_boundary() {
    let nrow = 3;
    let ncol = 4;
    let split = 2;
    let mut horizontal = vec![Some(edge_with_offset(1, 0)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge_with_offset(-1, 0)); (nrow - 1) * ncol];
    for row in 0..nrow {
        horizontal[horizontal_index(row, split - 1, ncol)] = Some(edge_with_offset(1, 200));
    }
    let mut labels = vec![0_i32; nrow * ncol];
    let before = objective(&labels, &horizontal, &vertical, nrow, ncol);

    let applied =
        refine_labels_by_barrier_component_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    let after = objective(&labels, &horizontal, &vertical, nrow, ncol);

    assert!(applied > 0);
    assert!(after < before);
    for row in 0..nrow {
        assert_eq!(
            labels[row * ncol + split] - labels[row * ncol + split - 1],
            -1
        );
        assert_eq!(labels[row * ncol], labels[row * ncol + split - 1]);
        assert_eq!(labels[row * ncol + split], labels[row * ncol + ncol - 1]);
    }
}

#[test]
fn barrier_component_shift_uses_lower_threshold_when_mean_is_outlier_inflated() {
    let nrow = 2;
    let ncol = 4;
    let split = 2;
    let mut horizontal = vec![Some(edge_with_offset(1, 0)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge_with_offset(-1, 0)); (nrow - 1) * ncol];
    for row in 0..nrow {
        horizontal[horizontal_index(row, split - 1, ncol)] = Some(edge_with_offset(1, 200));
    }
    horizontal[horizontal_index(0, 0, ncol)] = Some(edge_with_offset(1, 316));
    let mut labels = vec![0_i32; nrow * ncol];
    let before = objective(&labels, &horizontal, &vertical, nrow, ncol);

    let applied =
        refine_labels_by_barrier_component_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    let after = objective(&labels, &horizontal, &vertical, nrow, ncol);

    assert!(applied > 0);
    assert!(after < before);
    for row in 0..nrow {
        assert_eq!(
            labels[row * ncol + split] - labels[row * ncol + split - 1],
            -1
        );
    }
}

#[test]
fn barrier_component_shift_uses_quarter_threshold_when_half_mean_is_still_too_high() {
    let nrow = 2;
    let ncol = 4;
    let split = 2;
    let mut horizontal = vec![Some(edge_with_offset(1, 0)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge_with_offset(-1, 0)); (nrow - 1) * ncol];
    for row in 0..nrow {
        horizontal[horizontal_index(row, split - 1, ncol)] = Some(edge_with_offset(1, 142));
    }
    horizontal[horizontal_index(0, 0, ncol)] = Some(edge_with_offset(1, 316));
    let mut labels = vec![0_i32; nrow * ncol];
    let before = objective(&labels, &horizontal, &vertical, nrow, ncol);

    let applied =
        refine_labels_by_barrier_component_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    let after = objective(&labels, &horizontal, &vertical, nrow, ncol);

    assert!(applied > 0);
    assert!(after < before);
    for row in 0..nrow {
        assert_eq!(
            labels[row * ncol + split] - labels[row * ncol + split - 1],
            -1
        );
    }
}

#[test]
fn barrier_component_shift_uses_eighth_threshold_when_quarter_mean_is_still_too_high() {
    let nrow = 2;
    let ncol = 4;
    let split = 2;
    let mut horizontal = vec![Some(edge_with_offset(1, 0)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge_with_offset(-1, 0)); (nrow - 1) * ncol];
    for row in 0..nrow {
        horizontal[horizontal_index(row, split - 1, ncol)] = Some(edge_with_offset(1, 130));
    }
    horizontal[horizontal_index(0, 0, ncol)] = Some(edge_with_offset(1, 600));
    let mut labels = vec![0_i32; nrow * ncol];
    let before = objective(&labels, &horizontal, &vertical, nrow, ncol);

    let applied =
        refine_labels_by_barrier_component_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    let after = objective(&labels, &horizontal, &vertical, nrow, ncol);

    assert!(applied > 0);
    assert!(after < before);
    for row in 0..nrow {
        assert_eq!(
            labels[row * ncol + split] - labels[row * ncol + split - 1],
            -1
        );
    }
}

#[test]
fn barrier_component_shift_uses_sixteenth_threshold_when_eighth_mean_is_still_too_high() {
    let nrow = 2;
    let ncol = 4;
    let split = 2;
    let mut horizontal = vec![Some(edge_with_offset(1, 0)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge_with_offset(-1, 0)); (nrow - 1) * ncol];
    for row in 0..nrow {
        horizontal[horizontal_index(row, split - 1, ncol)] = Some(edge_with_offset(1, 180));
    }
    horizontal[horizontal_index(0, 0, ncol)] = Some(edge_with_offset(1, 1200));
    let mut labels = vec![0_i32; nrow * ncol];
    let before = objective(&labels, &horizontal, &vertical, nrow, ncol);

    let applied =
        refine_labels_by_barrier_component_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    let after = objective(&labels, &horizontal, &vertical, nrow, ncol);

    assert!(applied > 0);
    assert!(after < before);
    for row in 0..nrow {
        assert_eq!(
            labels[row * ncol + split] - labels[row * ncol + split - 1],
            -1
        );
    }
}

#[test]
fn barrier_component_shift_uses_thirty_second_threshold_when_sixteenth_mean_is_still_too_high() {
    let nrow = 2;
    let ncol = 4;
    let split = 2;
    let mut horizontal = vec![Some(edge_with_offset(1, 0)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge_with_offset(-1, 0)); (nrow - 1) * ncol];
    for row in 0..nrow {
        horizontal[horizontal_index(row, split - 1, ncol)] = Some(edge_with_offset(1, 255));
    }
    horizontal[horizontal_index(0, 0, ncol)] = Some(edge_with_offset(1, 2400));
    let mut labels = vec![0_i32; nrow * ncol];
    let before = objective(&labels, &horizontal, &vertical, nrow, ncol);

    let applied =
        refine_labels_by_barrier_component_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    let after = objective(&labels, &horizontal, &vertical, nrow, ncol);

    assert!(applied > 0);
    assert!(after < before);
    for row in 0..nrow {
        assert_eq!(
            labels[row * ncol + split] - labels[row * ncol + split - 1],
            -1
        );
    }
}

#[test]
fn barrier_component_shift_can_group_adjacent_components_when_each_is_local_minimum() {
    let nrow = 1;
    let ncol = 6;
    let mut horizontal = vec![None; nrow * (ncol - 1)];
    let vertical = Vec::new();
    for (col, offset) in [100, 150, 100, 100, -50].into_iter().enumerate() {
        horizontal[horizontal_index(0, col, ncol)] = Some(edge_with_offset(1, offset));
    }
    let mut labels = vec![0_i32; nrow * ncol];
    let before = objective(&labels, &horizontal, &vertical, nrow, ncol);

    let applied =
        refine_labels_by_barrier_component_shifts(&mut labels, &horizontal, &vertical, nrow, ncol);
    let after = objective(&labels, &horizontal, &vertical, nrow, ncol);

    assert!(applied > 0);
    assert!(after < before);
    assert_eq!(labels[0], 0);
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1] - labels[2], 1);
}
