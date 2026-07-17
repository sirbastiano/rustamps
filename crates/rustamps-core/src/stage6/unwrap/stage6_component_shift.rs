use crate::stage6::unwrap::native::EdgeDatum;

#[path = "stage6_component_shift_components.rs"]
mod stage6_component_shift_components;
#[path = "stage6_component_shift_cut.rs"]
mod stage6_component_shift_cut;
#[path = "stage6_component_shift_score.rs"]
mod stage6_component_shift_score;

use self::stage6_component_shift_components::{
    collect_barrier_component, collect_same_label_component,
};
use self::stage6_component_shift_cut::refine_labels_by_barrier_component_cut;
use self::stage6_component_shift_score::{component_shift_gain, positive_edge_mean_energy};

const MAX_COMPONENT_SHIFT_PASSES: usize = 4;
const MAX_BARRIER_COMPONENT_SHIFT_PASSES: usize = 2;
pub(super) const MAX_BARRIER_COMPONENT_CUT_COMPONENTS: usize = 65_536;
pub(super) const MAX_BARRIER_COMPONENT_CUT_BOUNDARIES: usize = 500_000;
pub(super) const CUT_INF_CAP: i64 = 1_i64 << 58;

pub(crate) fn refine_labels_by_component_shifts(
    labels: &mut [i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) -> usize {
    let cell_count = nrow.saturating_mul(ncol);
    if labels.len() != cell_count || cell_count == 0 {
        return 0;
    }

    let mut visited = vec![false; cell_count];
    let mut mark = vec![false; cell_count];
    let mut stack = Vec::new();
    let mut component = Vec::new();
    let mut applied = 0_usize;

    for _ in 0..MAX_COMPONENT_SHIFT_PASSES {
        visited.fill(false);
        let mut changed = false;
        for seed in 0..cell_count {
            if visited[seed] {
                continue;
            }
            collect_same_label_component(
                seed,
                labels,
                nrow,
                ncol,
                &mut visited,
                &mut stack,
                &mut component,
            );
            if component.len() == cell_count {
                continue;
            }
            for &node in &component {
                mark[node] = true;
            }
            let gain_neg = component_shift_gain(
                labels, &component, &mark, horizontal, vertical, nrow, ncol, -1,
            );
            let gain_pos = component_shift_gain(
                labels, &component, &mark, horizontal, vertical, nrow, ncol, 1,
            );
            let shift = if gain_pos > gain_neg { 1 } else { -1 };
            let gain = gain_pos.max(gain_neg);
            if gain > 0 {
                for &node in &component {
                    labels[node] += shift;
                }
                changed = true;
                applied += 1;
            }
            for &node in &component {
                mark[node] = false;
            }
        }
        if !changed {
            break;
        }
    }
    applied
}

pub(crate) fn refine_labels_by_barrier_component_shifts(
    labels: &mut [i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) -> usize {
    let cell_count = nrow.saturating_mul(ncol);
    if labels.len() != cell_count || cell_count == 0 {
        return 0;
    }
    let barrier = positive_edge_mean_energy(labels, horizontal, vertical, nrow, ncol);
    if barrier <= 0 {
        return 0;
    }

    let mut visited = vec![false; cell_count];
    let mut mark = vec![false; cell_count];
    let mut stack = Vec::new();
    let mut component = Vec::new();
    let mut applied = 0_usize;

    let lower_barrier = (barrier / 2).max(1);
    let lowest_barrier = (barrier / 4).max(1);
    let eighth_barrier = (barrier / 8).max(1);
    let sixteenth_barrier = (barrier / 16).max(1);
    let thirty_second_barrier = (barrier / 32).max(1);
    for active_barrier in [
        barrier,
        lower_barrier,
        lowest_barrier,
        eighth_barrier,
        sixteenth_barrier,
        thirty_second_barrier,
    ] {
        if active_barrier == barrier && applied > 0 {
            continue;
        }
        if active_barrier == barrier || active_barrier < barrier {
            for _ in 0..MAX_BARRIER_COMPONENT_SHIFT_PASSES {
                visited.fill(false);
                let mut changed = false;
                for seed in 0..cell_count {
                    if visited[seed] {
                        continue;
                    }
                    collect_barrier_component(
                        seed,
                        labels,
                        horizontal,
                        vertical,
                        nrow,
                        ncol,
                        active_barrier,
                        &mut visited,
                        &mut stack,
                        &mut component,
                    );
                    if component.len() == cell_count {
                        continue;
                    }
                    for &node in &component {
                        mark[node] = true;
                    }
                    let gain_neg = component_shift_gain(
                        labels, &component, &mark, horizontal, vertical, nrow, ncol, -1,
                    );
                    let gain_pos = component_shift_gain(
                        labels, &component, &mark, horizontal, vertical, nrow, ncol, 1,
                    );
                    let shift = if gain_pos > gain_neg { 1 } else { -1 };
                    let gain = gain_pos.max(gain_neg);
                    if gain > 0 {
                        for &node in &component {
                            labels[node] += shift;
                        }
                        changed = true;
                        applied += 1;
                    }
                    for &node in &component {
                        mark[node] = false;
                    }
                }
                if !changed {
                    if refine_labels_by_barrier_component_cut(
                        labels,
                        horizontal,
                        vertical,
                        nrow,
                        ncol,
                        active_barrier,
                    ) {
                        applied += 1;
                    } else {
                        break;
                    }
                }
            }
        }
    }
    applied
}
