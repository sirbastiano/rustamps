use std::collections::BinaryHeap;

use rayon::prelude::*;

pub(super) fn negative_candidate_arcs(
    arc_count: usize,
    limit: usize,
    parallel: bool,
    arc_cycle_cost: impl Fn(usize) -> Option<(i64, usize)> + Sync,
) -> Vec<(i64, usize)> {
    if limit == 0 {
        return Vec::new();
    }
    if parallel && arc_count > 4096 {
        return (0..arc_count)
            .into_par_iter()
            .fold(BinaryHeap::new, |mut local, index| {
                if let Some(candidate) = arc_cycle_cost(index) {
                    push_candidate(&mut local, candidate, limit);
                }
                local
            })
            .reduce(BinaryHeap::new, |mut left, right| {
                for candidate in right {
                    push_candidate(&mut left, candidate, limit);
                }
                left
            })
            .into_sorted_vec();
    }
    let mut candidates = BinaryHeap::new();
    for index in 0..arc_count {
        if let Some(candidate) = arc_cycle_cost(index) {
            push_candidate(&mut candidates, candidate, limit);
        }
    }
    candidates.into_sorted_vec()
}

fn push_candidate(
    candidates: &mut BinaryHeap<(i64, usize)>,
    candidate: (i64, usize),
    limit: usize,
) {
    if candidates.len() < limit {
        candidates.push(candidate);
    } else if candidates.peek().is_some_and(|worst| candidate < *worst) {
        candidates.pop();
        candidates.push(candidate);
    }
}

#[cfg(test)]
mod tests {
    use super::negative_candidate_arcs;

    #[test]
    fn bounded_heap_returns_the_same_ordered_smallest_candidates() {
        let score = |index: usize| Some((((index * 37) % 101) as i64 - 50, index));
        let sequential = negative_candidate_arcs(10_000, 32, false, score);
        let parallel = negative_candidate_arcs(10_000, 32, true, score);
        let mut expected = (0..10_000).filter_map(score).collect::<Vec<_>>();
        expected.sort_unstable();
        expected.truncate(32);
        assert_eq!(sequential, expected);
        assert_eq!(parallel, expected);
    }
}
