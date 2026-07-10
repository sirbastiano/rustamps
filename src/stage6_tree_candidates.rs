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
            .fold(Vec::new, |mut local, index| {
                if let Some(candidate) = arc_cycle_cost(index) {
                    push_candidate(&mut local, candidate, limit);
                }
                local
            })
            .reduce(Vec::new, |mut left, right| {
                for candidate in right {
                    push_candidate(&mut left, candidate, limit);
                }
                left
            });
    }
    let mut candidates = Vec::new();
    for index in 0..arc_count {
        if let Some(candidate) = arc_cycle_cost(index) {
            push_candidate(&mut candidates, candidate, limit);
        }
    }
    candidates
}

fn push_candidate(candidates: &mut Vec<(i64, usize)>, candidate: (i64, usize), limit: usize) {
    candidates.push(candidate);
    candidates
        .sort_unstable_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    candidates.truncate(limit);
}
