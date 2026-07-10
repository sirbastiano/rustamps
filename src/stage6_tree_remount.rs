use crate::stage6_residual_view::CompactResidualView;

use super::stage6_tree_basis::CompactTreeBasis;

#[cfg(test)]
pub(crate) fn relax_compact_tree_by_reduced_cost(
    view: &CompactResidualView<'_>,
    tree_arc_indices: &mut [usize],
    max_remounts: usize,
) -> usize {
    relax_compact_tree_by_reduced_cost_iter(
        view,
        tree_arc_indices,
        0..view.arc_count(),
        max_remounts,
    )
}

#[cfg(test)]
pub(crate) fn relax_compact_tree_by_reduced_cost_candidates(
    view: &CompactResidualView<'_>,
    tree_arc_indices: &mut [usize],
    candidates: &[usize],
    max_remounts: usize,
) -> usize {
    relax_compact_tree_by_reduced_cost_iter(
        view,
        tree_arc_indices,
        candidates.iter().copied(),
        max_remounts,
    )
}

fn relax_compact_tree_by_reduced_cost_iter<I>(
    view: &CompactResidualView<'_>,
    tree_arc_indices: &mut [usize],
    candidates: I,
    max_remounts: usize,
) -> usize
where
    I: Clone + IntoIterator<Item = usize>,
{
    let mut applied = 0;
    for _ in 0..max_remounts {
        let Some(basis) = CompactTreeBasis::new(view, tree_arc_indices) else {
            break;
        };
        let Some((entering, leaving_pair)) =
            basis.best_reduced_cost_relaxation(view, candidates.clone())
        else {
            break;
        };
        if tree_arc_indices
            .iter()
            .any(|&index| index / 2 == entering / 2)
        {
            break;
        }
        let Some(remove_pos) = tree_arc_indices
            .iter()
            .position(|&index| index / 2 == leaving_pair)
        else {
            break;
        };
        tree_arc_indices[remove_pos] = entering;
        applied += 1;
    }
    applied
}
