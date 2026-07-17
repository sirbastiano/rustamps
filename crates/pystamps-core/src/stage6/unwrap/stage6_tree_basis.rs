use crate::stage6::unwrap::residual_view::CompactResidualView;
use rayon::prelude::*;
use std::collections::VecDeque;

#[path = "stage6_tree_adjacency.rs"]
mod stage6_tree_adjacency;
#[path = "stage6_tree_candidates.rs"]
mod stage6_tree_candidates;
#[path = "stage6_tree_hierarchy.rs"]
mod stage6_tree_hierarchy;
use self::stage6_tree_adjacency::TreeAdjacency;
use self::stage6_tree_candidates::negative_candidate_arcs as collect_negative_candidate_arcs;
use self::stage6_tree_hierarchy::{chain_heads_into, lca as hierarchy_lca};

#[derive(Default)]
pub(crate) struct CompactTreeBasis {
    adjacency: TreeAdjacency,
    depth: Vec<usize>,
    parent_base: Vec<usize>,
    chain_head: Vec<u32>,
    up_arc: Vec<usize>,
    down_arc: Vec<usize>,
    order: Vec<usize>,
    seen: Vec<bool>,
    queue: VecDeque<usize>,
    subtree_size: Vec<usize>,
    heavy_child: Vec<usize>,
    up_root_cost: Vec<i64>,
    down_root_cost: Vec<i64>,
    node_count: usize,
}

impl CompactTreeBasis {
    pub(crate) fn new(view: &CompactResidualView<'_>, tree_arc_indices: &[usize]) -> Option<Self> {
        let mut basis = Self::default();
        basis.rebuild(view, tree_arc_indices)?;
        Some(basis)
    }

    pub(crate) fn rebuild(
        &mut self,
        view: &CompactResidualView<'_>,
        tree_arc_indices: &[usize],
    ) -> Option<()> {
        let node_count = view.node_count();
        if node_count == 0 || node_count > u32::MAX as usize {
            return None;
        }
        self.adjacency.rebuild(view, tree_arc_indices)?;
        reset_vec(&mut self.parent_base, node_count, usize::MAX);
        reset_vec(&mut self.up_arc, node_count, usize::MAX);
        reset_vec(&mut self.down_arc, node_count, usize::MAX);
        reset_vec(&mut self.depth, node_count, 0);
        reset_vec(&mut self.seen, node_count, false);
        self.order.clear();
        self.order.reserve(node_count);
        self.queue.clear();
        self.queue.reserve(node_count);
        self.queue.push_back(0);
        self.seen[0] = true;
        self.parent_base[0] = 0;

        while let Some(node) = self.queue.pop_front() {
            self.order.push(node);
            for edge in self.adjacency.neighbors(node) {
                let next = edge.next as usize;
                if self.seen[next] {
                    continue;
                }
                self.seen[next] = true;
                self.parent_base[next] = node;
                self.down_arc[next] = edge.down as usize;
                self.up_arc[next] = edge.up as usize;
                self.depth[next] = self.depth[node] + 1;
                self.queue.push_back(next);
            }
        }
        if self.seen.iter().any(|value| !*value) {
            return None;
        }
        chain_heads_into(
            &self.parent_base,
            &self.order,
            &mut self.chain_head,
            &mut self.subtree_size,
            &mut self.heavy_child,
        );
        self.node_count = node_count;
        Some(())
    }

    pub(crate) fn find_negative_cycle(
        &mut self,
        view: &CompactResidualView<'_>,
    ) -> Option<Vec<usize>> {
        self.refresh_root_costs(view)?;
        let best_arc = self.best_negative_arc(view, false)?;
        self.cycle_for_arc(view, best_arc)
    }

    pub(crate) fn find_negative_cycle_parallel(
        &mut self,
        view: &CompactResidualView<'_>,
    ) -> Option<Vec<usize>> {
        self.refresh_root_costs(view)?;
        let best_arc = self.best_negative_arc(view, true)?;
        self.cycle_for_arc(view, best_arc)
    }

    pub(crate) fn negative_cycles(
        &mut self,
        view: &CompactResidualView<'_>,
        limit: usize,
        parallel: bool,
    ) -> Vec<Vec<usize>> {
        if self.refresh_root_costs(view).is_none() {
            return Vec::new();
        }
        let arc_cycle_cost = |index| self.arc_cycle_cost(view, index);
        collect_negative_candidate_arcs(view.arc_count(), limit, parallel, arc_cycle_cost)
            .into_iter()
            .filter_map(|(_cost, index)| self.cycle_for_arc(view, index))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn best_reduced_cost_relaxation(
        &mut self,
        view: &CompactResidualView<'_>,
        candidates: impl IntoIterator<Item = usize>,
    ) -> Option<(usize, usize)> {
        self.refresh_root_costs(view)?;
        let mut best = None;
        let mut best_gain = 0_i64;
        for index in candidates {
            if self.adjacency.is_tree_arc(index) {
                continue;
            }
            let Some(arc) = view.arc(index) else { continue };
            if arc.to == 0 {
                continue;
            }
            let gain =
                self.down_root_cost[arc.to] - self.down_root_cost[arc.from] - i64::from(arc.cost);
            if gain <= best_gain || self.lca(arc.to, arc.from) == Some(arc.to) {
                continue;
            }
            best_gain = gain;
            best = Some((index, self.down_arc[arc.to] / 2));
        }
        best
    }

    fn best_negative_arc(&self, view: &CompactResidualView<'_>, parallel: bool) -> Option<usize> {
        if parallel && view.arc_count() > 4096 {
            return (0..view.arc_count())
                .into_par_iter()
                .filter_map(|index| self.arc_cycle_cost(view, index))
                .reduce_with(best_candidate)
                .map(|(_cost, index)| index);
        }
        let mut best_arc = None;
        let mut best_cost = 0_i64;
        for index in 0..view.arc_count() {
            let Some((cost, _index)) = self.arc_cycle_cost(view, index) else {
                continue;
            };
            if cost < best_cost {
                best_cost = cost;
                best_arc = Some(index);
            }
        }
        best_arc
    }

    #[inline(always)]
    fn arc_cycle_cost(&self, view: &CompactResidualView<'_>, index: usize) -> Option<(i64, usize)> {
        if self.adjacency.is_tree_arc(index) {
            return None;
        }
        let arc = view.arc(index)?;
        let path_cost = self.path_cost_with(arc.to, arc.from)?;
        let cost = i64::from(arc.cost) + path_cost;
        (cost < 0).then_some((cost, index))
    }

    fn refresh_root_costs(&mut self, view: &CompactResidualView<'_>) -> Option<()> {
        reset_vec(&mut self.up_root_cost, self.node_count, 0);
        reset_vec(&mut self.down_root_cost, self.node_count, 0);
        for &node in self.order.iter().skip(1) {
            let parent = self.parent_base[node];
            self.up_root_cost[node] =
                self.up_root_cost[parent] + i64::from(view.cost(self.up_arc[node])?);
            self.down_root_cost[node] =
                self.down_root_cost[parent] + i64::from(view.cost(self.down_arc[node])?);
        }
        Some(())
    }

    #[inline(always)]
    fn lca(&self, left: usize, right: usize) -> Option<usize> {
        hierarchy_lca(
            &self.depth,
            &self.parent_base,
            &self.chain_head,
            left,
            right,
        )
    }

    fn path_cost_with(&self, from: usize, to: usize) -> Option<i64> {
        let lca = self.lca(from, to)?;
        Some(
            (self.up_root_cost[from] - self.up_root_cost[lca])
                + (self.down_root_cost[to] - self.down_root_cost[lca]),
        )
    }

    fn cycle_for_arc(
        &self,
        view: &CompactResidualView<'_>,
        non_tree_arc_index: usize,
    ) -> Option<Vec<usize>> {
        let non_tree = view.arc(non_tree_arc_index)?;
        let mut path = self.path_arcs(non_tree.to, non_tree.from)?;
        let mut cycle = Vec::with_capacity(path.len() + 1);
        cycle.push(non_tree_arc_index);
        cycle.append(&mut path);
        Some(cycle)
    }

    fn path_arcs(&self, mut from: usize, mut to: usize) -> Option<Vec<usize>> {
        let lca = self.lca(from, to)?;
        let mut path = Vec::new();
        while from != lca {
            path.push(self.up_arc[from]);
            from = self.parent_base[from];
        }
        let mut down = Vec::new();
        while to != lca {
            down.push(self.down_arc[to]);
            to = self.parent_base[to];
        }
        down.reverse();
        path.extend(down);
        Some(path)
    }
}

fn reset_vec<T: Clone>(values: &mut Vec<T>, len: usize, value: T) {
    values.resize(len, value.clone());
    values.fill(value);
}

fn best_candidate(left: (i64, usize), right: (i64, usize)) -> (i64, usize) {
    if right.0 < left.0 || (right.0 == left.0 && right.1 < left.1) {
        right
    } else {
        left
    }
}

#[cfg(test)]
#[path = "stage6_tree_basis_tests.rs"]
mod tests;
