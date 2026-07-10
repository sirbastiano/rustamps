use crate::stage6_residual_view::CompactResidualView;
use rayon::prelude::*;
use std::collections::VecDeque;

#[path = "stage6_tree_candidates.rs"]
mod stage6_tree_candidates;
use self::stage6_tree_candidates::negative_candidate_arcs as collect_negative_candidate_arcs;

pub(crate) struct CompactTreeBasis {
    tree_mask: Vec<bool>,
    depth: Vec<usize>,
    parent: Vec<u32>,
    parent_base: Vec<usize>,
    up_arc: Vec<usize>,
    down_arc: Vec<usize>,
    order: Vec<usize>,
    levels: usize,
    node_count: usize,
}

impl CompactTreeBasis {
    pub(crate) fn new(view: &CompactResidualView<'_>, tree_arc_indices: &[usize]) -> Option<Self> {
        let node_count = view.node_count();
        if node_count == 0 || node_count > u32::MAX as usize {
            return None;
        }
        let mut tree_mask = vec![false; view.arc_count()];
        let mut adjacency = vec![Vec::<(usize, usize, usize)>::new(); node_count];
        for &index in tree_arc_indices {
            let Some(arc) = view.arc(index) else {
                continue;
            };
            let reverse_index = index ^ 1;
            let reverse = view.arc(reverse_index)?;
            if reverse.from != arc.to || reverse.to != arc.from {
                return None;
            }
            tree_mask[index] = true;
            tree_mask[reverse_index] = true;
            adjacency[arc.from].push((arc.to, index, reverse_index));
            adjacency[arc.to].push((arc.from, reverse_index, index));
        }

        let levels = (usize::BITS - node_count.saturating_sub(1).leading_zeros()) as usize + 1;
        let mut parent = vec![u32::MAX; levels * node_count];
        let mut parent_base = vec![usize::MAX; node_count];
        let mut up_arc = vec![usize::MAX; node_count];
        let mut down_arc = vec![usize::MAX; node_count];
        let mut depth = vec![0_usize; node_count];
        let mut seen = vec![false; node_count];
        let mut order = Vec::with_capacity(node_count);
        let mut queue = VecDeque::from([0_usize]);
        seen[0] = true;
        parent[0] = 0;
        parent_base[0] = 0;

        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &(next, down, up) in &adjacency[node] {
                if seen[next] {
                    continue;
                }
                seen[next] = true;
                parent[next] = node as u32;
                parent_base[next] = node;
                down_arc[next] = down;
                up_arc[next] = up;
                depth[next] = depth[node] + 1;
                queue.push_back(next);
            }
        }
        if seen.iter().any(|value| !*value) {
            return None;
        }
        for level in 1..levels {
            for node in 0..node_count {
                let prev = (level - 1) * node_count + node;
                let mid = parent[prev] as usize;
                parent[level * node_count + node] = parent[(level - 1) * node_count + mid];
            }
        }

        Some(Self {
            tree_mask,
            depth,
            parent,
            parent_base,
            up_arc,
            down_arc,
            order,
            levels,
            node_count,
        })
    }

    pub(crate) fn find_negative_cycle(&self, view: &CompactResidualView<'_>) -> Option<Vec<usize>> {
        let (up_root_cost, down_root_cost) = self.root_costs(view)?;
        let best_arc = self.best_negative_arc(view, &up_root_cost, &down_root_cost, false)?;
        self.cycle_for_arc(view, best_arc)
    }

    pub(crate) fn find_negative_cycle_parallel(
        &self,
        view: &CompactResidualView<'_>,
    ) -> Option<Vec<usize>> {
        let (up_root_cost, down_root_cost) = self.root_costs(view)?;
        let best_arc = self.best_negative_arc(view, &up_root_cost, &down_root_cost, true)?;
        self.cycle_for_arc(view, best_arc)
    }

    pub(crate) fn negative_cycles(
        &self,
        view: &CompactResidualView<'_>,
        limit: usize,
        parallel: bool,
    ) -> Vec<Vec<usize>> {
        let Some((up_root_cost, down_root_cost)) = self.root_costs(view) else {
            return Vec::new();
        };
        let arc_cycle_cost =
            |index| self.arc_cycle_cost(view, &up_root_cost, &down_root_cost, index);
        collect_negative_candidate_arcs(view.arc_count(), limit, parallel, arc_cycle_cost)
            .into_iter()
            .filter_map(|(_cost, index)| self.cycle_for_arc(view, index))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn best_reduced_cost_relaxation(
        &self,
        view: &CompactResidualView<'_>,
        candidates: impl IntoIterator<Item = usize>,
    ) -> Option<(usize, usize)> {
        let (_up_root_cost, down_root_cost) = self.root_costs(view)?;
        let mut best = None;
        let mut best_gain = 0_i64;
        for index in candidates {
            if self.tree_mask.get(index).copied().unwrap_or(false) {
                continue;
            }
            let Some(arc) = view.arc(index) else { continue };
            if arc.to == 0 {
                continue;
            }
            let gain = down_root_cost[arc.to] - down_root_cost[arc.from] - i64::from(arc.cost);
            if gain <= best_gain || self.lca(arc.to, arc.from) == Some(arc.to) {
                continue;
            }
            best_gain = gain;
            best = Some((index, self.down_arc[arc.to] / 2));
        }
        best
    }

    fn best_negative_arc(
        &self,
        view: &CompactResidualView<'_>,
        up_root_cost: &[i64],
        down_root_cost: &[i64],
        parallel: bool,
    ) -> Option<usize> {
        if parallel && view.arc_count() > 4096 {
            return (0..view.arc_count())
                .into_par_iter()
                .filter_map(|index| self.arc_cycle_cost(view, up_root_cost, down_root_cost, index))
                .reduce_with(best_candidate)
                .map(|(_cost, index)| index);
        }
        let mut best_arc = None;
        let mut best_cost = 0_i64;
        for index in 0..view.arc_count() {
            let Some((cost, _index)) =
                self.arc_cycle_cost(view, up_root_cost, down_root_cost, index)
            else {
                continue;
            };
            if cost < best_cost {
                best_cost = cost;
                best_arc = Some(index);
            }
        }
        best_arc
    }

    fn arc_cycle_cost(
        &self,
        view: &CompactResidualView<'_>,
        up_root_cost: &[i64],
        down_root_cost: &[i64],
        index: usize,
    ) -> Option<(i64, usize)> {
        if self.tree_mask.get(index).copied().unwrap_or(false) {
            return None;
        }
        let arc = view.arc(index)?;
        let path_cost = self.path_cost_with(arc.to, arc.from, up_root_cost, down_root_cost)?;
        let cost = i64::from(arc.cost) + path_cost;
        (cost < 0).then_some((cost, index))
    }

    fn root_costs(&self, view: &CompactResidualView<'_>) -> Option<(Vec<i64>, Vec<i64>)> {
        let mut up_root_cost = vec![0_i64; self.node_count];
        let mut down_root_cost = vec![0_i64; self.node_count];
        for &node in self.order.iter().skip(1) {
            let parent = self.parent_base[node];
            up_root_cost[node] =
                up_root_cost[parent] + i64::from(view.arc(self.up_arc[node])?.cost);
            down_root_cost[node] =
                down_root_cost[parent] + i64::from(view.arc(self.down_arc[node])?.cost);
        }
        Some((up_root_cost, down_root_cost))
    }

    fn index(&self, level: usize, node: usize) -> usize {
        level * self.node_count + node
    }

    fn raise(&self, mut node: usize, steps: usize) -> Option<usize> {
        if node >= self.node_count {
            return None;
        }
        for level in 0..self.levels {
            if ((steps >> level) & 1) == 1 {
                node = self.parent[self.index(level, node)] as usize;
            }
        }
        Some(node)
    }

    fn lca(&self, mut left: usize, mut right: usize) -> Option<usize> {
        if left >= self.node_count || right >= self.node_count {
            return None;
        }
        if self.depth[left] > self.depth[right] {
            left = self.raise(left, self.depth[left] - self.depth[right])?;
        } else if self.depth[right] > self.depth[left] {
            right = self.raise(right, self.depth[right] - self.depth[left])?;
        }
        if left == right {
            return Some(left);
        }
        for level in (0..self.levels).rev() {
            if self.parent[self.index(level, left)] != self.parent[self.index(level, right)] {
                left = self.parent[self.index(level, left)] as usize;
                right = self.parent[self.index(level, right)] as usize;
            }
        }
        Some(self.parent_base[left])
    }

    fn path_cost_with(
        &self,
        from: usize,
        to: usize,
        up_root_cost: &[i64],
        down_root_cost: &[i64],
    ) -> Option<i64> {
        let lca = self.lca(from, to)?;
        Some((up_root_cost[from] - up_root_cost[lca]) + (down_root_cost[to] - down_root_cost[lca]))
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

fn best_candidate(left: (i64, usize), right: (i64, usize)) -> (i64, usize) {
    if right.0 < left.0 || (right.0 == left.0 && right.1 < left.1) {
        right
    } else {
        left
    }
}
